import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import funcs
from skimage.transform import resize
import numpy as np
from scipy import io

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
    
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
        
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss
        
def masking(mask,img,ref):
    img_mask = mask*img+(1-mask)*ref
    return img_mask.float()
    
def ddpm_forward(deblur_origin,angles,group_number):
    """
    Algorithm 2.
    """
    ## device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## parameter
    deblur_size = 256
    output_size = 512
    deblur_origin = resize(deblur_origin,output_shape=(deblur_size,deblur_size))
    mask = np.ones((output_size,output_size))
    for i in range(output_size):
      for j in range(output_size):
        if (i-256)**2+(j-256)**2>40000:
          mask[i][j] = 0
    mask = mask[::2,::2]
    mask = torch.from_numpy(mask).to(device)
    
    beta_1 = 1e-4
    beta_T = 0.02
    T = 1000
    
    ## load data
    ref = io.loadmat('./data/level_7/htc2022_07a_recon_fbp_seg.mat')['reconFullFbpSeg'].astype(np.float32)
    ref = ref[::2,::2]
    ref = torch.from_numpy(ref).to(device)

    ##Define the network and load pretrained weights to gpu
    model = UNet(T=T, ch=64, ch_mult=[1, 2, 2, 2], attn=[1],num_res_blocks=2, dropout=0.1)

    model.load_state_dict(torch.load('./pre-trained-weights/ddpm.pkl',map_location='cuda:0'))
    #model.load_state_dict(torch.load("/home/wangrongqian/CT/DDPM-HTC-256/checkpoints/ddpm_256_covid_125.pkl"))
    model = model.to(device)

    ##Normalization
    
    x_T = torch.randn((1, 1, deblur_size, deblur_size)).to(device)
    
    betas = torch.linspace(beta_1, beta_T, T).double()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0).double()
    
    alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:-1]
    posterior_var = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
    
    model_log_var = torch.log(torch.cat([posterior_var[1:2], betas[1:]]))
    
    line = np.linspace(0.2, 1.0, T)
    a = line.copy()
    
    ##Parameters
    temp = 150
    a[0:temp] = 1
    b = 1 - a
    b[0:temp] = np.linspace(0.6, 0.2, temp)
    c = b.copy()
    #c[:] = 0
    c[temp:] = 0
    
    x_t = x_T.reshape((1,1, deblur_size, deblur_size))
    
    for time_step in reversed(range(T)):
        
        t = x_t.new_ones([1, ], dtype=torch.long) * time_step
        time_tensor = x_t.new_ones([1, ], dtype=torch.long) * time_step
    
        log_var = extract(model_log_var.cpu(), time_tensor.cpu(), x_t.shape)
    
        with torch.no_grad():
            eps = model(x_t, t)
        torch.cuda.empty_cache()
    
        ## x_0
        #mean = (x_t - (1-alphas[time_step])/torch.sqrt(1-alphas_bar[time_step]) * eps) / torch.sqrt(alphas[time_step])
        x_0 = torch.sqrt(1. / alphas_bar[time_step]) * x_t - torch.sqrt(1. / alphas_bar[time_step] - 1) * eps
        x_0 = x_0.cpu().numpy().reshape(deblur_size,deblur_size)
        x_0_512 = resize(x_0,output_shape=(output_size, output_size))
        #x_0_512 = sr.super_resolution(x_0[::2,::2],device)
        sinogram, BP, FBP = funcs.generate_all(x_0_512, angles, result_size=512, \
                  det_width=1.3484, det_count=560, source_origin=410.66, \
                  origin_det=143.08, eff_pixelsize=0.1483)

        ##deblur
        #BP=resize(BP,output_shape=(deblur_size, deblur_size))
        BP=BP[::2,::2]/23000
        perturb=funcs.Deep_Deblur(BP,group_number,device).reshape((deblur_size,deblur_size))

        x_0 = torch.from_numpy(a[time_step]*deblur_origin + b[time_step]*x_0 - c[time_step]*perturb).to(device)
        mean = torch.sqrt(alphas_bar_prev[time_step]) * betas[time_step] / (1. - alphas_bar[time_step]) * masking(mask,x_0,ref) + torch.sqrt(alphas[time_step]) * (1. - alphas_bar_prev[time_step]) / (1. - alphas_bar[time_step]) * x_t
    
        # no noise when t == 0
        if time_step > 0:
            noise = torch.randn_like(x_t).to(device)
        else:
            noise = 0
        x_t = mean.to(device) + torch.exp(0.5 * log_var).to(device) * noise
        
    x_t = x_t.reshape((deblur_size, deblur_size))
    x_t[x_t<0.5] = 0
    x_t[x_t>0.5] = 1
    
    ddpm = x_t.cpu().numpy()
    return ddpm
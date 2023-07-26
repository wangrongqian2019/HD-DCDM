## Functions used in "BP Deblur"

import numpy as np
from scipy import io
import astra
import torch
from torch.autograd import Variable
import pylab
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
from skimage.morphology import closing
from model import DeepRFT as myNet
import sr
import DDPM
def generate_all(model, angles, result_size=512, \
                      det_width=1.3484, det_count=560, source_origin=410.66, \
                      origin_det=143.08, eff_pixelsize=0.1483):

    '''
    The code is based on HelTomo.
    Inputs:
        angles: projection angles
        result_size: pixel number of the reconstruction domain
        det_width: distance between the centers of two adjacent detector pixels
        det_count: number of detector pixels in a single projection
        source_origin: distance between the source and the center of rotation
        origin_det: distance between the center of rotation and the detector array
        eff_pixelsize: effictive size of pixels
    Output:
        W: an operator that maps the models to signals
    '''
    
    ##Distances from specified in terms of effective pixel size
    source_origin=source_origin/eff_pixelsize
    origin_det=origin_det/eff_pixelsize
    
    ##Transform angles to radians
    angles=np.radians(angles)

    ##Define the geomotry
    vol_geom = astra.create_vol_geom(result_size, result_size) 
    proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, angles,source_origin,origin_det) 
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    
    sinogram_id, sinogram = astra.create_sino(model, proj_id)

    ## BP
    # Create a data object for the reconstruction
    
    rec_id = astra.data2d.create('-vol', vol_geom)
    
    # create configuration 
    cfg = astra.astra_dict('BP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    

    # possible values for FilterType:
    # none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    # triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    # blackman-nuttall, flat-top, kaiser, parzen

    
    # Create and run the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    # Get the result
    BP = astra.data2d.get(rec_id)

    ## FBP
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)

    # create configuration 
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = { 'FilterType': 'ram-lak'}

    # possible values for FilterType:
    # none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    # triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    # blackman-nuttall, flat-top, kaiser, parzen


    # Create and run the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Get the result
    FBP = astra.data2d.get(rec_id)


    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    
    return sinogram, BP, FBP
    


def get_coeff(group_number):
    
    
    return 23000*(100-10*group_number)/30

def BP_reconstruction(Input_signal, angles, result_size=512, \
                      det_width=1.3484, det_count=560, source_origin=410.66, \
                      origin_det=143.08, eff_pixelsize=0.1483,group_number=7 ):

    '''
    Back projection. The code is based on HelTomo.
    Inputs:
        Input_signal: measured sinogram
        angles: projection angles
        result_size: pixel number of the reconstruction domain
        det_width: distance between the centers of two adjacent detector pixels
        det_count: number of detector pixels in a single projection
        source_origin: distance between the source and the center of rotation
        origin_det: distance between the center of rotation and the detector array
        eff_pixelsize: effictive size of pixels
    Output:
        Bp: result of the back projection method
    '''

    ##Distances from specified in terms of effective pixel size
    source_origin=source_origin/eff_pixelsize
    origin_det=origin_det/eff_pixelsize
    
    ##Transform angles to radians
    angles=np.radians(angles)

    ##Define the geomotry
    vol_geom = astra.create_vol_geom(result_size, result_size) 
    proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, angles,source_origin,origin_det)   
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
   
    ##Get the projection matrix
    W = astra.optomo.OpTomo(proj_id)
    ##Back projection
    Bp = W.T.dot(Input_signal.ravel())
    Bp = np.reshape(Bp, (result_size,result_size))

    astra.projector.delete(proj_id)
    
    return Bp/get_coeff(group_number)


def Deep_Deblur(Input_albedo, group_number, device,img_resolution=256):
    '''
    Use network to enhance the result of back projection.
    Inputs:
        Input_albedo: result of the back projection method
        group_number: number of limited-angle tomography difficulty group
        img_resolution: resolution of input and output(if changed, the network should be retrained)
    Output:
        output: deblur result 
    '''
    
    ##Define the network and load pretrained weights to gpu
    net = myNet()
    try:
        net.load_state_dict(torch.load('./pre-trained-weights/level_%s.pkl'%(group_number)))
    except:
        #net=torch.nn.DataParallel(net)
        net.load_state_dict(torch.load('./pre-trained-weights/level_%s.pkl'%(group_number),map_location='cuda:0'))
    
    net = net.to(device)

    ##Normalization
    Input_albedo=Input_albedo/np.max(Input_albedo)
    
    ##Deblur
    with torch.no_grad():
        albedo = Variable(torch.from_numpy(Input_albedo)).reshape(1,1,img_resolution,img_resolution)
        albedo = albedo.to(device).type(torch.cuda.FloatTensor)

        output = net(albedo)
        output = output.data.cpu().numpy()
        output = output.reshape(1,1,img_resolution,img_resolution)
        output=np.squeeze(output/np.max(output))
    
    return output
def generate_parts_tosquare(input,output_size,stride1,stride2):
    '''
    
    
    '''
    ## get input size
    input_size1,input_size2=np.shape(input)
    
    ## calculate sub sample number
    slice1=(input_size1-output_size) // stride1 +1
    slice2=(input_size2-output_size) // stride2 +1
    
    output_num=slice1*slice2
    ## initialize output
    output=np.zeros([output_num,output_size,output_size])

    ## generate 
    for ii in range(slice1):

        for jj in range(slice2):

            output[ii*slice2+jj,:,:] = input[stride1*ii:stride1*ii+output_size,stride2*jj:stride2*jj+output_size]
    
    return output.reshape((output_num,1,output_size,output_size))

def reverse_generate_parts_tosquare(input,output_size1,output_size2,stride1,stride2):
    '''

    '''

    ## get input size
    input=np.squeeze(input)
    sample_num,input_size1,input_size2=np.shape(input) 
    ## calculate sub sample number
    slice1=(output_size1-input_size1) // stride1 +1
    slice2=(output_size2-input_size2) // stride2 +1
    
    output_num=slice1*slice2
    assert(output_num==sample_num, 'dimensions error')
    
    
    ## initialize output
    output=np.zeros([output_num,output_size1,output_size2])
    ## reverse
    for ii in range(slice1):

        for jj in range(slice2):

            output[ii*slice2+jj,stride1*ii:stride1*ii+input_size1,stride2*jj:stride2*jj+input_size2] = input[ii*slice2+jj,:,:]
    ## calculate the number of non-zero parts of each pixel
    mask=np.sum(output!=0,axis=0)
    mask[mask==0]=1
    output=np.sum(output,axis=0)/mask
    return output.reshape((output_size1,output_size2))




def Preprocess(Input_sinogram, device, group_number):

    #Use network to enhance the result of back projection.
    #Inputs:
    #    Input_albedo: result of the back projection method
    #    group_number: number of limited-angle tomography difficulty group
    #    img_resolution: resolution of input and output(if changed, the network should be retrained)
    #Output:
    #    output: deblur result 

    
    ##Define the network and load pretrained weights to gpu
    net = myNet()
    try:
        net.load_state_dict(torch.load('./pre-trained-weights/pre-process.pkl'))
    except:
        net=torch.nn.DataParallel(net)
        net.load_state_dict(torch.load('./pre-trained-weights/pre-process.pkl',map_location='cuda:0'))
    
    net = net.to(device)
    
    ## generate patches
    input_parts=generate_parts_tosquare(input=Input_sinogram,output_size=32,stride1=1,stride2=11)
    
    
    with torch.no_grad():
        input_parts = Variable(torch.from_numpy(input_parts))
        input_parts = input_parts.to(device)
        input_parts = input_parts.type(torch.cuda.FloatTensor)
        output_parts = net(input_parts)
        output_parts = output_parts.data.cpu().numpy()
        
    output=reverse_generate_parts_tosquare(input=output_parts,output_size1=2*(90-10*(group_number-1))+1,output_size2=560,stride1=1,stride2=11)
    
    return output

def Load_process(data_path,output_path,group_number):
    '''
    Load data from data path and reconstruct the phantom, then save the results to output path.
    Inputs:
        data_path: the path of the input mat file
        output_path: the path of the output png image
        group_number: difficulty level, to determine which pre-trained network to load 
    Output:
        None

    '''
    ##load data
    data=io.loadmat(data_path)['CtDataLimited'] 
    ##extract information from data
    sinogram=data['sinogram'][0][0]
    
    parameters=data['parameters'][0][0][0][0]  
      
    eff_pixel_size=parameters['effectivePixelSizePost'][0][0]
    det_width=parameters['geometricMagnification'][0][0]
    det_count=parameters['numDetectorsPost'][0][0]

    angles=parameters['angles'][0]
    output_angles=angles
    
    source_origin=parameters['distanceSourceOrigin'][0][0]
    origin_det=parameters['distanceSourceDetector'][0][0]-source_origin

    output_size=512
    deblur_size=256
    
    ##detecting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on ',device)
    
    ##let the angles start from 0
    angle_min=np.min(angles)
    angles=angles-angle_min
    
    ##preprocessing
    sinogram = Preprocess(sinogram,device,group_number)
    print('Data preprocessing finished......\n')
    torch.cuda.empty_cache()
    ##back projection
    BP=BP_reconstruction(sinogram,angles,result_size=output_size, det_width=det_width, det_count=det_count, source_origin=source_origin, origin_det=origin_det, eff_pixelsize=eff_pixel_size,group_number=group_number)

    

    ##deblur
    BP=resize(BP,output_shape=(deblur_size, deblur_size))
    result=Deep_Deblur(BP,group_number,device)
    print('Deblurring finished......\n')
    ##clear gpu memory
    torch.cuda.empty_cache()
    ##super resolution
    SR=sr.super_resolution(result,device)
    print('Super resolution finished......\n')
    torch.cuda.empty_cache()
    ##rotate the reconstruction to original orientation
    SR=rotate(SR,angle_min,order=0)    

    ##save results
    pylab.gray()
    #pylab.imsave('BP.png',BP)
    #pylab.imsave('Deblur.png',result)
    pylab.imsave(output_path,SR)

    #io.savemat('result.mat',{'albedo':SR})
    return BP, SR, output_angles

def find_mat(data_list):
    '''
    Find files with .mat format.
    Input:
        data_list: file names
    Output:
        tmp: name of mat files 
    '''
    tmp=[]
    for i in range(len(data_list)):
        tmp_name=data_list[i]
        if tmp_name[-4:]=='.mat':
            tmp.append(tmp_name)
    return tmp 
    
def calcScore(reconImg, groundtruthImg):
    Ir = reconImg
    It = groundtruthImg

    AND = lambda x, y: np.logical_and(x, y)
    NOT = lambda x: np.logical_not(x)

    # confusion matrix
    TP = float(len(np.where(AND(It, Ir))[0]))
    TN = float(len(np.where(AND(NOT(It), NOT(Ir)))[0]))
    FP = float(len(np.where(AND(NOT(It), Ir))[0]))
    FN = float(len(np.where(AND(It, NOT(Ir)))[0]))
    cmat = np.array([[TP, FN], [FP, TN]])

    # Matthews correlation coefficient (MCC)
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        score = 0
    else:
        score = numerator / denominator

    return score
def segmentation(img):
    img[img<0.5] = 0
    img[img>0.5] = 1
    
    return img

def derror(img1, img2,output_size,angles):   
    projections1 = forward(img1.reshape((output_size,output_size)),angles)
    projections2 = forward(img2.reshape((output_size,output_size)),angles)
    return np.mean((projections1 - projections2) ** 2) / np.mean(projections2 ** 2)
    
def forward(model, angles, result_size=512, \
                      det_width=1.3484, det_count=560, source_origin=410.66, \
                      origin_det=143.08, eff_pixelsize=0.1483):

    '''
    The code is based on HelTomo.
    Inputs:
        angles: projection angles
        result_size: pixel number of the reconstruction domain
        det_width: distance between the centers of two adjacent detector pixels
        det_count: number of detector pixels in a single projection
        source_origin: distance between the source and the center of rotation
        origin_det: distance between the center of rotation and the detector array
        eff_pixelsize: effictive size of pixels
    Output:
        W: an operator that maps the models to signals
    '''

    ##Distances from specified in terms of effective pixel size
    source_origin=source_origin/eff_pixelsize
    origin_det=origin_det/eff_pixelsize
    
    ##Transform angles to radians
    angles=np.radians(angles)

    ##Define the geomotry
    vol_geom = astra.create_vol_geom(result_size, result_size) 
    proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, angles,source_origin,origin_det) 
    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
    
    sinogram_id, sinogram = astra.create_sino(model, proj_id)
    ##Get the projection matrix
    #W = astra.optomo.OpTomo(proj_id)
    #W.T.dot(np.ones([560*181,512*512]).ravel())
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    
    return sinogram

def draw_figures(output_path,gt,Bp,deblur,ddpm,output_size,angles):
    '''
    
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extent = [0, 1, 1, 0]
    plt.figure(figsize=(18,12))
    cmax = 1
    cmin = 0
    colour = 'gray'
        
    gt_512 = resize(gt,(output_size,output_size))
    Bp_512 = resize(Bp,(output_size,output_size))
    deblur_512 = resize(deblur,(output_size,output_size))
    ddpm_512 = sr.super_resolution(ddpm,device)
        
    plt.subplot(2,3,1)
    plt.title('model',fontsize=18)
    plt.imshow(gt_512,vmax=cmax,vmin=cmin,extent=extent,cmap=colour)
    plt.yticks(size=15)
    plt.xticks(size=15)
    #plt.colorbar(shrink=0.7)
        
    plt.subplot(2,3,4)    
    plt.title('Bp, calcScore=%.2f, d=%.4f'%(calcScore(Bp_512,gt_512),derror(Bp_512,gt_512,output_size,angles)),fontsize=18)
    plt.imshow(Bp_512,vmax=cmax,vmin=0,extent=extent,cmap=colour)
    plt.yticks(size=15)
    plt.xticks(size=15)
        #plt.colorbar(shrink=0.7)
        
    plt.subplot(2,3,2)    
    plt.title('Deblur, calcScore=%.2f, d=%.4f'%(calcScore(deblur_512,gt_512),derror(deblur_512,gt_512,output_size,angles)),fontsize=18)
    plt.imshow(deblur_512,vmax=cmax,vmin=0,extent=extent,cmap=colour)
    plt.yticks(size=15)
    plt.xticks(size=15)  
        
    plt.subplot(2,3,5)    
    plt.title('Deblur-residual',fontsize=18)
    plt.imshow(deblur_512-gt_512,vmax=cmax,vmin=-1,extent=extent,cmap=colour)
    plt.yticks(size=15)
    plt.xticks(size=15)  

    plt.subplot(2,3,3)    
    plt.title('DDPM, calcScore=%.2f, d=%.4f'%(calcScore(ddpm_512,gt_512),derror(ddpm_512,gt_512,output_size,angles)),fontsize=18)
    plt.imshow(ddpm_512,vmax=cmax,vmin=0,extent=extent,cmap=colour)
    plt.yticks(size=15)
    plt.xticks(size=15)
      
    plt.subplot(2,3,6)    
    plt.title('DDPM-residual',fontsize=18)
    plt.imshow(ddpm_512-gt_512,vmax=cmax,vmin=-1,extent=extent,cmap=colour)
    plt.yticks(size=15)
    plt.xticks(size=15) 
       
    plt.tight_layout()
    plt.savefig(output_path)
    return 

def process_testdata(data_folder,output_folder,group_number,method_name,output_size):
    '''
    
    '''
    test_example=['a','b','c']
    ## Find the ground truth and process the signal
    #### process emample a
    for i in range(3):
        ## get the datapath and define the output name
        data_name='/htc2022_0%s%s_limited.mat'%(group_number,test_example[i])
        print('processing data:',data_name)
        output_file_name=data_name[0:-4]+'_'+method_name+'.png'
        data_path=data_folder+data_name
        output_path=output_folder+output_file_name
        ## reconstruct
        Bp,deblur,output_angles=Load_process(data_path,output_path,group_number)
        ddpm=DDPM.ddpm_forward(deblur_origin=deblur,angles=output_angles,group_number=group_number)
        ## segmentation
        deblur = segmentation(deblur)
        ddpm = segmentation(ddpm)
        ## load ground truth
        gt_path='/htc2022_0%s%s_recon_fbp_seg.mat'%(group_number,test_example[i])
        gt_path=data_folder+gt_path
        gt = io.loadmat(gt_path)['reconFullFbpSeg'].astype(np.float32)
        ## draw figures
        output_file_name=data_name[0:-4]+'_'+method_name+'_compare'+'.png'
        output_path=output_folder+output_file_name
        draw_figures(output_path,gt=gt,Bp=Bp,deblur=deblur,ddpm=ddpm,output_size=output_size,angles=output_angles)
    return 

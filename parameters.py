
## data path
data_path='/home/wangrongqian/CT/data-HTC/HTC-trainingset-256_angle30.h5'
val_data_path= data_path

## result path
result_path='./checkpoints/checkpoints'

## training parameters

img_resolution=256
## training set
sample_size_train=6800
## validation set
sample_id_val=6800
sample_size_val=10
## learning rate
learning_rate=1e-4
end_lr = 1e-7
## epochs
num_epochs=1000
checkpoint_epoch=0
## batch size
batchsize=8

## group number
group_number=7

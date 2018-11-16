
from CycleGAN import *

trainA_dir = '/home/xuagu37/CycleGAN/T1_subject_1_1000_slice_66.nii.gz'
trainB_dir = '/home/xuagu37/CycleGAN/FA_subject_1_1000_slice_66.nii.gz'
models_dir = '/home/xuagu37/CycleGAN/models'
batch_size = 10

myCycleGAN = CycleGAN(trainA_dir, trainB_dir, models_dir, batch_size)
myCycleGAN.train()

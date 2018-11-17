
# from importlib import reload  # Python 3.4+ only.
import CycleGAN #import the module here, so that it can be reloaded.
# reload(CycleGAN)
from CycleGAN import *

trainA_dir = '/home/xuagu37/CycleGAN/data/T1_subject_1_1000_slice_66.nii.gz'
trainB_dir = '/home/xuagu37/CycleGAN/data/FA_subject_1_1000_slice_66.nii.gz'
models_dir = '/home/xuagu37/CycleGAN/models'
batch_size = 1
epochs = 100
output_sample_flag = True
output_sample_dir = '/home/xuagu37/CycleGAN/output_sample.png'
normalization_factor_A = 1000
normalization_factor_B = 1
cycle_loss_type = 'L1_SSIM'

myCycleGAN = CycleGAN()
myCycleGAN.train(trainA_dir, normalization_factor_A, trainB_dir, normalization_factor_B, models_dir, batch_size, epochs, cycle_loss_type, output_sample_flag, output_sample_dir)


# G_X2Y = 'G_A2B'
# G_X2Y_dir = '/home/xuagu37/CycleGAN/G_A2B_weights_epoch_100.hdf5'
# test_X_dir = '/home/xuagu37/CycleGAN/data/T1_subject_1001_1065_slice_66.nii.gz'
# synthetic_Y_dir = '/home/xuagu37/CycleGAN/data/FA_subject_1001_1065_slice_66_synthetic.nii.gz'
# myCycleGAN.synthesize(G_X2Y, G_X2Y_dir, test_X_dir, synthetic_Y_dir)

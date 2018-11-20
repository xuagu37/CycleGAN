
import CycleGAN
from CycleGAN import *

# Create a CycleGAN
myCycleGAN = CycleGAN()

# Nifti data have the size [X, Y, C, N]
# [X, Y] is the slice size
# C is the channel size
# N is the number of data

# Train
trainA_dir = '/home/xuagu37/CycleGAN/data/T1_subject_1_1000_slice_66.nii.gz'
trainB_dir = '/home/xuagu37/CycleGAN/data/FA_subject_1_1000_slice_66.nii.gz'
models_dir = '/home/xuagu37/CycleGAN/models'
batch_size = 2
epochs = 100
output_sample_flag = True
output_sample_dir = '/home/xuagu37/CycleGAN/output_sample.png'
normalization_factor_A = 1000 # data/normalization_factor_A should be in [0, 1]
normalization_factor_B = 1
cycle_loss_type = 'L1'
myCycleGAN.train(trainA_dir, normalization_factor_A, trainB_dir, normalization_factor_B, models_dir, batch_size, epochs, cycle_loss_type, output_sample_flag, output_sample_dir)

# Synthesize from X to Y
G_X2Y = 'G_A2B'
G_X2Y_dir = '/home/xuagu37/CycleGAN/models/G_A2B_weights_epoch_40.hdf5'
test_X_dir = '/home/xuagu37/CycleGAN/data/T1_subject_1001_1065_slice_66.nii.gz'
normalization_factor_X = 1000
synthetic_Y_dir = '/home/xuagu37/CycleGAN/data/FA_subject_1001_1065_slice_66.nii.gz'
normalization_factor_Y = 1
myCycleGAN.synthesize(G_X2Y, G_X2Y_dir, test_X_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y)

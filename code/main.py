
import CycleGAN
from CycleGAN import *

# Create a CycleGAN on GPU 0
myCycleGAN = CycleGAN(0)

trainA_dir = '/home/xuagu37/CycleGAN/data/T1_training.nii.gz'
trainB_dir = '/home/xuagu37/CycleGAN/data/FA_training.nii.gz'
models_dir = '/home/xuagu37/CycleGAN/train_T1_FA/models'
output_sample_dir = '/home/xuagu37/CycleGAN/train_T1_FA/output_sample.png'
batch_size = 10
epochs = 200
normalization_factor_A = 1000
normalization_factor_B = 1
myCycleGAN.train(trainA_dir, normalization_factor_A, trainB_dir, normalization_factor_B, models_dir, batch_size, epochs, output_sample_dir=output_sample_dir, output_sample_channels=1)

for epoch in range(20, 201, 20):
    G_X2Y_dir = '/home/xuagu37/CycleGAN/train_T1_FA/models/G_A2B_weights_epoch_' + str(epoch) + '.hdf5'
    print(G_X2Y_dir)
    test_X_dir = '/home/xuagu37/CycleGAN/data/T1_test.nii.gz'
    synthetic_Y_dir = '/home/xuagu37/CycleGAN/train_T1_FA/synthetic/FA_synthetic_epoch_' + str(epoch) + '.nii.gz'
    normalization_factor_X = 1000
    normalization_factor_Y = 1
    myCycleGAN.synthesize(G_X2Y_dir, test_X_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y)


import CycleGAN
from CycleGAN import *

# Create a CycleGAN on GPU 0
myCycleGAN = CycleGAN(0)

trainA_dir = '/mnt/wd12t/CycleGAN/HCP/data/T1_subject_1_1000_slice_66.nii.gz'
trainB_dir = '/mnt/wd12t/CycleGAN/HCP/data/FA_subject_1_1000_slice_66.nii.gz'
models_dir = '/mnt/wd12t/CycleGAN/HCP/train_subject_1_1000_slice_66_T1_FA_us/models'
batch_size = 10
epochs = 200
output_sample_dir = '/mnt/wd12t/CycleGAN/HCP/train_subject_1001_1065_slice_66_T1_FA_us/output_sample.png'
normalization_factor_A = 1000
normalization_factor_B = 1
myCycleGAN.train(trainA_dir, normalization_factor_A, trainB_dir, normalization_factor_B, models_dir, batch_size, epochs, output_sample_dir=output_sample_dir, output_sample_channels=1, use_supervised_learning=True)

for epoch in range(20, 201, 20):
    G_X2Y_dir = '/mnt/wd12t/CycleGAN/HCP/train_subject_1_1000_slice_66_T1_FA_us/models/G_A2B_weights_epoch_' + str(epoch) + '.hdf5'
    print(G_X2Y_dir)
    test_X_dir = '/mnt/wd12t/CycleGAN/HCP/data/T1_subject_1001_1065_slice_66.nii.gz'
    normalization_factor_X = 1000
    synthetic_Y_dir = '/mnt/wd12t/CycleGAN/HCP/train_subject_1_1000_slice_66_T1_FA_us/synthetic/FA_synthetic_train_subject_1_1000_slice_66_test_subject_1001_1065_slice_66_epoch_' + str(epoch) + '.nii.gz'
    normalization_factor_Y = 1
    myCycleGAN.synthesize(G_X2Y_dir, test_X_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y)

# A Keras CycleGAN for nifti data  
We provide a Keras implementation for unpaired and image-to-image translation, i.e. CycleGAN [1], for nifti data.  

We proposed an application of CycleGAN to generate synthetic diffusion MRI scalar maps from structural T1-weighted images, see our paper [2].  

![T1 to FA/MD](https://github.com/xuagu37/CycleGAN/blob/master/images/T1_FA_MD.jpg)

## Getting started
### Prepare training data  
We prepare the training data by stacking subjects on the fourth dimention.  
For example, we extract 1 slice from each subject for 1000 subjects and then stack all slices to the fourth dimention.  
The created training data will have the size of [X, Y, 1, 1000].  
![T1_subject_1_1000_slice_66](https://github.com/xuagu37/CycleGAN/blob/master/images/T1_subject_1_1000_slice_66.png)

### Training  
\# Create a CycleGAN on GPU 0  
myCycleGAN = CycleGAN(0) 

\# Set directories  
trainA_dir = '/mnt/wd12t/CycleGAN/HCP/data/T1_subject_1_1000_slice_66.nii.gz'  
trainB_dir = '/mnt/wd12t/CycleGAN/HCP/data/FA_subject_1_1000_slice_66.nii.gz'  
models_dir = '/mnt/wd12t/CycleGAN/HCP/train_subject_1_1000_slice_66_T1_FA_us/models'  
output_sample_dir = '/mnt/wd12t/CycleGAN/HCP/train_subject_1001_1065_slice_66_T1_FA_us/output_sample.png'  

\# Set training parameters  
batch_size = 10  
epochs = 200  
normalization_factor_A = 1000  
normalization_factor_B = 1  

\# Start training  
myCycleGAN.train(trainA_dir, normalization_factor_A, trainB_dir, normalization_factor_B, models_dir, batch_size, epochs, output_sample_dir=output_sample_dir, output_sample_channels=1)


### Synthesize





[1] Zhu, J.Y., Park, T., Isola, P. and Efros, A.A., 2017. Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. In 2017 IEEE International Conference on Computer Vision (ICCV) (pp. 2242-2251). IEEE.  
[2] Gu, X., Knutsson, H., Nilsson, M. and Eklund, A., 2019. Generating diffusion MRI scalar maps from T1 weighted images using generative adversarial networks. In Scandinavian Conference on Image Analysis (pp. 489-498). Springer, Cham.


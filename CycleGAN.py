import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import nibabel as nib
from keras.layers import Dropout, Layer, Input, Conv2D, Activation, add, BatchNormalization, Conv2DTranspose, UpSampling2D
from keras_contrib.layers.normalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model
from keras.engine.topology import Network
from scipy.misc import imsave, toimage
import numpy as np
import random
import datetime
import time
import math
import sys
import keras.backend as K
import tensorflow as tf
import datetime



tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('CycleGAN loaded...')

class CycleGAN():
    def __init__(self, selected_gpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(selected_gpu)
        print('Initializing a CycleGAN on GPU ' + os.environ["CUDA_VISIBLE_DEVICES"])
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lr_D = 2e-4
        self.lr_G = 2e-4
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.supervised_weight = 10.0
        self.synthetic_pool_size = 50
        # optimizer
        self.opt_D = Adam(self.lr_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.lr_G, self.beta_1, self.beta_2)

        # TensorFlow wizardry
        config = tf.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
        # Create a session with the above options specified.
        session = tf.Session(config=config)
        K.tensorflow_backend.set_session(session)

    def create_discriminator_and_generator(self):
        print('Creating Discriminator and Generator ...')
        # Discriminator
        D_A = self.Discriminator()
        D_B = self.Discriminator()
        loss_weights_D = [0.5]
        image_A = Input(shape=self.data_shape)
        image_B = Input(shape=self.data_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B')
        self.D_A.compile(optimizer=self.opt_D, loss=self.lse, loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D, loss=self.lse, loss_weights=loss_weights_D)
        # Use containers to avoid falsy keras error about weight descripancies
        self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static')
        self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static')
        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        self.G_A2B = self.Generator(name='G_A2B')
        self.G_B2A = self.Generator(name='G_B2A')
        real_A = Input(shape=self.data_shape, name='real_A')
        real_B = Input(shape=self.data_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)
        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [self.cycle_loss, self.cycle_loss, self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2, self.lambda_D, self.lambda_D]
        model_outputs.append(dA_guess_synthetic)
        model_outputs.append(dB_guess_synthetic)
        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)
        self.G_model = Model(inputs=[real_A, real_B], outputs=model_outputs, name='G_model')
        self.G_model.compile(optimizer=self.opt_G, loss=compile_losses, loss_weights=compile_weights)

    def ck(self, x, k, use_normalization, stride):
        x = Conv2D(filters=k, kernel_size=4, strides=stride, padding='same', use_bias=True)(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same', use_bias=True)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1,1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=True)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=True)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=True)(x)
            #x = Dropout(0.1)(x, training=True)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same', use_bias=True)(x)  # this matches fractionally stided with stride 1/2
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        # x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Discriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.data_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False, 2)
        # Layer 2
        x = self.ck(x, 128, True, 2)
        # Layer 3
        x = self.ck(x, 256, True, 2)
        # Layer 4
        x = self.ck(x, 512, True, 1)
        # Output layer
        x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', use_bias=True)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def Generator(self, name=None):
        input_img = Input(shape=self.data_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)
        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)
            x = Dropout(self.dropout_rate)(x, training=True)
        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(filters=self.data_shape[2], kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)


    def train(self, train_A_dir, normalization_factor_A, train_B_dir, normalization_factor_B, models_dir, batch_size=10, epochs=200, cycle_loss_type='L1', use_resize_convolution=False, use_supervised_learning=False, output_sample_flag=True, output_sample_dir=None, output_sample_channels=1, dropout_rate=0):
        self.batch_size = batch_size
        self.epochs = epochs
        self.decay_epoch = self.epochs//2 # the epoch where linear decay of the learning rates starts
        self.cycle_loss_type = cycle_loss_type
        self.use_resize_convolution = use_resize_convolution
        self.use_supervised_learning = use_supervised_learning
        self.dropout_rate = dropout_rate
        # Data dir
        self.train_A_dir = train_A_dir
        self.train_B_dir = train_B_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.models_dir = models_dir
        self.train_A = load_data(self.train_A_dir, normalization_factor_A)
        self.train_B = load_data(self.train_B_dir, normalization_factor_B)
        # self.train_A, self.train_B = pad_data(self.train_A, self.train_B)
        self.data_shape = self.train_A.shape[1:4]
        self.data_num = self.train_A.shape[0]
        self.loop_num = self.data_num // self.batch_size
        print('Number of epochs: {}, number of loops per epoch: {}'.format(self.epochs, self.loop_num))
        self.create_discriminator_and_generator()

        # Image pools used to update the discriminators
        self.synthetic_A_pool = ImagePool(self.synthetic_pool_size)
        self.synthetic_B_pool = ImagePool(self.synthetic_pool_size)

        label_shape = (self.batch_size,) + self.D_A.output_shape[1:]
        ones = np.ones(shape=label_shape)
        zeros = ones * 0
        decay_D, decay_G = self.get_lr_linear_decay_rate()

        start_time = time.time()
        print("Dropout rate: {}".format(dropout_rate))
        print('Training ...')
        for epoch_i in range(self.epochs):
            # Update learning rates
            if epoch_i > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)
            random_indices = np.random.permutation(self.data_num)
            for loop_j in range(self.loop_num):
                # training data batches
                if self.use_supervised_learning:
                    random_indices_j = random_indices[loop_j*self.batch_size:(loop_j+1)*self.batch_size]
                    train_A_batch = self.train_A[random_indices_j]
                    train_B_batch = self.train_B[random_indices_j]
                else:
                    random_indices_j_A = random_indices[loop_j*self.batch_size:(loop_j+1)*self.batch_size]
                    random_indices_j_B = random_indices[loop_j*self.batch_size:(loop_j+1)*self.batch_size]
                    train_A_batch = self.train_A[random_indices_j_A]
                    train_B_batch = self.train_B[random_indices_j_B]
                # Synthetic data for training data batches
                synthetic_B_batch = self.G_A2B.predict(train_A_batch)
                synthetic_A_batch = self.G_B2A.predict(train_B_batch)
                synthetic_A_batch = self.synthetic_A_pool.query(synthetic_A_batch)
                synthetic_B_batch = self.synthetic_B_pool.query(synthetic_B_batch)

                # Train Discriminator
                DA_loss_train = self.D_A.train_on_batch(x=train_A_batch, y=ones)
                DB_loss_train = self.D_B.train_on_batch(x=train_B_batch, y=ones)
                DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_A_batch, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_B_batch, y=zeros)
                D_loss = DA_loss_train + DA_loss_synthetic + DB_loss_train + DB_loss_synthetic

                target_data = [train_A_batch, train_B_batch]
                target_data.append(ones)
                target_data.append(ones)
                if self.use_supervised_learning:
                    target_data.append(train_A_batch)
                    target_data.append(train_B_batch)
                # Train Generator
                G_loss = self.G_model.train_on_batch(x=[train_A_batch, train_B_batch], y=target_data)
                self.print_info(start_time, epoch_i, loop_j, D_loss, G_loss, DA_loss_train + DA_loss_synthetic, DB_loss_train + DB_loss_synthetic)
                if (output_sample_flag):
                    if (loop_j+1) % 5 == 0:
                        first_row = np.rot90(train_A_batch[0,:,:,0]) # training data A
                        second_row = np.rot90(train_B_batch[0,:,:,0]) # training data B
                        third_row = np.rot90(synthetic_B_batch[0,:,:,0]) # synthetic data B
                        if output_sample_channels>1:
                            for channel_i in range(output_sample_channels-1):
                                first_row = np.append(first_row, np.rot90(train_A_batch[0,:,:,channel_i+1]), axis=1)
                                second_row = np.append(second_row, np.rot90(train_B_batch[0,:,:,channel_i+1]), axis=1)
                                third_row = np.append(third_row, np.rot90(synthetic_B_batch[0,:,:,channel_i+1]), axis=1)
                        output_sample = np.append(np.append(first_row, second_row, axis=0), third_row, axis=0)
                        toimage(output_sample, cmin=-1, cmax=1).save(output_sample_dir)
            if (epoch_i+1) % 20 == 0:
                self.save_model(epoch_i)
        print("\u001b[12B")
        print("\u001b[1000D")
        print('Done')

    def synthesize(self, G_X2Y_dir, test_X_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y, use_resize_convolution=False, dropout_rate=0):
        test_X_img = nib.load(test_X_dir)
        test_X = load_data(test_X_dir, normalization_factor_X)
        self.data_shape = test_X.shape[1:4]
        self.data_num = test_X.shape[0]
        self.use_resize_convolution = use_resize_convolution
        self.dropout_rate = dropout_rate
        print('Synthesizing ...')
        print("Dropout rate: {}".format(dropout_rate))
        self.G_X2Y = self.Generator(name='G_X2Y')
        self.G_X2Y.load_weights(G_X2Y_dir)
        synthetic_Y = self.G_X2Y.predict(test_X)
        synthetic_Y = np.transpose(synthetic_Y, (1, 2, 3, 0))
        synthetic_Y = denormalize_data(synthetic_Y, normalization_factor_Y)
        synthetic_Y[synthetic_Y<0] = 0
        synthetic_Y = synthetic_Y[0:test_X_img.shape[0], 0:test_X_img.shape[1], :, :] # Remove padded zeros
        synthetic_Y_img = nib.Nifti1Image(synthetic_Y, test_X_img.affine, test_X_img.header)
        nib.save(synthetic_Y_img, synthetic_Y_dir)
        print('Done\n')

    def dropout_sample(self, G_X2Y_dir, test_X_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y, use_resize_convolution=False, dropout_rate=0, dropout_num=1):
        test_X_img = nib.load(test_X_dir)
        test_X = load_data(test_X_dir, normalization_factor_X)
        self.data_shape = test_X.shape[1:4]
        self.data_num = test_X.shape[0]
        self.use_resize_convolution = use_resize_convolution
        self.dropout_rate = dropout_rate
        self.G_X2Y = self.Generator(name='G_X2Y')
        self.G_X2Y.load_weights(G_X2Y_dir)
        print("Dropout rate: {}".format(dropout_rate))
        print("Dropout number: {}".format(dropout_num))
        for dropout_i in range(dropout_num):
            print("Dropout sample {}/{}".format(str(dropout_i+1), dropout_num))
            print("\u001b[3A")
            print("\u001b[1000D")
            sys.stdout.flush()
            synthetic_Y = self.G_X2Y.predict(test_X)
            synthetic_Y = np.transpose(synthetic_Y, (1, 2, 3, 0))
            synthetic_Y = denormalize_data(synthetic_Y, normalization_factor_Y)
            synthetic_Y[synthetic_Y<0] = 0
            synthetic_Y = synthetic_Y[0:test_X_img.shape[0], 0:test_X_img.shape[1], :, :] # Remove padded zeros
            synthetic_Y_img = nib.Nifti1Image(synthetic_Y, test_X_img.affine, test_X_img.header)
            nib.save(synthetic_Y_img, synthetic_Y_dir + "_" + str(dropout_i) + ".nii.gz")
        print("\u001b[1000D")
        print('Done\n')


    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        if self.cycle_loss_type == 'L1':
            # L1 norm
            loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        elif self.cycle_loss_type == 'L2':
            # L2 norm
            loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        elif self.cycle_loss_type == 'SSIM':
            # SSIM
            loss = 1 - tf.image.ssim(y_pred,y_true, max_val=1.0)[0]
        elif self.cycle_loss_type == 'L1_SSIM':
            # L1 + SSIM
            loss = 0.5*(1 - tf.image.ssim(y_pred,y_true, max_val=1.0)[0]) + 0.5*tf.reduce_mean(tf.abs(y_pred - y_true))
        elif self.cycle_loss_type == 'L2_SSIM':
            # L2 + SSIM
            loss = 0.5*(1 - tf.image.ssim(y_pred,y_true, max_val=1.0)[0]) + 0.5*tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        elif self.cycle_loss_type == 'L1_L2_SSIM':
            # L1 + L2 + SSIM
            loss = 1/3*(1 - tf.image.ssim(y_pred,y_true, max_val=1.0)[0]) + 1/3*tf.reduce_mean(tf.abs(y_pred - y_true)) + 1/3*tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def get_lr_linear_decay_rate(self):
        updates_per_epoch_D = 2 * self.data_num
        updates_per_epoch_G = self.data_num
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.lr_D / denominator_D
        decay_G = self.lr_G / denominator_G
        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        K.set_value(model.optimizer.lr, new_lr)

    def print_info(self, start_time, epoch_i, loop_j, D_loss, G_loss, DA_loss, DB_loss):
        print("\n")
        print("Epoch               : {:d}/{:d}{}".format(epoch_i + 1, self.epochs, "                         "))
        print("Loop                : {:d}/{:d}{}".format(loop_j + 1, self.loop_num, "                         "))
        print("D_loss              : {:5.4f}{}".format(D_loss, "                         "))
        print("G_loss              : {:5.4f}{}".format(G_loss[0], "                         "))
        print("reconstruction_loss : {:5.4f}{}".format(G_loss[3]+ G_loss[4], "                         "))
        print("DA_loss             : {:5.4f}{}".format(DA_loss, "                         "))
        print("DB_loss             : {:5.4f}{}".format(DB_loss, "                         "))
        passed_time = (time.time() - start_time)
        loops_finished = epoch_i * self.loop_num + loop_j
        loops_total = self.epochs * self.loop_num
        loops_left = loops_total - loops_finished
        remaining_time = (passed_time / (loops_finished + 1e-5) * loops_left)
        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        remaining_time_string = str(datetime.timedelta(seconds=round(remaining_time)))
        print("Time passed         : {}{}".format(passed_time_string, "                         "))
        print("Time remaining      : {}{}".format(remaining_time_string, "                         "))
        print("\u001b[13A")
        print("\u001b[1000D")
        sys.stdout.flush()

    def save_model(self, epoch_i):
        models_dir_epoch_i = os.path.join(self.models_dir, '{}_weights_epoch_{}.hdf5'.format(self.G_A2B.name, epoch_i+1))
        self.G_A2B.save_weights(models_dir_epoch_i)
        models_dir_epoch_i = os.path.join(self.models_dir, '{}_weights_epoch_{}.hdf5'.format(self.G_B2A.name, epoch_i+1))
        self.G_B2A.save_weights(models_dir_epoch_i)

def normalize_data(data, normalization_factor):
    # Normalize data to [-1, 1]
    if np.array(normalization_factor).size == 1:
        data = data/normalization_factor
    else:
        for i in range(data.shape[2]):
            data[:,:,i,:] = data[:,:,i,:]/normalization_factor[i] # normalize data for each channel
    data = data*2-1
    return data

def denormalize_data(data, normalization_factor):
    # Denormalize data to [-1, 1]
    data = (data+1)/2
    if np.array(normalization_factor).size == 1:
        data = data*normalization_factor
    else:
        for i in range(data.shape[2]):
            data[:,:,i,:] = data[:,:,i,:]*normalization_factor[i] # normalize data for each channel
    return data

def load_data(data_dir, normalization_factor):
        data = nib.load(data_dir).get_fdata()
        data[data<0] = 0
        if data.ndim == 2:
            data = data[:,:,np.newaxis, np.newaxis]
        data = normalize_data(data, normalization_factor)
        data = np.transpose(data, (3, 0, 1, 2))
        print('Loading data, data size: {}, number of data: {}'.format(data.shape[1:4], data.shape[0]))
        # Make sure that slice size is multiple 4
        if (data.shape[1]%4 != 0):
            data = np.append(data, np.zeros((data.shape[0], 4-data.shape[1]%4, data.shape[2], data.shape[3]))-1, axis=1)
        if (data.shape[2]%4 != 0):
            data = np.append(data, np.zeros((data.shape[0], data.shape[1], 4-data.shape[2]%4, data.shape[3]))-1, axis=2)
        return data

def pad_data(data_A, data_B):
    size_n = data_A.shape[0]
    size_x_A = data_A.shape[1]
    size_y_A = data_A.shape[2]
    size_c_A = data_A.shape[3]
    size_x_B = data_B.shape[1]
    size_y_B = data_B.shape[2]
    size_c_B = data_B.shape[3]
    size_x_new = np.maximum(size_x_A, size_x_B)
    size_y_new = np.maximum(size_y_A, size_y_B)
    size_c_new = np.maximum(size_c_A, size_c_B)

    data_A_new = -np.ones((size_n, size_x_new, size_y_new, size_c_new))
    data_B_new = -np.ones((size_n, size_x_new, size_y_new, size_c_new))
    data_A_new[:, int((size_x_new-size_x_A)/2):int((size_x_new-size_x_A)/2)+size_x_A, int((size_y_new-size_y_A)/2):int((size_y_new-size_y_A)/2)+size_y_A, 0:size_c_A] = data_A
    data_B_new[:, int((size_x_new-size_x_B)/2):int((size_x_new-size_x_B)/2)+size_x_B, int((size_y_new-size_y_B)/2):int((size_y_new-size_y_B)/2)+size_y_B, 0:size_c_B] = data_B

    return data_A_new, data_B_new

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)
    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])
    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))
            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))
        return return_images

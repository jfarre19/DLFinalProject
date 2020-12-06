import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# GAN class
class GAN():
    def __init__(self, data_generator, g_model=None, d_model=None):
        self.data_generator = data_generator
        self.g_model = g_model
        self.d_model = d_model
        if g_model is None:
            self.g_model = self.generator(dropout=0.4, depth=32, alpha=0.2)
        if d_model is None:
            self.d_model = self.discriminator(dropout=0.4, depth=32, alpha=0.2)
            
        # compile gan model
        self.gan_model = self.gan()

    # randomly generate latent space for generator input
    def gen_latent_space(self, latent_dim, batch_size):
        x_input = np.random.randn(latent_dim * batch_size)
        x_input = x_input.reshape(batch_size, latent_dim)
        return x_input

    def train(self, epochs):
        self.batch_size = self.data_generator.batch_size
        self.epochs = epochs

        batch_per_epo = self.data_generator.__len__()
        # half_batch = self.batch_size // 2
        
        # ground truth labels
        y_valid = np.ones((self.batch_size, 1)) 
        y_fake = np.zeros((self.batch_size, 1))
        
        # iterate over epochs and epochs_per_batch
        for i in range(self.epochs):
            for j in range(batch_per_epo):
                # collect batch data from datagenerator class
                [X_real, labels_real] = self.data_generator.__getitem__(j)

                # train discriminator on current 'real' batch data
                d_loss1, _ = self.d_model.train_on_batch([X_real, labels_real], y_valid)

                # train discriminator on 'fake' generated batch data
                latent_space = self.gen_latent_space(100, self.batch_size) # locally defined function
                X_fake = self.g_model.predict([latent_space, labels_real]) # generate fake data
                d_loss2, _ = self.d_model.train_on_batch([X_fake, labels_real], y_fake) # train 'fake data' on batch
                
                # prepare points in latent space as input for the generator
                z_input = self.gen_latent_space(100, 2 * self.batch_size)
                labels_fake = np.random.randint(0, self.data_generator.n_classes, 2 * self.batch_size)
                # create inverted labels for the fake samples
                y_gan = np.ones((2 * self.batch_size, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan_model.train_on_batch([z_input, labels_fake], y_gan)

                # summarize loss on this batch
                if j % 1000 == 0: 
                    print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                        (i+1, j+1, batch_per_epo, d_loss1, d_loss2, g_loss))
        # save models
        self.g_model.save('cgan_generator.h5')
        self.d_model.save('cgan_discriminator.h5')
        self.gan_model.save('cgan_GAN.h5')
    
    # define the combined generator and discriminator model, for updating the generator
    def gan(self):
        self.d_model.trainable = False
                
        # pipe generator inputs/outputs to discriminator
        gen_noise, gen_label = self.g_model.input
        gen_output = self.g_model.output
        gan_output = self.d_model([gen_output, gen_label])
        
        # finalize gan model definition
        # inputs: generator noise and label
        # output: discriminator output
        model = Model([gen_noise, gen_label], gan_output)
        
        # compile combined gan model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def discriminator(self, dropout=0.4, depth=64, alpha=0.2):        
        label = Input(shape=(1,))
        x = Embedding(345,50)(label)
        x = Dense(784)(x)
        l = Reshape((28,28,1))(x)
        
        image = Input(shape=(28,28,1))
        
        concat = Concatenate()([image,l])
        x = Conv2D(1*depth,5,strides=2,padding='same')(concat)
        x = Dropout(dropout)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Conv2D(2*depth,5,strides=2,padding='same')(x)
        x = Dropout(dropout)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Conv2D(4*depth,5,strides=2,padding='same')(x)
        x = Dropout(dropout)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Conv2D(8*depth,5,strides=1,padding='same')(x)
        x = Dropout(dropout)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Flatten()(x)
        out = Dense(1,activation='sigmoid')(x)
        
        model = Model(inputs=[image,label], outputs=out)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def generator(self, dropout=0.4, depth=32, alpha=0.2):  
        # category input (cGAN addition)      
        label = Input(shape=(1,))
        x = Embedding(345,50)(label)
        x = Dense(49)(x)
        l = Reshape((7,7,1))(x)
        
        # latent variables input (random inputs)
        z = Input(shape=(100,))
        x = Dense((8*depth-1)*49)(z)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Dropout(dropout)(x)
        image = Reshape((7,7,8*depth-1))(x)
        
        # combination of both inputs
        concat = Concatenate()([image,l])
        x = UpSampling2D()(concat)
        x = Conv2DTranspose(4*depth,5,padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(2*depth,5,padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Conv2DTranspose(1*depth,5,padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=alpha)(x)
        out = Conv2DTranspose(1,5,padding='same',activation='sigmoid')(x)
        
        model = Model(inputs=[z,label], outputs=out)
        return model
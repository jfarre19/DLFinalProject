from keras.models import Sequential, Model
from keras import Input
from keras.layers import Dense
from keras.layers import Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, BatchNormalization, Dropout
from keras.layers import Embedding

from keras.optimizers import Adam

# GAN class
class GAN():
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.generator = generator()
        self.discriminator = discriminator()
        self.gan = gan(self.generator, self.discriminator)

        # train(self.generator, self.discriminator, self.gan, self.batch_size, self.epochs)

    # randomly generate latent space for generator input
    def gen_latent_space(latent_dim, batch_size):
        x_input = randn(latent_dim * batch_size)
        x_input = x_input.reshape(batch_size, latent_dim)
        return x_input

    def train(batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs

        bat_per_epo = int(dataset[0].shape[0] / n_batch)
        
        # FIXME use this?
        half_batch = int(n_batch // 2)
        
        # iterate over epochs and epochs_per_batch
        for i in range(epochs):
            for j in range(batch_per_epo):

                # collect batch data from datagenerator class
                [X_real, labels_real] = self.data_generator.__getitem__(j)
                y_real = np.ones((half_batch, 1))

                # train discriminator on current 'real' batch data
                d_loss1, _ = discriminator.train_on_batch([X_real, labels_real], y_real)

                # train discriminator on 'fake' generated batch data
                latent_space = gen_latent_space(100, half_batch)                        # locally defined function
                [X_fake, labels_fake] = generator.predict(latent_space, category)       # create data
                y_fake = np.zeros((half_batch, 1))                                      # tell discriminator that these are fake
                d_loss2, _ = discriminator.train_on_batch([X_fake, labels_fake], y_fake)  # train 'fake data' on batch

                # prepare points in latent space as input for the generator
                [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # save the generator model
        g_model.save('cgan_generator.h5')

    
    # define the combined generator and discriminator model, for updating the generator
    def gan(generator, discriminator):
        # freeze discriminator to train generator
        discriminator.trainable = False

        # pipe generator inputs/outputs to discriminator
        gen_noise, gen_label = generator.input
        gen_output = generator.output
        gan_output = discriminator([gen_output, gen_label])
        
        # finalize gan model definition
        # inputs: generator noise and label
        # output: discriminator output
        model = Model([gen_noise, gen_label], gan_output)
        
        # compile combined gan model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def discriminator(dropout=0.4, depth=64, alpha=0.2):        
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

    def generator(dropout=0.4, depth=32, alpha=0.2):  
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
        # John -- generator not trained directly -> no compilation/optimizer needed
        # opt = Adam(lr=0.0002, beta_1=0.5)
        # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
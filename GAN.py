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
    def __init__(self, data_generator, btach_size, epochs):
        self.data_generator = data_generator
        self.generator = generator()
        self.discriminator = discriminator()
        self.gan = gan()
        train(self.generator, self.discriminator, self.gan, batch_size epochs)

    # randomly generate latent space for generator input
    def gen_latent_space(latent_dim, batch_size):
        x_input = randn(latent_dim * batch_size)
        x_input = x_input.reshape(batch_size, latent_dim)
        return x_input

    # from my understanding, training half batch real and fake = 1 full batch
    # this is hopefully still training well enough and equally distributing the real/fake for a single batch    
    # This does not translate to our Keras.sequence... I am leaving the code for now, but basically it isn't worth reading
    def train(generator, discriminator, gan, data_generator, batch_size, epochs):
        # get real samples and train discriminator
        # returns a history object (if we want that data). We can pull loss from this
        loss1, _ = discriminator.fit(x=data_generator, batch_size=batch_size//2, epochs=epochs)

        # generate fake examples from generator and train discriminator 
        # I was a little confused how specifically to generate these samples for a cGAN -- like iterating over categories or something?
        # returns another history object
        latent_space = gen_latent_space(100, batch_size//2)
        generator.predict(latent_space, category)
        loss2, _ = discriminator.fit()
        

        # get randomly selected 'real' samples
        # update discriminator model weights
        # generate 'fake' examples
        # update discriminator model weights
        # prepare points in latent space as input for the generator
        # create inverted labels for the fake samples
        # update the generator via the discriminator's error

    
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
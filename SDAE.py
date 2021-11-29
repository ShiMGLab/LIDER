from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
import math


class SDAE(object):
    
    
    def __init__(self, dims, act='relu', drop_rate=0.2, batch_size=32,actinlayer1="tanh",init="glorot_uniform"): #act relu
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.n_layers = 2*self.n_stacks  # exclude input layer
        self.activation = act
        #self.actinlayer1="tanh" #linear
        self.actinlayer1=actinlayer1 #linear
        self.drop_rate = drop_rate
        self.init=init
        self.batch_size = batch_size
        self.stacks = [self.make_stack(i) for i in range(self.n_stacks)]
        self.autoencoders ,self.encoder= self.make_autoencoders()
    def make_autoencoders(self):
        
       
        # input
        x = Input(shape=(self.dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(self.n_stacks-1):
            h = Dense(self.dims[i + 1], kernel_initializer=self.init,activation=self.activation, name='encoder_%d' % i)(h)

        # hidden layer,default activation is linear
        h = Dense(self.dims[-1],kernel_initializer=self.init, name='encoder_%d' % (self.n_stacks - 1),activation=self.actinlayer1)(h)  # features are extracted from here

        y=h
        # internal layers in decoder       
        for i in range(self.n_stacks-1, 0, -1): #2,1
            y = Dense(self.dims[i], kernel_initializer=self.init,activation=self.activation, name='decoder_%d' % i)(y)

        # output
        y = Dense(self.dims[0], kernel_initializer=self.init,name='decoder_0',activation=self.actinlayer1)(y)

        return Model(inputs=x, outputs=y,name="AE"),Model(inputs=x,outputs=h,name="encoder")

    def make_stack(self, ith):
       
        in_out_dim = self.dims[ith]
        hidden_dim = self.dims[ith+1]
        output_act = self.activation
        hidden_act = self.activation
        if ith == 0:
            output_act = self.actinlayer1# tanh, or linear
        if ith == self.n_stacks-1:
            hidden_act = self.actinlayer1 #tanh, or linear
        model = Sequential()
        model.add(Dropout(self.drop_rate, input_shape=(in_out_dim,)))
        model.add(Dense(units=hidden_dim, activation=hidden_act, name='encoder_%d' % ith))
        model.add(Dropout(self.drop_rate))
        model.add(Dense(units=in_out_dim, activation=output_act, name='decoder_%d' % ith))

        #plot_model(model, to_file='stack_%d.png' % ith, show_shapes=True)
        return model

    def pretrain_stacks(self, x, epochs=200):
        
        print("Doing SDAE: pretrain_stacks")  
        features = x
        for i in range(self.n_stacks):
            print( 'Pretraining the %dth layer...' % (i+1))
            for j in range(3):  # learning rate multiplies 0.1 every 'epochs/3' epochs
                print ('learning rate =', pow(10, -1-j))
                self.stacks[i].compile(optimizer=SGD(pow(10, -1-j), momentum=0.9), loss='mse')
                #callbacks=[EarlyStopping(monitor='loss',min_delta=10e-4,patience=4,verbose=1,mode='auto')]
                #self.stacks[i].fit(features, features, batch_size=self.batch_size, callbacks=callbacks,epochs=math.ceil(epochs/3))
                self.stacks[i].fit(features, features, batch_size=self.batch_size,epochs=math.ceil(epochs/3),verbose=0)
            print ('The %dth layer has been pretrained.' % (i+1))

            # update features to the inputs of the next layer
            feature_model = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder_%d'%i).output)
            features = feature_model.predict(features)

    def pretrain_autoencoders(self, x, epochs=500):
        
        print("Doing SDAE: pretrain_autoencoders")  
        print ('Copying layer-wise pretrained weights to deep autoencoders')
        for i in range(self.n_stacks):
            name = 'encoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
            name = 'decoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())

        print ('Fine-tuning autoencoder end-to-end')
        for j in range(math.ceil(epochs/50)):
            lr = 0.1*pow(10, -j)
            print ('learning rate =', lr)
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            #callbacks=[EarlyStopping(monitor='loss',min_delta=10e-4,patience=4,verbose=1,mode='auto')]
            #self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50,callbacks=callbacks)
            self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50,verbose=0)

    def fit(self, x, epochs=200):
        self.pretrain_stacks(x, epochs=epochs/2)
        self.pretrain_autoencoders(x, epochs=epochs)#fine tunning

    def fit2(self,x,epochs=200): #no stack directly traning 
        for j in range(math.ceil(epochs/50)):
            lr = 0.1*pow(10, -j)
            print ('learning rate =', lr)
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)

    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders.
        
        :param x: data
        :return: features
        """
        hidden_layer = self.autoencoders.get_layer(name='encoder_%d' % (self.n_stacks - 1))
        feature_model = Model(self.autoencoders.input, hidden_layer.output)
        return feature_model.predict(x, batch_size=self.batch_size)


if __name__ == "__main__":
    
    import numpy as np
    import scipy.io as sio

    
    db = 'zeisel'
    x = sio.loadmat('.../zeisel_normal.mat')
    # define and train SAE model
    sdae = SDAE(dims=[x.shape[-1], 500, 500, 2000, 1000])
    sdae.fit(x=x, epochs=400)
    

    # extract features
    print ('Finished training, extracting features using the trained SDAE model')
    features = sdae.extract_feature(x)
    np.savetxt(".../zeisel_features.txt",features)
    

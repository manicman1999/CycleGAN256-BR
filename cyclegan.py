# -*- coding: utf-8 -*-
"""

@author: Matthew Mann

DomainA / DomainB = Images[] - Pairs
DomainAs / DomainBs = Amages[] - Non-Pairs

"""



import numpy as np
from PIL import Image
from math import floor
import time

#Noise Level
nlev = 16

def zeros(n):
    return np.random.uniform(0.0, 0.01, size = [n, 1])

def ones(n):
    return np.random.uniform(0.99, 1.0, size = [n, 1])

def get_rand(array, amount):
    
    idx = np.random.randint(0, array.shape[0], amount)
    return array[idx]

def adjust_hue(image, amount):
    t0 = Image.fromarray(np.uint8(image*255))
    t1 = t0.convert('HSV')
    t2 = np.array(t1, dtype='float32')
    t2 = t2 / 255
    t2[...,0] = (t2[...,0] + amount) % 1
    t3 = Image.fromarray(np.uint8(t2*255), mode = "HSV")
    t4 = np.array(t3.convert('RGB'), dtype='float32') / 255
    
    return t4

#Import Images Function
def import_images(loc, n):
    
    out = []
    
    for n in range(1, n + 1):
        temp = Image.open("data/"+loc+"/im ("+str(n)+").png")
        
        temp = np.array(temp.convert('RGB'), dtype='float32')
        
        out.append(temp / 255)
        
        out.append(np.flip(out[-1], 1))
    
    return np.array(out)

#Keras Imports
from keras.models import model_from_json, Model, Sequential
from keras.layers import Conv2D, LeakyReLU, AveragePooling2D, BatchNormalization, Dense
from keras.layers import UpSampling2D, Activation, Dropout, concatenate, Input, Flatten
from keras.optimizers import Adam
#import keras.backend as K


#Defining Layers For U-Net
def conv(input_tensor, filters, bn = True, drop = 0):
    
    co = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(input_tensor)
    ac = LeakyReLU(0.2)(co)
    ap = AveragePooling2D()(ac)
    
    if bn:
        ap = BatchNormalization(momentum = 0.75)(ap)
        
    if drop > 0:
        ap = Dropout(drop)(ap)
    
    return ap

def deconv(input1, input2, filters, drop = 0):
    #Input 1 Should be half the size of Input 2
    up = UpSampling2D()(input1)
    co = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(up)
    ac = Activation('relu')(co)
    
    if drop > 0:
        ac = Dropout(drop)(ac)
        
    ba = BatchNormalization(momentum = 0.75)(ac)
    con = concatenate([ba, input2])
    
    return con

def d_block(f, b = True, p = True):
    temp = Sequential()
    temp.add(Conv2D(filters = f, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform'))
    if b:
        temp.add(BatchNormalization(momentum = 0.8))
    temp.add(Activation('relu'))
    if p:
        temp.add(AveragePooling2D())
        
    return temp



#Define The Actual Model Class
class GAN(object):
    
    def __init__(self):
        
        #Always 256x256 Images
        
        #Models
        
        #Generator (Domain A -> Domain B)
        self.G1 = None
        self.G2 = None
        
        #Discriminator (Domain B)
        self.D = None
        
        #Old Models For Rollback After Training Others
        self.OD = None
        self.OG = None
        self.OG2 = None
        self.OE = None
        
        #Training Models
        self.DM = None #Discriminator Model (D)
        self.AM = None #Adversarial Model (G1 + D)
        self.RM = None #Reconstruction Model (G1 + G2)
        
        
        #Other Config
        self.LR = 0.0002 #Learning Rate
        self.steps = 1 #Training Steps Taken
    
    def generator1(self):
        
        #Defining G1 // U-Net
        if self.G1:
            return self.G1
        
        #Image Input
        inp = Input(shape = [256, 256, 3])
        #128
        d1 = conv(inp, 8)
        #64
        d2 = conv(d1, 16)
        #32
        d3 = conv(d2, 32)
        #16
        d4 = conv(d3, 64)
        #8
        d5 = conv(d4, 128)
        d6 = conv(d5, 192)
        #4
        
        center = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(d6)
        ac = LeakyReLU(0.2)(center)
        
        #4
        u6 = deconv(ac, d5, 192)
        u5 = deconv(u6, d4, 128)
        u4 = deconv(u5, d3, 64)
        u3 = deconv(u4, d2, 32)
        u2 = deconv(u3, d1, 16)
        #64
        
        u1 = UpSampling2D()(u2)
        cc = concatenate([inp, u1])
        cl = Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(cc)
        #128
        out = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(cl)
        
        self.G1 = Model(inputs = inp, outputs = out)
        
        return self.G1
    
    def generator2(self):
        
        #Defining G2 // U-Net
        if self.G2:
            return self.G2
        
        #Image Input
        inp = Input(shape = [256, 256, 3])
        #128
        d1 = conv(inp, 8)
        #64
        d2 = conv(d1, 16)
        #32
        d3 = conv(d2, 32)
        #16
        d4 = conv(d3, 64)
        #8
        d5 = conv(d4, 128)
        d6 = conv(d5, 192)
        #4
        
        center = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(d6)
        ac = LeakyReLU(0.2)(center)
        
        #4
        u6 = deconv(ac, d5, 192)
        u5 = deconv(u6, d4, 128)
        u4 = deconv(u5, d3, 64)
        u3 = deconv(u4, d2, 32)
        u2 = deconv(u3, d1, 16)
        #64
        
        u1 = UpSampling2D()(u2)
        cc = concatenate([inp, u1])
        cl = Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(cc)
        #128
        out = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(cl)
        
        self.G2 = Model(inputs = inp, outputs = out)
        
        return self.G2
    
    def discriminator(self):
        
        if self.D:
            return self.D
        
        self.D = Sequential()
        
        self.D.add(Activation('linear', input_shape = [256, 256, 3]))
        
        #256
        self.D.add(d_block(8)) #128
        self.D.add(d_block(16)) #64
        self.D.add(d_block(32)) #32
        self.D.add(d_block(64)) #16
        self.D.add(d_block(96)) #32
        self.D.add(d_block(128)) #8
        self.D.add(d_block(192)) #4
        self.D.add(d_block(256, p = False)) #4
        self.D.add(Flatten())
        
        #8192
        
        self.D.add(Dropout(0.6))
        self.D.add(Dense(1, activation = 'linear'))
        
        return self.D
    
    def DisModel(self):
        
        #Defining DM
        if self.DM == None:
            self.DM = Sequential()
            self.DM.add(self.discriminator())
        
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.DM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))),
                        loss = 'mse')
        
        return self.DM
    
    def AdModel(self):
        
        #Defining AM
        if self.AM == None:
            self.AM = Sequential()
            self.AM.add(self.generator1())
            self.AM.add(self.discriminator())
        
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.AM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))),
                        loss = 'mse')
        
        return self.AM
    
    def RecModel(self):
        
        if self.RM == None:
            self.RM = Sequential()
            self.RM.add(self.generator1())
            self.RM.add(self.generator2())
        
        self.RM.compile(optimizer = Adam(lr = self.LR * 0.5 * (0.9 ** floor(self.steps / 10000))), loss = 'mae')
        
        return self.RM
    
    def sod(self):
        
        #Save Old Discriminator
        self.OD = self.D.get_weights()
    
    def lod(self):
        
        #Load Old Discriminator
        self.D.set_weights(self.OD)
        

#Now Define The Actual Model
class CycleGAN(object):
    
    def __init__(self, steps = -1, silent = True):
        
        #Models
        #Main
        self.GAN = GAN()
        
        #Set Steps, If Relevant
        if steps >= 0:
            self.GAN.steps = steps
        
        #Generators
        self.G1 = self.GAN.generator1()
        
        
        #Training Models
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.RecModel = self.GAN.RecModel()
        self.lastblip = time.clock()
        
        #Std Deviation
        self.std_dev = 1 + 4.0 / ((self.GAN.steps + 20000.0) / 10000.0)
        
        #Images
        self.ImagesA = import_images("DomainA", 962)
        self.ImagesB = import_images("DomainB", 403)
        
        self.silent = silent
        
    def train(self, batch = 2):
        
        #Train and Get Losses
        al = self.train_dis(batch)
        bl = self.train_gen(batch)
        cl = self.train_rec(batch)
        
        #Every 20 Steps Display Info
        if self.GAN.steps % 20 == 0 and not self.silent:
            ti = round((time.clock() - self.lastblip) * 100.0) / 100.0
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("Dis: " + str(round(al, 5)))
            print("Gen: " + str(round(bl, 5)))
            print("Rec: " + str(round(cl, 5)))
            print("T::: " + str(ti) + " seconds")
            self.lastblip = time.clock()
        
        #Save Every 500 steps
        if self.GAN.steps % 500 == 0:
            self.save(floor(self.GAN.steps / 10000))
            #self.adjust()
            #self.evaluate()
        
        #Re-Compile (Update Learning Rate) Every 10k Steps
        if self.GAN.steps % 10000 == 0:
            self.GAN.AM = None
            self.GAN.DM = None
            self.GAN.RM = None
            self.AdModel = self.GAN.AdModel()
            self.DisModel = self.GAN.DisModel()
            self.RecModel = self.GAN.RecModel()
        
        self.GAN.steps = self.GAN.steps + 1
        
        return True
    
    def train_dis(self, batch):
        
        #Get Real Images
        train_data = get_rand(self.ImagesB, batch)
        label_data = ones(batch)
            
        d_loss_real = self.DisModel.train_on_batch(train_data, label_data)
        
        #Get Fake Images
        train_data = self.G1.predict(get_rand(self.ImagesA, batch))
        label_data = zeros(batch)
            
        d_loss_fake = self.DisModel.train_on_batch(train_data, label_data)
        
        return (d_loss_real + d_loss_fake) * 0.5
        
    def train_gen(self, batch):
        
        self.GAN.sod()
        
        train_data = get_rand(self.ImagesA, batch)
        label_data = ones(batch)
        
        g_loss = self.AdModel.train_on_batch(train_data, label_data)
        
        self.GAN.lod()
        
        return g_loss
    
    def train_rec(self, batch):
        
        train_data = get_rand(self.ImagesA, batch)
        
        g_loss = self.RecModel.train_on_batch(train_data, train_data)
        
        return g_loss
        
    def evaluate(self, num):
        
        row = []
        
        #From left to right: Labels, GT, 6xGenerated Images
        
        #With Matching Ground Truth Image
        for _ in range(8):
            im = get_rand(self.ImagesA, 4)
            out = self.G1.predict(im)
            s = np.concatenate([im[0], out[0], im[1], out[1], im[2], out[2], im[3], out[3]], axis = 1)
            row.append(s)
        
        image = np.concatenate(row[0:8], axis = 0)
        
        x = Image.fromarray(np.uint8(image*255))
        
        x.save("Results/i"+str(num)+".png")
        
        del row, image, x
            
    
    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)
            
        model.save_weights("Models/"+name+"_"+str(num)+".h5")
        
    def loadModel(self, name, num):
        
        file = open("Models/"+name+".json", 'r')
        json = file.read()
        file.close()
        
        mod = model_from_json(json)
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")
        
        return mod
    
    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.G1, "gen1", num)
        self.saveModel(self.GAN.D, "dis", num)
        self.saveModel(self.GAN.G2, "gen2", num)
        

    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()
        
        self.GAN.steps = steps1

        #Load Models
        self.GAN.G1 = self.loadModel("gen1", num)
        self.GAN.D = self.loadModel("dis", num)
        self.GAN.G2 = self.loadModel("gen2", num)
        
        self.AdModel = self.GAN.AdModel()
        self.DisModel = self.GAN.DisModel()
        self.RecModel = self.GAN.RecModel()
        self.G1 = self.GAN.generator1()
        
    def sample(self, inp):
            
        return self.G1.predict(inp)

#Finally Onto The Main Function
if __name__ == "__main__":
    model = CycleGAN(139999, silent = False)
    model.load(12)
    
    while(True):
        model.train(2)
        
        if model.GAN.steps % 12000 == 0:
            time.sleep(15)
        
        #Evaluate Every 1k Steps
        if model.GAN.steps % 1000 == 0:
            model.evaluate(floor(model.GAN.steps / 1000))

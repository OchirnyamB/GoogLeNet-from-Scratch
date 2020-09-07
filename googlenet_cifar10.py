import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from model.minigooglenet import MiniGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler 
from callbacks.trainingmonitor import TrainingMonitor
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import os 

NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR*(1-(epoch/float(maxEpochs)))**power
    return alpha

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--model", required=True, help="path to output model")
ap.add_argument("-m", "--output", required=True, help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

# load the training and testing data, converting the images from integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subraction to the data
mean = np.mean(trainX, axis = 0)
trainX -= mean
testX  -= mean

# convert the baels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]

# initialize the optimizer and model
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), steps_per_epoch=len(trainX)//64, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

# save the network to disk
print("[INFO] serializing to disk...")
model.save(args["model"])

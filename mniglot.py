import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import glob
from skimage import io
import os
import scipy.misc
from scipy.misc import imread, imresize
from keras import regularizers
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


datasets_path = "/home/disk2/internship_anytime/zhangao/omni/omniglot/python/" 
def load_images(path,n=0):
    X = []
    Y=[]
    i=-1
    
    for back in os.listdir(path):
        back_path = os.path.join(path,back)
        for language in os.listdir(back_path):
            #print ("loading alphabet:" + alphabet)
            #Y.append(alphabet)
            alphabet_path = os.path.join(back_path,language)
            for letter in os.listdir(alphabet_path):
                category_images = []
                i=i+1
                letter_path = os.path.join(alphabet_path,letter)

                for filename in os.listdir(letter_path):
                    image_path = os.path.join(letter_path,filename)
                    image = imread(image_path)
                    #Y.append(image)
                    image = imresize(image,(28,28))
                    image = image/255
                    image = 1-image
                    Y.append(i)
                    X.append(image)
    return X,Y

print("training set")
x_train,y_train = load_images(datasets_path)


X = np.array(x_train)
y = np.array(y_train)
#print(y)
print("yshape:",y.shape)
print("xshape:",X.shape)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1,random_state = 42)
print("train",X_train.shape)
print("test",X_test.shape)



number_of_classes = 1623
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3 )))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1623))

model.add(Activation('softmax'))

gen = ImageDataGenerator()

test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=32)
test_generator = test_gen.flow(X_test, Y_test, batch_size=32)
lr_lsit=[0.01,0.001,0.0001,0.005,0.05,0.0005]
for lr in lr_list:
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr),metrics=['accuracy'])
    history=model.fit_generator(train_generator, epochs=100, validation_data=test_generator)
    plt.plot(history.history['val_acc'])
plt.title('val_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(lr_lsit,loc='upper left')
plt.savefig("acc.png")
score = model.evaluate(X_test, Y_test)
print()
print('Test loss: ', score[0])
print('Test Accuracy', score[1])

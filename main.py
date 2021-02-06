import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import os
import random

Train_Dir = './data/train.csv'
Test_Dir = './data/test.csv'
train_data = pd.read_csv(Train_Dir)
test_data = pd.read_csv(Test_Dir)

# use the previous value to fill the missing value
train_data.fillna(method='ffill', inplace=True)

# preparing training data
imga = []
for i in range(len(train_data)):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    #img = img / 255
    imga.append(img)

image_list = np.array(imga, dtype='float') / 255
X_train = image_list.reshape(-1, 96, 96, 1)

# preparing training label
training = train_data.drop('Image', axis=1)
y_train = []
for i in range(len(train_data)):
    y = training.iloc[i, 1:]
    y = (y - 48) / 48
    y_train.append(y)
y_train = np.array(y_train, dtype='float')

# preparing test data
timga = []
for i in range(len(test_data)):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    timga.append(timg)
timage_list = np.array(timga, dtype='float') /255
X_test = timage_list.reshape(-1, 96, 96, 1)

def show_result(image, points):
    plt.imshow(image, cmap='gray')
    for i in range(15):
        plt.plot(points[2*i], points[2*i + 1], 'ro')
    plt.show()

show_result(X_train[0].reshape(96,96), y_train[0])

def CNN(dim=9216):
    """keras NN model library
    
    Arguments:
        dim {int/tuple} -- input dimension (default: {9216})

    Return: compiled Keras model
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=dim))  # CONV1
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))  # FC1

    model.add(Conv2D(64, (2, 2)))           # Conv2
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))  # FC2

    model.add(Conv2D(128, (2, 2)))          # Conv3
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))  # FC3

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(30))                    # 30 coords

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    return model

modelCNN = CNN(dim=(96, 96, 1))
modelCNN.summary()
histCNN = modelCNN.fit(X_train, y_train, epochs=100, validation_split=0.2)

def normlabel(y, reverse=False):
    """normalize / de-normalize label 
    
    Arguments:
        y {np.array} -- raw or normalized label
    
    Keyword Arguments:
        reverse {bool} -- normalize / de-normalize label (default: {False})
    """
    if reverse:
        return  y*48 + 48
    else:
        return (y - 48) / 48

def histplot(hist, save=None, show=True):
    """plotting loss function
    
    Arguments:
        hist {keras.callbacks.History} -- hist = model.fit
    
    Keyword Arguments:
        save {str} -- filename if to save (default: {None})
        show {bool} -- show plot (default: {True})
    """
    plt.figure()
    plt.plot(hist.history['loss'], linewidth=2, label='train')
    plt.plot(hist.history['val_loss'], linewidth=2,label='valid set')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.grid()               # show metrics(0, 20, 40...)
    plt.legend(loc='best')
    figure = plt.gcf()

    if save != None:
        figure.savefig(save, dpi=300)
        print(save, 'saved.')

    if show:
        plt.show()
        
def predplot(xtest, ypred, save=None, show=True):
    """plotting pred outcome
    
    Arguments:
        xtest {np.array} -- test img
        ypred {np.array} -- pred label
    
    Keyword Arguments:
        save {str} -- filename if to save (default: {None})
        show {bool} -- show plot (default: {True})
    """
    total = xtest.shape[0]
    start = random.randint(0, total-16)
    print('image',start,'to',start+16)

    fig = plt.figure(figsize=(12, 12))
    # fig.subplots_adjust(left=0, right=0.5, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

    for i in range(start, start+16):
        x, y = xtest[i], ypred[i]
        img  = x.reshape(96, 96)
        axis = fig.add_subplot(4, 4, i-start+1, xticks=[], yticks=[])
        axis.imshow(img, cmap='gray')   # show image
        axis.scatter(normlabel(y[0::2], reverse=True),  # show lanmark
                     normlabel(y[1::2], reverse=True), marker='x', s=10)
    figure = plt.gcf()
    
    if save != None:    # deprecated
        figure.savefig(save, dpi=300)
        print(save, 'saved.')

    if show:
        plt.show()

histplot(histCNN, save='CNN-test.png', show=True)
y_pred = modelCNN.predict(X_test)
predplot(X_test, y_pred, save='pred-CNN.png', show=True)

y_pred1=normlabel(y_pred, reverse=True)

def export(pred_points, filename):
    """
    :param pred_points: result from your model use test.csv
    :return:
    """
    submission_data = pd.DataFrame(pred_points)
    submission_data.to_csv(filename, index=False)
    
export(y_pred1, '2018211348.csv')

def savemodel(model, name, toprint=True):
    """save model to file
    
    Arguments:
        model {keras.models.Sequential} -- keras model
    
    Keyword Arguments:
        name {str} -- file name to save
    """
    json_string = model.to_json()
    open(name+'_architecture.json', 'w').write(json_string)  # structure
    model.save_weights(name+'_weights.h5')                   # weights
    
    if toprint:
        print('Structure:', name+'_architecture.json')
        print('Weights:', name+'_weights.h5')
        
savemodel(modelCNN, name='model/CNN')

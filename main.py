# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import sys, getopt

def main(argv) :
    inputfile = ''
    outputfile = ''
    action = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:a:",["ifile=","ofile=","action="])
    except getopt.GetoptError:
        print('main.py -i <inputfile> -o <outputfile> -a <action>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputfile> -o <outputfile> -a <action>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-a", "--action"):
            action = arg

    print('Input file is: ', inputfile)
    print('Output file is: ', outputfile)
    print('Action is: ', action)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    if action == "new" :
        myModel = getModel()
        myModel.fit(train_images, train_labels, epochs=5)
        myModel.save(outputfile)

    elif action == "optimize" :
        myModel = loadModel(inputfile)
        myModel.fit(train_images, train_labels, epochs=5)
        myModel.save(outputfile)

    elif action == "use" :
        myModel = loadModel(inputfile)

def getModel() :

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

def loadModel(filename) :
    return keras.models.load_model(filename)

# train_images.shape

# train_labels

# test_images.shape

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# plt.figure(figsize=(10,10))

# for i in range(100):
#     plt.subplot(25,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])

# plt.show()

# model.fit(train_images, train_labels, epochs=10)

# model.save('my_model.h5')

# new_model = keras.models.load_model('my_model.h5')

# new_model.compile(optimizer=tf.train.AdamOptimizer(), 
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'])

# new_model.fit(train_images, train_labels, epochs=5)

# loss, acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == "__main__":
   main(sys.argv[1:])
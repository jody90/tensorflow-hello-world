# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import sys, getopt

from scipy import misc

import skimage

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images_flip = np.flip(train_images,2)
test_images_flip = np.flip(test_images,2)

train_images = np.append(train_images, train_images_flip, axis=0)
test_images = np.append(test_images, test_images_flip, axis=0)

train_labels = np.append(train_labels, train_labels, axis=0)
test_labels = np.append(test_labels, test_labels, axis=0)

train_images = train_images / 255.0

test_images = test_images / 255.0

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

    if action == "new" :
        model = getModel()
        model.fit(train_images, train_labels, epochs=5)
        model.save(outputfile)

    elif action == "optimize" :
        model = loadModel(inputfile)
        model.fit(train_images, train_labels, epochs=5)
        model.save(outputfile)
        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print('Test accuracy:', test_acc)

    elif action == "use" :
        model = loadModel(inputfile)

    elif action == "predict" :
        model = loadModel(inputfile)

        predictions = model.predict(test_images)

        # Plot the first X test images, their predicted label, and the true label
        # Color correct predictions in blue, incorrect predictions in red
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plot_image(i, predictions, test_labels, test_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plot_value_array(i, predictions, test_labels)
        
        plt.show()
    
    elif action == "predict-single" :
        model = loadModel(inputfile)

        img = test_images[4]

        img = (np.expand_dims(img,0))

        predictions_single = model.predict(img)

        plot_value_array(4, predictions_single, test_labels)
        _ = plt.xticks(range(10), class_names, rotation=45)
        plt.show()

    elif action == "predict-custom" :

        model = loadModel(inputfile)

        # customImage = misc.imread('sneaker_875.jpg', True)

        # customImage = skimage.transform.resize(customImage, output.shape)

        customImage = misc.imread('boot2.jpg', True)
        # customImage = misc.imread('shoe.jpg', True)
        customImage = customImage / 255.0

        plt.figure()
        plt.imshow(customImage)
        plt.colorbar()
        plt.grid(False)
        plt.show()

        img = (np.expand_dims(customImage,0))

        predictions_single = model.predict(img)

        plot_value_array(0, predictions_single, test_labels)
        _ = plt.xticks(range(10), class_names, rotation=45)
        plt.show()

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

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array = predictions_array[0]
  true_label = true_label[i]

  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

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
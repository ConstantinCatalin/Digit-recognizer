import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import model_from_json


class NeuralNet(object):

    # load the training and test data of MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # pre-process the data to fit the format required to train the model
    training_images = X_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    training_targets = to_categorical(y_train)

    test_images = X_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
    test_targets = to_categorical(y_test)

    # load json and create model
    json_file = open('./model_info/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./model_info/model_weights.h5")
    print("Loaded model from disk")
    print("--- Please wait ---")

    def info(self):
        print("Evaluate current model")
        print("--- Please wait ---")
        # evaluate loaded model on test data ########
        NeuralNet.loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = NeuralNet.loaded_model.evaluate(NeuralNet.training_images, NeuralNet.training_targets, verbose=0)
        print("Current %s: %.2f%%" % (NeuralNet.loaded_model.metrics_names[0], score[0] * 100))
        print("Current %s: %.2f%%" % (NeuralNet.loaded_model.metrics_names[1], score[1] * 100))

    def training(self):
        # create the model
        # convolution 16 5x5

        self.model = Sequential()
        self.model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Conv2D(32, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        # use softmax activation function to create the probability matrix for each digit
        self.model.add(Dense(10, activation='softmax'))

        # compile the model
        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

        # fit the model
        self.model.fit(NeuralNet.training_images,
                       NeuralNet.training_targets,
                       epochs=50,
                       validation_split=0.4,
                       callbacks=[EarlyStopping(patience=6)])

        # evaluate the model
        scores = self.model.evaluate(NeuralNet.training_images, NeuralNet.training_targets, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("./model_info/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights
        self.model.save_weights("./model_info/model_weights.h5")
        print("Saved model to disk")

    def predict(self, image):
        input = cv2.resize(image, (28, 28)).reshape((28, 28, 1)).astype('float32') / 255
        save_img = cv2.resize(image, (28, 28)).reshape((28, 28, 1)).astype('float32')
        cv2.imwrite("./model_info/input.png", save_img)
        return self.loaded_model.predict_classes(np.array([input]))

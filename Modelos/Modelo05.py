from TreinamentoPerfacemotion import TreinamentoPerfacemotion
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

class Modelo05(TreinamentoPerfacemotion):

    def ConstruirModelo(self, modelo, num_features, num_classes, width, height):
        print("MODELO 5")

        modelo.add(Conv2D(20, (3, 3), padding='same', activation='relu', input_shape=(width, height, 1)))
        modelo.add(Conv2D(30, (3, 3), padding='same', activation='relu'))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(BatchNormalization())
        modelo.add(Dropout(0.2))

        modelo.add(Conv2D(40, (3, 3), padding='same', activation='relu'))
        modelo.add(Conv2D(50, (3, 3), padding='same', activation='relu'))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(BatchNormalization())
        modelo.add(Dropout(0.2))

        modelo.add(Conv2D(60, (3, 3), padding='same', activation='relu'))
        modelo.add(Conv2D(70, (3, 3), padding='same', activation='relu'))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(Dropout(0.2))

        modelo.add(Conv2D(80, (3, 3), padding='same', activation='relu'))
        modelo.add(Conv2D(90, (3, 3), padding='same', activation='relu'))

        modelo.add(Flatten())

        modelo.add(Dense(1000, activation='relu'))
        modelo.add(Dense(512, activation='relu'))

        modelo.add(Dense(num_classes, activation='softmax'))
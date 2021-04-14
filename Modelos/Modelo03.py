from TreinamentoPerfacemotion import TreinamentoPerfacemotion
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

class Modelo03(TreinamentoPerfacemotion):

    def ConstruirModelo(self, modelo, num_features, num_classes, width, height):
        print("MODELO 3")

        modelo.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1),
                         data_format='channels_last', kernel_regularizer=l2(0.01)))
        modelo.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Flatten())

        modelo.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
        modelo.add(Dropout(0.4))
        modelo.add(Dense(2 * 2 * num_features, activation='relu'))
        modelo.add(Dropout(0.4))
        modelo.add(Dense(2 * num_features, activation='relu'))
        modelo.add(Dropout(0.5))

        modelo.add(Dense(num_classes, activation='softmax'))
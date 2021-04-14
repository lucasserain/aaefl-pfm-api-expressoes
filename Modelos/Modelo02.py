from TreinamentoPerfacemotion import TreinamentoPerfacemotion
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

class Modelo02(TreinamentoPerfacemotion):

    def ConstruirModelo(self, modelo, num_features, num_classes, width, height):
        print("MODELO 2")

        modelo.add(
            Conv2D(num_features, (3, 3), padding='same', kernel_initializer="he_normal",
                   input_shape=(width, height, 1)))

        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(
            Conv2D(num_features, (3, 3), padding="same", kernel_initializer="he_normal",
                   input_shape=(width, height, 1)))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(Dropout(0.2))

        modelo.add(
            Conv2D(2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(
            Conv2D(2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(Dropout(0.2))

        modelo.add(
            Conv2D(2 * 2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(
            Conv2D(2 * 2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(Dropout(0.2))

        modelo.add(
            Conv2D(2 * 2 * 2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(
            Conv2D(2 * 2 * 2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(Dropout(0.2))

        modelo.add(Flatten())
        modelo.add(Dense(2 * num_features, kernel_initializer="he_normal"))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(Dropout(0.5))

        modelo.add(Dense(2 * num_features, kernel_initializer="he_normal"))
        modelo.add(Activation('elu'))
        modelo.add(BatchNormalization())
        modelo.add(Dropout(0.5))

        modelo.add(Dense(num_classes, kernel_initializer="he_normal"))
        modelo.add(Activation("softmax"))

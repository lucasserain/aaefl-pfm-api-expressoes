from TreinamentoPerfacemotion import TreinamentoPerfacemotion
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2

class Modelo04(TreinamentoPerfacemotion):

    def ConstruirModelo(self, modelo, num_features, num_classes, width, height):
        print("MODELO 4")
        learning_rate = 0.001

        modelo.add(Conv2D(64, (3, 3), activation='relu', input_shape=(width, height, 1), kernel_regularizer=l2(0.01)))
        modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D(pool_size=(2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Flatten())
        modelo.add(Dense(512, activation='relu'))
        modelo.add(Dropout(0.5))
        modelo.add(Dense(256, activation='relu'))
        modelo.add(Dropout(0.5))
        modelo.add(Dense(128, activation='relu'))
        modelo.add(Dropout(0.5))
        modelo.add(Dense(64, activation='relu'))
        modelo.add(Dropout(0.5))
        modelo.add(Dense(7, activation='softmax'))
        adam = optimizers.Adam(lr=learning_rate)
        modelo.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
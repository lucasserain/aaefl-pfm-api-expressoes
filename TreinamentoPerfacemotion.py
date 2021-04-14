import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

from abc import ABC, abstractmethod


class TreinamentoPerfacemotion(ABC):

    # @abstractmethod
    # def __init__(self, caminhoArquivoTreinamento, caminhoSaidaModelo, caminhoSaidaModeloJson, caminhoSaidaFacesTeste,
    #              caminhoSaidaEmocoesTeste, expressoes, larguraImagem, alturaImagem):
    #     pass

    def InicializarModelo(self, caminhoArquivoTreinamento, caminhoSaidaModelo, caminhoSaidaModeloJson,
                          caminhoSaidaFacesTeste,
                          caminhoSaidaEmocoesTeste, expressoes, larguraImagem, alturaImagem):
        # Obtem o caminho para o arquivo de dados de treinamento
        self.__ARQUIVO_BASE_TREINAMENTO = caminhoArquivoTreinamento
        self.__ARQUIVO_SAIDA_MODELO = caminhoSaidaModelo
        self.__ARQUIVO_SAIDA_MODELO_JSON = caminhoSaidaModeloJson
        self.__ARQUIVO_SAIDA_FACES_TESTES = caminhoSaidaFacesTeste
        self.__ARQUIVO_SAIDA_EMOCOES_TESTES = caminhoSaidaEmocoesTeste

        # Le o arquivo de treinamento
        self.data_treinamento = pd.read_csv(self.__ARQUIVO_BASE_TREINAMENTO)
        self.expressoes = expressoes
        # Define o tamanho das imagens
        self.__LARGURA = larguraImagem
        self.__ALTURA = alturaImagem

    def __ConverterListaPixelsToMatriz(self, faces, largura, altura):
        # Obtem os pixels no arquivo de treinamento csv e tranforma em uma lista
        pixels = self.data_treinamento["pixels"].tolist()
        # Percorre a lista de pixels - Cada linha representa uma imagem
        for sequencia_pixels in pixels:
            # Quebra a lista de pixels entre os espacos
            face = [int(pixel) for pixel in sequencia_pixels.split(' ')]
            # Tranforma cada item da lista em uma matriz de dimensoes de acordo com a largura e altura definida
            face = np.asarray(face).reshape(largura, altura)
            # Inclui a imagem na matriz
            faces.append(face)

    def __NormalizarPixels(self, x):
        x = x.astype('float32')
        x = x / 255.0
        return x

    def __MontarBasesDados(self, faces, emocoes):
        # Separa 10% das imagens para testes e os outros 90% para treinamento
        self.faces_train, self.faces_test, self.emocoes_train, self.emocoes_test = train_test_split(faces, emocoes,
                                                                                                    test_size=0.1,
                                                                                                    random_state=15)

        # Separa 10% das imagens para validacao e os outros 90% para treinamento
        self.faces_train, self.faces_val, self.emocoes_train, self.emocoes_val = train_test_split(self.faces_train,
                                                                                                  self.emocoes_train,
                                                                                                  test_size=0.1,
                                                                                                  random_state=34)

        np.save(self.__ARQUIVO_SAIDA_FACES_TESTES, self.faces_test)
        np.save(self.__ARQUIVO_SAIDA_EMOCOES_TESTES, self.emocoes_test)

    @abstractmethod
    def ConstruirModelo(self, modelo, num_features, num_classes, width, height):
        pass

    def __ConstruirModelo01(self, num_features, num_classes, width, height):
        self.__modelo = Sequential()

        '''
        num_features -> numero de filtros
        (3, 3) -> tamanho da matriz para extração de caracteristicas
        padding -> Same - Não ignora os dados in
        kernel_initializer -> 
        input_shape -> Tamanho da entrada (Imagem) - 1: Quantidade de canais 
        Activation('elu') -> Remove numeros negativos para realçar partes principais das imagens
        
        '''
        self.__modelo.add(Conv2D(num_features, (3, 3), padding='same', kernel_initializer="he_normal",
                                 input_shape=(width, height, 1)))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())

        self.__modelo.add(Conv2D(num_features, (3, 3), padding="same", kernel_initializer="he_normal",
                                 input_shape=(width, height, 1)))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(MaxPooling2D(pool_size=(2, 2)))
        self.__modelo.add(Dropout(0.2))

        self.__modelo.add(Conv2D(2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(Conv2D(2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(MaxPooling2D(pool_size=(2, 2)))
        self.__modelo.add(Dropout(0.2))

        self.__modelo.add(Conv2D(2 * 2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(Conv2D(2 * 2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(MaxPooling2D(pool_size=(2, 2)))
        self.__modelo.add(Dropout(0.2))

        self.__modelo.add(Conv2D(2 * 2 * 2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(Conv2D(2 * 2 * 2 * num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(MaxPooling2D(pool_size=(2, 2)))
        self.__modelo.add(Dropout(0.2))

        self.__modelo.add(Flatten())
        self.__modelo.add(Dense(2 * num_features, kernel_initializer="he_normal"))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(Dropout(0.5))

        self.__modelo.add(Dense(2 * num_features, kernel_initializer="he_normal"))
        self.__modelo.add(Activation('elu'))
        self.__modelo.add(BatchNormalization())
        self.__modelo.add(Dropout(0.5))

        self.__modelo.add(Dense(num_classes, kernel_initializer="he_normal"))
        self.__modelo.add(Activation("softmax"))

        # print(self.__modelo.summary())

    def __CompilarModelo01(self):
        self.__modelo.compile(loss=categorical_crossentropy,
                              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                              metrics=["accuracy"])

    def __SalvarArquitetura(self):
        model_json = self.__modelo.to_json()
        with open(self.__ARQUIVO_SAIDA_MODELO_JSON, "w") as json_file:
            json_file.write(model_json)

    def __TreinarModelo(self, batch_size, epochs):
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(self.__ARQUIVO_SAIDA_MODELO, monitor='val_loss', verbose=1, save_best_only=True)

        print("Treinando modelo...")
        history = self.__modelo.fit(np.array(self.faces_train), np.array(self.emocoes_train),
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(np.array(self.faces_val), np.array(self.emocoes_val)),
                                    shuffle=True,
                                    callbacks=[lr_reducer, early_stopper, checkpointer])
        print("Treinamento finalizado.")

    def CarregarModelo(self):
        self.__modelo = load_model(self.__ARQUIVO_SAIDA_MODELO)

    def TestarModelo(self, imagem):
        prediction = self.__modelo.predict(imagem)[0]
        return self.expressoes[int(np.argmax(prediction))]

    def TreinarRedePerfacemotion(self, num_features, num_classes, batch_size, epochs):
        faces = []

        # Converte Lista com os pixels em matriz de dimensoes de acordo com largura e altura
        self.__ConverterListaPixelsToMatriz(faces, self.__LARGURA, self.__ALTURA)

        # Cria uma nova dimensao na matriz
        faces = np.expand_dims(faces, -1)

        # Normaliza os pois a rede só aceita numeros de 0 a 1
        faces = self.__NormalizarPixels(faces)

        # Converte as emocoes em uma matriz de dummies
        emocoes = pd.get_dummies(self.data_treinamento['emotion']).values

        # Monta as bases de dados de treinamento, teste, e validacao
        self.__MontarBasesDados(faces, emocoes)
        self.__modelo = Sequential()
        self.ConstruirModelo(self.__modelo, num_features, num_classes, self.__LARGURA, self.__ALTURA)
        self.__CompilarModelo01()
        self.__SalvarArquitetura()
        self.__TreinarModelo(batch_size, epochs)

        # print("Número de imagens no conjunto de treinamento:", len(faces_train))
        # print("Número de imagens no conjunto de testes:", len(faces_test))
        # print("Número de imagens no conjunto de validação:", len(faces_val))
        #
        # print("Treinamento X :", x_train)
        # print("Treinamento Y :", y_train)

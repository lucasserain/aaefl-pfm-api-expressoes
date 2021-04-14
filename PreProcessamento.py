import cv2
import path
import numpy as np
import math


class PreProcessamento:
    '''
    Construtor
    Define o tamanho da imagem que será trabalhada

    '''

    def __init__(self, haarCascade, larguraImagem, alturaImagem):
        self.__LARGURA = larguraImagem
        self.__ALTURA = alturaImagem
        self.__HaarCasdade = haarCascade

    def RedimensionarImagem(self, imagem):
        return cv2.resize(imagem, (self.__ALTURA, self.__LARGURA))

    def ConverterImagemEscalaCinza(self, imagem):
        return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    def RecortarImagem(self, imagem):
        # x, y, l, a = self.__HaarCasdade.CapturarRosto(imagem)
        faces = self.__HaarCasdade.CapturarRosto(imagem)

        if faces is not None and len(faces) == 1:
            x = faces[0, 0]
            y = faces[0, 1]
            l = faces[0, 2]
            a = faces[0, 3]
            imagemRecortada = imagem[y:y + a, x:x + l]
            return imagemRecortada
        return imagem

    '''
    Preparar imagem para entrar no modelo
    '''

    def NormalizarImagem(self, imagem):
        imagem = imagem.astype("float") / 255.0
        imagemExpandida = np.expand_dims(np.expand_dims(imagem, -1), 0)
        cv2.normalize(imagemExpandida, imagem, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        return imagem

    def EqualizarHistogramaImagem(self, imagem):
        cv2.equalizeHist(imagem, imagem)
        return imagem

    def RotacionarImagem(self, imagem, margem=10):
        if len(self.__HaarCasdade.CapturarOlhos(imagem)) < 2:
            return imagem
        olho1, olho2 = self.__HaarCasdade.CapturarOlhos(imagem)[:2]

        if abs(olho1[1] - olho2[1]) > margem:
            x1, y1 = olho1[:2]
            x2, y2 = olho2[:2]
            print("x1: " + str(x1))
            print("x2 : " + str(x2))
            print("y1 " + str(y1))
            print("y2 : " + str(y2))
            if x1 > x2:  # Virado pra Esquerda
                x = x1 - x2
                y = y1 - y2
                theta = math.atan2(y, x)
                graus = math.degrees(theta)
                M = cv2.getRotationMatrix2D((x2, y2), graus, 1.0)
            else:  # Virado pra Direita
                x = x2 - x1
                y = y2 - y1
                theta = math.atan2(y, x)
                graus = math.degrees(theta)
                M = cv2.getRotationMatrix2D((x1, y1), graus, 1.0)
            theta = math.atan2(y, x)
            graus = math.degrees(theta)
            altura, largura = imagem.shape
            print(graus)

            rotated = cv2.warpAffine(imagem, M, (largura, altura))
            return rotated

        return imagem

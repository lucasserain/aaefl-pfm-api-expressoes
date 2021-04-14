import numpy as np
import cv2
import math
from glob import glob
import os

class PropostaExperimental:

    def __init__(self, caminhoImagens, larguraImagem, alturaImagem):
        self.__LARGURA = larguraImagem
        self.__ALTURA = alturaImagem
        self.CaminhoImagens = caminhoImagens

    def RealizarPreProcessamento(self, preProcessamento):
        nomeImagens = glob(os.path.join(self.CaminhoImagens, "*jpg"))

        self.ArrayImagens = []

        for nome in nomeImagens:
            imagem = cv2.imread(nome)
            imagem = preProcessamento.ConverterImagemEscalaCinza(imagem)
            imagem = preProcessamento.EqualizarHistogramaImagem(imagem)
            imagem = preProcessamento.RotacionarImagem(imagem)
            imagem = preProcessamento.RecortarImagem(imagem)
            imagem = preProcessamento.RedimensionarImagem(imagem)
            self.ArrayImagens.append(imagem)

    def CalcularMediaEDesvioPadrao(self):
        media = np.zeros((self.__ALTURA, self.__LARGURA))
        desvioPadrao = np.zeros((self.__ALTURA, self.__LARGURA))

        for w in range(self.__ALTURA):
            for q in range(self.__LARGURA):
                soma = 0
                somaDP = 0
                for imagem in self.ArrayImagens:
                    soma += imagem[w, q]
                media[w, q] = soma / len(self.ArrayImagens)

                for imagem in self.ArrayImagens:
                    somaDP += (imagem[w, q] - media[w, q]) ** 2
                desvioPadrao[w, q] = math.sqrt(somaDP / len(self.ArrayImagens))

        self.__GravarImagem('Media.jpg', media)
        self.__GravarImagem('DesvioPadrao.jpg', desvioPadrao)

    def __ResetarArray(self, array):
        for i, e in enumerate(array):
            if isinstance(e, list):
                self.ResetarMatriz(e)
            else:
                array[i] = 0

    def __GravarImagem(self, nome, array):
        cv2.imwrite(os.path.join(self.CaminhoImagens, nome), array)

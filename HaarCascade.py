import cv2
import path
import numpy as np


class HaarCascade:
    # self.ARQUIVO_HAARCASCADE_FRONTALFACE = path.Path("resources\\haarcascade_frontalface_default.xml")
    def __init__(self, ARQUIVO_HAARCASCADE_FACE, ARQUIVO_HAARCASCADE_OLHOS, xMinImagem, yMinImagem):
        self.haarCascadeFace = cv2.CascadeClassifier(ARQUIVO_HAARCASCADE_FACE)
        self.haarCascadeOlhos = cv2.CascadeClassifier(ARQUIVO_HAARCASCADE_OLHOS)
        self.xMinImagem = xMinImagem
        self.yMinImagem = yMinImagem

    def CapturarRosto(self, imagem):
        facesDetectadas = self.haarCascadeFace.detectMultiScale(imagem, scaleFactor=1.5,
                                                                minSize=(self.xMinImagem, self.yMinImagem))
        return facesDetectadas

    def CapturarOlhos(self, imagem):
        olhos = self.haarCascadeOlhos.detectMultiScale(imagem)
        return olhos



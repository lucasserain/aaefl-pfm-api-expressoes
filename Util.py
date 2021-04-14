import cv2
import path
import numpy as np


class Util:

    def LerImagem(self, caminhoImagem):
        return cv2.imread(caminhoImagem)

    def ExibirImagem(self, imagem):
        cv2.imshow("Imagem", imagem)
        cv2.waitKey(5000)

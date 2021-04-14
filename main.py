import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import path
import matplotlib.pyplot as plt
import zipfile
# from . import TreinamentoPerfacemotion
# import TreinamentoPerfacemotion
import path

from Modelos.Modelo02 import Modelo02
from Modelos.Modelo03 import Modelo03
from Modelos.Modelo04 import Modelo04
from Modelos.Modelo05 import Modelo05
from TreinamentoPerfacemotion import TreinamentoPerfacemotion
from PreProcessamento import PreProcessamento
from Util import Util
from CapturaCamera import CapturaCamera
from HaarCascade import HaarCascade
from PropostaExperimental import PropostaExperimental
import config

import math


def __exit__():
    os._exit(0)


def __main__():

    expressoes = config.GetExpressoesBase("Fer2013")
    ALTURA, LARGURA = config.GetAlturaLarguraImagensBase("Fer2013")
    caminhoArquivoTreinamento = config.GetCaminhoArquivoTreinamento("Fer2013")
    caminhoImagensPreprocessamentoTeste = config.GetCaminhoImagensPreProcessamento()
    ARQUIVO_HAARCASCADE_FRONTALFACE = config.GetCaminhoHaarCascadeFrontalFace()
    ARQUIVO_HAARCASCADE_OLHOS = config.GetCaminhoHaarCascadeOlhos()

    # # Modelo 1
    # num_features = 32
    # num_classes = 7
    # width, height = 48, 48
    # batch_size = 16
    # epochs = 100
    # caminhoSaidaModelo = config.GetCaminhoSaidaModelo("Modelo01")
    # caminhoSaidaModeloJson = config.GetCaminhoSaidaModeloJSON("Modelo01")
    # caminhoSaidaFacesTeste = config.GetCaminhoSaidaFaceTeste("Modelo01")
    # caminhoSaidaEmocoesTeste = config.GetCaminhoSaidaEmocoesTest("Modelo01")

    # treinamento01 = Modelo01()
    # treinamento01.InicializarModelo(caminhoArquivoTreinamento, caminhoSaidaModelo, caminhoSaidaModeloJson,
    #                                 caminhoSaidaFacesTeste, caminhoSaidaEmocoesTeste, expressoes, ALTURA, LARGURA)
    # treinamento01.TreinarRedePerfacemotion(64, 7, 64, 100)

    # Modelo 2
    # caminhoSaidaModelo = config.GetCaminhoSaidaModelo("Modelo02")
    # caminhoSaidaModeloJson = config.GetCaminhoSaidaModeloJSON("Modelo02")
    # caminhoSaidaFacesTeste = config.GetCaminhoSaidaFaceTeste("Modelo02")
    # caminhoSaidaEmocoesTeste = config.GetCaminhoSaidaEmocoesTest("Modelo02")
    #
    # treinamento02 = Modelo02()
    # treinamento02.InicializarModelo(caminhoArquivoTreinamento, caminhoSaidaModelo, caminhoSaidaModeloJson,
    #                                 caminhoSaidaFacesTeste, caminhoSaidaEmocoesTeste, expressoes, ALTURA, LARGURA)
    # treinamento02.TreinarRedePerfacemotion(32, 7, 16, 100)

    # Modelo 3
    caminhoSaidaModelo = config.GetCaminhoSaidaModelo("Modelo03")
    caminhoSaidaModeloJson = config.GetCaminhoSaidaModeloJSON("Modelo03")
    caminhoSaidaFacesTeste = config.GetCaminhoSaidaFaceTeste("Modelo03")
    caminhoSaidaEmocoesTeste = config.GetCaminhoSaidaEmocoesTest("Modelo03")

    treinamento03 = Modelo03()
    treinamento03.InicializarModelo(caminhoArquivoTreinamento, caminhoSaidaModelo, caminhoSaidaModeloJson,
                                    caminhoSaidaFacesTeste, caminhoSaidaEmocoesTeste, expressoes, ALTURA, LARGURA)
    treinamento03.TreinarRedePerfacemotion(64, 7, 64, 100)

    # Modelo 4
    caminhoSaidaModelo = config.GetCaminhoSaidaModelo("Modelo04")
    caminhoSaidaModeloJson = config.GetCaminhoSaidaModeloJSON("Modelo04")
    caminhoSaidaFacesTeste = config.GetCaminhoSaidaFaceTeste("Modelo04")
    caminhoSaidaEmocoesTeste = config.GetCaminhoSaidaEmocoesTest("Modelo04")

    treinamento04 = Modelo04()
    treinamento04.InicializarModelo(caminhoArquivoTreinamento, caminhoSaidaModelo, caminhoSaidaModeloJson,
                                    caminhoSaidaFacesTeste, caminhoSaidaEmocoesTeste, expressoes, ALTURA, LARGURA)
    treinamento04.TreinarRedePerfacemotion(32, 7, 64, 100)

    caminhoSaidaModelo = config.GetCaminhoSaidaModelo("Modelo05")
    caminhoSaidaModeloJson = config.GetCaminhoSaidaModeloJSON("Modelo05")
    caminhoSaidaFacesTeste = config.GetCaminhoSaidaFaceTeste("Modelo05")
    caminhoSaidaEmocoesTeste = config.GetCaminhoSaidaEmocoesTest("Modelo05")

    # Modelo 5
    treinamento05 = Modelo05()
    treinamento05.InicializarModelo(caminhoArquivoTreinamento, caminhoSaidaModelo, caminhoSaidaModeloJson,
                                    caminhoSaidaFacesTeste, caminhoSaidaEmocoesTeste, expressoes, ALTURA, LARGURA)
    treinamento05.TreinarRedePerfacemotion(32, 7, 256, 100)

    return

    # Proposta Experimental
    haarCascade = HaarCascade(ARQUIVO_HAARCASCADE_FRONTALFACE, ARQUIVO_HAARCASCADE_OLHOS, 100, 100)
    preProc = PreProcessamento(haarCascade, LARGURA, ALTURA)

    camera = CapturaCamera(preProc, haarCascade)
    propExperimental = PropostaExperimental(caminhoImagensPreprocessamentoTeste, LARGURA, ALTURA)
    propExperimental.RealizarPreProcessamento(preProc)
    propExperimental.CalcularMediaEDesvioPadrao()

    #

    # treinamento.CarregarModelo()
    # camera.TestarComWebCam(treinamento)

    # treinamento.TestarModelo()

    # treinamento.TreinarRedePerfacemotion()

    __exit__()


__main__()

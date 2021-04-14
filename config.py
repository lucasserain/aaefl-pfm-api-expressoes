import path



# Base
def GetCaminhoArquivoTreinamento(nomeBase):
    if nomeBase == "Fer2013":
        return path.Path("bases-imagens\\Fer2013\\fer2013.csv")
    else:
        raise Exception("Base de dados deve ser parametrizada no config.")


def GetAlturaLarguraImagensBase(nomeBase):
    if nomeBase == "Fer2013":
        return 48, 48
    else:
        raise Exception("Base de dados deve ser parametrizada no config.")


def GetExpressoesBase(nomeBase):
    if nomeBase == "Fer2013":
        return ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]
    else:
        raise Exception("Base de dados deve ser parametrizada no config.")


# Caminhos Modelo
def GetCaminhoSaidaModelo(nomeModelo):
    return path.Path("resources\\{}\\modelo_perfacemotion.h5".format(nomeModelo))


def GetCaminhoSaidaModeloJSON(nomeModelo):
    return path.Path("resources\\{}\\modelo_perfacemotion.json".format(nomeModelo))


def GetCaminhoSaidaFaceTeste(nomeModelo):
    return path.Path("resources\\{}\\mod_faces_test.h5".format(nomeModelo))


def GetCaminhoSaidaEmocoesTest(nomeModeloJSON):
    return path.Path("resources\\{}\\mod_emocoes_test.json".format(nomeModeloJSON))

# Pre-processamento
def GetCaminhoImagensPreProcessamento():
    return path.Path("Imagens_Testes\\PropostaExperimental\\PreProcessamento")

# HaarCascade
def GetCaminhoHaarCascadeFrontalFace():
    return path.Path("Arquivos\\HaarCascade\\haarcascade_frontalface_default.xml")

def GetCaminhoHaarCascadeOlhos():
    return path.Path("Arquivos\\HaarCascade\\haarcascade_eye.xml")


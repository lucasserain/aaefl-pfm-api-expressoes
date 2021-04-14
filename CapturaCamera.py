import cv2
import path
import numpy as np
from PreProcessamento import PreProcessamento

# Obtem o camihho para o arquivo do haarcascade
ARQUIVO_HAARCASCADE_FRONTALFACE = path.Path("resources\\haarcascade_frontalface_default.xml")


class CapturaCamera:

    def __init__(self, preProc, classificadorFaces):
        self.PreProc = preProc
        # Cria um classificador com base no Haarcascade
        self.classificadorFaces = classificadorFaces

    def CapturarFrames(self):

        # cria a camera a partir do opencv
        camera = cv2.VideoCapture(0)
        while (True):
            # Lê da camera
            conectado, imagem = camera.read()
            # Cria uma imagem cinza
            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            # Obtem as faces detectadas pelo haarcascade
            # facesDetectadas = self.classificadorFaces.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100))
            facesDetectadas = self.classificadorFaces.CapturarRosto(imagemCinza)
            # Desenha os retangulos de acordo com as faces obtidas pelo haarcascade
            for (x, y, l, a) in facesDetectadas:
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
            # exibe a imagem com os retangulos de acordo com as faces localizadas
            cv2.imshow("Face", imagem)
            # Aguarda 100ms (Captura 10 fps)
            cv2.waitKey(100)

            # Fecha janela
            if cv2.getWindowProperty("Face", cv2.WND_PROP_VISIBLE) < 1:
                break

        # Destroi as janelas do open cv
        cv2.destroyAllWindows()
        # Encerra a camera
        camera.release()

    def TestarComWebCam(self, treinamento):

        # cria a camera a partir do opencv
        camera = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 30, (640, 480))
        while (True):
            # Lê da camera
            conectado, imagem = camera.read()
            # Cria uma imagem cinza
            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            # Obtem as faces detectadas pelo haarcascade
            # facesDetectadas = self.classificadorFaces.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100))
            facesDetectadas = self.classificadorFaces.CapturarRosto(imagemCinza)
            # Desenha os retangulos de acordo com as faces obtidas pelo haarcascade
            for (x, y, l, a) in facesDetectadas:
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
                roi_gray = imagemCinza[y:y + a, x:x + l]
                roi_gray = roi_gray.astype("float") / 255.0
                imagemRedimensionada = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                cv2.normalize(imagemRedimensionada, imagemRedimensionada, alpha=0, beta=1,
                              norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
                prediction = treinamento.TestarModelo(imagemRedimensionada)
                print(prediction)
                cv2.putText(imagem, prediction, (x, y - 10),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            # exibe a imagem com os retangulos de acordo com as faces localizadas
            cv2.imshow("Face", imagem)
            # Aguarda 100ms (Captura 10 fps)
            cv2.waitKey(100)
            out.write(imagem)
            # Fecha janela
            if cv2.getWindowProperty("Face", cv2.WND_PROP_VISIBLE) < 1:
                break

        # Destroi as janelas do open cv
        cv2.destroyAllWindows()
        # Encerra a camera
        camera.release()
        out.release()
        # treinamento.TestarModelo()

    def CapturarFramesVideo(self, treinamento):

        # cria a camera a partir do opencv
        camera = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 30,(640,480))
        while (camera.isOpened()):
            # Lê da camera
            conectado, imagem = camera.read()

            if conectado == True:
                # Cria uma imagem cinza
                imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
                # Obtem as faces detectadas pelo haarcascade
                imagemCinza = self.PreProc.EqualizarHistogramaImagem(imagemCinza)

                imagemCinza = self.PreProc.RotacionarImagem(imagemCinza, self.classificadorFaces)
                facesDetectadas = self.classificadorFaces.CapturarRosto(imagemCinza)
                # Desenha os retangulos de acordo com as faces obtidas pelo haarcascade
                for (x, y, l, a) in facesDetectadas:
                    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
                    roi_gray = imagemCinza[y:y + a, x:x + l]

                    roi_gray = roi_gray.astype("float") / 255.0
                    imagemRedimensionada = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    cv2.normalize(imagemRedimensionada, imagemRedimensionada, alpha=0, beta=1,
                                    norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
                    prediction = treinamento.TestarModelo(imagemRedimensionada)
                    print(prediction)
                    cv2.putText(roi_gray, prediction, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    imagem = cv2.flip(imagem, 0)
                    out.write(imagem)
                # exibe a imagem com os retangulos de acordo com as faces localizadas
                cv2.imshow("Face", imagemCinza)
                # Aguarda 100ms (Captura 10 fps)
                cv2.waitKey(100)
            else:
                break
            # Fecha janela
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty("Face", cv2.WND_PROP_VISIBLE) < 1:
                break

        # Destroi as janelas do open cv
        cv2.destroyAllWindows()
        # Encerra a camera
        camera.release()
        out.release()

    def ExibirImagem(self, imagem):
        cv2.imshow("Face", imagem)
        # Aguarda 100ms (Captura 10 fps)
        cv2.waitKey(100)
        cv2.destroyAllWindows()

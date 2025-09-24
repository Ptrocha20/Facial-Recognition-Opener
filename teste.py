import cv2
import numpy
import serial

# Verificar se os módulos de face do OpenCV estão disponíveis
recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Instalação bem-sucedida!")
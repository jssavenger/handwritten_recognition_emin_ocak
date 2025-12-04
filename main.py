import cv2
import pickle
import matplotlib.pyplot as plt

path = ""

def read_to_image(path):
    img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
    img = img.flatten()
    return img

model = pickle.load(open('model/read_hand_write' , "rb"))
print(f"\nModel is ready.\n")

img = read_to_image(path)

tahmin = model.predict([img])
print(tahmin)
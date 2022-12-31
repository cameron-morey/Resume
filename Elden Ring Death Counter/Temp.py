from PIL import ImageGrab, ImageOps, ImageTk, Image as im
import numpy as np
import cv2
import win32gui
import threading
import time

import tkinter as tk
from tkinter import ttk
from tkinter import * 

FULLSCREEN = False

is_done = False
window_cords = []
app_name = "ELDEN RING"
window_size_x = 1920
window_size_y = 1080


death_x = window_size_x / 2
death_y = window_size_y / 2


def main():
    win32gui.EnumWindows(callback, None)
    
def callback(hwnd, extra):
     if win32gui.GetWindowText(hwnd).replace(u"\u2122", "") == app_name:
          rect = win32gui.GetWindowRect(hwnd)
          window_cords.append((rect[0]))
          print(window_cords[0])
          window_cords.append((rect[1]))
          print(window_cords[1])
          window_cords.append((rect[2]))
          print(window_cords[2])
          window_cords.append((rect[3]))
          print(window_cords[3])
          window_size_x = rect[2] - rect[0]
          window_size_y = rect[3] - rect[1]
          death_x = window_size_x / 2
          death_y = window_size_y / 2

if __name__ == '__main__':
    main()
    
def CalculateMSE(actual, prediction):
    error = (np.square(np.array(actual) - np.array(prediction)) / 2)
    error = np.sum(error) / len(error)
    return error
    
death_picture_reference = 'death2.JPEG'
alive_picture_dark = 'alive.JPEG'
alive_picture_light = 'alive2.JPEG'

reference = im.open(death_picture_reference)
reference2 = im.open(alive_picture_dark)
reference3 = im.open(alive_picture_light)

reference = ImageOps.grayscale(reference)
reference2 = ImageOps.grayscale(reference2)
reference3 = ImageOps.grayscale(reference3)

# reference = np.array(reference) * 10

# reference = im.fromarray(reference)
reference.show()
reference.save('normal.JPEG')

img = ImageGrab.grab((window_cords[0] + death_x + 50, window_cords[1] + death_y + 125,
                      window_cords[2] - death_x - 50, window_cords[3] - death_y - 125))

img_np = np.array(img)

# cv2.imshow('Image', cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY))
# cv2.waitKey(10)

img = im.fromarray(img_np)
    
img = ImageOps.grayscale(img)

error = CalculateMSE(reference, img)

error_alive = CalculateMSE(reference2, img)
error_alive += CalculateMSE(reference3, img)
error_alive /= 2 

# while True:
#     time.sleep(2)
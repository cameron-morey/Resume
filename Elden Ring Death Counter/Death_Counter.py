from PIL import ImageGrab, ImageOps, ImageTk, Image as im
import numpy as np
import cv2
import win32gui
import threading

import tkinter as tk
from tkinter import ttk
from tkinter import * 

FULLSCREEN = False

is_done = False
window_cords = []
app_name = "ELDEN RING"
window_size_x = 1920
window_size_y = 1080

death_picture_reference = 'death5.JPEG'
alive_picture_dark = 'alive.JPEG'
alive_picture_light = 'alive2.JPEG'

death_x = window_size_x / 2
death_y = window_size_y / 2

reference = im.open(death_picture_reference)

reference = ImageOps.grayscale(reference)

reference = np.array(reference) * 2

_, reference_image_edges = cv2.threshold(reference, 50, 255, cv2.THRESH_BINARY)
reference_contours_mask, _ = cv2.findContours(reference_image_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

reference_canvas = np.zeros(reference.shape, np.uint8)
canvas_image = np.zeros(reference.shape, np.uint8)

reference_canvas.fill(255)

cv2.drawContours(reference_canvas, reference_contours_mask, -1, (0, 255, 0), 3)

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

def UpdateScreen():
    time = .5
    if not is_done:
        threading.Timer(time, UpdateScreen).start()
    img = ScreenGrab()
    # Do everything with screen here
    
def ScreenGrab():
    #Create reference so when app closes I can stop the thread
    error_max = 45
    print("Screen Grab")
    if not FULLSCREEN:
          img = ImageGrab.grab((window_cords[0] + death_x + 50, window_cords[1] + death_y + 125,
                                window_cords[2] - death_x - 50, window_cords[3] - death_y - 125))
    else:
          img = ImageGrab.grab((window_cords[0], window_cords[1], window_cords[2], window_cords[3]))
          
    img_np = np.array(img)

    img = im.fromarray(img_np)
    
    # img.save('death5.JPEG')
    
    # reference = im.open(death_picture_reference)
    # reference2 = im.open(alive_picture_dark)
    # reference3 = im.open(alive_picture_light)
    
    # reference = ImageOps.grayscale(reference)
    # reference = np.array(reference) * 10
    # reference = im.fromarray(reference)

    # reference2 = ImageOps.grayscale(reference2)
    # reference2 = np.array(reference2) * 10
    # reference2 = im.fromarray(reference2)
    
    # reference3 = ImageOps.grayscale(reference3)
    # reference3 = np.array(reference3) * 10
    # reference3 = im.fromarray(reference3)
    
    img = ImageOps.grayscale(img)
    
    img = np.array(img) * 2
    
    
    thresh, image_edges_screen = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours_mask, hierarchy = cv2.findContours(image_edges_screen, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    canvas_image.fill(255)
    cv2.drawContours(canvas_image, contours_mask, -1, (0, 255, 0), 3)
    
    # print(reference_contours_mask)
    # print(contours_mask)
    
    # error = cv2.matchShapes(reference_contours_mask[0], contours_mask[0], 1, 0.0)
    error = CalculateMSE(reference_canvas, canvas_image)
    print(error)
    

    # error = CalculateMSE(reference, img)
    
    # error_alive = CalculateMSE(reference2, img)
    # error_alive += CalculateMSE(reference3, img)
    # error_alive /= 2 
    
    # reference = im.open(death_picture_reference)
    
    # reference = ImageOps.grayscale(reference)
    
    # reference = np.array(reference) * 5
    
    # thresh, image_edges = cv2.threshold(reference, 100, 255, cv2.THRESH_BINARY)
    
    # canvas = np.zeros(reference.shape, np.uint8)
    # canvas.fill(255)
    
    # contours_mask, hierarchy = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # canvas = cv2.drawContours(canvas, contours_mask, -1, (0, 255, 0), 3)
    
    # cv2.imshow('Image', reference_canvas)
    # cv2.imshow('Image', canvas_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

    if error < error_max:
        print("Died")
    
    # cv2.imshow('Image', cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(10)
  
    return img

# this is the function called when the button is clicked
def btnClickFunction():
	print('clicked')


# this is the function called when the button is clicked
def btnClickFunction():
	print('clicked')


def Main_Function():
    #Starts the infinite screengrab process
    UpdateScreen()


UpdateScreen()

root = Tk()

# This is the section of code which creates the main window
root.geometry('720x480')
root.configure(background='#F0F8FF')
root.title('Hello, I\'m the main window')

# image = ScreenGrab()

canvas = Canvas(root, width = 300, height = 300)  
canvas.pack()  
# img = ImageTk.PhotoImage(image)  
# canvas.create_image(20, 20, anchor=NW, image=img) 


# This is the section of code which creates a button
Button(root, text='Button text!', bg='#F0F8FF', font=('arial', 12, 'normal'), command=btnClickFunction).place(x=64, y=371)


# This is the section of code which creates a button
Button(root, text='Stop', bg='#F0F8FF', font=('arial', 12, 'normal'), command=btnClickFunction).place(x=204, y=370)


# This is the section of code which creates the a label
Label(root, text='Counter', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=184, y=329)


root.mainloop()
import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import  Image
import numpy as np

#import file
# import detector

window = tk.Tk()
window.title("Face Recognition System")
# window.config(background="lime")

l1=tk.Label(window,text="NIM",font=("Cambria",20))
l1.grid(column=0, row=0)
t1=tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2=tk.Label(window,text="Nama",font=("Cambria",20))
l2.grid(column=0, row=1)
t2=tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

b1=tk.Button(window, text="Training", font=("Cambria", 20), bg="Blue", fg="White")
b1.grid(column=0, row=2)

def deteksi_wajah():
    import cv2
    import numpy as np
    import sqlite3
    import os
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    fname = "recognizer/trainingData.yml"
    if not os.path.isfile(fname):
        print("Please train the data first")
        exit(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(fname)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            ids, conf = recognizer.predict(gray[y:y + h, x:x + w])
            c.execute("select name from users where id = (?);", (ids,))
            result = c.fetchall()
            name = result[0][0]
            if conf < 50:
                cv2.putText(img, name, (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 0), 2)
            else:
                cv2.putText(img, 'No Match', (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Recognizer', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # ESC key
            break
    cap.release()
    cv2.destroyAllWindows()

b2=tk.Button(window, text="Deteksi Wajah", font=("Cambria", 20), bg="Blue", fg="White")
b2.grid(column=1, row=2)

b3=tk.Button(window, text="Generate Dataset", font=("Cambria", 20), bg="Blue", fg="White", command=deteksi_wajah)
b3.grid(column=2, row=2)

def menu2():
    window2 = tk.Tk()
    window2.title("Login")
    window2.geometry("800x200")
    window2.mainloop()

b3=tk.Button(window, text="Generate Dataset", font=("Cambria", 20), bg="Blue", fg="White", command=menu2)
b3.grid(column=2, row=3)

window.geometry("800x200")
window.mainloop()


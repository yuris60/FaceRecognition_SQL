import sqlite3
import eel
import os
import random
import cv2
import os
import numpy as np
from PIL import Image
import sqlite3

eel.init('web')
eel.mAddLogo()

@eel.expose                         # Expose this function to Javascript
def new_window():
    eel.start('main2.html', size=(500, 800))

@eel.expose
def handleinput(inp, inp2, inp3):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    c.execute('INSERT INTO users (nim, nama, kelas) VALUES (?,?,?)', (inp, inp2, inp3,))

    uid = c.lastrowid

    sampleNum = 0

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1
            cv2.imwrite("dataset/User." + str(uid) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.waitKey(100)
        cv2.imshow('img', img)
        cv2.waitKey(1);
        if sampleNum > 20:
            break
    cap.release()
    conn.commit()
    conn.close()
    cv2.destroyAllWindows()

    eel.info("Wajah "+ inp2 + "berhasil disimpan")
    #os.system("rm output.avi")


@eel.expose
def detect_faces():
    # import cv2
    eel.info("Press ESC to stop")

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    fname = "recognizer/trainingData.yml"
    # if not os.path.isfile(fname):
    #     print("Please train the data first")
    #     exit(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(fname)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            ids, conf = recognizer.predict(gray[y:y + h, x:x + w])
            c.execute("select nim, nama from users where id = (?);", (ids,))
            result = c.fetchall()
            nim = result[0][0]
            nama = result[0][1]
            if conf < 50:
                cv2.putText(img, nim+" "+nama, (x+2, y+h-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 0), 2)
                # cv2.putText(img, nama, (x + 3, y + h - 200),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 0), 2)
            else:
                cv2.putText(img, 'No Match', (x+2, y+h-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Deteksi Wajah', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # ESC key
            break
    cap.release()
    cv2.destroyAllWindows()
    eel.info("")


@eel.expose
def train_images():
    eel.info("Dataset wajah sedang dilatih, mohon untuk menunggu..")
    eel.mSpinner()
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path = 'dataset'
        if not os.path.exists('./recognizer'):
            os.makedirs('./recognizer')

        def getImagesWithID(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            IDs = []
            for imagePath in imagePaths:
                faceImg = Image.open(imagePath).convert('L')
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
                cv2.imshow("Latih Dataset Citra", faceNp)
                cv2.waitKey(10)
            return np.array(IDs), faces

        Ids, faces = getImagesWithID(path)
        recognizer.train(faces, Ids)
        recognizer.save('recognizer/trainingData.yml')
        cv2.destroyAllWindows()
        eel.info("Dataset wajah berhasil di latih")
        eel.mSpinner()
        eel.mAddTick()
    except:
        eel.info("Oops.. sepertinya ada kesalahan..")


eel.start('main.html', size=(900, 675))

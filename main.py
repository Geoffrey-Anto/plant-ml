print("Importing libraries...")
import config
import model
import tensorflow as tf
import numpy as np

print(tf.__version__)

model = tf.keras.models.load_model(config.save_model_path)

print("Enter the input format: ")
print("Press 1 for image")
print("Press 2 for video")
inputFormat = input()

if inputFormat == "image":
    from tkinter import Label,Tk
    from PIL import Image, ImageTk
    from tkinter import filedialog
    root = Tk()
    path=filedialog.askopenfilename(filetypes=config.file_types_for_image)
    im = Image.open(path)
    im_np = im.resize((256,256))
    im_np = np.array(im_np)
    im_np = im_np.reshape(1,256,256,3)
    prediction = model.predict(im_np)
    prediction = np.argmax(prediction)
    tkimage = ImageTk.PhotoImage(im)
    pred = config.classes[prediction]
    pred = pred.capitalize()
    pred = "Prediction: " + pred
    myvar=Label(root,text=pred,font=("Helvetica", 16))
    myvar.pack()
    myvar=Label(root,image = tkimage)
    myvar.image = tkimage
    myvar.pack()
    root.mainloop()
elif inputFormat == "video":
    import cv2 as cv
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("error opening camera")
        exit()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("error in retrieving frame")
            break
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_np = cv.resize(img, (256, 256))
        img_np = np.array(img_np)
        img_np = img_np.reshape(1, 256, 256, 3)
        prediction = model.predict(img_np)
        prediction = np.argmax(prediction)
        pred = config.classes[prediction]
        pred = pred.capitalize()
        pred = "Prediction: " + pred
        cv.putText(img, pred, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('frame', img)
        if cv.waitKey(1) == ord('q'):
            break
        
    cam.release()
    cv.destroyAllWindows()
else:
    print("Invalid input format")
    exit()
import os
import random
from flask import Flask, render_template, request, redirect
import time
from keras.models import Model
from keras.layers import Dense, GlobalMaxPooling2D,Input,Dropout
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import os,random
from keras.preprocessing.image import img_to_array, load_img
from keras.backend import tensorflow_backend as backend
import random
import string
import cv2

app = Flask(__name__)
app.config['DEBUG'] = True

ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'JPG'])
UPLOAD_FOLDER = './static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/dfg4s4dfg4sdf54g545df4sgsdfg6sd5fgs")
def hello():
    return render_template("index.html")


@app.route("/sdfsdfasls546465ASDFAD545asdffasdlk")
def detail():

    return render_template("detail.html",path=path,label_1=label_1,label_2=label_2,label_3=label_3,img_1=img_1,img_2=img_2,img_3=img_3)


@app.route('/dfg4s4dfg4sdf54g545df4sgsdfg6sd5fgs/upload', methods=['GET','POST'])
def result():

 if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename="graph" + str(time.time()) + ".jpg"
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            global path
            path=os.path.join(app.config['UPLOAD_FOLDER'], filename)


                    # 画像の読み込み
            image_gs = cv2.imread('static/images/'+filename)

            # 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
            cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
            #判定！
            face = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=1,minSize=(10,10))

            #print (type(face))
            #print (face)

            if isinstance(face,np.ndarray):
                cut_file='cut_'+filename
                for x,y,w,h in face:
                    face_cut = image_gs[y:y+h, x:x+w]
                    #切り抜き保存！
                    cv2.imwrite('static/images/'+cut_file, face_cut)





            #処理
            file_name='monkey'
            display_dir='static'
            images='images'
            label=['チンパンジー','ゴリラ','サル']

            N_CATEGORIES  = 3
            IMAGE_SIZE = 224
            BATCH_SIZE = 8
            input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
            base_model = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)
            x = base_model.output
            x = GlobalMaxPooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(2048, activation='relu')(x)
            x = Dropout(.25)(x)
            x = Dense(1024, activation='relu')(x)

            predictions = Dense(N_CATEGORIES, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

            model.load_weights(file_name+'.hdf5')

            model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])


            files=os.listdir(display_dir)





            plt.figure(figsize=(10,10)) #場所？

            for i in range(1):
                temp_img=load_img(os.path.join(display_dir,images,filename),target_size=(224,224))
                plt.subplot(5,5,i+1) #画像の大きさ
                plt.imshow(temp_img) #画像

                temp_img_array=img_to_array(temp_img)
                temp_img_array=temp_img_array.astype('float32')/255.0
                temp_img_array=temp_img_array.reshape((1,224,224,3))

                img_pred=model.predict(temp_img_array)
                plt.title(label[np.argmax(img_pred)]) #ラベル

                plt.xticks([]),plt.yticks([]) #座標を出さない

                backend.clear_session()

                global img_1,img_2,img_3,label_1,label_2,label_3

                label_1=label[0]
                img_1=round(img_pred[0][0]*100,2)
                label_2=label[1]
                img_2=round(img_pred[0][1]*100,2)
                label_3=label[2]
                img_3=round(img_pred[0][2]*100,2)



                if img_2 < img_3:
                   img_number=img_2
                   img_2=img_3
                   img_3=img_number
                   label_name=label_2
                   label_2=label_3
                   label_3=label_name

                if img_1 < img_2:
                   img_number=img_1
                   img_1=img_2
                   img_2=img_number
                   label_name=label_1
                   label_1=label_2
                   label_2=label_name

                if img_2 < img_3:
                   img_number=img_2
                   img_2=img_3
                   img_3=img_number
                   label_name=label_2
                   label_2=label_3
                   label_3=label_name




            return  render_template('result.html',path=path,label_1=label_1,label_2=label_2,label_3=label_3,img_1=img_1,img_2=img_2,img_3=img_3)
        else:
            return ''' <p>許可されていない拡張子です</p> '''
 else:
        return render_template("index.html")



if __name__ == "__main__":
    # webサーバー立ち上げ
    app.run(debug=True,port=6006)

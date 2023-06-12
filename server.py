from flask import Flask,jsonify
from flask import request
import os
import base64
import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import threading
from utils import train_model,inference
from text_spech import text_to_speech
from database import InsertDB,UpdateDB,GetDB

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

app=Flask(__name__)

@app.route('/')
def hello_world():
    return '<div>Hola mundo</div1'


@app.route('/attendance',methods=['POST'])
def SaveData():
    
    img_data=request.json['img']
    nombre=request.json['nombre']
    
    #instanciamos la fecha y hora para poder hacer consultas en la base de datos
    now=datetime.datetime.now()
    hora=now.strftime("%H:%M:%S")
    fecha=now.strftime("%d/%m/%Y")
    f=fecha.split('/')
    f=f[0]+'-'+f[1]+'-'+f[2]
    path='./historial/'+f+'/'+nombre
    
    #si el nombre del usuario con la fecha actual no existe, se inserta el nombre con la fecha actual
    getdb,data=GetDB(nombre,fecha)
    if getdb<1:
        #se crea una carpeta con el nombre del usuario
        
        path_enter=path+'/arrive/'
        if not os.path.exists(path_enter):
            os.makedirs(path_enter)
            
        path_img=path_enter+'arrive.png'
        info_db=(nombre,fecha,hora,None,path_img,None,0)
       
        cv2.imwrite(path_img,np.array(img_data))
        InsertDB(info_db)
        
        def worker():
            text_to_speech(nombre+", Bienvenido a smart city")
            
        
        
    else:# si el usuario con el nombre actual existe, se actualiza la hora de salida
        
        path_leave=path+'/leave/'
        if not os.path.exists(path_leave):
            os.makedirs(path_leave)
        path_img=path_leave+'leave.png'
        status=data[0][-1]
        
        msg=''
        if status==0:
            st=1
            msg='Nos vemos ' 
        else:
            st=0
            msg='Bienvenido de vuelta '
        
        UpdateDB(nombre,hora,fecha,path_img,st)
        cv2.imwrite(path_img,np.array(img_data))
        
        def worker():
            text_to_speech(msg+nombre)
        
    #instancias el trabajor para correr en un hilo diferente, la confirmacion de asistencia
    hilo1=threading.Thread(target=worker)
    hilo1.start()
    

    return nombre


@app.route('/users',methods=['GET'])
def User():
    def unique(list1):
        unique_list=[]
        for x in list1:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
    load_data=torch.load('data.pt')
    name_list=load_data[0]

    print(unique(name_list))
    #print(type(name_list))

    return jsonify({'lista':unique(name_list)})


@app.route('/SavePhoto',methods=['POST'])
def Data():
    directory='./train'
    if not os.path.exists(directory):
        os.mkdir(directory)
    img_data=request.json['img_lst']
    data_name=request.json['data']
    path_name=data_name["nombre"]+' '+data_name["apellido"]
    path=directory+'/'+path_name
    
    if not os.path.exists(path):
        os.makedirs(path)    
    
    print(path)
    
    
    
    
    for i,j in enumerate(img_data):
        with open(path+"/"+data_name["nombre"]+str(i)+".png","wb") as f:
            f.write(base64.b64decode(j.split(',')[1]))

    dim=len(os.listdir(path))
    
    x = datetime.datetime.now()
   
    res=jsonify({'msg':'Images have saved succesfully '+ x.strftime("%c"),
             "count":dim
             })
    return res

@app.route('/Train',methods=['GET'])
def Training():
    path_train='./train'
    data=train_model(path_train)
   #data == [name,embeddings]
    torch.save(data, './demo/data_server.pt') # saving data.pt file 
       
    if os.path.exists('./demo/data_server.pt'):
        response=jsonify({'msg':'data file was created successfully'})
        response.status_code=200
        return response
    
    response=jsonify({'msg':'the data file was not created successfully'})
    response.status_code=500
    return response

@app.route('/Inference',methods=['GET'])
def realTime():
    #path_file='./demo/data_server.pt'
    path_file='./demo/one_per.pt'
    inference(path_file)
    response=jsonify({'msg':'Inference was finished...'})
    response.status_code=200
    return response
    
if __name__=='__main__':
    app.run(debug=True)

from flask import Flask,jsonify
from flask import request
import os
import base64
import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import threading

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

    nombre=request.json['nombre']
    
    #instanciamos la fecha y hora para poder hacer consultas en la base de datos
    now=datetime.datetime.now()
    hora=now.strftime("%H:%M:%S")
    fecha=now.strftime("%d/%m/%Y")
    #si el nombre del usuario con la fecha actual no existe, se inserta el nombre con la fecha actual
    if GetDB(nombre,fecha)<1:
        info_db=(3,nombre,fecha,hora,None)
        InsertDB(info_db)
    else:# si el usuario con el nombre actual existe, se actuliza la hora de salida
        UpdateDB(nombre,hora,fecha)
    
    #instancias el trabajor para correr en un hilo diferente, la confirmacion de asistencia
    def worker():
        text_to_speech(nombre)
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
    name_list=load_data[1]

    print(unique(name_list))
    #print(type(name_list))

    return jsonify({'lista':unique(name_list)})


@app.route('/SavePhoto',methods=['POST'])
def Data():
    directory='train'
    img_data=request.json['img_lst']
    data_name=request.json['data']
    path_name=data_name["nombre"]+' '+data_name["apellido"]
    path=directory+'/'+path_name
    
    if not os.path.exists(path):
        os.makedirs(path)    
    
    print(path)
    #if os.path.exists()
    
    
    
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
    
    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,keep_all=False,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    workers = 0 if os.name == 'nt' else 4
    def collate_fn(x):
        return x[0]

    path_train='./train'
    dataset = datasets.ImageFolder(path_train)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    name_list=[]
    embedding_list=[]
    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True) 
        if face is not None and prob>0.92:
            emb = resnet(face.unsqueeze(0)).to(device) 
            embedding_list.append(emb.detach()) 
            name_list.append(dataset.idx_to_class[idx]) 

    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file    
    if os.path.exists('data.pt'):
        response=jsonify({'msg':'data file was created successfully'})
        response.status_code=200
        return response
    
    response=jsonify({'msg':'the data file was not created successfully'})
    response.status_code=500
    return response

if __name__=='__main__':
    app.run(debug=True)

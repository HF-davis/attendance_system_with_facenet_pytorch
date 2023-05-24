import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import datetime
import numpy as np
import faiss
import time
import requests
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

data=torch.load('./demo.pt')
names=data[0]
embeddings=data[1]


resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#mtcnn = MTCNN(
#    image_size=160, margin=20, min_face_size=20,
#    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
#    device=device
#)
mtcnn = MTCNN(image_size=160, margin=0,device=device)

cpu=torch.device('cpu')
m=[]
for emb in embeddings:
	arr=np.array(emb.to(cpu)).astype("float32")
	m.append(arr.flatten())

m=np.array(m)  #(180,1,512)
print(m.shape)

index=faiss.IndexFlatL2(m.shape[1])
index.add(m)

def makeSurePerson(idx):
        unique=[]
        all=[]
        for i in idx:
            all.append(names[i])
            if names[i] not in unique:
                unique.append(names[i])
        count=[0 for _ in range(len(unique))]

        for i in range(len(unique)):
            for j in range(len(all)):
                if unique[i]==all[j]:
                    count[i]=count[i]+1
        idc=all.index(unique[np.argmax(count)])
        #print(unique[np.argmax(count)])

        return unique[np.argmax(count)],idc
    
def recognition(frame_):
    img=frame_.copy()
    frame_=Image.fromarray(cv2.cvtColor(frame_,cv2.COLOR_BGR2RGB))
    img_cropped,prob=mtcnn(frame_,return_prob=True)
    
    boxes,_=mtcnn.detect(frame_)
    if img_cropped is not None:
        
        if prob>0.95:
            img_embedding=resnet(img_cropped.unsqueeze(0).to(device)).detach()
            
            
            img_emb=img_embedding.to(cpu)
            query=np.array(img_emb).astype("float32")
            D,I=index.search(query,5) #D es una lista de listas que contiene las distancias obtenidas por la busqueda de similaridades, I son los indices de estos.
            
            
            name,id=makeSurePerson(I.flatten())
            min_dist=D.flatten()[id]
            
            
            box=boxes[0]
            original=img.copy()
            if min_dist<0.6:
                #print(name)
                img=cv2.putText(img,name+' '+str(min_dist),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                #requests.post("http://127.0.0.1:5000/attendance",json={'nombre':name})
            img=cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
            time.sleep(1)
    return img




upper_left = (650, 280)
bottom_right = (1050, 830)

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        print("fail to grab frame, try again")
        break
    cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 255))
    rect_img=frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]=recognition(rect_img)
    cv2.imshow('webCam',frame)
    if cv2.waitKey(20)==27:
            break

capture.release()
cv2.destroyAllWindows()

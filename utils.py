from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm
import gc
import faiss
import cv2
import numpy as np
import requests
import time
#First we need to preproces data, we can do that with preprocess_data.py script
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


def train_model(path_t):
  print(path_t)
  
  if device=='cuda:0':
      gc.collect()
      torch.cuda.empty_cache()
      torch.backends.cuda.cufft_plan_cache.clear()
  
  mtcnn = MTCNN(
      image_size=160, margin=20, min_face_size=20,
      thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
      device=device
  )

  resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

  def collate_fn(x):
      return x[0]
  print(os.path.exists(path_t))

  dataset = datasets.ImageFolder(path_t)
  dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
  loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

  aligned = []
  names = []
  for x, y in tqdm(loader):
      x_aligned, prob = mtcnn(x, return_prob=True)
      if x_aligned is not None:
          #print('Face detected with probability: {:8f}'.format(prob))
          aligned.append(x_aligned)
          names.append(dataset.idx_to_class[y])

  batch=8
  start=0
  ebb=[]
  
  aligned = torch.stack(aligned).to(device)
  n=int(len(aligned)/batch)
  #start_t=time.time()
  for i in tqdm(range(1,n+1)):
    if i*batch<len(aligned):
      count=count+i*batch-start
      ebb.append(resnet(aligned[start:i*batch]).detach().to(device))
      start=i*batch
    if (i+1)*batch>len(aligned):
      ebb.append(resnet(aligned[start:len(aligned)]).detach().to(device))

  embeddings=torch.stack(ebb)
  m=embeddings[0]
 
  for i in tqdm(range(1,len(embeddings))):
    m=torch.vstack((m,embeddings[i]))
  print(m.size())
  
  return [names,m]


def makeSurePerson(idx,names):
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

def recognition(frame_,mtcnn,resnet,cpu,index,names):
    img=frame_.copy()
    frame_=Image.fromarray(cv2.cvtColor(frame_,cv2.COLOR_BGR2RGB))
    img_cropped,prob=mtcnn(frame_,return_prob=True)
    
    boxes,_=mtcnn.detect(frame_)
    if img_cropped is not None:
        
        if prob>0.95:
            img_embedding=resnet(img_cropped.unsqueeze(0).to(device)).detach()
            
            #st_fa=time.time()
            img_emb=img_embedding.to(cpu)
            
            query=np.array(img_emb).astype("float32")
            D,I=index.search(query,5) #D es una lista de listas que contiene las distancias obtenidas por la busqueda de similaridades, I son los indices de estos.
            
            #print("flatten: ",I.flatten())
            name,id=makeSurePerson(I.flatten(),names)
            min_dist=D.flatten()[id]
            #ed_fa=time.time()
            #print('search time: ',ed_fa-st_fa)
        
            
            #print('dist: ',min_dist)
            box=boxes[0]
            original=img.copy()
            if min_dist<0.6:
                #print(name)
                img=cv2.putText(img,name+' '+str(min_dist),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                #enc_string=base64.b64encode(img)
                #requests.post("http://127.0.0.1:5000/attendance",json={'nombre':name,'img':img.tolist()})
            img=cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
            time.sleep(0.5)
    return img
   
       
def inference(path_file):
  
  data=torch.load(path_file)
  names=data[0]
  embeddings=data[1]
  print("name_size: ",len(names))
  print("embeddings_size: ",len(embeddings))
  resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
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
  
  upper_left = (150, 70) 
  bottom_right = (550, 430)
  capture = cv2.VideoCapture(0)

  while True:
      ret, frame = capture.read()
      if not ret:
          print("fail to grab frame, try again")
          break
      cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 255))
      rect_img=frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
      frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]=recognition(rect_img,mtcnn,resnet,cpu,index,names)
      cv2.imshow('webCam',frame)
      if cv2.waitKey(20)==27:
              break

  capture.release()
  cv2.destroyAllWindows()

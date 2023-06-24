from threading import Thread
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import numpy as np
import faiss
import time

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

data=torch.load('./lab_data_sc.pt')
names=data[0]
embeddings=data[1]
print("name_size: ",len(names))
print("embeddings_size: ",len(embeddings))
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
        #print(idx)
        for i in idx:
            #print("idx: ",i)
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
            
            #st_fa=time.time()
            img_emb=img_embedding.to(cpu)
            
            query=np.array(img_emb).astype("float32")
            D,I=index.search(query,5) #D es una lista de listas que contiene las distancias obtenidas por la busqueda de similaridades, I son los indices de estos.
            
            #print("flatten: ",I.flatten())
            name,id=makeSurePerson(I.flatten())
            min_dist=D.flatten()[id]
            #ed_fa=time.time()
            #print('search time: ',ed_fa-st_fa)
        
            
            #print('dist: ',min_dist)
            box=boxes[0]
            original=img.copy()
            if min_dist<0.6:
                print(name)
                img=cv2.putText(img,name+' '+str(min_dist),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                #enc_string=base64.b64encode(img)
                requests.post("http://127.0.0.1:5000/attendance",json={'nombre':name,'img':img.tolist()})
		time.sleep(2)
            img=cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
            
    return img
  
  
class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id # default is 0 for main camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)        # self.stopped is initialized to False 
        self.stopped = True        # thread instantiation  
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t.start()    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()    # method to return latest read frame 
    def read(self):
        return self.frame    # method to stop reading frames 
    def stop(self):
        self.stopped = True

#cambiar el tamaño de la region de interes
upper_left = (600, 150)
bottom_right = (850, 500)

webcam_stream = WebcamStream(stream_id='') # 0 para la camara principal, colocar el link de la camara IP
webcam_stream.start()# processing frames in input stream
num_frames_processed = 0 
start = time.time()

while True :
    if webcam_stream.stopped is True :
        break
    else :
        frame = webcam_stream.read()    # adding a delay for simulating video processing time 
    delay = 0.03 # delay value in seconds
    time.sleep(delay) 
    num_frames_processed += 1    # displaying frame
    cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 255))
    rect_img=frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]=recognition(rect_img) 
    cv2.imshow('frame' , frame)
    key = cv2.waitKey(20)
    if key == 27:
        break
end = time.time()
webcam_stream.stop() # stop the webcam stream
cv2.destroyAllWindows()

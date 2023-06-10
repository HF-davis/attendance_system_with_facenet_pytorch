from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm
import gc
#First we need to preproces data, we can do that with preprocess_data.py script
path_t='./demo/lab_data'
path_dest='' #ruta de donde se va a almacenar el archivo .pt
print(path_t)
  
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device=='cuda:0':
      gc.collect()
      torch.cuda.empty_cache()
      torch.backends.cuda.cufft_plan_cache.clear()
print('Running on device: {}'.format(device))

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
      if x_aligned is not None and prob>0.92:
          #print('Face detected with probability: {:8f}'.format(prob))
          aligned.append(x_aligned)
          names.append(dataset.idx_to_class[y])

batch=8
start=0
ebb=[]
  
aligned = torch.stack(aligned).to(device)
n=int(len(aligned)/batch)
  #start_t=time.time()
print('aligned_size: ',len(aligned))
print('names_size:',len(names))

for i in tqdm(range(1,n+1)):
    if i*batch<len(aligned):
      #count=count+i*batch-start
      ebb.append(resnet(aligned[start:i*batch]).detach().to(device))
      start=i*batch
    if (i+1)*batch>len(aligned):
      ebb.append(resnet(aligned[start:len(aligned)]).detach().to(device))

##embeddings=torch.stack(ebb)
##
embeddings=[]
print('ebb[0][0]: ',ebb[0][0].size())
for i in range(len(ebb)):
      for j in ebb[i]:
            embeddings.append(j.unsqueeze(0))
            
print('embedding_size: ',len(embeddings))

data=[names,embeddings]
                                                                                                                                                                                                                                                                                                                                                                                                                                                               
torch.save(data,path_dest)
if os.path.exists(path_dest):
    print('data file was created successfully')

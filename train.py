#install requirements with:  pip install -r requirements.txt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm
import gc
import time
#First we need to preproces data, we can do that with preprocess_data.py script

#ruta en donde se encuentra nuestro conjunto de datos
path_train='./one_face_per_person'

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#si tenemos gpu, limpiamos la memoria caché de la gpu
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
print(os.path.exists(path_train))

dataset = datasets.ImageFolder(path_train)
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

        
batch=4 #el batch size suele ser potencia de 2
start=0
ebb=[]
aligned = torch.stack(aligned).to(device)
#aligned[0]
n=int(len(aligned)/batch)


for i in tqdm(range(1,n+2)):
  if i*batch<len(aligned):
    ebb.append(resnet(aligned[start:i*batch]).detach().to(device))
    start=i*batch
  else:
    ebb.append(resnet(aligned[start:len(aligned)]).detach().to(device))

embeddings=torch.stack(ebb)
m=embeddings[0]

for i in tqdm(range(1,len(embeddings))):
  m=torch.vstack((m,embeddings[i]))
m.size()                                                                                                                                                    
#embeddings = resnet(aligned).detach().cpu()                                                                                                                                                                                                                                                                                                           

data=[names,m] #el orden para guardar esto es muy importante, ya que al momento de la inferencia ser usará
torch.save(data,'embeddings.pt')
if os.path.exists('embeddings.pt'):
    print('data file was created successfully')

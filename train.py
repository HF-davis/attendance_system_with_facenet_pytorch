#install requirements with:  pip install -r requirements.txt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import time
import os
import argparse
from tqdm import tqdm

parse=argparse.ArgumentParser()
parse.add_argument('-t','--train',default='./train',help="ingrese la ruta de la carpeta de entrenamiento")
args=parse.parse_args()
path_train=args.train

print('ruta de la carpeta de entrenamiento: ',path_train)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#mtcnn = MTCNN(
#    image_size=160, margin=0, min_face_size=20,keep_all=False,
#    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
#    device=device
#)
mtcnn = MTCNN(image_size=160, margin=0,select_largest=False,post_process=False,device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

workers = 0 if os.name == 'nt' else 4

def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder(path_train)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

name_list=[]
embedding_list=[]

start=time.time()

for img, idx in tqdm(loader):
    face, prob = mtcnn(img, return_prob=True) 
    if face is not None and prob>0.92:
        
        emb = resnet(face.unsqueeze(0)).to(device) 
        embedding_list.append(emb.detach()) 
        name_list.append(dataset.idx_to_class[idx]) 

data = [embedding_list, name_list] 
torch.save(data, 'data_complete.pt') # saving data.pt file

end=time.time()
print("Training was finished in {} seconds".format(end-start))

if os.path.exists('data_complete.pt'):
    print('data file was created successfully')

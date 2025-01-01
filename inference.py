
from dataset import MNIST
from text_encoder import TextEncoder
import matplotlib.pyplot as plt 
import torch 
from clip import CLIP
import torch.nn.functional as F

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
dataset=MNIST()
text_enc = TextEncoder().to(DEVICE)
model=CLIP().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))
text_enc.eval()
model.eval()
strr = 'the number is '
kk = ['zero','one','two','three','four','five','six','seven','eight','nine']
pp = []
images,labels=zip(*[dataset[i] for i in range(10)])
images_tensor = torch.stack(images)
print('正确分类:',labels,images_tensor.size())

targets=torch.arange(0,10)
for i in range(10):
    pp.append(strr + kk[i])
logits=model(images_tensor.to(DEVICE),pp)
for row in  logits:
    print('CLIP分类:',strr+str(row.argmax(-1).item()))
print(logits.shape)
#print('CLIP分类:',strr+str(logits.argmax(-1).item()))


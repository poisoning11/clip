import torch 
from dataset import MNIST
from clip import CLIP
import torch.nn.functional as F
from text_encoder import TextEncoder
from torch.utils.data import DataLoader
import os 

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集

model=CLIP().to(DEVICE)

try:    # 加载模型
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 

optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)
'''
    训练模型
'''
ITER_BATCH_COUNT=100000
BATCH_SIZE=64
TARGET_COUNT=10
strr = 'the number is '
kk = ['zero','one','two','three','four','five','six','seven','eight','nine']
pp = []
text_enc = TextEncoder().to(DEVICE)
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)
if __name__ == '__main__':

    for i in range(ITER_BATCH_COUNT):
        while True:
            imgs,labels=next(iter(dataloader))
            if torch.unique(labels).shape[0]<TARGET_COUNT:
                continue
            target=set()
            indexes=[]
            for j in range(BATCH_SIZE):
                if labels[j].item() in target:
                    continue
                target.add(labels[j].item())
                indexes.append(j)
                if len(target)==TARGET_COUNT:
                    break
            imgs=imgs[indexes]
            labels=labels[indexes]
            break
        pp = []
        for value in labels:
            pp.append(strr + kk[value.item()])
        logits=model(imgs.to(DEVICE),pp)
        #print(labels)

        targets=torch.arange(0,TARGET_COUNT).to(DEVICE)

        #print(targets.size(),logits.size())
        loss=F.cross_entropy(logits,targets)
        #loss_t=F.cross_entropy(logits.permute(1,0),labels)
        #loss=(loss_i+loss_t)/2

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if i%1000==0:
            print('iter:{},loss:{}'.format(i,loss))
            torch.save(model.state_dict(),'.model.pth')
            os.replace('.model.pth','model.pth')
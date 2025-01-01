from torch import nn 
import torch 
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'

en_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', 'the', 'number', 'is', 'one', 'two', 'three','four','five','six','seven','eight','nine','ten']
en_vocab = build_vocab_from_iterator([en_tokens], specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM], special_first=True)
en_vocab.set_default_index(UNK_IDX)
# 初始化模型
embedding_dim = 16  #
def preprocess_sentence(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    token_ids = [vocab[token] if token in vocab else UNK_IDX for token in tokens]
    return torch.tensor(token_ids).unsqueeze(0)
strr = 'the number is '
kk = ['zero','one','two','three','four','five','six','seven','eight','nine']
pp = []
sentence_embedding = []
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding1 = nn.Embedding(len(en_vocab), embedding_dim)
        self.dense1=nn.Linear(in_features=16,out_features=64)
        self.dense2=nn.Linear(in_features=64,out_features=16)
        self.wt=nn.Linear(in_features=16,out_features=8)
        self.ln=nn.LayerNorm(8)
    
    def forward(self,x):
        sentence_embedding = []
        for it in x:
            token_ids = preprocess_sentence(it, en_tokenizer, en_vocab)
            gett = self.embedding1(token_ids.to(DEVICE))
            gett = gett.mean(dim=1)
            sentence_embedding.append(gett.to(DEVICE))
        x=torch.stack(sentence_embedding)
        x = x.squeeze(1)
        #print(x.size())
        x=F.relu(self.dense1(x))
        x=F.relu(self.dense2(x))
        x=self.wt(x)
        x=self.ln(x)
        return x

if __name__=='__main__':
    text_encoder=TextEncoder()
    #x='the number is one'
    #y=text_encoder(x)
    #print(y)
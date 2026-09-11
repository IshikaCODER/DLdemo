import os, numpy as np
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T, torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import add_self_loops, softmax as pyg_softmax
from skimage.segmentation import slic
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

# --- CBAM ---
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch//r, 8)
        self.mlp = nn.Sequential(nn.Linear(ch,mid,bias=False),nn.ReLU(True),nn.Linear(mid,ch,bias=False))
    def forward(self, x):
        B,C,_,_ = x.shape
        a = self.mlp(x.mean([2,3])); m = self.mlp(x.amax([2,3]))
        return x * torch.sigmoid(a+m).view(B,C,1,1)

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,k,padding=k//2,bias=False)
    def forward(self, x):
        d = torch.cat([x.mean(1,keepdim=True), x.amax(1,keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(d))

class CBAM(nn.Module):
    def __init__(self, ch, r=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(ch, r); self.sa = SpatialAttention(k)
    def forward(self, x): return self.sa(self.ca(x))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MEAN_NP = np.array(IMAGENET_MEAN)
STD_NP = np.array(IMAGENET_STD)

# --- Superpixel Graph Construction ---
def _unnorm(t):
    img=t.detach().cpu().numpy().transpose(1,2,0)
    return np.clip(img*STD_NP+MEAN_NP,0,1)

def build_superpixel_graph(img_t, feat_map, label, n_seg=50, compact=10):
    C,H,W = feat_map.shape
    img_np = _unnorm(img_t)
    segs = slic(img_np, n_segments=n_seg, compactness=compact, start_label=0, channel_axis=2)
    sh,sw = segs.shape[0]/H, segs.shape[1]/W
    sd = ndimage.zoom(segs.astype(float),(1/sh,1/sw),order=0).astype(int)[:H,:W]
    feat = feat_map.detach().cpu()
    uids = np.unique(sd); id2i = {s:i for i,s in enumerate(uids)}; N=len(uids)
    nf = torch.zeros(N, C)
    for s in uids:
        mask = torch.from_numpy((sd==s).astype(np.float32))
        cnt = mask.sum().clamp(min=1)
        nf[id2i[s]] = (feat*mask.unsqueeze(0)).sum([1,2])/cnt
    src,dst=[],[]
    for r in range(H):
        for c in range(W):
            cur=sd[r,c]
            for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<H and 0<=nc<W:
                    nb=sd[nr,nc]
                    if cur!=nb:
                        i,j=id2i[cur],id2i[nb]; src+=[i,j]; dst+=[j,i]
    if not src:
        for i in range(N):
            for j in range(i+1,N): src+=[i,j]; dst+=[j,i]
    es=set(zip(src,dst))
    if es: s,d=zip(*es)
    else: s,d=[],[]
    ei=torch.tensor([list(s),list(d)],dtype=torch.long)
    return Data(x=nf,edge_index=ei,y=torch.tensor([label],dtype=torch.long))

def build_multiscale_graphs(img_t, feat_map, label):
    fine=build_superpixel_graph(img_t,feat_map,label,n_seg=50,compact=10)
    coarse=build_superpixel_graph(img_t,feat_map,label,n_seg=15,compact=30)
    return fine, coarse


# --- ResNet50 + CBAM Backbone (Part 3) ---
class ResNet50CBAMBackbone(nn.Module):
    """ResNet50 with CBAM after each major block. [B,3,224,224]->[B,2048,7,7]"""
    def __init__(self, pretrained=True, freeze=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool)
        self.layer1=resnet.layer1; self.cbam1=CBAM(256)
        self.layer2=resnet.layer2; self.cbam2=CBAM(512)
        self.layer3=resnet.layer3; self.cbam3=CBAM(1024)
        self.layer4=resnet.layer4; self.cbam4=CBAM(2048)
        if freeze:
            for p in self.parameters(): p.requires_grad=False
    def forward(self, x):
        x=self.stem(x)
        x=self.cbam1(self.layer1(x))
        x=self.cbam2(self.layer2(x))
        x=self.cbam3(self.layer3(x))
        x=self.cbam4(self.layer4(x))
        return x

# --- Class-Aware GAT ---
class ClassAwareGATConv(MessagePassing):
    def __init__(self, in_ch, out_ch, n_cls=4, heads=4, drop=0.3, concat=True, ced=32):
        super().__init__(aggr='add', node_dim=0)
        self.out_channels=out_ch; self.heads=heads; self.concat=concat; self.dropout=drop
        self.lin=nn.Linear(in_ch, heads*out_ch, bias=False)
        self.class_embed=nn.Embedding(n_cls, ced)
        self.class_proj=nn.Linear(ced, heads, bias=False)
        self.att_src=nn.Parameter(torch.empty(1,heads,out_ch))
        self.att_dst=nn.Parameter(torch.empty(1,heads,out_ch))
        self.bias=nn.Parameter(torch.zeros(heads*out_ch if concat else out_ch))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src); nn.init.xavier_uniform_(self.att_dst)
    def forward(self, x, edge_index, class_logits=None):
        H,C=self.heads,self.out_channels
        xp=self.lin(x).view(-1,H,C)
        as_=( xp*self.att_src).sum(-1); ad=(xp*self.att_dst).sum(-1)
        if class_logits is not None and class_logits.dim()==2:
            pr=F.softmax(class_logits.detach(),-1)
            ce=pr@self.class_embed.weight
            self._cb=self.class_proj(ce).mean(0)
        else: self._cb=torch.zeros(H,device=x.device)
        ei,_=add_self_loops(edge_index,num_nodes=x.size(0))
        out=self.propagate(ei,x=xp,alpha=(as_,ad))
        if self.concat: return out.view(-1,H*C)+self.bias
        return out.mean(1)+self.bias
    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        a=alpha_i+alpha_j+self._cb.unsqueeze(0)
        a=F.leaky_relu(a,0.2); a=pyg_softmax(a,index,ptr,size_i)
        a=F.dropout(a,p=self.dropout,training=self.training)
        return x_j*a.unsqueeze(-1)

# --- Dual GAT + Fusion ---
class DualGATClassifier(nn.Module):
    def __init__(self, ind=2048, hid=256, nc=4, h=4, d=0.3):
        super().__init__(); self.drop=d
        self.fg1=ClassAwareGATConv(ind,hid,nc,h,d,True); self.fb1=nn.BatchNorm1d(hid*h)
        self.fg2=ClassAwareGATConv(hid*h,hid,nc,1,d,False); self.fb2=nn.BatchNorm1d(hid)
        self.cg1=ClassAwareGATConv(ind,hid,nc,h,d,True); self.cb1=nn.BatchNorm1d(hid*h)
        self.cg2=ClassAwareGATConv(hid*h,hid,nc,1,d,False); self.cb2=nn.BatchNorm1d(hid)
        self.gate=nn.Sequential(nn.Linear(hid*2,hid),nn.Sigmoid())
        self.fc1=nn.Linear(hid,64); self.fc2=nn.Linear(64,32); self.fc3=nn.Linear(32,nc)
        self.bn1=nn.BatchNorm1d(64); self.bn2=nn.BatchNorm1d(32)
    def _pipe(self,x,ei,b,g1,b1,g2,b2,cl=None):
        x=F.elu(b1(g1(x,ei,cl))); x=F.dropout(x,self.drop,self.training)
        x=F.elu(b2(g2(x,ei,cl))); return global_mean_pool(x,b)
    def forward(self,fx,fei,fb,cx,cei,cb,cl=None):
        f=self._pipe(fx,fei,fb,self.fg1,self.fb1,self.fg2,self.fb2,cl)
        c=self._pipe(cx,cei,cb,self.cg1,self.cb1,self.cg2,self.cb2,cl)
        cat=torch.cat([f,c],-1); g=self.gate(cat); fused=g*f+(1-g)*c
        x=F.dropout(F.elu(self.bn1(self.fc1(fused))),self.drop,self.training)
        x=F.dropout(F.elu(self.bn2(self.fc2(x))),self.drop,self.training)
        return F.log_softmax(self.fc3(x),-1)


# --- Full Model ---
class EnhancedResNet50GAT(nn.Module):
    def __init__(self, pretrained=True, freeze=False, num_classes=4):
        super().__init__()
        self.backbone = ResNet50CBAMBackbone(pretrained, freeze)
        self.gat = DualGATClassifier()
    def forward(self, images, labels=None):
        dev=images.device; B=images.size(0)
        feat_maps=self.backbone(images)
        fine_list, coarse_list = [], []
        for i in range(B):
            lbl = labels[i].item() if labels is not None else 0
            fg, cg = build_multiscale_graphs(images[i], feat_maps[i], lbl)
            fine_list.append(fg); coarse_list.append(cg)
        fb=Batch.from_data_list(fine_list).to(dev)
        cb=Batch.from_data_list(coarse_list).to(dev)
        return self.gat(fb.x,fb.edge_index,fb.batch,cb.x,cb.edge_index,cb.batch)
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

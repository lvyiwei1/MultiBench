from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import time
from utils.AUPRC import AUPRC
#from objective_functions.regularization import RegularizationLoss
#import pdb

softmax = nn.Softmax()

class MMDL(nn.Module):
    def __init__(self,encoders,fusion,head1,head2,has_padding=False):
        super(MMDL,self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head1 = head1
        self.head2 = head2
        self.has_padding=has_padding
    
    def forward(self,inputs,training=False):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i]([inputs[0][i],inputs[1][i]], training=training))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i], training=training))
        out = self.fuse(outs, training=training)
        #print(out) 
        return self.head1(out, training=training), self.head2(out, training=training)


def train(
    encoders,fusion,head1,head2,train_dataloader,valid_dataloader,total_epochs,is_packed=False,
    early_stop=False,task="classification",optimtype=torch.optim.SGD,lr=0.0001,weight_decay=0.0,
    criterion=nn.CrossEntropyLoss(),regularization=False,auprc=False,save='best.pt',validtime=False):
    
    model = MMDL(encoders,fusion,head1,head2,is_packed).cuda()
    op = optimtype([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=weight_decay)
    #scheduler = ExponentialLR(op, 0.9)
    bestvalloss = 10000
    bestacc = 0
    bestf1 = 0
    patience = 0
    
    if regularization:
        regularize = RegularizationLoss(criterion, model, 1e-10, is_packed)
    
    for epoch in range(total_epochs):
        totalloss = 0.0
        totalloss1 = 0.0
        totalloss2 = 0.0
        totals = 0
        model.train()
        for j in train_dataloader:
            #print([i for i in j[:-1]])
            op.zero_grad()
            if is_packed:
                with torch.backends.cudnn.flags(enabled=False):
                    out=model([[i.cuda() for i in j[0]], j[1]],training=True)
                    #print(j[-1])
                    #print(out)
                    loss1=criterion(out,j[-1].cuda())
                    loss2=regularize(out, [[i.cuda() for i in j[0]], j[1]]) if regularization else 0
                    loss = loss1+loss2
            else:
                out1,out2=model([i.float().cuda() for i in j[:-2]],training=True)
                #print(out, j[-1])
                loss1=criterion(out1, j[-2].long().cuda())
                loss2=criterion(out2, j[-1].long().cuda())
                #loss = loss1+loss2
                loss=loss2
            #print(loss)
            totalloss += loss * len(j[-1])
            totals+=len(j[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            op.step()
        if regularization:
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss1/totals)+" reg loss: "+str(totalloss2/totals))
        else:
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
        validstarttime=time.time()
        if validtime:
            print("train total: "+str(totals))
        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            pred1 = []
            true1 = []
            pred2 = []
            true2 = []
            pts = []
            for j in valid_dataloader:
                if is_packed:
                    out=model([[i.cuda() for i in j[0]], j[1]],training=False)
                else:
                    out1,out2 = model([i.float().cuda() for i in j[:-2]],training=False)
                loss1=criterion(out1, j[-2].long().cuda())
                loss2=criterion(out2, j[-1].long().cuda())
                totalloss += loss*len(j[-1])
                #print(totalloss)
                if task == "classification":
                    pred1.append(torch.argmax(out1, 1))
                    pred2.append(torch.argmax(out2, 1))
                true1.append(j[-2])
                true2.append(j[-1])
                if auprc:
                    #pdb.set_trace()
                    sm=softmax(out)
                    pts += [(sm[i][1].item(), j[-1][i].item()) for i in range(j[-1].size(0))]
        pred1 = torch.cat(pred1, 0).cpu().numpy()
        pred2 = torch.cat(pred2, 0).cpu().numpy()
        true1 = torch.cat(true1, 0).cpu().numpy()
        true2 = torch.cat(true2, 0).cpu().numpy()
        totals = true1.shape[0]
        valloss=totalloss/totals
        if task == "classification":
            acc1 = accuracy_score(true1, pred1)
            acc2 = accuracy_score(true2, pred2)
            
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+\
                " acc1: "+str(acc1)+" acc2: "+str(acc2))
            if acc1+acc2 > bestacc:
                patience = 0
                bestacc = acc1+acc2
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
        elif task == "multilabel":
            f1_micro = f1_score(true, pred, average="micro")
            f1_macro = f1_score(true, pred, average="macro")
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+\
                " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
            if f1_macro>bestf1:
                patience = 0
                bestf1=f1_macro
                print("Saving Best")
                torch.save(model,save)
            else:
                patience += 1
        elif task == "regression":
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
            if valloss<bestvalloss:
                patience = 0
                bestvalloss=valloss
                print("Saving Best")
                torch.save(model,save)
            else:
                patience += 1
        if early_stop and patience > 7:
            break
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        validendtime=time.time()
        if validtime:
            print("valid time:  "+str(validendtime-validstarttime))
            print("Valid total: "+str(totals))

        #scheduler.step()


def test(
    model,test_dataloader,is_packed=False,
    criterion=nn.CrossEntropyLoss(),task="classification",auprc=False):
    with torch.no_grad():
        totalloss = 0.0
        pred1=[]
        pred2=[]
        true1=[]
        true2=[]
        pts=[]
        for j in test_dataloader:
            if is_packed:
                out=model([[i.cuda() for i in j[0]], j[1]],training=False)
            else:
                out1,out2 = model([i.float().cuda() for i in j[:-2]],training=False)
            loss1=criterion(out1, j[-2].cuda())
            loss2=criterion(out1, j[-1].cuda())
            #print(torch.cat([out,j[-1].cuda()],dim=1))
            #totalloss += loss*len(j[-1])
            if task == "classification":
                pred1.append(torch.argmax(out1, 1))
                pred2.append(torch.argmax(out2, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            true1.append(j[-2])
            true2.append(j[-1])
            if auprc:
                #pdb.set_trace()
                sm=softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item()) for i in range(j[-1].size(0))]
        pred1 = torch.cat(pred1, 0).cpu().numpy()
        pred2 = torch.cat(pred2, 0).cpu().numpy()
        true1 = torch.cat(true1, 0).cpu().numpy()
        true2 = torch.cat(true2, 0).cpu().numpy()
        totals = true1.shape[0]
        #testloss=totalloss/totals
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            print("acc1: "+str(accuracy_score(true1, pred1)))
            print("acc2: "+str(accuracy_score(true2, pred2)))
            return accuracy_score(true1, pred1)
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro"))+\
                " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return f1_score(true, pred, average="micro"), f1_score(true, pred, average="macro")
        elif task == "regression":
            print("mse: "+str(testloss))
            return testloss.item()
        

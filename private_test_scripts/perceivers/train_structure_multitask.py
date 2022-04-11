import torch
from eval_scripts.performance import f1_score

def train(model,epochs,trains,valid,test,modalities,savedir,lr=0.001,weight_decay=0.0, optimizer=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss(),unsqueezing=[True,True], device="cuda:0",train_weights=[1.0,1.0],is_affect=[False,False],transpose=[False,False],ismmimdb=False,evalweights=None):
    #for param in model.parameters():
    #    print(param)
    optim = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    bestacc=0.0
    for ep in range(epochs):
        totalloss=[]
        totals=[]
        fulltrains=[]
        for i in range(len(trains)):
            count=0
            totalloss.append(0.0)
            totals.append(0)
            for j in trains[i]:
                #print('iter')
                if count >= len(fulltrains):
                    fulltrains.append({})
                if is_affect[i]:
                    jj=j[0]
                    if isinstance(criterion,torch.nn.CrossEntropyLoss):
                        jj.append((j[3].squeeze(1)>=0).long())
                    else:
                        jj.append(j[3])
                    fulltrains[count][str(i)]=jj
                #elif ismmimdb:
                #    jj=[j[0].transpose(1,2),j[1],j[2]]
                #    fulltrains[count][str(i)]=jj
                else:
                    #print("iter")
                    fulltrains[count][str(i)]=j
                count += 1
        fulltrains.reverse()
        for js in fulltrains:
            optim.zero_grad()
            losses=0.0
            for ii in js:
                model.to_logits=model.to_logitslist[int(ii)]
                indict={}
                for i in range(len(modalities[int(ii)])):
                    if unsqueezing[int(ii)]:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().unsqueeze(-1).to(device)
                    elif transpose[int(ii)]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().to(device)
                for mod in indict:
                    indict[mod].requires_grad=True
                #print(indict['colorlessimage'].size())
                #if 'audiospec' in indict:
                    #print(indict['colorlessimage'].size())
                    #print(indict['audiospec'].size())
                out=model(indict)
                if ismmimdb:
                    loss=criterion(out,js[ii][-1].float().to(device))
                else:
                    loss=criterion(out,js[ii][-1].long().to(device))
                losses += loss*train_weights[int(ii)]
                total=len(js[ii][0])
                totals[int(ii)] += total
                totalloss[int(ii)] += loss.item()*total
            losses.backward()
            optim.step()
            #print("We're at "+str(totals))
        for ii in range(len(trains)):
            print("epoch "+str(ep)+" train loss dataset " +str(ii)+": "+str(totalloss[ii]/totals[ii]))
        with torch.no_grad():
            accs=0.0
            for ii in range(len(valid)):
                totalloss=0.0
                totals=0
                corrects=0
                trues=[]
                preds=[]
                for jj in valid[ii]:
                    j=jj
                    if is_affect[ii]:
                        j=jj[0]
                        if isinstance(criterion,torch.nn.CrossEntropyLoss):
                            j.append((jj[3].squeeze(1)>=0).long())
                        else:
                            j.append(jj[3])
                    #if ismmimdb:
                    #    j[0]=j[0].transpose(1,2)
                    model.to_logits=model.to_logitslist[ii]
                    indict={}
                    for i in range(len(modalities[ii])):
                        if unsqueezing[ii]:
                            indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                        elif transpose[ii]:
                            indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                        else:
                            indict[modalities[ii][i]]=j[i].float().to(device)
                    out=model(indict)
                    if ismmimdb:
                        loss=criterion(out,j[-1].float().to(device))
                    else:
                        loss=criterion(out,j[-1].long().to(device))
                    totalloss += loss.item()*len(j[0])
                    if ismmimdb:
                        trues.append(j[-1])
                        preds.append(torch.sigmoid(out).round())
                        totals += len(j[-1])
                    else:
                        #print(preds)
                        for i in range(len(out)):
                            #print("we are here")
                            if isinstance(criterion,torch.nn.CrossEntropyLoss):
                                preds=torch.argmax(out,dim=1)
                                if preds[i].item()==j[-1].long()[i].item():
                                    corrects += 1
                            else:
                                if (out[i].item() >= 0) == (j[-1].long()[i].item() >= 0):
                                    corrects += 1
                            totals += 1
                if ismmimdb:
                    true=torch.cat(trues,0)
                    pred=torch.cat(preds,0)
                    f1_micro = f1_score(true, pred, average="micro")
                    f1_macro = f1_score(true, pred, average="macro")
                    accs = f1_macro
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
                else:
                    acc=float(corrects)/totals
                    if evalweights is None:
                        accs += acc
                    else:
                        accs += acc*evalweights[ii]
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" acc: "+str(acc))
            if accs > bestacc:
                print("save best")
                bestacc=accs
                torch.save(model,savedir)
    model=torch.load(savedir).to(device)
    with torch.no_grad():
        for ii in range(len(test)):
            model.to_logits=model.to_logitslist[ii]
            totals=0
            corrects=0
            trues=[]
            preds=[]
            for jj in test[ii]:            
                j=jj
                if is_affect[ii]:
                    j=jj[0]
                    j.append((jj[3].squeeze(1) >= 0).long())

                #if ismmimdb:
                #    j[0]=j[0].transpose(1,2)
                indict={}
                for i in range(0,len(modalities[ii])):
                    if unsqueezing[ii]:
                        indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                    elif transpose[ii]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[ii][i]]=j[i].float().to(device)
                out=model(indict)

                if ismmimdb:
                    trues.append(j[-1])
                    preds.append(torch.sigmoid(out).round())
                else:
                    for i in range(len(out)):
                        if isinstance(criterion,torch.nn.CrossEntropyLoss):
                            preds=torch.argmax(out,dim=1)
                            if preds[i].item()==j[-1].long()[i].item():
                                corrects += 1
                        else:
                            print(out[i].item(), j[-1][i].item())
                            if (out[i].item() >= 0) == j[-1].long()[i].item():
                                corrects += 1
                        totals += 1

            if ismmimdb:
                true=torch.cat(trues,0)
                pred=torch.cat(preds,0)
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                #accs = f1_macro
                print("test f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
            else:
                acc=float(corrects)/totals
                print("test acc dataset "+str(ii)+": "+str(ii)+" "+str(acc))










import os, sys, json, re
import numpy as np, torch
from scipy.stats import norm, pearsonr, spearmanr
from scipy.optimize import curve_fit
ROOT='/Users/chrischalkias/Projects/critiPrune'; sys.path.insert(0,ROOT); os.chdir(ROOT)
A_RAND=0.1

@torch.no_grad()
def f41_A(Ws,bs,Wout,bout,X,Y,s):
    mu=X; q=X**2
    for W,b in zip(Ws,bs):
        Wsq=W**2
        pm=s*(mu@W.t())+b
        pv=((s*q-s*s*mu*mu)@Wsq.t()).clamp(min=1e-12)
        ps=torch.sqrt(pv); rho=pm/ps; rn=rho.numpy()
        phi=torch.from_numpy(norm.pdf(rn).astype(np.float32))
        Phi=torch.from_numpy(norm.cdf(rn).astype(np.float32))
        mu=ps*phi+pm*Phi; q=(pm*pm+pv)*Phi+pm*ps*phi
    v=(q-mu*mu).clamp(min=0.0)
    zb=mu@Wout.t()+bout; zy=zb.gather(1,Y.unsqueeze(1)); M=zy-zb
    dW=Wout[Y].unsqueeze(1)-Wout.unsqueeze(0)
    Sig=(dW*dW*v.unsqueeze(1)).sum(-1).clamp(min=1e-30)
    r=M/torch.sqrt(Sig); C=Wout.shape[0]
    my=(torch.arange(C).unsqueeze(0)==Y.unsqueeze(1))
    r=torch.where(my,torch.full_like(r,float('inf')),r)
    return float(norm.cdf(r.numpy()).prod(1).mean())

def predict_s0(sd,X,Y):
    Lw=sorted([k for k in sd if k.endswith('.weight')],key=lambda k:int(re.findall(r'layers\.(\d+)\.',k)[0]))
    Ws=[sd[k].float() for k in Lw[:-1]]; bs=[sd[k.replace('weight','bias')].float() for k in Lw[:-1]]
    Wout=sd[Lw[-1]].float(); bout=sd[Lw[-1].replace('weight','bias')].float()
    sg=np.concatenate([np.geomspace(1e-3,0.95,40),[1.0]])
    A=np.array([f41_A(Ws,bs,Wout,bout,X,Y,float(s)) for s in sg])
    Ai=A[-1]; mid=(A_RAND+Ai)/2
    ab=A>=mid
    if not ab.any() or Ai<A_RAND+0.05: return np.nan,Ai
    i=int(np.argmax(ab))
    if i==0: return float(sg[0]),Ai
    return float(np.interp(mid,[A[i-1],A[i]],[sg[i-1],sg[i]])),Ai

def sig(s,Ai,s0,b): return A_RAND+(Ai-A_RAND)/(1+np.exp(-b*(s-s0)))
def emp_s0(e):
    s=np.array(e['densities']); a=np.array(e['accs_mean'])
    try: p,_=curve_fit(sig,s,a,p0=[a[-1],0.3,10],bounds=([0.2,0,1e-3],[1,1,50]),maxfev=20000)
    except Exception: return np.nan,np.nan
    res=a-sig(s,*p); ss=((a-a.mean())**2).sum(); r2=1-(res**2).sum()/ss if ss>0 else np.nan
    n=len(s); return float(p[1]), (1-(1-r2)*(n-1)/(n-3) if n>3 else r2)

def run(label, X_te, y_te, ckdir, jsonp, n_test):
    rng=np.random.default_rng(0); n=min(n_test,X_te.shape[0])
    idx=rng.choice(X_te.shape[0],n,replace=False)
    Xt=torch.tensor(X_te[idx],dtype=torch.float32); Yt=torch.tensor(np.asarray(y_te)[idx],dtype=torch.long)
    entries={(e['H'],e['L']):e for e in json.load(open(jsonp))}
    cells=sorted({tuple(map(int,re.findall(r'H(\d+)_L(\d+)_r0',f)[0])) for f in os.listdir(ckdir) if f.endswith('_r0.pt')})
    rows=[]
    for (H,L) in cells:
        if (H,L) not in entries: continue
        sd=torch.load(os.path.join(ckdir,f'H{H}_L{L}_r0.pt'),map_location='cpu',weights_only=False)['state_dict']
        sp,Ai=predict_s0(sd,Xt,Yt); se,r2=emp_s0(entries[(H,L)])
        rows.append([H,L,sp,se,r2,Ai])
    json.dump(rows,open(f'/tmp/heldout_{label}.json','w'))
    print(f'[{label}] cells={len(rows)} testpts={n}')
    return rows

# sklearn (instant)
from pruning.mnist_scaling import load_data
_,_,Xs,_,_,ys=load_data()
run('sklearn', Xs, ys, 'unstructured_pruning/checkpoints/unstructured_figures_sklearn_random',
    'unstructured_pruning/figures/unstructured_figures_sklearn_random/scaling_results.json', 3000)
# cifar resnet (download+extract features, cached)
from pruning.cifar_scaling import load_cifar10
_,_,Xc,_,_,yc=load_cifar10()
run('cifar_resnet', Xc, yc, 'unstructured_pruning/checkpoints/unstructured_figures_cifar_resnet_random',
    'unstructured_pruning/figures/unstructured_figures_cifar_resnet_random/scaling_results.json', 1500)
print('DONE')

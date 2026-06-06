"""Held-out prediction of the pruning critical density s0 from trained
weights alone, via the framework's F41 moment-propagation (no pruning sweep
used for the prediction). Compares to the empirically-fit s0 from the
random-pruning density sweep on MNIST-28."""
import os, sys, json, glob, re
import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import curve_fit

ROOT='/Users/chrischalkias/Projects/critiPrune'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from pruning.mnist28_scaling import load_mnist28

A_RAND = 0.1
CKDIR = 'unstructured_pruning/checkpoints/unstructured_figures_mnist28_random'
JSON  = 'unstructured_pruning/figures/unstructured_figures_mnist28_random/scaling_results.json'

# ---- data (same preprocessing as training) ----
X_tr, X_val, X_te, y_tr, y_val, y_te = load_mnist28()
# cap test pts for speed
rng = np.random.default_rng(0)
n = min(2500, X_te.shape[0])
idx = rng.choice(X_te.shape[0], n, replace=False)
Xt = torch.tensor(X_te[idx], dtype=torch.float32)
Yt = torch.tensor(y_te[idx], dtype=torch.long)
print('test pts', Xt.shape, 'classes', int(Yt.max())+1)

@torch.no_grad()
def f41_A(Ws, bs, Wout, bout, X, Y, s):
    mu = X; q = X**2
    for W, b in zip(Ws, bs):
        Wsq = W**2
        pm = s*(mu @ W.t()) + b
        pv = ((s*q - s*s*mu*mu) @ Wsq.t()).clamp(min=1e-12)
        ps = torch.sqrt(pv); rho = pm/ps
        rn = rho.numpy()
        phi = torch.from_numpy(norm.pdf(rn).astype(np.float32))
        Phi = torch.from_numpy(norm.cdf(rn).astype(np.float32))
        mu = ps*phi + pm*Phi
        q  = (pm*pm+pv)*Phi + pm*ps*phi
    v = (q - mu*mu).clamp(min=0.0)
    z_bar = mu @ Wout.t() + bout
    z_y = z_bar.gather(1, Y.unsqueeze(1))
    M = z_y - z_bar
    Wy = Wout[Y]
    dW = Wy.unsqueeze(1) - Wout.unsqueeze(0)
    Sig = (dW*dW*v.unsqueeze(1)).sum(-1).clamp(min=1e-30)
    r = M/torch.sqrt(Sig)
    C = Wout.shape[0]
    my = (torch.arange(C).unsqueeze(0)==Y.unsqueeze(1))
    r = torch.where(my, torch.full_like(r, float('inf')), r)
    A = norm.cdf(r.numpy()).prod(1)
    return float(A.mean())

def predict_s0(sd, A_rand=A_RAND):
    keys = [k for k in sd if k.endswith('.weight')]
    Lw = sorted(keys, key=lambda k:int(re.findall(r'layers\.(\d+)\.',k)[0]))
    Ws=[sd[k].float() for k in Lw[:-1]]; 
    bs=[sd[k.replace('weight','bias')].float() for k in Lw[:-1]]
    Wout=sd[Lw[-1]].float(); bout=sd[Lw[-1].replace('weight','bias')].float()
    s_grid = np.concatenate([np.geomspace(1e-3,0.95,40),[1.0]])
    A = np.array([f41_A(Ws,bs,Wout,bout,Xt,Yt,float(s)) for s in s_grid])
    A_inf = A[-1]
    mid = (A_rand + A_inf)/2.0
    # first crossing
    above = A >= mid
    if not above.any() or A_inf < A_rand+0.05:
        return np.nan, A_inf
    i = np.argmax(above)
    if i==0: return float(s_grid[0]), A_inf
    s0 = np.interp(mid, [A[i-1],A[i]], [s_grid[i-1],s_grid[i]])
    return float(s0), A_inf

def sigmoid(s, A_inf, s0, beta):
    return A_RAND + (A_inf-A_RAND)/(1+np.exp(-beta*(s-s0)))

def empirical_s0(entry):
    s=np.array(entry['densities']); a=np.array(entry['accs_mean'])
    try:
        p,_=curve_fit(sigmoid, s, a, p0=[a[-1],0.3,10],
                      bounds=([0.2,0.0,1e-3],[1.0,1.0,50]), maxfev=20000)
    except Exception:
        return np.nan, np.nan
    res=a-sigmoid(s,*p); ss=((a-a.mean())**2).sum()
    r2=1-(res**2).sum()/ss if ss>0 else np.nan
    n=len(s); r2adj=1-(1-r2)*(n-1)/(n-3) if n>3 else r2
    return float(p[1]), float(r2adj)

entries={(e['H'],e['L']):e for e in json.load(open(JSON))}
cells=sorted({tuple(map(int,re.findall(r'H(\d+)_L(\d+)_r0',f)[0]))
              for f in os.listdir(CKDIR) if f.endswith('_r0.pt')})
print('cells with r0 ckpt:', len(cells))

rows=[]
for (H,L) in cells:
    if (H,L) not in entries: continue
    sd=torch.load(os.path.join(CKDIR,f'H{H}_L{L}_r0.pt'),map_location='cpu',weights_only=False)['state_dict']
    s0_pred,Ainf=predict_s0(sd)
    s0_emp,r2=empirical_s0(entries[(H,L)])
    rows.append((H,L,s0_pred,s0_emp,r2,Ainf))
    print(f'H{H:4d} L{L:2d}  pred={s0_pred:.3f}  emp={s0_emp:.3f}  r2adj={r2:.3f}  Ainf_pred={Ainf:.3f}')

json.dump(rows, open('/tmp/heldout_rows.json','w'))
print('SAVED', len(rows))

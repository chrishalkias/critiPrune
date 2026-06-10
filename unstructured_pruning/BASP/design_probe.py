#!/usr/bin/env python3
"""Design probe: which (score x allocation x iteration) minimises s_0?

Throwaway experiment harness used to derive BASP empirically rather than by
literature prior. Not part of the package API.
"""
from __future__ import annotations
import copy, os, sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unstructured_pruning.base.mnist_scaling import load_data
from unstructured_pruning.base.pruning import fit_sigmoid
from unstructured_pruning.core import DEFAULT_DENSITIES, evaluate_masked_accuracy, load_fc_checkpoint

CKPT_DIR = 'checkpoints/sklearn_magnitude'
CELLS = ['H72_L4_r0', 'H72_L6_r0', 'H72_L8_r0']


def layer_acts(model, X):
    p = next(model.parameters())
    Xt = torch.as_tensor(np.asarray(X), dtype=p.dtype, device=p.device)
    acts = [Xt]; h = Xt
    model.eval()
    with torch.no_grad():
        for layer in model.layers[:-1]:
            h = torch.relu(layer(h)); acts.append(h)
    return acts


def score_tensors(model, X, kind):
    """Return list of per-hidden-layer score tensors [fan_out, fan_in]."""
    acts = layer_acts(model, X)
    out = []
    for l, layer in enumerate(model.layers[:-1]):
        W = layer.weight.detach().abs()
        xin = acts[l].norm(p=2, dim=0)         # [fan_in]
        aout = acts[l + 1].norm(p=2, dim=0)    # [fan_out]
        if kind == 'mag':
            sc = W
        elif kind == 'wanda':               # |W| * ||x||  (input drive)
            sc = W * xin.unsqueeze(0)
        elif kind == 'bidir':               # |W| * ||x|| * ||a||
            sc = W * xin.unsqueeze(0) * aout.unsqueeze(1)
        else:
            raise ValueError(kind)
        out.append(sc)
    return out


def mask_per_row(scores, s):
    masks = []
    for sc in scores:
        fan_in = sc.shape[1]; k = max(1, int(round(s * fan_in)))
        if k >= fan_in:
            masks.append(torch.ones_like(sc)); continue
        _, idx = sc.topk(k, dim=1)
        m = torch.zeros_like(sc); m.scatter_(1, idx, 1.0)
        masks.append(m)
    return [m.float().cpu() for m in masks]


def mask_per_layer(scores, s):
    masks = []
    for sc in scores:
        flat = sc.flatten(); N = flat.numel(); k = max(1, int(round(s * N)))
        if k >= N:
            masks.append(torch.ones_like(sc)); continue
        thr = torch.kthvalue(flat, N - k + 1).values
        masks.append((sc >= thr).float())
    return [m.cpu() for m in masks]


def build(model, X, s, kind, alloc, iters=1):
    if iters <= 1:
        sc = score_tensors(model, X, kind)
        return mask_per_row(sc, s) if alloc == 'row' else mask_per_layer(sc, s)
    sched = np.geomspace(1.0, max(s, 1e-4), iters + 1)[1:]
    masks = [torch.ones_like(layer.weight.detach()).cpu() for layer in model.layers[:-1]]
    for st in sched:
        pr = copy.deepcopy(model)
        with torch.no_grad():
            for layer, m in zip(pr.layers[:-1], masks):
                layer.weight.data.mul_(m.to(layer.weight.dtype))
        sc = [s_.cpu() * m for s_, m in zip(score_tensors(pr, X, kind), masks)]
        masks = mask_per_row(sc, float(st)) if alloc == 'row' else mask_per_layer(sc, float(st))
    return masks


VARIANTS = [
    ('mag/layer',       'mag',   'layer', 1),
    ('wanda/row',       'wanda', 'row',   1),
    ('wanda/layer',     'wanda', 'layer', 1),
    ('bidir/row',       'bidir', 'row',   1),
    ('bidir/layer',     'bidir', 'layer', 1),
    ('wanda/row/it5',   'wanda', 'row',   5),
    ('bidir/layer/it5', 'bidir', 'layer', 5),
    ('wanda/layer/it5', 'wanda', 'layer', 5),
]


def main():
    X_tr, _, X_te, y_tr, _, y_te = load_data()
    Xc = X_tr[:512]
    dens = sorted(set(DEFAULT_DENSITIES) | {0.02, 0.03})
    agg = {}
    for cell in CELLS:
        p = os.path.join(CKPT_DIR, f'{cell}.pt')
        model, ck = load_fc_checkpoint(p)
        print(f"\n=== {cell}  L={ck['arch']['num_hidden_layers']} ===")
        for name, kind, alloc, it in VARIANTS:
            mset = {s: [build(model, Xc, s, kind, alloc, it)] for s in dens}
            accs, normal = evaluate_masked_accuracy(model, X_te, y_te, mset)
            sv = sorted(accs); mean = {s: accs[s][0] for s in sv}
            popt, _, r2 = fit_sigmoid(sv, mean, normal)
            s0 = popt[2] if popt is not None else float('nan')
            agg.setdefault(name, []).append(s0)
            print(f"  {name:<18} s0={s0:.3f}  R2={r2:.3f}")
    print("\n=== mean s_0 across cells (lower=better) ===")
    for name in sorted(agg, key=lambda n: np.mean(agg[n])):
        print(f"  {name:<18} {np.mean(agg[name]):.3f}")


if __name__ == '__main__':
    main()

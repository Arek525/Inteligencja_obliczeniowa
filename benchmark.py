#!/usr/bin/env python3
import argparse, os, pandas as pd
from sklearn.model_selection import train_test_split
from data_utils import load_paths_labels
from models import tl_svm, custom_cnn, hist_baselines

# ── parser ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["tl","cnn","knn","nb"],
                    choices=["tl","cnn","knn","nb"])
parser.add_argument("--data", default="dataset-resized", help="Ścieżka do katalogu z podfolderami klas")
parser.add_argument("--out",  default="results")
cfg = parser.parse_args(); os.makedirs(cfg.out,exist_ok=True)

# ── dane ──────────────────────────────────────────────────────────────────
paths, labels = load_paths_labels(cfg.data)
p_tr, p_te, y_tr, y_te = train_test_split(paths, labels, test_size=0.2,
                                          stratify=labels, random_state=42)
CLASSES = sorted(set(labels))

rows=[]

# ── TL+SVM ────────────────────────────────────────────────────────────────
if "tl" in cfg.models:
    rpt, t = tl_svm.run(p_tr, p_te, y_tr, y_te, CLASSES, cfg.out)
    rows.append(dict(model="TL+SVM", accuracy=rpt["accuracy"],
                     macro_F1= sum(rpt[c]["f1-score"] for c in CLASSES)/len(CLASSES),
                     train_sec=t))

# ── CNN ───────────────────────────────────────────────────────────────────
if "cnn" in cfg.models:
    import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
    model, hist, t = custom_cnn.train(cfg.data, CLASSES)
    # learning curve
    plt.figure(); plt.plot(hist['accuracy']); plt.plot(hist['val_accuracy'])
    plt.legend(['train','val']); plt.tight_layout()
    plt.savefig(f"{cfg.out}/cnn_learning_curve.png"); plt.close()
    # infer
    idx_pred=[ custom_cnn.predict(model,[p])[0] for p in p_te ]
    y_pred=[ CLASSES[i] for i in idx_pred ]
    from sklearn.metrics import classification_report
    rpt = classification_report(y_te, y_pred, output_dict=True,zero_division=0)
    from data_utils import save_confusion
    save_confusion(y_te,y_pred,CLASSES,f"{cfg.out}/conf_cnn.png","CNN")
    rows.append(dict(model="CNN", accuracy=rpt["accuracy"],
                     macro_F1=sum(rpt[c]["f1-score"] for c in CLASSES)/len(CLASSES),
                     train_sec=t))

# ── kNN + NB ──────────────────────────────────────────────────────────────
if {"knn","nb"} & set(cfg.models):
    (r_knn,t_knn),(r_nb,t_nb)=hist_baselines.run(p_tr,p_te,y_tr,y_te,CLASSES,cfg.out)
    if "knn" in cfg.models:
        rows.append(dict(model="kNN", accuracy=r_knn["accuracy"],
                         macro_F1=sum(r_knn[c]["f1-score"] for c in CLASSES)/len(CLASSES),
                         train_sec=t_knn))
    if "nb" in cfg.models:
        rows.append(dict(model="NaiveBayes", accuracy=r_nb["accuracy"],
                         macro_F1=sum(r_nb[c]["f1-score"] for c in CLASSES)/len(CLASSES),
                         train_sec=t_nb))

# ── zapis zbiorczy ────────────────────────────────────────────────────────
df=pd.DataFrame(rows); df.to_csv(f"{cfg.out}/report.csv",index=False)
print(df.to_string(index=False))

import os, glob, cv2, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

def load_paths_labels(root):
    paths, labels = [], []
    for cls in sorted(os.listdir(root)):
        p_cls = os.path.join(root, cls)
        if not os.path.isdir(p_cls): continue
        for p in glob.glob(os.path.join(p_cls, "*")):
            if os.path.splitext(p.lower())[1] in ALLOWED_EXT:
                paths.append(p); labels.append(cls)
    return np.array(paths), np.array(labels)

def hsv_hist(path):
    img = cv2.imread(path); img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([cv2.calcHist([c],[0],None,[32],[0,256]).flatten()
                           for c in cv2.split(img)])
    return hist / (hist.sum()+1e-6)

def save_confusion(y_true, y_pred, classes, fname, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(fname); plt.close()

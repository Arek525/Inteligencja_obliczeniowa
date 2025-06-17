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
    img = cv2.imread(path)
    # zwraca tablicę NumPy o kształcie (H, W, 3) i typie uint8, gdzie 3 to kanały w przestrzeni BGR (Blue, Green, Red).

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # przekształca każdy piksel z BGR na HSV: H (Hue) – odcień (0–180 w OpenCV), S (Saturation) – nasycenie (0–255), V (Value) – jasność (0–255).
    # Teraz img ma kształt (H, W, 3), ale kanały to [H, S, V].


    hist = np.concatenate([cv2.calcHist([c],[0],None,[32],[0,256]).flatten()
                           for c in cv2.split(img)])
    # [c] – lista obrazów (tu pojedynczy kanał),
    #
    # [0] – indeks kanału w tej liście (zawsze 0, bo lista ma jeden element),
    #
    # None – brak maski (używamy całego obrazu),
    #
    # [32] – liczba koszyków (bins) histogramu,
    #
    # [0,256] – zakres wartości pikseli, tu od 0 do 255.
    #
    # W efekcie dostajesz tablicę (32,1), gdzie każda komórka to liczba pikseli z danego przedziału wartości (np. ile pikseli ma odcień H między 0–7, 8–15, …).


    # cv2.split(img) zwraca listę trzech tablic 2D:
    # c = img[:,:,0] → kanał H,
    # c = img[:,:,1] → kanał S,
    # c = img[:,:,2] → kanał V.
    # Każda taka tablica ma kształt (H, W) i wartości od 0 do 180 lub 255.

    # np.concatenate skleja trzy 32‐elementowe wektory w jedno 96‐elementowe [H_hist|S_hist|V_hist].



    return hist / (hist.sum()+1e-6)

def save_confusion(y_true, y_pred, classes, fname, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(fname); plt.close()

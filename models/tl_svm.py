import time, numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from data_utils import save_confusion
from features import mobilenet_vec

def run(p_train, p_test, y_train, y_test, classes, outdir):
    X_tr = np.vstack([mobilenet_vec(p) for p in p_train])
    X_te = np.vstack([mobilenet_vec(p) for p in p_test])

    t0=time.time(); svm=SVC(C=1,gamma='scale',kernel='rbf',probability=True)
    svm.fit(X_tr, y_train); train_sec=time.time()-t0
    y_pred=svm.predict(X_te)
    save_confusion(y_test,y_pred,classes,f"{outdir}/conf_tl_svm.png","TL+SVM")

    return classification_report(y_test,y_pred,output_dict=True,zero_division=0),train_sec

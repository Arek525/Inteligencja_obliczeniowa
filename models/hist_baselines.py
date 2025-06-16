import time, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from ..data_utils import hsv_hist, save_confusion

def run(p_train, p_test, y_train, y_test, classes, outdir):
    X_tr=np.vstack([hsv_hist(p) for p in p_train])
    X_te=np.vstack([hsv_hist(p) for p in p_test])

    knn=KNeighborsClassifier(n_neighbors=5); t0=time.time(); knn.fit(X_tr,y_train); t_knn=time.time()-t0
    nb =GaussianNB();                         t1=time.time(); nb .fit(X_tr,y_train); t_nb =time.time()-t1

    pred_knn=knn.predict(X_te); pred_nb=nb.predict(X_te)
    save_confusion(y_test,pred_knn,classes,f"{outdir}/conf_knn.png","kNN")
    save_confusion(y_test,pred_nb ,classes,f"{outdir}/conf_nb.png" ,"NaiveBayes")
    rpt_knn=classification_report(y_test,pred_knn,output_dict=True,zero_division=0)
    rpt_nb =classification_report(y_test,pred_nb ,output_dict=True,zero_division=0)
    return (rpt_knn,t_knn),(rpt_nb,t_nb)

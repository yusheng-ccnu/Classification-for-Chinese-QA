from sklearn import metrics
import numpy as np


def evaluate(model, x_test, y_test):
    p = []
    r = []
    f1 = []
    pre_vec = model.predict(x_test, batch_size=100)
    y_pred = np.argmax(pre_vec, axis=1)
    y_ = np.argmax(y_test, axis=1)
    prec = metrics.precision_score(y_, y_pred, average='weighted')
    p.append(prec)
    print('precision:{0:.3f}'.format(prec))
    rec = metrics.recall_score(y_, y_pred, average='weighted')
    r.append(rec)
    print('recall:{0:0.3f}'.format(rec))
    f1sco = metrics.f1_score(y_, y_pred, average='weighted')
    f1.append(f1sco)
    print('f1-score:{0:.3f}'.format(f1sco))

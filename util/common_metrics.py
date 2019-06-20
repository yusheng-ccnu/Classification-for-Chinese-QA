from sklearn import metrics
import numpy as np


def evaluate(model, x_test, y_test, batch_size=100):
    pre_vec = model.predict(x_test, batch_size=batch_size)
    y_pred = np.argmax(pre_vec, axis=1)
    y_ = np.argmax(y_test, axis=1)
    acc = metrics.accuracy_score(y_, y_pred)
    print('accuracy:{0:.4f}'.format(acc))
    prec = metrics.precision_score(y_, y_pred, average='weighted')
    print('precision:{0:.4f}'.format(prec))
    rec = metrics.recall_score(y_, y_pred, average='weighted')
    print('recall:{0:0.4f}'.format(rec))
    f1sco = metrics.f1_score(y_, y_pred, average='weighted')
    print('f1-score:{0:.4f}'.format(f1sco))
    return acc, prec, rec, f1sco


def evaluate_union(model, x_test, x_test_char, y_test, batch_size=100):
    pre_vec = model.predict([x_test, x_test_char], batch_size=batch_size)
    y_pred = np.argmax(pre_vec, axis=1)
    y_ = np.argmax(y_test, axis=1)
    acc = metrics.accuracy_score(y_, y_pred)
    print('accuracy:{0:.4f}'.format(acc))
    prec = metrics.precision_score(y_, y_pred, average='weighted')
    print('precision:{0:.4f}'.format(prec))
    rec = metrics.recall_score(y_, y_pred, average='weighted')
    print('recall:{0:0.4f}'.format(rec))
    f1sco = metrics.f1_score(y_, y_pred, average='weighted')
    print('f1-score:{0:.4f}'.format(f1sco))
    return acc, prec, rec, f1sco

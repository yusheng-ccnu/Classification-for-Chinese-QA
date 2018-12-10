from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


train_data = []
y_train_data = []
label_index = {}

###加载训练集
with open('../data/train_question.txt', 'r') as file:
    for line in file:
        terms = line.strip('\n').split('\t')
        label = terms[0].split('_')[0].strip(' ')
        if label not in label_index:
            label_index[label] = len(label_index)
        train_data.append(terms[1])
        y_train_data.append(label_index[label])


###记载训练集
test_data = []
y_test_data = []
with open('../data/test_question.txt', 'r') as file:
    for line in file:
        terms = line.strip('\n').split('\t')
        label = terms[0].split('_')[0].strip(' ')
        test_data.append(terms[1])
        y_test_data.append(label_index[label])

count_vect = CountVectorizer()
x_train_data = count_vect.fit_transform(train_data)
tf_transform = TfidfTransformer()
x_train_tfid = tf_transform.fit(x_train_data).transform(x_train_data)

x_test_data = count_vect.fit_transform(test_data)
x_test_tfid = tf_transform.fit(x_test_data).transform(x_test_data

                                                      )
##构建分类器
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(x_train_tfid, y_train_data)
###预测数据
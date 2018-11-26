import jieba
import fastText
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

label_index = {}


def load_stopword(path):
    stop_word = []
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            stop_word.append(line.strip('\n'))
    return stop_word


#字符分词
def load_data(in_path, out_path, stop_path):
    out_file = open(out_path, 'w')
    stop_word = load_stopword(stop_path)
    with open(in_path, 'r') as file:
        for line in file:
            words = line.strip('\n').split('\t')
            label = words[0].split('_')[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            terms = [term for term in words[1] if term not in stop_word]
            terms.append('__label__' + str(label_index[label]))
            out_file.write(' '.join(terms) + '\n')


#汉语分词
def load_jieba_data(in_path, out_path, stop_path):
    f_test = open(out_path, 'w')
    stop_word = load_stopword(stop_path)
    with open(in_path, 'r') as file:
        for line in file:
            terms = line.strip('\n').split('\t')
            label = terms[0].split('_')[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            words = jieba.cut(terms[1])
            f_test.write(' '.join([word for word in words if word not in stop_word]) + ' __label__' + str(label_index[label]) + '\n')


#训练数据
def train_model(data_path, model_path):
    clf = fastText.train_supervised(data_path, epoch=25, lr=0.6, wordNgrams=2, verbose=2, minCount=1, label="__label__")
    clf.save_model(model_path)
    return clf


#加载模型
def load_train_model(model_path):
    clf = fastText.load_model(model_path)
    return clf


#评价
def print_evalution(clf, test_path):
    print(clf.test(test_path))


#训练集路径
in_path = '../data/trainquestion.txt'
out_path = '../data/train_char_question.txt'
#测试集路径
in_test_path = '../data/testquestion.txt'
out_test_path = '../data/test_char_question.txt'
#停用词路径
stop_path = '../data/stopword.txt'
#按字符分词
load_data(in_test_path, out_test_path, stop_path)
load_data(in_path, '../data/train_char_question.txt', stop_path)
load_jieba_data(in_path, '../data/train_jieba_question.txt', stop_path)
load_jieba_data(in_test_path, '../data/test_jieba_question.txt', stop_path)

clf_jieba = train_model('../data/train_jieba_question.txt', '../model_file/fast_jiaba_model.model')
print(clf_jieba.test('../data/test_jieba_question.txt'))

clf = train_model('../data/train_char_question.txt', '../model_file/fast_model.model')
print(clf.test(out_test_path))




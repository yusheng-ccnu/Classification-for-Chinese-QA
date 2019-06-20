from util.data_preprocess import load_stopword
import jieba


def extract_label():
    hums = []
    with open('../dataset/nlp_bz.txt', 'r', encoding='utf-8') as f:
        for line in f:
            terms = line.strip('\n').split('\t')
            if terms[0] == 'TIME':
                hums.append(line.strip('\n'))
    print(len(hums))
    f_write = open('../question/time1.txt', 'w', encoding='utf-8')
    for sentence in hums:
        f_write.write(sentence + '\n')


label_index = {}


def seg(in_path, out_path, stop='../data/stop.txt'):
    f_test = open(out_path, 'w', encoding='utf-8')
    stop_word = load_stopword(stop)
    print(stop_word)
    with open(in_path, 'r', encoding='utf-8') as file:
        for line in file:
            print(line)
            terms = line.strip('\n').split('\t')
            label = terms[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            words = jieba.cut(terms[1].strip(' '))
            f_test.write(label + '\t' + ' '.join([word for word in words if (len(word) > 0 and word not in stop_word)]) + '\n')


def seg_char(in_path, out_path):
    label_index.clear()
    f_test = open(out_path, 'w', encoding='utf-8')
    with open(in_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = []
            terms = line.strip('\n').split('\t')
            label = terms[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            for word in terms[1]:
                words.append(word)
            f_test.write(label + '\t' + ' '.join(words) + '\n')


extract_label()


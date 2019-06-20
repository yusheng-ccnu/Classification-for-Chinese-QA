from util.data_preprocess import load_stopword
import jieba

def seg():
    f_test = open('../dataset/nlp_qa_seg.txt', 'w', encoding='utf-8')
    stop_word = load_stopword('../data/stop.txt')
    with open('../dataset/nlp_qa.txt', 'r', encoding='utf-8') as file:
        for line in file:
            words = jieba.cut(line.strip('\n'))
            f_test.write(' '.join([word for word in words if (len(word) > 0 and word not in stop_word)]) + '\n')


def extract_question():
    qa = open('../dataset/nlp_qa.txt', 'w', encoding='utf-8')
    with open('../data/nlpcc-iccpol-2016.kbqa.testing-data', 'r', encoding='utf-8') as file:
        tag = 0
        sens = []
        for line in file:
            if tag % 3 == 0:
                sens.append(line.split('\t')[1])
                qa.write(line.split('\t')[1])
            tag += 1
    print(sens)


seg()


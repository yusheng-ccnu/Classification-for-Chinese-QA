import re
import sys
import codecs
import jieba


def filte(input_file):
    p1 = re.compile('（）')
    p2 = re.compile('《》')
    p3 = re.compile('「')
    p4 = re.compile('」')
    p5 = re.compile('<doc (.*)>')
    p6 = re.compile('</doc>')
    outfile = codecs.open('../data/zh_wiki_00', 'a+', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as myfile:
        for line in myfile:
            line = p1.sub('', line)
            line = p2.sub('', line)
            line = p3.sub('', line)
            line = p4.sub('', line)
            line = p5.sub('', line)
            line = p6.sub('', line)
            outfile.write(line)
    outfile.close()


def seg():
    filename = '../data/zh_wiki'
    fileneedCut = '../data/zh_wiki_utf-8'
    fn = open(fileneedCut, "r", encoding="utf-8")
    f = open(filename, "w+", encoding="utf-8")
    for line in fn.readlines():
        words = jieba.cut(line)
        for w in words:
            f.write(str(w) + ' ')
    f.close()
    fn.close()


if __name__ == '__main__':
    seg()
from pypinyin import pinyin

with open('../data/testquestion.txt', 'r', encoding='utf-8') as files:
    f_write = open('../data/test_qa_char_7', 'w', encoding='utf-8')
    for line in files:
        terms = line.strip('\n').strip('\ufeff').split('\t')

        pys = pinyin(terms[1])
        words = []
        for word in pys:
            words.append(word[0])
        print(terms[0].split('_')[0])
        f_write.write(terms[0].split('_')[0] + '\t' + ' '.join(words) + '\n')

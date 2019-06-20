
f = open('../data/zh_wiki_2gram_char', 'w', encoding='utf-8')
with open('../data/zh_wiki', 'r', encoding='utf-8') as file:
    for line in file:
        for word in line.strip(' '):
            f.write(word + ' ')

import json

out = open('../data/zhihu_qa.txt', 'w', encoding='utf-8')
with open('../data/question.json', 'r', encoding='utf-8') as file:
    dict = json.load(file)
    for data in dict:
        question = data['question']
        print(question)
        out.write(question + '\n')

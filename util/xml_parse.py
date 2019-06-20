from xml.etree.ElementTree import ElementTree


tree = ElementTree(file='../data/fd_question.xml')
root = tree.getroot()
classification = []
data = {}
file = open('../data/fd_qa.txt', 'w', encoding='utf-8')
for child_root in root:
    if child_root[1].text.strip('\ufeff') not in classification:
        classification.append(child_root[1].text.strip('\ufeff'))

print(classification)

for cf in classification:
    cfs = []
    for child_root in root:
        if child_root[1].text == cf:
            cfs.append(child_root[0].text)
    data[cf] = cfs

for label in data:
    for question in data[label]:
        file.write(label + '\t' + question + '\n')




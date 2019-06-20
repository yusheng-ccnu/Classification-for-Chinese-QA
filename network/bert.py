from bert_serving.client import BertClient
bc = BertClient()
print(bc.encode(['中国', '美国']))
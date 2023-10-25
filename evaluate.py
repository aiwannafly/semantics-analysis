from math import floor

import conllu
import torch
from conllu import TokenList

from transformers import AutoTokenizer

import config

model = torch.load('models/term_recognizer.pt', map_location=torch.device('cpu'))

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

text = "Для создания нейронных сетей широко используется пакет PyTorch."
tokenized_text = tokenizer(text, return_tensors='pt')

model_outputs = model.forward(**tokenized_text)
labels_again = torch.argmax(model_outputs, dim=2)

print(tokenized_text['input_ids'])
print(labels_again)
print(labels_again.shape)

print(tokenizer('PyTorch', return_tensors='pt')['input_ids'])

ds_file = open(config.DS_PATH, 'r', encoding='utf-8')

ds_sentences: list[TokenList] = [*conllu.parse_incr(ds_file)]

ds_size = len(ds_sentences)

train_dataset = ds_sentences[:floor(ds_size * .9)]
test_dataset = ds_sentences[floor(ds_size * .9):]

labels = ['[PAD]', 'O', 'I-TERM']

TP, FP, TN, FN = 0, 0, 0, 0

for sent in test_dataset:
    tokens = []
    for marked_token in sent:
        if " " in marked_token['form']:
            split_idx = marked_token['form'].rfind(' ')
            (token, label) = marked_token['form'][:split_idx], marked_token['form'][split_idx + 1:]
        else:
            (token, label) = marked_token['form'], config.DEFAULT_CLASS
        tokens.append(token)
    inputs = tokenizer(' '.join(tokens), return_tensors='pt')
    outputs = model.forward(**inputs)
    predicted_labels = torch.squeeze(torch.argmax(outputs, dim=2))

    offset = 1
    for marked_token in sent:
        if " " in marked_token['form']:
            split_idx = marked_token['form'].rfind(' ')
            (token, label) = marked_token['form'][:split_idx], marked_token['form'][split_idx + 1:]
        else:
            (token, label) = marked_token['form'], config.DEFAULT_CLASS

        sub_tokens = tokenizer(token)['input_ids'][1:-1]

        expected = label

        count = 0
        for pl in predicted_labels[offset:offset + len(sub_tokens)]:
            if pl == labels.index('I-TERM'):
                count += 1
        if count > len(sub_tokens) / 2:
            predicted = 'I-TERM'
        else:
            predicted = 'O'
        offset += len(sub_tokens)

        if predicted == expected == 'I-TERM':
            TP += 1
        elif predicted == 'O' != expected:
            FN += 1
        elif predicted == 'I-TERM' != expected:
            FP += 1
        else:
            TN += 1

print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
acc = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy: {:.2f}".format(acc))
precision = TP / (TP + FP)
print("Precision: {:.2f}".format(precision))
recall = TP / (TP + FN)
print("Recall: {:.2f}".format(recall))
f_measure = 2 * precision * recall / (precision + recall)
print("F-measure: {:.2f}".format(f_measure))


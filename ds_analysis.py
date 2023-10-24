import conllu
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight

import config

ds_file = open(config.DS_PATH, 'r', encoding='utf-8')

iterator = conllu.parse_incr(ds_file)

# data = []
#
# tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
#
# for token_list in iterator:
#     tokens = [t['form'].split(' ')[0] for t in token_list]
#     # length = 0
#     # for t in tokens:
#     #     length += len(tokenizer(t)['input_ids'][1:-1])
#     data.append(len(tokens))
#
# plt.hist(data, color='lightgreen', ec='black', bins=15)
# plt.xlabel('Words per sentence')
# plt.savefig('length_distribution.png')


def compute_class_weights() -> list[float]:
    ds_file_ = open(config.DS_PATH, 'r', encoding='utf-8')

    iterator_ = conllu.parse_incr(ds_file)
    class_ids = []
    for marked_sent in iterator:
        for marked_token in marked_sent:
            if " " in marked_token['form']:
                split_idx = marked_token['form'].rfind(' ')
                (token, label) = marked_token['form'][:split_idx], marked_token['form'][split_idx + 1:]
            else:
                (token, label) = marked_token['form'], config.DEFAULT_CLASS
            class_ids.append(config.CLASSES.index(label) + 1)
    print(set(class_ids))
    ds_file_.close()
    return compute_class_weight('balanced', classes=[i + 1 for i in range(len(config.CLASSES))], y=class_ids)

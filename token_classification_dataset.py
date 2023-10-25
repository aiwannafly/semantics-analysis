from typing import List, Dict

import torch
from conllu import TokenList
from torch.utils.data.dataset import Dataset

import config


class TokenClassificationDataset(Dataset):

    def __init__(self, marked_sentences: list[TokenList], classes: List[str], tokenizer):
        super().__init__()
        self.examples = []
        self.labels = [config.PAD_TOKEN] + classes

        batches = []
        curr = 0
        while curr < len(marked_sentences):
            next_idx = curr + config.BATCH_SIZE
            if next_idx > len(marked_sentences):
                next_idx = len(marked_sentences) - 1
                if next_idx == curr + 1:
                    break
                batches.append(marked_sentences[curr:next_idx])
                break
            if next_idx == curr + 1:
                break
            batches.append(marked_sentences[curr:next_idx])
            curr = next_idx

        for batch in batches:
            batch_examples = []
            for marked_sent in batch:
                input_ids = [config.CLS_TOKEN_IDX]
                label_ids = [config.PAD_TOKEN_IDX]
                for marked_token in marked_sent:
                    if " " in marked_token['form']:
                        split_idx = marked_token['form'].rfind(' ')
                        (token, label) = marked_token['form'][:split_idx], marked_token['form'][split_idx + 1:]
                    else:
                        (token, label) = marked_token['form'], config.DEFAULT_CLASS

                    sub_token_ids = tokenizer(token)['input_ids'][1:-1]

                    for i in range(len(sub_token_ids)):
                        input_ids.append(sub_token_ids[i])

                        if i == len(sub_token_ids) - 1:
                            label_ids.append(self.labels.index(label))
                        else:
                            label_ids.append(self.labels.index(label))
                            # label_ids.append(config.PAD_TOKEN_IDX)

                input_ids.append(config.SEP_TOKEN_IDX)
                label_ids.append(config.PAD_TOKEN_IDX)

                attention_mask = [1] * len(input_ids)

                example = {
                    'input_ids': input_ids,
                    'token_type_ids': [0] * len(input_ids),
                    'attention_mask': attention_mask,
                    'label_ids': label_ids
                }

                batch_examples.append(example)
            max_length = max(len(e['input_ids']) for e in batch_examples)
            for e in batch_examples:
                diff = max_length - len(e['input_ids'])
                for k, v in e.items():
                    e[k] = torch.tensor(v + [0] * diff, dtype=torch.long, device=config.DEVICE)
            batch_example = {k: torch.stack([e[k] for e in batch_examples]) for k in batch_examples[0].keys()}
            self.examples.append(batch_example)
            if len(batch) < config.BATCH_SIZE:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        return self.examples[item]

import torch
from transformers import AutoTokenizer

import spacy
import config

model = torch.load('term_recognizer.pt', map_location=torch.device(config.DEVICE))
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
nlp = spacy.blank('ru')


def extract_terms(text: str) -> list[str]:
    tokens = [str(t) for t in nlp(text)]

    input_ids = [config.CLS_TOKEN_IDX]
    curr = 1
    ids_range_by_token_id = {}
    for i in range(len(tokens)):
        t = tokens[i]
        token_input_ids = tokenizer(t)['input_ids'][1:-1]

        ids_range_by_token_id[i] = (curr, curr + len(token_input_ids))
        curr += len(token_input_ids)

        input_ids.extend(token_input_ids)
    input_ids.append(config.SEP_TOKEN_IDX)

    outputs = model.forward(**tokenizer(text, return_tensors='pt'))
    predicted_labels = torch.squeeze(torch.argmax(outputs, dim=2))

    predicted_classes = [config.CLASSES[(l - 1)] for l in predicted_labels]

    assert len(input_ids) == len(predicted_classes)

    terms = []
    curr_terms = []
    for i in range(len(tokens)):
        t = tokens[i]
        l, r = ids_range_by_token_id[i]

        term_count = 0
        for j in range(l, r):
            if predicted_classes[j].endswith('TERM'):
                term_count += 1
        if term_count > (r - l) / 2:
            curr_terms.append(t)
        else:
            if len(curr_terms) > 0:
                terms.append(' '.join(curr_terms).replace(' - ', '-'))
                curr_terms.clear()

    terms.extend(curr_terms)
    curr_terms.clear()
    return terms


# print(extract_terms("Как хорошо использовать PyTorch для обучения нейронных сетей?"))

with open('prompt_tuning_revision.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

    lines.append("It is important to check that usual english words do not trigger the model like 'Tensorflow'.")

    for l in lines:
        if len(l) == 0 or l.isspace():
            continue
        l = l.strip()
        terms = extract_terms(l)

        print(f"[PARAGRAPH]: {l}")
        print(f"[TERMS]: {terms}")
        print()

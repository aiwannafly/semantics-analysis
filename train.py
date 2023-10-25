from math import floor

import conllu
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import config
from ds_analysis import compute_class_weights
from term_recognition import TermRecognizerModel
from token_classification_dataset import TokenClassificationDataset

ds_file = open(config.DS_PATH, 'r', encoding='utf-8')

ds_sentences = [*conllu.parse_incr(ds_file)]

ds_size = len(ds_sentences)

train_dataset = ds_sentences[:floor(ds_size * .9)]
test_dataset = ds_sentences[floor(ds_size * .9):]

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

train_dataset = TokenClassificationDataset(train_dataset, config.CLASSES, tokenizer)
test_dataset = TokenClassificationDataset(test_dataset, config.CLASSES, tokenizer)

print(f"Train dataset contains {len(train_dataset)} sentences.")
print(f"Test dataset contains {len(test_dataset)} sentences.")

train_loader = DataLoader(train_dataset, batch_size=None)
test_loader = DataLoader(train_dataset, batch_size=None)

weights = [0.0]
weights.extend(compute_class_weights())
weights = torch.tensor(weights, dtype=torch.float, device=config.DEVICE)

model = TermRecognizerModel()
model.to(device=config.DEVICE)
loss_fn = CrossEntropyLoss(weight=weights, ignore_index=config.PAD_TOKEN_IDX)
optim = AdamW(model.parameters(), lr=config.LEARNING_RATE)

train_loss = None
eval_loss = None
for e in range(config.EPOCHS):

    train_progress = tqdm(train_loader)
    train_progress.set_description(f"Epoch {e + 1}")

    model.train()
    for example in train_progress:
        target = example.pop('label_ids')

        optim.zero_grad()
        logits = torch.squeeze(model.forward(**example))

        B, T, P = logits.shape
        logits = logits.view(B * T, P)
        target = target.view(B * T)

        loss = loss_fn(logits, target)

        loss.backward()
        optim.step()

        train_loss = loss.item()

        train_progress.set_postfix({"train_loss": train_loss, "eval_loss": eval_loss})

    model.eval()
    total_eval_loss = 0
    for example in test_loader:
        target = example.pop('label_ids')
        logits = torch.squeeze(model.forward(**example))

        B, T, P = logits.shape
        logits = logits.view(B * T, P)
        target = target.view(B * T)

        loss = loss_fn(logits, target)
        total_eval_loss += loss.item()
    eval_loss = total_eval_loss / len(test_loader)
    train_progress.set_postfix({"train_loss": train_loss, "eval_loss": eval_loss})

    if e % 10 == 0:
        torch.save(model, 'models/term_recognizer.pt')


torch.save(model, 'models/term_recognizer.pt')

import os
import re

TEXT_FILE_REGEX = re.compile(r".+\.txt")

dirs = [f'train{i}/dataset_entity' for i in range(1, 4)]

all_file_names = []
for d in dirs:
    all_file_names += [f'{d}/{f}' for f in os.listdir(d)]

ds_file_names = [f for f in all_file_names if TEXT_FILE_REGEX.match(f)]

with open('class_term_dataset.conllu', 'w', encoding='utf-8') as wf:
    for f_name in ds_file_names:
        with open(f_name, 'r', encoding='utf-8') as f:
            wf.write(f.read())
            wf.write("\n")

import conllu

ds_file = open('data/formatted_term_dataset.conllu', 'r', encoding='utf-8')

iterator = conllu.parse_incr(ds_file)

for i in range(1):
    token_list = next(iterator)
    print(str([t['form'].split(' ')[0] for t in token_list]))


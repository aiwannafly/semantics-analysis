sent_id = 1

with open('class_formatted_term_dataset.conllu', 'w', encoding='utf-8') as wf:
    with open('class_term_dataset.conllu', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if line.startswith("# text") or line.startswith("#text"):
                wf.write(f"# sent_id = {sent_id}\n")
                sent_id += 1
            wf.write(line)

            token_id = 1
            while i < len(lines):
                line = lines[i]
                i += 1
                if line.startswith("# text") or line.startswith("#text"):
                    i -= 1
                    break
                if len(line) == 0 or line.isspace():
                    continue
                wf.write(f"{token_id}\t{line}")
                token_id += 1
            wf.write("\n")

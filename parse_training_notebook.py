import re
import json
import os


samples_file = os.path.join('data', 'sample_notebooks', 'train_v0_final.py')

with open(samples_file) as f:
    all_samples = f.read()

split_text = re.split(r'Prompt:\n###|:\n"""|# In\[[0-9]*\]:\n', all_samples)

idx_tuples = []



i = 2
sample_data = []
while(len(idx_tuples) < 45):
    idx_tuples.append((i,i+1))
    i = i + 3
for (i,t) in enumerate(idx_tuples):
    prompt = split_text[t[0]]
    if prompt[0] == '\n':
        prompt = prompt[1:]
    prompt = prompt + ':\n'
    completion = split_text[t[1]]
    completion = re.sub(r'\s+$', '', completion)
    samp_dict = {"prompt": prompt, "completion": completion}
    sample_data.append(samp_dict)

with open('train_v0.json', 'w') as fout:
    json.dump(sample_data, fout)
#   json.dump(sample_data, f)

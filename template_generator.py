from itertools import product, chain, combinations
import os
import functools

import prompt_parameters as params
from notebook_tools import *

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

base_prompt = params.components['base']
params_keys = params.components.keys()
option_prompts = {opt: params.components[opt] for opt in params_keys if opt != 'base'}
opt_keys = option_prompts.keys()

prompt_keys = powerset(opt_keys)
option_sets = dict()
for (i, pkeys) in enumerate(prompt_keys):
    blocks = tuple([option_prompts[k] for k in pkeys])
    if len(blocks) == 0:
        continue
    option_set_key = functools.reduce(lambda a, b: a + '-' + b, pkeys)
    option_sets[option_set_key] = blocks

outdir = os.path.join('template_notebooks', 'empty')

for (i, options_key) in enumerate(option_sets.keys()):
    opts = option_sets[options_key]
    opts_text = functools.reduce(lambda base, next_op: base + ' ' + next_op, opts) 
   
    prompt_text = base_prompt + ' ' + opts_text
    prompt_cell = build_prompt_cell(prompt_text, options_key)
    inputs = get_prompt_inputs(prompt_text)
    code_cells = build_code_cells(options_key, inputs=inputs)
    
    ntbk_cells = [prompt_cell] + code_cells
    ntbk_cell_txt = build_cell_chain(ntbk_cells)
    ntbk_txt = build_notebook(ntbk_cell_txt)
    
    outpath = os.path.join(outdir, '_' + str(i).zfill(4) + '_' + f'{options_key}' + '.ipynb')
    write_jptr_notebook(outpath, ntbk_txt)
breakpoint()
'''
Tentative cell structure -

(1) markup cell with contents - 
{prompt-key}
#BEGIN-prompt#
{Prompt}
#END-prompt#
#BEGIN-setparams#
Please set below parameters:
{prompt parameter} + ' = ' + '?' for each of parameters
#END-setparams#

(2) code cell - 1
# {prompt-key}
# sample_id : 0
#Begin-code#
#NOTIMPLEMENTED#
#End-code#

(3) code cell - 2
# {prompt-key} : 1
# sample_id : 1
#Begin-code#
#NOTIMPLEMENTED#
#End-code#

(4) code cell - 3
# {prompt-key} : 2
# sample_id : 2
#Begin-code#
#NOTIMPLEMENTED#
#End-code#
'''
# chart_types = ['scatter', 'line', 'bar', 'histogram', 'pie', 'stacked-bar', 'stem', 'compound']
# linestyle_types = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dotted', 'dotted', 'densely dotted', 'loosely dashed',
#         'dashed', 'densely dashed', 'loosely dashdotted', 'dashdotted', 'densely dashdotted', 'dashdotdotted', 
#         'loosely dashdotdotted', 'densely dashdotdotted']
# markerstyle_types = ['.', ',', 'o', 'v', '^', '<', '>']
# grid_shades = ['very light', 'light', 'dark', 'very dark', 'black']
# grid_extents = ['minimal', 'moderate', 'maximal']




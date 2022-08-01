from itertools import product, chain, combinations
import os
import functools

import prompt_parameters as params
from notebook_tools import *


add_optional_append = True

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

base_preface = params.components['base_preface']
base_suffix = params.components['base_suffix']
params_keys = params.components.keys()
option_prompts = {opt: params.components[opt] for opt in params_keys if 'base' not in opt}
chart_types = params.chart_type_dict
opt_keys = option_prompts.keys()

prompt_keys = powerset(opt_keys)
prompt_keys = [k for k in prompt_keys if len(list(k)) < 5]
option_sets = dict()
for (i, pkeys) in enumerate(prompt_keys):
    blocks = tuple([option_prompts[k] for k in pkeys])
    if len(pkeys) > 0:
        option_set_key = functools.reduce(lambda a, b: a + '-' + b, pkeys)
    else:
        option_set_key = pkeys
    option_sets[option_set_key] = blocks

outdir = os.path.join('template_notebooks', 'empty')
cnt = 0
for k in chart_types.keys():
    chart_str = chart_types[k]
    base_prompt = base_preface + chart_str + base_suffix
    for options_key in option_sets.keys():
        param_order = len(option_sets[options_key])
        if param_order > 2:
            continue
        opts = option_sets[options_key]
        if param_order > 0:
            num_samps = 5
            opts_text = functools.reduce(lambda base, next_op: base + ' ' + next_op, opts) 
            setting_key = k + '-' + options_key
            prompt_text = base_prompt + ' ' + opts_text
        else:
            num_samps = 10
            opts_text = ''
            setting_key = k
            prompt_text = base_prompt
       
        inputs = get_prompt_inputs(prompt_text)
        if add_optional_append:
            prompt_text = prompt_text + '{OPTIONAL_APPEND}'
        prompt_cell = build_prompt_cell(prompt_text, setting_key)
        code_cells = build_code_cells(setting_key, inputs=inputs,
                                                   num_samples=num_samps)
        
        ntbk_cells = [prompt_cell] + code_cells
        ntbk_cell_txt = build_cell_chain(ntbk_cells)
        ntbk_txt = build_notebook(ntbk_cell_txt)
        
        outpath = os.path.join(outdir,
                               f'order_{param_order}',
                               f'{setting_key}' + '.ipynb')
        write_jptr_notebook(outpath, ntbk_txt)
        cnt += 1
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




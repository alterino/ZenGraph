from itertools import product, chain, combinations
from notebook_tools import *

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

components = dict()

components['base'] = 'Write code that loads data from {CSV_NAME} and draws a {CHART_TYPE} chart of {VARIABLE_SPEC}.'
components['title'] = 'Make the title {TITLE}.'
components['color'] = 'Use {COLOR_SPEC} colors for the data.'
components['linestyle'] = 'Use the line style {LINE_STYLE}.'
components['markerstyle'] = 'Use the [marker, line] style {MARKER_STYLE}.'
components['bgcolor'] = 'Make the background color {BG_COLOR}.'
components['markerfill'] = 'Marker fill should be set to {MARKER_FILL}.'
components['gridspecs'] = 'Draw a {GRID_SHADE}, {GRID_EXTENT} grid'

components['xlabel'] = 'Make the x axis label {XLABEL}.'
components['ylabel'] = 'Make the y axis label {YLABEL}.'

components['mplstyle'] = 'Use the matplotlib style preset {STYLE_PRESET}.'

base_key = 'base'
opt_keys = ['title', 
            'color', 
            'linestyle',
            'markerstyle', 
            'bgcolor', 
            'markerfill', 
            'gridspecs', 
            'xlabel',
            'ylabel',
            'mplstyle'
            ]

prompt_keys = powerset(opt_keys)
prompt_blocks = dict()
for (i, pkeys) in enumerate(prompt_keys):
    blocks = tuple([components[k] for k in pkeys])
    prompt_blocks[pkeys] = blocks
    #print(pkeys, end='')
    #print(f' ({i}): ')
    #for b in blocks:
    #    print(b)
    #print()

for (i, k) in enumerate(prompt_blocks.keys()):
    print(i, k)
breakpoint()
# chart_types = ['scatter', 'line', 'bar', 'histogram', 'pie', 'stacked-bar', 'stem', 'compound']
# linestyle_types = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dotted', 'dotted', 'densely dotted', 'loosely dashed',
#         'dashed', 'densely dashed', 'loosely dashdotted', 'dashdotted', 'densely dashdotted', 'dashdotdotted', 
#         'loosely dashdotdotted', 'densely dashdotdotted']
# markerstyle_types = ['.', ',', 'o', 'v', '^', '<', '>']
# grid_shades = ['very light', 'light', 'dark', 'very dark', 'black']
# grid_extents = ['minimal', 'moderate', 'maximal']




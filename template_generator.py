





base_prompt = 'Write code that loads data from {CSV_NAME} and draws a {CHART_TYPE} chart of {VARIABLE_SPEC}.'
title_prompt = 'Make the title {TITLE}.'
color_prompt = 'Use {COLOR_SPEC} colors for the data.'
linestyle_prompt = 'Use the line style {LINE_STYLE}.'
markerstyle_prompt = 'Use the marker style {MARKER_STYLE}.'
bgcolor_prompt = 'Make the background color {BG_COLOR}.'
markerfill_prompt = 'Marker fill should be set to {MARKER_FILL}.'
gridshade_prompt = 'Draw a {GRID_SHADE}, {GRID_EXTENT} grid'

mplstyle_prompt = 'Use the matplotlib style preset {STYLE_PRESET}.'


chart_types = ['scatter', 'line', 'bar', 'histogram', 'pie', 'stacked-bar', 'stem', 'compound']
linestyle_types = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dotted', 'dotted', 'densely dotted', 'loosely dashed',
        'dashed', 'densely dashed', 'loosely dashdotted', 'dashdotted', 'densely dashdotted', 'dashdotdotted', 
        'loosely dashdotdotted', 'densely dashdotdotted']
markerstyle_types = ['.', ',', 'o', 'v', '^', '<', '>']
grid_shades = ['very light', 'light', 'dark', 'very dark', 'black']
grid_extents = ['minimal', 'moderate', 'maximal']

"""
combos - base base/title base/title/color base/title/linestyle base/title/markerstyle
         base/title/color/linestyle base/title/color/markerstyle
"""

test = base_prompt + ' ' + title_prompt

breakpoint()

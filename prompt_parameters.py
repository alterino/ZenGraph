components = dict()

chart_type_dict = {
                   'line'       : 'line graph',
                   'scatter'    : 'scatter plot',
                   'histogram'  : 'histogram',
                   'bar'        : 'bar chart',
                   'pie'        : 'pie chart',
                   'boxwhisker' : 'box-and-whisker plot'
                   }
components['base_preface'] = 'Write code that loads data from {CSV_NAME} and draws a '
components['base_suffix'] = ' of {VARIABLE_SPEC}.'
components['title'] = 'Make the title {TITLE}.'
components['color'] = 'Use {COLOR_SPEC} colors for the data.'
components['linestyle'] = 'Use the line style {LINE_STYLE}.'
components['markerstyle'] = 'Use the marker style {MARKER_STYLE}.'
components['bgcolor'] = 'Make the background color {BG_COLOR}.'
components['markerfill'] = 'Marker fill should be set to {MARKER_FILL}.'
components['gridspecs'] = '{GRID_SPEC}'

components['xlabel'] = 'Make the x axis label {XLABEL}.'
components['ylabel'] = 'Make the y axis label {YLABEL}.'

components['mplstyle'] = 'Use the matplotlib style preset {STYLE_PRESET}.'

#base_key = 'base'
#opt_keys = ['title', 
#            'color', 
#            'linestyle',
#            'markerstyle', 
#            'bgcolor', 
#            'markerfill', 
#            'gridspecs', 
#            'xlabel',
#            'ylabel',
#            'mplstyle'
#            ]

import re
import functools

def raw(string: str, replace: bool = False) -> str:
    r = repr(string)[1:-1]  # Strip the quotes from representation
    if replace:
        r = r.replace('\\\\', '\\')
    return r
    
def prime_factorization(n):
    prime_factors = []
    while (n % 2 == 0):
        n = n / 2
        prime_factors.append(2)
    for i in range(3, int(n**0.5 + 1), 2):
        while (n % i == 0):
            n = n / i
            prime_factors.append(i)
    if n > 2:
        prime_factors.append(int(n))
    return prime_factors

# templates for building notebook text
cell_templates = dict()
with open('code_template') as f:
    cell_templates['code'] = f.read()
with open('markdown_template') as f:
    cell_templates['markdown'] = f.read()
with open('empty_ntbk_template') as f:
    empty_ntbk_template = f.read()

# generic jupyter notebook text building functions
def write_jptr_notebook(filename, ntbk_text):
    with open(filename, 'w') as f:
        f.write(ntbk_text)

def build_notebook(input_cells):
    ntbk_str = empty_ntbk_template.replace('{CELLS}', input_cells)
    return re.sub(r'\s+$', '', ntbk_str)

def build_ntbk_cell(input_str, cell_type="code"):
    cell_str = cell_templates[cell_type].replace('{CELLTEXT}', input_str)
    cell_str = cell_str.replace(',\n', ',\n    ')

           # seems suspicious I have this re.sub in multiple places...
    return re.sub(r'\s+$', '', cell_str)

def build_cell_chain(cell_list):
    return functools.reduce(link_ntbk_cells, cell_list)

def link_ntbk_cells(base_chain, tail):
    return base_chain + ',\n' + tail

# template-specific functions
def build_prompt_cell(prompt_text, options_key):
    prompt_text = \
            f'\"options_key=\'{options_key}\'<br><br>\",\n\"Prompt:<br><br>\",\n\"**{prompt_text}**<br>\"\n'

    #re_inputs = re.compile(r' \{(\S*)\} ')
    #inputs = re_inputs.findall(prompt_text)
    #inputs = inputs
    #base = ''
    #params_text = functools.reduce(lambda a, b: a + '\"' + b +' = {?}<br>\",\n', inputs, base)
    #params_text = '\"Please specify these parameters in comments<br>\",\n' + params_text + '\"Format as above in specified section.\"'
    cell_text = prompt_text

    return build_ntbk_cell(cell_text, cell_type="markdown")
    

def build_code_cells(options_key, num_samples=5, inputs=None):
    cells = []
    lines = [options_key, ]
    code_cells = []
    for i in range(num_samples):
        code_text = '\"\'\'\'\\n\",\n'
        code_text += f'\"options_key=\'{options_key}\'\\n\",\n'
        code_text += f'\"sample_id=<{i}>\\n\",\n'
        code_text += '\"\\n\",\n'
        code_text += '\"Please specify the following inputs from your sample:\\n\",\n'
        code_text += '\"\\n\",\n'
        if inputs is not None:
            for inp in inputs:
                code_text += f'\"{inp} = ' + '{NOT SPECIFIED}\\n\",\n'
        code_text += '\"OPTIONAL_APPEND = {None}\\n\",\n'
        code_text += '\"\'\'\'\\n\",\n'

        code_text += '\"\\n\",\n'
        code_text += '\"# BEGIN-code\\n\",\n\"# NOT IMPLEMENTED\\n\",\n\"# END-code\"'
        code_cells.append(build_ntbk_cell(code_text, cell_type="code"))

    return code_cells

def get_prompt_inputs(prompt_text):

    re_inputs = re.compile(r'\{(\S*)\}')
    inputs = re_inputs.findall(prompt_text)

    return inputs




if __name__ == '__main__':

    with open('with_something_in_box.ipynb') as f:
        test_text = f.read()
    cell_str1 = build_ntbk_cell('this is a test markdown cell', cell_type="markdown")
    cell_str2 = build_ntbk_cell('#this is a test code cell', cell_type="code")
    cell_str3 = build_ntbk_cell('#this is another test code cell', cell_type="code")


    base_cell = cell_str1
    out_cells = base_cell
    for in_cell in [cell_str2, cell_str3]:
        out_cells = link_ntbk_cells(out_cells, in_cell)

    out_ntbk = build_notebook(out_cells)
    breakpoint()
    out_ntbk = re.sub(r'\s+$', '', out_ntbk)
    breakpoint()

    with open('test_ntbk.ipynb', 'w') as f:
        f.write(out_ntbk)

    print('\n--------------------------test file-------------------------------------')
    print(test_text)
    print('\n-----------------------generated file-----------------------------------')
    print(out_ntbk)
    breakpoint()

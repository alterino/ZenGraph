import re

cell_templates = dict()
with open('code_template') as f:
    cell_templates['code'] = f.read()
with open('markdown_template') as f:
    cell_templates['markdown'] = f.read()
with open('empty_ntbk_template') as f:
    empty_ntbk_template = f.read()

def build_notebook(input_cells):
    return empty_ntbk_template.replace('{CELLS}', input_cells)

def build_ntbk_cell(input_str, cell_type="code"):
    cell_str = cell_templates[cell_type].replace('{CELLTEXT}', input_str)
    cell_str = re.sub(r'\s+$', '', cell_str)

    return cell_str

def link_ntbk_cells(base_chain, tail):
    return base_chain + ',\n' + tail

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
    out_ntbk = re.sub(r'\s+$', '', out_ntbk)

    with open('test_ntbk.ipynb', 'w') as f:
        f.write(out_ntbk)

    print('\n--------------------------test file-------------------------------------')
    print(test_text)
    print('\n-----------------------generated file-----------------------------------')
    print(out_ntbk)
    breakpoint()

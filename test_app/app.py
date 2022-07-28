from flask import Flask, render_template, request, redirect, url_for, jsonify, abort 
from flask_bootstrap import Bootstrap

from functools import reduce
import requests
import json
import os
#from flask_fontawesome import FontAwesome

import sys

app = Flask(__name__)
bootstrap = Bootstrap(app)
#fa = FontAwesome(app)

def openai_api_call(prompt, max_tokens=128,
                            model="text-davinci-002",
                            temperature=0.7,
                            stop=None):

    api_key = os.environ.get("OPENAI_API_KEY")

    headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key
            }
    data = {
              "prompt": prompt,
              "max_tokens": max_tokens,
              "model" : model,
              "temperature" : temperature
           }
    if stop is not None:
        data["stop"] = stop
    data = json.dumps(data)

    url = "https://api.openai.com/v1/completions"

    return requests.post(url, headers=headers, data=data)

@app.route('/query_state', methods=['POST', 'GET'])
def get_query_state():
    return jsonify({'states_bool' : queryState['included_states'],
                    'stages_bool' : queryState['included_stages']})
    
@app.route('/submit_query', methods=['POST'])
def process_query():

    prompt_params = request.get_json()

    for k in prompt_params:
        print(k, prompt_params[k])
        



    return jsonify({'success': True})

@app.route('/load_csv_data', methods=['POST'])
def load_csv_data():

    data = request.get_json()
    varnames = list(data[0].keys())
    
    prepend = 'variable names:\n'

    varname_str = prepend + reduce(lambda base, name : base + ', ' + name, varnames)

    return jsonify({'success': True,
                    'variables': varnames,
                    'varstr' : varname_str})

@app.route('/submit_raw', methods=['POST'])
def submit_raw():

    raw_params = request.get_json()
    prompt = raw_params['raw_prompt']
    result = openai_api_call(prompt)
    completion = result.json()['choices'][0]['text']

    return jsonify({'success': True,
                    'prompt': prompt,
                    'completion': completion})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


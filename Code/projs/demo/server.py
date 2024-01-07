'''
demo file
for Flask server applications used in this project


Flask Server quick build instruction
'''

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
import json
import re
import warnings
from datetime import datetime
from markupsafe import escape
from pprint import pprint
import traceback
import sys
sys.dont_write_bytecode = True

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)


## request.args：针对GET方法，从request中获取参数 kv-args
@app.route('/')
def hello():
    print(request.path)
    print(request.full_path)
    return str(request.args.getlist('p'))



## 直接从URL中解析变量
@app.route('/<name>')
def user_page(name):
    return f'user name {escape(name)}'


# request.stream 和 request.form：针对POST方法，从request中获取表单数据 form data
@app.route('/register', methods=['POST'])
def register():
    print(request.headers)
    # print(request.stream.read())
    print(request.form)
    print(request.form.getlist('name'))
    return 'welcome'


# request.json：针对POST方法，从request中获取json数据 json data
@app.route('/add', methods=['POST'])
def add():
    print(request.headers)
    print(type(request.json))
    print(request.json)
    result = request.json['a'] + request.json['b']
    return str(result)







if __name__ == "__main__":
    app.run(port=8000, debug=True)
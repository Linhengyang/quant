from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
import json
import re
import warnings
from datetime import datetime
from pprint import pprint
import traceback
import sys
sys.dont_write_bytecode = True

from Code.Forecasters import BlackLitterman
from Code.Allocators import MeanVarOptimal
from Code.Allocators import RiskParity
from Code.Estimators import Risks

warnings.filterwarnings('ignore')
app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Welcome to My test flask!'
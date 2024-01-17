from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
import json
import re
import warnings
from datetime import datetime, timedelta
from markupsafe import escape
from pprint import pprint
import traceback
import sys
sys.dont_write_bytecode = True

from Code.projs.asset_allocate.functions import *
from Code.projs.asset_allocate.dataload import db_rtn_data, db_date_data
from Code.Utils.Sequence import strided_slicing_w_residual
from Code.BackTester.BT_AssetAllocate import rtn_multi_periods

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

asset_allocate_app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)




from Code.projs.asset_allocate.mean_var import mvopt_api
asset_allocate_app.register_blueprint(mvopt_api)

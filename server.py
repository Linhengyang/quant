from flask import Flask
import warnings
import sys
import numpy as np
sys.dont_write_bytecode = True



warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'



asset_allocate_app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)




from Code.projs.asset_allocate.runner import (
    mvopt_api,
    riskmng_api,
    fxdcomb_api
)



asset_allocate_app.register_blueprint(mvopt_api)
asset_allocate_app.register_blueprint(riskmng_api)
asset_allocate_app.register_blueprint(fxdcomb_api)










if __name__ == "__main__":
    asset_allocate_app.run(port=33000, debug=True)
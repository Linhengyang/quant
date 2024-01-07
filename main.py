from Code.projs.asset_allocate.server import asset_allocate_app
import sys
sys.dont_write_bytecode = True





if __name__ == "__main__":
    asset_allocate_app.run(port=8000, debug=True)
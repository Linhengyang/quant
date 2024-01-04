import pandas as pd 
import numpy as np
# from datetime import datetime, timedelta
from Code.DataLoaders.dbconnect import DatabaseConnection

# remotedb = {
#     'ip':'xx.xxx.xxx.xx',
#     'port':0000,
#     'user':'xxxxxx',
#     'pwd':'xxxxx',
#     'db':'xxxx'
# }

# localdb = {
#     'path':'XX/xx.db'
# }

localdb = {
    'path':'Data/aidx.db'
}

def db_rtn_data(assets:list, startdate:str, enddate:str, rtn_dilate):
    db = DatabaseConnection(localdb)
    assert len(assets) > 0, 'assets index list must not be empty'
    assert int(startdate) <= int(enddate), 'start date must no later than end date'

    sql_query = '''
    SELECT DISTINCT S_IRDCODE, TRADE_DT, {dilate}*PCHG as PCHG
    FROM aidx_eod_prices
    WHERE TRADE_DT >= {startdate} AND TRADE_DT <= {enddate}
    AND S_IRDCODE in {assets_tuple}
    '''.format(startdate=int(startdate), enddate=int(enddate),
               assets_tuple=tuple(assets), dilate=int(rtn_dilate))
    
    raw_data = db.GetSQL(sql_query, tbl_type='pddf')
    data = raw_data.pivot_table(index='S_IRDCODE', columns=['TRADE_DT'], values=['PCHG'])
    rtn_data = data.to_numpy(dtype=np.float32)
    assets_inds = data.index.to_list()
    
    return rtn_data, assets_inds
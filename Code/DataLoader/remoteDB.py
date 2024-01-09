import pandas as pd 
import numpy as np
# from datetime import datetime, timedelta
from Code.DataLoader.dbconnect import DatabaseConnection

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
    if raw_data.shape[0] <= 1:
        raise LookupError('Data 0 Extraction for table aidx_eod_prices from {start} to {end} among {assets_tuple}'.format(
            start=int(startdate), end=int(enddate), assets_tuple=tuple(assets)
        ))
    data = raw_data.pivot_table(index='S_IRDCODE', columns=['TRADE_DT'], values=['PCHG'])
    rtn_data = data.to_numpy(dtype=np.float32)
    assets_inds = data.index.to_list()
    
    return rtn_data, assets_inds



def db_date_data(assets:list, startdate:str, enddate:str):
    # get trade dates for assets from startdate to enddate
    db = DatabaseConnection(localdb)
    assert int(startdate) <= int(enddate), 'start date must no later than end date'

    sql_query = '''
    SELECT DISTINCT TRADE_DT
    FROM aidx_eod_prices
    WHERE TRADE_DT >= {startdate} AND TRADE_DT <= {enddate}
    AND S_IRDCODE in {assets_tuple}
    '''.format(startdate=int(startdate), enddate=int(enddate), assets_tuple=tuple(assets))
    
    raw_data = db.GetSQL(sql_query, tbl_type='pddf')['TRADE_DT'].astype(str)
    if raw_data.shape[0] <= 1:
        raise LookupError('Data 0 Extraction for table aidx_eod_prices from {start} to {end} among {assets_tuple}'.format(
            start=int(startdate), end=int(enddate), assets_tuple=tuple(assets)
        ))
    # data = raw_data.pivot_table(index='S_IRDCODE', columns=['TRADE_DT'], values=['PCHG'])
    # rtn_data = data.to_numpy(dtype=np.float32)
    # assets_inds = data.index.to_list()
    
    return raw_data




if __name__ == "__main__":
    print( db_date_data(['000001.SH', '000016.SH', '000002.SH', '000009.SH', '000010.SH'], '20230101', '20230103') )
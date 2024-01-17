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
    'path':'Data/asset_allocate/tydb.db'
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



def db_date_data(startdate:str, enddate:str):
    # get trade dates for assets from startdate to enddate
    db = DatabaseConnection(localdb)
    startdate, enddate = int(startdate), int(enddate)
    assert startdate <= enddate, 'start date must no later than end date'

    sql_query = '''
        SELECT DISTINCT TRADE_DT
        FROM dim_trade_date_ashare
        WHERE DATE_FLAG = '1' AND TRADE_DT >= {startdate} AND TRADE_DT <= {enddate}
        ORDER BY TRADE_DT ASC
    '''.format(startdate=startdate, enddate=enddate)

    raw_data = db.GetSQL(sql_query, tbl_type='pddf')['TRADE_DT'].astype(int).to_numpy()
    if len(raw_data) <= 1:
        raise LookupError('Data 0 Extraction for table dim_trade_date_ashare from {start} to {end}'.format(
            start=startdate, end=enddate
        ))
    return raw_data
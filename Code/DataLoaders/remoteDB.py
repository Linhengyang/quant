import pandas as pd 
import numpy as np
# from datetime import datetime, timedelta
from Code.DataLoaders.dbconnect import DatabaseConnection






def db_rtn_data(assets:list, startdate:str, enddate:str, rtn_dilate):
    db = DatabaseConnection()
    assert len(assets) > 0, 'assets index list must not be empty'
    assert int(startdate) <= int(enddate), 'start date must no later than end date'
    sql_query = '''
    SELECT *, {dilate}*rtn as rtn
    FROM table
    WHERE dt >= {startdate} AND dt <= {enddate}
    AND assets in {assets_tuple}
    '''.format(startdate=int(startdate), enddate=int(enddate), assets_tuple=tuple(assets), dilate=int(rtn_dilate))
    raw_data = db.GetSQL(sql_query, tbl_type='pddf')
    data = raw_data.pivot_table(index='assets', columns=['dt'], values=['rtn'])
    rtn_data = data.to_numpy(dtype=np.float32)
    assets_inds = data.index.to_list()
    return rtn_data, assets_inds
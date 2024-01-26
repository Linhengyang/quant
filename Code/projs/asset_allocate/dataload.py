import pandas as pd 
import numpy as np
# from datetime import datetime, timedelta
from Code.DataLoader.dbconnect import DatabaseConnection
from Code.Utils.Sequence import strided_slicing_w_residual

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

def db_rtn_data(
        assets:list,
        startdate:str,
        enddate:str,
        rtn_dilate:int,
        tbl_name:str,
        db_info:dict
        ):
    db = DatabaseConnection(db_info)

    assert len(assets) > 0, 'assets index list must not be empty'
    assert int(startdate) <= int(enddate), 'start date must no later than end date'

    sql_query = '''
    SELECT DISTINCT S_IRDCODE, TRADE_DT, {dilate}*PCHG as PCHG
    FROM {tbl_name}
    WHERE TRADE_DT >= {startdate} AND TRADE_DT <= {enddate}
    AND S_IRDCODE in ("{assets_tuple}")
    '''.format(tbl_name=tbl_name,
               startdate=int(startdate),
               enddate=int(enddate),
               assets_tuple='","'.join(assets),
               dilate=int(rtn_dilate)
               )
    
    raw_data = db.GetSQL(sql_query, tbl_type='pddf')

    if raw_data.shape[0] <= 1:
        raise LookupError(
            f'Data 0 Extraction for table {tbl_name} from {startdate} to {enddate} among {tuple(assets)}'
            )
    
    data = raw_data.pivot_table(index='S_IRDCODE',
                                columns=['TRADE_DT'],
                                values=['PCHG'])
    
    rtn_data = data.to_numpy(dtype=np.float32)
    assets_ids = data.index.to_list()
    
    return rtn_data, assets_ids



def db_rtn_data_multi_tbl(
        assets:list,
        startdate:str,
        enddate:str,
        rtn_dilate:int,
        tbl_names:list,
        db_info:dict
        ):
    db = DatabaseConnection(db_info)

    assert len(assets) > 0, 'assets index list must not be empty'
    assert int(startdate) <= int(enddate), 'start date must no later than end date'
    assert len(assets) == len(tbl_names), 'assets must be listed in different lists based on\
                                           their originated tables of tbl_names'

    # assets =    [["000001.SH", "000002.SH"], ["CBA0001.CB", "CBA00002.CB"]]
    # tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
    sql_queries = []
    for asset_lst, tbl_name in zip(assets, tbl_names):
        sql_query = '''
        SELECT DISTINCT S_IRDCODE, TRADE_DT, {dilate}*PCHG as PCHG
        FROM {tbl_name}
        WHERE TRADE_DT >= {startdate} AND TRADE_DT <= {enddate}
        AND S_IRDCODE in ("{assets_tuple}")
        '''.format(tbl_name=tbl_name,
                   startdate=int(startdate),
                   enddate=int(enddate),
                   assets_tuple='","'.join(asset_lst),
                   dilate=int(rtn_dilate)
                   )
        sql_queries.append(sql_query)

    query = "/n union /n".join(sql_queries)

    raw_data = db.GetSQL(query, tbl_type='pddf')

    if raw_data.shape[0] <= 1:
        raise LookupError(
            f'0 Extraction for tables {tuple(tbl_names)} from {startdate} to {enddate}\
              among {tuple([tuple(i) for i in assets])}'
            )
    
    data = raw_data.pivot_table(index='S_IRDCODE',
                                columns=['TRADE_DT'],
                                values=['PCHG'])
    
    rtn_data = data.to_numpy(dtype=np.float32)
    assets_ids = data.index.to_list()
    
    return rtn_data, assets_ids


def db_date_data(
        startdate:str,
        enddate:str,
        db_info:dict
        ):
    # get trade dates for assets from startdate to enddate
    db = DatabaseConnection(db_info)

    startdate, enddate = int(startdate), int(enddate)
    assert startdate <= enddate, 'start date must no later than end date'

    sql_query = '''
        SELECT DISTINCT TRADE_DT
        FROM dim_trade_date_ashare
        WHERE DATE_FLAG = '1' AND TRADE_DT >= {startdate} AND TRADE_DT <= {enddate}
        ORDER BY TRADE_DT ASC
    '''.format(startdate=startdate,
               enddate=enddate
               )

    raw_data = db.GetSQL(sql_query, tbl_type='pddf')['TRADE_DT'].astype(int).to_numpy()

    if len(raw_data) <= 1:
        raise LookupError(
            f'Data 0 Extraction for table dim_trade_date_ashare from {startdate} to {enddate}'
            )
    
    return raw_data



# get data for train and backtest
def get_train_rtn_data(
        begindate:str,
        termidate:str,
        gapday:int,
        back_window_size:int,
        dilate:int,
        assets_ids:list,
        tbl_names:str,
        db_info:dict=localdb
        ):
    '''
    assets_ids & tbl_names:
    1. if all assets come from 1 table, then arg {tbl_names} is the string of the table,
        assets_ids is a list of asset id codes

        e.g, assets_ids = ['000001.SH', '000002.SH'], tbl_names = 'aidx_eod_prices'
    2. if assets come from multiple tables, then arg {tbl_names} is the string of tables,
        assets_ids is a list of lists of asset id code 
        which come from corresponding table name by order.

        e.g, assets_ids = [['000001.SH', '000002.SH'], ['CBA0001.CBI']],
        tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
    '''
    # 取数据，一次io解决
    # 取出2000-01-01至终止日, 所有的交易日期，已排序
    all_mkt_dates = db_date_data('20000101',
                                 termidate,
                                 db_info) # 返回numpy of int
    # 涉及到的最早的date，是begindate往前数 back_window_size 个交易日的日期。因为begindate当日早上完成调仓，需要前一天至前back_window_size天
    
    begindate_idx = np.where(all_mkt_dates==int(begindate))[0].item()
    earlistdate_idx = begindate_idx-back_window_size
    if earlistdate_idx < 0:
        raise ValueError(
            f'Traceback earlier than 2000-01-01 from {begindate} going back with {back_window_size} days'
            )
    
    # 裁剪掉前面无用的日期
    all_mkt_dates = all_mkt_dates[earlistdate_idx:]
    earlistdate = all_mkt_dates[0] # 最早天数是第一天
    begindate_idx -= earlistdate_idx #
    earlistdate_idx = 0
    
    # all_rtn_data shape: (num_assets, begindate - back_window_size to begindate to termidate)
    # which is back_window_size + num_period_days_from_begin_to_termi
    if isinstance(assets_ids[0], list): # assets_ids = [['000001.SH', '000002.SH'], ['CBA0001.CB' ,'CBA0002.CB']]
        assert isinstance(tbl_names, list), "arg tbl_names must be a list for different tables\
                                             for function get_train_rtn_data"
        all_rtn_data, assets_idlst = db_rtn_data_multi_tbl(assets_ids,
                                                           str(earlistdate),
                                                           termidate, dilate,
                                                           tbl_names,
                                                           db_info
                                                           )
    else: # assets_ids = ['000001.SH', '000002.SH']
        assert isinstance(tbl_names, str), "arg tbl_names must be string for function get_train_rtn_data"
        all_rtn_data, assets_idlst = db_rtn_data(assets_ids,
                                                 str(earlistdate),
                                                 termidate,
                                                 dilate,
                                                 tbl_names,
                                                 db_info
                                                 )
    assert all_rtn_data.shape[1] == len(all_mkt_dates),\
        f'market dates with length {len(all_mkt_dates)} and Index return dates {all_rtn_data.shape[1]} mismatch'
    
    rtn_data = all_rtn_data[:, begindate_idx:] # 从 all_rtn_data 中，取出 begindate到termidate的列

    # 每一期持仓起始，往后持仓gapday天或最后一天
    strided_slices, _, last_range = strided_slicing_w_residual(rtn_data.shape[1],
                                                               gapday,
                                                               gapday
                                                               )
    
    hold_rtn_mat_list = list(rtn_data.T[strided_slices].transpose(0,2,1))
    if list(last_range): # rsd_range不为空
        hold_rtn_mat_list.append( rtn_data.T[last_range].T )
    
    # 每一期调仓日期起始，往前回溯back_window_size天。调仓日期在持仓日之前
    strided_slices, _, _ = strided_slicing_w_residual(all_rtn_data.shape[1]-1,
                                                      back_window_size,
                                                      gapday
                                                      )

    train_rtn_mat_list = list(all_rtn_data.T[strided_slices].transpose(0,2,1))

    assert len(train_rtn_mat_list) == len(hold_rtn_mat_list), 'train & hold period mismatch error. Please check code'

    return train_rtn_mat_list, hold_rtn_mat_list, assets_idlst
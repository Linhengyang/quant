import csv
import sqlite3
import pandas as pd


def csv2sqliteDB_pd(csv_path, db_path, tbl_name, encoding='utf-8'):

    conn = sqlite3.connect(db_path)
    with conn:
        df = pd.read_csv(csv_path)
        df['S_IRDNAME'] = df['S_IRDNAME'].astype(str).apply(lambda x: x.encode(encoding))
        df.to_sql(tbl_name, conn, if_exists='append', index=False)
    print('succeed')


# 打印csv文件的前几行
def printcsv(csv_path, print_numb=5, header=True, encoding='utf-8'):
    with open(csv_path, encoding=encoding) as csv_f:
        data = csv.reader(csv_f)
        r_numb = 0
        for row in data:
            print(row)
            print("-"*150)
            r_numb = r_numb + 1
            if r_numb >= print_numb:
                break


# csv文件输出到sqlite database文件
def csv2sqliteDB(csv_path, db_path, tbl_name, dtypes, headers=True, encoding='utf-8'):
    '''
    csv_path: .csv文件地址
    db_path: sqlite database .db文件地址
    tbl_name: 表名
    dtypes: list of "REAL", "INTEGER", "TEXT", "BLOB"
    headers:
        default True if first line of csv file is the column names
        else:
            list of column names
    '''
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    if isinstance(headers, list):
        assert len(dtypes) == len(headers), 'headers and datatypes must match in length'
    # 确认header
    elif headers:
        with open(csv_path, encoding=encoding) as csv_f:
            data = csv.reader(csv_f)
            for colnames in data:
                break
        headers = colnames
    else:
        raise ValueError('headers shall be True if first line is the header or be a list of column names')
    
    # create table
    cur.execute('''CREATE TABLE {tbl_name} ('''.format(tbl_name=tbl_name) + 
    ','.join([header + ' ' + dtype + '\n' for header, dtype in zip(headers, dtypes)]) + 
    ''')
    ''')
    with open(csv_path, encoding=encoding) as csv_f:
        if headers: # 当第一行为列名时，跳过第一行
            next(csv_f)
        row_numb = 0 # 行计数器
        for row in csv.reader(csv_f):
            row_numb += 1
            cur.execute('''INSERT INTO {tbl_name} '''.format(tbl_name=tbl_name) + '(' + 
                        ','.join(headers) + ')' + ' values ' + '(' + ','.join(['?']*len(row)) + ')', row)
    
    con.commit()
    con.close()

    print(
        f'{row_numb} rows of table {tbl_name} in {db_path} written successfully from {csv_path}'
        )
    















if __name__ == '__main__':
    csv_path = 'asset_allocate/cbidx2023.csv'
    db_path = 'asset_allocate/tydb.db'
    tbl_name = 'cbidx_eod_prices'
    # dtypes = ['INT', 'INT', 'INT', 'INT', 'TEXT', 'INT', 'TEXT', 'INT', 'TEXT', 'INT', 'INT', 'INT',]
    dtypes = ['INT', 'TEXT', 'TEXT', 'INT', 'TEXT', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL']
    # printcsv(csv_path, print_numb=5, header=True, encoding='utf-8-sig')
    csv2sqliteDB(csv_path, db_path, tbl_name, dtypes=dtypes, headers=True, encoding='utf-8-sig')
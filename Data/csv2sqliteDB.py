import csv
import sqlite3
import pandas as pd

def csv2sqliteDB(csv_path, db_path, tbl_name):

    conn = sqlite3.connect(db_path)
    with conn:
        df = pd.read_csv(csv_path)
        df.to_sql(tbl_name, conn, if_exists='append', index=False)
    print('succeed')






if __name__ == '__main__':
    csv_path = 'AIDX.csv'
    db_path = 'aidx.db'
    tbl_name = 'aidx_eod_prices'
    csv2sqliteDB(csv_path, db_path, tbl_name)
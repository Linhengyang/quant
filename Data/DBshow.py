import sqlite3

localdb = {
    "path":"asset_allocate/tydb.db"
}




def LocalDBshow(localdb, tbl_name, num_rows = 10):

    conn = sqlite3.connect(localdb['path'])
    cursor = conn.cursor()

    sql_query = \
    '''
    SELECT *
    FROM {tbl_name}
    LIMIT {num_rows}
    '''.format(
        tbl_name=tbl_name,
        num_rows=num_rows
        )

    cursor.execute(sql_query)
    data = cursor.fetchall()

    col_des = cursor.description
    # col_names = [col_des[i][0] for i in range(len(col_des))]
    cursor.close()
    conn.close()

    for row in data:
        print(row)
    
    print("description:", col_des)



















if __name__ == "__main__":
    # LocalDBshow(localdb, "aidx_eod_prices")
    LocalDBshow(localdb, "dim_trade_date_ashare")
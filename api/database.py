# STEP 1
import pymysql
import os

def send_query(query:str):
    # STEP 2: MySQL Connection 연결
    con = pymysql.connect(host=os.environ['MYSQLHOST'], user=os.environ['MYSQLUSER'], password=os.environ['MYSQLPASS'], port=os.environ['MYSQLPORT'],
                        db=os.environ['MYSQLDB'], charset='utf8') # 한글처리 (charset = 'utf8')



    # STEP 3: Connection 으로부터 Cursor 생성
    cur = con.cursor()
    
    # STEP 4: SQL문 실행 및 Fetch
    sql = query
    cur.execute(sql)

    # STEP 5: DB 연결 종료
    con.close()
    
    # 데이타 Fetch
    rows = cur.fetchall()
    print(rows)     # 전체 rows

    return rows



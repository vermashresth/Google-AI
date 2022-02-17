import mysql.connector
from mysql.connector.constants import ClientFlag
import pandas as pd
config = {
    'user': 'googleai',
    'password': '4UY(@{SqH{',
    'host': '34.93.237.61',
    'database': 'mmitrav2'
}

    
# now we establish our connection
cnxn = mysql.connector.connect(**config)
cursor = cnxn.cursor()
import sys


query = "INSERT INTO ai_gargi (ai_id,entrydatetime) VALUES (%s,%s)"
val = (int(sys.argv[1]),sys.argv[2])
cursor.execute(query, val)
cnxn.commit()


cnxn.close()
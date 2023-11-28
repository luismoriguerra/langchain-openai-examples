from dotenv import load_dotenv
load_dotenv()

import os
import snowflake.connector
import pandas as pd

def execute_df_query(sql):
    connection_params = {
        'user': os.environ["SF_USER"],
        'password': os.environ["SF_PASSWORD"],
        'account': os.environ["SF_ACCOUNT"],
        # 'warehouse': os.environ["SNOWFLAKE_WAREHOUSE"],
        'database': os.environ["SF_DATABASE"],
        'schema': os.environ["SF_SCHEMA"],
    }
    
    query=sql
    
    try:
        conn = snowflake.connector.connect(**connection_params)
        
        cur = conn.cursor()
        
        try:
            cur.execute(query)
        except snowflake.connector.errors.ProgrammingError as pe:
            print("query compilation error: " + pe)
            return("query compilation error")
        
        query_results = cur.fetchall()
        
        column_names = [x[0] for x in cur.description]
        
        data_frame = pd.DataFrame(query_results, columns=column_names)
        
        return data_frame
    
    except snowflake.connector.errors.ProgrammingError as de:
        print("snowflake connection error: " + de)
        
    except Exception as e:
        print("unexpected error: " + e)
        
    finally:
        try:
            cur.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass
        

if __name__ == "__main__":
    query = """
    SELECT * FROM FIVETRAN_INGEST.CROWD_PROD_PUBLIC.ACTIVITIES LIMIT 1
    """
    df = execute_df_query(query)
    print(df.head())
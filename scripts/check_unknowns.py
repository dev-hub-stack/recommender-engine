import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('PG_HOST'),
        database=os.getenv('PG_DB'),
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASSWORD'),
        sslmode='require' # or just remove if needed, but safe to keep
    )

def check_unknowns():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        print('--- Top Cities in Unknown Province ---')
        cur.execute("""
            SELECT 
                COALESCE(customer_city, 'NULL_CITY') as city, 
                COUNT(*) as count
            FROM orders 
            WHERE 
                province IS NULL 
                OR UPPER(province) IN ('UNKNOWN', 'OTHER', 'N/A')
                OR province = ''
            GROUP BY city
            ORDER BY count DESC
            LIMIT 500;
        """)
        
        rows = cur.fetchall()
        for r in rows:
            print(f'{r[0]}: {r[1]}')
        
        conn.close()
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    check_unknowns()

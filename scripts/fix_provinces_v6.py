import psycopg2
import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Extended map based on diagnostic query
NEW_MAPPINGS = {
    "sanghar": "Sindh",
    "risalpur": "Khyber Pakhtunkhwa",
    "kandh kot": "Sindh",
    "kandhkot": "Sindh",
    "sara - e - alamgir": "Punjab",
    "sarai alamgir": "Punjab",
    "depal pur": "Punjab",
    "depalpur": "Punjab",
    "abdul hakim": "Punjab",
    "hasilpur": "Punjab",
    "jahanian": "Punjab",
    "chunian": "Punjab",
    "daharki": "Sindh",
    "mian chanoo": "Punjab",
    "mian channu": "Punjab",
    "muzafarabad": "Azad Kashmir",
    "muzaffarabad": "Azad Kashmir",
    "kahuta": "Punjab",
    "thul": "Sindh",
    "farooqabad": "Punjab",
    "kala shah kaku": "Punjab",
    "dera murad jamali": "Balochistan",
    "liaquatpur": "Punjab",
    "talagang": "Punjab",
    "ali pur chattha": "Punjab",
    "laki marwat": "Khyber Pakhtunkhwa",
    "lakki marwat": "Khyber Pakhtunkhwa",
    "chashma": "Punjab",
    "khuddian khas": "Punjab",
    "zafarwal": "Punjab",
    "basirpur": "Punjab",
    "chowk azam": "Punjab",
    "jalalpur jattan": "Punjab",
    "pasrur": "Punjab",
    "yazman mandi": "Punjab",
    "sakrand": "Sindh",
    "jalal pur pirwala": "Punjab",
    "narang mandi": "Punjab",
    "alipur": "Punjab",
    "abottabad": "Khyber Pakhtunkhwa",
    "abbottabad": "Khyber Pakhtunkhwa",
    "shakar garh": "Punjab",
    "jehlum": "Punjab",
    "jhelum": "Punjab",
    "lalamusa": "Punjab",
    "ranipur": "Sindh",
    "tarbela": "Khyber Pakhtunkhwa",
    "fateh jang": "Punjab",
    "dina": "Punjab",
    "pindi bhatian": "Punjab",
    "malakwal": "Punjab",
    "kallar saidan": "Punjab"
}

def fix_provinces():
    try:
        conn = psycopg2.connect(
            host=os.getenv('PG_HOST'),
            database=os.getenv('PG_DB'),
            user=os.getenv('PG_USER'),
            password=os.getenv('PG_PASSWORD'),
            sslmode='require'
        )
        cur = conn.cursor()
        
        logger.info("Starting province backfill v6...")
        total_updated = 0
        
        for city, province in NEW_MAPPINGS.items():
            cur.execute("""
                UPDATE orders 
                SET province = %s 
                WHERE LOWER(TRIM(customer_city)) = %s 
                AND (province IS NULL OR province = '' OR province = 'Unknown')
            """, (province, city))
            count = cur.rowcount
            if count > 0:
                logger.info(f"Updated {count} orders for {city} -> {province}")
                total_updated += count
        
        conn.commit()
        logger.info(f"Total orders updated: {total_updated}")
        conn.close()
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    fix_provinces()

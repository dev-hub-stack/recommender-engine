import psycopg2
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

NEW_MAPPINGS = {
    "chak jhumra": "Punjab",
    "pannu akil": "Sindh",
    "pano aqil": "Sindh",
    "usta muhammad": "Balochistan",
    "tandlianwala": "Punjab",
    "pakpattan": "Punjab",
    "digri": "Sindh",
    "theeng more": "Punjab",
    "tando muhammad khan": "Sindh",
    "tando mohd khan": "Sindh",
    "khurian wala": "Punjab",
    "khurrianwala": "Punjab",
    "shahdadpur": "Sindh",
    "kamber ali khan": "Sindh",
    "master head office": "Punjab",
    "johi": "Sindh",
    "arif wala": "Punjab",
    "arifwala": "Punjab",
    "shor kot": "Punjab",
    "shorkot": "Punjab",
    "sharaqpur shareef": "Punjab",
    "qila deedar singh": "Punjab",
    "kunri": "Sindh",
    "mehmoodkot": "Punjab",
    "pir mahal": "Punjab",
    "pirmahal": "Punjab",
    "mithi": "Sindh",
    "fatehpur": "Punjab",
    "topi": "Khyber Pakhtunkhwa",
    "rato dero": "Sindh",
    "mian channun": "Punjab",
    "joharabad": "Punjab",
    "jauharabad": "Punjab",
    "kotla arab ali khan": "Punjab",
    "tank": "Khyber Pakhtunkhwa",
    "chenab nagar": "Punjab",
    "bhawana": "Punjab",
    "manga mandi": "Punjab",
    "kallar kahar": "Punjab",
    "shahkot": "Punjab",
    "kamoki": "Punjab",
    "kamoke": "Punjab",
    "hari pur": "Khyber Pakhtunkhwa",
    "haripur": "Khyber Pakhtunkhwa",
    "khudiyan khas": "Punjab",
    "dunyapur": "Punjab",
    "daud khel": "Punjab",
    "hujra shah muqeem": "Punjab",
    "tando jan muhammad": "Sindh",
    "kotri": "Sindh",
    "dinga": "Punjab",
    "pind dadan khan": "Punjab",
    "khewra dandot": "Punjab",
    "sharqpur": "Punjab",
    "chawinda": "Punjab",
    "shabqadar": "Khyber Pakhtunkhwa",
    "gawadar": "Balochistan",
    "gwadar": "Balochistan",
    "umer kot": "Sindh",
    "umerkot": "Sindh",
    "guddu": "Sindh",
    "timargera": "Khyber Pakhtunkhwa",
    "timergara": "Khyber Pakhtunkhwa",
    "warburton": "Punjab",
    "ahmed pur sial": "Punjab",
    "shinkiari": "Khyber Pakhtunkhwa",
    "safdarabad": "Punjab",
    "haroon abad": "Punjab",
    "haroonabad": "Punjab",
    "pindi bhattian": "Punjab",
    "kot mithan": "Punjab",
    "mandi faizabad": "Punjab",
    "buchekey": "Punjab",
    "bucheki": "Punjab",
    "samandri": "Punjab",
    "samundri": "Punjab",
    "kabir wala": "Punjab",
    "kabirwala": "Punjab",
    "noushehro feroz": "Sindh",
    "naushahro firoz": "Sindh",
    "sui": "Balochistan",
    "lala musa": "Punjab",
    "naudero": "Sindh",
    "bhera": "Punjab",
    "sujawal": "Sindh",
    "mangla": "Azad Kashmir",
    "jalal pur bhattian": "Punjab",
    "jalalpur bhattian": "Punjab",
    "mingaora": "Khyber Pakhtunkhwa",
    "mingora": "Khyber Pakhtunkhwa",
    "havelian": "Khyber Pakhtunkhwa",
    "changa manga": "Punjab",
    "matli": "Sindh",
    "ubauro": "Sindh",
    "kot momin": "Punjab",
    "gago mandi": "Punjab",
    "gaggo mandi": "Punjab",
    "haveli lakha": "Punjab",
    "badiana": "Punjab",
    "rajana": "Punjab",
    "pallandri": "Azad Kashmir",
    "nasirabad": "Balochistan",
    "choa saidan shah": "Punjab",
    "kot radha kishan": "Punjab",
    "dadyal ajk": "Azad Kashmir",
    "kahror pacca": "Punjab",
    "faislabad": "Punjab", 
    "faqirwali": "Punjab",
    "islamkot": "Sindh",
    "feroz watowan": "Punjab",
    "fort abbas": "Punjab",
    "shahdad kot": "Sindh",
    "rwp": "Punjab",
    "kalat": "Balochistan",
    "shinkayari": "Khyber Pakhtunkhwa",
    "uthal": "Balochistan",
    "dera allahyar": "Balochistan",
    "18 hazari city": "Punjab",
    "jhuddo": "Sindh",
    "sangla hill": "Punjab",
    "khi": "Sindh",
    "hydrabad": "Sindh"
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
        
        logger.info("Starting province backfill v7...")
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

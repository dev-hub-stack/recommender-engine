import psycopg2
from psycopg2.extras import RealDictCursor
import re
import os
from dotenv import load_dotenv

# Load env file to get DB credentials
load_dotenv('.env')

PG_HOST = os.getenv('PG_HOST', 'localhost')
PG_PORT = os.getenv('PG_PORT', '5432')
PG_DB = os.getenv('PG_DB', 'mastergroup_recommendations')
PG_USER = os.getenv('PG_USER', 'postgres')
PG_PASS = os.getenv('PG_PASSWORD', 'postgres')

def clean_phone(phone_val):
    if phone_val is None:
        return None
    phone_str = str(phone_val).strip()
    digits = re.sub(r'[^\d]', '', phone_str)
    if not digits:
        return None
        
    if len(digits) == 10 and digits.startswith('3'):
        standardized = '92' + digits
    elif len(digits) == 11 and digits.startswith('03'):
        standardized = '92' + digits[1:]
    elif len(digits) == 12 and (digits.startswith('923') or digits.startswith('920')):
        if digits.startswith('9203'):
            standardized = '923' + digits[4:]
        else:
            standardized = digits
    else:
        return None
    return '+' + standardized

def clean_city(city_val):
    if city_val is None:
        return None
    city_str = str(city_val).strip()
    if city_str.lower() in ('nan', 'none', ''):
        return None
    return city_str.title()

def clean_email(email_val):
    if email_val is None:
        return None
    email_str = str(email_val).strip().lower()
    if email_str in ('nan', 'none', ''):
        return None
    return email_str

def dry_run():
    print(f"Connecting to {PG_DB} at {PG_HOST}...")
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASS
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Check orders table
        print("\n--- Orders Table Dry Run ---")
        cur.execute("SELECT id, customer_phone, customer_city, customer_email, unified_customer_id, customer_name FROM orders LIMIT 1000")
        orders = cur.fetchall()
        
        phone_updates = 0
        city_updates = 0
        email_updates = 0
        invalid_phones = 0
        
        for order in orders:
            # Phone testing
            orig_phone = order['customer_phone']
            new_phone = clean_phone(orig_phone)
            if orig_phone != new_phone:
                if new_phone is None and orig_phone is not None:
                    invalid_phones += 1
                else:
                    phone_updates += 1
            
            # City testing
            orig_city = order['customer_city']
            new_city = clean_city(orig_city)
            if orig_city != new_city:
                city_updates += 1
                
            # Email testing
            orig_email = order['customer_email']
            new_email = clean_email(orig_email)
            if orig_email != new_email:
                email_updates += 1
                
        print(f"Sample Size: {len(orders)} orders")
        print(f"  Phones needing update: {phone_updates} (Found {invalid_phones} invalid phones)")
        print(f"  Cities needing update: {city_updates}")
        print(f"  Emails needing update: {email_updates}")
        
        # Example conversions
        print("\nExamples of phone conversions:")
        examples_shown = 0
        for order in orders:
            orig_phone = order['customer_phone']
            new_phone = clean_phone(orig_phone)
            if orig_phone != new_phone and new_phone is not None and examples_shown < 5:
                print(f"  {orig_phone} -> {new_phone}")
                examples_shown += 1
                
        # 2. Check customer_statistics
        print("\n--- Customer Statistics Table Dry Run ---")
        cur.execute("SELECT customer_id, customer_city FROM customer_statistics LIMIT 500")
        stats = cur.fetchall()
        
        stat_city_updates = 0
        for stat in stats:
            orig_city = stat['customer_city']
            new_city = clean_city(orig_city)
            if orig_city != new_city:
                stat_city_updates += 1
                
        print(f"Sample Size: {len(stats)} customer stats")
        print(f"  Cities needing update: {stat_city_updates}")

        print("\nDry run completed successfully. No changes were saved to the database.")
        
    except Exception as e:
        print(f"Error connecting to DB: {e}")
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()

if __name__ == "__main__":
    dry_run()

import pandas as pd
import re
import os

def clean_phone(phone_val):
    if pd.isna(phone_val):
        return None
    
    # Convert to string and strip spaces
    phone_str = str(phone_val).strip()
    
    # Remove all non-numeric characters
    digits = re.sub(r'[^\d]', '', phone_str)
    
    if not digits:
        return None
        
    # Standardize to +923XXXXXXXXX
    if len(digits) == 10 and digits.startswith('3'):
        standardized = '92' + digits
    elif len(digits) == 11 and digits.startswith('03'):
        standardized = '92' + digits[1:]
    elif len(digits) == 12 and (digits.startswith('923') or digits.startswith('920')):
        if digits.startswith('9203'):
            # handle cases like 9203001234567 -> 923001234567
            standardized = '923' + digits[4:]
        else:
            standardized = digits
    else:
        # Invalid or non-standard Pakistani mobile number
        return None
        
    return '+' + standardized

def get_data_completeness_score(row):
    score = 0
    if pd.notna(row.get('CustomerName')) and str(row.get('CustomerName')).strip() != '' and str(row.get('CustomerName')).strip().lower() != 'nan':
        score += 2
    if pd.notna(row.get('EmailAddress')) and str(row.get('EmailAddress')).strip() != '' and str(row.get('EmailAddress')).strip().lower() != 'nan' and str(row.get('EmailAddress')).strip().lower() != 'none':
        score += 3
    if pd.notna(row.get('CustomerAddress')) and str(row.get('CustomerAddress')).strip() != '' and str(row.get('CustomerAddress')).strip().lower() != 'nan':
        score += 1
    return score

def main():
    input_file = 'docs/CustomerDataMasterVerse.xlsx'
    output_file = 'docs/CustomerDataMasterVerse_Cleaned.csv'
    
    print(f"Loading data from {input_file}...")
    df = pd.read_excel(input_file)
    initial_count = len(df)
    print(f"Initial record count: {initial_count}")
    
    # 1. Phone Normalization
    print("Normalizing phone numbers...")
    df['normalized_phone'] = df['Mobileno'].apply(clean_phone)
    
    invalid_phones = df['normalized_phone'].isna().sum()
    print(f"Found {invalid_phones} records with invalid/unparsable phone numbers.")
    
    df = df.dropna(subset=['normalized_phone'])
    
    # 2. City Standardization
    print("Standardizing city names...")
    df['CityName'] = df['CityName'].astype(str).str.title().str.strip()
    df['CityName'] = df['CityName'].replace('Nan', pd.NA)
    
    # 3. Email Standardization
    print("Standardizing email addresses...")
    df['EmailAddress'] = df['EmailAddress'].astype(str).str.lower().str.strip()
    df.loc[df['EmailAddress'].isin(['nan', 'none']), 'EmailAddress'] = pd.NA
    
    # 4. Deduplication
    print("Deduplicating records...")
    df['completeness_score'] = df.apply(get_data_completeness_score, axis=1)
    
    # Sort by phone and completeness score descending
    df = df.sort_values(by=['normalized_phone', 'completeness_score'], ascending=[True, False])
    
    # Drop duplicates keeping the first (highest score)
    df_dedup = df.drop_duplicates(subset=['normalized_phone'], keep='first').copy()
    
    final_count = len(df_dedup)
    print(f"Deduplicated record count: {final_count} (Dropped {len(df) - final_count} duplicates)")
    
    df_dedup = df_dedup.drop(columns=['completeness_score'])
    df_dedup['Mobileno'] = df_dedup['normalized_phone']
    df_dedup = df_dedup.drop(columns=['normalized_phone'])
    
    print(f"Saving cleaned dataset to {output_file}...")
    df_dedup.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()

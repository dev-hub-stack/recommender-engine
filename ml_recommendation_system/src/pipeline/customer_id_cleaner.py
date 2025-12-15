"""
Customer ID Cleaner Module
Cleans and normalizes customer phone numbers and emails
"""
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerIDCleaner:
    """Clean and normalize customer identifiers"""
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'valid_phone': 0,
            'email_fallback': 0,
            'removed_invalid': 0,
            'phone_normalized': 0,
            'test_numbers_removed': 0
        }
    
    def clean_phone_number(self, phone):
        """
        Clean and normalize phone number
        
        Examples:
            "+92 300 4567890" → "03004567890"
            "0300-456-7890"   → "03004567890"
            "(0300) 4567890"  → "03004567890"
            "+1 (282) 468-4701" → "12824684701"
        
        Args:
            phone: Raw phone number
            
        Returns:
            Cleaned phone number or None
        """
        if pd.isna(phone) or phone == '':
            return None
        
        # Convert to string and strip
        phone = str(phone).strip()
        
        # Remove all non-digit characters
        digits_only = re.sub(r'[^0-9]', '', phone)
        
        if not digits_only:
            return None
        
        # Handle Pakistani international format (+92)
        if digits_only.startswith('92') and len(digits_only) == 12:
            # +92 300 4567890 → 03004567890
            digits_only = '0' + digits_only[2:]
            self.stats['phone_normalized'] += 1
        
        # Handle missing leading zero for Pakistani numbers
        elif len(digits_only) == 10 and digits_only[0] in ['3', '4', '5']:
            # 3004567890 → 03004567890
            digits_only = '0' + digits_only
            self.stats['phone_normalized'] += 1
        
        # Remove extra leading zeros
        elif len(digits_only) == 12 and digits_only.startswith('00'):
            # 003004567890 → 03004567890
            digits_only = digits_only[1:]
            self.stats['phone_normalized'] += 1
        
        return digits_only
    
    def is_valid_phone(self, phone):
        """
        Check if phone number is valid
        
        Valid criteria:
        - Between 10-15 digits
        - All digits
        - Not a test/placeholder number
        
        Args:
            phone: Cleaned phone number
            
        Returns:
            Boolean
        """
        if not phone:
            return False
        
        # Check length
        if len(phone) < 10 or len(phone) > 15:
            return False
        
        # Must be all digits
        if not phone.isdigit():
            return False
        
        return True
    
    def is_test_number(self, phone):
        """
        Detect test/fake/placeholder numbers
        
        Args:
            phone: Cleaned phone number
            
        Returns:
            Boolean (True if test number)
        """
        if not phone:
            return True
        
        # Check for all same digit
        if len(set(phone)) == 1:
            # "00000000000", "11111111111"
            return True
        
        # Check for obvious test patterns
        test_patterns = [
            '00000000000',
            '11111111111',
            '22222222222',
            '03000000000',
            '03111111111',
            '01111111111',
            '03001234567',
            '01234567890',
            '12345678901'
        ]
        
        if phone in test_patterns:
            return True
        
        # Check for sequential numbers (1234567890)
        if phone == ''.join(str(i) for i in range(10)):
            return True
        
        return False
    
    def is_valid_email(self, email):
        """
        Check if email is valid
        
        Args:
            email: Email address
            
        Returns:
            Boolean
        """
        if pd.isna(email) or email == '':
            return False
        
        email = str(email).strip().lower()
        
        # Basic format check
        if '@' not in email or '.' not in email:
            return False
        
        # Must have at least one character before @
        if email.startswith('@'):
            return False
        
        # Check for test/invalid emails
        invalid_patterns = [
            'test@test',
            'admin@admin',
            'example@example',
            'noreply@',
            'no-reply@',
            'donotreply@',
            '@test.com',
            '@example.com',
            '@shopdev.co',
            'test@',
            '@test',
            'dummy@',
            'fake@'
        ]
        
        for pattern in invalid_patterns:
            if pattern in email:
                return False
        
        # Split and validate parts
        parts = email.split('@')
        if len(parts) != 2:
            return False
        
        username, domain = parts
        
        # Username must be at least 2 characters
        if len(username) < 2:
            return False
        
        # Domain must have at least one dot and be at least 3 characters
        if '.' not in domain or len(domain) < 3:
            return False
        
        return True
    
    def create_customer_id(self, row):
        """
        Create clean customer ID with fallback logic
        
        Priority:
        1. Valid phone number (Pakistani or international)
        2. Valid email (if phone is invalid)
        3. None (order will be removed)
        
        Args:
            row: DataFrame row with customer_phone and customer_email
            
        Returns:
            Clean customer ID or None
        """
        self.stats['total_processed'] += 1
        
        phone = row.get('customer_phone')
        email = row.get('customer_email')
        
        # Try to clean phone
        cleaned_phone = self.clean_phone_number(phone)
        
        # Check if valid phone
        if cleaned_phone and self.is_valid_phone(cleaned_phone):
            # Check if not a test number
            if not self.is_test_number(cleaned_phone):
                self.stats['valid_phone'] += 1
                return cleaned_phone
            else:
                self.stats['test_numbers_removed'] += 1
        
        # Fallback to email
        if self.is_valid_email(email):
            self.stats['email_fallback'] += 1
            return str(email).strip().lower()
        
        # No valid identifier
        self.stats['removed_invalid'] += 1
        return None
    
    def clean_dataframe(self, df):
        """
        Clean customer IDs in entire DataFrame
        
        Args:
            df: DataFrame with customer_phone and customer_email columns
            
        Returns:
            DataFrame with cleaned customer_id column
        """
        logger.info("Starting customer ID cleaning...")
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'valid_phone': 0,
            'email_fallback': 0,
            'removed_invalid': 0,
            'phone_normalized': 0,
            'test_numbers_removed': 0
        }
        
        # Create clean customer IDs
        df['customer_id_clean'] = df.apply(self.create_customer_id, axis=1)
        
        # Log statistics
        logger.info("\n" + "="*60)
        logger.info("CUSTOMER ID CLEANING RESULTS")
        logger.info("="*60)
        logger.info(f"Total orders processed: {self.stats['total_processed']:,}")
        logger.info(f"Valid phone numbers: {self.stats['valid_phone']:,} ({self.stats['valid_phone']/self.stats['total_processed']*100:.1f}%)")
        logger.info(f"Phone numbers normalized: {self.stats['phone_normalized']:,}")
        logger.info(f"Email fallback used: {self.stats['email_fallback']:,} ({self.stats['email_fallback']/self.stats['total_processed']*100:.1f}%)")
        logger.info(f"Test numbers removed: {self.stats['test_numbers_removed']:,}")
        logger.info(f"Invalid (will be removed): {self.stats['removed_invalid']:,} ({self.stats['removed_invalid']/self.stats['total_processed']*100:.1f}%)")
        logger.info("="*60)
        
        # Remove rows with no valid customer ID
        df_clean = df[df['customer_id_clean'].notna()].copy()
        
        # Replace old customer_id with clean one
        df_clean['customer_id'] = df_clean['customer_id_clean']
        df_clean = df_clean.drop(columns=['customer_id_clean'])
        
        logger.info(f"\nOrders retained: {len(df_clean):,} / {len(df):,} ({len(df_clean)/len(df)*100:.1f}%)")
        logger.info(f"Orders removed: {len(df) - len(df_clean):,}")
        
        return df_clean
    
    def get_stats(self):
        """Get cleaning statistics"""
        return self.stats

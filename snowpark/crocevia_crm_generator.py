"""
Crocevia CRM Generator - Standalone Snowpark Version

Based on the structure pattern from example_data_gen.sql:
- Class-based architecture with proper initialization
- Session-based data loading from Snowflake
- Efficient pandas operations with preprocessing
- Structured main function with clear data flow

Usage:
    python crocevia_crm_generator.py
"""

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col, when, lit, uniform, rand
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import uuid
import random


class CroceviaCRMGenerator:
    """
    Generates synthetic Crocevia grocery chain CRM data with controlled overlaps
    to Summit Sports CRM for customer identity resolution testing.
    """
    
    def __init__(self, 
                 source_crm_df: pd.DataFrame,
                 target_size: int = 10000,
                 source_sample_size: int = 5000):
        """
        Initialize the CRM generator with source data and parameters.
        
        Args:
            source_crm_df: Source Summit Sports CRM data
            target_size: Number of Crocevia customers to generate
            source_sample_size: Sample size from source for overlaps
        """
        self.source_crm_df = source_crm_df
        self.target_size = target_size
        self.source_sample_size = min(source_sample_size, len(source_crm_df))
        
        # Overlap configuration
        self.TRIPLE_MATCH_PCT = 0.20    # 20% match on all three (email, phone, name)
        self.EMAIL_OVERLAP_PCT = 0.60   # 60% total email matches
        self.PHONE_OVERLAP_PCT = 0.50   # 50% total phone matches  
        self.NAME_OVERLAP_PCT = 0.35    # 35% total name matches
        
        # Data quality configuration
        self.DOB_PRESENT_PCT = 0.40     # 40% have date of birth
        self.POSTAL_PRESENT_PCT = 0.60  # 60% have postal codes
        self.EMAIL_MISSING_PCT = 0.15   # 15% missing emails
        self.PHONE_MISSING_PCT = 0.20   # 20% missing phones
        self.DUPLICATE_PCT = 0.10       # 10% duplicate customers
        
        # French localization data
        self.FRENCH_FIRST_NAMES = [
            'Jean', 'Marie', 'Pierre', 'Sophie', 'Michel', 'Catherine',
            'Philippe', 'Nathalie', 'Alain', 'Isabelle', 'François', 'Sylvie',
            'Bernard', 'Martine', 'Patrick', 'Christine', 'Daniel', 'Françoise',
            'Thierry', 'Monique', 'Laurent', 'Brigitte', 'André', 'Dominique',
            'Nicolas', 'Véronique', 'Stéphane', 'Sandrine', 'Julien', 'Céline'
        ]
        
        self.FRENCH_LAST_NAMES = [
            'Martin', 'Bernard', 'Dubois', 'Thomas', 'Robert', 'Petit',
            'Richard', 'Durand', 'Leroy', 'Moreau', 'Simon', 'Laurent',
            'Lefebvre', 'Michel', 'Garcia', 'David', 'Bertrand', 'Roux',
            'Vincent', 'Fournier', 'Morel', 'Girard', 'Andre', 'Lefevre',
            'Mercier', 'Dupont', 'Lambert', 'Bonnet', 'François', 'Martinez'
        ]
        
        self.FRENCH_EMAIL_DOMAINS = [
            'gmail.com', 'orange.fr', 'free.fr', 'wanadoo.fr', 
            'sfr.fr', 'laposte.net', 'hotmail.fr', 'yahoo.fr'
        ]
        
        # Preprocessing
        self._preprocess_source_data()
        self._create_overlap_mappings()

    def _preprocess_source_data(self):
        """Preprocess source CRM data for efficient lookup."""
        # Sample source data for overlaps
        self.source_sample = self.source_crm_df.sample(
            n=self.source_sample_size, 
            random_state=42
        ).reset_index(drop=True)
        
        # Create lookup arrays for fast access
        self.src_emails = self.source_sample['EMAIL'].dropna().astype(str).values
        self.src_phones = self.source_sample['PHONE'].dropna().astype(str).values
        self.src_first_names = self.source_sample['FIRST_NAME'].dropna().astype(str).values
        self.src_last_names = self.source_sample['LAST_NAME'].dropna().astype(str).values
        
        # Validate we have sufficient data
        if (len(self.src_emails) == 0 or len(self.src_phones) == 0 or 
            len(self.src_first_names) == 0 or len(self.src_last_names) == 0):
            raise ValueError("Source dataset lacks sufficient non-null identity fields")

    def _create_overlap_mappings(self):
        """Create index mappings for controlled overlaps."""
        np.random.seed(42)
        
        # Calculate overlap counts
        self.triple_match_count = int(self.target_size * self.TRIPLE_MATCH_PCT)
        self.email_extra_count = int(self.target_size * self.EMAIL_OVERLAP_PCT) - self.triple_match_count
        self.phone_extra_count = int(self.target_size * self.PHONE_OVERLAP_PCT) - self.triple_match_count
        self.name_extra_count = int(self.target_size * self.NAME_OVERLAP_PCT) - self.triple_match_count
        
        # Create index sets
        all_indices = np.arange(self.target_size)
        
        # Triple matches (exact overlap on all three fields)
        self.triple_indices = np.random.choice(all_indices, size=self.triple_match_count, replace=False)
        
        # Remaining pool for other overlaps
        remaining = np.setdiff1d(all_indices, self.triple_indices)
        
        # Additional overlaps (can intersect with each other but not with triple)
        self.email_indices = np.random.choice(remaining, size=self.email_extra_count, replace=False)
        self.phone_indices = np.random.choice(remaining, size=self.phone_extra_count, replace=False)
        self.name_indices = np.random.choice(remaining, size=self.name_extra_count, replace=False)

    def _generate_synthetic_customer_data(self) -> pd.DataFrame:
        """Generate base synthetic customer data with French localization."""
        np.random.seed(42)
        random.seed(42)
        
        customers = []
        
        for i in range(self.target_size):
            # Generate base synthetic data
            first_name = random.choice(self.FRENCH_FIRST_NAMES)
            last_name = random.choice(self.FRENCH_LAST_NAMES)
            
            # Generate synthetic email
            local = f"{first_name.lower()}.{last_name.lower()}".replace("'", "").replace("ç", "c")
            domain = random.choice(self.FRENCH_EMAIL_DOMAINS)
            email = f"{local}@{domain}"
            
            # Generate French phone number
            area_code = random.choice(['01', '02', '03', '04', '05', '06', '07', '08', '09'])
            phone = f"{area_code} {random.randint(10, 99)} {random.randint(10, 99)} {random.randint(10, 99)} {random.randint(10, 99)}"
            
            # Generate postal code (French format)
            postal_code = f"{random.randint(1000, 95999):05d}" if random.random() < self.POSTAL_PRESENT_PCT else None
            
            # Generate date of birth (18-90 years old)
            if random.random() < self.DOB_PRESENT_PCT:
                birth_year = random.randint(1934, 2006)
                birth_month = random.randint(1, 12)
                birth_day = random.randint(1, 28)  # Safe day range
                date_of_birth = date(birth_year, birth_month, birth_day)
            else:
                date_of_birth = None
            
            customers.append({
                'row_id': i,
                'customer_id': f"CRV-{i:010d}",
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'phone': phone,
                'postal_code': postal_code,
                'date_of_birth': date_of_birth,
                'overlap_type': 'NONE'
            })
        
        return pd.DataFrame(customers)

    def _apply_overlaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply controlled overlaps with source CRM data."""
        df = df.copy()
        
        # Apply triple matches (same source record for consistency)
        if len(self.triple_indices) > 0:
            src_indices = np.random.randint(0, len(self.source_sample), size=len(self.triple_indices))
            for i, src_idx in zip(self.triple_indices, src_indices):
                source_row = self.source_sample.iloc[src_idx]
                df.loc[i, 'email'] = source_row['EMAIL']
                df.loc[i, 'phone'] = source_row['PHONE'] 
                df.loc[i, 'first_name'] = source_row['FIRST_NAME']
                df.loc[i, 'last_name'] = source_row['LAST_NAME']
                df.loc[i, 'overlap_type'] = 'TRIPLE'
        
        # Apply email-only overlaps
        if len(self.email_indices) > 0:
            df.loc[self.email_indices, 'email'] = np.random.choice(
                self.src_emails, size=len(self.email_indices), replace=True
            )
            df.loc[self.email_indices, 'overlap_type'] = 'EMAIL'
        
        # Apply phone-only overlaps
        if len(self.phone_indices) > 0:
            df.loc[self.phone_indices, 'phone'] = np.random.choice(
                self.src_phones, size=len(self.phone_indices), replace=True
            )
            # Update overlap type (might overwrite EMAIL)
            phone_only = np.setdiff1d(self.phone_indices, self.email_indices)
            df.loc[phone_only, 'overlap_type'] = 'PHONE'
            if len(np.intersect1d(self.phone_indices, self.email_indices)) > 0:
                df.loc[np.intersect1d(self.phone_indices, self.email_indices), 'overlap_type'] = 'EMAIL_PHONE'
        
        # Apply name-only overlaps
        if len(self.name_indices) > 0:
            name_pairs = np.column_stack([
                np.random.choice(self.src_first_names, size=len(self.name_indices), replace=True),
                np.random.choice(self.src_last_names, size=len(self.name_indices), replace=True)
            ])
            df.loc[self.name_indices, 'first_name'] = name_pairs[:, 0]
            df.loc[self.name_indices, 'last_name'] = name_pairs[:, 1]
            # Update overlap types appropriately
            name_only = np.setdiff1d(self.name_indices, np.concatenate([self.email_indices, self.phone_indices]))
            df.loc[name_only, 'overlap_type'] = 'NAME'
        
        return df

    def _add_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic data quality issues and duplicates."""
        df = df.copy()
        
        # Add missing emails (outside email overlap sets)
        overlap_email_indices = np.concatenate([self.triple_indices, self.email_indices])
        non_email_overlap = np.setdiff1d(np.arange(len(df)), overlap_email_indices)
        missing_email_count = int(self.EMAIL_MISSING_PCT * len(df))
        if len(non_email_overlap) > 0 and missing_email_count > 0:
            missing_email_indices = np.random.choice(
                non_email_overlap, 
                size=min(missing_email_count, len(non_email_overlap)), 
                replace=False
            )
            df.loc[missing_email_indices, 'email'] = None
        
        # Add missing phones (outside phone overlap sets)
        overlap_phone_indices = np.concatenate([self.triple_indices, self.phone_indices])
        non_phone_overlap = np.setdiff1d(np.arange(len(df)), overlap_phone_indices)
        missing_phone_count = int(self.PHONE_MISSING_PCT * len(df))
        if len(non_phone_overlap) > 0 and missing_phone_count > 0:
            missing_phone_indices = np.random.choice(
                non_phone_overlap,
                size=min(missing_phone_count, len(non_phone_overlap)),
                replace=False
            )
            df.loc[missing_phone_indices, 'phone'] = None
        
        # Add duplicate customers with variations
        duplicate_count = int(self.DUPLICATE_PCT * len(df))
        if duplicate_count > 0:
            source_indices = np.random.choice(len(df), size=duplicate_count, replace=False)
            
            duplicates = []
            for src_idx in source_indices:
                src_row = df.iloc[src_idx].copy()
                src_row['customer_id'] = f"{src_row['customer_id']}_DUP"
                src_row['overlap_type'] = 'DUPLICATE'
                
                # Add slight variations for realism
                if random.random() < 0.5 and src_row['email']:
                    # Modify email slightly
                    email_parts = str(src_row['email']).split('@')
                    if len(email_parts) == 2:
                        src_row['email'] = f"{email_parts[0]}{random.randint(0, 9)}@{email_parts[1]}"
                
                if random.random() < 0.5 and src_row['phone']:
                    # Modify last digit of phone
                    phone_str = str(src_row['phone'])
                    if phone_str:
                        src_row['phone'] = phone_str[:-1] + str(random.randint(0, 9))
                
                duplicates.append(src_row)
            
            # Append duplicates to DataFrame
            df = pd.concat([df, pd.DataFrame(duplicates)], ignore_index=True)
        
        return df

    def generate_crm_data(self) -> pd.DataFrame:
        """
        Main generation method that orchestrates the entire process.
        
        Returns:
            Complete Crocevia CRM DataFrame with overlaps and quality issues
        """
        print(f"Generating {self.target_size} Crocevia CRM records...")
        
        # Step 1: Generate base synthetic data
        print("Generating base synthetic customer data...")
        df = self._generate_synthetic_customer_data()
        
        # Step 2: Apply controlled overlaps
        print("Applying controlled overlaps with source CRM...")
        df = self._apply_overlaps(df)
        
        # Step 3: Add data quality issues
        print("Adding realistic data quality issues...")
        df = self._add_data_quality_issues(df)
        
        # Step 4: Final cleanup and formatting
        df['created_at'] = datetime.now()
        
        # Reorder columns for consistency
        column_order = [
            'customer_id', 'email', 'phone', 'first_name', 'last_name',
            'postal_code', 'date_of_birth', 'overlap_type', 'created_at'
        ]
        df = df[column_order]
        
        print(f"Generated {len(df)} total records (including duplicates)")
        return df

    def validate_results(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate that overlap percentages meet targets."""
        total_records = len(df)
        overlap_counts = df['overlap_type'].value_counts()
        
        # Calculate actual overlap percentages
        triple_pct = overlap_counts.get('TRIPLE', 0) / total_records
        email_pct = (overlap_counts.get('TRIPLE', 0) + overlap_counts.get('EMAIL', 0) + 
                    overlap_counts.get('EMAIL_PHONE', 0)) / total_records
        phone_pct = (overlap_counts.get('TRIPLE', 0) + overlap_counts.get('PHONE', 0) + 
                    overlap_counts.get('EMAIL_PHONE', 0)) / total_records
        name_pct = (overlap_counts.get('TRIPLE', 0) + overlap_counts.get('NAME', 0)) / total_records
        duplicate_pct = overlap_counts.get('DUPLICATE', 0) / total_records
        
        return {
            'total_records': total_records,
            'triple_match_pct': triple_pct,
            'email_overlap_pct': email_pct,
            'phone_overlap_pct': phone_pct,
            'name_overlap_pct': name_pct,
            'duplicate_pct': duplicate_pct
        }


def main(session: snowpark.Session, 
         target_size: int = 10000, 
         source_sample_size: int = 5000,
         writemode: str = "overwrite") -> str:
    """
    Main execution function following the example_data_gen.sql pattern.
    
    Args:
        session: Snowflake Snowpark session
        target_size: Number of Crocevia customers to generate
        source_sample_size: Sample size from source Summit Sports CRM
        writemode: Write mode for output table ('overwrite' or 'append')
    
    Returns:
        Success message
    """
    print("Starting Crocevia CRM generation...")
    
    # Configuration
    SOURCE_TABLE = "SS_101.RAW_CUSTOMER.CUSTOMER_LOYALTY"
    TARGET_TABLE = "CROCEVIA.RAW_DATA.CROCEVIA_CRM"
    
    try:
        # Step 1: Load source CRM data
        print(f"Loading source data from {SOURCE_TABLE}...")
        source_df = session.table(SOURCE_TABLE).to_pandas()
        print(f"Loaded {len(source_df)} source records")
        
        # Step 2: Initialize generator
        print("Initializing CRM generator...")
        generator = CroceviaCRMGenerator(
            source_crm_df=source_df,
            target_size=target_size,
            source_sample_size=source_sample_size
        )
        
        # Step 3: Generate CRM data
        crm_df = generator.generate_crm_data()
        
        # Step 4: Validate results
        print("Validating results...")
        validation = generator.validate_results(crm_df)
        
        print("\n=== Generation Results ===")
        print(f"Total records: {validation['total_records']:,}")
        print(f"Triple match: {validation['triple_match_pct']:.1%}")
        print(f"Email overlap: {validation['email_overlap_pct']:.1%}")
        print(f"Phone overlap: {validation['phone_overlap_pct']:.1%}")
        print(f"Name overlap: {validation['name_overlap_pct']:.1%}")
        print(f"Duplicates: {validation['duplicate_pct']:.1%}")
        
        # Step 5: Write to Snowflake
        print(f"\nWriting results to {TARGET_TABLE}...")
        session.create_dataframe(crm_df).write.mode(writemode).save_as_table(TARGET_TABLE)
        
        return f"Crocevia CRM generation complete! Generated {len(crm_df)} records."
        
    except Exception as e:
        error_msg = f"Error in CRM generation: {str(e)}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    # For standalone execution
    from snowflake.snowpark import Session
    
    # Create session (assumes default connection)
    session = Session.builder.getOrCreate()
    
    try:
        result = main(session, target_size=10000, source_sample_size=5000)
        print(result)
    finally:
        session.close()

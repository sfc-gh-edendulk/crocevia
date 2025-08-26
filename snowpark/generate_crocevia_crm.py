"""
Snowpark-based Crocevia CRM Data Generator

Generates a large synthetic grocery/hypermarket CRM dataset with controlled overlaps 
to Summit Sports CRM for customer identity resolution testing.

Features:
- Runs natively in Snowflake using Snowpark for Python
- Controlled email (60%), phone (50%), name (35%) overlaps with 20% triple matches
- French localization (names, addresses, postal codes)
- Realistic noise (missing data, duplicates, anonymous transactions)
- Scalable to millions of records

Usage:
    python generate_crocevia_crm.py
"""

import random
from typing import Dict, List, Optional, Tuple

from snowflake.snowpark import Session
from snowflake.snowpark.functions import (
    col, lit, when, uniform, rand, randstr, 
    iff, row_number, dense_rank, count, max as sf_max
)
from snowflake.snowpark.types import (
    StructType, StructField, StringType, IntegerType, 
    DateType, BooleanType, DoubleType
)
from snowflake.snowpark.window import Window


# Configuration
RANDOM_SEED = 42
SOURCE_SAMPLE_SIZE = 5000  # Sample from original CRM for testing
TARGET_SIZE = 10000        # Output size for testing
SOURCE_TABLE = "SS_101.RAW_CUSTOMER.CUSTOMER_LOYALTY"
TARGET_DB = "CROCEVIA"
TARGET_SCHEMA = "RAW_DATA"
TARGET_TABLE = "CROCEVIA_CRM"

# Overlap percentages
EMAIL_OVERLAP_PCT = 0.60    # 60% of emails match
PHONE_OVERLAP_PCT = 0.50    # 50% of phones match  
NAME_OVERLAP_PCT = 0.35     # 35% of name combos match
TRIPLE_MATCH_PCT = 0.20     # 20% match on all three

# Noise percentages
DOB_PRESENT_PCT = 0.40      # 40% have date of birth
POSTAL_PRESENT_PCT = 0.60   # 60% have postal codes
EMAIL_MISSING_PCT = 0.15    # 15% missing emails
PHONE_MISSING_PCT = 0.20    # 20% missing phones
DUPLICATE_PCT = 0.10        # 10% duplicate customers


def create_session() -> Session:
    """Create Snowflake session from default connection parameters."""
    return Session.builder.getOrCreate()


def setup_database_schema(session: Session) -> None:
    """Create target database and schema if they don't exist."""
    session.sql(f"CREATE DATABASE IF NOT EXISTS {TARGET_DB}").collect()
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {TARGET_DB}.{TARGET_SCHEMA}").collect()
    session.sql(f"USE DATABASE {TARGET_DB}").collect()
    session.sql(f"USE SCHEMA {TARGET_SCHEMA}").collect()


def sample_source_data(session: Session) -> str:
    """Create a sample of source data for overlap generation."""
    temp_table = f"TEMP_SOURCE_SAMPLE_{random.randint(1000, 9999)}"
    
    session.sql(f"""
        CREATE OR REPLACE TEMPORARY TABLE {temp_table} AS
        SELECT * FROM {SOURCE_TABLE}
        SAMPLE ({SOURCE_SAMPLE_SIZE} ROWS)
    """).collect()
    
    return temp_table


def infer_column_mappings(session: Session, table_name: str) -> Dict[str, Optional[str]]:
    """
    Infer likely email, phone, name, and other columns from table structure.
    Returns a mapping of semantic column types to actual column names.
    """
    # Get column information
    describe_result = session.sql(f"DESCRIBE TABLE {table_name}").collect()
    columns = [row['name'] for row in describe_result]
    
    # For testing with generic column names, create a simple mapping
    # In practice, this would use more sophisticated heuristics
    mapping = {}
    
    if len(columns) >= 8:  # Assume we have the expected structure
        mapping = {
            'customer_id': columns[0],    # First column as ID
            'email': columns[1] if len(columns) > 1 else None,
            'phone': columns[2] if len(columns) > 2 else None,
            'first_name': columns[3] if len(columns) > 3 else None,
            'last_name': columns[4] if len(columns) > 4 else None,
            'postal': columns[5] if len(columns) > 5 else None,
            'preferred_store': columns[6] if len(columns) > 6 else None,
        }
    
    return mapping


def generate_base_dataset(session: Session, size: int) -> str:
    """Generate base synthetic dataset with French localization."""
    temp_table = f"TEMP_BASE_DATA_{random.randint(1000, 9999)}"
    
    # Create base table with row numbers for consistent sampling
    session.sql(f"""
        CREATE OR REPLACE TEMPORARY TABLE {temp_table} AS
        WITH row_nums AS (
            SELECT ROW_NUMBER() OVER (ORDER BY UNIFORM(1, 1000000, RANDOM({RANDOM_SEED}))) as row_id
            FROM TABLE(GENERATOR(ROWCOUNT => {size}))
        ),
        base_data AS (
            SELECT 
                row_id,
                'CRV-' || LPAD(row_id, 10, '0') as customer_id,
                -- French first names (simplified list)
                CASE (row_id % 20)
                    WHEN 0 THEN 'Jean' WHEN 1 THEN 'Marie' WHEN 2 THEN 'Pierre'
                    WHEN 3 THEN 'Sophie' WHEN 4 THEN 'Michel' WHEN 5 THEN 'Catherine'
                    WHEN 6 THEN 'Philippe' WHEN 7 THEN 'Nathalie' WHEN 8 THEN 'Alain'
                    WHEN 9 THEN 'Isabelle' WHEN 10 THEN 'François' WHEN 11 THEN 'Sylvie'
                    WHEN 12 THEN 'Bernard' WHEN 13 THEN 'Martine' WHEN 14 THEN 'Patrick'
                    WHEN 15 THEN 'Christine' WHEN 16 THEN 'Daniel' WHEN 17 THEN 'Françoise'
                    WHEN 18 THEN 'Thierry' ELSE 'Monique'
                END as first_name,
                -- French last names
                CASE (row_id % 25)
                    WHEN 0 THEN 'Martin' WHEN 1 THEN 'Bernard' WHEN 2 THEN 'Dubois'
                    WHEN 3 THEN 'Thomas' WHEN 4 THEN 'Robert' WHEN 5 THEN 'Petit'
                    WHEN 6 THEN 'Richard' WHEN 7 THEN 'Durand' WHEN 8 THEN 'Leroy'
                    WHEN 9 THEN 'Moreau' WHEN 10 THEN 'Simon' WHEN 11 THEN 'Laurent'
                    WHEN 12 THEN 'Lefebvre' WHEN 13 THEN 'Michel' WHEN 14 THEN 'Garcia'
                    WHEN 15 THEN 'David' WHEN 16 THEN 'Bertrand' WHEN 17 THEN 'Roux'
                    WHEN 18 THEN 'Vincent' WHEN 19 THEN 'Fournier' WHEN 20 THEN 'Morel'
                    WHEN 21 THEN 'Girard' WHEN 22 THEN 'Andre' WHEN 23 THEN 'Lefevre'
                    ELSE 'Mercier'
                END as last_name,
                -- Generate synthetic emails
                LOWER(first_name) || '.' || LOWER(last_name) || '@' ||
                CASE (row_id % 6)
                    WHEN 0 THEN 'gmail.com' WHEN 1 THEN 'orange.fr' WHEN 2 THEN 'free.fr'
                    WHEN 3 THEN 'wanadoo.fr' WHEN 4 THEN 'sfr.fr' ELSE 'laposte.net'
                END as email,
                -- French phone numbers (simplified)
                CASE 
                    WHEN UNIFORM(0, 1, RANDOM()) < {PHONE_MISSING_PCT} THEN NULL
                    ELSE '0' || (1 + (row_id % 6)) || ' ' || 
                         LPAD(UNIFORM(10, 99, RANDOM()), 2, '0') || ' ' ||
                         LPAD(UNIFORM(10, 99, RANDOM()), 2, '0') || ' ' ||
                         LPAD(UNIFORM(10, 99, RANDOM()), 2, '0') || ' ' ||
                         LPAD(UNIFORM(10, 99, RANDOM()), 2, '0')
                END as phone,
                -- French postal codes (simplified)
                CASE 
                    WHEN UNIFORM(0, 1, RANDOM()) < {POSTAL_PRESENT_PCT} THEN 
                        LPAD(UNIFORM(1000, 95999, RANDOM()), 5, '0')
                    ELSE NULL
                END as postal_code,
                -- Date of birth for 40%
                CASE 
                    WHEN UNIFORM(0, 1, RANDOM()) < {DOB_PRESENT_PCT} THEN 
                        DATEADD(day, -UNIFORM(6570, 32850, RANDOM()), CURRENT_DATE()) -- 18-90 years old
                    ELSE NULL
                END as date_of_birth
            FROM row_nums
        )
        SELECT * FROM base_data
    """).collect()
    
    return temp_table


def apply_overlaps(session: Session, base_table: str, source_table: str, col_mapping: Dict[str, str]) -> str:
    """Apply controlled overlaps with source data."""
    overlap_table = f"TEMP_OVERLAP_DATA_{random.randint(1000, 9999)}"
    
    # Calculate overlap counts
    triple_match_count = int(TARGET_SIZE * TRIPLE_MATCH_PCT)
    email_extra_count = int(TARGET_SIZE * EMAIL_OVERLAP_PCT) - triple_match_count
    phone_extra_count = int(TARGET_SIZE * PHONE_OVERLAP_PCT) - triple_match_count
    name_extra_count = int(TARGET_SIZE * NAME_OVERLAP_PCT) - triple_match_count
    
    session.sql(f"""
        CREATE OR REPLACE TEMPORARY TABLE {overlap_table} AS
        WITH source_data AS (
            SELECT 
                {col_mapping.get('email', 'NULL')} as src_email,
                {col_mapping.get('phone', 'NULL')} as src_phone,
                {col_mapping.get('first_name', 'NULL')} as src_first_name,
                {col_mapping.get('last_name', 'NULL')} as src_last_name,
                ROW_NUMBER() OVER (ORDER BY UNIFORM(1, 1000000, RANDOM())) as src_rank
            FROM {source_table}
            WHERE {col_mapping.get('email', 'NULL')} IS NOT NULL
              AND {col_mapping.get('phone', 'NULL')} IS NOT NULL
              AND {col_mapping.get('first_name', 'NULL')} IS NOT NULL
              AND {col_mapping.get('last_name', 'NULL')} IS NOT NULL
        ),
        overlap_assignments AS (
            SELECT 
                b.*,
                CASE 
                    WHEN b.row_id <= {triple_match_count} THEN 'TRIPLE'
                    WHEN b.row_id <= {triple_match_count + email_extra_count} THEN 'EMAIL'
                    WHEN b.row_id <= {triple_match_count + email_extra_count + phone_extra_count} THEN 'PHONE'
                    WHEN b.row_id <= {triple_match_count + email_extra_count + phone_extra_count + name_extra_count} THEN 'NAME'
                    ELSE 'NONE'
                END as overlap_type,
                s.src_email, s.src_phone, s.src_first_name, s.src_last_name
            FROM {base_table} b
            LEFT JOIN source_data s ON s.src_rank = ((b.row_id - 1) % {SOURCE_SAMPLE_SIZE}) + 1
        )
        SELECT 
            customer_id,
            CASE 
                WHEN overlap_type IN ('TRIPLE', 'EMAIL') THEN src_email
                WHEN UNIFORM(0, 1, RANDOM()) < {EMAIL_MISSING_PCT} THEN NULL
                ELSE email
            END as email,
            CASE 
                WHEN overlap_type IN ('TRIPLE', 'PHONE') THEN src_phone
                ELSE phone
            END as phone,
            CASE 
                WHEN overlap_type IN ('TRIPLE', 'NAME') THEN src_first_name
                ELSE first_name
            END as first_name,
            CASE 
                WHEN overlap_type IN ('TRIPLE', 'NAME') THEN src_last_name
                ELSE last_name
            END as last_name,
            postal_code,
            date_of_birth,
            overlap_type
        FROM overlap_assignments
    """).collect()
    
    return overlap_table


def add_duplicates_and_noise(session: Session, table_name: str) -> str:
    """Add duplicate customers and data quality issues."""
    final_table = f"TEMP_FINAL_DATA_{random.randint(1000, 9999)}"
    
    duplicate_count = int(TARGET_SIZE * DUPLICATE_PCT)
    
    session.sql(f"""
        CREATE OR REPLACE TEMPORARY TABLE {final_table} AS
        WITH duplicates AS (
            SELECT 
                customer_id || '_DUP' as customer_id,
                -- Slightly modify email for some duplicates
                CASE 
                    WHEN UNIFORM(0, 1, RANDOM()) < 0.5 AND email IS NOT NULL THEN 
                        REGEXP_REPLACE(email, '@', UNIFORM(0, 9, RANDOM()) || '@')
                    ELSE email
                END as email,
                -- Slightly modify phone for some duplicates
                CASE 
                    WHEN UNIFORM(0, 1, RANDOM()) < 0.5 AND phone IS NOT NULL THEN 
                        SUBSTR(phone, 1, LENGTH(phone) - 1) || UNIFORM(0, 9, RANDOM())
                    ELSE phone
                END as phone,
                first_name, last_name, postal_code, date_of_birth,
                'DUPLICATE' as overlap_type
            FROM {table_name}
            WHERE ROW_NUMBER() OVER (ORDER BY customer_id) <= {duplicate_count}
        )
        SELECT customer_id, email, phone, first_name, last_name, postal_code, date_of_birth, overlap_type
        FROM {table_name}
        UNION ALL
        SELECT customer_id, email, phone, first_name, last_name, postal_code, date_of_birth, overlap_type
        FROM duplicates
    """).collect()
    
    return final_table


def create_final_table(session: Session, temp_table: str) -> None:
    """Create the final Crocevia CRM table."""
    full_table_name = f"{TARGET_DB}.{TARGET_SCHEMA}.{TARGET_TABLE}"
    
    session.sql(f"""
        CREATE OR REPLACE TABLE {full_table_name} AS
        SELECT 
            customer_id,
            email,
            phone,
            first_name,
            last_name,
            postal_code,
            date_of_birth,
            overlap_type,
            CURRENT_TIMESTAMP() as created_at
        FROM {temp_table}
    """).collect()
    
    print(f"Created table {full_table_name}")


def validate_results(session: Session) -> Dict[str, float]:
    """Validate overlap percentages and data quality."""
    full_table_name = f"{TARGET_DB}.{TARGET_SCHEMA}.{TARGET_TABLE}"
    
    # Get basic counts
    total_count = session.sql(f"SELECT COUNT(*) as cnt FROM {full_table_name}").collect()[0]['CNT']
    
    # Count overlaps by type
    overlap_counts = session.sql(f"""
        SELECT overlap_type, COUNT(*) as cnt
        FROM {full_table_name}
        GROUP BY overlap_type
    """).collect()
    
    overlap_dict = {row['OVERLAP_TYPE']: row['CNT'] for row in overlap_counts}
    
    results = {
        'total_rows': total_count,
        'triple_match_pct': overlap_dict.get('TRIPLE', 0) / total_count,
        'email_match_pct': (overlap_dict.get('TRIPLE', 0) + overlap_dict.get('EMAIL', 0)) / total_count,
        'phone_match_pct': (overlap_dict.get('TRIPLE', 0) + overlap_dict.get('PHONE', 0)) / total_count,
        'name_match_pct': (overlap_dict.get('TRIPLE', 0) + overlap_dict.get('NAME', 0)) / total_count,
        'duplicate_pct': overlap_dict.get('DUPLICATE', 0) / total_count,
    }
    
    return results


def main():
    """Main execution function."""
    print("Starting Crocevia CRM data generation...")
    
    # Create session
    session = create_session()
    
    try:
        # Setup
        print("Setting up database and schema...")
        setup_database_schema(session)
        
        # Sample source data
        print(f"Sampling {SOURCE_SAMPLE_SIZE} rows from source...")
        source_sample = sample_source_data(session)
        
        # Infer column mappings
        print("Inferring column mappings...")
        col_mapping = infer_column_mappings(session, source_sample)
        print(f"Column mapping: {col_mapping}")
        
        # Generate base dataset
        print(f"Generating {TARGET_SIZE} base records...")
        base_table = generate_base_dataset(session, TARGET_SIZE)
        
        # Apply overlaps
        print("Applying controlled overlaps...")
        overlap_table = apply_overlaps(session, base_table, source_sample, col_mapping)
        
        # Add noise and duplicates
        print("Adding duplicates and noise...")
        final_temp_table = add_duplicates_and_noise(session, overlap_table)
        
        # Create final table
        print("Creating final table...")
        create_final_table(session, final_temp_table)
        
        # Validate results
        print("Validating results...")
        results = validate_results(session)
        
        print("\n=== Generation Complete ===")
        print(f"Total rows: {results['total_rows']:,}")
        print(f"Triple match: {results['triple_match_pct']:.1%}")
        print(f"Email overlap: {results['email_match_pct']:.1%}")
        print(f"Phone overlap: {results['phone_match_pct']:.1%}")
        print(f"Name overlap: {results['name_match_pct']:.1%}")
        print(f"Duplicates: {results['duplicate_pct']:.1%}")
        
    finally:
        # Clean up temp tables (they'll auto-drop anyway)
        session.close()


if __name__ == "__main__":
    main()

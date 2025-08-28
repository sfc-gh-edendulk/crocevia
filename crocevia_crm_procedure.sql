-- =======================================================================
-- CROCEVIA CRM GENERATOR - STORED PROCEDURE VERSION
-- =======================================================================
-- 
-- Generates synthetic Crocevia grocery chain CRM data with controlled overlaps
-- to Summit Sports CRM for customer identity resolution testing.
--
-- Based on structure pattern from example_data_gen.sql:
-- - Stored procedure with parameter handling
-- - Class-based Python implementation
-- - Session-based data loading
-- - Efficient pandas operations
--
-- Usage:
--   CALL CROCEVIA.RAW_DATA.GENERATE_CROCEVIA_CRM(10000, 5000, 'overwrite');
--
-- =======================================================================

-- Create database and schema if they don't exist
CREATE DATABASE IF NOT EXISTS CROCEVIA;
CREATE SCHEMA IF NOT EXISTS CROCEVIA.RAW_DATA;
USE DATABASE CROCEVIA;
USE SCHEMA RAW_DATA;

-- Create the stored procedure
CREATE OR REPLACE PROCEDURE CROCEVIA.RAW_DATA.GENERATE_CROCEVIA_CRM(
    "TARGET_SIZE" NUMBER(38,0) DEFAULT 10000,
    "SOURCE_SAMPLE_SIZE" NUMBER(38,0) DEFAULT 5000
)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('numpy==2.2.2', 'pandas==2.2.3', 'snowflake-snowpark-python==*', 'Faker==24.9.0')
HANDLER = 'main'
EXECUTE AS OWNER
AS '
import snowflake.snowpark as snowpark
import pandas as pd
import numpy as np
import random
from faker import Faker


def generate_french_addresses(fake: Faker, num_addresses: int, unique_pct: float = 0.80) -> list:
    """
    Generate realistic French addresses with coordinates.
    
    Args:
        fake: Faker instance with fr_FR locale
        num_addresses: Number of addresses to generate
        unique_pct: Percentage of addresses that should be unique
    
    Returns:
        List of address dictionaries with street, city, postal_code, lat, lng
    """
    # Sample of real French cities with approximate coordinates
    french_cities = [
        {"city": "Paris", "postal_prefix": "75", "lat": 48.8566, "lng": 2.3522},
        {"city": "Marseille", "postal_prefix": "13", "lat": 43.2965, "lng": 5.3698},
        {"city": "Lyon", "postal_prefix": "69", "lat": 45.7640, "lng": 4.8357},
        {"city": "Toulouse", "postal_prefix": "31", "lat": 43.6047, "lng": 1.4442},
        {"city": "Nice", "postal_prefix": "06", "lat": 43.7102, "lng": 7.2620},
        {"city": "Nantes", "postal_prefix": "44", "lat": 47.2184, "lng": -1.5536},
        {"city": "Strasbourg", "postal_prefix": "67", "lat": 48.5734, "lng": 7.7521},
        {"city": "Montpellier", "postal_prefix": "34", "lat": 43.6110, "lng": 3.8767},
        {"city": "Bordeaux", "postal_prefix": "33", "lat": 44.8378, "lng": -0.5792},
        {"city": "Lille", "postal_prefix": "59", "lat": 50.6292, "lng": 3.0573},
        {"city": "Rennes", "postal_prefix": "35", "lat": 48.1173, "lng": -1.6778},
        {"city": "Reims", "postal_prefix": "51", "lat": 49.2583, "lng": 4.0317},
        {"city": "Le Havre", "postal_prefix": "76", "lat": 49.4944, "lng": 0.1079},
        {"city": "Saint-Étienne", "postal_prefix": "42", "lat": 45.4397, "lng": 4.3872},
        {"city": "Toulon", "postal_prefix": "83", "lat": 43.1242, "lng": 5.9280},
        {"city": "Angers", "postal_prefix": "49", "lat": 47.4784, "lng": -0.5632},
        {"city": "Grenoble", "postal_prefix": "38", "lat": 45.1885, "lng": 5.7245},
        {"city": "Dijon", "postal_prefix": "21", "lat": 47.3220, "lng": 5.0415},
        {"city": "Nîmes", "postal_prefix": "30", "lat": 43.8367, "lng": 4.3601},
        {"city": "Aix-en-Provence", "postal_prefix": "13", "lat": 43.5297, "lng": 5.4474},
        {"city": "Le Mans", "postal_prefix": "72", "lat": 48.0061, "lng": 0.1996},
        {"city": "Brest", "postal_prefix": "29", "lat": 48.3904, "lng": -4.4861},
        {"city": "Tours", "postal_prefix": "37", "lat": 47.3941, "lng": 0.6848},
        {"city": "Limoges", "postal_prefix": "87", "lat": 45.8336, "lng": 1.2611},
        {"city": "Clermont-Ferrand", "postal_prefix": "63", "lat": 45.7797, "lng": 3.0863},
        {"city": "Villeurbanne", "postal_prefix": "69", "lat": 45.7665, "lng": 4.8795},
        {"city": "Amiens", "postal_prefix": "80", "lat": 49.8941, "lng": 2.2958},
        {"city": "Metz", "postal_prefix": "57", "lat": 49.1193, "lng": 6.1757},
        {"city": "Besançon", "postal_prefix": "25", "lat": 47.2380, "lng": 6.0243},
        {"city": "Perpignan", "postal_prefix": "66", "lat": 42.6886, "lng": 2.8956},
        {"city": "Orléans", "postal_prefix": "45", "lat": 47.9029, "lng": 1.9093},
        {"city": "Caen", "postal_prefix": "14", "lat": 49.1829, "lng": -0.3707},
        {"city": "Rouen", "postal_prefix": "76", "lat": 49.4431, "lng": 1.0993},
        {"city": "Nancy", "postal_prefix": "54", "lat": 48.6921, "lng": 6.1844},
        {"city": "Argenteuil", "postal_prefix": "95", "lat": 48.9474, "lng": 2.2473},
        {"city": "Montreuil", "postal_prefix": "93", "lat": 48.8630, "lng": 2.4447},
        {"city": "Mulhouse", "postal_prefix": "68", "lat": 47.7508, "lng": 7.3359},
        {"city": "Roubaix", "postal_prefix": "59", "lat": 50.6927, "lng": 3.1746},
        {"city": "Tourcoing", "postal_prefix": "59", "lat": 50.7236, "lng": 3.1609},
        {"city": "La Rochelle", "postal_prefix": "17", "lat": 46.1603, "lng": -1.1511}
    ]
    
    # Common French street prefixes
    street_prefixes = [
        "rue", "avenue", "boulevard", "place", "allée", "impasse", 
        "chemin", "route", "passage", "square", "quai", "cours"
    ]
    
    # Generate unique addresses
    unique_count = int(num_addresses * unique_pct)
    duplicate_count = num_addresses - unique_count
    
    addresses = []
    
    # Generate unique addresses
    for i in range(unique_count):
        city_info = random.choice(french_cities)
        street_prefix = random.choice(street_prefixes)
        street_name = fake.street_name()
        
        # Generate realistic street number
        street_number = random.randint(1, 999)
        
        # Create full street address
        street = str(street_number) + " " + street_prefix + " " + street_name
        
        # Generate postal code based on city prefix
        postal_suffix = str(random.randint(100, 999)).zfill(3)
        postal_code = city_info["postal_prefix"] + postal_suffix
        
        # Add small random offset to coordinates for realistic variance
        lat_offset = random.uniform(-0.05, 0.05)  # ~5km variance
        lng_offset = random.uniform(-0.05, 0.05)
        
        addresses.append({
            "street": street,
            "city": city_info["city"],
            "postal_code": postal_code,
            "latitude": round(city_info["lat"] + lat_offset, 6),
            "longitude": round(city_info["lng"] + lng_offset, 6)
        })
    
    # Add duplicate addresses by repeating some of the unique ones
    if duplicate_count > 0:
        duplicate_sources = random.choices(addresses, k=duplicate_count)
        addresses.extend(duplicate_sources)
    
    # Shuffle to distribute duplicates randomly
    random.shuffle(addresses)
    
    return addresses


def generate_crocevia_customers(source_customers_df: pd.DataFrame, 
                              num_customers: int = 10000,
                              source_sample_size: int = 5000) -> pd.DataFrame:
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
        self.ADDRESS_PRESENT_PCT = 0.50 # 50% have complete addresses
        self.EMAIL_MISSING_PCT = 0.15   # 15% missing emails
        self.PHONE_MISSING_PCT = 0.20   # 20% missing phones
        self.DUPLICATE_PCT = 0.10       # 10% duplicate customers
        
        # Initialize French Faker
        self.faker = Faker("fr_FR")
        Faker.seed(42)  # For reproducible results
        
        # French email domains for realistic localization
        self.FRENCH_EMAIL_DOMAINS = [
            "gmail.com", "orange.fr", "free.fr", "wanadoo.fr", 
            "sfr.fr", "laposte.net", "hotmail.fr", "yahoo.fr"
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
        self.src_emails = self.source_sample["EMAIL"].dropna().astype(str).values
        self.src_phones = self.source_sample["PHONE"].dropna().astype(str).values
        self.src_first_names = self.source_sample["FIRST_NAME"].dropna().astype(str).values
        self.src_last_names = self.source_sample["LAST_NAME"].dropna().astype(str).values
        
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
        
        # Generate addresses for 50% of customers
        address_count = int(self.target_size * self.ADDRESS_PRESENT_PCT)
        self.addresses = generate_french_addresses(self.faker, address_count, unique_pct=0.80)
        self.address_indices = set(np.random.choice(range(self.target_size), size=address_count, replace=False))
        self.address_idx = 0

    def _generate_synthetic_customer_data(self) -> pd.DataFrame:
        """Generate base synthetic customer data with French localization using Faker."""
        np.random.seed(42)
        random.seed(42)
        self.faker.seed_instance(42)  # Ensure reproducible Faker results
        
        customers = []
        
        for i in range(self.target_size):
            # Generate French names using Faker
            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
            
            # Generate realistic French email using Faker
            # Use Faker email method but with French domains for localization
            if random.random() < 0.7:  # 70% use name-based emails
                local = (first_name.lower() + "." + last_name.lower()).replace("\'", "").replace("ç", "c").replace("é", "e").replace("è", "e").replace("à", "a").replace("ù", "u")
                domain = random.choice(self.FRENCH_EMAIL_DOMAINS)
                email = local + "@" + domain
            else:  # 30% use more varied patterns
                email = self.faker.email()
                # Replace domain with French one
                local_part = email.split("@")[0]
                domain = random.choice(self.FRENCH_EMAIL_DOMAINS)
                email = local_part + "@" + domain
            
            # Generate French phone number using Faker
            phone = self.faker.phone_number()
            
            # Generate date of birth using Faker (18-90 years old)
            if random.random() < self.DOB_PRESENT_PCT:
                date_of_birth = self.faker.date_of_birth(minimum_age=18, maximum_age=90)
            else:
                date_of_birth = None
            
            # Add address information if this customer should have one
            if i in self.address_indices:
                address = self.addresses[self.address_idx]
                self.address_idx += 1
                street = address["street"]
                city = address["city"]
                postal_code = address["postal_code"]
                latitude = address["latitude"]
                longitude = address["longitude"]
            else:
                street = None
                city = None
                postal_code = self.faker.postcode() if random.random() < self.POSTAL_PRESENT_PCT else None
                latitude = None
                longitude = None
            
            customers.append({
                "row_id": i,
                "customer_id": "CRV-" + str(i).zfill(10),
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone": phone,
                "street": street,
                "city": city,
                "postal_code": postal_code,
                "latitude": latitude,
                "longitude": longitude,
                "date_of_birth": date_of_birth,
                "overlap_type": "NONE"
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
                df.loc[i, "email"] = source_row["EMAIL"]
                df.loc[i, "phone"] = source_row["PHONE"] 
                df.loc[i, "first_name"] = source_row["FIRST_NAME"]
                df.loc[i, "last_name"] = source_row["LAST_NAME"]
                df.loc[i, "overlap_type"] = "TRIPLE"
        
        # Apply email-only overlaps
        if len(self.email_indices) > 0:
            df.loc[self.email_indices, "email"] = np.random.choice(
                self.src_emails, size=len(self.email_indices), replace=True
            )
            df.loc[self.email_indices, "overlap_type"] = "EMAIL"
        
        # Apply phone-only overlaps
        if len(self.phone_indices) > 0:
            df.loc[self.phone_indices, "phone"] = np.random.choice(
                self.src_phones, size=len(self.phone_indices), replace=True
            )
            # Update overlap type (might overwrite EMAIL)
            phone_only = np.setdiff1d(self.phone_indices, self.email_indices)
            df.loc[phone_only, "overlap_type"] = "PHONE"
            if len(np.intersect1d(self.phone_indices, self.email_indices)) > 0:
                df.loc[np.intersect1d(self.phone_indices, self.email_indices), "overlap_type"] = "EMAIL_PHONE"
        
        # Apply name-only overlaps
        if len(self.name_indices) > 0:
            name_pairs = np.column_stack([
                np.random.choice(self.src_first_names, size=len(self.name_indices), replace=True),
                np.random.choice(self.src_last_names, size=len(self.name_indices), replace=True)
            ])
            df.loc[self.name_indices, "first_name"] = name_pairs[:, 0]
            df.loc[self.name_indices, "last_name"] = name_pairs[:, 1]
            # Update overlap types appropriately
            name_only = np.setdiff1d(self.name_indices, np.concatenate([self.email_indices, self.phone_indices]))
            df.loc[name_only, "overlap_type"] = "NAME"
        
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
            df.loc[missing_email_indices, "email"] = None
        
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
            df.loc[missing_phone_indices, "phone"] = None
        
        # Add duplicate customers with variations
        duplicate_count = int(self.DUPLICATE_PCT * len(df))
        if duplicate_count > 0:
            source_indices = np.random.choice(len(df), size=duplicate_count, replace=False)
            
            duplicates = []
            for src_idx in source_indices:
                src_row = df.iloc[src_idx].copy()
                src_row["customer_id"] = src_row["customer_id"] + "_DUP"
                src_row["overlap_type"] = "DUPLICATE"
                
                # Add slight variations for realism
                if random.random() < 0.5 and src_row["email"]:
                    # Modify email slightly
                    email_parts = str(src_row["email"]).split("@")
                    if len(email_parts) == 2:
                        src_row["email"] = email_parts[0] + str(random.randint(0, 9)) + "@" + email_parts[1]
                
                if random.random() < 0.5 and src_row["phone"]:
                    # Modify last digit of phone
                    phone_str = str(src_row["phone"])
                    if phone_str:
                        src_row["phone"] = phone_str[:-1] + str(random.randint(0, 9))
                
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
        df["created_at"] = datetime.now()
        
        # Reorder columns for consistency
        column_order = [
            "customer_id", "email", "phone", "first_name", "last_name",
            "postal_code", "date_of_birth", "overlap_type", "created_at"
        ]
        df = df[column_order]
        
        print(f"Generated {len(df)} total records (including duplicates)")
        return df

    def validate_results(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate that overlap percentages meet targets."""
        total_records = len(df)
        overlap_counts = df["overlap_type"].value_counts()
        
        # Calculate actual overlap percentages
        triple_pct = overlap_counts.get("TRIPLE", 0) / total_records
        email_pct = (overlap_counts.get("TRIPLE", 0) + overlap_counts.get("EMAIL", 0) + 
                    overlap_counts.get("EMAIL_PHONE", 0)) / total_records
        phone_pct = (overlap_counts.get("TRIPLE", 0) + overlap_counts.get("PHONE", 0) + 
                    overlap_counts.get("EMAIL_PHONE", 0)) / total_records
        name_pct = (overlap_counts.get("TRIPLE", 0) + overlap_counts.get("NAME", 0)) / total_records
        duplicate_pct = overlap_counts.get("DUPLICATE", 0) / total_records
        
        return {
            "total_records": total_records,
            "triple_match_pct": triple_pct,
            "email_overlap_pct": email_pct,
            "phone_overlap_pct": phone_pct,
            "name_overlap_pct": name_pct,
            "duplicate_pct": duplicate_pct
        }


def main(session: snowpark.Session) -> str:
    """
    Main function to generate Crocevia CRM with controlled overlaps.
    
    Args:
        session: Snowflake Snowpark session
    
    Returns:
        Success message
    """
    print("Loading Summit Sports customer data...")
    source_customers_df = session.table("SS_101.RAW_CUSTOMER.CUSTOMER_LOYALTY").to_pandas()
    print("Loaded " + str(len(source_customers_df)) + " source customers")
    
    print("Generating " + str(TARGET_SIZE) + " Crocevia customers with controlled overlaps...")
    customers_df = generate_crocevia_customers(
        source_customers_df, 
        num_customers=TARGET_SIZE,
        source_sample_size=SOURCE_SAMPLE_SIZE
    )
    
    print("Adding duplicate customers for realism...")
    customers_df = add_duplicate_customers(customers_df)
    
    print("Validating overlap results...")
    validation = validate_overlap_results(customers_df, source_customers_df)
    
    print("\\n=== Generation Results ===")
    print("Total records: " + str(validation["total_records"]))
    print("Overlap breakdown: " + str(validation["overlap_type_breakdown"]))
    print("Triple match: {:.1%}".format(validation["target_triple_pct"]))
    print("Email overlap: {:.1%} (actual: {:.1%})".format(validation["target_email_pct"], validation["actual_email_overlap"]))
    print("Phone overlap: {:.1%} (actual: {:.1%})".format(validation["target_phone_pct"], validation["actual_phone_overlap"]))
    print("Name overlap: {:.1%} (actual: {:.1%})".format(validation["target_name_pct"], validation["actual_name_overlap"]))
    print("Duplicates: {:.1%}".format(validation["duplicate_pct"]))
    
    print("Writing Crocevia customers to Snowflake...")
    session.write_pandas(
        customers_df, 
        "CROCEVIA_CRM", 
        auto_create_table=True, 
        overwrite=True
    )
    
    print("Crocevia CRM generation complete!")
    
    return "Crocevia CRM generation complete! Generated " + str(len(customers_df)) + " records."
';

-- =======================================================================
-- USAGE EXAMPLES AND TESTING
-- =======================================================================

-- Test with small dataset (1000 records)
CALL CROCEVIA.RAW_DATA.GENERATE_CROCEVIA_CRM(1000, 500);

-- Validate the generated data
SELECT 
    overlap_type,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
FROM CROCEVIA.RAW_DATA.CROCEVIA_CRM
GROUP BY overlap_type
ORDER BY count DESC;

-- Sample the results
SELECT * FROM CROCEVIA.RAW_DATA.CROCEVIA_CRM LIMIT 10;

-- Full production run (10,000 records)
-- CALL CROCEVIA.RAW_DATA.GENERATE_CROCEVIA_CRM(10000, 5000);

-- Monitor table size and structure
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(email) as records_with_email,
    COUNT(phone) as records_with_phone,
    COUNT(date_of_birth) as records_with_dob,
    COUNT(postal_code) as records_with_postal
FROM CROCEVIA.RAW_DATA.CROCEVIA_CRM;

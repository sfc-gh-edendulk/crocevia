CREATE OR REPLACE PROCEDURE SPORTS_DB.SPORTS_DATA.SPORTS_RETAILER_SALES_GENERATOR_CRM_V2("END_DATE" DATE, "NUM_DAYS" NUMBER(38,0), "WRITEMODE" VARCHAR)
RETURNS VARCHAR

LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('holidays==0.48','numpy==2.2.2','pandas==2.2.3','snowflake-snowpark-python==*')
HANDLER = 'main'
EXECUTE AS OWNER
AS 'import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col, when, lit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from typing import Dict, List, Optional
import uuid

class CRMSalesDataGenerator:
    def __init__(self,
                 product_catalog_df: pd.DataFrame,
                 store_catalog_df: pd.DataFrame,
                 chdry_data: pd.DataFrame,
                 customers_df: pd.DataFrame,
                 loyalty_cards_df: pd.DataFrame):
        
        # Store DataFrames as class attributes
        self.product_catalog_df = product_catalog_df
        self.store_catalog_df = store_catalog_df
        self.customers_df = customers_df
        self.loyalty_cards_df = loyalty_cards_df
        
        # Preprocess CAC40 data
        self.chdry_data = self._preprocess_chdry_data(chdry_data)
        
        # Constants
        self.AVG_DAILY_SALES_TARGET = 10000
        self.HOLIDAYS_FR = holidays.France()
        self.PAYMENT_METHODS = ["Credit Card", "Debit Card", "Gift Card", "Cash"]
        
        # Create sales periods lookup
        current_year = datetime.now().year
        self.sales_periods = {
            "winter": (datetime(current_year, 12, 1).date(), datetime(current_year, 2, 28).date()),
            "summer": (datetime(current_year, 6, 15).date(), datetime(current_year, 8, 31).date()),
            "back_to_school": (datetime(current_year, 9, 1).date(), datetime(current_year, 9, 30).date())
        }
        
        # Pre-calculate customer-card mappings
        self._create_customer_card_mapping()

    def _preprocess_chdry_data(self, chdry_data: pd.DataFrame) -> Dict[datetime.date, float]:
        """Preprocess CAC40 data into a lookup dictionary for faster access."""
        chdry_data["DATE"] = pd.to_datetime(chdry_data["DATE"]).dt.date
        chdry_data["NORMALIZED_chdry"] = (chdry_data["CLOSE"] - chdry_data["CLOSE"].min()) / \\
                                        (chdry_data["CLOSE"].max() - chdry_data["CLOSE"].min())
        return dict(zip(chdry_data["DATE"], chdry_data["NORMALIZED_chdry"]))

    def _create_customer_card_mapping(self):
        """Create a lookup dictionary for customer-card relationships."""
        self.customer_card_mapping = self.loyalty_cards_df.groupby(''CUSTOMER_ID'')[''CARD_ID''].first().to_dict()

    def _get_peak_sales_multiplier(self, date\: datetime, store_type\: str) -> float\:
        """Calculate sales multiplier based on date and store type."""
        date_to_compare = date.date() if isinstance(date, datetime) else date
    
        if "Montagne" in store_type.lower()\:
            return 1.5 if date_to_compare.month in {12, 1, 2} else 0.8
        return 1.3 if any(start <= date_to_compare <= end for start, end in self.sales_periods.values()) else 1.0
    
    def _generate_order_details(self) -> tuple:
        """Generate customer and loyalty card details for an order."""
        if np.random.random() > 0.30:  # 70% chance for customer association
            customer = self.customers_df.sample(n=1).iloc[0]
            customer_id = customer["CUSTOMER_ID"]
            card_id = self.customer_card_mapping.get(customer_id) if np.random.random() > 0.20 else None
            return customer_id, card_id
        return None, None

    def generate_sales_data(self, end_date: datetime, num_days: int = 365) -> pd.DataFrame:
        """Generate sales data more efficiently using vectorized operations where possible."""
        sales_data = []
        
        for day_delta in range(num_days):
            date = end_date - timedelta(days=day_delta)
            if date in self.HOLIDAYS_FR:
                continue
                
            normalized_chdry = self.chdry_data.get(date)
            if normalized_chdry is None:
                continue

            for _, store in self.store_catalog_df.iterrows():
                store_target = self.AVG_DAILY_SALES_TARGET * (0.7 + 0.6 * normalized_chdry)
                store_target *= self._get_peak_sales_multiplier(date, store["STORE_TYPE"])
                accumulated_sales = 0

                while accumulated_sales < store_target:
                    order_id = str(uuid.uuid4())
                    customer_id, card_id = self._generate_order_details()
                    
                    # Generate items for this order
                    num_items = np.random.randint(2, 6)
                    products = self.product_catalog_df.sample(n=num_items)
                    
                    for _, product in products.iterrows():
                        quantity = 1 if np.random.random() < 0.8 else np.random.randint(2, 6)
                        applicable_price = (product["SALE_PRICE"] 
                                         if any(start <= date <= end for start, end in self.sales_periods.values())
                                         else product["MRP"])
                        
                        sales_data.append([
                            order_id,
                            store["STOREID"],
                            store["STORE_NAME"],
                            date.date() if isinstance(date, datetime) else date,
                            product["PRODUCTID"],
                            quantity,
                            round(applicable_price * quantity, 2),
                            round((product["MRP"] - product["SALE_PRICE"]) * quantity, 2),
                            np.random.choice(self.PAYMENT_METHODS, p=[0.3, 0.3, 0.2, 0.2]),
                            f"ASSISTANT_{np.random.randint(1, 801)}",
                            customer_id,
                            card_id
                        ])
                        
                        accumulated_sales += applicable_price * quantity

        return pd.DataFrame(sales_data, columns=[
            "ORDER_ID", "STOREID", "STORE_NAME", "SALE_DATE", "PRODUCT_ID", "QUANTITY",
            "SALES_PRICE_EURO", "DISCOUNT_AMOUNT_EURO", "PAYMENT_METHOD", "SALES_ASSISTANT_ID",
            "CUSTOMER_ID", "CARD_ID"
        ])

def main(session: snowpark.Session, end_date: datetime = datetime(2025, 3, 14), num_days: int = 30, writemode: str = "append") -> pd.DataFrame:
    # Load data from Snowflake
    tables = {
        "product_catalogue": "SPORTS_DB.SPORTS_DATA.SPORTS_PRODUCT_CATALOGUE",
        "store_catalogue": "SPORTS_DB.SPORTS_DATA.SPORTS_STORES",
        "chdry_data": "SPORTS_DB.SPORTS_DATA.chdry0120_0325_COMPLETE",
        "customers": "CUSTOMERS",
        "loyalty_cards": "FIDELITY_CARDS"
    }
    
    # Load all tables in parallel using dictionary comprehension
    dfs = {name: session.table(path).to_pandas() for name, path in tables.items()}
    
    # Initialize generator and generate data
    generator = CRMSalesDataGenerator(
        dfs["product_catalogue"],
        dfs["store_catalogue"],
        dfs["chdry_data"],
        dfs["customers"],
        dfs["loyalty_cards"]
    )
    
    sales_df = generator.generate_sales_data(end_date=end_date, num_days=num_days)
    
    # Write to Snowflake
    session.create_dataframe(sales_df).write.mode(writemode).save_as_table(
        "SPORTS_DB.SPORTS_DATA.INSTORE_SALES_DATA_CRM3"
    )
    
    print("CRM-integrated sales data generation complete!")
    return "CRM-integrated sales data generation complete!"';

    call SPORTS_RETAILER_SALES_GENERATOR_CRM_V2('2025-02-23'::DATE, 53, 'append');

    select * from SPORTS_DB.SPORTS_DATA.INSTORE_SALES_DATA_CRM3 limit 5;
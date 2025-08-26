# Snowflake Notebook Code Examples and Templates

This file contains comprehensive code examples and templates for creating Snowflake notebooks as an alternative to DBT for data engineering pipelines.

## Table of Contents
1. [Notebook Templates](#notebook-templates)
2. [Code Examples](#code-examples)
3. [Deployment Scripts](#deployment-scripts)
4. [Best Practices](#best-practices)

---

## Notebook Templates

### 1. JSON Processing Notebook Template

**File: `notebooks/01_json_processing.ipynb`**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# JSON Processing: Bronze to Silver Layer\n",
    "\n",
    "This notebook demonstrates how to extract structured data from JSON documents stored in the Bronze layer.\n",
    "\n",
    "## Objectives:\n",
    "- Extract client information from JSON documents\n",
    "- Extract vehicle information from JSON documents\n",
    "- Create normalized Silver layer tables\n",
    "- Validate data quality and completeness\n",
    "\n",
    "## Prerequisites:\n",
    "- Bronze layer data loaded (DEMANDES table with JSON)\n",
    "- Database and schemas created (BRONZE, SILVER, GOLD)\n",
    "- Appropriate warehouse permissions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Set context for the notebook\n",
    "USE DATABASE your_demo_database;\n",
    "USE SCHEMA silver;\n",
    "USE WAREHOUSE your_warehouse;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Explore JSON Structure\n",
    "\n",
    "First, let's examine the structure of our JSON data to understand what we're working with."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Examine JSON structure in Bronze layer\n",
    "SELECT \n",
    "    demande_id,\n",
    "    donnees_demande_json\n",
    "FROM bronze.demandes \n",
    "LIMIT 3;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Extract Client Information\n",
    "\n",
    "Create a view to extract client information from the JSON structure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Create CLIENTS view by extracting from JSON\n",
    "CREATE OR REPLACE VIEW silver.clients AS\n",
    "SELECT \n",
    "    demande_id,\n",
    "    donnees_demande_json:client:nom::STRING AS nom,\n",
    "    donnees_demande_json:client:prenom::STRING AS prenom,\n",
    "    donnees_demande_json:client:email::STRING AS email,\n",
    "    donnees_demande_json:client:age::INTEGER AS age,\n",
    "    donnees_demande_json:client:ville::STRING AS ville,\n",
    "    donnees_demande_json:client:code_postal::STRING AS code_postal,\n",
    "    donnees_demande_json:client:telephone::STRING AS telephone,\n",
    "    CURRENT_TIMESTAMP() AS date_creation\n",
    "FROM bronze.demandes\n",
    "WHERE donnees_demande_json:client IS NOT NULL;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Validate Client Data\n",
    "\n",
    "Check the quality and completeness of the extracted client data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Validate client data extraction\n",
    "SELECT \n",
    "    COUNT(*) as total_clients,\n",
    "    COUNT(DISTINCT email) as unique_emails,\n",
    "    COUNT(CASE WHEN nom IS NULL THEN 1 END) as missing_names,\n",
    "    COUNT(CASE WHEN email IS NULL THEN 1 END) as missing_emails,\n",
    "    AVG(age) as average_age,\n",
    "    MIN(age) as min_age,\n",
    "    MAX(age) as max_age\n",
    "FROM silver.clients;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "python"
   },
   "outputs": [],
   "source": [
    "# Visualize client data distribution\n",
    "from snowflake.snowpark.context import get_active_session\n",
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "\n",
    "session = get_active_session()\n",
    "\n",
    "# Get client data for visualization\n",
    "client_data = session.sql(\"\"\"\n",
    "    SELECT \n",
    "        ville,\n",
    "        COUNT(*) as client_count,\n",
    "        AVG(age) as avg_age\n",
    "    FROM silver.clients \n",
    "    GROUP BY ville \n",
    "    ORDER BY client_count DESC\n",
    "    LIMIT 10\n",
    "\"\"\").to_pandas()\n",
    "\n",
    "# Create bar chart\n",
    "fig = px.bar(client_data, \n",
    "             x='VILLE', \n",
    "             y='CLIENT_COUNT',\n",
    "             title='Top 10 Cities by Client Count')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Extract Vehicle Information\n",
    "\n",
    "Create a view to extract vehicle information from the JSON structure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Create VEHICULES view by extracting from JSON\n",
    "CREATE OR REPLACE VIEW silver.vehicules AS\n",
    "SELECT \n",
    "    demande_id,\n",
    "    donnees_demande_json:vehicule:marque::STRING AS marque,\n",
    "    donnees_demande_json:vehicule:modele::STRING AS modele,\n",
    "    donnees_demande_json:vehicule:annee::INTEGER AS annee,\n",
    "    donnees_demande_json:vehicule:carburant::STRING AS carburant,\n",
    "    donnees_demande_json:vehicule:puissance::INTEGER AS puissance,\n",
    "    donnees_demande_json:vehicule:valeur_vehicule::NUMBER AS valeur_vehicule,\n",
    "    CURRENT_TIMESTAMP() AS date_creation\n",
    "FROM bronze.demandes\n",
    "WHERE donnees_demande_json:vehicule IS NOT NULL;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Summary and Next Steps\n",
    "\n",
    "Review what we've accomplished and prepare for the next notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Final validation of both views\n",
    "SELECT 'CLIENTS' as table_name, COUNT(*) as record_count FROM silver.clients\n",
    "UNION ALL\n",
    "SELECT 'VEHICULES' as table_name, COUNT(*) as record_count FROM silver.vehicules;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Key Takeaways:\n",
    "\n",
    "1. **JSON Extraction**: Successfully extracted client and vehicle data from JSON documents\n",
    "2. **Data Quality**: Validated completeness and identified any data quality issues\n",
    "3. **Performance**: Views provide efficient access to structured data\n",
    "4. **Next Steps**: Proceed to notebook 02 for data quality and enrichment\n",
    "\n",
    "## Troubleshooting:\n",
    "\n",
    "- If JSON paths return NULL, verify the JSON structure\n",
    "- If data types don't match, adjust the casting (::STRING, ::INTEGER, etc.)\n",
    "- If views are empty, check the source Bronze table has data"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Snowflake",
   "language": "sql",
   "name": "snowflake"
  },
  "language_info": {
   "name": "sql",
   "version": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

### 2. Data Quality Notebook Template

**File: `notebooks/02_data_quality.ipynb`**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Quality: Bronze to Silver Enrichment\n",
    "\n",
    "This notebook focuses on data quality checks, validation, and enrichment of Bronze layer data.\n",
    "\n",
    "## Objectives:\n",
    "- Clean and validate Bronze layer CSV data\n",
    "- Enrich data with business logic\n",
    "- Create high-quality Silver layer tables\n",
    "- Implement data quality checks\n",
    "\n",
    "## Prerequisites:\n",
    "- Bronze layer data loaded (DEVIS, FOURNISSEURS, CAMPAGNES_MARKETING)\n",
    "- JSON processing completed (notebook 01)\n",
    "- Database and schemas created"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Set context\n",
    "USE DATABASE your_demo_database;\n",
    "USE SCHEMA silver;\n",
    "USE WAREHOUSE your_warehouse;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Analyze Bronze Data Quality\n",
    "\n",
    "First, let's examine the quality of our Bronze layer data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Check data quality in Bronze DEVIS table\n",
    "SELECT \n",
    "    COUNT(*) as total_records,\n",
    "    COUNT(DISTINCT devis_id) as unique_devis,\n",
    "    COUNT(CASE WHEN prime_mensuelle IS NULL THEN 1 END) as missing_prime,\n",
    "    COUNT(CASE WHEN prime_mensuelle <= 0 THEN 1 END) as invalid_prime,\n",
    "    COUNT(CASE WHEN fournisseur IS NULL THEN 1 END) as missing_fournisseur,\n",
    "    MIN(prime_mensuelle) as min_prime,\n",
    "    MAX(prime_mensuelle) as max_prime,\n",
    "    AVG(prime_mensuelle) as avg_prime\n",
    "FROM bronze.devis;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Create Clean DEVIS View\n",
    "\n",
    "Clean and enrich the DEVIS data with business logic."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Create clean DEVIS view with business logic\n",
    "CREATE OR REPLACE VIEW silver.devis_clean AS\n",
    "SELECT \n",
    "    d.devis_id,\n",
    "    d.demande_id,\n",
    "    d.date_devis,\n",
    "    d.type_assurance,\n",
    "    d.franchise,\n",
    "    d.prime_mensuelle,\n",
    "    d.fournisseur,\n",
    "    d.rang_affichage,\n",
    "    d.source_acquisition,\n",
    "    d.canal_marketing,\n",
    "    d.campagne_id,\n",
    "    d.mot_cle,\n",
    "    d.referrer_url,\n",
    "    d.device_type,\n",
    "    d.session_id,\n",
    "    -- Business logic enrichments\n",
    "    CASE \n",
    "        WHEN d.prime_mensuelle < 50 THEN 'Budget'\n",
    "        WHEN d.prime_mensuelle < 100 THEN 'Standard'\n",
    "        WHEN d.prime_mensuelle < 200 THEN 'Premium'\n",
    "        ELSE 'Luxury'\n",
    "    END as categorie_prix,\n",
    "    \n",
    "    -- Calculate annual premium\n",
    "    d.prime_mensuelle * 12 as prime_annuelle,\n",
    "    \n",
    "    -- Data quality flags\n",
    "    CASE \n",
    "        WHEN d.prime_mensuelle IS NULL OR d.prime_mensuelle <= 0 THEN 'INVALID'\n",
    "        WHEN d.fournisseur IS NULL THEN 'MISSING_PROVIDER'\n",
    "        ELSE 'VALID'\n",
    "    END as data_quality_flag,\n",
    "    \n",
    "    CURRENT_TIMESTAMP() as date_creation\n",
    "FROM bronze.devis d\n",
    "WHERE d.devis_id IS NOT NULL;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Validate Data Quality\n",
    "\n",
    "Check the results of our data cleaning process."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Validate cleaned data\n",
    "SELECT \n",
    "    data_quality_flag,\n",
    "    COUNT(*) as record_count,\n",
    "    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage\n",
    "FROM silver.devis_clean\n",
    "GROUP BY data_quality_flag\n",
    "ORDER BY record_count DESC;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "python"
   },
   "outputs": [],
   "source": [
    "# Visualize data quality results\n",
    "from snowflake.snowpark.context import get_active_session\n",
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "\n",
    "session = get_active_session()\n",
    "\n",
    "# Get price category distribution\n",
    "price_dist = session.sql(\"\"\"\n",
    "    SELECT \n",
    "        categorie_prix,\n",
    "        COUNT(*) as count,\n",
    "        AVG(prime_mensuelle) as avg_premium\n",
    "    FROM silver.devis_clean \n",
    "    WHERE data_quality_flag = 'VALID'\n",
    "    GROUP BY categorie_prix\n",
    "    ORDER BY avg_premium\n",
    "\"\"\").to_pandas()\n",
    "\n",
    "# Create visualization\n",
    "fig = px.bar(price_dist, \n",
    "             x='CATEGORIE_PRIX', \n",
    "             y='COUNT',\n",
    "             title='Distribution of Insurance Quotes by Price Category')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Create Enhanced DEMANDE View\n",
    "\n",
    "Create a cleaned version of the DEMANDE data by extracting from JSON."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Create clean DEMANDE view\n",
    "CREATE OR REPLACE VIEW silver.demande_clean AS\n",
    "SELECT \n",
    "    d.demande_id,\n",
    "    d.donnees_demande_json:type_assurance::STRING AS type_assurance,\n",
    "    d.donnees_demande_json:timestamp_creation::TIMESTAMP AS timestamp_creation,\n",
    "    d.donnees_demande_json:statut_demande::STRING AS statut_demande,\n",
    "    \n",
    "    -- Extract demande details\n",
    "    d.donnees_demande_json:demande:nouveau_vehicule::BOOLEAN AS nouveau_vehicule,\n",
    "    d.donnees_demande_json:demande:raison_changement::STRING AS raison_changement,\n",
    "    d.donnees_demande_json:demande:assureur_actuel::STRING AS assureur_actuel,\n",
    "    d.donnees_demande_json:demande:sinistres_12_mois::INTEGER AS sinistres_12_mois,\n",
    "    d.donnees_demande_json:demande:bonus_malus::FLOAT AS bonus_malus,\n",
    "    d.donnees_demande_json:demande:permis_depuis_annees::INTEGER AS permis_depuis_annees,\n",
    "    d.donnees_demande_json:demande:usage_vehicule::STRING AS usage_vehicule,\n",
    "    d.donnees_demande_json:demande:stationnement_nuit::STRING AS stationnement_nuit,\n",
    "    d.donnees_demande_json:demande:km_annuel_estime::INTEGER AS km_annuel_estime,\n",
    "    \n",
    "    -- Business logic enrichments\n",
    "    CASE \n",
    "        WHEN d.donnees_demande_json:demande:permis_depuis_annees::INTEGER < 2 THEN 'Nouveau conducteur'\n",
    "        WHEN d.donnees_demande_json:demande:permis_depuis_annees::INTEGER < 5 THEN 'Conducteur junior'\n",
    "        WHEN d.donnees_demande_json:demande:permis_depuis_annees::INTEGER < 10 THEN 'Conducteur expérimenté'\n",
    "        ELSE 'Conducteur senior'\n",
    "    END as profil_conducteur,\n",
    "    \n",
    "    CASE \n",
    "        WHEN d.donnees_demande_json:demande:sinistres_12_mois::INTEGER = 0 THEN 'Aucun sinistre'\n",
    "        WHEN d.donnees_demande_json:demande:sinistres_12_mois::INTEGER = 1 THEN 'Un sinistre'\n",
    "        ELSE 'Plusieurs sinistres'\n",
    "    END as profil_sinistres,\n",
    "    \n",
    "    CURRENT_TIMESTAMP() as date_creation\n",
    "FROM bronze.demandes d\n",
    "WHERE d.demande_id IS NOT NULL;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Final Data Quality Summary\n",
    "\n",
    "Provide a comprehensive overview of our Silver layer data quality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "language": "sql"
   },
   "outputs": [],
   "source": [
    "-- Final data quality summary\n",
    "SELECT \n",
    "    'CLIENTS' as table_name,\n",
    "    COUNT(*) as record_count,\n",
    "    COUNT(DISTINCT demande_id) as unique_keys,\n",
    "    COUNT(CASE WHEN email IS NULL THEN 1 END) as missing_critical_fields\n",
    "FROM silver.clients\n",
    "\n",
    "UNION ALL\n",
    "\n",
    "SELECT \n",
    "    'VEHICULES' as table_name,\n",
    "    COUNT(*) as record_count,\n",
    "    COUNT(DISTINCT demande_id) as unique_keys,\n",
    "    COUNT(CASE WHEN marque IS NULL THEN 1 END) as missing_critical_fields\n",
    "FROM silver.vehicules\n",
    "\n",
    "UNION ALL\n",
    "\n",
    "SELECT \n",
    "    'DEVIS_CLEAN' as table_name,\n",
    "    COUNT(*) as record_count,\n",
    "    COUNT(DISTINCT devis_id) as unique_keys,\n",
    "    COUNT(CASE WHEN data_quality_flag != 'VALID' THEN 1 END) as missing_critical_fields\n",
    "FROM silver.devis_clean\n",
    "\n",
    "UNION ALL\n",
    "\n",
    "SELECT \n",
    "    'DEMANDE_CLEAN' as table_name,\n",
    "    COUNT(*) as record_count,\n",
    "    COUNT(DISTINCT demande_id) as unique_keys,\n",
    "    COUNT(CASE WHEN statut_demande IS NULL THEN 1 END) as missing_critical_fields\n",
    "FROM silver.demande_clean;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "### What We Accomplished:\n",
    "1. **Data Quality Assessment**: Identified and flagged data quality issues\n",
    "2. **Business Logic**: Added meaningful categorizations and enrichments\n",
    "3. **Clean Views**: Created high-quality Silver layer views\n",
    "4. **Validation**: Confirmed data integrity and completeness\n",
    "\n",
    "### Next Steps:\n",
    "- Proceed to notebook 03 for Gold layer analytics\n",
    "- Use these clean Silver views for business intelligence\n",
    "- Monitor data quality over time\n",
    "\n",
    "### Key Learnings:\n",
    "- Data quality is critical for reliable analytics\n",
    "- Business logic should be applied at the Silver layer\n",
    "- Validation and monitoring are essential"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Snowflake",
   "language": "sql",
   "name": "snowflake"
  },
  "language_info": {
   "name": "sql",
   "version": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

---

## Deployment Scripts

### 1. Notebook Deployment SQL Script

**File: `setup/notebooks/deploy_notebooks.sql`**

```sql
-- ================================================================
-- Snowflake Notebook Deployment Script
-- ================================================================

-- Set context
USE DATABASE your_demo_database;
USE WAREHOUSE your_warehouse;

-- Create notebooks schema
CREATE SCHEMA IF NOT EXISTS notebooks
    COMMENT = 'Schema for Snowflake notebooks - alternative to DBT';

-- Deploy JSON Processing Notebook
CREATE OR REPLACE NOTEBOOK notebooks."01_JSON_PROCESSING"
    QUERY_WAREHOUSE = 'your_warehouse'
    MAIN_FILE = '01_json_processing.ipynb'
    COMMENT = 'JSON processing notebook - Bronze to Silver extraction';

ALTER NOTEBOOK notebooks."01_JSON_PROCESSING"
    ADD LIVE VERSION FROM LAST;

-- Deploy Data Quality Notebook
CREATE OR REPLACE NOTEBOOK notebooks."02_DATA_QUALITY"
    QUERY_WAREHOUSE = 'your_warehouse'
    MAIN_FILE = '02_data_quality.ipynb'
    COMMENT = 'Data quality notebook - Bronze to Silver enrichment';

ALTER NOTEBOOK notebooks."02_DATA_QUALITY"
    ADD LIVE VERSION FROM LAST;

-- Deploy Analytics Notebook
CREATE OR REPLACE NOTEBOOK notebooks."03_ANALYTICS"
    QUERY_WAREHOUSE = 'your_warehouse'
    MAIN_FILE = '03_analytics.ipynb'
    COMMENT = 'Analytics notebook - Silver to Gold metrics';

ALTER NOTEBOOK notebooks."03_ANALYTICS"
    ADD LIVE VERSION FROM LAST;

-- Deploy Validation Notebook
CREATE OR REPLACE NOTEBOOK notebooks."04_VALIDATION"
    QUERY_WAREHOUSE = 'your_warehouse'
    MAIN_FILE = '04_validation.ipynb'
    COMMENT = 'Validation notebook - Data quality checks';

ALTER NOTEBOOK notebooks."04_VALIDATION"
    ADD LIVE VERSION FROM LAST;

-- Verify notebook deployment
SHOW NOTEBOOKS IN SCHEMA notebooks;
```

### 2. Notebook Upload Script

**File: `setup/notebooks/upload_notebooks.sh`**

```bash
#!/bin/bash
# ================================================================
# Snowflake Notebook Upload Script
# ================================================================

# Configuration
DATABASE="your_demo_database"
SCHEMA="notebooks"
NOTEBOOKS_DIR="notebooks"

echo "Uploading Snowflake notebooks..."

# Check if SnowSQL is configured
if ! command -v snowsql &> /dev/null; then
    echo "Error: SnowSQL not found. Please install and configure SnowSQL."
    exit 1
fi

# Check if notebooks directory exists
if [ ! -d "$NOTEBOOKS_DIR" ]; then
    echo "Error: Notebooks directory not found: $NOTEBOOKS_DIR"
    exit 1
fi

# Upload JSON Processing Notebook
echo "Uploading 01_json_processing.ipynb..."
snowsql -q "PUT file://$NOTEBOOKS_DIR/01_json_processing.ipynb @\"$DATABASE\".\"$SCHEMA\".\"01_JSON_PROCESSING\"/versions/live/;"

# Upload Data Quality Notebook
echo "Uploading 02_data_quality.ipynb..."
snowsql -q "PUT file://$NOTEBOOKS_DIR/02_data_quality.ipynb @\"$DATABASE\".\"$SCHEMA\".\"02_DATA_QUALITY\"/versions/live/;"

# Upload Analytics Notebook
echo "Uploading 03_analytics.ipynb..."
snowsql -q "PUT file://$NOTEBOOKS_DIR/03_analytics.ipynb @\"$DATABASE\".\"$SCHEMA\".\"03_ANALYTICS\"/versions/live/;"

# Upload Validation Notebook
echo "Uploading 04_validation.ipynb..."
snowsql -q "PUT file://$NOTEBOOKS_DIR/04_validation.ipynb @\"$DATABASE\".\"$SCHEMA\".\"04_VALIDATION\"/versions/live/;"

echo "Verifying uploads..."
snowsql -q "LIST @\"$DATABASE\".\"$SCHEMA\".\"01_JSON_PROCESSING\"/versions/live/;"
snowsql -q "LIST @\"$DATABASE\".\"$SCHEMA\".\"02_DATA_QUALITY\"/versions/live/;"
snowsql -q "LIST @\"$DATABASE\".\"$SCHEMA\".\"03_ANALYTICS\"/versions/live/;"
snowsql -q "LIST @\"$DATABASE\".\"$SCHEMA\".\"04_VALIDATION\"/versions/live/;"

echo "Notebook deployment completed successfully!"
```

### 3. Complete Deployment Script

**File: `setup/notebooks/setup_notebooks.sh`**

```bash
#!/bin/bash
# ================================================================
# Complete Notebook Setup Script
# ================================================================

echo "Starting Snowflake notebook setup..."

# Step 1: Deploy notebook objects
echo "Step 1: Deploying notebook objects..."
snowsql -f setup/notebooks/deploy_notebooks.sql

if [ $? -eq 0 ]; then
    echo "✓ Notebook objects deployed successfully"
else
    echo "✗ Error deploying notebook objects"
    exit 1
fi

# Step 2: Upload notebook files
echo "Step 2: Uploading notebook files..."
bash setup/notebooks/upload_notebooks.sh

if [ $? -eq 0 ]; then
    echo "✓ Notebook files uploaded successfully"
else
    echo "✗ Error uploading notebook files"
    exit 1
fi

# Step 3: Verify deployment
echo "Step 3: Verifying deployment..."
snowsql -q "SHOW NOTEBOOKS IN SCHEMA notebooks;"

echo "✓ Snowflake notebook setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Access notebooks through Snowflake UI"
echo "2. Navigate to your_demo_database.notebooks schema"
echo "3. Open notebooks in order: 01, 02, 03, 04"
echo "4. Execute cells step by step for hands-on learning"
```

---

## Python Code Examples

### 1. Session Management

```python
# Standard session setup for Snowflake notebooks
from snowflake.snowpark.context import get_active_session
session = get_active_session()

# Verify session is active
print(f"Current database: {session.get_current_database()}")
print(f"Current schema: {session.get_current_schema()}")
print(f"Current warehouse: {session.get_current_warehouse()}")
```

### 2. Data Processing

```python
# Data processing with Snowpark
from snowflake.snowpark.functions import col, when, avg, count, sum as sum_
from snowflake.snowpark.types import IntegerType, StringType, FloatType
import pandas as pd

# Read data from Bronze layer
bronze_df = session.table("bronze.demandes")

# Process JSON data
processed_df = bronze_df.select(
    col("demande_id"),
    col("donnees_demande_json")["client"]["nom"].as_("client_nom"),
    col("donnees_demande_json")["client"]["age"].as_("client_age"),
    col("donnees_demande_json")["vehicule"]["marque"].as_("vehicule_marque")
).filter(col("donnees_demande_json")["client"]["nom"].is_not_null())

# Create business logic
enriched_df = processed_df.with_column(
    "age_category",
    when(col("client_age") < 25, "Young")
    .when(col("client_age") < 40, "Adult")
    .when(col("client_age") < 60, "Mature")
    .otherwise("Senior")
)

# Show results
enriched_df.show()
```

### 3. Data Visualization

```python
# Create visualizations
import plotly.express as px
import plotly.graph_objects as go

# Get data for visualization
viz_data = session.sql("""
    SELECT 
        vehicule_marque,
        COUNT(*) as count,
        AVG(client_age) as avg_age
    FROM silver.temp_processed_data
    GROUP BY vehicule_marque
    ORDER BY count DESC
    LIMIT 10
""").to_pandas()

# Create interactive chart
fig = px.bar(
    viz_data, 
    x='VEHICULE_MARQUE', 
    y='COUNT',
    title='Top Vehicle Brands by Request Count',
    labels={'COUNT': 'Number of Requests', 'VEHICULE_MARQUE': 'Vehicle Brand'}
)

fig.update_layout(
    xaxis_tickangle=-45,
    height=500
)

fig.show()
```

### 4. Data Quality Checks

```python
# Data quality validation
def validate_data_quality(table_name, session):
    """Validate data quality for a given table"""
    
    # Get basic statistics
    stats = session.sql(f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT demande_id) as unique_keys,
            COUNT(CASE WHEN demande_id IS NULL THEN 1 END) as null_keys
        FROM {table_name}
    """).collect()
    
    # Print results
    for row in stats:
        print(f"Table: {table_name}")
        print(f"Total records: {row['TOTAL_RECORDS']}")
        print(f"Unique keys: {row['UNIQUE_KEYS']}")
        print(f"Null keys: {row['NULL_KEYS']}")
        
        # Calculate data quality score
        quality_score = (row['UNIQUE_KEYS'] / row['TOTAL_RECORDS']) * 100
        print(f"Data quality score: {quality_score:.2f}%")
    
    return stats

# Use the validation function
validate_data_quality("silver.clients", session)
validate_data_quality("silver.vehicules", session)
```

---

## Best Practices

### 1. Notebook Structure

- **Clear Documentation**: Every notebook should have clear objectives and prerequisites
- **Step-by-Step Process**: Break complex transformations into logical steps
- **Validation**: Include data quality checks after each transformation
- **Visualization**: Use charts to show transformation results
- **Error Handling**: Include graceful error handling and troubleshooting tips

### 2. Code Organization

- **Modular Cells**: Each cell should have a single, clear purpose
- **Context Setting**: Always set database, schema, and warehouse context
- **Comments**: Add clear comments explaining business logic
- **Variable Names**: Use descriptive names for views and variables

### 3. Performance Considerations

- **Efficient Queries**: Use appropriate filters and aggregations
- **Result Limiting**: Use LIMIT for exploratory queries
- **Caching**: Leverage Snowflake's result caching
- **Monitoring**: Include query performance monitoring

### 4. Security and Compliance

- **No Hardcoded Values**: Use variables for database and schema names
- **Data Masking**: Consider data masking for sensitive information
- **Access Control**: Ensure appropriate role-based access
- **Audit Trail**: Include creation timestamps and user tracking

---

## Troubleshooting Guide

### Common Issues and Solutions

1. **Session Not Available**
   - Error: `get_active_session()` returns error
   - Solution: Ensure you're running in Snowflake notebook environment

2. **Permission Denied**
   - Error: Cannot create notebook or access data
   - Solution: Check role permissions and warehouse access

3. **JSON Path Issues**
   - Error: JSON path returns NULL
   - Solution: Verify JSON structure and path syntax

4. **Upload Failures**
   - Error: PUT command fails
   - Solution: Check file paths and SnowSQL configuration

5. **Notebook Not Visible**
   - Error: Notebook doesn't appear in UI
   - Solution: Check schema context and SHOW NOTEBOOKS command

### Debug Commands

```sql
-- Check notebook status
SHOW NOTEBOOKS IN SCHEMA notebooks;

-- Verify notebook details
DESC NOTEBOOK notebooks."01_JSON_PROCESSING";

-- List uploaded files
LIST @"database"."schema"."NOTEBOOK_NAME"/versions/live/;

-- Check permissions
SHOW GRANTS ON SCHEMA notebooks;
```

---

This comprehensive guide provides all the necessary templates, examples, and deployment scripts for creating and managing Snowflake notebooks as an alternative to DBT for data engineering pipelines. 
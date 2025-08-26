# Crocevia CRM Data Generation

A Snowflake/Snowpark-based data generation system for creating synthetic CRM datasets with controlled overlap patterns for data linkage and customer identity resolution testing.

## Overview

This project generates a large synthetic grocery/hypermarket CRM dataset (Crocevia) with controlled overlaps to an existing sports retailer CRM (Summit Sports), simulating real-world customer identity resolution challenges.

## Features

- **Snowpark Integration**: Runs natively in Snowflake using Snowpark for Python
- **Controlled Overlaps**: Precise control over email, phone, and name matching percentages
- **Realistic Noise**: Anonymous transactions, duplicates, missing data, cash transactions
- **French Localization**: French names, addresses, phone numbers, postal codes
- **Scalable**: Handles millions of records efficiently in Snowflake

## Quick Start

1. Set up Snowflake connection
2. Run database setup: `snowflake.sql`
3. Execute CRM generation: `snowpark/generate_crocevia_crm.py`

## Structure

```
/
├── snowflake.sql              # Database setup and export commands
├── snowpark/                  # Snowpark Python code
│   └── generate_crocevia_crm.py
├── setup/                     # Local Python utilities
├── data/                      # Data files
├── requirements.txt           # Python dependencies
└── environment.yml           # Conda environment
```

## Requirements

- Snowflake account with Snowpark enabled
- Python 3.11+
- Conda environment (see environment.yml)

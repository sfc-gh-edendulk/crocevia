# Synthetic Data Generation Rules for Snowflake Demos
# Strategic Framework for Realistic Demo Data Creation
# Version: 1.0 | Context: Cursor AI Conversation Analysis

## Executive Summary
This document outlines the strategic approach to generating realistic synthetic data for Snowflake demos, based on analysis of various tools and methodologies. The goal is to create industry-specific datasets that showcase Snowflake capabilities while maintaining business realism.

## Key Insights from Analysis

### Tool Comparison Summary
- **jafgen (Jaffle Shop)**: Best for retail/e-commerce, realistic business logic, minimal skew
- **Faker**: Best for custom industries, maximum flexibility, requires skew mitigation
- **Snowflake Native**: Best for showcasing platform capabilities, infinite customization
- **Cortex AI**: Best for content generation, multilingual support, text analysis demos

### Critical Issue: Data Skew
- **Faker Problem**: Generates uniform random data, lacks business realism
- **Real Business Data**: Contains natural skews (Pareto distributions, seasonal patterns, correlations)
- **Solution**: Layer business logic on top of raw generation tools

## Data Generation Framework

### Tier 1: Foundation Data
**Primary: jafgen Approach**
- Use jaffle_shop as base structure for retail scenarios
- Provides realistic customer behaviors, order patterns, product relationships
- Built-in business logic prevents unrealistic data distributions

**Secondary: Enhanced Faker**
- Custom industry templates with realistic distributions
- Business rules layer to create proper correlations
- Snowflake post-processing for final realism adjustments

### Tier 2: Industry Transformation
**Mapping Strategy:**
- Retail: Direct jaffle_shop usage
- Healthcare: Orders → Treatments, Customers → Patients, Products → Medications
- Finance: Orders → Transactions, Customers → Account Holders, Products → Financial Products
- Manufacturing: Orders → Work Orders, Customers → Suppliers, Products → Components
- Asset Management: Orders → Trades, Customers → Portfolio Managers, Products → Assets

### Tier 3: AI Enhancement
**Cortex AI Applications:**
- Generate realistic industry-specific text content
- Add sentiment analysis for customer feedback
- Create multilingual content for global demos
- Generate classification and tagging data

## Industry-Specific Rules

### Asset Management Data Requirements
**Core Entities:**
- Portfolio Managers (customers)
- Investment Accounts (portfolios)
- Assets (stocks, bonds, funds, alternatives)
- Transactions (trades, dividends, rebalancing)
- Performance Metrics (returns, risk measures, benchmarks)

**Realistic Distributions:**
- 80/20 rule: 20% of managers handle 80% of AUM
- Asset allocation follows institutional patterns
- Transaction frequency correlates with strategy type
- Performance should follow market-realistic patterns

**Temporal Patterns:**
- Trading activity higher during market hours
- Rebalancing typically quarterly/monthly
- Dividend payments on realistic schedules
- Market volatility affects transaction volumes

## Implementation Standards

### Data Quality Requirements
- No uniform random distributions for business metrics
- All financial data must follow realistic market patterns
- Customer behavior must reflect actual industry segments
- Geographic distribution must match real wealth concentration
- Temporal patterns must include market cycles and seasonality

### Scale Requirements
- 100 records: Quick demos, proof of concept
- 1,000 records: Detailed analysis, customer meetings
- 10,000 records: Performance testing, scalability demos
- 100,000 records: Enterprise-scale demonstrations

### Realism Validation
- Statistical tests for distribution appropriateness
- Business logic validation (e.g., no negative account balances)
- Temporal consistency checks
- Cross-entity relationship validation

## Tool Selection Matrix

| Industry | Primary Tool | Secondary Tool | Transformation Layer |
|----------|-------------|----------------|---------------------|
| Retail/E-commerce | jafgen | Snowflake Native | Minimal |
| Healthcare | Faker + Templates | Cortex AI | Heavy |
| Finance | Faker + Templates | Snowflake Native | Medium |
| Asset Management | Faker + Templates | Snowflake Native | Heavy |
| Manufacturing | jaffle_shop Transform | Faker | Medium |

## Asset Management Specific Implementation

### Data Model Design
```
Portfolio Managers (Customers)
├── manager_id, name, firm, aum_range, strategy_type
├── experience_years, risk_tolerance, geographic_focus

Investment Accounts (Portfolios)  
├── account_id, manager_id, account_type, inception_date
├── current_value, benchmark, investment_objective

Assets (Products)
├── asset_id, symbol, asset_class, sector, geography
├── price, volatility, liquidity_rank, market_cap

Transactions (Orders)
├── transaction_id, account_id, asset_id, transaction_type
├── quantity, price, timestamp, strategy_reason
```

### Business Logic Rules
- High-AUM managers have more diversified portfolios
- Transaction sizes correlate with account values
- Asset allocation reflects stated investment objectives
- Performance attribution follows factor model patterns
- Risk metrics correlate with portfolio composition

## Success Metrics
- Data passes statistical realism tests
- Business users recognize realistic patterns
- Demos effectively showcase Snowflake capabilities
- Data supports multiple use case demonstrations
- Generation time remains under 10 minutes for standard datasets

## Next Steps
1. Implement asset management data generator
2. Create validation framework for realism testing
3. Build industry transformation templates
4. Develop automated quality checks
5. Create performance benchmarking tools 
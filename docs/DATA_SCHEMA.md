# Data Schema Format

Assuming you want to create a table named MarketingData (you can change this name as needed) in a BigQuery Dataset, here is the DDL:

```sql
CREATE TABLE MeridianData (
  geo STRING NOT NULL,
  time DATE NOT NULL,
  Channel0_impression FLOAT64 NOT NULL,
  Channel1_impression FLOAT64 NOT NULL,
  Channel2_impression FLOAT64 NOT NULL,
  Channel3_impression FLOAT64 NOT NULL,
  Channel4_impression FLOAT64 NOT NULL,
  Channel5_impression FLOAT64 NOT NULL,
  Competitor_Sales FLOAT64 NOT NULL,
  Discount FLOAT64 NOT NULL,
  GQV FLOAT64 NOT NULL,
  Channel0_spend FLOAT64 NOT NULL,
  Channel1_spend FLOAT64 NOT NULL,
  Channel2_spend FLOAT64 NOT NULL,
  Channel3_spend FLOAT64 NOT NULL,
  Channel4_spend FLOAT64 NOT NULL,
  Channel5_spend FLOAT64 NOT NULL,
  conversions FLOAT64 NOT NULL,
  revenue_per_conversion FLOAT64 NOT NULL,
  population FLOAT64 NOT NULL,
  Channel4_reach FLOAT64 NOT NULL,
  Channel5_reach FLOAT64 NOT NULL,
  Channel4_frequency FLOAT64 NOT NULL,
  Channel5_frequency FLOAT64 NOT NULL
);
```

Explanation of Data Type Mapping:
* `STRING` is a `STRING` in GoogleSQL.
* `DATE` is a `DATE` in GoogleSQL.
* `FLOAT` is `FLOAT64` in GoogleSQL, which is a double-precision floating-point number.
* All columns are defined as non nullable

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
* INTEGER from the image is mapped to INT64 in GoogleSQL, which is a 64-bit signed integer.
* STRING from the image is mapped to STRING in GoogleSQL.
* DATE from the image is mapped to DATE in GoogleSQL.
* FLOAT from the image is mapped to FLOAT64 in GoogleSQL, which is a double-precision floating-point number. GoogleSQL also has a FLOAT32 type, but FLOAT64 is generally preferred for broader compatibility and precision unless storage is a critical concern. The alias FLOAT in GoogleSQL refers to FLOAT64.
* All columns are defined as non nullable

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


## Column Descriptions

Here are the column descriptions for each column in the `MarketingData` table, based on common marketing analytics terminology:

* **`geo`**: The geographic area or region to which the marketing data pertains (e.g., country, state, city, DMA).
* **`time`**: The date or time period for which the data was recorded in the format YYYY-MM-DD (idealy aggregated weekly).
* **`Channel0_impression`**: The total number of times advertisements or content from Channel 0 were displayed or seen.
* **`Channel1_impression`**: The total number of times advertisements or content from Channel 1 were displayed or seen.
* **`Channel2_impression`**: The total number of times advertisements or content from Channel 2 were displayed or seen.
* **`Channel3_impression`**: The total number of times advertisements or content from Channel 3 were displayed or seen.
* **`Channel4_impression`**: The total number of times advertisements or content from Channel 4 were displayed or seen.
* **`Channel5_impression`**: The total number of times advertisements or content from Channel 5 were displayed or seen.
* **`Competitor_Sales`** (Optional): Sales figures or estimated sales volume for a competitor or the overall competitor market within the specified geo and time.
* **`Discount`** (Optional): The average discount percentage or monetary amount offered on products/services during the specified period.
* **`GQV`** (Google Query Volume): Represents the volume or number of search queries performed on Google related to specific keywords, brand terms, or product categories within the specified geo and time. This indicates public interest and search trends.
* **`Channel0_spend`**: The amount of money spent on marketing activities for Channel 0.
* **`Channel1_spend`**: The amount of money spent on marketing activities for Channel 1.
* **`Channel2_spend`**: The amount of money spent on marketing activities for Channel 2.
* **`Channel3_spend`**: The amount of money spent on marketing activities for Channel 3.
* **`Channel4_spend`**: The amount of money spent on marketing activities for Channel 4.
* **`Channel5_spend`**: The amount of money spent on marketing activities for Channel 5.
* **`conversions`**: The total number of desired actions taken by users as a result of marketing efforts (e.g., sales, sign-ups, downloads).
* **`revenue_per_conversion`**: The average amount of revenue generated for each conversion.
* **`population`**: The total population in the specified geographic area (`geo`).
* **`Channel4_reach`**: The number of unique individuals who were exposed to advertisements or content from Channel 4.
* **`Channel5_reach`**: The number of unique individuals who were exposed to advertisements or content from Channel 5.
* **`Channel4_frequency`**: The average number of times an individual was exposed to advertisements or content from Channel 4 within a specific period.
* **`Channel5_frequency`**: The average number of times an individual was exposed to advertisements or content from Channel 5 within a specific period.

**Note on "ChannelX" columns:**
"Channel0" through "Channel5" represent different marketing channels. These could be specific platforms like: Search Engine Marketing (SEM), Social Media (e.g., Facebook, Instagram, TikTok), Display Advertising, Email Marketing, Television, Radio, Out-of-Home (OOH) advertising, etc.

The specific mapping of "Channel0" to "Channel5" would depend on the actual data source and how the channels were defined during data collection.

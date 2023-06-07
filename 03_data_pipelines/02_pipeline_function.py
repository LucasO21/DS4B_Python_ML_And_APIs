# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA PROCESSING PIPELINE FUNCTION
# ----

# LIBRARIES ----

# Core
import pandas as pd
import numpy as np

# EDA 
import re
import email_lead_scoring as els

# Data Import
df_leads = els.db_read_els_data()

df_tags = els.db_read_els_raw_table(table_name="Tags")


# 1.0 Create Processing Function ----
def process_lead_tags(df_leads, df_tags):
    
    
    # Leads Data
    df_1 = df_leads \
        .assign(optin_days = lambda x: (x["optin_time"] - x["optin_time"].max()).dt.days) \
        .assign(email_provider = lambda x: x["user_email"].str.split("@").str[1]) \
        .assign(tag_count_by_optin_day = lambda x: x["tag_count"] / abs(x["optin_days"] - 1))
        
    # Tags Wide Data
    df_2 = df_tags \
        .assign(value = lambda x: 1) \
        .pivot(
            index   = "mailchimp_id",
            columns = "tag",
            values  = "value"            
        ) \
        .fillna(value = 0) \
        .rename(columns = lambda x: x.replace("-", "_").lower()) \
        .add_prefix("tag_") \
        .reset_index()
    
    # Merge
    df_leads_tags = df_1 \
        .merge(df_2, how = "left") \
        .fillna({col: 0 for col in df_2.columns if col.startswith("tag_")})
    
    # High Cardinality
    # countries_to_keep =  els.explore_sales_by_category(data=df_leads_tags, category="country_code") \
    #     .query("sales >= 6") \
    #     .index \
    #     .to_list()
    
    countries_to_keep = ['US',
                    'IN',
                    'AU',
                    'UK',
                    'BR',
                    'CA',
                    'DE',
                    'FR',
                    'ES',
                    'MX',
                    'NL',
                    'SG',
                    'DK',
                    'MY',
                    'PL',
                    'AE',
                    'ID',
                    'CO',
                    'BE',
                    'JP',
                    'NG'
                ]

    df_leads_tags = df_leads_tags \
        .assign(country_code = np.where(df_leads_tags["country_code"] \
            .isin(countries_to_keep), df_leads_tags["country_code"], "other"))        
    
    # Return
    return df_leads_tags


# 2.0 Test Out ----
process_lead_tags(df_leads, df_tags).head()



# 3.0 Improve On Pipeline ----
def db_read_and_process_els_data(conn_string="sqlite://" + "/00_database/crm_database.sqlite"):
    
    df_leads = els.db_read_els_data(conn_string=conn_string)
    
    df_tags = els.db_read_els_raw_table(conn_string=conn_string, table_name="Tags")
    
    df = process_lead_tags(df_leads, df_tags)
    
    return(df)



# 4.0 Try Out Package ----

import email_lead_scoring as els

els.db_read_and_process_els_data()

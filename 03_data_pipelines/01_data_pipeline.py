# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 3: DATA PROCESSING PIPELINE 
# ----

# LIBRARIES ----

# Core
import pandas as pd
import numpy as np
# EDA 
import re
import janitor as jn
import sweetviz as sv
# Email Lead Scoring
import email_lead_scoring as els

# Recap  ---
# - We see some promise in the data
# - Now we need to continue to develop features
# - Prepare for Machine Learning

leads_df = els.db_read_els_data()

# 1.0 FEATURE EXPLORATION ----

# High Colinearity


# Ordinal Feature


# Interaction



# 2.0 BUILDING ENGINEERED FEATURES ----

# Date Features



# Email Features



# Activity Features


# 3.0 ONE-TO-MANY FEATURES  ----
# - TAGS: 1 Customer can have many tags

# Specific Tag Features (Actions)



# Merge Tags



# Fill NA selectively


# 4.0 ADJUSTING FEATURES ----
# - Country Code: Has high cardinality

# High Cardinality Features: Country Code



# 5.0 EXPLORATORY REPORT PART 2

# Examine with Sweetviz



# CONCLUSIONS ----

# - Member Rating: 5 has 14% of purchasers within group
# - Country Code: US getting 11% conversion, AU getting 12%
# - Tag Count: 
#    - At zero tags, almost no chance of purchase
#    - At 5 tags, about 10% chance of purchase
#    - At 10 tags, about 35% chance of purchase
# - Optin Days: After 100 days, increases to 2% to 4-5% likelihood
# - Ratio of Tag Counts to Length of Time: Goes from zero to 15% at 0.10 (1 tag for every 10 days on list)
# - Attending Learning Labs & Webinars: Increases from 2% to 10%




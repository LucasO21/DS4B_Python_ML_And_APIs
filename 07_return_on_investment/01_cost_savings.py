# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 7: ROI
# PART 1: COST VS SAVINGS TRADEOFF & ESTIMATE
# ----

# *************************************************************************
# LIBRARIES -----
# *************************************************************************

import pandas as pd
import numpy as np
import email_lead_scoring as els

# *************************************************************************
# RECAP ----
# *************************************************************************

leads_df = els.db_read_and_process_els_data()

leads_scored_df = els.model_score_leads(
    data       = leads_df,
    model_path = "models/pycaret/blended_tuned_models_calibrated_finalized"
)

# Optional (MLFlow)
# leads_scored_df_2 = els.mlflow_score_leads(
#     data=leads_df,
#     run_id = els.mlflow_get_best_run('automl_lead_scoring_1')
# )

# REVIEW COSTS ----

els.cost_calc_monthly_cost_table()

els.cost_simulate_unsub_cost()

# 1.0 LEAD TARGETING STRATEGY ----

# 1.1 Make Lead Strategy



# 1.2 Aggregate Results



# 2.0 CONFUSION MATRIX ANALYSIS ----

email_list_size = 1e5
unsub_rate_per_sales_email = 0.005
sales_emails_per_month = 5

avg_sales_per_month = 250000
avg_sales_emails_per_month = 5

customer_conversion_rate = 0.05
avg_customer_value = 2000

sample_factor = 5

# 2.1 Confusion Matrix Calculations ----


# 2.2 Confusion Matrix Summaries ----


# 3.0 PRELIMINARY EXPECTED VALUE CALCULATIONS

# 3.1 [Savings] Cold That Are Not Targeted



# 3.2 [Cost] Missed Sales That Are Not Targeted



# 3.3 [Cost] Hot Leads Targeted That Unsubscribe



# 3.4 [Savings] Sales Achieved



# 4.0 FINAL EXPECTED VALUE TO REPORT TO MANAGEMENT

# 4.1 Expected Monthly Sales (Realized)


# 4.2 Expected Monthly Value (Unrealized because of delayed nuture effect)


# 4.3 Expected Monthly Savings (Unrealized until nurture takes effect)


# 4.4 Expected Saved Customers (Unrealized until nuture takes effect)


# 4.5 EXPECTED VALUE SUMMARY OUTPUT





# CONCLUSIONS -----
# - Can save a lot of money with this strategy
# - But there is a tradeoff with the threshold selected
# - Need to somehow optimize for threshold




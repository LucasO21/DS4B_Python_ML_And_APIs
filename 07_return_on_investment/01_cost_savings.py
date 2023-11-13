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

# leads_scored_df = els.model_score_leads(
#     data       = leads_df,
#     model_path = "models/pycaret/blended_tuned_models_calibrated_finalized"
# )

leads_scored_df = els.model_score_leads(
    data       = leads_df,
    model_path = "models/pycaret/xgb_model_single_tuned_finalized"
)

# Optional (MLFlow)
# leads_scored_df_2 = els.mlflow_score_leads(
#     data=leads_df,
#     run_id = els.mlflow_get_best_run('automl_lead_scoring_1')
# )

# *************************************************************************
# REVIEW COSTS ----
# *************************************************************************

els.cost_calc_monthly_cost_table()

els.cost_simulate_unsub_cost()


# *************************************************************************
# 1.0 LEAD TARGETING STRATEGY ----
# *************************************************************************

# 1.1 Make Lead Strategy
leads_scored_small_df = leads_scored_df[['user_email', 'Score', 'made_purchase']]

leads_ranked_df = (
	leads_scored_small_df
		.sort_values('Score', ascending = False)
		.assign(rank = lambda x: np.arange(0, len(x['made_purchase'])) + 1)
		.assign(gain = lambda x: np.cumsum(x['made_purchase']) / np.sum(x['made_purchase']))
)

# Threshold Selection
threshold = 0.95

# Strategry Table
strategy_df = (
	leads_ranked_df
		.assign(category = lambda x: np.where(x['gain'] <= threshold, 'Hot-Lead', 'Cold-Lead'))
)

# Strategy for Marketing Table
strategy_for_marketing_df = (
	leads_scored_df
		.merge(
			right       = strategy_df[['category']],
			how         = 'left',
			left_index  = True,
			right_index = True
		)
)


# 1.2 Aggregate Results
strategy_aggregate_df = (
	strategy_df
		.groupby('category')
		.agg(
			count = ('made_purchase', 'count'),
			sum_made_purchase = ('made_purchase', 'sum')
		)
)


# *************************************************************************
# 2.0 CONFUSION MATRIX ANALYSIS ----
# *************************************************************************

email_list_size = 100000
unsub_rate_per_sales_email = 0.005
sales_emails_per_month = 5

avg_sales_per_month = 250000
avg_sales_emails_per_month = 5

customer_conversion_rate = 0.05
avg_customer_value = 2000

sample_factor = 5


# 2.1 Confusion Matrix Calculations ----
strategy_aggregate_df

# # Cold Lead Count Variable
# try:
#     cold_lead_count = strategy_aggregate_df['count']['Cold-Lead']
# except:
#     cold_lead_count = 0

# # Hot Lead Count Variable
# try:
#     hot_lead_count = strategy_aggregate_df['count']['Hot-Lead']
# except:
#     hot_lead_count = 0

# # Missed Purchases Variable
# try:
#     missed_purchases = strategy_aggregate_df['sum_made_purchase']['Cold-Lead']
# except:
#     missed_purchases = 0

# # Made Purchases Variable
# try:
#     made_purchases = strategy_aggregate_df['sum_made_purchase']['Hot-Lead']
# except:
#     made_purchases = 0

# Cold Lead Count Variable
cold_lead_count = strategy_aggregate_df['count'].get('Cold-Lead', 0)

# Hot Lead Count Variable
hot_lead_count = strategy_aggregate_df['count'].get('Hot-Lead', 0)

# Missed Purchases Variable
missed_purchases = strategy_aggregate_df['sum_made_purchase'].get('Cold-Lead', 0)

# Made Purchases Variable
made_purchases = strategy_aggregate_df['sum_made_purchase'].get('Hot-Lead', 0)


# 2.2 Confusion Matrix Summaries ----
total_count = (cold_lead_count + hot_lead_count)

total_purchases = (missed_purchases + made_purchases)

sample_factor = email_list_size / total_count

sales_per_email_sent = avg_sales_per_month / avg_sales_emails_per_month


# *************************************************************************
# 3.0 PRELIMINARY EXPECTED VALUE CALCULATIONS
# *************************************************************************

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




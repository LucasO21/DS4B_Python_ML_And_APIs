# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 7: ROI
# PART 2: THRESHOLD OPTIMIZATION & PROFIT MAXIMIZATION
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                   LIBRARIES                                  #
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import plotly.express as px
import email_lead_scoring as els


# ---------------------------------------------------------------------------- #
#                                  RECAP                                       #
# ---------------------------------------------------------------------------- #
leads_df = els.db_read_and_process_els_data()

leads_scored_df = els.model_score_leads(
	data       = leads_df,
	model_path = "models/pycaret/xgb_model_single_tuned_finalized"
)


 # ---------------------------------------------------------------------------- #
 #            1.0 MAKE THE LEAD STRATEGY FROM THE SCORED SUBSCRIBERS:           #
 # ---------------------------------------------------------------------------- #

#   lead_make_strategy()

def lead_make_strategy(
    data               = leads_scored_df,
    thresh             = 0.95,
    for_marketing_team = False,
    verbose            = False
):

    # ranking leads
    leads_ranked_df = (
        data
        .sort_values('Score', ascending=False)
        .assign(rank=lambda x: np.arange(0, len(x['made_purchase'])) + 1)
        .assign(gain=lambda x: np.cumsum(
            x['made_purchase']) / np.sum(x['made_purchase'])
        )
    )

    # make the strategy
    strategy_df = (
        leads_ranked_df
        .assign(category=lambda x: np.where(x['gain'] <= thresh, 'Hot-Lead', 'Cold-Lead'))
    )

    if for_marketing_team:
        strategy_for_marketing_df = (
            leads_scored_df
            .merge(
                right=strategy_df[['category']],
                how='left',
                left_index=True,
                right_index=True
            )
        )

    if verbose:
        print("Strategy Created.")

    # return
    return strategy_df

# ------------------------------------------------------------------------------------------------ #


# Workflow
lead_make_strategy(
	data   = leads_scored_df,
	thresh = 0.90,
	for_marketing_team = False
)



# ------------------------------------------------------------------------------------------------ #
#                              2.0 AGGREGATE THE LEAD STRATEGY RESULTS                             #
# ------------------------------------------------------------------------------------------------ #

#  lead_aggregate_strategy_results()

def lead_aggregate_strategy_results(data):

    # aggregate results
    results_df = (
		data
			.groupby('category')
			.agg(
				count = ('made_purchase', 'count'),
				sum_made_purchase = ('made_purchase', 'sum')
			)
	)

    # return
    return results_df

# ------------------------------------------------------------------------------------------------ #

# Workflow
(
	lead_make_strategy(leads_scored_df, thresh = 0.90)
		.pipe(lead_aggregate_strategy_results)
)


# ------------------------------------------------------------------------------------------------ #
#                       3.0 CALCULATE EXPECTED VALUE (FOR ONE GROUP OF INPUTS)                       #
# ------------------------------------------------------------------------------------------------ #

# lead_strategy_calc_expected_value()

def lead_strategy_calc_expected_value(
	data,
	email_list_size = 100000,
	unsub_rate_per_sales_email = 0.001,
	sales_emails_per_month = 5,
	avg_sales_per_month = 250000,
	avg_sales_emails_per_month = 5,
	customer_conversion_rate = 0.05,
	avg_customer_value = 2000,
	verbose = False
):
    # Define Variables ----
	cold_lead_count  = data['count'].get('Cold-Lead', 0)
	hot_lead_count   = data['count'].get('Hot-Lead', 0)
	missed_purchases = data['sum_made_purchase'].get('Cold-Lead', 0)
	made_purchases   = data['sum_made_purchase'].get('Hot-Lead', 0)


	# Confusion Matrix Summaries ----
	total_count          = (cold_lead_count + hot_lead_count)
	total_purchases      = (missed_purchases + made_purchases)
	sample_factor        = email_list_size / total_count
	sales_per_email_sent = avg_sales_per_month / avg_sales_emails_per_month

	# ---- PRELIMINARY EXPECTED VALUE CALCULATIONS ---- #
	# [Savings] Cold That Are Not Targeted ----
	savings_cold_no_target = (
		cold_lead_count
			* (sales_emails_per_month * unsub_rate_per_sales_email)
			* (customer_conversion_rate * avg_customer_value)
			* sample_factor

	)

 	# [Cost] Missed Sales That Are Not Targeted ----
	missed_purchase_ratio = missed_purchases / (missed_purchases + made_purchases)
	cost_missed_purchases = (sales_per_email_sent * sales_emails_per_month * missed_purchase_ratio)

	# [Cost] Hot Leads Targeted That Unsubscribe ----
	cost_hot_target_but_unsub = (
		hot_lead_count
			* sales_emails_per_month * unsub_rate_per_sales_email
			* customer_conversion_rate * avg_customer_value
			* sample_factor
	)

	# [Savings] Sales Achieved ----
	made_purchase_ratio = made_purchases / (missed_purchases + made_purchases)
	savings_made_purchases = (sales_per_email_sent * sales_emails_per_month * made_purchase_ratio)


	# ---- FINAL EXPECTED VALUE TO REPORT TO MANAGEMENT ---- #

	# Expected Monthly Sales (Realized) ----
	savings_made_purchases

	# Expected Monthly Value (Unrealized because of delayed nuture effect)
	ev = savings_made_purchases + savings_cold_no_target - cost_missed_purchases

	# Expected Monthly Savings (Unrealized until nurture takes effect)
	es = savings_cold_no_target - cost_missed_purchases

	# Expected Saved Customers (Unrealized until nuture takes effect)
	esc = savings_cold_no_target / avg_customer_value

	# ---- EXPECTED VALUE SUMMARY OUTPUT ---- #
	if verbose:
		print(f"Expected Value: {'${:,.0f}'.format(ev)}")
		print(f"Expected Savings: {'${:,.0f}'.format(es)}")
		print(f"Monthly Sales: {'${:,.0f}'.format(savings_made_purchases)}")
		print(f"Saved Customers: {'${:,.0f}'.format(esc)}")

	# Return ----
	return {
		'expected_value': ev,
		'expected_savings': es,
		'monthly_sales': savings_made_purchases,
		'expected_customers_saved': esc
	}



# Workflow:
lead_make_strategy(
	data = leads_scored_df,
	thresh = 0.99,
	verbose = True
) \
    .pipe(
        lead_aggregate_strategy_results) \
    .pipe(
        lead_strategy_calc_expected_value,
        email_list_size = 200000,
		unsub_rate_per_sales_email = 0.001,
		sales_emails_per_month = 5,
		avg_sales_per_month = 250000,
		avg_sales_emails_per_month = 5,
		customer_conversion_rate = 0.05,
		avg_customer_value = 2000,
		verbose = False
    )



# ------------------------------------------------------------------------------------------------ #
#                          4.0 OPTIMIZE THE THRESHOLD AND GENERATE A TABLE                         #
# ------------------------------------------------------------------------------------------------ #
#  lead_strategy_create_thresh_table()



# Workflow:



# 5.0 SELECT THE BEST THRESHOLD
#  def lead_select_optimum_thresh()



# Workflow



# 6.0 GET EXPECTED VALUE RESULTS ----
#  def lead_get_expected_value()




# 7.0 PLOT THE OPTIMAL THRESHOLD ----
#  def lead_plot_optim_thresh()



# Workflow



# 8.0 MAKE THE OPTIMAL STRATEGY ----



# FINAL OPTIMIZATION RESULTS ----
# - Use one function to perform the automation and collect the results
#   def lead_score_strategy_optimization()



# Workflow



# CONCLUSIONS ----

# Business Leaders may freak out if they see a big hit in sales
#  even though it could be beneficial in long term
# Recommend to start with a smaller strategy of focusing on
#  top 95%
# We can set this up with the Threshold Override

# ---------------------------------------------------------------------------- #
#                            Shift + Opt + X = Title                           #
# ---------------------------------------------------------------------------- #

# ---------------------------- Opt + X = Subtitle ---------------------------- #

# ---------------------------------------------------------------------------- #


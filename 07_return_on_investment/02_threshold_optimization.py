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
 #            1.0 MAKE THE LEAD STRATEGY FROM THE SCORED SUBSCRIBERS            #
 # ---------------------------------------------------------------------------- #

#   lead_make_strategy()

def lead_make_strategy(
    data               = leads_scored_df,
    thresh             = 0.95,
    for_marketing_team = False,
    verbose            = False
):

    # Rank Leads
    leads_scored_small_df = data[['user_email', 'Score', 'made_purchase']]

    leads_ranked_df = leads_scored_small_df \
		.sort_values('Score', ascending = False) \
		.assign(rank = lambda x: np.arange(0, len(x['made_purchase'])) + 1) \
		.assign(
      gain = lambda x: np.cumsum(x['made_purchase']) / np.sum(x['made_purchase'])
    )

    # Make Strategy
    strategy_df = leads_ranked_df \
        	.assign(
             category = lambda x: np.where(x['gain'] <= thresh, 'Hot-Lead', 'Cold-Lead')
        )

	# Format for Marketing
    if for_marketing_team:
        strategy_for_marketing_df = data \
            .merge(
                right       = strategy_df[['category']],
                how         = 'left',
                left_index  = True,
                right_index = True
            )

	# Verbose
    if verbose:
        print("===================================================================")
        print(f"lead_make_strategy: thresh = {thresh}, strategy created!")
        print("===================================================================")

    # Return
    return strategy_df

# ---- End Function ---- #


# Workflow
lead_make_strategy(
	data   = leads_scored_df,
	thresh = 0.90,
	for_marketing_team = False,
	verbose = True
)



# ------------------------------------------------------------------------------------------------ #
#                              2.0 AGGREGATE THE LEAD STRATEGY RESULTS                             #
# ------------------------------------------------------------------------------------------------ #

#  lead_aggregate_strategy_results()

def lead_aggregate_strategy_results(data):

    # Aggregate Results
    results_df = data \
			.groupby('category') \
			.agg(
				count = ('made_purchase', 'count'),
				sum_made_purchase = ('made_purchase', 'sum')
			)


    # return
    return results_df

# ---- End Function ---- #


# Workflow
lead_make_strategy(leads_scored_df, thresh = 0.90) \
	.pipe(lead_aggregate_strategy_results)



# ------------------------------------------------------------------------------------------------ #
#                       3.0 CALCULATE EXPECTED VALUE (FOR ONE GROUP OF INPUTS)                     #
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
    # Define Variables
	cold_lead_count  = data['count'].get('Cold-Lead', 0)
	hot_lead_count   = data['count'].get('Hot-Lead', 0)
	missed_purchases = data['sum_made_purchase'].get('Cold-Lead', 0)
	made_purchases   = data['sum_made_purchase'].get('Hot-Lead', 0)

	# Confusion Matrix Summaries
	total_count          = (cold_lead_count + hot_lead_count)
	total_purchases      = (missed_purchases + made_purchases)
	sample_factor        = email_list_size / total_count
	sales_per_email_sent = avg_sales_per_month / avg_sales_emails_per_month

	# ---- PRELIMINARY EXPECTED VALUE CALCULATIONS ---- #
	# [Savings] Cold That Are Not Targeted
	savings_cold_no_target = cold_lead_count \
			* (sales_emails_per_month * unsub_rate_per_sales_email) \
			* (customer_conversion_rate * avg_customer_value) \
			* sample_factor

 	# [Cost] Missed Sales That Are Not Targeted
	missed_purchase_ratio = missed_purchases / (missed_purchases + made_purchases)
	cost_missed_purchases = (sales_per_email_sent * sales_emails_per_month * missed_purchase_ratio)

	# [Cost] Hot Leads Targeted That Unsubscribe ----
	cost_hot_target_but_unsub = hot_lead_count \
			* sales_emails_per_month * unsub_rate_per_sales_email \
			* customer_conversion_rate * avg_customer_value \
			* sample_factor


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

	# Verbose
	if verbose:
		print(f"Expected Value: {'${:,.0f}'.format(ev)}")
		print(f"Expected Savings: {'${:,.0f}'.format(es)}")
		print(f"Monthly Sales: {'${:,.0f}'.format(savings_made_purchases)}")
		print(f"Saved Customers: {'${:,.0f}'.format(esc)}")

	# Return
	return {
		'expected_value': ev,
		'expected_savings': es,
		'monthly_sales': savings_made_purchases,
		'expected_customers_saved': esc
	}

 #! ---- End Function ---- #



# Workflow:
lead_make_strategy(
	data = leads_scored_df,
	thresh = 0.99,
	verbose = True
) \
    .pipe(lead_aggregate_strategy_results) \
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



# -------------------------------------------------------------------------------------------- #
#                          4.0 OPTIMIZE THE THRESHOLD AND GENERATE A TABLE                     #
# -------------------------------------------------------------------------------------------- #
# lead_strategy_create_thresh_table()
# Optimize for multiple thresholds

def lead_strategy_create_thresh_table(
	data,
    thresh                     = np.linspace(0, 1, num = 100),
	email_list_size            = 100000,
	unsub_rate_per_sales_email = 0.005,
	sales_emails_per_month     = 5,
	avg_sales_per_month        = 250000,
	avg_sales_emails_per_month = 5,
	customer_conversion_rate   = 0.05,
	avg_customer_value         = 2000,
	highlight_max              = True,
	highlight_max_color        = "green",
	verbose                    = False
):
    # Thereshold Table
    thresh_df = pd.Series(thresh, name = "thresh").to_frame()

    # List Comp
    #[tup[0] for tup in zip(thresh_df['thresh'])]

    sim_results_list = [
		lead_make_strategy(
			data = data,
			thresh = tup[0],
			verbose = verbose
		) \
			.pipe(
				lead_aggregate_strategy_results
    		) \
			.pipe(
				lead_strategy_calc_expected_value,
				email_list_size = email_list_size,
				unsub_rate_per_sales_email = unsub_rate_per_sales_email,
				sales_emails_per_month = sales_emails_per_month,
				avg_sales_per_month = avg_sales_per_month,
				avg_sales_emails_per_month = avg_sales_emails_per_month,
				customer_conversion_rate = customer_conversion_rate,
				avg_customer_value = avg_customer_value,
				verbose = verbose,
			)
		for tup in zip(thresh_df['thresh'])
	]

    # List to Frame
    sim_results_df = pd.Series(sim_results_list, name = 'sim_results').to_frame()

    sim_results_df = sim_results_df['sim_results'].apply(pd.Series)

    thresh_optim_df = pd.concat([thresh_df, sim_results_df], axis = 1)

    # Highlighting
    if highlight_max:
        thresh_optim_df = thresh_optim_df.style.highlight_max(
			color = highlight_max_color,
			axis = 0
		)

    # Return
    return thresh_optim_df

#! ---- End Function ---- #


# Workflow:
lead_strategy_create_thresh_table(
    data = leads_scored_df,
    highlight_max_color = "green",
    verbose = True
)

# data = lead_strategy_create_thresh_table(
#     data = leads_scored_df,
#     highlight_max_color = "green",
#     verbose = True
# )



# -------------------------------------------------------------------------------------- #
#                              5.0 DETERMINE THE BEST THRESHOLD                          #
# -------------------------------------------------------------------------------------- #
#  def lead_select_optimum_thresh()

def lead_select_optimum_thresh(
	data,
	optim_col = 'expected_value',
	monthly_sales_reduction_safe_guard = 0.90,
	verbose = False
):

    # Handle Styler Object
    try:
        data = data.data
    except:
        data = data

    # Find Optim
    _filter_1 = data[optim_col] == data[optim_col].max()

    # Find Safeguard
    _filter_2 = data['monthly_sales'] >= monthly_sales_reduction_safe_guard \
        * data['monthly_sales'].max()

    # Test if optim is in the safegard range
    if (all(_filter_1 + _filter_2 == _filter_2)):
        _filter = _filter_1
    else:
        _filter = _filter_2

    # Apply Filter
    thresh_selected = data[_filter].head(1)

    # Values
    ret = thresh_selected['thresh'].values[0]


    # Verbose
    if (verbose):
        print("===================================================================")
        print(f'lead_select_optimum_thresh: Optimal Threshold: {ret}')
        print("===================================================================")

    # Return
    return ret

#! ---- End Function ---- #


# Workflow
thresh_optim_df = lead_strategy_create_thresh_table(
    data = leads_scored_df,
    highlight_max_color = "green",
    verbose = True
)

thresh_optim = lead_select_optimum_thresh(
	data = thresh_optim_df,
	monthly_sales_reduction_safe_guard = 0.90
)



# -------------------------------------------------------------------------------------- #
#                             6.0 GET EXPECTED VALUE RESULTS                             #
# -------------------------------------------------------------------------------------- #
#  def lead_get_expected_value()

def lead_get_expected_value(data = thresh_optim_df, threshold = None, verbose = False):

     # Handle Styler Object
    try:
        data = data.data
    except:
        data = data

    # Expected Value Table
    df = data[data.thresh >= threshold].head(1)

    # Verbose
    if verbose:
        print("===================================================================")
        print("lead_get_expected_value: Expected Value Table:")
        print("===================================================================")
        print(df)

    # Return
    return df

#! ---- End Function ---- #


# workflow
lead_get_expected_value(
	data = thresh_optim_df,
	threshold = lead_select_optimum_thresh(
		data = thresh_optim_df,
		monthly_sales_reduction_safe_guard = 0.85
	)
)


# -------------------------------------------------------------------------------------- #
#                             7.0 PLOT THE OPTIMAL THRESHOLD                             #
# -------------------------------------------------------------------------------------- #
#  def lead_plot_optim_thresh()

# fig = px.line(
# 	data,
# 	x = 'thresh',
# 	y = 'expected_value'
# )

# fig.add_hline(y = 0, line_color = 'black')

# fig.add_vline(
#     x = lead_select_optimum_thresh(data),
#     line_color = 'red',
#     line_dash = 'dash'
# )

def lead_plot_optim_thresh(
    data,
    optim_col = 'expected_value',
    monthly_sales_reduction_safe_guard = 0.85,
    verbose = False
):

    # Handle Styler Object
    try:
        data = data.data
    except:
        data = data

    # ---- MAKE PLOT ----- #

    # Plot: Initial Plot
    fig = px.line(
		data,
		x = 'thresh',
		y = 'expected_value'
	)

	# Plot: Add Hline
    fig.add_hline(y = 0, line_color = 'black')

	# Plot:: Add Vline
    fig.add_vline(
		x = lead_select_optimum_thresh(
				data = data,
				optim_col = optim_col,
				monthly_sales_reduction_safe_guard = monthly_sales_reduction_safe_guard
		),
		line_color = 'red',
		line_dash = 'dash'
	)

    # Verbose
    if verbose:
        print("===================================================================")
        print("lead_plot_optin_thresh: Plot Created!")
        print("===================================================================")

    # Return
    return fig

#! ---- End Function ---- #


# Workflow
lead_plot_optim_thresh(
	data = thresh_optim_df,
	optim_col = 'expected_value',
	monthly_sales_reduction_safe_guard = 0.80,
	verbose = True
)



# -------------------------------------------------------------------------------------- #
#                              8.0 MAKE THE OPTIMAL STRATEGY                             #
# -------------------------------------------------------------------------------------- #

# FINAL OPTIMIZATION RESULTS ----
# - Use one function to perform the automation and collect the results
#   def lead_score_strategy_optimization()

def lead_score_strategy_optimization(
    data,

    thresh = np.linspace(0, 1, num = 100),
    optim_col = 'expected_value',
	monthly_sales_reduction_safe_guard = 0.90,
	for_marketing_team = True,

	email_list_size = 100000,
	unsub_rate_per_sales_email = 0.005,
	sales_emails_per_month = 5,
	avg_sales_per_month = 250000,
	avg_sales_emails_per_month = 5,
	customer_conversion_rate = 0.05,
	avg_customer_value = 2000,
	highlight_max = True,
	highlight_max_color = "green",

	verbose = False

):

    # Lead Strategy Create Thresh Table
    thresh_optim_df = lead_strategy_create_thresh_table(
		data = data,
		thresh = thresh,
  		email_list_size = email_list_size,
		unsub_rate_per_sales_email = unsub_rate_per_sales_email,
		sales_emails_per_month = sales_emails_per_month,
		avg_sales_per_month = avg_sales_per_month,
		avg_sales_emails_per_month = avg_sales_emails_per_month,
		customer_conversion_rate = customer_conversion_rate,
		avg_customer_value = avg_customer_value,
		highlight_max = highlight_max,
		highlight_max_color = highlight_max_color,
		verbose = verbose

	)

    # Lead Select Optimum Thresh
    thresh_optim = lead_select_optimum_thresh(
		data = thresh_optim_df,
		optim_col = optim_col,
		monthly_sales_reduction_safe_guard = monthly_sales_reduction_safe_guard,
		verbose = verbose
	)

    # Expected Value
    expected_value = lead_get_expected_value(
		data = thresh_optim_df,
		threshold = thresh_optim,
		verbose = verbose
	)

    # Lead Plot
    thresh_plot = lead_plot_optim_thresh(
		data = thresh_optim_df,
		optim_col = optim_col,
		monthly_sales_reduction_safe_guard = monthly_sales_reduction_safe_guard,
		verbose = verbose
	)

    # Re-Calculate Lead Strategy
    lead_strategy_df = lead_make_strategy(
		data = data,
		thresh = thresh_optim,
		for_marketing_team = for_marketing_team,
		verbose = verbose
	)

    # Return Dictionary
    ret = dict(
		lead_strategy_df = lead_strategy_df,
		expected_value = expected_value,
		thresh_optim_df = thresh_optim_df,
		thresh_plot = thresh_plot
	)

    # Return
    return ret

    #! ---- End Function ---- #



# Workflow
optimization_results_dict = lead_score_strategy_optimization(
	data = leads_scored_df,
	monthly_sales_reduction_safe_guard = 0.90,
	verbose = True
)

optimization_results_dict.keys()

optimization_results_dict['lead_strategy_df']

optimization_results_dict['expected_value']

optimization_results_dict['thresh_optim_df']

optimization_results_dict['thresh_plot']



# CONCLUSIONS ----

# Business Leaders may freak out if they see a big hit in sales
#  even though it could be beneficial in long term
# Recommend to start with a smaller strategy of focusing on
#  top 95%
# We can set this up with the Threshold Override





# -------------------------------------------------------------------------------------- #
#                                         IMPORTS                                        #
# -------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import plotly.express as px
import email_lead_scoring as els


# -------------------------------------------------------------------------------------- #
#                                   LEAD MAKE STRATEGY                                   #
# -------------------------------------------------------------------------------------- #
def lead_make_strategy(
    data,
    thresh             = 0.95,
    for_marketing_team = False,
    verbose            = False
):
    """
    Generate a lead strategy based on lead scores.

    Parameters:
    - data: DataFrame,
        The input data containing lead scores. In this workflow, this data should be
        called `lead_scored_df`.
    - thresh: float, default=0.95
        The threshold value for categorizing leads as Hot-Lead or Cold-Lead.
    - for_marketing_team: bool, default=False
        Flag indicating whether to format the strategy for the marketing team.
    - verbose: bool, default=False
        Flag indicating whether to print verbose information.

    Returns:
    - strategy_df: DataFrame
        The lead strategy DataFrame containing lead ranks and categories.
    """

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

#! ---- End Function ---- #


# -------------------------------------------------------------------------------------- #
#                               AGGREGATE STRATEGY RESULTS                               #
# -------------------------------------------------------------------------------------- #
def lead_aggregate_strategy_results(data):
    """
    Aggregate the results of the lead strategy.

    Parameters:
    - data: DataFrame
        The input data containing lead strategy results. In this workflow, this should be
        called `strategy_df` and is the output of the `lead_make_strategy()` function.

    Returns:
    - results_df: DataFrame
        The aggregated results DataFrame containing the count and sum of made purchases for each category.
    """
    # Aggregate Results
    results_df = data \
            .groupby('category') \
            .agg(
                count = ('made_purchase', 'count'),
                sum_made_purchase = ('made_purchase', 'sum')
            )

    # return
    return results_df

#! ---- End Function ---- #


# -------------------------------------------------------------------------------------- #
#                                CALCULATE EXPECTED VALUE                                #
# -------------------------------------------------------------------------------------- #
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
    """
    Calculate the expected value of the lead strategy.

    Parameters:
    - data: DataFrame
        The input data containing lead strategy results. In this workflow, this should be
        called `results_df` and is the output of the `lead_aggregate_strategy_results()` function.
    - email_list_size: int, default=100000
        The size of the email list.
    - unsub_rate_per_sales_email: float, default=0.001
        The unsubscribe rate per sales email.
    - sales_emails_per_month: int, default=5
        The number of sales emails sent per month.
    - avg_sales_per_month: int, default=250000
        The average sales per month.
    - avg_sales_emails_per_month: int, default=5
        The average number of sales emails sent per month.
    - customer_conversion_rate: float, default=0.05
        The customer conversion rate.
    - avg_customer_value: int, default=2000
        The average customer value.
    - verbose: bool, default=False
        Flag indicating whether to print verbose information.

    Returns:
    - expected_value: float
        The expected value of the lead strategy.
    - expected_savings: float
        The expected savings of the lead strategy.
    - monthly_sales: float
        The expected monthly sales.
    - expected_customers_saved: float
        The expected number of customers saved.
    """
    # Define Variables
    cold_lead_count = data['count'].get('Cold-Lead', 0)
    hot_lead_count = data['count'].get('Hot-Lead', 0)
    missed_purchases = data['sum_made_purchase'].get('Cold-Lead', 0)
    made_purchases = data['sum_made_purchase'].get('Hot-Lead', 0)

    # Confusion Matrix Summaries
    total_count = (cold_lead_count + hot_lead_count)
    total_purchases = (missed_purchases + made_purchases)
    sample_factor = email_list_size / total_count
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

    # Expected Monthly Value (Unrealized because of delayed nurture effect)
    ev = savings_made_purchases + savings_cold_no_target - cost_missed_purchases

    # Expected Monthly Savings (Unrealized until nurture takes effect)
    es = savings_cold_no_target - cost_missed_purchases

    # Expected Saved Customers (Unrealized until nurture takes effect)
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


# -------------------------------------------------------------------------------------- #
#                       OPTIMIZE THE THRESHOLD AND GENERATE A TABLE                      #
# -------------------------------------------------------------------------------------- #
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
    """
    Optimize the threshold and generate a table of lead strategy results.

    Parameters:
    - data: DataFrame
        The input data containing lead scores. In this workflow, this data should be
        called `lead_scored_df`.
    - thresh: array-like, default=np.linspace(0, 1, num=100)
        The array of threshold values to test.
    - email_list_size: int, default=100000
        The size of the email list.
    - unsub_rate_per_sales_email: float, default=0.005
        The unsubscribe rate per sales email.
    - sales_emails_per_month: int, default=5
        The number of sales emails sent per month.
    - avg_sales_per_month: int, default=250000
        The average sales per month.
    - avg_sales_emails_per_month: int, default=5
        The average number of sales emails sent per month.
    - customer_conversion_rate: float, default=0.05
        The customer conversion rate.
    - avg_customer_value: int, default=2000
        The average customer value.
    - highlight_max: bool, default=True
        Flag indicating whether to highlight the maximum values in the table.
    - highlight_max_color: str, default="green"
        The color to use for highlighting the maximum values.
    - verbose: bool, default=False
        Flag indicating whether to print verbose information.

    Returns:
    - thresh_optim_df: DataFrame
        The table of lead strategy results for each threshold value.
    """
    # Thereshold Table
    thresh_df = pd.Series(thresh, name="thresh").to_frame()

    # List Comp
    sim_results_list = [
        lead_make_strategy(
            data=data,
            thresh=tup[0],
            verbose=verbose
        ).pipe(
            lead_aggregate_strategy_results
        ).pipe(
            lead_strategy_calc_expected_value,
            email_list_size=email_list_size,
            unsub_rate_per_sales_email=unsub_rate_per_sales_email,
            sales_emails_per_month=sales_emails_per_month,
            avg_sales_per_month=avg_sales_per_month,
            avg_sales_emails_per_month=avg_sales_emails_per_month,
            customer_conversion_rate=customer_conversion_rate,
            avg_customer_value=avg_customer_value,
            verbose=verbose,
        )
        for tup in zip(thresh_df['thresh'])
    ]

    # List to Frame
    sim_results_df = pd.Series(sim_results_list, name='sim_results').to_frame()

    sim_results_df = sim_results_df['sim_results'].apply(pd.Series)
    thresh_optim_df = pd.concat([thresh_df, sim_results_df], axis=1)

    # Highlighting
    if highlight_max:
        thresh_optim_df = thresh_optim_df.style.highlight_max(
            color=highlight_max_color,
            axis=0
        )

    # Return
    return thresh_optim_df

#! ---- End Function ---- #


# -------------------------------------------------------------------------------------- #
#                              DETERMINE THE BEST THRESHOLD                              #
# -------------------------------------------------------------------------------------- #
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

# -------------------------------------------------------------------------------------- #
#                                 GET EXPECTED VALUE RESULTS                             #
# -------------------------------------------------------------------------------------- #
def lead_get_expected_value(data, threshold = None, verbose = False):

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


# -------------------------------------------------------------------------------------- #
#                                 PLOT THE OPTIMAL THRESHOLD                             #
# -------------------------------------------------------------------------------------- #
def lead_plot_optim_thresh(
    data,
    optim_col = 'expected_value',
    monthly_sales_reduction_safe_guard = 0.85,
    #fig_title = "Expected Value by Threshold Plot",
    verbose = False
):

    # Handle Styler Object
    try:
        data = data.data
    except:
        data = data

    # ---- MAKE PLOT ----- #

    # Fig: Initial Plot
    fig = px.line(
		data,
		x = 'thresh',
		y = 'expected_value'
	)

	# Fig: Add Hline
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

    # Fig Format Y-Axis Label
    fig.update_yaxes(tickformat = '$,.0f')

    # Fig Adjust Y-Axlis Label
    fig.update_yaxes(ticklabelposition = "inside top")

    # Fig Tooltip
    fig.update_traces(
        hovertemplate=(
            "<b>Thresh:</b> %{x:.2f}<br>"
            "<b>Expected Value:</b> %{y:$,.0f}<br>"
            "<b>Monthly Sales:</b> %{customdata[1]:$,.0f}<br>"
            "<b>Expected Customers Saved:</b> %{customdata[0]:.0f}<extra></extra>"
        ),
        customdata=np.stack((data['expected_customers_saved'], data['monthly_sales']), axis=-1),
        hoverlabel=dict(font_size=16)  # Increase the font size as needed
    )

    # Fig Title
    # fig.update_layout(title = fig_title)


    # Verbose
    if verbose:
        print("===================================================================")
        print("lead_plot_optin_thresh: Plot Created!")
        print("===================================================================")

    # Return
    return fig

#! ---- End Function ---- #


# -------------------------------------------------------------------------------------- #
#                                  MAKE THE OPTIMAL STRATEGY                             #
# -------------------------------------------------------------------------------------- #
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

    #fig_title = "Expected Value Plot",

	verbose = False

):
    """
    Lead score strategy and optimization that returns:

    1. lead_strategy_df
    2. expected_value
    3. thresh_optim_df
    4. thresh_plot

    This function optimizes the lead scoring strategy by testing various thresholds
    and evaluating their impact on key performance metrics. It returns a comprehensive
    analysis including an optimized threshold, expected values, and visual representations
    for decision-making support.

    Args:
        - data (DataFrame): Output of `els.model_score_leads(leads_df)`

        - thresh (ndarray, optional): Threshold values to test for optimization. Defaults to np.linspace(0, 1, num=100).

        - optim_col (str, optional): The column name to optimize. Defaults to 'expected_value'.

        - monthly_sales_reduction_safe_guard (float, optional): Minimum acceptable monthly sales reduction as a percentage of total sales. Defaults to 0.90.
        for_marketing_team (bool, optional): Whether to format `lead_strategy_df` as dataframe
        for the marketing team. Defaults to True.

        - email_list_size (int, optional): The size of the email list. Defaults to 100000.

        - unsub_rate_per_sales_email (float, optional): Unsubscription rate per sales email. Defaults to 0.005.

        - sales_emails_per_month (int, optional): Number of sales emails sent per month. Defaults to 5.

        - avg_sales_per_month (int, optional): Average sales per month. Defaults to 250000.

        - avg_sales_emails_per_month (int, optional): Average number of sales emails per month. Defaults to 5.

        - customer_conversion_rate (float, optional): Conversion monthly rate of customers. Defaults to 0.05.

        - avg_customer_value (int, optional): Average value of a customer. Defaults to 2000.

        - highlight_max (bool, optional): Whether to highlight the maximum values. Defaults to True.

        - highlight_max_color (str, optional): Color to use for highlighting maximum values. Defaults to "green".

        - verbose (bool, optional): If True, prints additional information. Defaults to False.

        Returns:
        dict: A dictionary containing the lead strategy DataFrame, expected value, threshold
        optimization DataFrame, and threshold plot.
    """

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
        #fig_title = fig_title,
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
        #fig_title = fig_title,
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






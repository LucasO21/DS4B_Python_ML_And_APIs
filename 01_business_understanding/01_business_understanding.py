# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING
# ----

# LIBRARIES ----
import pandas as pd
import numpy as np
import janitor as jn
import plotly.express as px


# BUSINESS SCIENCE PROBLEM FRAMEWORK ----

# View Business as a Machine ----


# Business Units:
#   - Marketing Department
#   - Responsible for sales emails
# Project Objectives:
#   - Target Subscribers Likely To Purchase
#   - Nurture Subscribers to take actions that are known to increase probability of purchase
# Define Machine:
#   - Marketing sends out email blasts to everyone
#   - Generates Sales
#   - Also ticks some members off
#   - Members unsubscribe, this reduces email growth and profitability
# Collect Outcomes:
#   - Revenue has slowed, Email growth has slowed


# Understand the Drivers ----

#   - Key Insights:
#     - Company has Large Email List: 100,000
#     - Email list is growing at 6,000/month less 3500 unsub for total of 2500
#     - High unsubscribe rates: 500 people per sales email
#   - Revenue:
#     - Company sales cycle is generating about $250,000 per month
#     - Average Customer Lifetime Value: Estimate $2000/customer
#   - Costs:
#     - Marketing sends 5 Sales Emails Per Month
#     - 5% of lost customers likely to convert if nutured


# COLLECT OUTCOMES ----
email_list_size_1 = 100000

unsub_count_per_sales_email_1 = 500

unsub_rate_1 = unsub_count_per_sales_email_1 / email_list_size_1

sales_emails_per_month_1 = 5

conversion_rate_1 = 0.05

lost_customers_1 = email_list_size_1 * unsub_rate_1 * \
    sales_emails_per_month_1 * conversion_rate_1

average_customer_value_1 = 2000

lost_revenue_per_month_1 = lost_customers_1 * average_customer_value_1


# No-growth scenario $3M cost
annual_cost_no_growth_1 = lost_revenue_per_month_1 * 12


# 2.5% Growth Scenario:
# amount = principle * ((1+rate)**time)

growth_rate = 3500 / 100000

100000 * ((1 + growth_rate) ** 0)

100000 * ((1 + growth_rate) ** 1)

100000 * ((1 + growth_rate) ** 12)


# Cost Table
time = 12

# Period
period_series = pd.Series(np.arange(0, 12), name="period")

cost_table_df = period_series.to_frame()

# Email Size - No Growth
cost_table_df["email_size_no_growth"] = np.repeat(email_list_size_1, time)

# Lost Customers - No Growth
cost_table_df["lost_customers_no_growth"] = cost_table_df["email_size_no_growth"] * \
    unsub_rate_1 * sales_emails_per_month_1

# Cost - No Growth
cost_table_df["cost_no_growth"] = cost_table_df["lost_customers_no_growth"] * \
    conversion_rate_1 * average_customer_value_1

# Email Size - With Growth
cost_table_df["email_size_with_growth"] = cost_table_df["email_size_no_growth"] * \
    ((1 + growth_rate) ** cost_table_df["period"])

# Lost Customers - With Growth
cost_table_df["lost_customers_with_growth"] = cost_table_df["email_size_with_growth"] * \
    unsub_rate_1 * sales_emails_per_month_1

# Cost - With Growth
cost_table_df["cost_with_growth"] = cost_table_df["lost_customers_with_growth"] * \
    conversion_rate_1 * average_customer_value_1

# Compare Cost - With / No Growth
cost_table_df[["cost_no_growth", "cost_with_growth"]] \
    .sum()

# If reduce unsubscribe rate by 30%
cost_table_df["cost_no_growth"].sum() * 0.30

cost_table_df["cost_with_growth"].sum() * 0.30


# COST CALCULATION FUNCTIONS ----

# Function: Calculate Monthly Unsubscriber Cost Table ----
def cost_calc_monthly_cost_table(
    email_list_size=100000,
    email_list_growth_rate=0.035,
    sales_emails_per_month=5,
    unsub_rate_per_sales_email=0.005,
    customer_conversion_rate=0.05,
    average_customer_value=2000,
    n_periods=24
):

    # Period
    period_series = pd.Series(np.arange(0, n_periods), name="period")

    # Cost Table
    cost_table_df = period_series.to_frame()

    # No Growth & Growth Scenarios
    cost_table_df = cost_table_df \
        .assign(email_size_no_growth=np.repeat(email_list_size, n_periods)) \
        .assign(lost_customers_no_growth=lambda x: x["email_size_no_growth"] * unsub_rate_per_sales_email * sales_emails_per_month) \
        .assign(cost_no_growth=lambda x: x["lost_customers_no_growth"] * customer_conversion_rate * average_customer_value) \
        .assign(email_size_with_growth=lambda x: x["email_size_no_growth"] * ((1 + email_list_growth_rate) ** x["period"])) \
        .assign(lost_customers_with_growth=lambda x: x["email_size_with_growth"] * unsub_rate_per_sales_email * sales_emails_per_month) \
        .assign(cost_with_growth=lambda x: x["lost_customers_with_growth"] * customer_conversion_rate * average_customer_value)

    return cost_table_df


df = cost_calc_monthly_cost_table(
    email_list_size=50000,
    sales_emails_per_month=1,
    unsub_rate_per_sales_email=0.001,
    n_periods=24
)


# Function: Sumarize Cost ----
def cost_total_unsub_cost(cost_table):

    summary_df = cost_table[["cost_no_growth", "cost_with_growth"]] \
        .sum() \
        .to_frame() \
        .transpose()

    return summary_df


cost_total_unsub_cost(df)

# ARE OBJECTIVES BEING MET?
# - We can see a large cost due to unsubscription
# - However, some attributes may vary causing costs to change


# SYNTHESIZE OUTCOMES (COST SIMULATION) ----
# - Make a cartesian product of inputs that can vary
# - Use list comprehension to perform simulation
# - Visualize results

# Cartesian Product (Expand Grid)


# Function


# VISUALIZE COSTS


# Function: Plot Simulated Unsubscriber Costs


# ARE OBJECTIVES BEING MET?
# - Even with simulation, we see high costs
# - What if we could reduce by 30% through better targeting?


# - What if we could reduce unsubscribe rate from 0.5% to 0.17% (marketing average)?
# - Source: https://www.campaignmonitor.com/resources/knowledge-base/what-is-a-good-unsubscribe-rate/


# HYPOTHESIZE DRIVERS

# - What causes a customer to convert of drop off?
# - If we know what makes them likely to convert, we can target the ones that are unlikely to nurture them (instead of sending sales emails)
# - Meet with Marketing Team
# - Notice increases in sales after webinars (called Learning Labs)
# - Next: Begin Data Collection and Understanding

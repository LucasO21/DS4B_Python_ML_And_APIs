# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING
# ----

# LIBRARIES ----
import pandas as pd
import numpy as np
import janitor as jn
from itertools import product
import plotly.express as px
from mizani.formatters import custom_format
from plotnine import *
import locale


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


# Cost Table ----
time = 12

# Period
period_series = pd.Series(np.arange(0, 12), name="period")

cost_table_df = period_series.to_frame()

# Email Size - No Growth
cost_table_df["email_size_no_growth"] = np.repeat(email_list_size_1, time)

# Lost Customers - No Growth
cost_table_df["lost_customers_no_growth"] = cost_table_df["email_size_no_growth"] * \
    unsub_rate_1 * sales_emails_per_month_1 * conversion_rate_1

# Cost - No Growth
cost_table_df["cost_no_growth"] = cost_table_df["lost_customers_no_growth"] * average_customer_value_1

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

df = cost_table_df.copy()

df["Email Size (No Growth)"] = df["email_size_no_growth"].apply(
    lambda x: "{:.2f}".format(x))


# COST CALCULATION FUNCTIONS ----

# Function: Calculate Monthly Unsubscriber Cost Table ----
def cost_calc_monthly_cost_table(
    email_list_size            = 100000,
    email_list_growth_rate     = 0.035,
    sales_emails_per_month     = 5,
    unsub_rate_per_sales_email = 0.005,
    customer_conversion_rate   = 0.05,
    average_customer_value     = 2000,
    n_periods                  = 12
):

    # Period
    period_series = pd.Series(np.arange(0, n_periods), name="period")

    # Cost Table
    cost_table_df = period_series.to_frame()

    # No Growth & Growth Scenarios
    cost_table_df = cost_table_df \
        .assign(email_size_no_growth=np.repeat(email_list_size, n_periods)) \
        .assign(lost_customers_no_growth=lambda x: x["email_size_no_growth"] * unsub_rate_per_sales_email * sales_emails_per_month * customer_conversion_rate) \
        .assign(cost_no_growth=lambda x: x["lost_customers_no_growth"] * average_customer_value) \
        .assign(email_size_with_growth=lambda x: x["email_size_no_growth"] * ((1 + email_list_growth_rate) ** x["period"])) \
        .assign(lost_customers_with_growth=lambda x: x["email_size_with_growth"] * unsub_rate_per_sales_email * sales_emails_per_month * customer_conversion_rate) \
        .assign(cost_with_growth=lambda x: x["lost_customers_with_growth"] * average_customer_value)

    return cost_table_df


cost_calc_monthly_cost_table(
    email_list_size            = 100000,
    sales_emails_per_month     = 1,
    unsub_rate_per_sales_email = 0.005,
    n_periods                  = 12
)


# Function: Sumarize Cost ----
def cost_total_unsub_cost(cost_table):

    summary_df = cost_table[["cost_no_growth", "cost_with_growth"]] \
        .sum() \
        .to_frame() \
        .transpose()

    return summary_df


cost_total_unsub_cost(cost_table_df)

# ARE OBJECTIVES BEING MET?
# - We can see a large cost due to unsubscription
# - However, some attributes may vary causing costs to change


# SYNTHESIZE OUTCOMES (COST SIMULATION) ----------------------------------------------
# - Make a cartesian product of inputs that can vary
# - Use list comprehension to perform simulation
# - Visualize results

# Cartesian Product (Expand Grid)

data_dict = dict(
    email_list_monthly_growth_rate=np.linspace(0, 0.05, num=10),
    customer_conversion_rate=np.linspace(0.04, 0.06, num=3)
)

combinations = list(
    product(
        data_dict['email_list_monthly_growth_rate'],
        data_dict['customer_conversion_rate']
    )
)

parameter_grid_df = pd.DataFrame(
    combinations,
    columns=['email_list_monthly_growth_rate', 'customer_conversion_rate']
)

# parameter_grid_df = jn.expand_grid(others=data_dict)


# List Comprehension
def temporary_function(x, y):

    cost_table_df = cost_calc_monthly_cost_table(
        email_list_growth_rate=x,
        customer_conversion_rate=y
    )

    summary_df = cost_total_unsub_cost(cost_table_df)

    return summary_df


temporary_function(x=0.035, y=0.05)


summary_list = [
    temporary_function(x, y) for x, y in zip(
        parameter_grid_df["email_list_monthly_growth_rate"],
        parameter_grid_df["customer_conversion_rate"]
    )
]

simulation_results_df = pd.concat(summary_list, axis=0) \
    .reset_index() \
    .drop("index", axis=1) \
    .merge(parameter_grid_df, left_index=True, right_index=True)


# Function -------------------------------------------------------------------------------------


def cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0, 0.035],
    customer_conversion_rate=[0.04, 0.05, 0.06],
    **kwargs
):

    # -- Parameter Grid -- #
    data_dict = dict(
        email_list_monthly_growth_rate=email_list_monthly_growth_rate,
        customer_conversion_rate=customer_conversion_rate
    )

    combinations = list(
        product(
            data_dict['email_list_monthly_growth_rate'],
            data_dict['customer_conversion_rate']
        )
    )

    parameter_grid_df = pd.DataFrame(
        combinations,
        columns=['email_list_monthly_growth_rate', 'customer_conversion_rate']
    )

    # -- Temp Function -- #
    def temporary_function(x, y):

        cost_table_df = cost_calc_monthly_cost_table(
            email_list_growth_rate=x,
            customer_conversion_rate=y,
            **kwargs
        )

        summary_df = cost_total_unsub_cost(cost_table_df)

        return summary_df

    # -- List Comprehension -- #
    summary_list = [
        temporary_function(x, y) for x, y in zip(
            parameter_grid_df["email_list_monthly_growth_rate"],
            parameter_grid_df["customer_conversion_rate"]
        )
    ]

    simulation_results_df = pd.concat(summary_list, axis=0) \
        .reset_index() \
        .drop("index", axis=1) \
        .merge(parameter_grid_df, left_index=True, right_index=True)

    # -- Return -- #
    return simulation_results_df


cost_simulate_unsub_cost()


# VISUALIZE COSTS ---------------------------------------------------------------------------

# Plotly
simulation_results_wide_df = cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0.01, 0.02],
    customer_conversion_rate=[0.04, 0.06],
    email_list_size=100000
) \
    .drop("cost_no_growth", axis=1) \
    .pivot(
        index="email_list_monthly_growth_rate",
        columns="customer_conversion_rate",
        values="cost_with_growth"
)

px.imshow(
    simulation_results_wide_df,
    origin="lower",
    aspect="auto",
    title="Lead Cost Simulation",
    labels=dict(x="Customer Conversion Rate",
                y="Monthly Email Growth Rate",
                color="Unsubscrib Cost")
)

# Plotnine
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

df = cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0.01, 0.02],
    customer_conversion_rate=[0.04, 0.06],
    email_list_size=100000
) \
    .drop("cost_no_growth", axis=1) \
    .assign(cost_with_growth_text=lambda x: x['cost_with_growth'].apply(lambda y: '${:.2f}M'.format(y / 1000000))) \
    .assign(growth_rate_text=lambda x: x["email_list_monthly_growth_rate"].apply(lambda y: '{:.1%}'.format(y))) \
    .assign(conv_rate_text=lambda x: x["customer_conversion_rate"].apply(lambda y: '{:.1%}'.format(y))) \
    .assign(
        label_text=lambda x: "Cost: " + x["cost_with_growth_text"].str.cat([
            "Growth Rate: " + x['growth_rate_text'].astype(str).str.cat([
                "Conv Rate: " + x["conv_rate_text"].astype(str)
            ], sep="\n")
        ], sep="\n")
)

(
    ggplot(df, aes("email_list_monthly_growth_rate",
           "customer_conversion_rate", fill="cost_with_growth"))
    + geom_tile()
    + geom_text(aes(label="label_text"), size=8)
    + theme_bw()
    + labs(
        title="Lead Cost Simulation",
        x="Email List Growth Rate (%)",
        y="Customer Converstion Rate (%)",
        fill="Cost with Growth"
    )
    + theme(
        axis_text=element_text(size=8),
        axis_title=element_text(size=8),
        legend_title=element_text(size=9),
        legend_text=element_text(size=7),
        legend_key_size=10
    )
)

# Function: Plot Simulated Unsubscriber Costs ---------------------------------------


def cost_plot_simulated_unsub_cost(simulation_results):

    simulation_results_wide_df = simulation_results \
        .drop("cost_no_growth", axis=1) \
        .pivot(
            index="email_list_monthly_growth_rate",
            columns="customer_conversion_rate",
            values="cost_with_growth"
        )

    fig = px.imshow(
        simulation_results_wide_df,
        origin="lower",
        aspect="auto",
        text_auto=True,
        title="Lead Cost Simulation",
        labels=dict(
            x="Customer Conversion Rate",
            y="Monthly Email Growth Rate",
            color="Unsubscrib Cost"
        )
    )

    fig.update_layout(
        xaxis=dict(title=dict(font=dict(size=11))),
        yaxis=dict(title=dict(font=dict(size=11)))
    )

    return fig


cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0.01, 0.02, 0.03],
    customer_conversion_rate=[0.04, 0.05, 0.06],
    email_list_size=100000
) \
    .pipe(cost_plot_simulated_unsub_costs)

# Plotnine


def cost_plot_simulated_unsub_cost_plotnine(
    simulation_results,
    title, sub_title, x_lab, y_lab, legend_title=""
):

    simulation_results_wide_df = simulation_results \
        .drop("cost_no_growth", axis=1) \
        .assign(cost_with_growth_text=lambda x: x['cost_with_growth'].apply(lambda y: '${:.2f}M'.format(y / 1000000))) \
        .assign(growth_rate_text=lambda x: x["email_list_monthly_growth_rate"].apply(lambda y: '{:.1%}'.format(y))) \
        .assign(conv_rate_text=lambda x: x["customer_conversion_rate"].apply(lambda y: '{:.1%}'.format(y))) \
        .assign(
            label_text=lambda x: "Cost: " + x["cost_with_growth_text"].str.cat([
                "Growth Rate: " + x['growth_rate_text'].astype(str).str.cat([
                    "Conv Rate: " + x["conv_rate_text"].astype(str)
                ], sep="\n")
            ], sep="\n")
        )

    p = (
        ggplot(
            simulation_results_wide_df,
            aes(
                "email_list_monthly_growth_rate",
                "customer_conversion_rate",
                fill="cost_with_growth"
            )
        )
        + geom_tile()
        + geom_text(aes(label="label_text"), size=8)
        + theme_bw()
        + labs(
            title=title,
            subtitle=sub_title,
            x=x_lab,
            y=y_lab,
            fill=legend_title
        )
        + theme(
            axis_text=element_text(size=8),
            axis_title=element_text(size=8),
            legend_title=element_text(size=9),
            legend_text=element_text(size=7),
            legend_key_size=10,
            legend_position="none"
        )
        + scale_fill_gradient(low="lightblue", high="orange")
    )

    return p


cost_plot_simulated_unsub_costs_plotnine(
    simulation_results=cost_simulate_unsub_cost(
        email_list_monthly_growth_rate=[0.01, 0.02, 0.03],
        customer_conversion_rate=[0.04, 0.05, 0.06],
        email_list_size=100000
    ),
    title="Cost Simulation with Variability in Cost Drivers",
    sub_title="Including variability in cost drivers",
    x_lab="Email List Growth Rate (%)",
    y_lab="Customer Conversion Rate (%)"
)


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

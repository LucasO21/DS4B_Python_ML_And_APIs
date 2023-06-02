

# LIBRARIES ----
import pandas as pd
import numpy as np
import janitor as jn
from itertools import product
import plotly.express as px
from plotnine import *
import pandas_flavor as pf


# Function: Calculate Monthly Unsubscriber Cost Table ----
@pf.register_dataframe_method
def cost_calc_monthly_cost_table(
    email_list_size=100000,
    email_list_growth_rate=0.035,
    sales_emails_per_month=5,
    unsub_rate_per_sales_email=0.005,
    customer_conversion_rate=0.05,
    average_customer_value=2000,
    n_periods=12
):
    """This function generates a cost table.

    Args:
        email_list_size (int, optional): Email list size. Defaults to 100000.
        email_list_growth_rate (float, optional): Monthly email list growth rate. Defaults to 0.035.
        sales_emails_per_month (int, optional): Number of sales emails sent out per month. Defaults to 5.
        unsub_rate_per_sales_email (float, optional): Unsubscription rate per email. Defaults to 0.005.
        customer_conversion_rate (float, optional): Customer conversion rate. Defaults to 0.05.
        average_customer_value (int, optional): Average customer value. Defaults to 2000.
        n_periods (int, optional): Number of months for our cost table. Defaults to 24.

    Returns:
        Dataframe: Returns a cost table. 
    """

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


# Function: Sumarize Cost ----
@pf.register_dataframe_method
def cost_total_unsub_cost(cost_table):
    """Takes the input from [cost_calc_monthly_cost_table()], and produces a summary of the 
    total cost. 

    Args:
        cost_table (Dataframe): Output from [cost_calc_monthly_cost_table()]

    Returns:
        Dataframe: Summarized total costs for email unsubscription.
    """

    summary_df = cost_table[["cost_no_growth", "cost_with_growth"]] \
        .sum() \
        .to_frame() \
        .transpose()

    return summary_df


# Function: Simulate Unsubscriber Cost ----
def cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0, 0.035],
    customer_conversion_rate=[0.04, 0.05, 0.06],
    **kwargs
):
    """Generate a cost analysis siumlation to characterize cost uncertainty.

    Args:
        email_list_monthly_growth_rate (list, optional): List of values for email monthly 
        growth rate to simulate. Defaults to [0, 0.035].
        customer_conversion_rate (list, optional): List of values for customer conversion
        rate to simulate. Defaults to [0.04, 0.05, 0.06].

    Returns:
        Dataframe: Cartesian product of the email list and customer conversion rate is 
        calculated and total unsubscriber costs are calculated.
    """

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


# Function: Plot Simulated Costs ----
@pf.register_dataframe_method
def cost_plot_simulated_unsub_cost(simulation_results):
    """This is a plotting function to plot the results from [cost_simulated_unsub_cost()].

    Args:
        simulation_results (Dataframe): The output from [cost_simulate_unsub_cost()]

    Returns:
        Plotly Plot: Heatmap that visualizes the cost simulation.
    """

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
        # text_auto=True,
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


# Function: Plot Simulated Costs (Plotnine) ----
@pf.register_dataframe_method
def cost_plot_simulated_unsub_cost_plotnine(
    simulation_results,
    title=None, sub_title=None, x_lab=None, y_lab=None, legend_title=None
):
    """This is a plotnine plotting function to plot the results 
    from [cost_simulated_unsub_cost()]

    Args:
        simulation_results (Dataframe): The output from [cost_simulate_unsub_cost()].
        title (str): Title of plot.
        sub_title (str): Subtitle of plot.
        x_lab (str): X - Axis label.
        y_lab (str): Y - Axis label.
        legend_title (str, optional): Legend title. Defaults to "".

    Returns:
        Plotnine Plot: Heatmap that visualizes the cost simulation.
    """

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

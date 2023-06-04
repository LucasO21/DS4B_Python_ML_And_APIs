# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING
# ----


# TEST CALCULATIONS ----

import email_lead_scoring as els

# import email_lead_scoring_modules.cost_calculations as cost

# cost.cost_calc_monthly_cost_table()

# ?cost.cost_calc_monthly_cost_table

# Once __init__.py file imports the submodule, function
# ?els.cost_calc_monthly_cost_table

# ?els.cost_total_unsub_cost

# Pandas Flavor - Chaining Dataframes
# els.cost_calc_monthly_costs_table() \
#     .cost_total_unsub_costs()


# ?els.cost_simulate_unsub_cost
# els.cost_simulate_unsub_cost()


# ?els.cost_plot_simulated_unsub_costs

# ?els.cost_plot_simulated_unsub_costs_plotnine


df = els.cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0.015, 0.025, 0.035]
)

els.cost_plot_simulated_unsub_cost(df)

els.cost_plot_simulated_unsub_cost_plotnine(
    simulation_results=df,
    title="Cost Simulation"
)


# Database
els.db_read_els_data().head()

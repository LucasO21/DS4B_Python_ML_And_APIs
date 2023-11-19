# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 7: ROI
# PART 3: LEAD STRATEGY FUNCTIONAL WORKFLOW
# ----

# -------------------------------------------------------------------------------------- #
#                                         IMPORTS                                        #
# -------------------------------------------------------------------------------------- #
import email_lead_scoring as els

# -------------------------------------------------------------------------------------- #
#                                        WORKFLOW                                        #
# -------------------------------------------------------------------------------------- #

leads_df = els.db_read_and_process_els_data()

leads_scored_df = els.model_score_leads(
    data = leads_df,
    model_path = "models/pycaret/xgb_model_single_tuned_finalized"
)


# -------------------------------------------------------------------------------------- #
#                                     BUILD FUNCTIONS                                    #
# -------------------------------------------------------------------------------------- #
#  els > lead_strategy.py

?els.lead_score_strategy_optimization

optimization_results_list = els.lead_score_strategy_optimization(
    data = leads_scored_df
)

optimization_results_list.keys()

keys_list = list(optimization_results_list.keys())

optimization_results_list[keys_list[0]]
optimization_results_list[keys_list[1]]
optimization_results_list[keys_list[2]]
optimization_results_list[keys_list[3]]




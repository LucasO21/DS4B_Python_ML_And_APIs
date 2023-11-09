# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 7: ROI 
# PART 3: LEAD STRATEGY FUNCTIONAL WORKFLOW
# ----

# IMPORTS
import email_lead_scoring as els

# WORKFLOW ----

leads_df = els.db_read_and_process_els_data()
leads_df

leads_scored_df = els.model_score_leads(leads_df)
leads_scored_df

# CREATE FUNCTIONS ----
#  els > lead_strategy.py




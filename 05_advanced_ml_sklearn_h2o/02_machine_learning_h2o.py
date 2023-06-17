# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 5: ADVANCED MACHINE LEARNING 
# PART 2: H2O AUTOML
# ----
# ========================================================================
# LIBRARIES ----
# ========================================================================
import pandas as pd
import numpy as np

import h2o
from h2o.automl import H2OAutoML

import email_lead_scoring as els


# ========================================================================
# DATA IMPORT ----
# ========================================================================

leads_df = els.db_read_and_process_els_data()


# ========================================================================
# 1.0 H2O PREPARATION
# ========================================================================

# - Initialize H2O
h2o.init( max_mem_size = 4 )


# - Convert to H2O Frame
leads_h2o = h2o.H2OFrame(leads_df)

leads_h2o["made_purchase"] = leads_h2o["made_purchase"].asfactor()

leads_h2o.describe()


# - Prep for AutoML
leads_h2o_cols = leads_h2o.columns
drop_cols = [
    "mailchimp_id", "user_full_name", "user_email", 
    "optin_time", "email_provider", "made_purchase"
    ]

x_cols = [col for col in leads_h2o_cols if col not in drop_cols]

y_col = "made_purchase"



# ========================================================================
# 2.0 RUN H2O AUTOML ----
# ========================================================================

# - H2OAutoML
aml = H2OAutoML(
    project_name = "lead_scoring_prototype",
    nfolds = 5,
    exclude_algos = ["DeepLearning"],
    max_runtime_secs = 3 * 60, 
    seed = 123    
)

aml.train(x = x_cols, y = y_col, training_frame = leads_h2o)

aml.leaderboard


# ========================================================================
# MAKING PREDICTIONS
# ========================================================================
model_h2o_se = h2o.get_model(model_id = "StackedEnsemble_AllModels_3_AutoML_1_20230617_82716")

predictions_h2o = model_h2o_se.predict(leads_h2o)

h2o_predictino_df = predictions_h2o.as_data_frame()

pd.concat([h2o_predictino_df, leads_df], axis = 1)\
    .sort_values("p1", ascending = False)
    
# ========================================================================
# PERFORMANCE
# ========================================================================
best_model_performace = model_h2o_se.model_performance()

best_model_performace.plot(type = "roc")

model_h2o_se.explain_row(leads_h2o, row_index = 1, figsize = (8, 5))



# ========================================================================
# SAVE / LOAD
# ========================================================================

# - Save 
h2o.save_model(
    model = model_h2o_se,
    path = "models/h2o",
    filename = "h2o_stacked_ensemble",
    force = True
)

# - Load
# - Must always use the same h2o training version for building and using h2o models
h2o.load_model("models/h2o/h2o_stacked_ensemble")



# CONCLUSIONS ----
# 1. H2O AutoML handles A LOT of stuff for you (preprocessing)
# 2. H2O is highly scalable
# 3. (CON) H2O depends on Java, which adds another complexity when you take your model into production

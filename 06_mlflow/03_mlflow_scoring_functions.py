# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 6: MLFLOW 
# PART 3: PREDICTION FUNCTION 
# ----

import pandas as pd
import mlflow
import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()

# ========================================================================
# 1.0 GETTING THE BEST RUN FOR PRODUCTION ----
# ========================================================================
EXPERIMENT_NAME = "email_lead_scoring_0"
EXPERIMENT_NAME = "automl_lead_scoring_1"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

experiment_id = experiment.experiment_id

logs_df = mlflow.search_runs(experiment_id)\
    .rename(columns = lambda x: x.lower())

logs_df\
    .query("`tags.source` in ['finalize_model', 'h2o_automl_model']")\
    .sort_values("metrics.AUC", ascending = False)\
    ["run_id"]\
    .values[0]
    



# Function
def mlflow_get_best_run(
    experiment_name, 
    n          = 1, 
    metric     = "metrics.auc", 
    tag_source = ["finalize_model", "h2o_automl_model"],
    ascending  = False
):
    
    experiment = mlflow.get_experiment_by_name(experiment_name)    
    experiment_id = experiment.experiment_id
    
    logs_df = mlflow.search_runs(experiment_id)\
    .rename(columns = lambda x: x.lower())
    
    best_run_id = logs_df\
        .query(f'`tags.source` in {tag_source}')\
        .sort_values(metric, ascending = ascending)\
        ["run_id"]\
        .values[n - 1]
    
    return(best_run_id)


mlflow_get_best_run(experiment_name = "automl_lead_scoring_1")

mlflow_get_best_run(experiment_name = "email_lead_scoring_0")


# ========================================================================
# 2.0 PREDICT WITH THE MODEL (LEAD SCORING FUNCTION)
# ========================================================================

# Load model as a PyFuncModel

# H2O
run_id = mlflow_get_best_run(experiment_name = "automl_lead_scoring_1")

logged_model = f"runs:/{run_id}/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
loaded_model.predict(leads_df)["p1"]

# Sklearn / Pycaret (Extract)
run_id = mlflow_get_best_run(experiment_name = "email_lead_scoring_0")

logged_model = f"runs:/{run_id}/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
loaded_model._model_impl.predict_proba(leads_df)[:, 1]

# Function
def mlflow_score_leads(data, run_id):
    
    logged_model = f"runs:/{run_id}/model"
    print(logged_model)

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    # Predict
    try:
        predictions_array = loaded_model.predict(pd.DataFrame(data))["p1"]
    except:
        predictions_array = loaded_model._model_impl.predict_proba(data)[:, 1]
        
    predictions_series = pd.Series(predictions_array, name = "Score")
    
    ret = pd.concat([predictions_series, data], axis = 1)
    
    
    # Return
    return ret

# End Function ======================
mlflow_score_leads(data = leads_df, run_id = mlflow_get_best_run("automl_lead_scoring_1"))

mlflow_score_leads(data = leads_df, run_id = mlflow_get_best_run("email_lead_scoring_0"))
    


# ========================================================================
# 3.0 TEST WORKFLOW ----
# ========================================================================
import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()

best_run_id = els.mlflow_get_best_run("automl_lead_scoring_1")

els.mlflow_score_leads(leads_df, best_run_id)


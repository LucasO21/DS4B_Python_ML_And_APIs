# =========================================================================
# LEAD SCORING MODULE 
# =========================================================================


# =========================================================================
# LIBRARIES 
# =========================================================================
import pandas as pd
import numpy as np
import pycaret.classification as clf
import mlflow

# =========================================================================
# MODEL LOAD FUNCTION
# =========================================================================
def model_score_leads(data, model_path = "models/blended_model_final"):
    """_summary_

    Args:
        data (DataFrame): DataFrame of leads to predict/score.
        model_path (str, optional): Path to leads scoring model. Defaults to "models/blended_model_final".

    Returns:
        DataFrame: DataFrame of leads along with prediction and score.
    """
    
    # - Load model
    model = clf.load_model(model_path)
    
    # - Get predictions
    predictions_df = clf.predict_model(estimator = model, data = data)

    df = predictions_df
    
    predictions_df["Score"] = np.where(df["Label"] == 0, 1 - df["Score"], df["Score"])
        
    # - Score leads
    leads_scored_df = pd.concat([predictions_df["Score"], data], axis = 1)    
    
    # - Return
    return leads_scored_df


# =========================================================================
# MLFLOW GET BEST RUN
# =========================================================================
def mlflow_get_best_run(
    experiment_name, 
    n          = 1, 
    metric     = "metrics.auc", 
    tag_source = ["finalize_model", "h2o_automl_model"],
    ascending  = False
):
    """Returns the best run from an MLflow experiment name.

    Args:
        experiment_name (str): MLflow experiment name.
        n (int, optional): _description_. Defaults to 1.
        metric (str, optional): Metric to use. Defaults to "metrics.auc".
        tag_source (list, optional): Tag.Source in MLflow to use in production. Defaults to ["finalize_model", "h2o_automl_model"].
        ascending (bool, optional): Whether or not to sor the metric ascending or descending. AUC should be descending. Metrics like log loss should be ascending.  
    
    Returns:
        string: The best run id found. 
    
    """
    
    
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


# =========================================================================
# MLFLOW SCORE LEADS
# =========================================================================
def mlflow_score_leads(data, run_id):
    
    """This function scores the leads using an MLflow Run Id to select a model. 
    
    Args:
        data (DataFrame): Leads data from els.db_read_and_process_data()
        run_id (string): An MLflow Run Id. Recommend to use mlflow_get_best_run().

    Returns:
        DataFrame: A dataframe with the lead scores column, concatenated to the leads_df.
    """
    
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
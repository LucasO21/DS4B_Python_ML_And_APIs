# =========================================================================
# LEAD SCORING MODULE 
# =========================================================================


# =========================================================================
# LIBRARIES 
# =========================================================================
import pandas as pd
import numpy as np
import pycaret.classification as clf

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



# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING & APIS
# MODULE 4: MACHINE LEARNING | MODEL LEAD SCORING FUNCTION
# ====

# =========================================================================
# LIBRARIES 
# =========================================================================
import pandas as pd
import numpy as np
import pycaret.classification as clf
import email_lead_scoring as els


# =========================================================================
# DATA IMPORT
# =========================================================================
leads_df = els.db_read_and_process_els_data()

# =========================================================================
# MODEL LOAD FUNCTION
# =========================================================================
def model_score_leads(data, model_path = "models/blended_model_final"):
    
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

model_score_leads(data = leads_df)


# data = leads_df
# df = predictions_df 
# predictions_df[["user_full_name", "Score", "Label"]]
# leads_scored_df[["user_full_name", "Score"]]



# =========================================================================
# TEST OUT
# =========================================================================

leads_df = els.db_read_and_process_els_data()

test_df = els.model_score_leads(
    data = leads_df,
    model_path = "models/xgb_model_tuned_finalized"
) \
    .sort_values("Score", ascending = False)
    
test_df.query("user_full_name == 'Mikeal Bernhard'")
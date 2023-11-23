# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 9: STREAMLIT
# FRONTEND USER STREAMLIT APP FOR API - NO SECURITY
# ----

# To run app (put this in Terminal):
#   streamlit run 09_streamlit/01_app_streamlit.py


# ---------------------------------------------------------------------------- #
#                                   PACKAGES                                   #
# ---------------------------------------------------------------------------- #
import sys
import pathlib
import streamlit as st
import requests
import pandas as pd

# import altair
# from altair.vegalite.v4.api import Chart

# ---------------------------------------------------------------------------- #
#                                     PATHS                                    #
# ---------------------------------------------------------------------------- #
# NEEDED FOR EMAIL LEAD SCORING TO BE DETECTED
# APPEND PROJECT DIRECTORY TO PYTHONPATH

working_dir = str(pathlib.Path().absolute())
print(working_dir)
sys.path.append(working_dir)

import email_lead_scoring as els

ENDPOINT = 'http://localhost:8000'

# ---------------------------------------------------------------------------- #
#                                   1.0 TITLE                                  #
# ---------------------------------------------------------------------------- #
st.title("Email Lead Scoring Streamlit Frontend App")

st.write("---")


# -------------------------------------------------------------------------------------- #
#                                     2.0 DATA INPUT                                     #
# -------------------------------------------------------------------------------------- #
# CACHING DATA - NEEDED TO PREVENT REQUIRING DATA TO BE RE-INPUT
uploaded_file = st.file_uploader(
    "Choose a CSV File",
    type = "csv",
    accept_multiple_files = False
)

@st.cache_data()
def load_data(filename):
    leads_df = pd.read_csv(uploaded_file)
    return leads_df


 # -------------------------------------------------------------------------------------- #
 #                                      3.0 APP BODY                                      #
 # -------------------------------------------------------------------------------------- #
#  What Happens Once Data Is Loaded?
if uploaded_file:

    leads_df = load_data(uploaded_file)
    full_data_json = leads_df.to_json()

    # Checkbox - Show Table
    if st.checkbox("Show Raw Data"):
        st.subheader("Sample of Raw Data (First 10 Rows)")
        st.write(leads_df.head(10))

    st.write("---")
    st.markdown("## Lead Scoring Analysis")

    # User Inputs - Add Sliders / Buttons
    estimated_monthly_sales = st.number_input(
        "How much on average in email sales per month ($)",
        min_value = 0,
        value     = 250000,
        step      = 1000
    )
    print(estimated_monthly_sales)

    monthly_sales_reduction_safe_guard = st.slider(
        "How much of the monthly sales should be maintained (%)",
        min_value = 0.,
        max_value = 1.,
        value     = 0.9,
        step      = 0.01
    )
    print(monthly_sales_reduction_safe_guard)

    sales_limit = "${:,.0f}".format(monthly_sales_reduction_safe_guard * estimated_monthly_sales)
    st.subheader(f"monthly sales will not go below: {sales_limit}")

    # Run Analysis
    if st.button("Run Analysis"):

        # Spinner
        with st.spinner("Lead scoring in progress..."):

            # Make Request
            res = requests.post(
                    url = f"{ENDPOINT}/calculate_lead_strategy",
                    json = full_data_json,
                    params = dict(
                    monthly_sales_reduction_safe_guard = float(monthly_sales_reduction_safe_guard),
                    email_list_size = 100000,
                    unsub_rate_per_sales_email = 0.005,
                    sales_emails_per_month = 5,
                    avg_sales_per_month = float(estimated_monthly_sales),
                    avg_sales_emails_per_month = 5,
                    customer_conversion_rate = 0.05,
                    avg_customer_value = 2000.0
                )
            )
            print(pd.read_json(res.json()["expected_value"]))


            # Collect JSON / Convert Data



            # Display Results


            # Display Strategy Summary


            # Display Expected Value Plot



            # Display Sample Lead Strategy


            # Download button - Get lead scoring results




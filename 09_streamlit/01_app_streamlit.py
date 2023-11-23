# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 9: STREAMLIT 
# FRONTEND USER STREAMLIT APP FOR API - NO SECURITY
# ----

# To run app (put this in Terminal):
#   streamlit run 09_streamlit/01_app_streamlit.py

import streamlit as st
import requests
import pandas as pd

import sys
import pathlib

# NEEDED FOR EMAIL LEAD SCORING TO BE DETECTED
# APPEND PROJECT DIRECTORY TO PYTHONPATH
working_dir = str(pathlib.Path().absolute())
print(working_dir)
sys.path.append(working_dir)

import email_lead_scoring as els

ENDPOINT = 'http://localhost:8000'

# 1.0 TITLE


# 2.0 DATA INPUT
#   CACHING DATA - NEEDED TO PREVENT REQUIRING DATA TO BE RE-INPUT

# 3.0 APP BODY


    
    # Checkbox - Show Table 
    
    
    
    # User Inputs - Add Sliders / Buttons
    
    
    
    
    # Run Analysis 
    
        # Spinner
        
            
            # Make Request
            
            
            
            # Collect JSON / Convert Data
            
            
            
            # Display Results
            
            
            # Display Strategy Summary
            
            
            # Display Expected Value Plot
            
            
            
            # Display Sample Lead Strategy
            
            
            # Download button - Get lead scoring results
            
        
    

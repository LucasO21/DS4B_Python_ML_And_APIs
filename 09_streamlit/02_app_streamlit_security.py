# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 9: STREAMLIT 
# FRONTEND USER STREAMLIT APP FOR API - SECURITY ENABLED
# ----

# To run app (put this in Terminal):
#   streamlit run 09_streamlit/02_app_streamlit_security.py

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

AUTHORIZED_API_KEYS = {"X-API-KEY": "my-secret-api-key"}



# [INSERT APP CODE HERE]


        
    

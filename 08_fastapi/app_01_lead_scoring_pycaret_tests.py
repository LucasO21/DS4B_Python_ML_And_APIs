# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 8: FASTAPI
# PART 1: TESTING THE API
# ----

# -------------------------------------------------------------------------------------- #
#                                        LIBRARIES                                       #
# -------------------------------------------------------------------------------------- #
import pandas as pd
import requests

import email_lead_scoring as els


# Converting to JSON
full_data = els.db_read_and_process_els_data()

full_data_json = full_data.to_json()

sample_data = els.db_read_and_process_els_data().head()

sample_data_json = sample_data.to_json()


# -------------------------------------------------------------------------------------- #
#                          1.0 GET: EXPOSE DATA FROM AN ENDPOINT                         #
# -------------------------------------------------------------------------------------- #
res = requests.get(
    "http://127.0.0.1:8000/get_email_subscribers"
)

res.json()

pd.read_json(res.json())
# -------------------------------------------------------------------------------------- #
#                              2.0 POST: PASS DATA TO AN API                             #
# -------------------------------------------------------------------------------------- #




# -------------------------------------------------------------------------------------- #
#                        3.0 POST: PASS DATA AND MAKE PREDICTIONS                        #
# -------------------------------------------------------------------------------------- #




# -------------------------------------------------------------------------------------- #
#                           4.0 POST: PASS DATA AND PARAMETERS                           #
# -------------------------------------------------------------------------------------- #
#   TO CALCULATE LEAD STRATEGY FOR MARKETING





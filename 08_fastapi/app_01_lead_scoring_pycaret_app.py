# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 8: FASTAPI
# PART 1: BUILDING THE API
# ----

# To Run this App:
# - Open Terminal
# - uvicorn 08_fastapi.app_01_lead_scoring_pycaret_app:app --reload --port 8000
# - Navigate to localhost:8000
# - Navigate to localhost:8000/docs
# - Shutdown App: Ctrl/Cmd + C

# -------------------------------------------------------------------------------------- #
#                                        PACKAGES                                        #
# -------------------------------------------------------------------------------------- #

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import json
import pandas as pd
import email_lead_scoring as els


app = FastAPI()

# -------------------------------------------------------------------------------------- #
#                                          DATA                                          #
# -------------------------------------------------------------------------------------- #
leads_df = els.db_read_and_process_els_data()


# -------------------------------------------------------------------------------------- #
#                                 0.0 INTRODUCE YOUR API                                 #
# -------------------------------------------------------------------------------------- #
@app.get("/")
async def main():

    content = """
                <head>
                <style>
                    body {
                        font-family: 'Arial', sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #2c3e50;
                        color: #ffffff;
                        text-align: center;
                    }

                    h1 {
                        color: #18bc9c;
                    }

                    p {
                        color: #ffffff;
                    }

                    code {
                        background-color: #18bc9c;
                        color: #ffffff;
                        padding: 2px 6px;
                        border-radius: 4px;
                        font-family: 'Courier New', monospace;
                    }

                    .container {
                        padding: 50px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Welcome to the Email Lead Scoring Project</h1>
                    <p>This API helps users score leads using our proprietary lead scoring models.</p>
                    <p>Navigate to the <code>/docs</code> to see the API documentation.</p>
                </div>
            </body>
    """

    return HTMLResponse(content = content)


#18bc9c
# -------------------------------------------------------------------------------------- #
#                1.0 GET: EXPOSE THE EMAIL SUBSCRIBER DATA AS AN ENDPOINT                #
# -------------------------------------------------------------------------------------- #
@app.get("/get_email_subscribers")
async def get_email_subscribers():

    json = leads_df.to_json()

    return JSONResponse(json)





# -------------------------------------------------------------------------------------- #
#                            2.0 POST: PASSING DATA TO AN API                            #
# -------------------------------------------------------------------------------------- #




# -------------------------------------------------------------------------------------- #
#                           3.0 MAKING PREDICTIONS FROM AN API                           #
# -------------------------------------------------------------------------------------- #




# -------------------------------------------------------------------------------------- #
#                          4.0 POST: MAKE LEAD SCORING STRATEGY                          #
# -------------------------------------------------------------------------------------- #




# -------------------------------------------------------------------------------------- #
#                                   DEFINE THE API PORT                                  #
# -------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


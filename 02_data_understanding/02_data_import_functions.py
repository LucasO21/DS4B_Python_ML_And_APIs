# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING
# PART 2: DATA IMPORT FUNCTIONS
# ----

# LIBRARIES ----

import pandas as pd
import numpy as np
import sqlalchemy as sql


# IMPORT RAW DATA ----

# Read & Combine Raw Data
def db_read_els_data(conn_string="sqlite://" + "/00_database/crm_database.sqlite"):

    # Connect To Engine
    engine = sql.create_engine(conn_string)

    # Raw Data Collect
    with engine.connect() as conn:

        # Subscribers
        subscribers_df = pd.read_sql(sql="select * from Subscribers", con=conn)

        subscribers_df = subscribers_df \
            .assign(mailchimp_id=lambda x: x["mailchimp_id"].astype("int32")) \
            .assign(member_rating=lambda x: x["member_rating"].astype("int32")) \
            .assign(optin_time=lambda x: x["optin_time"].astype("datetime64"))

        # Tags
        tags_df = pd.read_sql(sql="select * from Tags", con=conn)

        tags_df = tags_df \
            .assign(mailchimp_id=lambda x: x["mailchimp_id"].astype("int32"))

        # Transactions
        transactions_df = pd.read_sql(
            sql="select * from Transactions", con=conn)

        transactions_df = transactions_df \
            .assign(purchased_at=lambda x: x["purchased_at"].astype("datetime64")) \
            .assign(product_id=lambda x: x["product_id"].astype("int32"))

        # Merge Tag Counts
        user_events_df = tags_df \
            .groupby("mailchimp_id") \
            .agg(dict(tag="count")) \
            .set_axis(["tag_count"], axis=1) \
            .reset_index()

        subscribers_joined_df = subscribers_df \
            .merge(user_events_df, how="left") \
            .fillna(dict(tag_count=0)) \
            .assign(tag_count=lambda x: x["tag_count"].astype("int32"))

        # Merge Target Variable
        emails_made_purchase = transactions_df["user_email"].unique()

        subscribers_joined_df = subscribers_joined_df \
            .assign(country_code=lambda x: x["country_code"].str.upper()) \
            .assign(made_purchase=lambda x: x["user_email"].isin(emails_made_purchase).astype("int32"))

    return subscribers_joined_df


db_read_els_data().head()

db_read_els_data().info()


# Read Table Names
def db_read_els_table_names(conn_string="sqlite://" + "/00_database/crm_database.sqlite"):

    engine = sql.create_engine(conn_string)

    table_names = sql.inspect(engine).get_table_names()

    return table_names


db_read_els_table_names()


# Get Raw Table
def db_read_els_raw_table(conn_string="sqlite://" + "/00_database/crm_database.sqlite",
                          table_name="Products"):

    engine = sql.create_engine(conn_string)

    with engine.connect() as conn:

        df = pd.read_sql(sql=f"select * from {table_name}", con=conn)

    return df


db_read_els_raw_table(table_name="Website")


# TEST IT OUT -----

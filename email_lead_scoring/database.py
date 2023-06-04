# LIBRARIES ----

import pandas as pd
import numpy as np
import sqlalchemy as sql


# IMPORT RAW DATA ----

# Read & Combine Raw Data
def db_read_els_data(conn_string="sqlite://" + "/00_database/crm_database.sqlite"):
    """Function to read in the Subscribers, Tags, and Transactions tables and combine
    them into a DataFrame with `tag_count` and `made_purchase` columns.

    Args:
        conn_string (str, optional): Database connection string. Defaults to "sqlite://"+"/00_database/crm_database.sqlite".

    Returns:
        _type_: Pandas DataFrame
    """

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


# Read Table Names
def db_read_els_table_names(conn_string="sqlite://" + "/00_database/crm_database.sqlite"):
    """Get table tames for each table in the crm database. 

    Args:
        conn_string (str, optional): _description_. Defaults to "sqlite://"+"/00_database/crm_database.sqlite".

    Returns:
        _type_: List
    """

    engine = sql.create_engine(conn_string)

    table_names = sql.inspect(engine).get_table_names()

    return table_names


# Get Raw Table
def db_read_els_raw_table(
    conn_string="sqlite://" + "/00_database/crm_database.sqlite",
    table_name="Products"
):
    """Reads a single raw table for each table in the crm database

    Args:
        conn_string (str, optional): _description_. Defaults to "sqlite://"+"/00_database/crm_database.sqlite".
        table_name (str, optional): Table name. Defaults to "Products". See [db_read_els_table_names()]
        to get the full list of table names.

    Returns:
        _type_: Pandas DataFrame
    """

    engine = sql.create_engine(conn_string)

    with engine.connect() as conn:

        df = pd.read_sql(sql=f"select * from {table_name}", con=conn)

    return df

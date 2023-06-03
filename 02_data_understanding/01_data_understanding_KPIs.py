# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING
# PART 1: DATA UNDERSTANDING & KPIS
# ----

# GOAL: ----
# - Saw high costs, feedback showed problems
# - Now need to work with departments to collect data and develop project KPIs

# LIBRARIES ----

# Data Analysis:
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_bar, geom_line, scale_y_continuous, theme_bw
from plotnine import labs, theme, element_text
import seaborn as sns

# New Libraries:
import sweetviz as sv
import sqlalchemy as sql

# Email Lead Scoring:
import email_lead_scoring as els


# ?els.cost_calc_monthly_unsub_cost_table

els.cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=np.linspace(0, 0.03, 5),
    customer_conversion_rate=np.linspace(0.4, 0.6, 3),
    sales_emails_per_month=5,
    unsub_rate_per_sales_email=0.001,
    email_list_size=1e5
) \
    .pipe(func=els.cost_plot_simulated_unsub_cost)


# 1.0 CONNECTING TO SQLITE DATABASE ----
engine = sql.create_engine("sqlite://" + "/00_database/crm_database.sqlite")

conn = engine.connect()

sql.inspect(engine).get_table_names()


# 2.0 COLLECT DATA ----

# Products ----
products_df = pd.read_sql(sql = "select * from Products", con=conn)

products_df = products_df \
    .assign(product_id = lambda x: x["product_id"].astype("int"))

products_df.head()

products_df.shape

products_df.info()


# Subscribers ----
subscribers_df = pd.read_sql(sql = "select * from Subscribers", con=conn)

subscribers_df = subscribers_df \
    .assign(mailchimp_id = lambda x: x["mailchimp_id"].astype("int32")) \
    .assign(member_rating = lambda x: x["member_rating"].astype("int32")) \
    .assign(optin_time = lambda x: x["optin_time"].astype("datetime64")) 

subscribers_df.head()

subscribers_df.shape

subscribers_df.info()


# Tags ----
tags_df = pd.read_sql(sql = "select * from Tags", con=conn)

tags_df = tags_df \
    .assign(mailchimp_id = lambda x: x["mailchimp_id"].astype("int32"))

tags_df.head()

tags_df.shape

tags_df.info()


# Transactions ----
transactions_df = pd.read_sql(sql = "select * from Transactions", con=conn)

transactions_df = transactions_df \
    .assign(purchased_at = lambda x: x["purchased_at"].astype("datetime64")) \
    .assign(product_id = lambda x: x["product_id"].astype("int32"))

transactions_df.head()

transactions_df.shape

transactions_df.info()


# Website ----
website_df = pd.read_sql(sql = "select * from Website", con=conn)

website_df = website_df \
    .assign(date = lambda x: x["date"].astype("datetime64")) \
    .assign(pageviews = lambda x: x["pageviews"].astype("int32")) \
    .assign(organicsearches = lambda x: x["organicsearches"].astype("int32")) \
    .assign(sessions = lambda x: x["sessions"].astype("int32"))

website_df.info()


# Close Connection ----
# - Note: a better practice is to use `with`
conn.close()


with engine.connect() as conn:
    


# 3.0 COMBINE & ORGANIZE DATA ----
# - Problem is related to probability of purchase from email list
# - Need to understand what increases probability of purchase
# - Learning Labs could be a key event
# - Website data would be interesting but can't link it to email
# - Products really aren't important to our initial question - just want to know if they made a purchase or not and identify which are likely

# Make Target Feature
emails_made_purchase = transactions_df["user_email"].unique()

subscribers_df["user_email"] \
    .isin(emails_made_purchase) 
    
subscribers_df = subscribers_df \
    .assign(country_code = lambda x: x["country_code"].str.upper()) \
    .assign(made_purchase = lambda x: x["user_email"].isin(emails_made_purchase).astype("int32"))

# Who is purchasing?
subscribers_df["made_purchase"].sum()/len(subscribers_df)


# By Geographic Regions (Countries)
by_geography_df = subscribers_df \
    .groupby("country_code") \
    .agg(
        dict(made_purchase = ["sum", lambda x: sum(x) / len(x)])
    ) \
    .set_axis(["sales", "prop_in_group"], axis=1) \
    .assign(prop_overall = lambda x: x["sales"] / sum(x["sales"])) \
    .sort_values(by = "sales", ascending=False) \
    .assign(prop_cumsum = lambda x: x["prop_overall"].cumsum())
    
subscribers_df

# - Top 80% countries
by_geography_df.query("prop_cumsum <= 0.80")

# - High Conversion Countries (>8% conversion)
by_geography_df.query("prop_in_group <= 0.80")

# - Prop In Group Quantile 
by_geography_df.quantile(q=[0.10, 0.50, 0.90])

# - Prop In Group Mean 
by_geography_df.mean()


# By Tags (Events)
# - Probablity of making a purchase may be impacted by the amount of tags (events) a customer has

# - Count of Tags
tags_df \
    .groupby("tag") \
    .agg(dict(tag = "count"))
    
# - Count of Tags by User
user_events_df = tags_df \
    .groupby("mailchimp_id") \
    .agg(dict(tag = "count")) \
    .set_axis(["tag_count"], axis = 1) \
    .reset_index()

# - Merge `user_events_df` and `subscribers_df`
subscribers_joined_df = subscribers_df \
    .merge(user_events_df, how = "left") \
    .fillna(dict(tag_count = 0)) \
    .assign(tag_count = lambda x: x["tag_count"].astype("int32"))
    
subscribers_joined_df.info()

# - Analyzing Tag Count Proportions
subscribers_joined_df \
    .groupby("made_purchase") \
    .quantile(q = [0.10, 0.50, 0.90])


# 4.0 SWEETVIZ EDA REPORT ----
report = sv.analyze(subscribers_joined_df, target_feat="made_purchase")

report.show_html(filepath="02_data_understanding/subscriber_eda_report.html")


# 5.0 DEVELOP KPI'S ----
# - Reduce unnecessary sales emails by 30% while maintaing 99% of sales
# - Segment list - apply sales (hot leads), nuture (cold leads)

# EVALUATE PERFORMANCE -----
subscribers_joined_df[["made_purchase", "tag_count"]] \
    .groupby("made_purchase") \
    .agg(
        mean_tag_count = ("tag_count", "mean"),
        median_tag_count = ("tag_count", "median"),
        count_subscriber = ("tag_count", "count")
    )


# WHAT COULD BE MISSED?
# - More information on which tags are most important


######################################################################################################

# EDA Section For Article

print(sns.color_palette("pastel").as_hex())

# - Member Rating: Count & Prop of `made_purchase`    
df = subscribers_joined_df \
    .groupby("member_rating") \
    .agg(
        count = ("member_rating", "size"),
        made_purchase_prop = ("made_purchase", "mean") \
    ) \
    .reset_index() \
    .assign(member_rating = lambda x: x["member_rating"].astype("str"))
    
# Set up the plot
sns.set_theme(style="white", palette=None)
fig, ax1 = plt.subplots()

# Plot count as bars
sns.barplot(x='member_rating', y='count', data=df, color='#a1c9f4', alpha=0.5, ax=ax1,
            saturation=1)

# Create a second y-axis for made_purchase_proportion
ax2 = ax1.twinx()

# Plot made_purchase_proportion as a line chart
sns.lineplot(x='member_rating', y='made_purchase_prop', data=df, color='#ffb482', linewidth=2, ax=ax2,
             marker='o')

# Remove gridlines
ax1.grid(False)
ax2.grid(False)

# Set labels and title
ax1.set_ylabel('Frequency', fontsize=9)
ax2.set_ylabel('\n Made Purchase (%)', fontsize=9)
ax1.set_xlabel('Member Rating', fontsize=9)
ax1.set_title('Count and Made Purchase Proportion', fontsize=9)

# Reduce the fontsize of the axis values
ax1.tick_params(axis='both', labelsize=8)
ax2.tick_params(axis='both', labelsize=8)

# Add data labels to the line plot
for x, y in zip(df['member_rating'], df['made_purchase_prop']):
    ax2.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize = 8, fontweight="bold")

# Show the plot
plt.show()


df = subscribers_joined_df \
    .groupby("made_purchase") \
    .agg(count = ("made_purchase", "size")) \
    .reset_index() \
    .assign(made_purchase_prop = lambda x: x["count"] / x["count"].sum())



# Function: Plot Bar
def plot_bar(data, x, y, fill_color="#a1c9f4", alpha=0.8,
             xlab=None, ylab=None, title=None, data_label=None,
             bar_width=0.7, **kwargs):
    
    sns.set_theme(style="white", palette=None)
    fig, ax1 = plt.subplots()
    
    # Plot count as bars
    ax1 = sns.barplot(x=x, y=y, data=data, color=fill_color, alpha=alpha, ax=ax1, saturation=1,
                      width=bar_width)
    
    # Set labels and title
    ax1.set_ylabel(ylab, fontsize=9)
    ax1.set_xlabel(xlab, fontsize=9)
    ax1.set_title(title, fontsize=10)
    
    # Reduce the fontsize of the axis values
    ax1.tick_params(axis='both', labelsize=8)
    
    # Add data labels to the bars
    if data_label is not None:
        for i, row in data.iterrows():
            label = f'{row[data_label]:.2%}' if isinstance(row[data_label], float) else str(row[data_label])
            ax1.text(row.name, row[y], label, ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig

    
    
    
plot_bar(
    data=df, 
    x="made_purchase", 
    y="count", 
    data_label="made_purchase_prop",
    title="Only 5% of Users Have Made a Previous Purchase",
    bar_width=0.5
)
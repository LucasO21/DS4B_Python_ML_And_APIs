# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# EDA FOR ANALYSIS WRITE UP
# ----


# LIBRARIES ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mizani.formatters import *
import email_lead_scoring as els


# LOAD DATA ----
data = els.db_read_els_data()

# EDA ----

# Made Purchase

# - Data
df_made_purchase_prop = data \
    .groupby("made_purchase") \
    .agg(count=("made_purchase", "size")) \
    .reset_index() \
    .assign(made_purchase_prop=lambda x: x["count"] / x["count"].sum()) \
    .assign(made_purchase=lambda x: x["made_purchase"].replace({0: "No", 1: "Yes"})) \
    .assign(data_label=lambda x: x["made_purchase_prop"].apply(lambda x: f'{x:.0%}'))

# - Plot
sns.set_theme(style="darkgrid", palette=None)
ax = sns.barplot(data=df_made_purchase_prop, x="made_purchase", y="count", color="#1f77b4", alpha=0.8, width=0.6)

# Add Data Labels
ax.bar_label(ax.containers[0], labels=df_made_purchase_prop["data_label"], fontweight="bold")

# Set labels and title
# Set labels and title
ax.set_ylabel("Frequency", fontsize=9)
ax.set_xlabel("", fontsize=9)
ax.set_title("Proportion of Users with Previous Purchase", fontsize=12, fontweight="bold");


# Member Rating
df = data \
    .groupby("member_rating") \
    .agg(
        count = ("member_rating", "size"),
        made_purchase_prop = ("made_purchase", "mean") \
    ) \
    .reset_index() \
    .assign(member_rating = lambda x: x["member_rating"].astype("str")) 

# Theme
sns.set_theme(style="darkgrid", palette=None)

# Subplots
fig, ax1 = plt.subplots()

# Barplot
sns.barplot(data=df, x="member_rating", y="count", width=0.6, color="#1f77b4", alpha=0.8, ax=ax1)

# Create a second y-axis for made_purchase_proportion
ax2 = ax1.twinx()

# Plot made_purchase_proportion as a line chart
sns.lineplot(x='member_rating', y='made_purchase_prop', data=df, linewidth=2, markers='o', color="#ffb482", ax=ax2)

# Remove gridliens from second axis
ax2.grid(b=False)

# Set labels and title
ax1.set_ylabel('Frequency', fontsize=9)
ax2.set_ylabel('\n Made Purchase (%)', fontsize=9)
ax1.set_xlabel('Member Rating', fontsize=9)
ax1.set_title('Count and Made Purchase Proportion', fontsize=12)

# Reduce the fontsize of the axis values
ax1.tick_params(axis='both', labelsize=8)
ax2.tick_params(axis='both', labelsize=8)

# Add data labels to the line plot
for x, y in zip(df['member_rating'], df['made_purchase_prop']):
    ax2.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize = 9, fontweight="bold")
    
# Legend
# ax1.legend(["Count of Users"], loc="upper center", fontsize=8, frameon=False)
# ax2.legend(["Made Purchase (%)"], loc="upper center", bbox_to_anchor=(0.5, 1), fontsize=8, frameon=False)


# Country
df = data \
    .groupby("country_code") \
    .agg(
        count = ("country_code", "size"),
        made_purchase_prop = ("made_purchase", "mean") \
    ) \
    .reset_index() \
    .sort_values("count", ascending=False) \
    .head(10)
    
    
def bar_plot_dual_axis(data, x, y, y2, bar_width=0.6, bar_fill="#1f77b4", alpha=0.8,
                       xlab=None, ylab=None, y2lab=None, title=None):
    
    # Setup
    sns.set_theme(style="darkgrid", palette=None)
    fig, ax1 = plt.subplots()
    
    # Barplot
    sns.barplot(data=df, x=x, y=y, width=bar_width, color=bar_fill, alpha=alpha, ax=ax1)
    
    # Create a second y-axis for made_purchase_proportion
    ax2 = ax1.twinx()
    
    # Plot made_purchase_proportion as a line chart
    sns.lineplot(x=x, y=y2, data=df, linewidth=2, markers='o', color="#ffb482", ax=ax2)
    
    # Remove gridliens from second axis
    ax2.grid(b=False)

    # Set labels and title
    ax1.set_ylabel(ylab, fontsize=9)
    ax2.set_ylabel(y2lab, fontsize=9)
    ax1.set_xlabel(xlab, fontsize=9)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    
    # Reduce the fontsize of the axis values
    ax1.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    
    # Add data labels to the line plot
    for x, y in zip(df[x], df[y2]):
        ax2.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize = 9, fontweight="bold")
    
    
bar_plot_dual_axis(
    df, 
    x="country_code", y="count", y2="made_purchase_prop",
    ylab="Frequency", y2lab="\n Made Purchase (%)", 
    title="Count of Users vs Made Purchase % by Country"
)


# Tag Count
df = data \
    .groupby("tag_count") \
    .agg(
        count = ("tag_count", "size"),
        made_purchase_prop = ("made_purchase", "mean") \
    ) \
    .reset_index() \
    .sort_values("count", ascending=False) \
    .query("tag_count in (5, 10, 15, 20, 25, 30, 35, 40)") \
    .assign(tag_count = lambda x: x["tag_count"].astype("str"))
    
bar_plot_dual_axis(
    df, 
    x="tag_count", y="count", y2="made_purchase_prop",
    ylab="Frequency", y2lab="\n Made Purchase (%)", 
    title="Count of Users vs Made Purchase % by Country"
)


# Correlation ----

# Data
df_corr = data \
    .select_dtypes(include=("int32")) \
    .drop("mailchimp_id", axis=1) \
    .corr() 
    
# Plot
# mask = np.zeros_like(df_corr)
# mask[np.triu_indices_from(mask)] = True
sns.set_theme(style="darkgrid", palette=None)
ax = sns.heatmap(data=df_corr, annot=True, cmap="Blues")
ax.set_title("Correlation Heatmap", fontsize=12, fontweight="bold");


# KPIs ---
# - Reduce unnecessary sales emails by 30% while maintaning 99% of sales
# - Segment list - apply sales (hot leads), nuture (cold leads)

# Create a styler object
styler = data.style

# KPI 1: Median Tag Count 
df_kp1_1 = data[["made_purchase", "tag_count"]] \
    .groupby("made_purchase") \
    .agg(
        mean_tag_count    = ("tag_count", "mean"),
        median_tag_count  = ("tag_count", "median"),
        count_subscribers = ("tag_count", "count")
    ) \
    .reset_index() \
    .assign(made_purchase=lambda x: x["made_purchase"].replace({0: "No", 1: "Yes"})) \
    .rename(columns = lambda x: x.replace("_", " ").title()) \
    #.rename_axis(index=lambda x: x.title().replace("_", " ")) 
    
    
df_kp1_1 \
    .style \
    .set_table_styles([
          {'selector': 'th', 'props': [
                ('background-color', '#2c3e50'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('font-size', '16px')

        ]}
    ], overwrite=False)
        
  
    

# ! ARCHIVE

#  Dual Axis Function ----
sns.set_theme(style="darkgrid", palette=None)

# Subplots
fig, ax1 = plt.subplots()

# Barplot
sns.barplot(data=df, x="country_code", y="count", width=0.6, color="#1f77b4", alpha=0.8, ax=ax1)

# Create a second y-axis for made_purchase_proportion
ax2 = ax1.twinx()

# Plot made_purchase_proportion as a line chart
sns.lineplot(x='country_code', y='made_purchase_prop', data=df, linewidth=2, markers='o', color="#ffb482", ax=ax2)

# Remove gridliens from second axis
ax2.grid(b=False)

# Set labels and title
ax1.set_ylabel('Frequency', fontsize=9)
ax2.set_ylabel('\n Made Purchase (%)', fontsize=9)
ax1.set_xlabel('Country Code', fontsize=9)
ax1.set_title('Count and Made Purchase Proportion', fontsize=12)

# Reduce the fontsize of the axis values
ax1.tick_params(axis='both', labelsize=8)
ax2.tick_params(axis='both', labelsize=8)

# Add data labels to the line plot
for x, y in zip(df['country_code'], df['made_purchase_prop']):
    ax2.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize = 9, fontweight="bold")


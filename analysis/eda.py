# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# EDA FOR ANALYSIS WRITE UP
# ****

# ***************************************************************************************************
# LIBRARIES 
# ***************************************************************************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mizani.formatters import *
import email_lead_scoring as els
import random

# ***************************************************************************************************
# LOAD DATA 
# ***************************************************************************************************
data = els.db_read_els_data()

# ***************************************************************************************************
# EDA ----
# ***************************************************************************************************
# - Made Purchase

# -- Data
df_made_purchase_prop = data \
    .groupby("made_purchase") \
    .agg(count=("made_purchase", "size")) \
    .reset_index() \
    .assign(prop = lambda x: x["count"] / x["count"].sum()) \
    .assign(made_purchase = lambda x: x["made_purchase"].replace({0: "No", 1: "Yes"})) \
    #.assign(data_label=lambda x: x["made_purchase_prop"].apply(lambda x: f'{x:.0%}'))

# -- Plot
sns.set_theme(style="darkgrid", palette=None)
ax = sns.barplot(data=df_made_purchase_prop, x="made_purchase", y="count", alpha=0.8, width=0.6)

# Add data labels
data_labels = ['{:.0%}'.format(x) for x in df_made_purchase_prop["prop"]]
ax.bar_label(ax.containers[0], labels = data_labels, fontsize = 10)

# Set labels and title
ax.set_ylabel("Frequency", fontsize=9)
ax.tick_params(axis='y', labelsize=9)
ax.set_xlabel(None)
ax.set_title("Proportion of Users with Previous Purchase", fontsize=13, fontweight="bold");

# ***************************************************************************************************
# FUNCTION: DUAL AXIS BAR/LINE PLOT 
def bar_plot_dual_axis(data, x, y, y2, bar_width=0.6, bar_fill="#1f77b4", alpha=0.8,
                       xlab=None, ylab=None, y2lab=None, title=None):
    
    # Setup
    sns.set_theme(style="darkgrid", palette=None)
    fig, ax1 = plt.subplots()
    
    # Barplot
    sns.barplot(data=data, x=x, y=y, width=bar_width, color=bar_fill, alpha=alpha, ax=ax1)
    
    # Create a second y-axis for made_purchase_proportion
    ax2 = ax1.twinx()
    
    # Plot made_purchase_proportion as a line chart
    sns.lineplot(x=x, y=y2, data=data, linewidth=2, markers='o', color="#ffb482", ax=ax2)
    
    # Remove gridliens from second axis
    ax2.grid(b=False)

    # Set labels and title
    ax1.set_ylabel(ylab, fontsize=9)
    ax2.set_ylabel(y2lab, fontsize=9)
    ax1.set_xlabel(xlab, fontsize=9)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    
    # Reduce the fontsize of the axis values
    ax1.tick_params(axis='both', labelsize=9)
    ax2.tick_params(axis='both', labelsize=9)
    
    # Add data labels to the line plot
    for x, y in zip(data[x], data[y2]):
        ax2.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize = 10, fontweight="bold")


# ***************************************************************************************************
# - Member Rating

# -- Data
df_member_rating = data \
    .groupby("member_rating") \
    .agg(
        count = ("member_rating", "size"),
        made_purchase_prop = ("made_purchase", "mean") \
    ) \
    .reset_index() \
    .assign(member_rating = lambda x: x["member_rating"].astype("str")) 
    
# -- Plot
bar_plot_dual_axis(
    data  = df_member_rating, 
    x     = "member_rating", 
    y     = "count", 
    y2    = "made_purchase_prop",
    ylab  = "Frequency", 
    y2lab = "\n Made Purchase (%)", 
    title = "Member Rating vs Made Purchase"
)


# ***************************************************************************************************
# - Country

# -- Data
df_country = data \
    .groupby("country_code", as_index=True) \
    .agg(
        count = ("country_code", "size"),
        made_purchase_prop = ("made_purchase", "mean") \
    ) \
    .reset_index() \
    .sort_values("count", ascending=False) \
    .head(10) 
    
# -- Plot
bar_plot_dual_axis(
    data  = df_country, 
    x     = "country_code", 
    y     = "count", 
    y2    = "made_purchase_prop",
    ylab  = "Frequency", 
    y2lab = "\n Made Purchase (%)", 
    title = "Country Code vs Made Purchase"
)

# ***************************************************************************************************
# - Tag Count

# -- Data
df_tags = data \
    .groupby("tag_count") \
    .agg(
        count = ("tag_count", "size"),
        made_purchase_prop = ("made_purchase", "mean") \
    ) \
    .reset_index() \
    .sort_values("count", ascending=False) \
    .query("tag_count in (5, 10, 15, 20, 25, 30, 35, 40)") \
    .assign(tag_count = lambda x: x["tag_count"].astype("str"))

# -- Plot
bar_plot_dual_axis(
    data  = df_tags, 
    x     = "tag_count", 
    y     = "count", 
    y2    = "made_purchase_prop",
    ylab  = "Frequency", 
    y2lab = "\n Made Purchase (%)", 
    title = "Count of Tags (Events) vs Made Purchase"
)

# ***************************************************************************************************
# - Correlation 

# -- Data
df_corr = data \
    .select_dtypes(include=("int32")) \
    .drop("mailchimp_id", axis=1) \
    .corr() 
    
# -- Plot
# mask = np.zeros_like(df_corr)
# mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(6, 4))
sns.set_theme(style="darkgrid", palette=None)
ax = sns.heatmap(data=df_corr, annot=True, cmap="Blues")
ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
ax.tick_params(axis='both', labelsize=9)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)

# ***************************************************************************************************
# KPIs
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
        
# ***************************************************************************************************
# Cost Simulation Heatmap

# - Data
df_cost_simulation = els.cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0.015, 0.025, 0.035],
    customer_conversion_rate=[0.04, 0.05, 0.06]
)

# - Plot
els.cost_plot_simulated_unsub_cost_plotnine(
    simulation_results = df_cost_simulation,
    title              = "Cost Simulation",
    x_lab              = "Customer Conversion Rate (%)",
    y_lab              = "Email List Growth Rate (%)"  
)

# ***************************************************************************************************
# Made Purchase Prop by Tag (Post Data Preprocessing)

# - Data With Tag Flags
df_binary_tags = els.db_read_and_process_els_data()

# - Sample Tag Columns to Plot 
columns = [
    'made_purchase',
    'tag_learning_lab_09', 
    'tag_learning_lab_24',
    'tag_learning_lab_12',
    'tag_learning_lab_33',
    'tag_webinar_no_degree',
    'tag_learning_lab_13',
    'tag_learning_lab_05',
    'tag_learning_lab_31',
    'tag_learning_lab_41'
]

# - Data
df = df_binary_tags[columns]

# - Plot

# -- Grid setup
sns.set_style("darkgrid")
fig, axs = plt.subplots(3, 3, figsize=(16, 11))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# -- Data Aggregation
for feat, ax in zip(columns[1:], axs.ravel()):  
    
    df_feat = df[["made_purchase", feat]] \
                .groupby([feat, "made_purchase"]) \
                .agg(count=(feat, "size")) \
                .reset_index() \
                .groupby(feat) \
                .apply(lambda x: x.assign(proportion=x["count"] / x["count"].sum())) \
                .reset_index(drop = True) \
                .query("made_purchase == 1") 
                
    df_feat[feat] = df_feat[feat].replace({0: "No", 1: "Yes"})   
   
    
    # -- Plot
    sns.barplot(data = df_feat, x = feat, y = "proportion", alpha = 0.8, width = 0.6, ax = ax)
    
    # -- Data labels
    ax.bar_label(ax.containers[0], labels=['{:.0%}'.format(x) for x in df_feat["proportion"]])
    
    # -- Increase y axis limit
    current_ylim = ax.get_ylim()
    new_ylim = (current_ylim[0], current_ylim[1] + 0.01)
    ax.set_ylim(new_ylim)
    
    # Axis labs
    ax.set_title(f'{feat}')
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    
     # Overall plot title
    fig.suptitle("Proportion of Subscribers With A Purchase by Tag", fontsize = 13, fontweight = "bold")
    plt.show()
    
        
    
# ***************************************************************************************************
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


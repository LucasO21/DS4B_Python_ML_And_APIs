# Email Lead Scoring using Machine Learning

## Table of Contents

- [Email Lead Scoring using Machine Learning](#email-lead-scoring-using-machine-learning)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Problem Statement](#problem-statement)
  - [Solution](#solution)
    - [Business Understanding](#business-understanding)

## Introduction

Businesses face challenges in identifying and prioritizing potential customers and/or identifying future purchase potential of current customers based
on their email interactions, leading to suboptimal allocation of resourcesa nd missed opportunities.

Email lead scoring plays a crucial role in determining the quality and conversion potential of leads generated through email marketing campaigns. Email lead scoring is a method
used by marketers and sales teams to evaluate and prioritize leads based on their potential to become customers. It involves assigning scores or ratings to individual leads
based on their behaviour, interations, and other characteristics. However, the traditional manual lead scoring methods are time-consuming, subjective, and often produce inconsistent results. Additionally, these methods do not fully leverage the available data, such as email content, sender information, and historical customer interactions.

The need of the hour is to develop a machine learning-based solution that can effectively evaluate the probability of leads converting into customers based on various data points extracted from email interactions. This solution should take into account factors like email open rates, click-through rates, response times, engagement patterns, and historical customer data to provide a comprehensive lead score.

By leveraging machine learning algorithms, such as classification models or predictive analytics techniques, the aim is to create a reliable and automated system that can accurately score and rank email leads according to their conversion potential. The solution will empower businesses to prioritize their efforts and resources more effectively, enabling them to focus on the most promising leads and improve overall sales and marketing efficiency.

This analysis works through and end-to-end email lead scoring solution for a business, from analysis to deployment. The projects also uses the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) methodology to see through the project. Skills demonstrated in this project include -

- Business understanding
- Business return on investment and sensitivity analysis
- Machine learning (using tools like python, pycaret, mlflow)
- Model deploying (using tools like fastapi and streamlit)

---

## Problem Statement

The company possesses a large email list of **100,000** subscribers, with a monthly growth rate of **6,000** new subscribers. However, the email list also experiences a significant number of unsubscribes, with an average of **2,500** per month. High unsubscribe rates of **500** people per sales email indicate potential inefficiencies in the email marketing strategy.

The company's sales cycle generates approximately **$250,000** in revenue per month, and the estimated average customer lifetime value is **$2,000**. To sustain and increase revenue, it is crucial to optimize the email marketing approach and maximize customer conversion rates.

The costs associated with email marketing include sending five sales emails per month. Additionally, the company believes that nurturing lost customers has the potential to convert approximately 5% of them back into active customers.

Given these key insights, the problem at hand is to develop an effective email list scoring and segmentation strategy. The goal is to identify and prioritize the most valuable subscribers while reducing unsubscribe rates and increasing overall customer conversions. By segmenting the email list based on various factors and implementing tailored communication and nurturing strategies, the company aims to optimize the use of marketing resources and enhance revenue generation.

In summary, the primary objective is to leverage email list scoring and segmentation techniques to improve customer engagement, reduce unsubscribes, increase customer conversion rates, and ultimately maximize revenue and customer lifetime value.

## Solution

### Business Understanding

In this phase, the key is to emphasize why this is a problem for the business. One way to achive this is to understand the understnad the business problem by **calculating the cost of the business problem**.
We do this by -

- Business Understanding - Understand how high unsubscribe rates leads to lower revenue, thus indicating the need for email segmentation.
- Cost Assessment - Assign a cost to high unsubscribe rates, thus giving the business a point estimate of annual costs of unsubscribe rates. This step does NOT account of growth rate of email lists.
- Improve Cost Analysis - Improve on cost assesment by account for email list growth uncertainty.
- Business Cost Simulation - This is also necessary when accounting for uncertainty and helps model cost when key inputs change.

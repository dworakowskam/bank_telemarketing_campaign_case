# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:18:19 2023

@author: MD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from btcc_eda1 import read_file
from btcc_eda1 import save_file


ver4 = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v4.csv'
ver5 = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v5.csv'


def compare_groups(col_name):
    ct = pd.crosstab(
        index=df[col_name], 
        columns=df["response"], 
        values=df["Client"], 
        aggfunc="count",  
        normalize="columns")
    ct["Diff"] = ct.apply(lambda x: x[0] - x[1], axis=1)
    return ct

def visualize_comparison(ct):
    _ct = ct.drop(columns=["Diff"])
    fig = _ct.plot(kind="bar", title="Comparision between groups", figsize=(15, 10), ylabel="Total ratio")
    return fig




if __name__ == "__main__":
    
    
    df = read_file(ver4)

    # To check whether any correlation between data exists, we need to calculate Pearson's correlation coefficient
    pearson_correlation = df.corr(numeric_only=True)
    plt.figure(figsize=(14,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues')
    plt.title = "Heatmap of correlations"
    # There is no strong or moderate correlation, as all Pearson's r values are lower than 0.3 (and higher than -0.3)
    # except for salary - education, which is quite obvious, as salary is connected with education level
    
    
    # For non-numerical data we can analyse percentage distribution of "yes"/"no" groups
    df['Client'] = df.reset_index().index
    
    # Age
    age_comparison = compare_groups("age")
    visualize_comparison(age_comparison)
    # Result is the same as in previous age analysis
    
    # Marital
    marital_comparison = compare_groups("marital")
    visualize_comparison(marital_comparison)
    # "Single" is the group with about 9% heigher number of "yes" response
    
    # Default
    default_comparison = compare_groups("default")
    visualize_comparison(default_comparison)
    # There are no visible differences between groups
    
    # Housing
    housing_comparison = compare_groups("housing")
    visualize_comparison(housing_comparison)
    # There is a big difference
    
    # Loan
    loan_comparison = compare_groups("loan")
    visualize_comparison(loan_comparison)
    # There is a difference but not as big as in housing
    
    # Contact
    contact_comparison = compare_groups("contact")
    visualize_comparison(contact_comparison)
    # "Unknown" contact correlates with lower probability of "yes" response
    
    # Day
    day_comparison = compare_groups("day")
    visualize_comparison(day_comparison)
    # No valuable conclusions
    
    # Month
    month_comparison = compare_groups("month")
    visualize_comparison(month_comparison)
    # No valuable conclusions
    
    # Campaign
    campaign_comparison = compare_groups("campaign")
    visualize_comparison(campaign_comparison)
    # No valuable conclusions
    
    # Previous
    previous_comparison = compare_groups("previous")
    visualize_comparison(previous_comparison)
    # No valuable conclusions
    
    # Poutcome
    poutcome_comparison = compare_groups("poutcome")
    visualize_comparison(poutcome_comparison)
    # There is significant difference in "success" group
    
    # Job
    job_comparison = compare_groups("Job")
    visualize_comparison(job_comparison)
    # "student" and "retired" job status are groups with the heighest differences
    
    # Education
    education_comparison = compare_groups("Education")
    visualize_comparison(education_comparison)
    # There is 9% difference in "tertiary" group
    
    # Balance groups
    balance_groups_comparison = compare_groups("balance groups")
    visualize_comparison(balance_groups_comparison)
    # The differences are too low
    
    # Pdays groups
    pdays_groups_comparison = compare_groups("pdays groups")
    visualize_comparison(pdays_groups_comparison)
    # The heighest difference is for (50, 100] group
    
    # Previous groups
    previous_groups_comparison = compare_groups("previous groups")
    visualize_comparison(previous_groups_comparison)
    # No significant difference
    
    # SUMMARY - GROUPS FOR FURTHER ANALYSIS:
    # - age equal or under 30 and equal or over 60 - the same as for previous analysis,
    # - "single",
    # - housing loan - negative correlation - the same as for previous analysis,
    # - successfull previous contact - the same as for previous analysis,
    # - "student" and "retired" job status - the same as for previous analysis,
    # - "tertiary" education level,
    # - number of days passed by since the customer has been reached for any of the other products: 50 - 100
    #   - similarly as for previous analysis.
    
    save_file(df, ver5)
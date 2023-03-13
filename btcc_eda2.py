# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:11:17 2023

@author: MD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from btcc_eda1 import read_file
from btcc_eda1 import save_file


ver3 = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v3.csv'
ver4 = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v4.csv'


def analyse_column(column_name):
    result = df.groupby(by=[column_name])["response"].\
        agg(["mean",
             "count",
             lambda x: sum(1-x),
             lambda x: sum(x),
             # 11.7 is total % of "yes" response
             lambda x: sum(x)/len(x) - 0.117
             ])
    result.columns = ["Yes_ratio", "Group_Size",
                      "No_Amount", "Yes_Amount", "Deviation_From_Global"]
    return result

def visualize_analysis_scatter(df, global_mean, title, xlabel, xticks=[]):
    plt.figure(figsize=(15, 5))
    x = df.index.astype(str)
    plt.xticks(xticks)
    y = df["Yes_ratio"]
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel("Yes ratio")
    plt.title(title)
    plt.axhline(global_mean, color="red", linestyle="--")
    plt.show()

def visualize_analysis_bar(df, global_mean, title, xlabel):
    plt.figure(figsize=(15, 5))
    x = df.index.astype(str)
    y = df["Yes_ratio"]
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel("Yes ratio")
    plt.title(title)
    plt.axhline(global_mean, color="red", linestyle="--")
    plt.show()




if __name__ == "__main__":
    
    df = read_file(ver3)
    global_mean = df["response"].mean()
    
    
    # DATA ANALYSIS
    
    # Age
    age_analysis = analyse_column("age")
    visualize_analysis_scatter(
        age_analysis, global_mean, "Response - age analysis", "Age", np.arange(2, 91, step=5))
    # Significantly high response is for age group under 28 and over 60.
    # There is low number of counts for both groups but results seems still reasonable
    
    # Salary
    salary_analysis = analyse_column("salary")
    visualize_analysis_bar(salary_analysis, global_mean, "Response - salary analysis", "Salary")
    # There are two groups with deviation from global and size high enough for further analysis:
    # 4000 (deviation 17%) and 55000 (deviation 11%). Nevertheless there is no visible trend in this chart,
    # so results may be misleading
    
    # Balance
    balance_groups_analysis = analyse_column("balance groups")
    visualize_analysis_bar(balance_groups_analysis, global_mean, "Response - balance groups analysis", "Balance groups")
    # The lowest probability of "yes" response is for group with the lowest balance, then it grows with
    # every next group except for the last. Nevertheless, the differences are too low
    
    # Marital
    response_marital_analysis = analyse_column("marital")
    visualize_analysis_bar(response_marital_analysis, global_mean, "Response - marital status analysis", "Marital status")
    # The heighest yes ratio is for "single" group,
    # but there is too low difference comparing to other groups and mean
    
    # Default
    default_analysis = analyse_column("default")
    visualize_analysis_bar(default_analysis, global_mean, "Response - defaulted loan analysis", "Defaulted loan")
    # There is significant difference between both groups; when having a defaulted loan ,
    # probability of "no" response is much higher, which is logical as people with defaulted loan
    # won't like to open term deposit account
    
    # Housing
    housing_analysis = analyse_column("housing")
    visualize_analysis_bar(housing_analysis, global_mean, "Response - housing loan analysis", "Housing loan")
    # There is significant difference between both groups; when not having a housing loan,
    # probability of "yes" response is two times higher
    
    # Loan
    loan_analysis = analyse_column("loan")
    visualize_analysis_bar(loan_analysis, global_mean, "Response - personal loan analysis", "Personal loan")
    # There is significant difference between both groups; when not having a personal loan,
    # probability of "yes" response is two times higher
    
    # Contact
    contact_analysis = analyse_column("contact")
    visualize_analysis_bar(contact_analysis, global_mean, "Response - contact analysis", "Contact")
    # "Unknown" contact correlates with lower probability of "yes" response
    
    # Day
    day_analysis = analyse_column("day")
    visualize_analysis_bar(day_analysis, global_mean, "Response - day analysis", "Day of month")
    # There seems to be much dispersion in the data, altough there are some peaks which could
    # correlate with paydays (1st and 10th day of month). To check whether this correlation is
    # of significant meaning it needs to be checked whether the values are included in
    # three standard deviations + mean (normal distribution)
    day_mean = day_analysis.Yes_ratio.mean()
    standard_deviation = day_analysis.Yes_ratio.std()
    plt.figure(figsize=(15, 5))
    x = day_analysis.index.astype(str)
    y = day_analysis["Yes_ratio"]
    plt.bar(x, y)
    plt.xlabel("Day of month")
    plt.ylabel("Yes ratio")
    plt.title("Response - day additional analysis")
    plt.axhline(global_mean, color="red", linestyle="--")
    plt.axhline(day_mean+(standard_deviation*3), color="green", linestyle="--")
    plt.axhline(day_mean+(standard_deviation*(-3)),
                color="green", linestyle="--")
    plt.show()
    # Only 1st day of month's value is beyond the normal distribution
    
    # Month
    month_analysis = analyse_column("month")
    visualize_analysis_bar(month_analysis, global_mean, "Response - month analysis", "Month")
    # There seems to be much dispersion in the data, altough there are four months with
    # highest "yes" ratio: March, September, October and December. To be sure if the "month"
    # value is of any meaning, the data should be provided for more than one year to
    # check if this correlation is of any meaning
    
    # Duration
    duration_analysis = analyse_column("duration")
    visualize_analysis_scatter(duration_analysis, global_mean, "Response - call duration analysis", "Call duration")
    # The longer call duration, the heiger "yes" ratio which seems logical as clients
    # less interested finish the call earlier than those who are interested. Nevertheless,
    # call duration is rather effect, not the reason of "yes" response
    
    # Campaign
    campaign_analysis = analyse_column("campaign")
    visualize_analysis_scatter(campaign_analysis, global_mean, "Response - campaign number analysis", "Campaign number", np.arange(0, 58))
    # Only campaigns 1 - 6 have group sizes higher than 1000. Group size decreases consistently with number of campaign.
    # We can visualize campaign number - month analysis to see whether number of campaign is dependent on month
    plt.figure(figsize=(15, 5))
    x = df.month
    y = df.campaign
    plt.scatter(x, y)
    plt.xlabel("Month")
    plt.ylabel("Campaign number")
    plt.xticks(np.arange(0, 13))
    plt.title("Campaign - month analysis")
    plt.show()
    # There is no relevant relationship between the data
    
    # Pdays
    pdays_groups_analysis = analyse_column("pdays groups")
    pdays_analysis = analyse_column("pdays")
    visualize_analysis_bar(pdays_groups_analysis, global_mean, "Response - days passed since last contact by groups", "Days passed since last contact")
    visualize_analysis_scatter(pdays_analysis, global_mean, "Response - days passed since last contact", "Days passed since last contact", np.arange(0,900,step=50))
    # Group 0 - 100 has significantly high yes ratio. Group 400 - 900 is too low to be analyzed
    
    # Previous
    previous_analysis = analyse_column("previous")
    visualize_analysis_scatter(previous_analysis, global_mean, "Response - number of contacts", "Number of contacts", np.arange(0,41))
    previous_groups_analysis = analyse_column("previous groups")
    visualize_analysis_bar(previous_groups_analysis, global_mean, "Response - groups of number of contacts", "Number of contacts")
    # There is no significant difference in any of the groups
    
    # Poutcome
    poutcome_analysis = analyse_column("poutcome")
    visualize_analysis_bar(poutcome_analysis, global_mean, "Response - outcome of previous reach outs", "outcome of previous reach outs")
    # There is significant difference in "success" group, above 50%
    
    # Job
    job_analysis = analyse_column("Job")
    visualize_analysis_bar(job_analysis, global_mean, "Response - job analysis", "Job")
    # There are two groups with deviation from global and size high enough for further analysis:
    # "student" (deviation 17%) and retired (deviation 11%)
    
    # Education
    education_analysis = analyse_column("Education")
    visualize_analysis_bar(education_analysis, global_mean, "Response - education analysis", "Education level")
    # There is visible difference between groups but under 4%
    
    # SUMMARY - GROUPS FOR FURTHER ANALYSIS:
    # - age equal or under 30 and equal or over 60,
    # - salary 4000 and 55000,
    # - defaulted loan - negative correlation,
    # - housing loan - negative correlation,
    # - personal loan - negative correlation,
    # - 1st day of month,
    # - number of days passed by since the customer has been reached for any of the other products: 0 - 100,
    # - successfull previous contact,
    # - "student" and "retired" job status.
    
    save_file(df, ver4)
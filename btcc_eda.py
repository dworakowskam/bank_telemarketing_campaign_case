# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 07:17:44 2023

@author: MD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

kaggle_file = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v1.csv'
fixed_rows_columns = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v2.csv'


def read_kaggle_file(filepath, sep=',', header=0, encoding='UTF-8'):
    return pd.read_csv(filepath, sep=sep, header=header, encoding=encoding)


def save_file(data_frame, filepath, index=False, sep=','):
    return data_frame.to_csv(filepath, index=index, sep=sep)


def read_file(filepath, sep=','):
    return pd.read_csv(filepath, sep=sep)


def create_boxplot(column_name):
    plt.figure(figsize=(15, 5))
    plt.xlabel(f'{column_name}')
    plt.title(f'{column_name} box plot')
    plt.boxplot(df[column_name], vert=False)
    plt.show()


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

    # DATA CLEANING AND PROCESSING

    # COLUMNS
    df = read_kaggle_file(kaggle_file, header=2)

    # Customerid
    # Customerid column can be dropped as its values are duplicated in index column
    df.drop('customerid', axis='columns', inplace=True)

    # Jobedu
    # Jobedu column can be separated into two as it contains two different values
    df[['Job', 'Education']] = df["jobedu"].apply(
        lambda x: pd.Series(str(x).split(",")))
    df.drop('jobedu', axis='columns', inplace=True)

    # Saving file
    save_file(df, fixed_rows_columns)

    # HANDLING MISSING DATA
    df = read_file(fixed_rows_columns)

    # Checking missing values
    missing_values = df.isnull().sum()
    # We have missing values in age, month and response columns

    # Missing values in age column
    missing_age = round(100*(df.age.isnull().sum()/len(df.age)), 2)
    # Missing values are 0.04% of all. As it is low enough, the rows can be removed
    df = df.dropna(subset=['age'])

    # Missing values in month column
    missing_month = round(100*(df.month.isnull().sum()/len(df.month)), 2)
    # Missing values are 0.11% of all. As it is low enough, the rows can be removed
    df = df.dropna(subset=['month'])

    # Missing values in response column
    missing_response = round(
        100*(df.response.isnull().sum()/len(df.response)), 2)
    # Missing values are 0.11% of all. As it is low enough, the rows can be removed
    df = df.dropna(subset=['response'])

    # Missing values in pdays column
    missing_pdays = df.pdays.describe()
    # There are -1 values in pdays column which indicate that customer had never been reached,
    # they will be converted to np.NaN
    df.loc[df.pdays < 0, "pdays"] = np.NaN

    # HANDLING OUTLIERS

    # Age
    age_describe = df.age.describe()
    create_boxplot("age")

    # Salary
    salary_describe = df.salary.describe()
    create_boxplot("salary")

    # Balance
    balance_describe = df.salary.describe()
    create_boxplot("balance")

    # STANDARISING AND CATEGORIZING VALUES

    age_dtype = df.age.dtype
    # Age dtype is float64, it will be changed to integer
    df = df.astype({"age": int})

    # Salary
    salary_describe = df.salary.describe()
    salary_unique = sorted(df.salary.unique())
    # There are only eleven values of salaries which probably represent groups

    # Balance
    # Balance is a continuous variable, it will be divided into categories
    balance_describe = df.balance.describe()
    bins = [-10000, -100, 0, 100, 250, 500, 1000, 2500, 5000, 110000]
    df["balance groups"] = pd.cut(df["balance"], bins=bins)

    # Targeted
    # As there is no description for targeted column, it will be skipped in analysis

    # Default, housing, loan
    default_describe = df.default.describe()
    housing_describe = df.housing.describe()
    loan_describe = df.default.describe()
    # Default, housing and loan consist only of two values: yes and no

    # Contact
    contact_describe = df.contact.describe()
    # Contact consists only of three values

    # Day
    day_describe = df.day.describe()
    # Day consists of 31 values

    # Month
    month_describe = df.month.describe()
    # Month consists of 12 values. It would be convenient for visualization to
    # enumerate values for sorting
    jan = df["month"] == "jan, 2017"
    df.loc[jan, "month"] = 1
    feb = df["month"] == "feb, 2017"
    df.loc[feb, "month"] = 2
    mar = df["month"] == "mar, 2017"
    df.loc[mar, "month"] = 3
    apr = df["month"] == "apr, 2017"
    df.loc[apr, "month"] = 4
    may = df["month"] == "may, 2017"
    df.loc[may, "month"] = 5
    jun = df["month"] == "jun, 2017"
    df.loc[jun, "month"] = 6
    jul = df["month"] == "jul, 2017"
    df.loc[jul, "month"] = 7
    aug = df["month"] == "aug, 2017"
    df.loc[aug, "month"] = 8
    sep = df["month"] == "sep, 2017"
    df.loc[sep, "month"] = 9
    octo = df["month"] == "oct, 2017"
    df.loc[octo, "month"] = 10
    nov = df["month"] == "nov, 2017"
    df.loc[nov, "month"] = 11
    dec = df["month"] == "dec, 2017"
    df.loc[dec, "month"] = 12

    # Duration
    duration_describe = df.duration.describe()
    # Duration should be converted into minutes and into float values
    df.duration = df.duration.apply(lambda x: float(
        x.split()[0])/60 if x.find("sec") > 0 else float(x.split()[0]))
    duration_describe = df.duration.describe()

    # Campaign
    campaign_describe = df.campaign.describe()
    campaign_unique = df.campaign.unique()
    # Campaign consists of 48 values

    # Pdays
    pdays_describe = df.pdays.describe()
    # Pdays is a continuous variable, it will be divided into categories
    bins = list(range(0, 401, 50))
    bins.append(900)
    df["pdays groups"] = pd.cut(df["pdays"], bins=bins)

    # Previous
    previous_describe = df.previous.describe()
    previous_unique = df.previous.unique()
    # Previous is a continuous variable, it will be divided into categories
    bins = [0, 1, 2, 3]
    bins.append(5)
    bins.append(276)
    df["previous groups"] = pd.cut(df["previous"], bins=bins)
    previous_groups_describe = df["previous groups"].describe()

    # Education
    # It would be convenient for further presentation to enumerate education categories for easier sorting
    education_unique = df.Education.unique()
    primary = df["Education"] == "primary"
    df.loc[primary, "Education"] = "1. Primary"
    secondary = df["Education"] == "secondary"
    df.loc[secondary, "Education"] = "2. Secondary"
    tertiary = df["Education"] == "tertiary"
    df.loc[tertiary, "Education"] = "3. Tertiary"
    unknown = df["Education"] == "unknown"
    df.loc[unknown, "Education"] = "4. Unknown"

    # Poutcome
    poutcome_percentage = df.poutcome.value_counts(normalize=True)*100
    # There is a high percentage (80%) of unknown values
    poutcome_without_unknown = df[~(df.poutcome == 'unknown')].poutcome.value_counts(normalize=True)*100

    # Response - target variable
    response_percentage = df.response.value_counts(normalize=True)*100
    # For further analysis it would be convenient to change yes/no values for 1/0 int values
    yes = df["response"] == "yes"
    df.loc[yes, "response"] = 1
    no = df["response"] == "no"
    df.loc[no, "response"] = 0
    global_mean = df["response"].mean()
    # Response represents whether the customer has opened the term deposit account
    # (1 for "yes", 0 for "no")

    # -------------------------------------------------------------------------
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
    # - age equal or under 28 and equal or over 60,
    # - salary 4000 and 55000,
    # - defaulted loan - negative correlation,
    # - housing loan - negative correlation,
    # - personal loan - negative correlation,
    # - 1st day of month,
    # - number of days passed by since the customer has been reached for any of the other products: 0 - 100,
    # - successfull previous contact,
    # - "student" and "retired" job status.

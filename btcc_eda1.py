# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:07:05 2023

@author: MD
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 07:17:44 2023

@author: MD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ver1 = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v1.csv'
ver2 = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v2.csv'
ver3 = 'C:/MD/Dokumenty/python/data_analysis/bank_telemarketing_campaign_case/bank_marketing_updated_v3.csv'



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




if __name__ == "__main__":

    
    # DATA CLEANING AND PROCESSING

    # COLUMNS
    df = read_kaggle_file(ver1, header=2)

    # Customerid
    # Customerid column can be dropped as its values are duplicated in index column
    df.drop('customerid', axis='columns', inplace=True)

    # Jobedu
    # Jobedu column can be separated into two as it contains two different values
    df[['Job', 'Education']] = df["jobedu"].apply(
        lambda x: pd.Series(str(x).split(",")))
    df.drop('jobedu', axis='columns', inplace=True)

    # Saving file
    save_file(df, ver2)

    # HANDLING MISSING DATA
    df = read_file(ver2)

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
    primary_num = df["Education"] == "1. Primary"
    df.loc[primary_num, "Education_num"] = 1
    secondary_num = df["Education"] == "2. Secondary"
    df.loc[secondary_num, "Education_num"] = 2
    tertiary_num = df["Education"] == "3. Tertiary"
    df.loc[tertiary_num, "Education_num"] = 3
    unknown_num = df["Education"] == "4. Unknown"
    df.loc[unknown_num, "Education_num"] = 4
        
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
    # Response represents whether the customer has opened the term deposit account
    # (1 for "yes", 0 for "no")
    
    save_file(df, ver3)
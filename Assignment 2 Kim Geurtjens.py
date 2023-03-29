# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:59:16 2023

@author: 20181846
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def df_plus_transposeddf(filename):
    """Function that reads a dataframe in World-bank format and returns a
    dataframe with years as columns and a dataframe with countries as columns
    """
    df = pd.read_csv(filename, skiprows=4)
    
    df_tr = pd.DataFrame.transpose(df)
    df_tr.columns = df_tr.iloc[0]
    df_tr = df_tr.iloc[2:]
    return df, df_tr


climate_change, climate_tr = \
    df_plus_transposeddf("API_19_DS2_en_csv_v2_5346672.csv")

# Delete empty column / row
climate_change = climate_change.drop("Unnamed: 66", axis=1)
climate_tr = climate_tr.drop("Unnamed: 66")

print(climate_change)
print(climate_tr)

# Create dataframe with only relevant information for six largest countries
# including world (arable land and forest area)
arable_forest_all = climate_change.loc[(climate_change["Indicator Name"] 
                                            == "Arable land (% of land area)")
                                           | (climate_change["Indicator Name"]
                                            == "Forest area (% of land area)")]

arable_forest_six = arable_forest_all[(arable_forest_all["Country Name"] ==
                               "Russian Federation") |
                              (arable_forest_all["Country Name"] == "Canada") |
                              (arable_forest_all["Country Name"] == "China") |
                              (arable_forest_all["Country Name"] ==
                               "United States") |
                              (arable_forest_all["Country Name"] == "Brazil") |
                              (arable_forest_all["Country Name"] ==
                               "Australia") |
                              (arable_forest_all["Country Name"] == "World")]
print(arable_forest_six)

# Create transposed dataframe with only relevant information for six largest
# countries
arable_forest_six_tr = pd.DataFrame.transpose(arable_forest_six)
arable_forest_six_tr.columns = arable_forest_six_tr.iloc[0]
arable_forest_six_tr = arable_forest_six_tr.iloc[2:]
print(arable_forest_six_tr)

# Statistical properties of arable land and forest area over six countries
arable_six = arable_forest_six.loc[climate_change["Indicator Name"]
                           == "Arable land (% of land area)"]
print(arable_six.describe())

forest_six = arable_forest_six.loc[climate_change["Indicator Name"]
                           == "Forest area (% of land area)"]
print(forest_six.describe())

# Statistical properties of arable land and forest area over all countries
arable_all = arable_forest_all.loc[climate_change["Indicator Name"]
                           == "Arable land (% of land area)"]
print(arable_all.describe())

forest_all = arable_forest_all.loc[climate_change["Indicator Name"]
                           == "Forest area (% of land area)"]
print(forest_all.describe())

# Statistical properties of arable land and forest area over all years for six
# countries
arable_tr = pd.DataFrame.transpose(arable_six)
arable_tr.columns = arable_tr.iloc[0]
arable_tr = arable_tr.iloc[4:]
print(arable_tr.mean())
print(arable_tr.std())
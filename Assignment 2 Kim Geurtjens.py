# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:59:16 2023

@author: 20181846
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
# Columns without NaN values
arable_six1 = arable_six[["1970", "1980", "1990", "2000", "2010", "2020"]]
print(arable_six1.describe())

forest_six = arable_forest_six.loc[climate_change["Indicator Name"]
                           == "Forest area (% of land area)"]
# Columns without NaN values
forest_six1 = forest_six[["1990", "2000", "2010", "2020"]]
print(forest_six1.describe())

# Statistical properties of arable land and forest area over all countries
arable_all = arable_forest_all.loc[climate_change["Indicator Name"]
                           == "Arable land (% of land area)"]
# Columns without NaN values
arable_all1 = arable_all[["1970", "1980", "1990", "2000", "2010", "2020"]]
print(arable_all1.describe())

forest_all = arable_forest_all.loc[climate_change["Indicator Name"]
                           == "Forest area (% of land area)"]
# Columns without NaN values
forest_all1 = forest_all[["1990", "2000", "2010", "2020"]]
print(forest_all1.describe())

# Statistical properties of arable land and forest area over all years for six
# countries
arable_tr = pd.DataFrame.transpose(arable_six)
arable_tr.columns = arable_tr.iloc[0]
arable_tr = arable_tr.iloc[4:]
print(arable_tr.mean())
print(arable_tr.std())

forest_tr = pd.DataFrame.transpose(forest_six)
forest_tr.columns = forest_tr.iloc[0]
forest_tr = forest_tr.iloc[4:]
print(forest_tr.mean())
print(forest_tr.std())

# Skewness of six different countries and world over all years
arable_tr = arable_tr.astype(float)
arable_tr_clean = arable_tr.dropna()
print("skewness", stats.skew(arable_tr_clean))
print ("kurtosis", stats.kurtosis(arable_tr_clean))

forest_tr = forest_tr.astype(float)
forest_tr_clean = forest_tr.dropna()
print("skewness", stats.skew(forest_tr_clean))
print ("kurtosis", stats.kurtosis(forest_tr_clean))

# Correlation matrices and heatmaps
arable_corr = arable_tr.corr()
print(arable_corr)

forest_corr = forest_tr.corr()
print(forest_corr)

heatmap_arable = sns.heatmap(arable_tr.corr(), vmin=-1, vmax=1, annot=True,
                             cmap="PiYG")
heatmap_arable.set_title("Correlation heatmap arable land")
plt.savefig("heatmap_arable.png", dpi=300, bbox_inches="tight")

# Last heatmap is commented because otherwise they will occur in one plot
#heatmap_forest = sns.heatmap(forest_tr.corr(), vmin=-1, vmax=1, annot=True,
#                             cmap="PiYG")
#heatmap_forest.set_title("Correlation heatmap forest area")
#plt.savefig("heatmap_forest.png", dpi=300, bbox_inches="tight")

# Visualization of data over time
arable_tr["Year"] = arable_tr.index
arable_tr = arable_tr.astype(float)
arable_tr.plot("Year", ["Australia", "Brazil", "Canada", "China",
                  "Russian Federation", "United States", "World"])
plt.xlim(1961, 2020)
plt.ylabel("Arable land (% of land area)")
plt.legend(loc="upper left", fontsize="9")
plt.title("Arable land of biggest countries of the world over time")
plt.savefig("arable_land.png")
plt.show()

forest_tr["Year"] = forest_tr.index
forest_tr = forest_tr.astype(float)
forest_tr.plot("Year", ["Australia", "Brazil", "Canada", "China",
                  "Russian Federation", "United States", "World"])
plt.xlim(1990, 2020)
plt.ylabel("Forest area (% of land area)")
plt.legend(loc="upper left", fontsize="9")
plt.title("Forest area of biggest countries of the world over time")
plt.savefig("forest_area.png")
plt.show()
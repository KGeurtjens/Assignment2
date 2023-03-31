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

# Create dataframes with only relevant information for six largest countries
# with or without world data (arable land and forest area)
arable_forest_all = climate_change.loc[(climate_change["Indicator Name"] 
                                            == "Arable land (% of land area)")
                                           | (climate_change["Indicator Name"]
                                            == "Forest area (% of land area)")]

arable_forest_sixw = arable_forest_all[(arable_forest_all["Country Name"] ==
                               "Russian Federation") |
                              (arable_forest_all["Country Name"] == "Canada") |
                              (arable_forest_all["Country Name"] == "China") |
                              (arable_forest_all["Country Name"] ==
                               "United States") |
                              (arable_forest_all["Country Name"] == "Brazil") |
                              (arable_forest_all["Country Name"] ==
                               "Australia") |
                              (arable_forest_all["Country Name"] == "World")]

arable_forest_six = arable_forest_all[(arable_forest_all["Country Name"] ==
                               "Russian Federation") |
                              (arable_forest_all["Country Name"] == "Canada") |
                              (arable_forest_all["Country Name"] == "China") |
                              (arable_forest_all["Country Name"] ==
                               "United States") |
                              (arable_forest_all["Country Name"] == "Brazil") |
                              (arable_forest_all["Country Name"] ==
                               "Australia")]

# Create transposed dataframe with only relevant information for six largest
# countries and world
#arable_forest_sixw_tr = pd.DataFrame.transpose(arable_forest_sixw)
#arable_forest_sixw_tr.columns = arable_forest_sixw_tr.iloc[0]
#arable_forest_sixw_tr = arable_forest_sixw_tr.iloc[2:]
#DELETE IN THE END IF NOT NEEDED

# Statistical properties of arable land and forest area over all countries
arable_all = arable_forest_all.loc[climate_change["Indicator Name"]
                           == "Arable land (% of land area)"]
# Columns without NaN values
arable_all1 = arable_all[["Country Name", "1970", "1980", "1990", "2000",
                          "2010", "2020"]]
print(arable_all1.describe())

forest_all = arable_forest_all.loc[climate_change["Indicator Name"]
                           == "Forest area (% of land area)"]
# Columns without NaN values
forest_all1 = forest_all[["Country Name", "1990", "2000", "2010", "2020"]]
print(forest_all1.describe())

# Statistical properties of arable land and forest area over six countries
arable_six = arable_forest_six.loc[climate_change["Indicator Name"]
                           == "Arable land (% of land area)"]
# Columns without NaN values
arable_six1 = arable_six[["Country Name", "1970", "1980", "1990", "2000",
                          "2010", "2020"]]
print(arable_six1.describe())

forest_six = arable_forest_six.loc[climate_change["Indicator Name"]
                           == "Forest area (% of land area)"]
# Columns without NaN values
forest_six1 = forest_six[["Country Name", "1990", "2000", "2010", "2020"]]
print(forest_six1.describe())


#years = np.array([1990, 2000, 2010, 2020])
#australia = ([17.4274, 17.1582, 16.8629, 17.4213])
#brazil = ([70.458, 65.9344, 61.2075, 59.4175])
#canada = ([38.8455, 38.793, 38.7395, 38.6955])
#china = ([16.6733, 18.7805, 21.2856, 23.3406])
#russia = (["NaN", 49.4018, 49.7736, 49.7843])
#us = ([33.0223, 33.1302, 33.7494, 33.8669])
#DELETE IN THE END IF NOT NEEDED

names = ["Australia", "Brazil", "Canada", "China", "Russia", "US"]
year1990 = [17.4274, 70.4580, 38.8455,
             16.6733, 0, 33.0223]
year2000 = [17.1582, 65.9344, 38.7930,
             18.7805, 49.4018, 33.1302]
year2010 = [16.8629, 61.2075, 38.7395,
             21.2856, 49.7736, 33.7494]
year2020 = [17.4213, 59.4175, 38.6955,
             23.3406, 49.7843, 33.8669]

X = np.arange(6)
plt.bar(X - 0.3, year1990, 0.2, label="1990")
plt.bar(X - 0.1, year2000, 0.2, label="2000")
plt.bar(X + 0.1, year2010, 0.2, label="2010")
plt.bar(X + 0.3, year2020, 0.2, label="2020")
plt.xticks(X, names)
plt.ylabel("Forest area (% of land area)")
plt.title("Forest area over the years for six largest countries")
plt.legend()
plt.savefig("bar_forest.png")

plt.show()

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

# Skewness of six different countries over all years
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

# First two heatmaps are commented because otherwise they will occur in the
# same plot
#heatmap_arable = sns.heatmap(arable_tr.corr(), vmin=-1, vmax=1, annot=True,
#                             cmap="PiYG")
#heatmap_arable.set_title("Correlation heatmap arable land")
#plt.savefig("heatmap_arable.png", dpi=300, bbox_inches="tight")

#heatmap_forest = sns.heatmap(forest_tr.corr(), vmin=-1, vmax=1, annot=True,
#                             cmap="PiYG")
#heatmap_forest.set_title("Correlation heatmap forest area")
#plt.savefig("heatmap_forest.png", dpi=300, bbox_inches="tight")


climate_bra = climate_change.loc[climate_change["Country Name"] == "Brazil"]
climate_tr_bra = pd.DataFrame.transpose(climate_bra)
climate_tr_bra.columns = climate_tr_bra.iloc[0]
climate_tr_bra = climate_tr_bra.iloc[2:]
climate_tr_bra.columns = climate_tr_bra.iloc[0]
climate_tr_bra = climate_tr_bra.iloc[2:]
climate_tr_bra = climate_tr_bra.astype(float)
climate_tr_bra = climate_tr_bra.rename(columns={
    "Urban population (% of total population)":"Urban pop (% of total)",
    "Population, total":"Total pop",
    "Total greenhouse gas emissions (kt of CO2 equivalent)":
    "Greenhouse emissions",
    "Agriculture, forestry, and fishing, value added (% of GDP)":
    "Agriculture, forestry, fishing",
    "Energy use (kg of oil equivalent per capita)":"Energy use"})
climate_tr_bra = climate_tr_bra[["Urban pop (% of total)",
                "Total pop", "Arable land (% of land area)",
                "Forest area (% of land area)",
                "Greenhouse emissions",
                "Agriculture, forestry, fishing",
                "Energy use"]]

heatmap_climate_bra = sns.heatmap(climate_tr_bra.corr(), vmin=-1, vmax=1,
                                    annot=True, cmap="PiYG")
heatmap_climate_bra.set_title("Correlation heatmap Brazil")
plt.savefig("heatmap_brazil.png", dpi=300, bbox_inches="tight")

# Visualization of data over time for six biggest countries and the world
arable_sixw = arable_forest_sixw.loc[climate_change["Indicator Name"]
                           == "Arable land (% of land area)"]
arablew_tr = pd.DataFrame.transpose(arable_sixw)
arablew_tr.columns = arablew_tr.iloc[0]
arablew_tr = arablew_tr.iloc[4:]

arablew_tr["Year"] = arablew_tr.index
arablew_tr = arablew_tr.astype(float)
arablew_tr.plot("Year", ["Australia", "Brazil", "Canada", "China",
                  "Russian Federation", "United States", "World"])
plt.xlim(1961, 2020)
plt.ylabel("Arable land (% of land area)")
plt.legend(loc="upper left", fontsize="9")
plt.title("Arable land of biggest countries of the world over time")
plt.savefig("arable_land.png")
plt.show()

forest_sixw = arable_forest_sixw.loc[climate_change["Indicator Name"]
                           == "Forest area (% of land area)"]
forestw_tr = pd.DataFrame.transpose(forest_sixw)
forestw_tr.columns = forestw_tr.iloc[0]
forestw_tr = forestw_tr.iloc[4:]

forestw_tr["Year"] = forestw_tr.index
forestw_tr = forestw_tr.astype(float)
forestw_tr.plot("Year", ["Australia", "Brazil", "Canada", "China",
                  "Russian Federation", "United States", "World"])
plt.xlim(1990, 2020)
plt.ylabel("Forest area (% of land area)")
plt.legend(loc="upper left", fontsize="9")
plt.title("Forest area of biggest countries of the world over time")
plt.savefig("forest_area.png")
plt.show()

# Visualization of data over time as the average over six biggest countries
av_six = arable_forest_six.groupby("Indicator Name").mean()
print(av_six)

av_tr_six = pd.DataFrame.transpose(av_six)
print(av_tr_six)

av_tr_six["Year"] = av_tr_six.index
av_tr_six = av_tr_six.astype(float)
av_tr_six.plot("Year", ["Arable land (% of land area)",
                         "Forest area (% of land area)"])
plt.xlim(1960, 2020)
plt.ylabel("% of land area")
plt.legend(loc="center left", fontsize="9")
plt.title("Average arable land and forest area of biggest countries over time")
plt.savefig("arable_forest_sum.png")
plt.show()
### Obtaining and cleaning the data
#The capacity factors for solar and wind and the power demand data were retrieved from the Open Power System Data (OPSD) database {cite}`OPSD2020`.

import pandas as pd
from datetime import datetime
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_excel("./data/Power_system_data_DE.xlsx")

#raw_data.head(20)

raw_data.columns = ["utc_timestamp", "cet_timestamp", "demand", "wind", "offshore", "onshore", "solar"]
data = raw_data.drop(index=[0,1,2,3,4,5]).fillna(0)
for i, row in data.iterrows():
    data.at[i,"year"] = row["utc_timestamp"].year
    data.at[i,"month"] = row["utc_timestamp"].month
    data.at[i,"day"] = row["utc_timestamp"].day
    data.at[i,"date"] = row["utc_timestamp"].date()
    data.at[i,"time"] = row["utc_timestamp"].time()

data_2018 = data.loc[data["year"]==2018]
data_2018_pivot = data_2018.pivot(index=["date"], columns=["time"], values=["wind","solar","demand"])

data_2018.to_pickle('./data/2018_data.pkl')

f, (ax1, ax2, ax3) = plt.subplots(1,3)

for i, row in data_2018_pivot.iterrows():
    ax1.plot(row["wind"].values, lw=0.2, alpha=0.5, color=plt.get_cmap("Blues")(np.random.rand(1)))  
    ax2.plot(row["solar"].values, lw=0.2, alpha=0.5, color=plt.get_cmap("YlOrBr")(np.random.rand(1)*0.5))
    ax3.plot(row["demand"].values, lw=0.2, alpha=0.5, color=plt.get_cmap("Greys")(np.random.rand(1)))
    
ax1.set_xticks([0, 23, 47, 71, 95], labels=["0:00", "6:00", "12:00", "18:00", "24:00"])
ax2.set_xticks([0, 23, 47, 71, 95], labels=["0:00", "6:00", "12:00", "18:00", "24:00"])
ax3.set_xticks([0, 23, 47, 71, 95], labels=["0:00", "6:00", "12:00", "18:00", "24:00"])

ax1.set_ylim([0,1])
ax2.set_ylim([0,1])

ax1.set_xlim([0,95])
ax2.set_xlim([0,95])
ax3.set_xlim([0,95])

ax1.set_ylabel("wind capacity factor")
ax2.set_ylabel("solar capacity factor")
ax3.set_ylabel("power demand / MW")

plt.subplots_adjust(wspace=0.4)

f.set_size_inches([10,4])
plt.savefig('./figures/capacity_factors.png')

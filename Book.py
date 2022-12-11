# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:07:22 2022

@author: ashwi
"""
import numpy as np
import pandas as pd
df = pd.read_csv("book.csv")

df.shape
df.dtypes
df.head()
df.values[0]
df.values[1]

# apriori algorithm 
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(df,min_support=0.01,use_colnames=True,)
frequent_itemsets

# i have applyed different min_support values then i have taken 0.01 value

from mlxtend.frequent_patterns import association_rules
res = association_rules(frequent_itemsets,metric="confidence", min_threshold=0.6)

pd.set_option("display.max_columns",9)
res
# i have applyed different min_threshold values then i have taken 0.6 value

res1 = res[["antecedents","consequents","support","confidence","lift"]]
res1
# showing the results of highest lifting values
res1.nlargest(10, columns="lift")
# therefor the people who are purchasing (ArtBks,ItalAtlas,Italcook,) they are also purchasing(ItalArt,Cookbks,RefBks) 58.27 times more, like that the result is showing top 10 probability.














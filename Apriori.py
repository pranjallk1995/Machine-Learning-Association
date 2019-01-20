#Apriori Algorithm

#importing libraries.
import matplotlib.pyplot as plt
import pandas as pd

#importing data.
dataset = pd.read_csv("Market_Basket_Optimisation.csv")
#Apriori packages requires a list of all transactions.
transactions = []
lendata = len(dataset)
lendatacols = len(dataset.columns)
#takes forever have patience.
for i in range(0, lendata):
    transactions.append([str(dataset.values[i, j]) for j in range(0, lendatacols)])

#applying apriori.
#pip install apyori.
from apyori import apriori
"""min_supp: items bought atleast 3 times a day, data contains items bought over a week.
   confidence: 0.8 means learned rules must follow in 80% of data (but this would be too high)
   parameters must be tuned based on the problem."""
rules = apriori(transactions, min_support = (3 * 7/lendata), min_confidence = 0.2, min_lift = 3, min_lenght = 2)

#visualization.
result = list(rules)
print(result)
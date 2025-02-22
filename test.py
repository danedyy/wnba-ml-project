# import the output.csv file
import pandas as pd
df = pd.read_csv('output.csv')

print(df['playoff_prob'].sum())
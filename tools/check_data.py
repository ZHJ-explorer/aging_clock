import pandas as pd
df = pd.read_csv('preprocessed_data/merged_scaled.csv', index_col=0)
print(f"Samples: {len(df)}, Genes: {len(df.columns)-1}")
print(f"Age range: {df['age'].min():.2f} - {df['age'].max():.2f}")
print(df.head(2))
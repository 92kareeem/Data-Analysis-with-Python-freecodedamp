import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add import statement for seaborn
import seaborn as sns

# Import data
df = pd.read_csv('medical_examination.csv')

# 1. Add an overweight column to the data
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 2. Normalize data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 3. Draw the Categorical Plot
def draw_cat_plot():
    # 4. Create a DataFrame for the cat plot using pd.melt

# Option 1 (remove name argument):
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()

# Option 2 (rename index column):
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(rename_axis='count')

    g = sns.catplot(data=df_cat, x='variable', y='total', hue='value', kind='bar', col='cardio')
    g.set_axis_labels("variable", "total")
    plt.show()
    return g

# 7. Draw the Heat Map
def draw_heat_map():
    # 8. Clean the data in df_heat
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # 9. Calculate the correlation matrix
    corr = df_heat.corr()
    # 10. Generate a mask for the upper triangle
    mask = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))  # Replace np.bool with bool
    # 11. Plot the correlation matrix using sns.heatmap()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    fig.savefig('heatmap.png')
    return fig
    plt.show()

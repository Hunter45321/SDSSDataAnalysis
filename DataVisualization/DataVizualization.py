import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Preprocessing import *
# Create a figure and a grid of subplots with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3,figsize=(18, 12))

# Plot each dataset on a separate subplot
sns.scatterplot(x='u', y='g', hue='class', data=df, ax=axes[0, 0])
sns.scatterplot(x='u', y='r', hue='class', data=df, ax=axes[0, 1])
sns.scatterplot(x='g', y='r', hue='class', data=df, ax=axes[0, 2])
sns.scatterplot(x='redshift', y='r', hue='class', data=df, ax=axes[1, 0])
sns.scatterplot(x='redshift', y='g', hue='class', data=df, ax=axes[1, 1])
sns.scatterplot(x='redshift', y='u', hue='class', data=df, ax=axes[1, 2])

# Add titles to each subplot
axes[0, 0].set_title('g-u plot')
axes[0, 1].set_title('r-u plot')
axes[0, 2].set_title('r-g plot')
axes[1, 0].set_title('r-redshift plot')
axes[1, 1].set_title('g-redshift plot')
axes[1, 2].set_title('u-redshift plot')

plt.tight_layout()
plt.show()

sns.scatterplot(x='ra', y='dec', hue='class', data=df)

sns.scatterplot(x='ra', y='u', hue='class', data=df)

sns.scatterplot(x='dec', y='u', hue='class', data=df)

sns.scatterplot(x='u', y='g', hue='class', data=df)

sns.scatterplot(x='u', y='r', hue='class', data=df)

sns.scatterplot(x='u', y='i', hue='class', data=df[df['i']>0])

sns.scatterplot(x='u', y='z', hue='class', data=df[df['z']>0])

sns.scatterplot(x='g', y='r', hue='class', data=df)

sns.scatterplot(x='g', y='i', hue='class', data=df[df['i']>0])

sns.scatterplot(x='g', y='z', hue='class', data=df[df['z']>0])

sns.scatterplot(x='r', y='i', hue='class', data=df[df['i']>0])

sns.scatterplot(x='r', y='z', hue='class', data=df[df['z']>0])

sns.scatterplot(x='i', y='z', hue='class', data=df)

sns.scatterplot(x='redshift', y='z', hue='class', data=df[df['z']>0])

sns.scatterplot(x='redshift', y='u', hue='class', data=df)

sns.scatterplot(x='redshift', y='g', hue='class', data=df)

sns.scatterplot(x='redshift', y='r', hue='class', data=df)

sns.scatterplot(x='redshift', y='i', hue='class', data=df[df['i']>0])

sns.scatterplot(x='redshift', y='dec', hue='class', data=df)

sns.scatterplot(x='redshift', y='ra', hue='class', data=df)

sns.boxplot(x='class',y="ra",data=df)

sns.boxplot(x='class',y="dec",data=df)

sns.boxplot(x='class',y="u",data=df)

sns.boxplot(x='class',y="g",data=df)

sns.boxplot(x='class',y='r',data=df[df['i']>0])

sns.boxplot(x='class',y='i',data=df[df['i']>0])

sns.boxplot(x='class',y='z',data=df[df['z']>0])

sns.boxplot(x='class',y="redshift",data=df)

# Create a figure and a grid of subplots with 2 rows and 3 columns
fig, axes = plt.subplots(1, 3,figsize=(6, 3))

# Plot each dataset on a separate subplot
sns.boxplot(x='class',y="redshift",data=df, ax=axes[0])
sns.boxplot(x='class',y="u",data=df, ax=axes[1])
sns.boxplot(x='class',y='z',data=df[df['z']>0],ax=axes[2])

# Add titles to each subplot
axes[0].set_title('redshift distribution')
axes[1].set_title('u distribution')
axes[2].set_title('z distribution')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

sns.histplot(df[df['class']=='STAR']['g'])

sns.histplot(data=df, x='g', hue='class')

sns.histplot(data=df, x='u', hue='class')

sns.histplot(data=df, x='r', hue='class')

sns.histplot(data=df[df['i']>0], x='i', hue='class')

sns.histplot(data=df[df['z']>0], x='z', hue='class')

sns.histplot(data= df, x='redshift', hue='class')
plt.xlim(0, 3) 
plt.ylim(0, 1000) 
plt.show()

sns.countplot(x='class', data=df)
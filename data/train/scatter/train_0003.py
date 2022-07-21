"""
###
Prompt:
###
---
Using the following parameters:
legend: on
title: off
---
Write code that draws a simple line graph using 3 sets of reproducible randomized data. The data should
simulate random cumulative sums over days of a year starting from January 1st, 1942. Be sure to import the
necessary packages and initialize all necessary variables. Use of pandas for inputs would preferred. Makes it
look clean and minimalist. Comment the code to show how you followed the instructions.

Code:
"""
import numpy as np
import pandas as pd

# make the generated data reproducible
np.random.seed(0)

# data labels inferred from data specification,
# specify in prompt for more precise results
labels = ('Timeseries A', 'Timeseries B', 'Timeseries C')

# input data variables - 3 random lines representing cumulative
# sums over one year beginning January, 1st 1942
df = pd.DataFrame({labels[0]: np.random.randn(365).cumsum(0), 
                   labels[1]: np.random.randn(365).cumsum(0) + 20,
                   labels[2]: np.random.randn(365).cumsum(0) - 20}, 
                  index=pd.date_range('1/1/1942', periods=365))

# plot the data using pandas, as per the preference mentioned
df.plot();

ax = plt.gca();

# eliminate borders clean, minimalist look
for spine in ax.spines.values():
    spine.set_visible(False)
    plt.tick_params(top=False, bottom=True, left=True, right=False, 
                    labelleft=True, labelbottom=True)

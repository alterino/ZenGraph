"""
###
Prompt:
###
Using the following input variables:
baseline_data = np.array([1,2,3,4,5,6,7,8])
competition_data = baseline_data**2
---
and the following parameters:
legend: on
---
Write code that draws a simple line graph showing the input variables. Be sure to import the necessary
packages and initialize all necessary variables. Choose appropriate titles and labels. Comment the code to
show how you followed the instructions.

Code:
"""

import numpy as np

# input data variables
baseline_data = np.array([1,2,3,4,5,6,7,8])
competition_data = baseline_data**2

plt.figure()

# create line plot
plt.plot(baseline_data, '-o', competition_data, '-o')

# set axes labels and title -- inferred from variable names --
# Set explicitly for more customized result.
plt.xlabel('Index')
plt.ylabel('Performance')
plt.title('Baseline vs. Competition')

# include a legend
plt.legend(['Baseline', 'Competition'])

plt.show()

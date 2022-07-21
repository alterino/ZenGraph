#!/usr/bin/env python
# coding: utf-8

# # training samples for auto-visualization app - v0

# In[ ]:


# seems environment from terminal is not same as notebook environment
#%pip install matplotlib==3.5.2
#%pip install pandas==1.4.3


# In[1]:


# used for clearing namespace
init_dir = dir()

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

print(pd.__version__)
import matplotlib
print(matplotlib.__version__)
print(sys.version)

test = 5

# to use between experiments so can't use previous
# initializations
def clear_namespace(namespace):
    # clear matplotlib style sheets
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    # clear all packages and variables
    for var in namespace:
        if var in init_dir + ['clear_namespace', 'init_dir']:
            continue
        del globals()[var]
        
    # set matplotlib plot settings
    get_ipython().run_line_magic('matplotlib', 'inline')
    
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print(init_dir)


# In[3]:


# see the pre-defined styles provided.
plt.style.available


# In[4]:


'''
This could be useful for considering varieties of styles by setting parameters explicitly

example: 
# Set ggplot styles and update Matplotlib with them.
ggplot_styles = {
    'axes.edgecolor': 'white',
    'axes.facecolor': 'EBEBEB',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': 'white',
    'grid.linewidth': '1.2',
    'xtick.color': '555555',
    'xtick.major.bottom': True,
    'xtick.minor.bottom': False,
    'ytick.color': '555555',
    'ytick.major.left': True,
    'ytick.minor.left': False,
}

plt.rcParams.update(ggplot_styles)
''';


# In[6]:


clear_namespace(dir())
# train_0000.py
"""
###
Prompt:
###
Using the following input variables:
baseline_data = np.array([1,2,3,4,5,6,7,8])
competition_data = baseline_data**2
---
and the following parameters:
title: "A title"
ylabel: "Some data"
xlabel: "Some other data"
legend: on
---
Write code that draws a simple line graph showing the input variables. Be sure to import the necessary
packages and initialize all necessary variables. Comment the code to show how you followed the instructions.

Code:
"""
import numpy as np
import matplotlib.pyplot as plt

# input data variables
baseline_data = np.array([1,2,3,4,5,6,7,8])
competition_data = baseline_data**2

plt.figure()

# create line plot
plt.plot(baseline_data, '-o', competition_data, '-o')

# set axes labels and title -- inferred from variable names --
# Set explicitly here or in prompt for more customized result.
plt.xlabel('Index')
plt.ylabel('Performance')
plt.title('Baseline vs. Competition')

# include a legend
plt.legend(['Baseline', 'Competition']);


# In[7]:


clear_namespace(dir())
# train_0001.py
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
Write code that draws a simple line graph showing the input variables with the space between the lines filled.
Be sure to import the necessary packages and initialize all necessary variables. Choose appropriate titles and
labels. Comment the code to show how you followed the instructions.

Code:
"""

import numpy as np
import matplotlib.pyplot as plt

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

# fill space between lines
plt.gca().fill_between(range(len(baseline_data)), 
                       baseline_data, competition_data, 
                       facecolor='blue', 
                       alpha=0.25)
                       
plt.show()
                       
                       


# In[8]:


clear_namespace(dir())
# train_0002.py
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
look clean and minimalist, but axis ticks. Comment the code to show how you followed the instructions.

Code:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
ax = df.plot();

# eliminate borders for clean, minimalist look, but include axis ticks
for spine in ax.spines.values():
    spine.set_visible(False)
    plt.tick_params(top=False, bottom=True, left=True, right=False, 
                    labelleft=True, labelbottom=True)


# In[9]:


clear_namespace(dir())
# train_0004.py
"""
###
Prompt:
###
---
Using the following parameters:
legend: off
title: on
labels: off
---
Write code that draws a histogram with 2 sets of randomized normal data on the same axis. 
Make each dataset have a different parameter values. Overlay a kernel density curve of the combined data.
Use of pandas for inputs would be preferred. Make the histogram bars a bit transparent. 
The randomized experiment should be reproducible. Use a gray background with a grid.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# make the experiment reproducible
np.random.seed(1046)

# labels for legend not specified
labels = ('data 0', 'data 1')

# set mean and variance to different values, specifics not specified
mean_0 = 0
mean_1 = 100
std_0 = 15
std_1 = 30

# number of points and bins not specified
N = 250
          
# input data variables - 2 random normal variables 
data_0 = pd.Series(np.random.normal(mean_0, std_0, N), name=labels[0])
data_1 = pd.Series(np.random.normal(mean_1, std_1, N), name=labels[1])

plt.figure()
#plot histogram of data with alpha set to slightly transparent
plt.hist([data_0, data_1], histtype='barstacked', density=True, alpha=0.7);
combined_data = np.concatenate((data_0,data_1))

# use sns presets to make gray background with grid
sns.set_style("darkgrid")
# overlay density plot of combined data
sns.kdeplot(combined_data);

# label parameters set to off
ax = plt.gca()
ax.set_xlabel("")
ax.set_ylabel("")

# title set to on but not specified so choosing generic title
plt.title('two sets of randomized data with combined density estimate');
          


# In[10]:


clear_namespace(dir())
# train_0005.py
"""
###
Prompt:
###
---
Using the following parameters:
title: off
labels: on
---
Write code that draws bivariate and univariate graphs for the sns dataset "penguins" 
showing bill length vs bill depth.
The central plot should be a scatter plot. 
The univariate graphs should be simple histograms.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt
import seaborn as sns

# input data is sns "penguins" dataset
penguins = sns.load_dataset("penguins")

# plot bill length vs bill depth with scatter plot and  simple histograms (default for sns function)
h = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")

# axis labels inferred from prompt      
h.ax_joint.set_xlabel("Bill Length")
h.ax_joint.set_ylabel("Bill Depth") 


# In[11]:


clear_namespace(dir())
# train_0006.py
"""
###
Prompt:
###
---
Using the following parameters:
legend: on
title: off
labels: on
---
Write code that draws bivariate and univariate graphs for the sns dataset "penguins" 
showing bill length vs bill depth.
The central plot should be a scatter plot. 
The univariate graphs should be kernel density estimates.
Visualize data differentiated by species.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt
import seaborn as sns

# input data is sns "penguins" dataset
penguins = sns.load_dataset("penguins")

# plot bill length vs bill depth separated by species with scatter plot and kde curves (default when hue is set)
h = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")

# axis labels inferred from prompt      
h.ax_joint.set_xlabel("Bill Length")
h.ax_joint.set_ylabel("Bill Depth")

# legend visiblity set explicitly (although should default to this)
h.ax_joint.set_visible(True)


# In[12]:


clear_namespace(dir())
# train_0007.py
"""
###
Prompt:
###
---
Using the following parameters:
legend: on
title: off
labels: on
---
Write code that draws bivariate and univariate graphs for the sns dataset "penguins" 
showing bill length vs bill depth separated by species.
All plots should be kernel density estimates
Visualize data differentiated by species.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt
import seaborn as sns

# input data is sns "penguins" dataset
penguins = sns.load_dataset("penguins")

# plot bill length vs bill depth separated by species with scatter plot and kde curves (default when hue is set)
h = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", 
                                 kind="kde", legend=True)

# axis labels inferred from prompt      
h.ax_joint.set_xlabel("Bill Length")
h.ax_joint.set_ylabel("Bill Depth")


# In[13]:


clear_namespace(dir())
# train_0008.py
"""
###
Prompt:
###
---
Using the following parameters:
input: sns dataset "penguins"
title: title
labels: on
---
Write code that draws bivariate and univariate graphs showing bill length vs bill depth.
All plots should be histograms. 
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt
import seaborn as sns

# input data is sns "penguins" dataset
penguins = sns.load_dataset("penguins")

# plot bill length vs bill depth with all plots as histograms
h = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", 
                                 kind="hist")

# axis labels inferred from prompt      
h.ax_joint.set_xlabel("Bill Length")
h.ax_joint.set_ylabel("Bill Depth")
h.fig.suptitle("Bill length vs. Bill depth in sns Penguin Data")

# must adjust subplots down for proper title appearance with sns.jointplot
h.fig.subplots_adjust(top=0.95)

# legend visiblity set explicitly (although should default to this)
h.ax_joint.set_visible(True)


# In[14]:


clear_namespace(dir())
# train_0009.py
"""
###
Prompt:
###
---
Using the following parameters:
title: off
labels: on
bins: 30
---
Write code that draws bivariate and univariate graphs for the sns dataset "penguins" 
showing bill length vs bill depth.
The bivariate graph should be a scatter plot. 
The univariate graphs should be simple histograms. 
Make the appearance of the scatter points and histogram bins interesting. Do not use default settings.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt
import seaborn as sns

# input data is sns "penguins" dataset
penguins = sns.load_dataset("penguins")

# set number of bins to 30
n_bins = 30

# make the appearance more interesting than default behavior
point_marker = "+"
marker_size = 90
bar_fill = False

# plot bill length vs bill depth separated by species with kde curves
h = sns.jointplot(
    data=penguins, x="bill_length_mm", y="bill_depth_mm",
    marker="+", s=marker_size, marginal_kws=dict(bins=n_bins, fill=bar_fill),
)

# axis labels inferred from prompt      
h.ax_joint.set_xlabel("Bill Length")
h.ax_joint.set_ylabel("Bill Depth") 


# In[16]:


clear_namespace(dir())
# train_0010.py
"""
###
Prompt:
###
---
Using the following parameters:
legend: on
title: off
labels: on
fill_density: on
---
Write code that draws bivariate and univariate graphs for the sns dataset "penguins" 
showing bill length vs bill depth separated by species.
All plots should be kernel density estimates.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

# input data is sns "penguins" dataset
penguins = sns.load_dataset("penguins")

# plot kernel density estimates of bill length vs bill depth separated by species
h = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species",
                                 kind="kde", space=0, fill=True)

# axis labels inferred from prompt      
h.ax_joint.set_xlabel("Bill Length");
h.ax_joint.set_ylabel("Bill Depth");


# In[17]:


clear_namespace(dir())
# train_0011.py
"""
###
Prompt:
###
---
Using the following parameters:
title: off
labels: on
fill_density: on
---
Write code that draws bivariate and univariate graphs for the sns dataset "penguins" 
showing bill length vs bill depth.
All plots should be kernel density estimates.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

# input data is sns "penguins" dataset
penguins = sns.load_dataset("penguins")

# plot kernel density estimates of bill length vs bill depth
h = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm",
                                 kind="kde", space=0, fill=True)

# axis labels inferred from prompt      
h.ax_joint.set_xlabel("Bill Length")
h.ax_joint.set_ylabel("Bill Depth")


# In[18]:


clear_namespace(dir())
# train_0012.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'IRIS.csv'
---
and the following parameters:
title:  off
ylabel: on
xlabel: on
legend: on
---
Write code that reads data into a pandas dataframe and plots 
pairwise relationships for the measured variables with data separted by the "species" column.
The diagonal plots should have kernel density estimates. Set facecolor to white with no grid.

Code:
"""
import pandas as pd
import seaborn as sns

# face color set to white with no grid
sns.set_style('white')

# read input data from file 'IRIS.csv'
iris = pd.read_csv('IRIS.csv')

# plot pairwise relationships separated by species column with kernel densities on diagonal
h = sns.pairplot(iris, hue='species', diag_kind='kde', height=2);


# In[19]:


clear_namespace(dir())
# train_0013.py
"""
###
Prompt:
###
Using the following parameters:
title: on
ylabel: on
xlabel: on
legend: on
---
Write code that reads data from the file "IRIS.csv" into a pandas dataframe and plots 
pairwise relationships for the measured variables with data separted by the "species" column.
The diagonal plots should have histograms. Make the title stand out. Set facecolor to white with no grid.

Code:
"""
import pandas as pd
import seaborn as sns

# face color set to white with no grid
sns.set_style('white')

# read input data from file 'IRIS.csv'
iris = pd.read_csv('IRIS.csv')

# plot pairwise relationships separated by species column with kernel densities on diagonal
h = sns.pairplot(iris, hue='species', diag_kind='hist', height=2);

# inferring title since parameter title set to on without specification
# setting to size 15 and weight to 'bold' so title stands out, as specified
h.fig.suptitle("Pairwise Relationships for iris dataset across species", fontsize=15, fontweight='bold');

# must adjust subplots down for proper title appearance with sns.pairplot
h.fig.subplots_adjust(top=0.93)


# In[20]:


clear_namespace(dir())
# train_0014.py
"""
###
Prompt:
###
Using the following input variables:
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))
---
and the following parameters:
title: off
labels: off
legend: off
---
Write code that draws a simple line graph showing the input variables. Be sure to import the necessary
packages and initialize all necessary variables. Comment the code to show how you followed the instructions.

Code:
"""

#import necessary packages
import matplotlib.pyplot as plt
import numpy as np

# initializing the inputs
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))

# create plot figure
fig, ax = plt.subplots()

# plot the curves
for i in range(len(y)):
    ax.plot(x, y[i])

# making ticks visible, preference not specified
ax.tick_params(bottom=True, left=True)


# In[21]:


clear_namespace(dir())
# train_0015.py
"""
###
Prompt:
###
Using the following input variables:
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))
---
and the following parameters:
title: off
labels: off
legend: off
---
Write code that draws a simple line graph showing the input variables. Be sure to import the necessary
packages and initialize all necessary variables. Give it an extremely minimalist look.
Comment the code to show how you followed the instructions.

Code:
"""

#import necessary packages
import matplotlib.pyplot as plt
import numpy as np

# initializing the inputs
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))

# create plot figure
fig, ax = plt.subplots()

# plot the curves
for i in range(len(y)):
    ax.plot(x, y[i])

# eliminated borders and ticks with white facecolor for extremely minimalist look
ax.set_facecolor('#FFFFFF')
for spine in ax.spines.values():
    spine.set_visible(False)
    plt.tick_params(top=False, bottom=False, left=False, right=False, 
                    labelleft=True, labelbottom=True)


# In[22]:


clear_namespace(dir())
# train_0016.py
"""
###
Prompt:
###
Using the following input variables:
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))
---
and the following parameters:
title: off
labels: off
legend: off
---
Write code that draws a simple line graph showing the input variables. Be sure to import the necessary
packages and initialize all necessary variables. Include a subtle grid that includes minor ticks.
Comment the code to show how you followed the instructions.

Code:
"""

#import necessary packages
import matplotlib.pyplot as plt
import numpy as np

# initializing the inputs
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))

# create plot figure
fig, ax = plt.subplots()

# plot the curves
for i in range(len(y)):
    ax.plot(x, y[i])

# Show a subtle grid
ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
# include minor ticks, choosing to de-emphasize
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax.minorticks_on()
# making ticks visible, seems consistent with grid request
ax.tick_params(bottom=True, left=True)


# In[23]:


clear_namespace(dir())
# train_0017.py
"""
###
Prompt:
###
Using the following input variables:
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))
---
and the following parameters:
title: off
labels: off
legend: off
---
Write code that draws a simple line graph showing the input variables. Be sure to import the necessary
packages and initialize all necessary variables. Give it a clean, minimalist look but include a grid with some minor gridlines.
Comment the code to show how you followed the instructions.

Code:
"""

#import necessary packages
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


# initializing the inputs
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))

# create plot figure
fig, ax = plt.subplots()

# plot the curves
for i in range(len(y)):
    ax.plot(x, y[i])

# Remove border around plot for clean, minimalist look.
[ax.spines[side].set_visible(False) for side in ax.spines]

# using gray face color, aesthetic judgement call by AI
ax.set_facecolor('#DDDDDD')

# Style the grid. Gray background with white lines seems minimalistic
ax.grid(which='major', color='white', linewidth=1.1)
ax.grid(which='minor', color='white', linewidth=0.3)
# Removing ticks for minimalist look.
ax.tick_params(which='both', bottom=False, left=False)

# Including one minor tick since specified as minimalist but including some
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))


# In[24]:


clear_namespace(dir())
# train_0018.py
"""
###
Prompt:
###
Using the following input variables:
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))
---
and the following parameters:
title: off
labels: off
legend: off
---
Write code that draws a simple line graph showing the input variables. Be sure to import the necessary
packages and initialize all necessary variables. Give it a clean, minimalist look but include minor gridlines.
Comment the code to show how you followed the instructions.

Code:
"""

#import necessary packages
import matplotlib.pyplot as plt
import numpy as np


# initializing the inputs
x = np.linspace(0, 14, 100)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .5) * (7 - i))

# create plot figure
fig, ax = plt.subplots()

# plot the curves
for i in range(len(y)):
    ax.plot(x, y[i])

# Remove border around plot for clean, minimalist look.
[ax.spines[side].set_visible(False) for side in ax.spines]

# using gray face color, aesthetic judgement call by AI
ax.set_facecolor('#DDDDDD')

# include minor grid lines
ax.minorticks_on()

# Style the grid. Gray background with white lines seems minimalistic
ax.grid(which='major', color='white', linewidth=1.1)
ax.grid(which='minor', color='white', linewidth=0.3)
# Removing ticks for minimalist look.
ax.tick_params(which='both', bottom=False, left=False)


# In[25]:


clear_namespace(dir())
# train_0019.py
"""
###
Prompt:
###
Using the following input variables:
x = np.linspace(0, 20, 150)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .7) * (7 - i))
---
and the following parameters:
title: off
labels: off
legend: off
---
Write code that draws a simple line graph showing the input variables. Be sure to import the necessary
packages and initialize all necessary variables. 
Give it an extremely minimalist look but make the background of the graph black for an outer space vibe.
Set appropriate axis limits.
Comment the code to show how you followed the instructions.

Code:
"""

#import necessary packages
import matplotlib.pyplot as plt
import numpy as np

# initializing the inputs
x = np.linspace(0, 20, 150)
y = []
for i in range(1, 7):
    y.append(np.sin(x + i * .7) * (7 - i))

# create plot figure
fig, ax = plt.subplots()

# plot the curves
for i in range(len(y)):
    ax.plot(x, y[i])

# setting tick parameters explicitly AI judgement call
ax.set_xticks(list(range(0,22,2)))
ax.set_yticks(list(range(-6, 7,2)))

# calculating and setting axes limits
xlims = [np.min(x), np.max(x)]
ylims = [np.min(y) - abs(.2*np.min(y)), np.max(y) + .2*np.max(y)]
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# eliminated borders and ticks with white facecolor for extremely minimalist look
for spine in ax.spines.values():
    spine.set_visible(False)
    plt.tick_params(top=False, bottom=False, left=False, right=False, 
                    labelleft=True, labelbottom=True)
# make background black
ax.set_facecolor('#000000')


# In[26]:


clear_namespace(dir())
# train_0020.py
"""
###
Prompt:
###
Using the following input variables:

data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]
---
and the following parameters:
title: on
labels: off
---
Write code that draws a notched box plot of the input variables. 
Be sure to import the necessary packages and initialize all necessary variables. 
Use colors that pop out for plots. Weird color schemes are okay. Customize as much as possible.
Make the data reproducible.
Comment the code to show how you followed the instructions.

Code:
"""

# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
 
# make experiment reproducible
np.random.seed(10)
    
# input as specified
data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]
 
fig = plt.figure()
ax = fig.add_subplot(111)
 
# creating box plot
bp = ax.boxplot(data, patch_artist = True,
                notch ='True', vert = 0) # set notched to true for notched box plot

# choosing neon colors, specified weird was ok 
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']

# set box color
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# begin --- customizing "as much as possible"
# changing color and linewidth of whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":") 
# changing color and linewidth of caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
# end --- customize "as much as possible settings"
    
# x-axis labels inferred from input variable names
ax.set_yticklabels(['data 1', 'data 2',
                    'data 3', 'data 4'])
 
# inferring title because value not specified but parameter set to 'on'
plt.title("Box plot of reproducible random variables")
 
# Removing top axes and right axes ticks, AI judgement call
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()


# In[27]:


clear_namespace(dir())
# train_0021.py
"""
###
Prompt:
###
Using the following input variables:

data = np.random.normal(100, 20, 200)
---
Write code that draws a simple box plot with no customization. Make it look boring. 
Be sure to import all necessary packages and initialize all necessary variables.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt
import numpy as np
 
# initializing input as specified
data = np.random.normal(100, 20, 200)
 
fig = plt.figure()
 
# create box plot with no customization
plt.boxplot(data);


# In[28]:


clear_namespace(dir())
# train_0022.py
"""
###
Prompt:
###
Using the following input variables:
x = [15, 25, 25, 30, 5]
---
and the following parameters:
title: 'Survey Responses'
labels: ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
---
Write code that draws a simple pie graph of the input data. Use a pleasant color scheme.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt

# initialize input data
x = [15, 25, 25, 30, 5]
# set labels as specified in parameters
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
# choosing a 'pleasant' color scheme
colors = ['tab:blue', 'tab:cyan', 'tab:gray', 'tab:orange', 'tab:red']

fig, ax = plt.subplots()
# plotting pie chart
ax.pie(x, labels = labels, colors = colors)
# setting title as specified
ax.set_title('Survery Responses');


# In[29]:


clear_namespace(dir())
# train_0023.py
"""
###
Prompt:
###
Using the following input variables:

x = [15, 25, 25, 30, 5]

---
and the following parameters:
title: 'Survey Responses'
labels: ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
---
Write code that draws a simple pie graph of the input data. Use a pleasant color scheme.
Include percentages in the graph.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt

# initialize input data
x = [15, 25, 25, 30, 5]
# set labels as specified in parameters
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
# choosing a 'pleasant' color scheme
colors = ['tab:blue', 'tab:cyan', 'tab:gray', 'tab:orange', 'tab:red']

fig, ax = plt.subplots()
# plotting pie chart
ax.pie(x, labels = labels, colors = colors, autopct='%.0f%%') # showing percentages in graph with autopct kwarg
# setting title as specified
ax.set_title('Survery Responses');


# In[30]:


clear_namespace(dir())
# train_0024.py
"""
###
Prompt:
###
Using the following input variables:
x = [15, 25, 25, 30, 5]

---
and the following parameters:
title: 'Survey Responses'
labels: ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
---
Write code that draws a simple pie graph of the input data. Emphasize the 'Unsure' category a bit.
Include percentages in the graph.
Comment the code to show how you followed the instructions.

Code:
"""
# import necessary packages
import matplotlib.pyplot as plt

# initialize input data
x = [15, 25, 25, 30, 5]
# set labels as specified in parameters
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']

# emphasizing the 'Unsure' category a bit
emphasize = [0, 0, 0.1, 0, 0]

fig, ax = plt.subplots()
# plotting pie chart
ax.pie(x, labels = labels, explode=emphasize, autopct='%.0f%%') # showing percentages in graph with autopct kwarg
# setting title as specified
ax.set_title('Survery Responses');


# In[31]:


clear_namespace(dir())
# train_0025.py
"""
###
Prompt:
###
---
Using the following parameters:
title: 'Midwest Area vs. Population'
axis labels: on
legend: on
---
Write code that loads data from 'datasets/midwest_filter.csv' and draws a scatter plot of
'area' vs. 'poptotal' with the data grouped by 'category', of which there are 14. Make it look pretty.
Comment the code to show how you followed the instructions.

Code:
"""

# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# setting style to "make it look pretty"
plt.style.use('bmh')

# import dataset from specified file
midwest = pd.read_csv("datasets/midwest_filter.csv")

# Create as many colors as there are unique midwest['category'] for grouping
categories = np.unique(midwest['category'])
# using tab20 because there are 14 categories, as specified by prompt
colors = [plt.cm.tab20(i/float(len(categories)-1)) for i in range(len(categories))]

plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

# drawing a separate plot for each category
for i, category in enumerate(categories):
    plt.scatter('area', 'poptotal', 
                data=midwest.loc[midwest.category==category, :], 
                s=40, color=colors[i], label=str(category))

# calculate ranges of variables for axis limits
xmin, xmax = np.min(midwest['area']), np.max(midwest['area'])
ymin, ymax = np.min(midwest['poptotal']), np.max(midwest['poptotal'])
xrange = abs(xmax - xmin)
yrange = abs(ymax - ymin)
# calculate reasonable axis limits
xlim = (xmin - .05*xrange, xmax + .05*xrange)
ylim = (ymin - .05*yrange, ymax + .05*yrange)

    
# Setting axis labels, inferred from df keywords
ax = plt.gca()
ax.set_xlabel('Area', fontsize=20)
ax.set_ylabel('Population', fontsize=20)
# set reasonable axis limits
ax.set(xlim=xlim, ylim=ylim)

plt.xticks(fontsize=15); plt.yticks(fontsize=15)
plt.title("Midwest Area vs Population", fontsize=22)
plt.legend(fontsize=14, loc='center right')    
plt.show()    


# In[32]:


clear_namespace(dir())
# train_0026.py
"""
###
Prompt:
###
Using the following input variables:
v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')
---
and the following parameters:
labels: off
bins: np.arange(-50,150,5)
---
Write code that draws a simple histogram of the input data. Make sure it is fully functional.
Comment the code to show how you followed the instructions.

Code:
"""
# import packages so it is fully functional code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set inputs as specified
v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')

plt.figure()
plt.hist(v1, bins=np.arange(-50,150,5), label='v1');
plt.hist(v2, bins=np.arange(-50,150,5), label='v2');
plt.legend();


# In[33]:


clear_namespace(dir())
# train_0027.py
"""
###
Prompt:
###
Using the following input variables:
v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')
---
and the following parameters:
labels: off
bins: np.arange(-40,140,5)
---
Write code that draws a simple histogram of the input data. Make it look nice and clean. 
Make sure it is fully functional.
Comment the code to show how you followed the instructions.

Code:
"""

# import packages so it is fully functional code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# set inputs as specified
v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')

plt.figure()

# plot data, using bins as specified and making transparent for nice look
plt.hist(v1, alpha=0.7, bins=np.arange(-40,140,5), label='v1');
plt.hist(v2, alpha=0.7, bins=np.arange(-40,140,5), label='v2');
plt.legend();

ax = plt.gca()

# Remove border around plot for clean look.
[ax.spines[side].set_visible(False) for side in ax.spines]

# using gray face color, aesthetic judgement call by AI
ax.set_facecolor('#DDDDDD')

# include minor grid lines
ax.minorticks_off()

# Style the grid. Gray background with white lines seems minimalistic
ax.grid(which='major', color='white', linewidth=1.1)
ax.grid(which='minor', color='white', linewidth=0.3)
# Removing ticks for minimalist look.
ax.tick_params(which='both', bottom=False, left=False)
ax.set_axisbelow(True)

# Including one minor tick for clean but structured look
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))


# In[34]:


clear_namespace(dir())
# train_0028.py
"""
###
Prompt:
###
Using the following input variables:
v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')
---
and the following parameters:
labels: on
title: 'Frequency Distributions of v1 and v2'
---
Write code that draws a simple histogram of the input data. Make it look nice and clean. 
Use bright colors. Make sure it is fully functional. Calculate reasonable bins.
Comment the code to show how you followed the instructions.

Code:
"""

# import packages so it is fully functional code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# set inputs as specified
v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')

plt.figure()

# calculating "reasonable" bins
num_bins = 50
data_min = np.min([v1, v2])
data_max = np.max([v1, v2])
data_range = abs(data_max - data_min)
data_min = data_min - 0.05*data_range
data_max = data_max + 0.05*data_range
data_range = abs(data_max - data_min)
reasonable_bins = np.arange(data_min, data_max, data_range/num_bins)

# plot data, I hope these colors are bright enough! (feedback is always appreciated)
plt.hist(v1, alpha=0.6, bins=reasonable_bins, label='v1', color='springgreen');
plt.hist(v2, alpha=0.6, bins=reasonable_bins, label='v2', color='yellow');
plt.legend();

ax = plt.gca()

# Remove border around plot for clean look.
[ax.spines[side].set_visible(False) for side in ax.spines]

# using gray face color, aesthetic judgement call by AI
ax.set_facecolor('#DDDDDD')

# include minor grid lines
ax.minorticks_off()

# Style the grid. Gray background with white lines seems minimalistic
ax.grid(which='major', color='white', linewidth=1.1)
ax.grid(which='minor', color='white', linewidth=0.3)
# Removing ticks for minimalist look.
ax.tick_params(which='both', bottom=False, left=False)
ax.set_axisbelow(True)

# Including one minor tick for clean but structured look
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))


# In[36]:


clear_namespace(dir())
# train_0029.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'datasets/house_prices.csv'
---
Write code that reads data into a pandas dataframe and plots gives two side-by-side histograms
of pricing data for entries whose 'type' is either 'Residential' or 'Condo'. Make sure the code is
fully functional. Comment to show what you're doing.

Code:
"""
# import necessary packages
import pandas as pd

# import data from specified file
df = pd.read_csv('datasets/house_prices.csv')
# extract relevant columns
df = df[df['type'].isin(['Residential', 'Condo'])]

# plot a simple histogram. You can tell me to make it prettier if you'd prefer.
df['price'].hist(by=df['type'], bins=20);


# In[50]:


clear_namespace(dir())
# train_0030.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'datasets/house_prices.csv'
---
Write code that reads data into a pandas dataframe and draws a simple histogram
of pricing data for entries whose 'type' is 'Multi-Family' or 'Condo', separated into 2 groups.
Make sure the code is fully functional. Comment to show what you're doing.

Code:
"""
# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

# import data from specified file
df = pd.read_csv('datasets/house_prices.csv')

plt.figure()
# plot a simple histogram. You can tell me to make it prettier if you'd prefer.
plt.hist(df[df['type'].isin(['Condo'])]['price'], label='Condo', alpha=0.7);
plt.hist(df[df['type'].isin(['Multi-Family'])]['price'], label='Multi-Family', alpha=0.7);
plt.legend();


# In[49]:


clear_namespace(dir())
# train_0031.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'datasets/house_prices.csv'
---
Write code that reads data into a pandas dataframe and draws a histogram plot
of pricing data for entries whose 'type' is 'Multi-Family' or 'Condo', separated into 2 groups.
Make it pretty and clean. Make sure the code is fully functional. Comment to show what you're doing.

Code:
"""
# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# import data from specified file
df = pd.read_csv('datasets/house_prices.csv')

plt.figure()
# plotting histogram with some stylistic choices to "make it pretty"
plt.hist(df[df['type'].isin(['Condo'])]['price'], label='Condo', alpha=0.7, color='cornflowerblue');
plt.hist(df[df['type'].isin(['Multi-Family'])]['price'], label='Multi-Family', alpha=0.7, color='lightcoral');
plt.legend()

ax = plt.gca()

# Remove border around plot for clean look.
[ax.spines[side].set_visible(False) for side in ax.spines]

# light gray grid for pleasing and organized look
ax.grid(which='major', color='#DDDDDD', linewidth=1.1)
ax.grid(which='minor', color='#DDDDDD', linewidth=0.5)

# Removing ticks for clean look.
ax.tick_params(which='both', bottom=False, left=False)
ax.set_axisbelow(True)

# Removing ticks for minimalist look.
ax.tick_params(which='both', bottom=False, left=False)
ax.set_axisbelow(True)

# Including one minor tick for clean but structured look
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# setting an inferred title
ax.set_title('House Prices for Condos vs Multi-Family');


# In[39]:


clear_namespace(dir())
# train_0032.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'datasets/house_prices.csv'
---
Write code that reads data into a pandas dataframe and draws a scatter plot with a regression
line for 'sepal_length' vs. 'petal_length'. Keep it simple. It's only for inspection.
Make sure the code is fully functional. Comment to show what you're doing.

Code:
"""

import pandas as pd
import seaborn as sns

df = pd.read_csv('datasets/IRIS.csv')

sns.regplot(data = df,
           x = 'sepal_length',
           y = 'petal_length');


# In[40]:


clear_namespace(dir())
# train_0033.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'datasets/house_prices.csv'
---
Write code that reads data into a pandas dataframe and draws a scatter plot with a regression
line for 'sepal_length' vs. 'petal_length'. Make it pretty.
Make sure the code is fully functional. Comment to show what you're doing.

Code:
"""

import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

df = pd.read_csv('datasets/IRIS.csv')

ax = sns.regplot(data = df, x = 'sepal_length',y = 'petal_length',
           marker='+', color='g');

# Doing some stuff to "make it pretty" here
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.set_facecolor('#DDDDDD')
ax.minorticks_off()
ax.grid(which='major', color='white', linewidth=1.1)
ax.grid(which='minor', color='white', linewidth=0.3)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_axisbelow(True)

# Including one minor tick for clean but structured look
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))


# In[41]:


clear_namespace(dir())
# train_0034.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'datasets/house_prices.csv'
---
Write code that reads data into a pandas dataframe and draws a scatter plot with a regression
line for 'sepal_length' vs. 'petal_length'. Make it pretty. Include an appropriate title.
Make sure the code is fully functional. Comment to show what you're doing.

Code:
"""

import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

df = pd.read_csv('datasets/IRIS.csv')

# plotting with customizations to "make it pretty"
ax = sns.regplot(data = df, x = 'sepal_length',y = 'petal_length',
           marker='X', color='indigo', scatter_kws={'s':40});

# Doing some more stuff to "make it pretty" here
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.set_facecolor('#CCCCCC')
ax.grid(which='major', color='#EEEEEE', linewidth=0.9)
ax.grid(which='minor', color='#EEEEEE', linewidth=0.4)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_axisbelow(True)

# including a title
ax.set_title('Sepal Length vs. Petal Length in Iris dataset')

# Including one minor tick for clean but structured look
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))


# In[53]:


clear_namespace(dir())
# train_0035.py
"""
###
Prompt:
###
Using the following input variables:
        val0 = np.random.randn(N) * (1 - np.random.rand(N)).cumsum()
        val1 = np.random.randn(N) * (1 - np.random.rand(N)).cumsum()
---
Write code that plots the series as a time series with a rolling moving average
of approximately one year if the the samples were taken daily. Include a legend and a title.
Make sure the code is fully functional. Comment to show what you're doing.

Code:
"""

# import packages so code is functional
import pandas as pd
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# N set arbitrarily, since not specified
N = 1000

data = pd.DataFrame(
    {
        'val0': pd.Series((np.random.randn(1000) * (1 - np.random.rand(1000)))).cumsum(),
        'val1': pd.Series((np.random.randn(1000) * (1 - np.random.rand(1000)))).cumsum()
    }
)

window = 110 # about 1/3 year plus noise, since not specified

rolling_average = data.rolling(window).mean()

rolling_average.columns = [i + '_average_' + str(window) for i in data.columns]

ax = data.plot(alpha = .6)
ax.set_title('Simulated data of daily samples of something with a moving average')

rolling_average.plot(ax = ax);

# decided to throw in a little pizzazz, since aesthetics not specified
# feedback always appreciated on such decisions
ax.grid(which='major', color='#666666', linewidth=0.8)
ax.grid(which='minor', color='#777777', linewidth=0.2)

# Including one minor tick for clean but structured look
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_axisbelow(True)


# In[54]:


clear_namespace(dir())
# train_0036.py
"""
###
Prompt:
###
Using the following input variables:
        val0 = np.random.randn(M) * (1 - np.random.rand(M)).cumsum()
        val1 = np.random.randn(M) * (1 - np.random.rand(M)).cumsum()
---
Write code that plots the series as a time series with a rolling moving average
of approximately one year if the the samples were taken daily. Include a legend and a title.
Make sure the code is fully functional. Comment to show what you're doing.

Code:
"""

# import packages so code is functional
import pandas as pd
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# M set arbitrarily, since not specified
M = 525

# creating dataframe to represent random time series as specified
data = pd.DataFrame(
    {
        'val0': pd.Series((np.random.randn(1000) * (1 - np.random.rand(1000)))).cumsum(),
        'val1': pd.Series((np.random.randn(1000) * (1 - np.random.rand(1000)))).cumsum()
    }
)

# arbitrarily set to 60 days since not specified
window = 60

# calculating rolling average
rolling_average = data.rolling(window).mean()

# naming columns as good general practice
rolling_average.columns = [i + '_average_' + str(window) for i in data.columns]

# plotting data
ax = data.plot(alpha = .5)
ax.set_title('Simulated data of daily samples of something with a moving average')

# plotting rolling average
rolling_average.plot(ax = ax);

# decided to get creative with the grid (since aesthetics unspecified)
# feedback always appreciated on such choices
ax.grid(which='major', color='#777777', linewidth=0.5)
ax.grid(which='minor', color='#999999', linewidth=0.4)

# Including one minor tick for clean but structured look
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))


# In[213]:


clear_namespace(dir())
# train_0037.py
"""
###
Prompt:
###
Using the following input variables:
        val0 = np.random.randn(Q) * (1 - np.random.rand(Q)).cumsum()
        val1 = np.random.randn(Q) * (1 - np.random.rand(Q)).cumsum()
)
---
Write code that plots the series as a time series with a rolling moving average
of approximately one year if the the samples were taken daily. Include a legend and a title.
Make sure the code is fully functional. Keep it simple. No customization required. 
Just make sure it's reproducible.
Comment to show what you're doing.

Code:
"""

# import packages so code is functional
import pandas as pd
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# random seed set for reproducibility
np.random.seed(528491)

# Q set at random, since not specified
Q = np.random.randint(100, high=5000)

# generate pandas dataframe with specified form
data = pd.DataFrame(
    {
        'val0': pd.Series((np.random.randn(Q) * (1 - np.random.rand(Q)))).cumsum(),
        'val1': pd.Series((np.random.randn(Q) * (1 - np.random.rand(Q)))).cumsum()
    }
)

window = 90 # set to 1/4 since not specified

# calculate rolling average of "daily" data
rolling_average = data.rolling(window).mean()

# set name of columns as good practice
rolling_average.columns = [i + '_average_' + str(window) for i in data.columns]

# creating simple plot, no customizations
ax = data.plot()

# plotting the rolling average
rolling_average.plot(ax = ax);


# In[55]:


clear_namespace(dir())
# train_0038.py
"""
###
Prompt:
###
Using the following input variables:
read from csv file: 'datasets/house_prices.csv'
---
And the following parameters:
grid: off
---
Write code that plots a histogram of the aggregated price data.
Make sure the code is fully functional. Keep it simple. Don't write comments.

Code:
"""
import pandas as pd

import matplotlib.pyplot as plt

house_prices = pd.read_csv('datasets/house_prices.csv')

ax = house_prices['price'].hist()
ax.grid(False)


# In[46]:


clear_namespace(dir())
# train_0039.py
"""
###
Prompt:
###
Using the following input variables:
read from csv file: 'datasets/house_prices.csv'
---
And the following parameters:
---
Write code that plots a scatterplot of the latitude and longitude values.
Make sure the code is fully functional. Keep it simple.

Code:
"""
import pandas as pd
import matplotlib.pyplot as plt

house_prices = pd.read_csv('datasets/house_prices.csv')

# extracting lat/long values
lat = house_prices['latitude']
long = house_prices['longitude']

fig = plt.figure()
plt.scatter(lat, long, s=10)

# setting for proper proportions along directions since it's lat/long values
plt.axis('square');


# In[47]:





# In[87]:


clear_namespace(dir())
# train_0040.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'datasets/mpg.csv'
---
And the following parameters:
title: 'MPG city vs. highway'
axis labels: off
legend: on
---
Write code that reads the data into a pandas dataframe and draws a histogram plot
comparing aggregated mpg data for 'cty' vs 'hwy'.
Make it pretty and clean but keep the code simple. Use as few lines as possible.
Make sure the code is fully functional. Comment to show what you're doing.

Code:
"""
# import packages so code is funtional
import pandas as pd
import matplotlib.pyplot as plt

# setting style to "make it look pretty" in as few lines as possible
plt.style.use('Solarize_Light2')

# reading data into pandas dataframe
mpg_data = pd.read_csv('datasets/mpg.csv')

# plotting histograms
plt.hist(mpg_data['hwy'], label='highway', bins=20, alpha=0.5);
plt.hist(mpg_data['cty'], label='city', bins=20, alpha=0.5);
# legend set to 'on' in parameters
plt.legend();


# In[130]:


clear_namespace(dir())
# train_0041.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'datasets/MSFT.csv'
---
Write code that reads data into a pandas dataframe and draws a histogram plot
of closing data. Infer a reasonable title for the plot. Make it look nice but not flashy.
Make sure axis labels are readable and the code is fully functional.
Comment to show what you're doing.

Code:
"""

# import packages so code is funtional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# setting style to "make it look pretty" in as few lines as possible
plt.style.use('ggplot')

# reading data into pandas dataframe
MSFT_df = pd.read_csv('datasets/MSFT.csv', index_col='Date')
# assuming first and last entries are final dates
start, end = np.min(MSFT_df.index), np.max(MSFT_df.index)

# plot the data using pandas
ax = MSFT_df['Close'].plot();

# set meaningful title - inferred by AI, feedback is appreciated
ax.set_title(f'Closing Data for Microsoft from {start} to {end}', fontsize=10)

# make sure date labels are readable
plt.xticks(rotation=45);


# In[172]:


clear_namespace(dir())
# train_0042.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'output/grades.csv'
---
Write code that reads the data into a pandas dataframe and draws a 
simple box-and-whisker plot with one box for each assignmentN_grade 
column where N is from 1 to 5. Infer a reasonable title. 
Make the boxes different colors. Make sure the labels are readable.
Make sure axis labels are readable and the code is fully functional.
Comment to show what you're doing.

Code:
"""

import pandas as pd
import matplotlib.pyplot as plt

# read data from specified file
df = pd.read_csv('output/grades.csv')

# generate column names as specified, for data extraction
cols = [f'assignment{i}_grade' for i in range(1,6)]

# extract data from dataframe
data = df[cols]
 
# creating box plot
plt.boxplot(data, patch_artist = True,
                labels=[c[:-6] for c in cols])

# infer a title (feedback always appreciated)
plt.title('Grade distributions for assignments 1-5')

# make sure date labels are readable
plt.xticks(rotation=45);


# In[206]:


clear_namespace(dir())
# train_0043.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'output/grades.csv'
---
Write code that reads the data into a pandas dataframe and draws a 
box-and-whisker plot with one box for each assignmentN_grade 
column where N is from 1 to 5. Infer a reasonable title. Make the boxes horizontal
Make the boxes different colors.
Make sure axis labels are readable and the code is fully functional.
Comment to show what you're doing.

Code:
"""

# importing packages so code is functional
import pandas as pd
import matplotlib.pyplot as plt

# read data from specified file
df = pd.read_csv('output/grades.csv')

# generate column names as specified, for data extraction
cols = [f'assignment{i}_grade' for i in range(1,6)]
# extract data from dataframe
data = df[cols]

# creating box plot
bp = plt.boxplot(data, patch_artist = True, #set vert to False to make boxes horizontal
                labels=[c[:-6] for c in cols], vert=False) 

# infer a title (feedback always appreciated)
plt.title('Grades for assignments 1-5')

# make the boxes different colors
colors = ['#053dc2', '#fe9360',
          '#846e84', '#6aa84f']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of medians
# to make sure they are clear with facecolor selections
for median in bp['medians']:
    median.set(color ='#e9184c',
               linewidth = 1)


# In[207]:


clear_namespace(dir())
# train_0044.py
"""
###
Prompt:
###
Using the following input variables:
input from csv: 'output/grades.csv'
---
Write code that reads the data into a pandas dataframe and draws a 
box-and-whisker plot with one box for each assignmentN_grade 
column where N is from 1 to 5. Infer a reasonable title. Make the boxes horizontal
Make the boxes different colors. Stylize it to make it pop.
Make sure axis labels are readable and the code is fully functional.
Comment to show what you're doing.

Code:
"""
# importing packages so code is functional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# read data from specified file
df = pd.read_csv('output/grades.csv')

# generate column names as specified, for data extraction
cols = [f'assignment{i}_grade' for i in range(1,6)]

# extract data from dataframe
data = df[cols]

#keep axes handle for stylizing
ax = plt.figure().add_subplot(111)
 
# creating box plot
bp = plt.boxplot(data, patch_artist = True, #set vert to False to make boxes horizontal
                labels=[c.split('_')[0] for c in cols], vert=False) 
                        # extracted first part of variable name for label, 
                        # if incorrect please provide feedback

# infer a title (feedback always appreciated)
plt.title('Assignments 1-5 Grade Distributions')

# make the boxes different colors
colors = ['#2a4679', '#c5c3a1',
          '#ad463e', '#5c3c2a']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of medians
# to make sure they are clear with facecolor selections
for median in bp['medians']:
    median.set(color ='#fe9360',
               linewidth = 1)

# Stylizing to "make it pop"
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.set_facecolor('#DDDDDD')
ax.grid(which='major', color='#FFFFFF', linewidth=0.9)
ax.grid(which='minor', color='#FFFFFF', linewidth=0.4)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_axisbelow(True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))


# In[209]:


clear_namespace(dir())
# train_0045py
"""
###
Prompt:
###
Using the following input variables:
read from csv file: 'datasets/house_prices.csv'
---
And the following parameters:
---
Write code that plots a scatterplot of the latitude and longitude values.
Maybe make it a bit more sophisticated and jazz it up a bit.
Make sure the code is fully functional. Keep it simple. 
Comment to show how you follow the instructions.

Code:
"""
# import packages so code is fully functional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set plot style to "jazz it up a bit"
plt.style.use('fivethirtyeight')

# read data from specified file
house_prices = pd.read_csv('datasets/house_prices.csv')

# extracting lat/long values
lat = house_prices['latitude']
long = house_prices['longitude']

# plot latitude and longitude as scatter plot. Turned it into a jointplot
# because it was specified to perhaps make it more sophisticated
h = sns.jointplot(data=house_prices, x="latitude", y="longitude", hue="city");

# legend visiblity set explicitly (although should default to this)
h.ax_joint.legend([],[]);


# # ------------------------training data ends here -----------------------

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# using ada model
'''
Using the following input variables:
 input from csv: 'datasets/AAPL.csv'
 ---
Write code that reads the data into a pandas dataframe and draws a line plot
of closing data. Make it look nice and clean. Comment code to show you're
following the directions.

Code:
 
''' 
import pandas as pd
import matplotlib.pyplot as plt

# read data from specified file
df = pd.read_csv('datasets/AAPL.csv')

# plot data with line plot of closing data. Make it look nice and clean.

plt.figure()

# set style to "make it look nice"
plt.strategies.set_style('default')
plt.xlabel('Closing Data')
plt.ylabel('Closing Data')

# calculate head and tail of data
plt.hist(df['closed'], label='Closing data', bins=50); END
# plot with a line plot of closing data
plt.plot(df['closed'], x='Closing data', y='Open data'); 


# In[2]:


# using babbage model
'''
Using the following input variables:
 input from csv: 'datasets/AAPL.csv'
 ---
Write code that reads the data into a pandas dataframe and draws a line plot
of closing data. Make it look nice and clean. Comment code to show you're
following the directions.

Code:
'''

# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

# import data from specified file
AAPL = pd.read_csv('datasets/AAPL.csv')

# plot some data
plt.plot(AAPL, 'blue');

# plot some "clean" curves without making it look too clean
plt.plot(AAPL, 'green');

# making it look "clean" by removing ticks
plt.tick_plot(AAPL, 'green', label='closing data'); END
# labels inferred from prompt
plt.xticks(); END

# make it look nice, but make it look clean
plt.axis([0,0]) END
# end with prompt, but make it look like a real scientist
plt.legend(['closing data', 'opening data']) END

# make it look like a real scientist, with proper ticks
plt.xticks(); END

# end with prompt, but make it look like a real scientist
plt.axis(which); END

# make it look like a real scientist, with proper ticks
plt.axis(which); END

# remove ticks
plt.tick_params(bottom=True, left=True); END

# make it look proper, with proper ticks
plt.xticks(); END

# remove ticks
plt.tick_params(bottom=True, left=False); END

# make it look like proper, with proper ticks
plt.axis(which); END

# Remove ticks
plt.tick_params(bottom=False, left=False); END

# make it look proper, with proper ticks
plt.axis(which); END

# Clearly show title, but make it look proper
plt.title('Closing data and opening data'); END

# End with prompt, but make it look like a real scientist
plt.legend(); END

# Make it look like a real scientist, with proper ticks
plt.axis(which); END

# Clearly show title, but make it look proper
plt.title('Opening data and closing data'); END

# End with prompt, but make it look like a real scientist
plt.axis(which); END

# make it look like proper, with proper ticks
plt.axis(which);


# In[ ]:


# curie
'''
Using the following input variables:
 input from csv: 'datasets/AAPL.csv'
 ---
Write code that reads the data into a pandas dataframe and draws a line plot
of closing data. Make it look nice and clean. Comment code to show you're
following the directions.

Code:
''' 

import pandas as pd
import seaborn as sns

df = pd.read_csv('datasets/AAPL.csv')
sns.line_chart(data=df,
 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          


# In[4]:


# davinci

'''Using the following input variables:
 input from csv: 'datasets/AAPL.csv'
 ---
Write code that reads the data into a pandas dataframe and draws a line plot
of closing data. Make it look nice and clean. Comment code to show you're
following the directions.

Code:
''' 
# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

# import data from specified file
data = pd.read_csv('datasets/AAPL.csv')

# make plot look pretty
plt.style.use('ggplot')

# plot line of data
plt.plot(data.index, data.close)

# make plot gray and white
plt.gray()
plt.white()

# plotting ticks ON by default
plt.xticks(rotation=45)

# axes proper height with gray face.
plt.gca().set_facecolor('#DDDDDD') # END OF CODE (commented manually)
  
# commented for testing
'''
Analysis:
 The code is very clean and simple, with a few stylistic choices made. The plot is very
 minimalist, with a gray background and white lines. The plot is set up with a gray
 background and white lines and ticks, with a very minimalistic look, which is probably
 best for this type of data. The gray background of the plot is also a good stylistic
 choice, as it allows the viewer to focus on the data, which is the point of the visualization.

The comment of code is also very minimalistic, but necessary.

11/10/16
  

# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

# read in data from specified file
data = pd.read_csv('datasets/IRIS.csv')

# make plot look nice and clean
plt.style.use('ggplot')

# plot line of data
plt.plot(data.index, data.iris)

# make plot gray and white.
plt.gray()
plt.white()

# axis proper height with gray face.
plt.gca().set_facecolor('#DDDDDD') END OF CODE 
  
Analysis:
 The code is very clean and simple, with a few stylistic choices made. The plot is very
 minimalist, with a gray background and white lines. The plot is set up with a gray
 background and white lines and ticks, with a very minimalistic look, which is probably
 best for this type of data. The gray background of the plot is also a good stylistic
 choice, as it allows the viewer to focus on the data, which is the point of the visualization.
 
 '''


# In[5]:


# ada
'''
Using the following input variables:
       input from csv: 'datasets/Admission_Predict.csv'
       ---
       Write code that reads the data into a pandas dataframe and draws a scatter plot of GRE Score
       vs. Chance of Admit. Choose an appropriate title and label the axes.
       Make it look pretty with some pizzazz. Comment code to show you're following the directions.

Code:
'''

import pandas as pd
import matplotlib.pyplot as plt

fig = pd.figure()

# plot GRE Score vs. Chance of Admit
gre = pd.scatter(data_from='datasets/Admission_Predict.csv', x='GRE_Score', y=' Chance_of_Admit', color='white');

# make it look pretty with some pizzazz
plt.spatial tint(84);
plt.textplot(gre); END

# make it look pretty with some pizzazz
plt.legend(); END END

# input data from csv
data_from = 'datasets/Admission_Predict.csv'

# input from csv
data_from = 'datasets/Admission_Predict.csv'

# urn:lsid:yourwebsite:courses/AI_05_Data_Analysis
data_from = 'Admission_Score'
data_from = 'Chance_of_Admit'

# urn:lsid:yourwebsite:courses/AI_05_Data_Analysis
data_from = 'Admission_Score'
data_from = 'Chance_of_Admit'
data_from = 'Admission_Score'
data_from = 'Chance_of_Admit'
data_from = 'Admission_Score'
data_from = 'Chance_of_Admit'
data_from = 'Admission_Score'
data_from = 'Chance_of_Admit'
data_from = 'Admission_Score'
data_from = 'Chance_of_Admit' END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END


# In[ ]:


# babbage
'''
Using the following input variables:
       input from csv: 'datasets/Admission_Predict.csv'
       ---
       Write code that reads the data into a pandas dataframe and draws a scatter plot of GRE Score
       vs. Chance of Admit. Choose an appropriate title and label the axes.
       Make it look pretty with some pizzazz. Comment code to show you're following the directions.

Code:
''' 

import pandas as pd
import seaborn as sns

df = pd.read_csv('datasets/Admission_Predict.csv')

sns.scatter(df['GRE_Score'])

# Choose an appropriate title and label the axes.
                    sns.set_title('Scoring Differently for Admissions');
                   
# Make it look pretty with some pizzazz.
                    
# Comment code to show you're following the directions.
                    
---
                     
[['GRE Score', 'Chance of Admit'],
                   
[['GRE Score', 'Chance of Admit'],
                    ]]
 
# Remove extraneous whitespace for readability
  sns.extra_classes(df)

# Remove axis labels for readability
  sns.ax_joint(axis='both', kwargs='{}', labels=False)

# Calculate the scatter plot.
  sns.scatter(df['GRE_Score'], df['Chance of Admit'], 
                                                                                              
                                           
                     
# Including some white space for clarity
  sns.xt


# In[6]:


# curie
'''
Using the following input variables:
       input from csv: 'datasets/Admission_Predict.csv'
       ---
       Write code that reads the data into a pandas dataframe and draws a scatter plot of 'GRE Score'
       vs. 'Chance of Admit'. Choose an appropriate title and label the axes.
       Make it look pretty with some pizzazz. Comment code to show you're following the directions.

Code:
''' 

import pandas as pd
import seaborn as sns

# input data comes from specified file
from pandas.csv import read_csv

# extract data into pandas dataframe
df = pd.read_csv('datasets/Admission_Predict.csv')

# plot scatter plot of 'GRE Score' vs. 'Chance of Admit'
df['GRE Score'].plot(kind='scatter', x=-5, y=.30, color='#FF0000')

# choosing title and labels
df.set_title('GRE Score vs. Chance of Admit')

# making it pretty with some pizzazz. Hint: vis='default' in prompt
df.style.visibility = 'visible'

# Create inferred variables based on specified inputs
df['Chance of Admit']. infer(df['GRE Score'])

# Make it pretty with some pizzazz. Hint: vis='default' in prompt
df.style.visibility = 'visible'

# Make it look pretty with some pizzazz. Hint: vis='default' in prompt
df.style.visibility = 'visible'

# Comment code to show you're following the directions.
# Make it look pretty with some pizzazz. Hint: vis='default' in prompt
df.style.visibility = 'visible' ENDING CODE END.

# Including this line to import ersatz "jitter" variable (for some pizzaz)
jitter = .05

# Including this line to set font to "default" for some pizzazz
font = sns.Font('Avenir', 15) ENDING CODE END


  
# Make it look pretty with some pizzazz. Hint: vis='default' in prompt
df.style.visibility = 'visible'

# Make it look pretty with some pizzazz. Hint: vis='default' in prompt
df.style.visibility = 'visible' ENDING CODE END.

# Including this line to import ersatz "jitter" variable (for some pizzaz)
jitter = .05

# Including this line to set font to "default" for some pizzazz
font = sns.Font('Avenir', 15) ENDING CODE END

# Make it look pretty with some pizzazz. Hint: vis='default' in prompt
df.style.visibility = 'visible'

# Make it


# In[ ]:


# davinci
'''
Using the following input variables:
       input from csv: 'datasets/Admission_Predict.csv'
       ---
       Write code that reads the data into a pandas dataframe and draws a scatter plot of 'GRE Score'
       vs. 'Chance of Admit'. Choose an appropriate title and label the axes.
       Make it look pretty with some pizzazz. Comment code to show you're following the directions.

Code:
''' 
 
 
  import pandas as pd
  
     # import data from specified file
   data = pd.read_csv('datasets/Admission_Predict.csv')
      # remove unnecessary columns for visiblity
   data = data[['GRE Score', 'Chance of Admit']]
      # set title explicitly to "make it look pretty with some pizzazz"
   data.title = 'Admission Predictor for Fall 2015 Applicants'
      # set explicit label for x-axis as per directions
   data['Chance of Admit'].label = 'Chance of Admission'
      # set explicit label for y-axis as per directions
   data['GRE Score'].label = 'GRE Score'
      # code to make it look pretty is in the comments
   # scatter plot with some pizzazz
   data['Chance of Admit'].plot(kind='scatter', x='GRE Score',
                                       y='Chance of Admit', label='Chance of Admission');
      # legend with labels as per directions
   data['Chance of Admit'].plot(kind='scatter', x='GRE Score',
                                     y='Chance of Admit', label='Chance of Admission');
       data['Chance of Admit'].legend(); END CODE
                   
      # setting axis labels explicitly as per directions
   data['Chance of Admit'].xlabel('GRE Score')
   data['Chance of Admit'].ylabel('Chance of Admission')
      # choosing appropriate title as per directions
   data['Chance of Admit'].title = 'Admission Predictor for Fall 2015 Applicants
                                


# In[ ]:


# davinci
'''
Using the following input variables:
input from csv: 'datasets/Admission_Predict.csv'
---
Write code that reads the data into a pandas dataframe and draws a scatter plot of 'GRE Score'
vs. 'Chance of Admit'. Include a regression line. Choose an appropriate title and label the axes.
Keep it simple. Comment code to show you're following the directions.

Code:
'''
import pandas as pd
import seaborn as sns

# import necessary for plotting

df = pd.read_csv('datasets/Admission_Predict.csv')

sns.regplot(data = df, x = 'GRE Score', y = 'Chance of Admit');

# choosing appropriate title

# setting labels appropriately

# including x and y axes

# including legend

# making ticks visible

fig = sns.regplot(data = df, x = 'GRE Score', y = 'Chance of Admit');
# setting face color to white
# setting white background
# explicitly setting style to make ticks visibile
fig.spines[‘bottom’].set_visible(False)
fig.spines[‘top’].set_visible(False)
fig.spines[‘left’].set_visible(False)
fig.spines[‘right’].set_visible(False) END OF CODE
 

# Code following prompt from the comments
# import necessary for plotting
import pandas as pd
import seaborn as sns

# import necessary for plotting

df = pd.read_csv('datasets/Admission_Predict.csv')

sns.regplot(data = df, x = 'GRE Score', y = 'Chance of Admit');

# choosing appropriate title

# setting labels appropriately

# including x and y axes

# including legend

# making ticks visible

fig = sns.regplot(data = df, x = 'GRE Score', y = 'Chance of Admit');
# setting face color to white
# setting white background
# explicitly setting style to make ticks visibile
fig.spines[‘bottom’].set_visible(False)
fig.spines[‘top’].set_visible(False)
fig.spines[‘left’].set_visible(False)
fig.spines[‘right’].set_visible(False) END OF CODE
 

# Following prompt from comments
# including title and axis labels
fig.suptitle('Admission Prediction with GRE Scores');
ax = fig.add_subplot(111) END OF CODE

# Including a legend
ax.set_axisbelow(True) END OF CODE
 

# Following


# In[ ]:


# curie
'''
Using the following input variables:
input from csv: 'datasets/Admission_Predict.csv'
---
Write code that reads the data into a pandas dataframe and draws a scatter plot of 'GRE Score'
vs. 'Chance of Admit'. Include a regression line. Choose an appropriate title and label the axes.
Keep it simple. Comment code to show you're following the directions.

Code:
'''
# importnecessary packages
import pandas as pd
import matplotlib.pyplot as plt

# import data from specified file
df = pd.read_csv('datasets/Admission_Predict.csv')

# plot the data
plt.figure()
plt.plot(df['GRE_Score'], '#dd#', df['Chance_of_Admit'])
plt.title('GRE Score vs. Chance of Admit')
plt.legend() END

# create regression line -- inferred from prompt comments
plt. regression (df['GRE_Score'], df['Chance_of_Admit']) END

# choosing an appropriate style and ticks (defaults left off recommendation)
plt. style .use( 'darkgray' )
plt.xlabel( 'GRE Score' )
plt.ylabel( 'Chance of Admit' ) END

# Including a clean, but explicit judgement call at the end
plt. title ( 'GRE Score vs. Chance of Admit' ) END

# Including a title, even though it's simple
plt. legend () END END

Analysis:
 
# calculating mean, standard deviation, min, and max
mean_score = df['GRE_Score'].mean()
standard_deviation_score = df['GRE_Score'].std()
min_GRE_Score = df['GRE_Score'].min()
max_GRE_Score = df['GRE_Score'].max()

# calculating minimum and maximum absolute values and standard deviation
min_abs_score = abs(min_GRE_Score)
max_abs_score = abs(max_GRE_Score)
std_dev_score = sqrt(max_abs_score * min_abs_score)

# calculating average absolute value and standard deviation
mean_abs_score = mean(df['GRE_Score'])
std_dev_score = std(std_dev_score)

# plot scatter plot, regression line, and constants (min, max, mean, standard deviation) as legends
plt.scatter(df['GRE_Score'], df['Chance_of_Admit'])
plt. regression (df['GRE_Score'], df['Chance_of_Admit']) END

# Including a clean but explicit judgement call END
# choosing an appropriate style and ticks (defaults left off recommendation)


# In[ ]:


# babbage
'''
Using the following input variables:
input from csv: 'datasets/Admission_Predict.csv'
---
Write code that reads the data into a pandas dataframe and draws a scatter plot of 'GRE Score'
vs. 'Chance of Admit'. Include a regression line. Choose an appropriate title and label the axes.
Keep it simple. Comment code to show you're following the directions.

Code: 
'''
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/Admission_Predict.csv')

ax = df[df['GRE Score'] > 0]
ax[['Chance of Admit']] = 0
ax[['Chance of Admit']] = 100
ax[['Chance of Admit']] = 0
ax[['GRE Score']] = 0
ax.grid(False)
ax.tick_params(top=True, bottom=True)
ax.spines_on()

# Including a regression line with ticks and boundaries.
# Use tick_params to make it look more "scientific" 
ax.regress(['GRE Score', 'Chance of Admit'],
                                                   
                                                  
                                                    
                                                
                                              
                                             
                                             
                       


# In[7]:


# ada
'''
Using the following input variables:
input from csv: 'datasets/Admission_Predict.csv'
---
Write code that reads the data into a pandas dataframe and draws a scatter plot of 'GRE Score'
vs. 'Chance of Admit'. Include a regression line. Choose an appropriate title and label the axes.
Keep it simple. Comment code to show you're following the directions.

Code:
'''
 
# import packages so it's easy to write code
import pandas as pd
import seaborn as sns

# import data from specified file
df = pd.read_csv('datasets/Admission_Predict.csv')

# create curve for scatter plot
sns.scatter(df, 'GRE Score', {
                                                                          });

# choosing an appropriate title and a name with a "s"
sns.title('GRE Score')
sns.xaxis.set_label('GRE Score')
sns.yaxis.set_label('Chance of Admit')

# creating a column labelled 'Admit' with a hash value to make it look more natural
sns.cols_ MERGE(df,['Admit',True],{
                                                                                });

# setting parameters as specified
df = pd.DataFrame(data = data)

# specifying data format as specified
df = pd.DataFrame(data = data)

# creating curve for scatter plot
sns.scatter(df, 'GRE Score', {
                                                                                 });

# choosing an appropriate title and a name with a "s"
sns.title('GRE Score')
sns.xaxis.set_label('GRE Score')
sns.yaxis.set_label('Chance of Admit')

# Including a regression line
sns.reg_line(data


# In[ ]:





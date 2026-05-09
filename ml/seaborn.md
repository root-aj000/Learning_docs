# Seaborn

## Table of Contents

1. [Seaborn - Introduction](#seaborn---introduction)
2. [Seaborn - Environment Setup](#seaborn---environment-setup)
3. [Seaborn - Importing DataSets and Libraries](#seaborn---importing-datasets-and-libraries)
4. [Seaborn - Figure Aesthetic](#seaborn---figure-aesthetic)
5. [Seaborn - Color Palette](#seaborn---color-palette)
6. [Seaborn - Histogram](#seaborn---histogram)
7. [Seaborn - Kernel Density Estimates](#seaborn---kernel-density-estimates)
8. [Seaborn - Visualizing Pairwise Relationship](#seaborn---visualizing-pairwise-relationship)
9. [Seaborn - Plotting Categorical Data](#seaborn---plotting-categorical-data)
10. [Seaborn - Distribution of Observations](#seaborn---distribution-of-observations)
11. [Seaborn - Statistical Estimation](#seaborn---statistical-estimation)
12. [Seaborn - Plotting Wide Form Data](#seaborn---plotting-wide-form-data)
13. [Seaborn - Multipanel Categorical Plots](#seaborn---multipanel-categorical-plots)
14. [Seaborn - Linear Relationships](#seaborn---linear-relationships)
15. [Seaborn - Facet Grid](#seaborn---facet-grid)
16. [Seaborn - Pair Grid](#seaborn---pair-grid)
17. [Seaborn - Function Reference](#seaborn---function-reference)
18. [Seaborn - Relational Plots](#seaborn---relational-plots)
19. [Seaborn - Distribution Plots](#seaborn---distribution-plots)
20. [Seaborn - Categorical Plots](#seaborn---categorical-plots)
21. [Seaborn - Regression Plots](#seaborn---regression-plots)
22. [Seaborn Matrix Plots - Introduction](#seaborn-matrix-plots---introduction)
23. [Seaborn Multi plot grids - Introduction](#seaborn-multi-plot-grids---introduction)
24. [Seaborn Themeing - Introduction](#seaborn-themeing---introduction)
25. [Seaborn Color Palettes - Introduction](#seaborn-color-palettes---introduction)
26. [Seaborn Palette widgets - Introduction](#seaborn-palette-widgets---introduction)
27. [Seaborn Utility Functions - Introduction](#seaborn-utility-functions---introduction)

---

## 1. Seaborn - Introduction

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_introduction.htm)*

---

---
[Previous](/seaborn/index.htm)[Quiz](/seaborn/quiz_on_seaborn_introduction.htm)[Next](/seaborn/seaborn_environment_setup.htm)
In the world of Analytics, the best way to get insights is by visualizing the data. Data can be visualized by representing it as plots which is easy to understand, explore and grasp. Such data helps in drawing the attention of key elements.

To analyse a set of data using Python, we make use of Matplotlib, a widely implemented 2D plotting library. Likewise, Seaborn is a visualization library in Python. It is built on top of Matplotlib.

## Seaborn Vs Matplotlib

It is summarized that if Matplotlib tries to make easy things easy and hard things possible, Seaborn tries to make a well-defined set of hard things easy too.

Seaborn helps resolve the two major problems faced by Matplotlib; the problems are −

- 
Default Matplotlib parameters

- 
Working with data frames

As Seaborn compliments and extends Matplotlib, the learning curve is quite gradual. If you know Matplotlib, you are already half way through Seaborn.

## Important Features of Seaborn

Seaborn is built on top of Pythons core visualization library Matplotlib. It is meant to serve as a complement, and not a replacement. However, Seaborn comes with some very important features. Let us see a few of them here. The features help in −

- 
Built in themes for styling matplotlib graphics

- 
Visualizing univariate and bivariate data

- 
Fitting in and visualizing linear regression models

- 
Plotting statistical time series data

- 
Seaborn works well with NumPy and Pandas data structures

- 
It comes with built in themes for styling Matplotlib graphics

In most cases, you will still use Matplotlib for simple plotting. The knowledge of Matplotlib is recommended to tweak Seaborns default plots.

---

## 2. Seaborn - Environment Setup

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_environment_setup.htm](https://www.tutorialspoint.com/seaborn/seaborn_environment_setup.htm)*

---

---
[Previous](/seaborn/seaborn_introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_environment_setup.htm)[Next](/seaborn/seaborn_importing_datasets_and_libraries.htm)
In this chapter, we will discuss the environment setup for Seaborn. Let us begin with the installation and understand how to get started as we move ahead.

## Installing Seaborn and getting started

In this section, we will understand the steps involved in the installation of Seaborn.

### Using Pip Installer

To install the latest release of Seaborn, you can use pip −

```
pip3 install seaborn
Collecting seaborn
  Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
  ...
Installing collected packages: seaborn
Successfully installed seaborn-0.13.2
```

### For Windows, Linux & Mac using Anaconda

Anaconda (from
[https://www.anaconda.com/](https://www.anaconda.com/)is a free Python distribution for SciPy stack. It is also available for Linux and Mac.
It is also possible to install the released version using conda −

```
conda install seaborn
```

## Installing Using Source Code

### To install the development version of Seaborn directly from github
[https://github.com/mwaskom/seaborn"](https://github.com/mwaskom/seaborn)
### Dependencies

Consider the following dependencies of Seaborn −

- Python 2.7 or 3.4+
- numpy
- scipy
- pandas
- matplotlib

---

## 3. Seaborn - Importing DataSets and Libraries

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_importing_datasets_and_libraries.htm](https://www.tutorialspoint.com/seaborn/seaborn_importing_datasets_and_libraries.htm)*

---

---
[Previous](/seaborn/seaborn_environment_setup.htm)[Quiz](/seaborn/quiz_on_seaborn_importing_datasets_and_libraries.htm)[Next](/seaborn/seaborn_figure_aesthetic.htm)
In this chapter, we will discuss how to import Datasets and Libraries. Let us begin by understanding how to import libraries.

## Importing Libraries

Let us start by importing Pandas, which is a great library for managing relational (table-format) datasets. Seaborn comes handy when dealing with DataFrames, which is most widely used data structure for data analysis.

The following command will help you import Pandas −

```
# Pandas for managing datasets
import pandas as pd
```

Now, let us import the Matplotlib library, which helps us customize our plots.

```
# Matplotlib for additional customization
from matplotlib import pyplot as plt
```

We will import the Seaborn library with the following command −

```
# Seaborn for plotting and styling
import seaborn as sb
```

## Importing Datasets

We have imported the required libraries. In this section, we will understand how to import the required datasets.

Seaborn comes with a few important datasets in the library. When Seaborn is installed, the datasets download automatically.

You can use any of these datasets for your learning. With the help of the following function you can load the required dataset

```
load_dataset()
```

## Example - Importing Data as Pandas DataFrame

In this section, we will import a dataset. This dataset loads as Pandas DataFrame by default. If there is any function in the Pandas DataFrame, it works on this DataFrame.

The following line of code will help you import the dataset −

### main.py

```
# Seaborn for plotting and styling
import seaborn as sb
df = sb.load_dataset('tips')
print(df.head())
```

#### Output

The above line of code will generate the following output −

```
total_bill  tip   sex    smoker day  time   size
0    16.99    1.01   Female  No    Sun  Dinner  2
1    10.34    1.66   Male    No    Sun  Dinner  3
2    21.01    3.50   Male    No    Sun  Dinner  3
3    23.68    3.31   Male    No    Sun  Dinner  2
4    24.59    3.61   Female  No    Sun  Dinner  4
```

## Example - Viewing All Datasets

To view all the available data sets in the Seaborn library, you can use the following command with the
**get_dataset_names()**function as shown below −
### main.py

```
import seaborn as sb
print(sb.get_dataset_names())
```

#### Output

The above line of code will return the list of datasets available as the following output

```
[u'anscombe', u'attention', u'brain_networks', u'car_crashes', u'dots', 
u'exercise', u'flights', u'fmri', u'gammas', u'iris', u'planets', u'tips', 
u'titanic']
```
**DataFrames**store data in the form of rectangular grids by which the data can be over viewed easily. Each row of the rectangular grid contains values of an instance, and each column of the grid is a vector which holds data for a specific variable. This means that rows of a DataFrame do not need to contain, values of same data type, they can be numeric, character, logical, etc. DataFrames for Python come with the Pandas library, and they are defined as two-dimensional labeled data structures with potentially different types of columns.
For more details on DataFrames, visit our
[tutorial](/python_pandas/index.htm)on pandas.

---

## 4. Seaborn - Figure Aesthetic

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_figure_aesthetic.htm](https://www.tutorialspoint.com/seaborn/seaborn_figure_aesthetic.htm)*

---

---
[Previous](/seaborn/seaborn_importing_datasets_and_libraries.htm)[Quiz](/seaborn/quiz_on_seaborn_figure_aesthetic.htm)[Next](/seaborn/seaborn_color_palette.htm)
Visualizing data is one step and further making the visualized data more pleasing is another step. Visualization plays a vital role in communicating quantitative insights to an audience to catch their attention.

Aesthetics means a set of principles concerned with the nature and appreciation of beauty, especially in art. Visualization is an art of representing data in effective and easiest possible way.

## Example - Using Matplotlib

Matplotlib library highly supports customization, but knowing what settings to tweak to achieve an attractive and anticipated plot is what one should be aware of to make use of it. Unlike Matplotlib, Seaborn comes packed with customized themes and a high-level interface for customizing and controlling the look of Matplotlib figures.

### main.py

```
import numpy as np
from matplotlib import pyplot as plt
def sinplot(flip = 1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5): 
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
sinplot()
plt.show()
```

#### Output

This is how a plot looks with the defaults Matplotlib −
![matplotlib](/seaborn/images/matplotlib.jpg)
## Example - Using Seaborn

To change the same plot to Seaborn defaults, use the
**set()**function −
### main.py

```
import numpy as np
from matplotlib import pyplot as plt
def sinplot(flip = 1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5):
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
import seaborn as sb
sb.set()
sinplot()
plt.show()
```

#### Output
![output](/seaborn/images/output.jpg)
The above two figures show the difference in the default Matplotlib and Seaborn plots. The representation of data is same, but the representation style varies in both.

Basically, Seaborn splits the Matplotlib parameters into two groups−

- 
Plot styles

- 
Plot scale

## Seaborn Figure Styles

The interface for manipulating the styles is
**set_style()**. Using this function you can set the theme of the plot. As per the latest updated version, below are the five themes available.
- 
Darkgrid

- 
Whitegrid

- 
Dark

- 
White

- 
Ticks

Let us try applying a theme from the above-mentioned list. The default theme of the plot will be
**darkgrid**which we have seen in the previous example.
### main.py

```
import numpy as np
from matplotlib import pyplot as plt
def sinplot(flip=1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5):
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
import seaborn as sb
sb.set_style("whitegrid")
sinplot()
plt.show()
```

#### Output
![Darkside](/seaborn/images/darkside.jpg)
The difference between the above two plots is the background color

## Example - Removing Axes Spines

In the white and ticks themes, we can remove the top and right axis spines using the
**despine()**function.
### main.py

```
import numpy as np
from matplotlib import pyplot as plt
def sinplot(flip=1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5):
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
import seaborn as sb
sb.set_style("white")
sinplot()
sb.despine()
plt.show()
```

#### Output
![Spines](/seaborn/images/spines.jpg)
In the regular plots, we use left and bottom axes only. Using the
**despine()**function, we can avoid the unnecessary right and top axes spines, which is not supported in Matplotlib.
## Overriding the Elements

If you want to customize the Seaborn styles, you can pass a dictionary of parameters to the
**set_style()**function. Parameters available are viewed using**axes_style()**function.
### main.py

```
import seaborn as sb
print sb.axes_style
```

#### Output

```
{'axes.axisbelow'     : False,
'axes.edgecolor'      : 'white',
'axes.facecolor'      : '#EAEAF2',
'axes.grid'           : True,
'axes.labelcolor'     : '.15',
'axes.linewidth'      : 0.0,
'figure.facecolor'    : 'white',
'font.family'         : [u'sans-serif'],
'font.sans-serif'     : [u'Arial', u'Liberation  
                        Sans', u'Bitstream Vera Sans', u'sans-serif'],
'grid.color'          : 'white',
'grid.linestyle'      : u'-',
'image.cmap'          : u'Greys',
'legend.frameon'      : False,
'legend.numpoints'    : 1,
'legend.scatterpoints': 1,
'lines.solid_capstyle': u'round',
'text.color'          : '.15',
'xtick.color'         : '.15',
'xtick.direction'     : u'out',
'xtick.major.size'    : 0.0,
'xtick.minor.size'    : 0.0,
'ytick.color'         : '.15',
'ytick.direction'     : u'out',
'ytick.major.size'    : 0.0,
'ytick.minor.size'    : 0.0}
```

## Example - Changing Plot Style

Altering the values of any of the parameter will alter the plot style.

### main.py

```
import numpy as np
from matplotlib import pyplot as plt
def sinplot(flip=1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5):
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
import seaborn as sb
sb.set_style("darkgrid", {'axes.axisbelow': False})
sinplot()
sb.despine()
plt.show()
```

#### Output
![Elements](/seaborn/images/elements.jpg)
## Scaling Plot Elements

We also have control on the plot elements and can control the scale of plot using the
**set_context()**function. We have four preset templates for contexts, based on relative size, the contexts are named as follows
- 
Paper

- 
Notebook

- 
Talk

- 
Poster

By default, context is set to notebook; and was used in the plots above.

### main.py

```
import numpy as np
from matplotlib import pyplot as plt
def sinplot(flip = 1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5):
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
import seaborn as sb
sb.set_style("darkgrid", {'axes.axisbelow': False})
sinplot()
sb.despine()
plt.show()
```

#### Output
![Scaling](/seaborn/images/scaling.jpg)
The output size of the actual plot is bigger in size when compared to the above plots.
**Note**− Due to scaling of images on our web page, you might miss the actual difference in our example plots.

---

## 5. Seaborn - Color Palette

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_color_palette.htm](https://www.tutorialspoint.com/seaborn/seaborn_color_palette.htm)*

---

---
[Previous](/seaborn/seaborn_figure_aesthetic.htm)[Quiz](/seaborn/quiz_on_seaborn_color_palette.htm)[Next](/seaborn/seaborn_histogram.htm)
Color plays an important role than any other aspect in the visualizations. When used effectively, color adds more value to the plot. A palette means a flat surface on which a painter arranges and mixes paints.

## Building Color Palette

Seaborn provides a function called
**color_palette()**, which can be used to give colors to plots and adding more aesthetic value to it.
### Usage

```
seaborn.color_palette(palette = None, n_colors = None, desat = None)
```

### Parameter

The following table lists down the parameters for building color palette −
Sr.No.Palatte & Description1**n_colors**
Number of colors in the palette. If None, the default will depend on how palette is specified. By default the value of
**n_colors**is 6 colors.2**desat**
Proportion to desaturate each color.

### Return

Return refers to the list of RGB tuples. Following are the readily available Seaborn palettes −

- 
Deep

- 
Muted

- 
Bright

- 
Pastel

- 
Dark

- 
Colorblind

Besides these, one can also generate new palette

It is hard to decide which palette should be used for a given data set without knowing the characteristics of data. Being aware of it, we will classify the different ways for using
**color_palette()**types −
- 
qualitative

- 
sequential

- 
diverging

We have another function
**seaborn.palplot()**which deals with color palettes. This function plots the color palette as horizontal array. We will know more regarding**seaborn.palplot()**in the coming examples.
## Example - Qualitative Color Palettes

Qualitative or categorical palettes are best suitable to plot the categorical data.

### main.py

```
from matplotlib import pyplot as plt
import seaborn as sb
current_palette = sb.color_palette()
sb.palplot(current_palette)
plt.show()
```

#### Output
![Colour](/seaborn/images/colour.jpg)
We havent passed any parameters in
**color_palette();**by default, we are seeing 6 colors. You can see the desired number of colors by passing a value to the**n_colors**parameter. Here, the**palplot()**is used to plot the array of colors horizontally.
## Example - Sequential Color Palettes

Sequential plots are suitable to express the distribution of data ranging from relative lower values to higher values within a range.

Appending an additional character s to the color passed to the color parameter will plot the Sequential plot.

### main.py

```
from matplotlib import pyplot as plt
import seaborn as sb
current_palette = sb.color_palette()
sb.palplot(sb.color_palette("Greens"))
plt.show()
```

#### Output
![grid](/seaborn/images/grid.jpg)**Note −**We need to append s to the parameter like Greens in the above example.
## Example - Diverging Color Palette

Diverging palettes use two different colors. Each color represents variation in the value ranging from a common point in either direction.

Assume plotting the data ranging from -1 to 1. The values from -1 to 0 takes one color and 0 to +1 takes another color.

By default, the values are centered from zero. You can control it with parameter center by passing a value.

### main.py

```
from matplotlib import pyplot as plt
import seaborn as sb
current_palette = sb.color_palette()
sb.palplot(sb.color_palette("BrBG", 7))
plt.show()
```

#### Output
![brown](/seaborn/images/brown.jpg)
## Example - Setting the Default Color Palette

The functions
**color_palette()**has a companion called**set_palette()**The relationship between them is similar to the pairs covered in the aesthetics chapter. The arguments are same for both**set_palette()**and**color_palette(),**but the default Matplotlib parameters are changed so that the palette is used for all plots.
### main.py

```
import numpy as np
from matplotlib import pyplot as plt
def sinplot(flip = 1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5):
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

import seaborn as sb
sb.set_style("white")
sb.set_palette("husl")
sinplot()
plt.show()
```

#### Output
![graph](/seaborn/images/pallete.jpg)
## Plotting Univariate Distribution

Distribution of data is the foremost thing that we need to understand while analysing the data. Here, we will see how seaborn helps us in understanding the univariate distribution of the data.

Function
**distplot()**provides the most convenient way to take a quick look at univariate distribution. This function will plot a histogram that fits the kernel density estimation of the data.
### Usage

```
seaborn.distplot()
```

### Parameters

The following table lists down the parameters and their description −
Sr.No.Parameter & Description1**data**
Series, 1d array or a list
2**bins**
Specification of hist bins
3**hist**
bool
4**kde**
bool

These are basic and important parameters to look into.

---

## 6. Seaborn - Histogram

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_histogram.htm](https://www.tutorialspoint.com/seaborn/seaborn_histogram.htm)*

---

---

## 7. Seaborn - Kernel Density Estimates

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_kernel_density_estimates.htm](https://www.tutorialspoint.com/seaborn/seaborn_kernel_density_estimates.htm)*

---

---
[Previous](/seaborn/seaborn_histogram.htm)[Quiz](/seaborn/quiz_on_seaborn_kernel_density_estimates.htm)[Next](/seaborn/seaborn_visualizing_pairwise_relationship.htm)
Kernel Density Estimation (KDE) is a way to estimate the probability density function of a continuous random variable. It is used for non-parametric analysis.

Setting the
**hist**flag to False in**distplot**will yield the kernel density estimation plot.
## Example - Ploting Kernel Density Estimation

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.distplot(df['petal_length'],hist=False)
plt.show()
```

#### Output
![graph](/seaborn/images/curve.jpg)
## Example - Fitting Parametric Distribution
**distplot()**is used to visualize the parametric distribution of a dataset.
### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.distplot(df['petal_length'])
plt.show()
```

#### Output
![histogram](/seaborn/images/bar.jpg)
## Example - Plotting Bivariate Distribution

Bivariate Distribution is used to determine the relation between two variables. This mainly deals with relationship between two variables and how one variable is behaving with respect to the other.

The best way to analyze Bivariate Distribution in seaborn is by using the
**jointplot()**function.
Jointplot creates a multi-panel figure that projects the bivariate relationship between two variables and also the univariate distribution of each variable on separate axes.

## Scatter Plot

Scatter plot is the most convenient way to visualize the distribution where each observation is represented in two-dimensional plot via x and y axis.

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.jointplot(x = 'petal_length',y = 'petal_width',data = df)
plt.show()
```

#### Output
![bar and histogram](/seaborn/images/clubed.jpg)
The above figure shows the relationship between the
**petal_length**and**petal_width**in the Iris data. A trend in the plot says that positive correlation exists between the variables under study.
## Example - Hexbin Plot

Hexagonal binning is used in bivariate data analysis when the data is sparse in density i.e., when the data is very scattered and difficult to analyze through scatterplots.

An addition parameter called kind and value hex plots the hexbin plot.

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.jointplot(x = 'petal_length',y = 'petal_width',data = df,kind = 'hex')
plt.show()
```

#### Output
![HExagon and bar](/seaborn/images/hexagon.jpg)
## Example - Kernel Density Estimation

Kernel density estimation is a non-parametric way to estimate the distribution of a variable. In seaborn, we can plot a kde using
**jointplot().**
Pass value kde to the parameter kind to plot kernel plot.

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.jointplot(x = 'petal_length',y = 'petal_width',data = df,kind = 'hex')
plt.show()
```

#### Output
![xray](/seaborn/images/xray.jpg)

---

## 8. Seaborn - Visualizing Pairwise Relationship

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_visualizing_pairwise_relationship.htm](https://www.tutorialspoint.com/seaborn/seaborn_visualizing_pairwise_relationship.htm)*

---

---
[Previous](/seaborn/seaborn_kernel_density_estimates.htm)[Quiz](/seaborn/quiz_on_seaborn_visualizing_pairwise_relationship.htm)[Next](/seaborn/seaborn_plotting_categorical_data.htm)
Datasets under real-time study contain many variables. In such cases, the relation between each and every variable should be analyzed. Plotting Bivariate Distribution for (n,2) combinations will be a very complex and time taking process.

To plot multiple pairwise bivariate distributions in a dataset, you can use the
**pairplot()**function. This shows the relationship for (n,2) combination of variable in a DataFrame as a matrix of plots and the diagonal plots are the univariate plots.
## Axes

In this section, we will learn what are Axes, their usage, parameters, and so on.

### Usage

```
seaborn.pairplot(data,)
```

### Parameters

Following table lists down the parameters for Axes −
Sr.No.Parameter & Description1**data**
Dataframe
2**hue**
Variable in data to map plot aspects to different colors.
3**palette**
Set of colors for mapping the hue variable
4**kind**
Kind of plot for the non-identity relationships. {scatter, reg}
5**diag_kind**
Kind of plot for the diagonal subplots. {hist, kde}

Except data, all other parameters are optional. There are few other parameters which
**pairplot**can accept. The above mentioned are often used params.
## Example - Using pairplot() function

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.set_style("ticks")
sb.pairplot(df,hue = 'species',diag_kind = "kde",kind = "scatter",palette = "husl")
plt.show()
```

#### Output
![multigraphs](/seaborn/images/multi.jpg)
We can observe the variations in each plot. The plots are in matrix format where the row name represents x axis and column name represents the y axis.

The diagonal plots are kernel density plots where the other plots are scatter plots as mentioned.

---

## 9. Seaborn - Plotting Categorical Data

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_plotting_categorical_data.htm](https://www.tutorialspoint.com/seaborn/seaborn_plotting_categorical_data.htm)*

---

---
[Previous](/seaborn/seaborn_visualizing_pairwise_relationship.htm)[Quiz](/seaborn/quiz_on_seaborn_plotting_categorical_data.htm)[Next](/seaborn/seaborn_distribution_of_observations.htm)
In our previous chapters we learnt about scatter plots, hexbin plots and kde plots which are used to analyze the continuous variables under study. These plots are not suitable when the variable under study is categorical.

When one or both the variables under study are categorical, we use plots like striplot(), swarmplot(), etc,. Seaborn provides interface to do so.

## Categorical Scatter Plots

In this section, we will learn about categorical scatter plots.

## Example - Using stripplot()

stripplot() is used when one of the variable under study is categorical. It represents the data in sorted order along any one of the axis.

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.stripplot(x = "species", y = "petal_length", data = df)
plt.show()
```

#### Output
![Box](/seaborn/images/box.jpg)
In the above plot, we can clearly see the difference of
**petal_length**in each species. But, the major problem with the above scatter plot is that the points on the scatter plot are overlapped. We use the Jitter parameter to handle this kind of scenario.
## Example - Adding Jitter

Jitter adds some random noise to the data. This parameter will adjust the positions along the categorical axis.

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.stripplot(x = "species", y = "petal_length", data = df, jitter = Ture)
plt.show()
```

#### Output
![Dots](/seaborn/images/dots.jpg)
Now, the distribution of points can be seen easily.

## Example - Using Swarmplot()

Another option which can be used as an alternate to Jitter is function
**swarmplot()**. This function positions each point of scatter plot on the categorical axis and thereby avoids overlapping points −
### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.swarmplot(x = "species", y = "petal_length", data = df)
plt.show()
```

#### Output
![Shades](/seaborn/images/shades.jpg)

---

## 10. Seaborn - Distribution of Observations

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_distribution_of_observations.htm](https://www.tutorialspoint.com/seaborn/seaborn_distribution_of_observations.htm)*

---

---

## 11. Seaborn - Statistical Estimation

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_statistical_estimation.htm](https://www.tutorialspoint.com/seaborn/seaborn_statistical_estimation.htm)*

---

---
[Previous](/seaborn/seaborn_distribution_of_observations.htm)[Quiz](/seaborn/quiz_on_seaborn_statistical_estimation.htm)[Next](/seaborn/seaborn_plotting_wide_form_data.htm)
In most of the situations, we deal with estimations of the whole distribution of the data. But when it comes to central tendency estimation, we need a specific way to summarize the distribution. Mean and median are the very often used techniques to estimate the central tendency of the distribution.

In all the plots that we learnt in the above section, we made the visualization of the whole distribution. Now, let us discuss regarding the plots with which we can estimate the central tendency of the distribution.

## Bar Plot

The
**barplot()**shows the relation between a categorical variable and a continuous variable. The data is represented in rectangular bars where the length the bar represents the proportion of the data in that category.
Bar plot represents the estimate of central tendency. Let us use the titanic dataset to learn bar plots.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('titanic')
sb.barplot(x = "sex", y = "survived", hue = "class", data = df)
plt.show()
```

#### Output
![barplot](/seaborn/images/barplot.jpg)
In the above example, we can see that the average number of survivals of male and female in each class. From the plot we can understand that more number of females survived than males. In both males and females more number of survivals are from first class.

A special case in barplot is to show the no of observations in each category rather than computing a statistic for a second variable. For this, we use
**countplot().**
### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('titanic')
sb.countplot(x = " class ", data = df, palette = "Blues");
plt.show()
```

#### Output
![Bargraph](/seaborn/images/bargraph.jpg)
Plot says that, the number of passengers in the third class are higher than first and second class.

## Point Plots

Point plots serve same as bar plots but in a different style. Rather than the full bar, the value of the estimate is represented by the point at a certain height on the other axis.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('titanic')
sb.pointplot(x = "sex", y = "survived", hue = "class", data = df)
plt.show()
```

#### Output
![zSymbol](/seaborn/images/z.jpg)

---

## 12. Seaborn - Plotting Wide Form Data

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_plotting_wide_form_data.htm](https://www.tutorialspoint.com/seaborn/seaborn_plotting_wide_form_data.htm)*

---

---
[Previous](/seaborn/seaborn_statistical_estimation.htm)[Quiz](/seaborn/quiz_on_seaborn_plotting_wide_form_data.htm)[Next](/seaborn/seaborn_multi_panel_categorical_plots.htm)
It is always preferable to use long-from or tidy datasets. But at times when we are left with no option rather than to use a wide-form dataset, same functions can also be applied to wide-form data in a variety of formats, including Pandas Data Frames or two-dimensional NumPy arrays. These objects should be passed directly to the data parameter the x and y variables must be specified as strings

## Example - Plotting a Wide Form Data

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.boxplot(data = df, orient = "h")
plt.show()
```

#### Output
![Models](/seaborn/images/models.jpg)
## Example - Using Vectors

Additionally, these functions accept vectors of Pandas or NumPy objects rather than variables in a DataFrame.

### main.py

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.boxplot(data = df, orient = "h")
plt.show()
```

#### Output
![Designs](/seaborn/images/designs.jpg)
The major advantage of using Seaborn for many developers in Python world is because it can take pandas DataFrame object as parameter.

---

## 13. Seaborn - Multipanel Categorical Plots

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_multi_panel_categorical_plots.htm](https://www.tutorialspoint.com/seaborn/seaborn_multi_panel_categorical_plots.htm)*

---

---
[Previous](/seaborn/seaborn_plotting_wide_form_data.htm)[Quiz](/seaborn/quiz_on_seaborn_multi_panel_categorical_plots.htm)[Next](/seaborn/seaborn_linear_relationships.htm)
Categorical data can we visualized using two plots, you can either use the functions
**pointplot()**, or the higher-level function**factorplot()**.
## Factorplot

Factorplot draws a categorical plot on a FacetGrid. Using kind parameter we can choose the plot like boxplot, violinplot, barplot and stripplot. FacetGrid uses pointplot by default.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('exercise')
sb.factorplot(x = "time", y = pulse", hue = "kind",data = df);
plt.show()
```

#### Output
![Lshape](/seaborn/images/l.jpg)
We can use different plot to visualize the same data using the
**kind**parameter.
### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('exercise')
sb.factorplot(x = "time", y = "pulse", hue = "kind", kind = 'violin',data = df);
plt.show()
```

#### Output
![Sharp](/seaborn/images/sharp.jpg)
In factorplot, the data is plotted on a facet grid.

## What is Facet Grid?
**Facet grid**forms a matrix of panels defined by row and column by dividing the variables. Due of panels, a single plot looks like multiple plots. It is very helpful to analyze all combinations in two discrete variables.
Let us visualize the above the definition with an example

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('exercise')
sb.factorplot(x = "time", y = "pulse", hue = "kind", kind = 'violin', col = "diet", data = df);
plt.show()
```

#### Output
![Two Types](/seaborn/images/two_types.jpg)
The advantage of using Facet is, we can input another variable into the plot. The above plot is divided into two plots based on a third variable called diet using the col parameter.

We can make many column facets and align them with the rows of the grid −

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('titanic')
sb.factorplot("alive", col = "deck", col_wrap = 3,data = df[df.deck.notnull()],kind = "count")
plt.show()
```

#### output
![various types](/seaborn/images/varioustypes.jpg)

---

## 14. Seaborn - Linear Relationships

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_linear_relationships.htm](https://www.tutorialspoint.com/seaborn/seaborn_linear_relationships.htm)*

---

---
[Previous](/seaborn/seaborn_multi_panel_categorical_plots.htm)[Quiz](/seaborn/quiz_on_seaborn_linear_relationships.htm)[Next](/seaborn/seaborn_facet_grid.htm)
Most of the times, we use datasets that contain multiple quantitative variables, and the goal of an analysis is often to relate those variables to each other. This can be done through the regression lines.

While building the regression models, we often check for
**multicollinearity,**where we had to see the correlation between all the combinations of continuous variables and will take necessary action to remove multicollinearity if exists. In such cases, the following techniques helps.
## Functions to Draw Linear Regression Models

There are two main functions in Seaborn to visualize a linear relationship determined through regression. These functions are
**regplot()**and**lmplot()**.
### regplot vs lmplot
regplotlmplotaccepts the x and y variables in a variety of formats including simple numpy arrays, pandas Series objects, or as references to variables in a pandas DataFramehas data as a required parameter and the x and y variables must be specified as strings. This data format is called long-form data
Let us now draw the plots.

### Example

Plotting the regplot and then lmplot with the same data in this example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('tips')
sb.regplot(x = "total_bill", y = "tip", data = df)
sb.lmplot(x = "total_bill", y = "tip", data = df)
plt.show()
```

#### Output

You can see the difference in the size between two plots.
![Zoomed and Magnifier](/seaborn/images/zoomed_magnifier.jpg)
We can also fit a linear regression when one of the variables takes discrete values

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('tips')
sb.lmplot(x = "size", y = "tip", data = df)
plt.show()
```

#### Output
![Rugged](/seaborn/images/rugged.jpg)
## Fitting Different Kinds of Models

The simple linear regression model used above is very simple to fit, but in most of the cases, the data is non-linear and the above methods cannot generalize the regression line.

Let us use Anscombes dataset with the regression plots −

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('anscombe')
sb.lmplot(x="x", y="y", data=df.query("dataset == 'I'"))
plt.show()
```

#### Output
![Dotted Graph](/seaborn/images/dotted.jpg)
In this case, the data is good fit for linear regression model with less variance.

Let us see another example where the data takes high deviation which shows the line of best fit is not good.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('anscombe')
sb.lmplot(x = "x", y = "y", data = df.query("dataset == 'II'"))
plt.show()
```

#### Output
![Half](/seaborn/images/halfs.jpg)
The plot shows the high deviation of data points from the regression line. Such non-linear, higher order can be visualized using the
**lmplot()**and**regplot()**.These can fit a polynomial regression model to explore simple kinds of nonlinear trends in the dataset −
### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('anscombe')
sb.lmplot(x = "x", y = "y", data = df.query("dataset == 'II'"),order = 2)
plt.show()
```

#### Output
![parabola](/seaborn/images/parabola.jpg)

---

## 15. Seaborn - Facet Grid

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_facet_grid.htm](https://www.tutorialspoint.com/seaborn/seaborn_facet_grid.htm)*

---

---
[Previous](/seaborn/seaborn_linear_relationships.htm)[Quiz](/seaborn/quiz_on_seaborn_facet_grid.htm)[Next](/seaborn/seaborn_pair_grid.htm)
A useful approach to explore medium-dimensional data, is by drawing multiple instances of the same plot on different subsets of your dataset.

This technique is commonly called as lattice, or trellis plotting, and it is related to the idea of small multiples.

To use these features, your data has to be in a Pandas DataFrame.

## Plotting Small Multiples of Data Subsets

In the previous chapter, we have seen the FacetGrid example where FacetGrid class helps in visualizing distribution of one variable as well as the relationship between multiple variables separately within subsets of your dataset using multiple panels.

A FacetGrid can be drawn with up to three dimensions − row, col, and hue. The first two have obvious correspondence with the resulting array of axes; think of the hue variable as a third dimension along a depth axis, where different levels are plotted with different colors.
**FacetGrid**object takes a dataframe as input and the names of the variables that will form the row, column, or hue dimensions of the grid.
The variables should be categorical and the data at each level of the variable will be used for a facet along that axis.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('tips')
g = sb.FacetGrid(df, col = "time")
plt.show()
```

#### Output
![Graphs](/seaborn/images/graphs.jpg)
In the above example, we have just initialized the
**facetgrid**object which doesnt draw anything on them.
The main approach for visualizing data on this grid is with the
**FacetGrid.map()**method. Let us look at the distribution of tips in each of these subsets, using a histogram.
### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('tips')
g = sb.FacetGrid(df, col = "time")
g.map(plt.hist, "tip")
plt.show()
```

#### Output
![differentiation](/seaborn/images/differentiation.jpg)
The number of plots is more than one because of the parameter col. We discussed about col parameter in our previous chapters.

To make a relational plot, pass the multiple variable names.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('tips')
g = sb.FacetGrid(df, col = "sex", hue = "smoker")
g.map(plt.scatter, "total_bill", "tip")
plt.show()
```

#### Output
![Ratio](/seaborn/images/ratio.jpg)

---

## 16. Seaborn - Pair Grid

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_pair_grid.htm](https://www.tutorialspoint.com/seaborn/seaborn_pair_grid.htm)*

---

---
[Previous](/seaborn/seaborn_facet_grid.htm)[Quiz](/seaborn/quiz_on_seaborn_pair_grid.htm)[Next](/seaborn/seaborn_function_reference.htm)
PairGrid allows us to draw a grid of subplots using the same plot type to visualize data.

Unlike FacetGrid, it uses different pair of variable for each subplot. It forms a matrix of sub-plots. It is also sometimes called as scatterplot matrix.

The usage of pairgrid is similar to facetgrid. First initialise the grid and then pass the plotting function.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
g = sb.PairGrid(df)
g.map(plt.scatter);
plt.show()
```

#### Output
![Various models](/seaborn/images/variousmodels.jpg)
It is also possible to plot a different function on the diagonal to show the univariate distribution of the variable in each column.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
g = sb.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);
plt.show()
```

#### Output
![Histogram Dots](/seaborn/images/histogram_dots.jpg)
We can customize the color of these plots using another categorical variable. For example, the iris dataset has four measurements for each of three different species of iris flowers so you can see how they differ.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
g = sb.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);
plt.show()
```

#### Output
![Colored ](/seaborn/images/colored.jpg)
We can use a different function in the upper and lower triangles to see different aspects of the relationship.

### Example

```
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
g = sb.PairGrid(df)
g.map_upper(plt.scatter)
g.map_lower(sb.kdeplot, cmap = "Blues_d")
g.map_diag(sb.kdeplot, lw = 3, legend = False);
plt.show()
```

#### Output
![various plots](/seaborn/images/various_plots.jpg)

---

## 17. Seaborn - Function Reference

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_function_reference.htm](https://www.tutorialspoint.com/seaborn/seaborn_function_reference.htm)*

---

---

## 18. Seaborn - Relational Plots

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_relational_plots_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_relational_plots_introduction.htm)*

---

---
[Previous](/seaborn/seaborn_function_reference.htm)[Quiz](/seaborn/quiz_on_seaborn_relational_plots_introduction.htm)[Next](/seaborn/seaborn_distribution_plot_introduction.htm)
The statistical link between the data points is visualized using relational graphs. Because visualization helps humans to detect trends and patterns in data, it is vital. Statistical analysis is the process of determining how the variables in a dataset relate to one another and their relationships.

Relational plots like the name suggests shows how two variables of a dataset are associated with each other.

Seaborn provides a few methods to make plotting relational plots easier. In the coming articles we will look at all the functions that will enable relational plotting.

## Functions

The list of all the Seaborn relational functions is given below.
S.NoMethod and Description1[relplot()](/seaborn/seaborn_relplot_method.htm)**relplot()**
This method allows the user to plot association between variables and mapping between different semantic groups.
2[lineplot()](/seaborn/seaborn_lineplot_method.htm)**lineplot()**
This method allows the user to draw a line plot with possibility of several semantic groupings.
3[scatterplot()](/seaborn/seaborn_scatterplot_method.htm)**scatterplot()**
This method allows the user to draw a scatter plot with possibility of several semantic groupings.

Before moving on, we require data to plot graphs and in case, data is not available readily for your use, you can use the datasets present in the seaborn library.

Seaborn contains various default datasets in addition to being a statistical charting toolkit. We'll use the one of the in-built datasets as an example of a default dataset.

Let us consider the tips dataset in the first example. The 'tips' dataset comprises information about people who have likely eaten at a restaurant and whether or not they left a tip for the servers, as well as their gender, smoking status, and other factors.

The
**get_dataset_names()**method helps to retrieve all the names of the in-built datasets.
```
seaborn.get_dataset_names()
```

The
**load_dataset()**method helps to load the dataset with the name into a data structure.
```
Tips=seaborn.load_dataset('tips')
```

The above line of code helps to load the dataset with the name 'tips' into a data structure called tips.

Now that we have data to plot, let us understand how to plot relational graphs using Seaborn library.

---

## 19. Seaborn - Distribution Plots

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_distribution_plot_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_distribution_plot_introduction.htm)*

---

---

## 20. Seaborn - Categorical Plots

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_categorial_plots_Introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_categorial_plots_Introduction.htm)*

---

---
[Previous](/seaborn/seaborn_distribution_plot_introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_categorial_plots_Introduction.htm)[Next](/seaborn/seaborn_regression_plots_introduction.htm)
Plots are mostly used to depict the relationship between two or more variables. Those variables can be entirely numerical or represent a category such as a group, class, or division. This article discusses categorical variables and how they may be visualized with Python's Seaborn package.

Seaborn contains various default datasets in addition to being a statistical charting toolkit. We'll use the one of the in-built datasets as an example of a default dataset.

Let us consider the tips dataset in the first example. The 'tips' dataset comprises information about people who have likely eaten at a restaurant and whether or not they left a tip for the servers, as well as their gender, smoking status, and other factors.

Seaborn.get_dataset_names() method helps to retrieve all the names of the in-built datasets.

```
seaborn.get_dataset_names()
```

The
**load_dataset()**method helps to load the dataset with the name into a data structure.
```
Tips=seaborn.load_dataset('tips')
```

The above line of code helps to load the dataset with the name 'tips' into a data structure called tips.

There are different kinds of categorical plots such as distribution, estimate and scatter plots. Each of these are categories contain a few plots each.
S.NoTypePlot Names1**Categorical Scatter Plots**
- [Cat plot](/seaborn/seaborn_catplot_method.htm)**Cat plot**
- [Strip plot](/seaborn/seaborn_stripplot_method.htm)**Strip plot**
- [Swarm plot](/seaborn/seaborn_swarmplot_method.htm)**Swarm plot**2**Categorical Distribution Plots**
- [Box plot](/seaborn/seaborn_boxplot_method.htm)**Box plot**
- [Violin plot](/seaborn/seaborn_violinplot_method.htm)**Violin plot**
- [Boxen plot](/seaborn/seaborn_boxenplot_method.htm)**Boxen plot**3**Categorical Estimate Plots**
- [Bar plot](/seaborn/seaborn_barplot_method.htm)**Bar plot**
- [Point plot](/seaborn/seaborn_pointplot_method.htm)**Point plot**
- [Count plot](/seaborn/seaborn_countplot_method.htm)**Count plot**

---

## 21. Seaborn - Regression Plots

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_regression_plots_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_regression_plots_introduction.htm)*

---

---
[Previous](/seaborn/seaborn_categorial_plots_Introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_regression_plots_introduction.htm)[Next](/seaborn/seaborn_matrix_plots_introduction.htm)
A statistical charting tool called Seaborn offers straightforward techniques for adding regression lines to your scatter diagrams and can read Pandas dataframes as well as other types of data formats. An excellent tool for examining links or trends in data is a scatter plot. But by including a regression line, those patterns can become more obvious.

In an effort to discover the scatter plot's points that fit together the best, the regression line is used.

Drawing regression plots with small data. We begin by importing the seaborn library

```
import seaborn as sns
```

We can draw a scatterplot for any dataset by using the below code snippet. Considering the following case, we are using the in-built titanic dataset in seaborn library and x,y parameters are passed to the scatterplot() method and a graph is obtained.

```
import seaborn as sns
import matplotlib.pyplot as plt
dts=sns.load_dataset("titanic")
dts.head()
sns.scatterplot(data=dts,x="age",y="fare",hue="sex")
plt.show()
```

#### Output

The output is obtained as follows −
![Regression plots](/seaborn/images/basic_scatterplot.jpg)
On observing the above plot, you will be able to understand the linear relationship but drawing a regression line makes it easier.

```
import seaborn as sns
import matplotlib.pyplot as plt
titanic=sns.load_dataset("titanic")
titanic.head()
sns.regplot(x="fare", y="age",color="g", data=titanic)
plt.show()
```

#### Output

The output is obtained as follows −
![Seaborn Regression plots](/seaborn/images/regplot_method.jpg)
The above figure shows the regression line drawn over the plot.

Since we have understood how regression lines help us, now we will see how to add regression lines to your plots. This can be done with the help of the regression plot methods in seaborn.

There are three different types of regression plots in seaborn.
S.NoMethod & Description1[The regplot() method](/seaborn/seaborn_regplot_method.htm)**The regplot() method**
seaborn.regplot() method is used to plot data and draw a linear regression model fit.
2[The lmplot() method](/seaborn/seaborn_implot_method.htm)**The lmplot() method**
seaborn.lmplot() method is used to plot data and draw regression model fits across grids where multiple plots can be plotted.
3[The residplot() method](/seaborn/seaborn_residplot_method.htm)**The residplot() method**
seaborn.residplot() method is used to plot residual data of a linear regression.

---

## 22. Seaborn Matrix Plots - Introduction

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_matrix_plots_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_matrix_plots_introduction.htm)*

---

---
[Previous](/seaborn/seaborn_regression_plots_introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_matrix_plots_introduction.htm)[Next](/seaborn/seaborn_multi_plot_grids_introduction.htm)
A plot of matrix data is called a matrix plot. A matrix plot is a color-coded figure with values, data in the rows, and data in the columns. You can create matrix plots in seaborn by using either heatmap() or clustermap() functions.

Heatmap() is used to produce rectangular data as a color-coded matrix and Clustermap() is used to plot a dataset as a hierarchically clustered heat map.

To plot any data into a map, we need to import the data. You can either use datasets available in the seaborn library or import them as per your choice from elsewhere.

## In-built datasets in Seaborn library

Seaborn contains various default datasets in addition to being a statistical charting toolkit. We'll use the one of the in-built datasets as an example of a default dataset.

Let us consider the tips dataset in the first example. The 'tips' dataset comprises information about people who have likely eaten at a restaurant and whether or not they left a tip for the servers, as well as their gender, smoking status, and other factors.

The
**Seaborn.get_dataset_names()**method helps to retrieve all the names of the in-built datasets.
```
seaborn.get_dataset_names() 
>> ['anagrams', 'anscombe', 'attention', 'brain_networks', 'car_crashes', 'diamonds', 'dots', 'exercise', 'flights', 'fmri', 'gammas', 'geyser', 'iris', 'mpg', 'penguins', 'planets', 'taxis', 'tips', 'titanic']
```

From the above list, we will for now, use the tips dataset. To load the dataset into a data frame for use, the following method can be used.
**load_dataset()**method helps to load the dataset with the name into a data structure.
```
Tips=seaborn.load_dataset('tips')
```

The above line of code helps to load the dataset with the name 'tips' into a data structure called tips.

Now that we understand how to load an existing dataset, we will move on to understanding the working of the matrix plots.
S.NoMethod and Description1[Heatmap()](/seaborn/seaborn_heatmap_method.htm)**Heatmap()**
Produces rectangular data as a color-coded matrix.
2[Clustermap()](/seaborn/seaborn_clustermap_method.htm)**Clustermap()**
Plots a dataset as a hierarchically clustered heat map.

The matrix plots, heatmap() and clustermap() are discussed in the coming articles.

---

## 23. Seaborn Multi plot grids - Introduction

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_multi_plot_grids_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_multi_plot_grids_introduction.htm)*

---

---
[Previous](/seaborn/seaborn_matrix_plots_introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_multi_plot_grids_introduction.htm)[Next](/seaborn/seaborn_themeing_introduction.htm)
We'll look at multi-dimensional plot data in this post. Drawing the same plot numerous times on various dataset subsets is a valuable strategy. It enables a viewer to swiftly extract a significant amount of data from a complicated dataset. We will plot numerous graphs in Seaborn in two different ways. With the first using the Facetgrid() method and the second implicitly using Matplotlib.

There are different multi-plot grids available in seaborn and they are listed below.
S.NoName and Description1[FacetGrid()](/seaborn/seaborn_facetgrid_method.htm)**FacetGrid()**
The FacetGrid class is useful when you want to visualize the distribution of a variable or the relationship between multiple variables separately within subsets of your dataset.
2[Pairplot()](/seaborn/seaborn_pairplot_method.htm)**Pairplot()**
Is used to plotpairwise relationships in a dataset.
3[PairGrid()](/seaborn/seaborn_pairgrid_class.htm)**PairGrid()**
Is used to subplot grid for plotting pairwise relationships in a dataset.
4[Jointplot()](/seaborn/seaborn_jointplot_method.htm)**Jointplot()**
Is used to draw a plot of two variables with bivariate and univariate graphs.
5[JointGrid()](/seaborn/seaborn_jointgrid_class.htm)**JointGrid()**
Is used as a Grid for drawing a bivariate plot with marginal univariate plots

Before moving on to understanding the working of these plots, we will understand how to load in-built datasets from the seaborn library since we will be using these datasets to learn about these functions.

Seaborn contains various default datasets in addition to being a statistical charting toolkit. We'll use the one of the in-built datasets as an example of a default dataset.

Let us consider the tips dataset in the first example. The 'tips' dataset comprises information about people who have likely eaten at a restaurant and whether or not they left a tip for the servers, as well as their gender, smoking status, and other factors.

The
**Seaborn.get_dataset_names()**method helps to retrieve all the names of the in-built datasets.
```
seaborn.get_dataset_names()
```

load_dataset() method helps to load the dataset with the name into a data structure.

```
Tips=seaborn.load_dataset('tips')
```

The above line of code helps to load the dataset with the name 'tips' into a data structure called tips.

Now that we know where to load datasets from, we can move onto understanding the working of multi-plot grids in seaborn.

---

## 24. Seaborn Themeing - Introduction

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_themeing_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_themeing_introduction.htm)*

---

---
[Previous](/seaborn/seaborn_multi_plot_grids_introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_themeing_introduction.htm)[Next](/seaborn/seaborn_color_palettes_introduction.htm)
The main aim while producing a data visualization is to convey the data's insights. While styling affects how your audience understands what you're trying to say, visualizing conveys crucial information.

The final step of data visualization is styling, which comes after formatting and visualizing your data. Customizing the overall appearance of your visualization or figure is the process of styling. Making deliberate choices will enhance the impact of the visualization and distinguish your work. Using backdrop colors, grids, spines, and ticks, we'll look at how to change the overall appearance of your figure in the following articles.

There are many theming methods available in Seaborn. They are named below.
S.NoMethod and Description1[set_theme()](/seaborn/seaborn_set_theme_method.htm)**set_theme()**
Sets the visual theme components for all seaborn and matplotlib plots.
2[axes_styles()](/seaborn/seaborn_axes_style_method.htm)**axes_styles()**
Obtain the settings that govern the plots' overall look.
3[set_style()](/seaborn/seaborn_set_style_method.htm)**set_style()**
Sets the parameters that control the general style of the plots.
4[plotting_context()](/seaborn/seaborn_plotting_context_method.htm)**plotting_context()**
Gets the parameters that control the scaling of plot elements.
5[set_context()](/seaborn/seaborn_set_context_method.htm)**set_context()**
Sets the parameters that control the scaling of plot elements.
6[set_color_codes()](/seaborn/seaborn_set_color_codes_method.htm)**set_color_codes()**
Changes how matplotlib color shorthands are interpreted.
7**reset_default()**
Restores all RC params to default settings.
8**reset_orig()**
Restores all RC params to original settings
9**set()**
It is another name for set_theme() which is the preferred interface.

---

## 25. Seaborn Color Palettes - Introduction

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_color_palettes_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_color_palettes_introduction.htm)*

---

---
[Previous](/seaborn/seaborn_themeing_introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_color_palettes_introduction.htm)[Next](/seaborn/seaborn_palette_widgets_introduction.htm)
Color palettes in seaborn makes visualizing data a lot more interesting. The main aim while producing a data visualization is to convey the data's insights. While styling affects how your audience understands what you're trying to say, visualizing conveys crucial information.

The final step of data visualization is styling, which comes after formatting and visualizing your data. Customizing the overall appearance of your visualization or figure is the process of styling. Making deliberate choices will enhance the impact of the visualization and distinguish your work. Using backdrop colors, grids, spines, and ticks, we'll look at how to change the overall appearance of your figure in the following articles. We may generate the point with various colors using the palette.
In the following chapters we will discuss a few of the color palette methods that Seaborn offers.
## Functions

The list of all the Seaborn functions regarding
is given below −**Color Palettes**S.NoMethod and Description1[set_palette()](/seaborn/seaborn_set_palette_method.htm)**set_palette()**
This method sets the palette for plots.
2[color_palette()](/seaborn/seaborn_color_palette_method.htm)**color_palette()**
This method returns a palette defined by a list of colors or a continuous colormap.
3[husl_palette()](/seaborn/seaborn_husl_palette_method.htm)**husl_palette()**
This method retrieves a set of evenly spaced colors that are a part of the HUSL color space.
4[hls_palette()](/seaborn/seaborn_hls_palette_method.htm)**hls_palette()**
This method retrieves set of evenly spaced colors that are a part of the HSL color space.
5[cubehelix_palette()](/seaborn/seaborn_cubehelix_palette_method.htm)**cubehelix_palette()**
This method creates a sequential palette from cubehelix system.
6[dark_palette()](/seaborn/seaborn_dark_palette_method.htm)**dark_palette()**
This method creates a sequential palette that blends from dark colors to lighter colors.
7[light_palette()](/seaborn/seaborn_light_palette_method.htm)**light_palette()**
This method creates a sequential palette that blends from light to color.
8[diverging_palette()](/seaborn/seaborn_diverging_palette_method.htm)**diverging_palette()**
This method creates a diverging plot between two HUSL colors.
9[blend_palette()](/seaborn/seaborn_blend_palette_method.htm)**blend_palette()**
This method is used to blend between a list of colors.
10[xkcd_palette()](/seaborn/seaborn_xkcd_palette_method.htm)**xkcd_palette()**
This method is used to create a palette with color names form the xkcd color survey.
11[crayon_palette()](/seaborn/seaborn_crayon_palette_method.htm)**crayon_palette()**
This method is used to create a palette with color names from Crayola crayons.
12[mpl_palette()](/seaborn/seaborn_mpl_palette_method.htm)**mpl_palette()**
This method returns a palette's discrete color values.

---

## 26. Seaborn Palette widgets - Introduction

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_palette_widgets_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_palette_widgets_introduction.htm)*

---

---
[Previous](/seaborn/seaborn_color_palettes_introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_palette_widgets_introduction.htm)[Next](/seaborn/seaborn_utility_functions_introduction.htm)
Palette widgets can be categorized into the various three kinds. Namely, qualitative, sequential and diverging. In the coming chapters we will understand and explore all these in great detail.

## Qualitative palettes

When a variable is categorical in nature, a qualitative palette is employed, and the color assigned to each group must be distinct. One color from a qualitative palette is assigned to each potential value of the variable. To understand an example of this, the following graph can be used.

```
sns.barplot(x=["A", "B", "C"], y=[10, 30, 23])
```

The following output is produced.
![Palette_widgets](/seaborn/images/Palette_widgets.jpg)
## Sequential Palettes

Colors in successive palettes changed gradually from lighter to darker. A sequential palette can be used to represent a variable when it has numerical values or intrinsically ordered values, as seen in the figure.
![Sequential_Palettes](/seaborn/images/Sequential_Palettes.jpg)
## Diverging palettes

Diverging palettes are the ideal for visualizing mixed value tasks, such as +ve and -ve (low and high values).

## Functions

The list of all the Seaborn functions regarding
is given below −**Palette widgets**S.NoMethod and Description1[choose_colorbrewer_palette()](/seaborn/seaborn_choose_colorbrewer_palette.htm)**choose_colorbrewer_palette()**
This method enables the user to choose colors present in the given options.
2[choose_cubehelix_palette() method](/seaborn/seaborn_choose_cubehelix_palette_method.htm)**choose_cubehelix_palette() method**
This method enable widgets to choose colors in the cubehelix palette.
3[choose_light_palette() method](/seaborn/seaborn_choose_light_palette_method.htm)**choose_light_palette() method**
This method is used to create a light sequential palette as an interactive widget.
4[choose_dark_palette() method](/seaborn/seaborn_choose_dark_palette_method.htm)**choose_dark_palette() method**
This method is used to create a dark sequential palette as an interactive widget.
5[choose_diverging_palette() method](/seaborn/seaborn_choose_diverging_palette_method.htm)**choose_diverging_palette() method**
This method launches a widget that enables easy access to the user in choosing a diverging palette.

---

## 27. Seaborn Utility Functions - Introduction

*Source: [https://www.tutorialspoint.com/seaborn/seaborn_utility_functions_introduction.htm](https://www.tutorialspoint.com/seaborn/seaborn_utility_functions_introduction.htm)*

---

---
[Previous](/seaborn/seaborn_palette_widgets_introduction.htm)[Quiz](/seaborn/quiz_on_seaborn_utility_functions_introduction.htm)[Next](/seaborn/seaborn_quick_guide.htm)
There is no clear definition for the word "utility." If a piece of code appears too small to be deemed a standalone application and too general-purpose to be considered a component of a specific program, it may be referred to as a utility. For instance, a database program would not constitute a utility, but a function that only operated on a list once might. The majority of utilities are similar to the existing functions and macros in Seaborn.

In the following chapters we will look at the different utility functions and understand how they work. We have been working with some of these functions already in the previous chapters but now, we will understand only the working of these and not as a part of other topics.

There are many utility functions and in nay library they play a crucial role. In Seaborn we have the following utility functions.
S.NoMethod and Description1[Despine](/seaborn/seaborn_despine_method.htm)**Despine**
This method helps to removes the top and right spines of the plot.
2[Move_legend](/seaborn/seaborn_move_legend_method.htm)**Move_legend**
This method helps to move the plots legend to a new location.
3[Saturate](/seaborn/seaborn_saturate_method.htm)**Saturate**
This method helps to return a fully saturated color with the same hue.
4[Desaturate](/seaborn/seaborn_desaturate_method.htm)**Desaturate**
This method helps to decrease the saturation channel of a color by a particular percent.
5[Set_hls_values](/seaborn/seaborn_set_hls_values_method.htm)**Set_hls_values**
This method helps to manipulate the H, L and S colors individually.
6[Load_dataset](/seaborn/seaborn_load_dataset_method.htm)**Load_dataset**
This method helps to load inbuilt seaborn datasets to work with them.
7[Get_dataset_names](/seaborn/seaborn_get_dataset_names_method.htm)**Get_dataset_names**
This method helps to view all the dataset names present by default in the seaborn library.
8[Get_data_home](/seaborn/seaborn_get_data_home_method.htm)**Get_data_home**
This method helps to retrieve a path to the home directory for exemplar directory.

---


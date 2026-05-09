# Scipy

## Table of Contents

1. [SciPy - Introduction](#scipy---introduction)
2. [SciPy - Environment Setup](#scipy---environment-setup)
3. [SciPy - Basic Functionality](#scipy---basic-functionality)
4. [SciPy - Relationship with NumPy](#scipy---relationship-with-numpy)
5. [SciPy - Cluster](#scipy---cluster)
6. [SciPy - Hiearchical Clustering](#scipy---hiearchical-clustering)
7. [SciPy - Hiearchical Clustering](#scipy---hiearchical-clustering)
8. [SciPy - Distance Metrics](#scipy---distance-metrics)
9. [SciPy - Constants](#scipy---constants)
10. [SciPy - Mathematical Constants](#scipy---mathematical-constants)
11. [SciPy Physical Constants](#scipy-physical-constants)
12. [SciPy - Unit Conversion Constants](#scipy---unit-conversion-constants)
13. [SciPy - Astronomical Constants](#scipy---astronomical-constants)
14. [SciPy - FFT Pack](#scipy---fft-pack)
15. [SciPy - Discrete Fourier Transform](#scipy---discrete-fourier-transform)
16. [SciPy - Fast Fourier Transform](#scipy---fast-fourier-transform)
17. [SciPy - Integrate](#scipy---integrate)
18. [SciPy - Single Integration](#scipy---single-integration)
19. [SciPy - Double Integration](#scipy---double-integration)
20. [SciPy - Triple Integration](#scipy---triple-integration)
21. [SciPy - Multiple Integration](#scipy---multiple-integration)
22. [SciPy - Differential Equations](#scipy---differential-equations)
23. [SciPy - Integration of Stochastic Differential Equations](#scipy---integration-of-stochastic-differential-equations)
24. [SciPy - Integration of Ordinary Differential Equations](#scipy---integration-of-ordinary-differential-equations)
25. [SciPy - Discontinous Functions](#scipy---discontinous-functions)
26. [SciPy - Oscillatory Functions](#scipy---oscillatory-functions)
27. [SciPy - Partial Differential Equations](#scipy---partial-differential-equations)
28. [SciPy - Interpolate](#scipy---interpolate)
29. [SciPy - Linear 1-D Interpolation](#scipy---linear-1-d-interpolation)
30. [SciPy - Polynomial 1-D Interpolation](#scipy---polynomial-1-d-interpolation)
31. [SciPy - Spline 1-D Interpolation](#scipy---spline-1-d-interpolation)
32. [SciPy - Grid Data Multi-Dimensional Interpolation](#scipy---grid-data-multi-dimensional-interpolation)
33. [SciPy - Radial Basis Function(RBF) Multi-Dimensional Interpolation](#scipy---radial-basis-functionrbf-multi-dimensional-interpolation)
34. [SciPy - Curve Fitting](#scipy---curve-fitting)
35. [SciPy - Linear Curve Fitting](#scipy---linear-curve-fitting)
36. [SciPy - Non-Linear Curve Fitting](#scipy---non-linear-curve-fitting)
37. [SciPy - Input and Output](#scipy---input-and-output)
38. [SciPy - Reading and Writing Files](#scipy---reading-and-writing-files)
39. [SciPy - Working With Different File Formats](#scipy---working-with-different-file-formats)
40. [Scipy - Efficient Data Storage with HDF5](#scipy---efficient-data-storage-with-hdf5)
41. [SciPy - Data Serialization](#scipy---data-serialization)
42. [SciPy - linalg](#scipy---linalg)
43. [SciPy - Matrix Creation & Basic Operations](#scipy---matrix-creation--basic-operations)
44. [SciPy - Matrix LU Decomposition](#scipy---matrix-lu-decomposition)
45. [SciPy - Matrix QU Decomposition](#scipy---matrix-qu-decomposition)
46. [SciPy - Singular Value Decomposition](#scipy---singular-value-decomposition)
47. [SciPy - Cholesky Decomposition](#scipy---cholesky-decomposition)
48. [SciPy - Solving Linear Systems](#scipy---solving-linear-systems)
49. [SciPy - Eigenvalues and Eigenvectors](#scipy---eigenvalues-and-eigenvectors)
50. [SciPy - Ndimage](#scipy---ndimage)
51. [SciPy - Reading and Writing Images](#scipy---reading-and-writing-images)
52. [SciPy - Image Transformation](#scipy---image-transformation)
53. [SciPy - Filtering and Edge Detection](#scipy---filtering-and-edge-detection)
54. [SciPy - Top-Hat Filters](#scipy---top-hat-filters)
55. [SciPy - Morphological Filters](#scipy---morphological-filters)
56. [SciPy - Low Pass Filters](#scipy---low-pass-filters)
57. [SciPy - High Pass Filters](#scipy---high-pass-filters)
58. [SciPy - Bilateral Filter](#scipy---bilateral-filter)
59. [SciPy - Median Filter](#scipy---median-filter)
60. [SciPy - Non-Linear Filters in Image processing](#scipy---non-linear-filters-in-image-processing)
61. [SciPy - High Boost Filter](#scipy---high-boost-filter)
62. [SciPy - Laplacian Filter](#scipy---laplacian-filter)
63. [SciPy - Morphological Operations](#scipy---morphological-operations)
64. [SciPy Image Segmentation](#scipy-image-segmentation)
65. [SciPy - Thresholding in Image Segmentation](#scipy---thresholding-in-image-segmentation)
66. [SciPy - Region-based Segmentation](#scipy---region-based-segmentation)
67. [SciPy - Connected Component Labeling](#scipy---connected-component-labeling)
68. [SciPy - Optimize](#scipy---optimize)
69. [SciPy - Special Matrices and Functions](#scipy---special-matrices-and-functions)
70. [SciPy - Unconstrained Optimization](#scipy---unconstrained-optimization)
71. [SciPy - Constrained Optimization](#scipy---constrained-optimization)
72. [SciPy - Matrix Norms](#scipy---matrix-norms)
73. [SciPy - Sparse Matrix](#scipy---sparse-matrix)
74. [SciPy - Frobenius Norm(fro)](#scipy---frobenius-normfro)
75. [SciPy - L2 Norm(Spectral Norm)](#scipy---l2-normspectral-norm)
76. [SciPy - Condition Numbers](#scipy---condition-numbers)
77. [SciPy - Linear Least Squares](#scipy---linear-least-squares)
78. [SciPy - Non-Linear Least Squares](#scipy---non-linear-least-squares)
79. [SciPy - Finding Roots of Scalar Functions](#scipy---finding-roots-of-scalar-functions)
80. [SciPy - Finding Roots of Multivariate Functions](#scipy---finding-roots-of-multivariate-functions)
81. [SciPy - Signal Filtering and Smoothing](#scipy---signal-filtering-and-smoothing)
82. [SciPy - Short-Time Fourier Transform (STFT)](#scipy---short-time-fourier-transform-stft)
83. [SciPy - Discrete Wavelet Transform (DWT)](#scipy---discrete-wavelet-transform-dwt)
84. [SciPy - Continuous Wavelet Transform (CWT)](#scipy---continuous-wavelet-transform-cwt)
85. [SciPy - Wavelet Packet Transform](#scipy---wavelet-packet-transform)
86. [SciPy - Multi-Resolution Analysis (MRA)](#scipy---multi-resolution-analysis-mra)
87. [SciPy - Stationary Wavelet Transform](#scipy---stationary-wavelet-transform)
88. [SciPy - Stats](#scipy---stats)
89. [SciPy - Descriptive Statistics](#scipy---descriptive-statistics)
90. [SciPy - Continous Probability Distributions](#scipy---continous-probability-distributions)
91. [SciPy - Discrete Probability Distributions](#scipy---discrete-probability-distributions)
92. [SciPy - Statistical and Tests Inference](#scipy---statistical-and-tests-inference)
93. [SciPy - Generating Random Samples](#scipy---generating-random-samples)
94. [SciPy - Kaplan-Meier Estimator Survival Analysis](#scipy---kaplan-meier-estimator-survival-analysis)
95. [SciPy - Cox Proportional Hazards Model Survival Analysis](#scipy---cox-proportional-hazards-model-survival-analysis)
96. [SciPy - Spatial](#scipy---spatial)
97. [SciPy - Special Packages](#scipy---special-packages)
98. [SciPy - CSGraph](#scipy---csgraph)
99. [SciPy - ODR](#scipy---odr)
100. [SciPy - Reference](#scipy---reference)

---

## 1. SciPy - Introduction

*Source: [https://www.tutorialspoint.com/scipy/scipy_introduction.htm](https://www.tutorialspoint.com/scipy/scipy_introduction.htm)*

---

---
[Previous](/scipy/index.htm)[Quiz](/scipy/quiz_on_scipy_introduction.htm)[Next](/scipy/scipy_environment_setup.htm)**SciPy**is pronounced as**Sigh Pie**. It is an open-source Python library designed for scientific and technical computing. It builds on NumPy by providing advanced mathematical functions for optimization, integration, interpolation, linear algebra, statistics and signal processing.**SciPy**is organized into submodules such as**scipy.optimize**,**scipy.integrate**,**scipy.stats**etc which divides based on various scientific needs. It is widely used in academia and industry for tasks such as data analysis, engineering simulations and scientific research.
Its integration with other libraries such as NumPy, Matplotlib and pandas makes it a cornerstone of the Python scientific computing ecosystem.

## History & Development of SciPy
**SciPy**was created in 2001 by Travis Oliphant, Pearu Peterson and Eric Jones as part of an effort to enhance Python's capabilities for scientific computing. It evolved from earlier libraries such as Numeric, which eventually became NumPy by providing a more extensive suite of scientific functions.
SciPy's development was driven by the need for an open-source, easy-to-use library that could handle complex mathematical computations across various scientific domains.

## Core Functionality of SciPy

SciPy is a powerful library that extends the capabilities of NumPy by providing a wide range of functions and tools for scientific and technical computing.

The Scipy core functionality encompasses various domains by making it suitable for a diverse set of applications. Below are the key features and functionalities provided by SciPy −

- **Optimization**− SciPy offers several optimization algorithms such as linear programming, curve fitting and root finding.
- **Integration**− The Scipy library provides functions for numerical integration such as single, double and multiple integrals.
- **Interpolation**− SciPy supports various methods for interpolating data points such as linear, cubic and spline interpolation.
- **Linear Algebra**− Beyond basic matrix operations the SciPy library includes advanced linear algebra functions like matrix decomposition e.g., LU, QR, SVD and solving systems of linear equations.
- **Statistics**− SciPy offers an extensive collection of statistical functions including probability distributions, hypothesis testing and descriptive statistics.
- **Signal Processing**− The library provides tools for working with signals such as filtering, convolution, Fourier transforms and spectral analysis.
- **Special Functions**− SciPy includes numerous special functions such as Bessel functions, gamma functions and hyper-geometric functions, which are crucial in many scientific applications.
- **Image Processing**− The library includes basic image manipulation tools like filtering, morphology and object measurement.
## Modules in SciPy

Following is the list of modules in SciPy −
ModuleDescriptionKey Functions/Classes**scipy.optimize**Provides algorithms for function optimization, root finding and curve fitting.minimize, curve_fit, root, least_squares**scipy.integrate**This offers functions for numerical integration of functions and solving differential equations.quad, dblquad, solve_ivp, odeint**scipy.interpolate**Contains tools for interpolating data points in one, two and three dimensions.interp1d, interp2d, Rbf, UnivariateSpline**scipy.linalg**Extends NumPys linear algebra capabilities with more advanced matrix operations and decompositions.inv, det, eig, svd, lu, qr**scipy.stats**Provides a wide range of statistical functions, probability distributions and tests.norm, t-test, chi2_contingency, describe**scipy.fftpack**Contains functions for performing fast Fourier transforms (FFT) and related operations.fft, ifft, fftfreq, dct, dst**scipy.ndimage**Focuses on image processing and analysis in n-dimensional arrays.convolve, gaussian_filter, morphology, label**scipy.signal**Provides tools for signal processing, including filtering, spectral analysis and convolution.butter, convolve, spectrogram, welch**scipy.sparse**Handles sparse matrices, which are efficient for large matrices with many zeros.csr_matrix, csc_matrix, lil_matrix, dok_matrix**scipy.spatial**Offers functions for spatial data structures and algorithms, including nearest neighbors and distance computations.KDTree, Delaunay, distance_matrix, ConvexHull**scipy.special**Contains numerous special mathematical functions often used in scientific computations.gamma, bessel, erf, hypergeometric**scipy.constants**Provides a large collection of physical and mathematical constants.physical_constants, value, unit, precision**scipy.cluster**Includes functions for hierarchical and k-means clustering.linkage, fcluster, kmeans, dendrogram**scipy.io**Offers functions for reading and writing data in various formats such as MATLAB files.loadmat, savemat, mmread, mmwrite**scipy.odr**Orthogonal Distance Regression module for fitting models to data.ODR, Model, Data
## Usage and Applications of Scipy

SciPy is widely used in academia and industry for tasks ranging from basic numerical operations to complex scientific simulations. Some common applications of SciPy as mentioned below −

- **Data Analysis**− Researchers use SciPy to analyze and visualize data by applying statistical methods and signal processing techniques.
- **Engineering**− Engineers leverage SciPy for simulations, modeling and solving differential equations in mechanical, electrical and civil engineering.
- **Machine Learning**− While SciPy is not a machine learning library it is often used in conjunction with libraries like scikit-learn for pre-processing data and optimizing algorithms.
- **Physics and Chemistry**− SciPy's special functions and integration tools are frequently used in physics and chemistry for solving equations related to quantum mechanics, thermodynamics, and other fields.

---

## 2. SciPy - Environment Setup

*Source: [https://www.tutorialspoint.com/scipy/scipy_environment_setup.htm](https://www.tutorialspoint.com/scipy/scipy_environment_setup.htm)*

---

---
[Previous](/scipy/scipy_introduction.htm)[Quiz](/scipy/quiz_on_scipy_environment_setup.htm)[Next](/scipy/scipy_basic_functionality.htm)**SciPy Environment Setup**refers to the process of preparing our system to use the SciPy library, which is a Python-based ecosystem of open-source software for mathematics, science and engineering.
This setup involves installing Python i.e. if not already installed, along with SciPy and its dependencies, so that we can use the library for scientific computing tasks.

## Key Steps in SciPy Environment Setup

The following are the different steps involved in SciPy Environment setup −

### Install Python

Python is the programming language required to use SciPy library. we should need to ensure that Python is installed on our system.

If Python is not installed, it can be downloaded from the
[official Python website](https://www.python.org/downloads/). During installation its important to add Python to our system's PATH to ensure that we can run Python commands from the terminal or command prompt.
### Install SciPy

Once Python is installed we can install SciPy using a package manager like pip or conda in Anaconda.

### Verify the Installation

After installation we should verify that SciPy is installed correctly by importing it in a Python session or script and then have to check its version.

### Optional Tools

Depending on our needs we might also install additional libraries commonly used with SciPy such as Matplotlib for plotting, pandas for data manipulation and Jupyter Notebook for an interactive coding environment.

> Setting up the SciPy environment across different operating systems such as Windows, macOS and Linux generally involves similar steps, though there are some platform-specific considerations. Below is a guide to setting up SciPy on each of these operating systems.

## Installing SciPy on Windows

The below are the steps to setup the environment of SciPy in Windows Operating System −

### Installing python

First we have to make sure that python is installed in our PC with the help of below code executed in command prompt −

```
py --version
```

As Python is already installed in the PC, the version of python is as below −

```
Python 3.14.2
```

If Python is not installed in the PC then we have to download and install it from the
[official Python website](https://www.python.org/downloads/). Ensure we check the option to "Add Python to PATH" during installation.
### Installing SciPy

The Installation of SciPy can be done with the help of either using
**pip**or**conda**.**Installing using pip**
We can install SciPy library with the help of command prompt by executing the below command and this will install SciPy along with its dependencies −

```
pip install scipy
```

Following is the output of installing SciPy library with the help of
**pip**command.![pip_windows](/scipy/images/pip_windows.jpg)**Installing using conda**
First we need to install
**Anaconda**software from the[official website](https://www.anaconda.com/download#download-section)then we can install SciPy library using the below command executing in the Anaconda Prompt −
```
conda install scipy
```

#### Verification of Installation

The Verification of Installation can be done by checking the version of python by executing the below code in command prompt, if the version is printed then installation successful −

```
(myenv) D:\Projects\python\myenv>py
Python 3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import scipy
>>> print(scipy.__version__)
```

Following is the version of Scipy as installation is successful −

```
1.16.3
```

## Installing SciPy on macOS

Below are the steps that need to be followed for installing SciPy in macOS −

### Installing Python

macOS comes with Python 2.x pre-installed but its recommended to use Python 3.x which can be installed via Homebrew or directly from the official Python website.

### Installation of SciPy

The Scipy library can be installed in two ways as same as discussed in windows OS −
**Installation using pip**
```
pip install scipy
```
**Installation using Anaconda**
```
conda install scipy
```

## Installing SciPy on Linux

Here are the steps that need to be followed for installing SciPy in Linus OS −

### Installing Python

Before proceeding with the installation of SciPy, we have to check whether it is already installed or not by checking the version with below code −

```
python3 --version
```

Most of Linux distributions come with Python pre-installed. If its not installed we can install it using our package manager.
**Installation for Ubuntu/Debian**
```
sudo apt-get update
sudo apt-get install python3 python3-pip
```
**Installation for Arch Linux**
```
sudo pacman -S python python-pip
```

### Installation of Scipy

The Scipy library can be installed in two ways one is using pip, conda −
**Installation using pip**
```
pip3 install scipy
```
**Installation using script**
```
bash Anaconda3-*.sh
```
**Install SciPy using the Terminal**
```
conda install scipy
```

### Checking the Version

After installing SciPy on any of the above platforms, we can verify that the installation was successful with the help of below steps −

- 
Open the terminal either command prompt or Anaconda prompt. Start a Python session with the help of below code.

```
python3
```

- 
Import SciPy and check its version using the following program −

```
import scipy
print(scipy.__version__)
```

If no errors are thrown and the version number is printed then, the SciPy installation was successful.

---

## 3. SciPy - Basic Functionality

*Source: [https://www.tutorialspoint.com/scipy/scipy_basic_functionality.htm](https://www.tutorialspoint.com/scipy/scipy_basic_functionality.htm)*

---

---
[Previous](/scipy/scipy_environment_setup.htm)[Quiz](/scipy/quiz_on_scipy_basic_functionality.htm)[Next](/scipy/scipy_relationship_with_numpy.htm)**SciPy**is an open-source Python library that is widely used in the scientific community for various tasks involving numerical computation. This library is built on top of NumPy.
SciPy extends its capabilities by providing additional functionality that is essential for scientific and engineering applications. This includes algorithms and functions for optimization, integration, interpolation, eigenvalue problems and solving differential equations.

Here is the overview of the basic functionalities of SciPy library −

## Numerical Integration

Numerical integration refers to techniques for calculating the integral of a function when an analytical solution is difficult or impossible to obtain. SciPy provides several methods for numerical integration as discussed below −

- **Quad Integration:**This is used to compute the integral of a function over a specified interval. In SciPy using**scipy.integrate.quad()**we can perform adaptive quadrature to compute definite integrals. It returns both the integral result and an estimate of the error.
- **Simpsons Rule:**This is a numerical method for estimating the definite integral of a function. It is particularly effective for integrating smooth functions and is based on approximating the function by a quadratic polynomial.
This can be implemented by
**scipy.integrate.simps**this method uses polynomial interpolation to provide a more accurate estimate than simple trapezoidal rules.
### Example

Below is the example of computing the integral
**f(x) = e**, which is the area under the curve of the Gaussian function and it converges to a known value by using the**quad()**function of the**scipy.integrate**module.
```
from scipy import integrate
import numpy as np

# Function to integrate
def f(x):
   return np.exp(-x**2)

# Compute the integral from 0 to infinity
result, error = integrate.quad(f, 0, np.inf)
print(f"Integral result: {result}, Error estimate: {error}")
```

Following is the output of the
**quad()**function −
```
Integral result: 0.8862269254527579, Error estimate: 7.101318378329813e-09
```

## Optimization

Optimization involves finding the best solution i.e. maximum or minimum of a given function. SciPy offers several optimization algorithms through the
**scipy.optimize()**. Following are the methods that optimization can be done −
- **Minimization:**Use functions like minimize to find the minimum of a scalar function of one or more variables. It supports various methods such as Nelder-Mead, BFGS and Powell etc.
- **Root Finding:**The**fsolve()**function helps find the roots of a system of equations.
### Example

The following example shows how to use the
**minimize()**function of the scipy.optimize module to find the minimum of a simple quadratic function −
```
from scipy import optimize

# Define a function to minimize
def objective_function(x):
   return (x - 2)**2 + 1

# Find the minimum of the function
result = optimize.minimize(objective_function, x0=0)
print(f"Minimum value: {result.fun} at x = {result.x}")
```

Following is the output of the
**minimize()**function −
```
Minimum value: 1.0000000000000007 at x = [1.99999997]
```

## Interpolation

Interpolation is a technique to estimate unknown values that fall between known data points. SciPys
**scipy.interpolate**module provides various methods for interpolation −
- **Linear and Cubic Interpolation**− Use interp1d for one-dimensional interpolation which can be linear, quadratic or cubic.
- **Barycentric Interpolation**− This provides high-order polynomial interpolation and is more numerically stable.
### Example

Here is the example which shows how to perform linear interpolation using SciPy and visualize the results with Matplotlib −

```
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.array([0, 1, 2, 3, 4])  # Original x values
y = np.array([1, 3, 2, 5, 4])  # Corresponding y values

# Create a linear interpolation function
linear_interp = interpolate.interp1d(x, y)

# New x values for interpolation
x_new = np.linspace(0, 4, 100)  # 100 points between 0 and 4
y_new = linear_interp(x_new)  # Interpolated y values

# Plot the original data and the interpolation
plt.plot(x, y, 'o', label='Data points')  # Original data points as circles
plt.plot(x_new, y_new, '-', label='Linear interpolation')  # Interpolated line
plt.legend()  # Show legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Interpolation')
plt.grid()  # Add a grid
plt.show()  # Display the plot
```

Following is the output of the Linear interpolation −
![interpolation](/scipy/images/basic_fun_interpolation.jpg)
## Eigenvalue Problems

Eigenvalue problems are a fundamental concept in linear algebra, arising in various fields such as physics, engineering, and data science. They involve finding eigenvalues and eigenvectors of a matrix. An eigenvalue
and its corresponding eigenvectorof a square matrixsatisfy the equation as defined below −
```
A v = λ v
```
**A****v**=**λ****v**
### Example

Below is the example which shows how to compute the eigenvalues and eigenvectors of a given matrix using the
**eig()**function of the scipy.linalg module.
```
from scipy.linalg import eig
import numpy as np
# Define a matrix
A = np.array([[1, 2], [2, 1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
```

Here is the output of the eigenvalues and eigenvectors of a given matrix −

```
Eigenvalues: [ 3.+0.j -1.+0.j]
Eigenvectors:
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]
```

## Algebraic Equations

SciPy provides tools for solving linear algebraic equations. The
**scipy.linalg**module includes functions for matrix operations, solving linear systems and computing determinants.
### Example

This is the example which shows how to use the
**solve()**function of the**scipy.linalg**module on linear equations −
```
import numpy as np
from scipy.linalg import solve

# Define a coefficient matrix A and a right-hand side vector b
A = np.array([[3, 2], [1, 2]])
b = np.array([5, 5])

# Solve the linear system
x = solve(A, b)
print(f"Solution of the linear system: {x}")
```

Following is the output of the above program −

```
Solution of the linear system: [0.  2.5]
```

## Statistical Functions

SciPy also includes the
**scipy.stats**module which contains a large collection of statistical distributions and functions for statistical testing.
- **Probability Distributions:**SciPy supports many continuous and discrete distributions such as normal, binomial, Poisson etc.
- **Statistical Tests:**There are functions for performing statistical tests such as t-tests and chi-square tests.
### Example

This example shows how to perform a Shapiro-Wilk test for normality using the
**scipy.stats**module in Python. This statistical test checks whether a given dataset follows a normal distribution −
```
from scipy import stats
import numpy as np
# Generate random data
data = np.random.normal(0, 1, 1000)

# Perform a Shapiro-Wilk test for normality
statistic, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk test statistic: {statistic}, p-value: {p_value}")
```

Following is the output of the above program −

```
Shapiro-Wilk test statistic: 0.9984066795076805, p-value: 0.49518026390115066
```

---

## 4. SciPy - Relationship with NumPy

*Source: [https://www.tutorialspoint.com/scipy/scipy_relationship_with_numpy.htm](https://www.tutorialspoint.com/scipy/scipy_relationship_with_numpy.htm)*

---

---

## 5. SciPy - Cluster

*Source: [https://www.tutorialspoint.com/scipy/scipy_cluster.htm](https://www.tutorialspoint.com/scipy/scipy_cluster.htm)*

---

---
[Previous](/scipy/scipy_relationship_with_numpy.htm)[Quiz](/scipy/quiz_on_scipy_cluster.htm)[Next](/scipy/scipy_hierarchical_clustering.htm)
## What is SciPy Clustering?

In SciPy
**Clustering**refers to the process of grouping a set of data points into clusters based on their similarities or distances from one another.
The goal of clustering is to partition the data into subsets where data points within each subset are more similar to each other than to those in other subsets. SciPy provides a range of clustering algorithms within its
**scipy.cluster**module to perform this task.
## Types of SciPy Clusters

SciPy offers a variety of clustering techniques through its
**scipy.cluster**module. Below is the image which shows types of clusters in SciPy −![Types of Clusters](/scipy/images/types_clusters.jpg)
Here's a detailed overview of the clustering methods available −

## Hierarchical Clustering
**Hierarchical clustering**is a method of cluster analysis that seeks to build a hierarchy of clusters. This approach can be visualized as a tree structure known as a dendrogram.
In hierarchical clustering the data is grouped either by successively merging smaller clusters into larger ones i.e. agglomerative or by splitting a large cluster into smaller ones i.e. divisive.

Following are the different types of Scipy
**Hierarchical Clustering**−
### Agglomerative Hierarchical Clustering
**Agglomerative Hierarchical Clustering**is a widely-used clustering technique that builds a hierarchy of clusters by progressively merging the closest clusters.
This approach is a bottom-up method which means it starts with each data point as its own cluster and then merges clusters step by step until all data points are grouped into a single cluster. The process is visualized using a dendrogram, a tree-like structure that represents the merging steps.

In Python the
**scipy.cluster.hierarchy**module provides the linkage function to perform agglomerative hierarchical clustering. This function is versatile and allows for the use of various linkage methods each of which defines a different criterion for merging clusters.
The choice of linkage method affects how the distances between clusters are calculated and consequently, how clusters are merged. Let's see the different linkage methods available in SciPy
**Agglomerative Clustering**−
#### Single Linkage

Single Linkage Clustering is a type of hierarchical clustering that is also known as the "minimum linkage" or "nearest neighbor" method. It is one of the simplest methods for agglomerative clustering.

The distance between two clusters is defined as the shortest distance between any single pair of points, where each point is from one of the two clusters. This method tends to produce long "chain-like" clusters and can be sensitive to noise.

Following is the mathematical formula given to calculate the distance −

```
d(A,B) = min{d(a,b):aA,bB}
```

#### Complete Linkage

Complete Linkage Clustering is a type of hierarchical clustering which is also known as "maximum linkage" or "farthest neighbor" method. It is an agglomerative clustering technique where the distance between two clusters is defined as the maximum distance between any single point in one cluster and any single point in the other cluster.

Here is the mathematical formula to calculate the distance −

```
d(A,B) = max{d(a,b):aA,bB}
```

#### Average Linkage

Average Linkage Clustering is also known as the "UPGMA" (Unweighted Pair Group Method with Arithmetic Mean) or "group average" method which is another type of hierarchical clustering. In this method the distance between two clusters is defined as the average distance between all pairs of points where one point is from each cluster.

Here is the mathematical formula to calculate the distance of average linkage −
![Average linkage](/scipy/images/average_linkage.jpg)
#### Ward Linkage

Ward linkage is a type of Agglomerative Clustering method used in hierarchical clustering to compute the distances between clusters based on the variance of their combined data points.

It is designed to minimize the total within-cluster variance or equivalently, to minimize the sum of squared differences within clusters. The mathematical formula to calculate the distance is given as follows −
![ward_linkage](/scipy/images/ward_linkage.jpg)
#### Dendrogram

A dendrogram is a tree-like diagram that is used to illustrate the arrangement of clusters formed by hierarchical clustering. It visually represents the hierarchical relationships between clusters and helps to understand the data structure and clustering process.
**Key Elements of a Dendrogram**
The Key elements of the Dendrogram is given as follows −

- **Leaves**− Represent the individual data points or objects.
- **Nodes**− Represent clusters formed at each level of the hierarchy. The height of a node indicates the distance at which clusters are merged.
- **Branches**− Connect nodes and show the relationship between clusters at different levels.
### Divisive Clustering

Divisive Clustering is also known as Top-Down Hierarchical Clustering which is a type of hierarchical clustering method that begins with all data points in a single cluster and then recursively splits the clusters into smaller clusters until each data point is its own cluster or a desired number of clusters is reached.

This is in contrast to agglomerative clustering which starts with each data point as its own cluster and merges them into larger clusters.

## K - Means Clustering

K-means clustering is a popular and straightforward unsupervised machine learning algorithm used to partition a dataset into a set of distinct, non-overlapping groups (clusters).

It is particularly useful for tasks where we need to group similar data points together but the exact number of groups (clusters) is not known beforehand.
**Key Concepts of K-means Clustering**
Following are the key concepts of K - Means Clustering −

- **Centroids**− The K-means algorithm partitions the data into k clusters. Each cluster is represented by its centroid, which is the mean of all the data points within that cluster.
- **Cluster**− A cluster is a collection of data points grouped together because of their similarity to one another. Each cluster is represented by a centroid which is the mean of all points in the cluster.
- **Inertia**− Inertia quantifies how tightly the data points are clustered around their centroids. It is the sum of the squared distances between each data point and the centroid of the cluster to which it belongs. Mathematically it's defined as −![inertia](/scipy/images/inertia.png)
### Types of K-Means clustering

In SciPy the K-means clustering algorithm is primarily implemented through the
**scipy.cluster.vq module**. Following are the different types of K-Means Clustering −
#### Standard K-Means Clustering

Standard K-Means Clustering is a widely used clustering algorithm that partitions a dataset into a specified number of clusters (K) by minimizing the variance within each cluster.

This is widely used in various applications such as customer segmentation, image compression and anomaly detection. However it is important to consider its limitations and possibly use more advanced variants or techniques depending on the specific data and clustering requirements.

#### Standard K-Means Clustering

K-Means++ is an enhancement of the standard K-means clustering algorithm designed to improve the initialization step. The primary goal of K-Means++ is to address the problem of poor clustering results due to suboptimal initial centroid placement which can occur with random initialization in the standard K-means algorithm.

## SciPy Cluster Module

The cluster module provide the functionality related to cluster algorithm. Following are the methods of the SciPy
**Cluster**−Sr.No.Types & Description1**fcluster()**
This method is a part of hierarchical algorithm which group the data points into a specified number of cluster.
2**fclusterdata()**
This method grouped the similar data into cluster.
3**leaders()**
This method is used to identify the cluster center.
4**linkage()**
This method works on hierarchical cluster which can be used to perform the task of linkage matrix.
5**single()**
This method performs the task of single/minimum/nearest linkage on a condensed matrix.
6**complete()**
This Method perform the task of complete linkage(largest point) on a condensed distance matrix.
7**average()**
This method is used to perform the task of arithmetic mean on a distance matrix.
8**weighted()**
This method depends on other functions which user can perform such as weighted means, weighted sums, and weighted operations.
9**centroid()**
This method define an one-dimensional array in which data values are calculated with the help of average weight and these weights itself represent a value.
10**median()**
This method is used to find the median value of an array.
11**ward()**
This method is a part of agglomerative cluster which minimize the total cluster variance within its control.
12**cophenet()**
This method calculates the cophenetic distance between each observation of the hierarchical cluster.
13**from_mlab_linkage()**
This method is used to work with clustering algorithm(mlab.linkage) and converts it into a format that can be used for the references of other SciPy clustering functions.
14**inconsistent()**
This method is used to perform the calculation of inconsistency statistics on a linkage matrix.
15**maxinconsts()**
This method is used to calculate the distances between two datasets.
16**maxdists()**
This method calculate the pairwise distances between the points from the given set.
17**maxRstat()**
This method perform the task of maximum value obtained by a column R for each non-singleton cluster and its children.
18**to_mlab_linkage()**
This method is used to convert the clustering output into MATLAB compatible format.
19**dendrogram()**
This method determine its functionality by cutting clusters at a particular height.
20**set_link_color_palette()**
This method perform the task of matplotlib color codes while representing different level of clusters.
21**DisjointSet()**
This method is used to manage the data partition set into a disjoint subsets.

## Representing Hierarchies as Tree Objects

SciPy offers functions for managing hierarchical clustering trees, allowing you to group data into clusters based on their similarities. These functions help you visualize, reorganize, and retrieve clusters from hierarchical tree structures.
Sr.No.Function & Description1[scipy.clusternode()](/scipy/scipy_clusternode_function.htm)
This method represents a node in a hierarchical clustering tree.
2[scipy.optimal.leaf.ordering()](/scipy/scipy_optimal_leaf_ordering_function.htm)
This method reorders the leaves to minimize the distance between adjacent clusters.
3[scipy.leaves.list()](/scipy/scipy_leaves_list_function.htm)
This method returns the leaf node order in a hierarchical clustering tree.
4[scipy.to.ree()](/scipy/scipy_to_tree_function.htm)
This method converts a linkage matrix into a tree object.
5[scipy.cut.tree()](/scipy/scipy_cut_tree_function.htm)
This method extracts cluster memberships by cutting the hierarchical tree at a given depth.

---

## 6. SciPy - Hiearchical Clustering

*Source: [https://www.tutorialspoint.com/scipy/scipy_hierarchical_clustering.htm](https://www.tutorialspoint.com/scipy/scipy_hierarchical_clustering.htm)*

---

---
[Previous](/scipy/scipy_cluster.htm)[Quiz](/scipy/quiz_on_scipy_hierarchical_clustering.htm)[Next](/scipy/scipy_k_means_clustering.htm)
## What is Hierarchical Clustering?

In Scipy
**Hierarchical clustering**is a method of cluster analysis that builds a hierarchy of clusters by either successively merging smaller clusters into larger ones i.e. agglomerative approach or splitting larger clusters into smaller ones i.e. divisive approach.
This method does not require specifying the number of clusters beforehand. The result is typically visualized using a dendrogram which is a tree-like diagram showing the arrangement and distance between clusters at each step.
**Hierarchical clustering**helps us to reveal the data's natural structure and relationships by making it useful for exploratory data analysis and identifying patterns or groupings in complex datasets.
## Types of Hierarchical Clustering
**Hierarchical clustering**can be categorized based on its approach to forming clusters. Each type of Hierarchical clustering has different methods for building clusters and varies in how it handles the data.
Following are the two primary types of hierarchical clustering −

- **Agglomerative Hierarchical Clustering**− This approach is bottom-up. It starts with each data point as its own individual cluster and progressively merges the closest pairs of clusters.
- **Divisive Hierarchical Clustering**− This approach is top-down. It starts with all data points in a single cluster and recursively splits it into smaller clusters.
Now let's see in detail about each type of Hierarchical Clustering.

## Agglomerative Hierarchical Clustering
**Agglomerative hierarchical clustering**is a bottom-up approach where each data point starts as its own cluster. It iteratively merges the closest pairs of clusters based on a chosen linkage criterion such as single, complete or average linkage until all points are grouped into a single cluster or a predefined number of clusters is reached.
This method builds a hierarchy of clusters which is often visualized using a dendrogram, illustrating the sequence of merges and the distances at which they occurred. It is widely used in data analysis to uncover the underlying structure and relationships within data.

### Agglomerative hierarchical clustering in SciPy

SciPy has
**scipy.cluster.hierarchy**module which provides comprehensive tools for performing agglomerative hierarchical clustering which is a method of cluster analysis that builds a hierarchy of clusters through a series of merging operations.
Below are the functions which are used to perform
**Agglomerative Hierarchical Clustering**−
### Linkage Computation

The
**linkage()**function computes the hierarchical clustering encoded in a linkage matrix. This matrix describes the clustering process and is used for further analysis.
#### Syntax

Here is the syntax of Scipy Agglomerative Hierarchical Clustering
**linkage()**function −
```
scipy.cluster.hierarchy.linkage(Y, method='ward', metric='euclidean')
```

#### Parameters

Following are the parameters of the
**linkage()**function of the Agglomerative Hierarchical Clustering −
- **Y**− Distance matrix i.e. condensed form from pdist or a square matrix representing pairwise distances.
- **method**− This is the linkage method such as 'single', 'complete', 'average', 'ward'.
- **metric**− This Parameter is Distance metric. The default value is 'euclidean'.
#### Example

Following is the example of using the
**linkage()**function of Agglomerative Hierarchical Clustering.  This example performs hierarchical clustering using SciPys linkage function with Wards method −
```
from scipy.cluster.hierarchy import linkage
import numpy as np

# Generate random sample data
data = np.random.rand(10, 2)  # 10 points in 2D space

# Compute the linkage matrix
Z = linkage(data, method='ward')
print(Z)
```

##### Output

Following is the output of the linkage() function −

```
[[ 0.          7.          0.0505634   2.        ]
 [ 4.         10.          0.09255057  3.        ]
 [ 1.          5.          0.15725673  2.        ]
 [ 2.          8.          0.22920974  2.        ]
 [ 9.         11.          0.24129559  4.        ]
 [ 3.         13.          0.29270489  3.        ]
 [ 6.         14.          0.32005747  5.        ]
 [12.         15.          0.93642962  5.        ]
 [16.         17.          0.98112101 10.        ]]
```

### Dendrogram Visualization

The
**dendrogram()**function creates a dendrogram which is a tree-like diagram that shows the arrangement and distances of clusters as they are merged.
#### Syntax

Here is the syntax of Scipy Agglomerative Hierarchical Clustering
**dendrogram()**function −
```
scipy.cluster.hierarchy.dendrogram(Z, **kwargs)
```

#### Parameters

Below are the parameters of the
**dendrogram()**function of the agglomerative hierarchical clustering−
- **Z**− Linkage Matrix
- **kwargs**− These are the optional arguments for customization such as color_threshold, labels, leaf_rotation.
#### Example

Here is the example of using the
**dendrogram()**function of Agglomerative Hierarchical Clustering which generates the image of the matrix computed with the help of**linkage()**function −
```
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import numpy as np
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
Z = linkage(data, method='ward') 
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

##### Output

Here is the output image generated with the help of dendrogram function −
![Dendrogram](/scipy/images/dendrogram.jpg)
### Forming Flat Clusters

The
**fcluster**function in SciPy is used to extract flat clusters from hierarchical clustering results defined by a linkage matrix. This function converts the hierarchical clustering into a specific number of clusters or based on a distance threshold by making it easier to work with and interpret the results.
#### Syntax

Here is the syntax of Scipy Agglomerative Hierarchical Clustering
**dendrogram()**function −
```
scipy.cluster.hierarchy.fcluster(Z, t, criterion='inconsistent', **kwargs)
```

#### Parameters

Here are the parameters of the
**fcluster()**function of the Scipy Agglomerative Hierarchical Clustering −
- **Z:**The linkage matrix obtained from the linkage function which represents the hierarchical clustering of the data.
- **t:**The threshold for forming flat clusters.
- **criterion:**This parameter determines how the flat clusters are formed.
- ****kwargs:**Additional keyword arguments which can defined depending on the criterion
#### Example

Here is the example of the
**fcluster()**function which converts the hierarchical clustering results into a format that is easier to analyze and interpret by allowing for practical application of clustering results −
```
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
import numpy as np
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
Z = linkage(data, method='ward') 
clusters = fcluster(Z, t=3, criterion='maxclust')  # Form 3 clusters
print(f"Cluster assignments: {clusters}")
```

##### Output

Below is the output of the
**fcluster()**function −
```
Cluster assignments: [1 1 2 2]
```

## Divisive Hierarchical Clustering
**Divisive Hierarchical Clustering**is a clustering method where the process starts with a single, all-encompassing cluster containing all data points and iteratively splits it into smaller clusters until each cluster meets a certain criterion or the desired number of clusters is achieved.
This is in contrast to Agglomerative Hierarchical Clustering which begins with individual data points and merges them into larger clusters.

SciPy does not provide a built-in implementation for Divisive Hierarchical Clustering. Let's see how we can illustrate how it might be implemented manually in Python. Below are the steps to be followed to implement the Divisive Hierarchical Clustering manually −

- 
Initialize with all data points in one cluster.

- 
Iteratively split the largest cluster.

- 
Repeat this until the desired number of clusters are achieved.

For simplicity the below example will use a basic approach to splitting clusters but note that real-world implementations might be more sophisticated.

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def split_cluster(cluster):
    """ Split a cluster into two sub-clusters based on a simple approach. """
    # Compute the centroid of the cluster
    centroid = np.mean(cluster, axis=0)
    # Compute distances from each point to the centroid
    distances = cdist(cluster, [centroid], metric='euclidean').ravel()
    # Split the cluster into two based on distance from the centroid
    median_distance = np.median(distances)
    cluster1 = cluster[distances < median_distance]
    cluster2 = cluster[distances >= median_distance]
    return cluster1, cluster2

def divisive_clustering(data, n_clusters):
    """ Perform Divisive Hierarchical Clustering. """
    clusters = [data]
    while len(clusters) < n_clusters:
        # Find the largest cluster to split
        largest_cluster = max(clusters, key=len)
        clusters.remove(largest_cluster)
        # Split the largest cluster
        cluster1, cluster2 = split_cluster(largest_cluster)
        # Add the new clusters
        clusters.append(cluster1)
        clusters.append(cluster2)
    return clusters

# Generate synthetic data
np.random.seed(0)
data = np.random.rand(100, 2)  # 100 points in 2D space

# Perform divisive hierarchical clustering
n_clusters = 4
clusters = divisive_clustering(data, n_clusters)

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster in clusters:
    plt.scatter(cluster[:, 0], cluster[:, 1])
plt.title('Divisive Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

##### Output

Following is the output of the Divisive Hierarchical Clustering −
![Divisive](/scipy/images/divisive.jpg)

---

## 7. SciPy - Hiearchical Clustering

*Source: [https://www.tutorialspoint.com/scipy/scipy_k_means_clustering.htm](https://www.tutorialspoint.com/scipy/scipy_k_means_clustering.htm)*

---

---
[Previous](/scipy/scipy_hierarchical_clustering.htm)[Quiz](/scipy/quiz_on_scipy_k_means_clustering.htm)[Next](/scipy/scipy_distance_metrics.htm)
## What is K-Means Clustering?
**SciPy K-means clustering**is a technique for partitioning data into 'K' clusters implemented in the**scipy.cluster.vq**module. It includes functions like 'kmeans' for clustering and 'vq' for assigning data points to clusters.
The algorithm works by iteratively updating cluster centroids and reassigning data points based on their distance to these centroids by aiming to minimize the within-cluster variance. SciPy's implementation allows for different initialization methods such as random or K-Means++ which improves centroid placement.

This clustering is useful for identifying groupings in datasets which although it is less feature-rich compared to other libraries such as scikit-learn.

## Types of K-Means Clustering

K-means clustering is a widely used algorithm for partitioning data into clusters. Various types and variations of K-means clustering are employed to address specific needs or improve performance. Let's see them in detail −

### Standard K-Means Clustering

Standard K-Means Clustering is a popular algorithm used for partitioning data into K distinct clusters. It works through an iterative process to assign data points to clusters and update cluster centroids.

### How Does K-Means Clustering Work?

Here are the steps which shows how does the Standard K-Means Clustering works −

#### Initialization

Initialization is a crucial step in K-means clustering as it significantly impacts the algorithms performance and the quality of the final clustering results.

- **Select the Number of Clusters (K)**− Here we will decide the number of clusters that we want to form in the dataset. This is a crucial parameter that affects the results. We can use the methods such as Elbow, Silhouette Score, Gap Statistic, Cross-Validation as per our requiremnet.
- **Initialize Centroids**− We can start by randomly selecting (K) data points from our dataset to serve as the initial centroids (cluster centers). Alternatively, more advanced initialization methods such as K-Means++ can be employed to achieve better clustering results.
#### Assignment Step

The Assignment Step in K-means clustering is the phase where each data point in the dataset is assigned to the nearest cluster is based on the current positions of the centroids. This step is crucial because it determines the composition of each cluster which directly influencing the subsequent update step.

- **Compute Distances**− For each data point we calculate the distance to each of the K centroids. Common distance metrics such as Euclidean distance, Manhattan distance, etc.
- **Assign Clusters**− Assign each data point to the cluster associated with the nearest centroid. This forms K clusters where each data point belongs to the cluster with the closest centroid.
#### Update Step

The Update Step in the K-means clustering algorithm is crucial for refining the positions of the centroids based on the current cluster assignments of the data points. After the data points have been assigned to the nearest centroid i.e calculated in assignment step, the centroids are updated to reflect the mean position of all data points assigned to each cluster. This process continues iteratively until the centroids stabilize.

- **Recalculate Centroids**− Once all data points are assigned to clusters we can recompute the centroids for each cluster. The new centroid is calculated as the mean of all the data points within that cluster. Mathematically the formula is given as follows −![centroids](/scipy/images/centroids.png)
- **Update Centroids**− Replace the old centroids with the newly recalculated centroids.
- **Convergence Check**− After updating the centroids the algorithm checks whether the centroids have moved significantly compared to their previous positions.
If the movement or change of centroids is below a certain threshold then the algorithm considers this as convergence and stops the iterations. Otherwise, the algorithm goes back to the Assignment Step to reassign data points to the nearest updated centroid.

### Example

Following is the example which shows how to apply
**standard K-means clustering**using SciPy, visualize the results and interpret the output −
```
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(0)
data = np.vstack([np.random.normal(0, 0.5, (50, 2)), 
                  np.random.normal(3, 0.5, (50, 2)), 
                  np.random.normal(6, 0.5, (50, 2))])

# Number of clusters
k = 3

# Perform K-means clustering
centroids, distortion = kmeans(data, k)

# Assign each sample to a cluster
labels, _ = vq(data, centroids)

# Plot the results
plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering using SciPy')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(f"Centroids:\n{centroids}")
print(f"Distortion: {distortion}")
```

#### Output

Following is the output of the Standard K-Means clustering −

```
Centroids:
[[0.00840308 0.05140493]
 [5.95346338 5.98730436]
 [2.99063924 3.09137373]]
Distortion: 0.6258523704544776
```
![standard_k_means](/scipy/images/standard_k_means.jpg)
### K-Means++ Clustering
**K-means++**is an advanced version of the standard K-means clustering algorithm which is designed to improve the initialization step by choosing the initial centroids more strategically. This approach enhances the performance and accuracy of the K-means algorithm by reducing the likelihood of poor clustering results due to random initialization.
#### How K-means++ Works?

Following are the steps which shows how the K-Means++ clustering works −

- **First Centroid Selection**− The algorithm begins by randomly selecting the first centroid from the data points.
- **Subsequent Centroid Selection**− For each data point ( x_i) that has not yet been selected as a centroid, calculate the distance ( D(x_i) ) between ( x_i ) and the nearest already chosen centroid.
Select the next centroid from the remaining data points with a probability proportional to ( D(x_i)^2 ). This means that data points farther from the existing centroids have a higher probability of being chosen as new centroids. The probability is given as follows −
![k_means_formula](/scipy/images/k_means_formula.png)
Repeat this process until ( k ) centroids have been selected.

- **Standard K-means**− Once the initial centroids are selected using the K-means++ method, the standard K-means clustering algorithm is applied. This involves iteratively assigning data points to the nearest centroid, updating the centroids and repeating until convergence.
### Example

SciPy does not directly implement K-means++ in the scipy.cluster.vq module but it can be used through the kmeans function by setting the minit parameter to '++'. This ensures that the centroids are initialized using the K-means++ strategy. Below is the example −

```
import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(0)
data = np.vstack([np.random.normal(0, 0.5, (50, 2)), 
                  np.random.normal(3, 0.5, (50, 2)), 
                  np.random.normal(6, 0.5, (50, 2))])

# Number of clusters
k = 3

# Perform K-means++ clustering
centroids, labels = kmeans2(data, k, minit='++')

# Plot the results
plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means++ Clustering using SciPy')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(f"Centroids:\n{centroids}")
```

#### Output

Here is the output of the K-means++ clustering −

```
Centroids:
[[5.95346338 5.98730436]
 [0.00840308 0.05140493]
 [2.99063924 3.09137373]]
```
![k_means++](/scipy/images/k_means_plus_plus.jpg)
### Vector quantization
**Vector quantization**involves partitioning a large set of vectors into a smaller set of clusters. Each vector in the dataset is approximated by the nearest representative vector which is known as a codebook vector or centroid. The set of these codebook vectors is called a codebook.
Vector quantization (VQ) is a technique in signal processing and machine learning used to compress and encode vector data by mapping it to a finite set of representative vectors. It is widely used in data compression, pattern recognition and clustering.

#### How Vector Quantization Works

Here are the steps which shows how the Vector quantization works −
**Training**
- **Initialization:**Choose an initial set of codebook vectors. This can be done randomly or using methods like K-means clustering.
- **Assignment:**Assign each data vector to the nearest codebook vector.
- **Update:**Recalculate the codebook vectors as the mean of all vectors assigned to each codebook vector.
- **Iteration:**Repeat the assignment and update steps until convergence, meaning that the codebook vectors no longer change significantly.**Encoding**
After training each data vector is encoded by its index in the codebook rather than by the vector itself. This reduces the amount of data needed to represent the original data.
**Decoding**
To reconstruct the data we have to replace each index with the corresponding codebook vector. This results in a compressed approximation of the original data.

### Example

Heres an example of vector quantization using K-means clustering from scipy.cluster.vq which effectively performs vector quantization −

```
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
data = np.vstack([np.random.normal(0, 0.5, (100, 2)), 
                  np.random.normal(3, 0.5, (100, 2)), 
                  np.random.normal(6, 0.5, (100, 2))])

# Number of codebook vectors (clusters)
k = 3

# Perform K-means clustering to get codebook vectors
centroids, distortion = kmeans(data, k)

# Assign each sample to a cluster
labels, _ = vq(data, centroids)

# Plot the results
plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Codebook Vectors')
plt.title('Vector Quantization using K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(f"Codebook Vectors:\n{centroids}")
print(f"Distortion: {distortion}")
```

#### Output

The output of the Vector quantization using K-Means Clustering is given below −
![vector_quant](/scipy/images/vector_quant.jpg)
```
Codebook Vectors:
[[-4.78840902e-04  7.13893340e-02]
 [ 5.94382124e+00  5.94843116e+00]
 [ 2.92846083e+00  2.94352468e+00]]
Distortion: 0.6249447014860251
```

---

## 8. SciPy - Distance Metrics

*Source: [https://www.tutorialspoint.com/scipy/scipy_distance_metrics.htm](https://www.tutorialspoint.com/scipy/scipy_distance_metrics.htm)*

---

---
[Previous](/scipy/scipy_k_means_clustering.htm)[Quiz](/scipy/quiz_on_scipy_distance_metrics.htm)[Next](/scipy/scipy_constants.htm)
## What are Distance Metrics?

In SciPy library
**distance metrics**are crucial for measuring similarity or dissimilarity between two points in a given space. These metrics are widely used in fields such as machine learning, data analysis and clustering for tasks such as classification, clustering and nearest neighbor searches.
The
**scipy.spatial.distance**module offers a variety of these metrics such as Euclidean, Manhattan, Cosine and Hamming distances, among others. Each metric serves different purposes for helping to determine the relationships and structures within datasets.
## Types of Distance Metrics

As we know that the
**scipy.spatial.distance**module provides a wide range of distance metrics with serving a different purpose as per the requirement. Below are the different Distance Metrics available in Scipy −
## Euclidean Distance

In SciPy
**Euclidean distance**is a measure of the straight-line distance between two points in Euclidean space. It is commonly used to quantify the similarity between two vectors by calculating the length of the shortest path connecting them.
The
**scipy.spatial.distance.euclidean()**function is used to calculate the**Euclidean Distance**in Scipy.
Mathematically, it is defined as the square root of the sum of the squared differences between corresponding components of the two vectors. The formula is given as follows −
![Euclidean Distance](/scipy/images/euclidean_dist.jpg)
Where −

- **x = (x**and, x, ....., x)**y = (y**− are the vectors representing the points in the space., y,....., y)
- **(x**−  is the difference between the x and y, y)
### Syntax

Following is the syntax of
**scipy.spatial.distance.euclidean()**function −
```
scipy.spatial.distance.euclidean(u, v)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.euclidean()**function −
- **u:**The first point or vector in n-dimensional space.
- **v:**The second point or vector in n-dimensional space.
### Return Value

This function returns the Euclidean distance between the points u and v.

### Example

Following is a simple example showing how to compute the Euclidean distance between two points using SciPy's euclidean() function −

```
from scipy.spatial.distance import euclidean

# Define two points in 2D space
point1 = [1, 2]
point2 = [4, 6]

# Calculate the Euclidean distance between the two points
distance = euclidean(point1, point2)

print(f"Euclidean Distance: {distance}")
```

Following is the output of the Euclidean Distance calculated for two points −

```
Euclidean Distance: 5.0
```

## Manhattan Distance

Manhattan Distance is also known as City-block Distance or L1 Norm which is a metric used to measure the distance between two points in a grid-like path. This is similar to how one would navigate a city grid.

Unlike Euclidean distance which measures the straight-line distance where as Manhattan distance calculates the total distance traveled along the grid lines.

Mathematically, the formula for calculating the Manhattan Distance −
![Manhattan Distance](/scipy/images/manhattan_dist.jpg)
Where −

- **x = (x**and,x,.....,x)**y = (y**− are the vectors representing the points.,y,.....,y)
- **|x**−  is the absolute difference between the x and y., y|
### Syntax

Following is the syntax of
**scipy.spatial.distance.cityblock()**function −
```
scipy.spatial.distance.cityblock(u, v)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.cityblock()**function −
- **u:**The first point or vector.
- **v:**The second point or vector.
### Return Value

This function returns the City block distance between the vectors u and v.

### Example

Here is the example which calculates the Manhattan Distance with the help of Scipy
**cityblock()**function −
```
from scipy.spatial.distance import cityblock

# Define two vectors
vector1 = [1, 2, 3]
vector2 = [4, 6, 8]

# Calculate the City Block distance
distance = cityblock(vector1, vector2)

print(f"City Block Distance: {distance}")
```

Following is the output of the Cityblock Distance calculated for two points −

```
City Block Distance: 12
```

## Minkowski Distance
**Minkowski Distance**is a generalization of both Euclidean and Manhattan distances and is used to measure the distance between two points in a normed vector space.
It provides a flexible framework by introducing a parameter
**p**which determines the specific distance metric being used. Mathematically, the formula for calculating the Manhattan Distance −![Minkowski Distance](/scipy/images/minkowski_dist.jpg)
Where −

- **x = (x**and, x,....., x)**y = (y**− are the vectors representing the points., y,....., y)
- **|x**−  is the absolute difference between the x and y., y|
- **p**− is a parameter that defines the distance metric.
### Syntax

Following is the syntax of
**scipy.spatial.distance.minkowski()**function −
```
scipy.spatial.distance.minkowski(u, v, p=2)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.minkowski()**function −
- **u:**The first point or vector which is an array of coordinates.
- **v:**The second point or vector which is an array of coordinates.
- **p(float, optional):**The power parameter for the Minkowski distance. Default is 2.
Note that,

When
**p = 1**, it calculates the Manhattan Distance.
When
**p = 2**, it calculates the Euclidean Distance.
When values of
**p > 2**measures a more general Minkowski distance.
### Return Value

This function returns the Minkowski distance between the two points.

### Example

Below is the example of finding the Minkowski distance between two points with the help of
**minkowski()**function −
```
from scipy.spatial.distance import minkowski

# Define two points in 2D space
point1 = [1, 2]
point2 = [4, 6]

# Calculate Minkowski distance with p=3
distance = minkowski(point1, point2, p=3)

print(f"Minkowski Distance (p=3): {distance}")
```

Following is the output of the Minkowski Distance calculated for two points −

```
Minkowski Distance (p=3): 4.497941445275415
```

## Chebyshev Distance
**Chebyshev Distance**is also known as the Maximum Metric or**L**Norm which is a distance metric used to measure the distance between two points in a grid-like system.
It is defined as the greatest of the absolute differences along any coordinate dimension. Mathematically the formula for calculating the Chebyshev Distance −
![chebyshev Distance](/scipy/images/chebyshev_dist.jpg)
Where −

- **x = (x**and,x,.....,x)**y = (y**− are the vectors representing the points.,y,.....,y)
- **|x**−  is the absolute difference between the x and y.,y|
### Syntax

Following is the syntax of
**scipy.spatial.distance.chebyshev()**function −
```
scipy.spatial.distance.chebyshev(u, v)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.chebyshev()**function −
- **u:**An array-like object representing the first point in the space.
- **v:**An array-like object representing the second point in the space.
### Return Value

This function returns the Chebyshev distance between the two points u and v.

### Example

Below is the example of finding the Chebyshev distance between two points with the help of
**Chebyshev()**function −
```
from scipy.spatial.distance import chebyshev

# Define two points
point1 = [1, 2]
point2 = [4, 6]

# Calculate the Chebyshev distance
distance = chebyshev(point1, point2)

print(f"Chebyshev Distance: {distance}")
```

Following is the output of the Chebyshev Distance calculated for two points −

```
Chebyshev Distance: 4
```

## Cosine Distance
**Cosine Distance**is a measure of dissimilarity between two vectors based on the angle between them. It quantifies how different the vectors are by calculating the cosine of the angle between them with the distance being derived from this similarity measure.
It is often used in text analysis and clustering when the magnitude of the vectors is less important than their orientation. Mathematically the formula for calculating the Cosine Distance −
![Cosine Distance](/scipy/images/cosine_dist.jpg)
### Syntax

Following is the syntax of
**scipy.spatial.distance.cosine()**function −
```
scipy.spatial.distance.cosine(u, v)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.cosine()**function −
- **u:**An array-like object representing the first vector.
- **v:**An array-like object representing the second vector.
### Return Value

This function returns the Cosine distance between the two points u and v.

### Example

Below is the example of finding the Cosine distance between two points with the help of
**Cosine()**function −
```
from scipy.spatial.distance import cosine

# Example vectors
vector1 = [1, 0, 1]
vector2 = [0, 1, 1]

# Compute Cosine distance
distance = cosine(vector1, vector2)

print(f"Cosine Distance: {distance}")
```

Following is the output of the Cosine Distance calculated for two points −

```
Cosine Distance: 0.5
```

## Hamming Distance
**Hamming Distance**is a measure of dissimilarity between two strings or binary vectors of equal length. It quantifies the number of positions at which the corresponding elements differ.
It is often used in error detection and correction algorithms as well as in various applications involving binary data.

A Hamming distance of 0 indicates that the vectors are identical while a distance closer to 1 indicates more dissimilarity. Mathematically the formula for calculating the Hamming Distance −
![Hamming Distance](/scipy/images/hamming_dist.jpg)
### Syntax

Following is the syntax of
**scipy.spatial.distance.hamming()**function −
```
scipy.spatial.distance.hamming(u, v)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.hamming()**function −
- **u:**An array-like object or list representing the first vector or string.
- **v:**An array-like object or list representing the second vector or string.
### Return Value

This function returns the Hamming distance between the two points u and v.

### Example

In this example the Hamming distance represents the fraction of positions where the two binary vectors differ −

```
from scipy.spatial.distance import hamming

# Example binary vectors
vector1 = [1, 0, 1, 0, 1]
vector2 = [1, 1, 0, 0, 1]

# Compute Hamming distance
distance = hamming(vector1, vector2)

print(f"Hamming Distance: {distance}")
```

Below is the output of the Hamming Distance calculated for two points −

```
Hamming Distance: 0.4
```

## Jaccard Distance
**Jaccard Distance**is a measure of dissimilarity between two sets. It is calculated as one minus the Jaccard similarity coefficient which is the ratio of the size of the intersection of the sets to the size of their union.
Jaccard distance is often used in binary or categorical data analysis which is particularly in fields like clustering and classification.

In SciPy library the Jaccard distance can be computed using the
**scipy.spatial.distance.jaccard()**function. Mathematically the formula for calculating the Jaccard Distance −![Jaccard Distance](/scipy/images/jaccard_dist.jpg)
Where −

- **|u∩v|:**is the size of the intersection of the two sets.
- **|u∪ ∪ v|:**is the size of the union of the two sets.
### Syntax

Following is the syntax of
**scipy.spatial.distance.jaccard()**function −
```
scipy.spatial.distance.jaccard(u, v)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.jaccard()**function −
- **u:**An array-like object representing the first binary vector or set.
- **v:**An array-like object representing the second binary vector or set.
### Return Value

This function returns the Jaccard distance between the two points u and v.

### Example

Following is the example of using the jaccard() function to calculate the Jaccard Distance in SciPy −

```
from scipy.spatial.distance import jaccard

# Example binary vectors
vector1 = [1, 0, 1, 0, 1, 1]
vector2 = [0, 1, 1, 0, 1, 0]

# Compute Jaccard distance
distance = jaccard(vector1, vector2)

print(f"Jaccard Distance: {distance}")
```

Following is the output of the Jaccard Distance calculated for two points −

```
Jaccard Distance: 0.6
```

## Canberra Distance
**Canberra Distance**is a metric that measures the dissimilarity between two points by summing the absolute differences between their coordinates and normalized by the sum of their absolute values.
- It is particularly sensitive to differences when both coordinates are small by making it useful for cases where values can be zero or near-zero.
- The Canberra distance is often used in various fields such as environmental science and economics where proportional differences are more significant than absolute differences.
Mathematically the formula for calculating the Canberra Distance is given as follows −
![Canberra Distance](/scipy/images/canberra_dist.jpg)
- **|u**− is the absolute difference between the u and v.-v|
- **|u**− is the sum of the absolute values of the th coordinates.|+|v|
### Syntax

Following is the syntax of
**scipy.spatial.distance.canberra()**function −
```
scipy.spatial.distance.canberra(u, v)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.canberra()**function −
- **u:**An array-like object representing the first vector.
- **v:**An array-like object representing the second vector.
### Return Value

This function returns the Canberra distance between the two points u and v.

### Example

Following is the example of using the canberra() function to calculate the Canberra Distance in SciPy −

```
from scipy.spatial.distance import canberra

# Example vectors
vector1 = [10, 20, 30]
vector2 = [15, 24, 36]

# Compute Canberra distance
distance = canberra(vector1, vector2)

print(f"Canberra Distance: {distance}")
```

Below is the output of the Canberra Distance calculated for two points −

```
Canberra Distance: 0.38181818181818183
```

## Bray-Curtis Distance
**Bray-Curtis Distance**is a measure of dissimilarity between two non-negative numerical vectors which often used in ecology and biology for comparing species abundances.
It quantifies the difference between two samples by taking into account the magnitude of their elements by making it particularly useful for datasets where the absolute differences are more important than their relative differences.

In SciPy the Bray-Curtis distance can be calculated using the scipy.spatial.distance.braycurtis() function.

Mathematically the formula for calculating the Canberra Distance is given as follows −
![Bray-Curtis Distance](/scipy/images/bray_dist.jpg)
Where −

- **|u**− is the absolute difference between the corresponding elements of vectors u and v.-v|
- **u**− is the sum of the corresponding elements.+v
### Syntax

Following is the syntax of
**scipy.spatial.distance.braycurtis()**function −
```
scipy.spatial.distance.braycurtis(u, v)
```

### Parameters

Here are the Parameters of the
**scipy.spatial.distance.braycurtis()**function −
- **u:**An array-like object representing the first vector.
- **v:**An array-like object representing the second vector.
### Return Value

This function returns the Bray-Curtis distance between the two points u and v.

### Example

Here is the example of using the braycurtis() function to calculate the Bray-Curtis Distance in SciPy −

```
from scipy.spatial.distance import braycurtis

# Example vectors
vector1 = [1, 3, 5, 7]
vector2 = [2, 4, 6, 8]

# Compute Bray-Curtis distance
distance = braycurtis(vector1, vector2)

print(f"Bray-Curtis Distance: {distance}")
```

Below is the output of the Canberra Distance calculated for two points −

```
Bray-Distance: 0.1111111111111111
```

---

## 9. SciPy - Constants

*Source: [https://www.tutorialspoint.com/scipy/scipy_constants.htm](https://www.tutorialspoint.com/scipy/scipy_constants.htm)*

---

---

## 10. SciPy - Mathematical Constants

*Source: [https://www.tutorialspoint.com/scipy/scipy_mathematical_constants.htm](https://www.tutorialspoint.com/scipy/scipy_mathematical_constants.htm)*

---

---
[Previous](/scipy/scipy_constants.htm)[Quiz](/scipy/quiz_on_scipy_mathematical_constants.htm)[Next](/scipy/scipy_physical_constants.htm)
SciPy provides a set of mathematical constants that are useful across various computational tasks. These constants are pre-defined values that are fundamental in mathematical computations and scientific research.

Below are the key mathematical constants available in SciPy and their theoretical definitions. Let's see them one by one in detail −

## Euler's Number (e)
**Euler's number**is approximately equal to 2.71828 which is the base of the natural logarithm. It is a fundamental constant in mathematics, especially in calculus and complex analysis. It arises naturally in the study of exponential growth, compound interest and in the solutions to differential equations.
### Syntax

Following is the syntax of calculating the Euler's Number with the help of SciPy −

```
scipy.constants.e
```

### Example

Here's an example of how we can use Euler's Number in SciPy for computing continuous growth or decay.

```
import numpy as np
from scipy.constants import e

# Parameters
P0 = 1000  # Initial population
r = 0.05   # Growth rate (5% per unit time)
t = 10     # Time (10 units)

# Calculate the population at time t
P_t = P0 * np.exp(r * t)

print(f"Population after {t} units of time: {P_t}")
```

Following is the output of the above program −

```
Population after 10 units of time: 1648.7212707001281
```

## Pi ()
**Pi**is the ratio of the circumference of a circle to its diameter. It is a transcendental number with an approximate value of 3.14159. This is crucial in geometry, trigonometry and various areas of science and engineering.
### Syntax

Below is the syntax of calculating the Pi with the help of SciPy −

```
scipy.constants.pi
```

### Example

Following is an example of how we can calculate the
**pi**in SciPy −
```
import numpy as np
from scipy.constants import pi
print("Pi:",pi)
```

Following is the output of the above program −

```
Pi: 3.141592653589793
```

## The Golden Ratio ()

The
**Golden ratio**is denoted as (phi) and is approximately given as 1.61803 which is a special number that appears in various contexts in art, architecture and nature. It is defined algebraically as −![Golden Ratio](/scipy/images/golden_ratio.jpg)
It is associated with aesthetically pleasing proportions and is seen in the Fibonacci sequence.

### Syntax

Below is the syntax of calculating the Golden Ratio () with the help of SciPy −

```
scipy.constants.golden
```

### Example

In SciPy the Golden Ratio is provided in the
**scipy.constants.golden**method. Heres how we can access and use it −
```
from scipy.constants import golden
print(f"The Golden Ratio () is: {golden}")
```

Following is the output of the above program −

```
The Golden Ratio () is: 1.618033988749895
```

## Avogadro Constant
**Avogadro constant**is represented as Avogadro or N_A. It provides the number of entities such as atoms or molecules in one mole of a substance.
The Avogadro constant is a fundamental quantity in chemistry and physics. The value of Avogadro constant is given as
**6.02214076e+23 (mol)**.
### Syntax

Here is the syntax of calculating the Avogadro constant with the help of SciPy −

```
scipy.constants.Avogadro
```

### Example

Here is the example which shows how to calculate the Avogadro constant using the method
**scipy.constants.Avogadro**−
```
from scipy.constants import Avogadro
# Print Avogadro's Number
print(f"Avogadro's Number (N_A) is: {Avogadro}")
```

Following is the output of the above program −

```
Avogadro's Number (N_A) is: 6.02214076e+23
```

## Boltzmann Constant (k_B)
**Boltzmann constant**is represented as k_B which is approximately equal to 1.381x10J/K, relates the average kinetic energy of particles in a gas with the temperature of the gas. It is crucial in statistical mechanics and thermodynamics.
### Syntax

Following is the syntax of calculating the Boltzmann Constant with the help of Scipy −

```
scipy.constants.Boltzmann
```

### Example

In this example we are calculating the Boltzmann constant with the help of scipy method
**scipy.constants.Boltzmann**−
```
from scipy.constants import Boltzmann

# Print the Boltzmann constant
print(f"The Boltzmann constant (k) is: {Boltzmann}")
```

Following is the output of the above program −

```
The Boltzmann constant (k) is: 1.380649e-23
```

## Gas Constant
**Gas Constant**is denoted as R,is a fundamental constant that appears in the ideal gas law equation. It relates the pressure, volume and temperature of an ideal gas to the number of moles.
The gas constant provides a bridge between macroscopic and microscopic descriptions of gases. The value of Gas constant is given as 8.314462618 (J/(molK)).

### Syntax

Here is the syntax of calculating the Gas Constant with the help of Scipy −

```
scipy.constants.gas_constant
```

### Example

Below is the example of calculating the Gas constant by using the
**scipy.constants.gas_constant**method availabe in scipy −
```
from scipy.constants import gas_constant

# Print the Gas Constant
print(f"The Gas Constant (R) is: {gas_constant}")
```

Following is the output of the above program −

```
The Gas Constant (R) is: 8.314462618
```

## Elementary Charge(e)
**Elementary Charge**denoted as , which is the magnitude of the electric charge of a proton or the negative charge of an electron.
It is a fundamental constant in electromagnetism and quantum mechanics. The value of Elementary Charge is given as 1.602176634e-19 (Coulombs).

### Syntax

Here is the syntax of calculating the Elementary Charge with the help of SciPy −

```
scipy.constants.e
```

### Example

Here is the example of calculating the Elementary Charge using the method
**scipy.constants.e**−
```
from scipy.constants import e

# Print the elementary charge
print(f"The elementary charge (e) is: {e}")
```

Following is the output of the above program −

```
The elementary charge (e) is: 1.602176634e-19
```

## List of Mathematical Constants

There are many more Mathematical constants but here in this tutorial we discussed about few Mathematical constants which can be used with the help of SciPy library.

We can get all the constants available in the
**scipy.constants**module which include Mathematical Constants, with the help of below code −
```
import scipy
from scipy import constants
print(dir(scipy.constants))
```

Following are the constants list −

```
['Avogadro', 'Boltzmann', 'Btu', 'Btu_IT', 'Btu_th', 'ConstantWarning', 'G', 'Julian_year', 'N_A', 'Planck', 'R', 'Rydberg', 'Stefan_Boltzmann', 'Wien', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_codata', '_constants', '_obsolete_constants', 'acre', 'alpha', 'angstrom', 'arcmin', 'arcminute', 'arcsec', 'arcsecond', 'astronomical_unit',.....................................
........................................................
 'u', 'unit', 'value', 'week', 'yard', 'year', 'yobi', 'yocto', 'yotta', 'zebi', 'zepto', 'zero_Celsius', 'zetta']
```

---

## 11. SciPy Physical Constants

*Source: [https://www.tutorialspoint.com/scipy/scipy_physical_constants.htm](https://www.tutorialspoint.com/scipy/scipy_physical_constants.htm)*

---

---
[Previous](/scipy/scipy_mathematical_constants.htm)[Quiz](/scipy/quiz_on_scipy_physical_constants.htm)[Next](/scipy/scipy_unit_conversion_constants.htm)
The
**scipy.constants**module provides access to fundamental physical constants and it also ensures that these values are maintained with the highest accuracy and are regularly updated according to the latest scientific standards.
The constants such as Planck constant, Speed of light etc, play a critical role in converting between units, performing precise calculations and modeling physical phenomena.

Using these standardized constants, scientists and engineers can ensure that their results are consistent with those obtained globally, facilitating collaboration and reproducibility in research.

This module also includes conversion factors allowing seamless transitions between different units of measurement by further enhancing its utility in scientific computation. Following are the different Physical Constants and let's see them in detail −

## Planck Constant (h)
**Planck Constant**is denoted as h,which is a fundamental constant in quantum mechanics. This constant value is 6.626 x 10Js that relates the energy of a photon to its frequency. It is essential in the study of quantum phenomena.
### Syntax

Here is the syntax of calculating the Planck Constant with the help of Scipy −

```
scipy.constants.Planck
```

### Example

Below is the example of calculating the Planck constant by using the
**scipy.constants.Planck**method available in SciPy −
```
from scipy.constants import Planck

# Print the Planck constant
print(f"The Planck constant (h) is: {Planck}")
```

Following is the output of the above program −

```
The Planck constant (h) is: 6.62607015e-34
```

## Planck Mass
**Planck mass**is a natural unit of mass in the system of Planck units which is used in theoretical physics. It is the mass at which the gravitational force between two objects is comparable to other fundamental forces such as electromagnetism.
Theoretically the value of
**Planck Mass**is given as 2.176434x10Kilograms. Following is the formula for calculating the Planck Mass(mp) −![planck mass](/scipy/images/planck_mass.jpg)
Where −

- **(h-bar):**is the reduced Planck constant.
- **c:**is the speed of light in a vacuum.
- **G:**is the gravitational constant.
### Example

We know that there is no direct method to calculate the Planck mass with the help SciPy but we can calculate it by using the fundamental constants hbar, c and G. Below is the example −

```
from scipy.constants import hbar, c, G

# Calculate Planck mass
m_planck = (hbar * c / G)**0.5

print(f"Planck mass: {m_planck} kg")
```

Following is the output of the above program −

```
Planck mass: 2.1764343427178984e-08 kg
```

## Speed of Light
**Speed of Light**in a vacuum is a fundamental constant that represents the maximum speed at which all energy, matter and information in the universe can travel. It is a critical value in both classical and modern physics particularly in the theory of relativity.
The speed of light in a vacuum is approximately given as 299,792,458 meters per second. Following is the mathematical formula for calculating the speed of light −

```
E = pc
```

Where −

- **E:**is the energy of the photon.
- **p:**is the momentum of the photon
- **c:**is the speed of light.
### Syntax

Following is the syntax of calculating the speed of light with the help of Scipy −

```
scipy.constants.c
```

### Example

Below is the example of calculating the speed of light with the help of
**scipy.constants.c**method −
```
from scipy.constants import c
print(f"Speed of light (c): {c} meters per second")
```

Following is the output of the above program −

```
Speed of light (c): 299792458.0 meters per second
```

## Gravitational Constant (G)

The
**Gravitational constant**is a fundamental constant used in Newton's law of universal gravitation. It is denoted as**G**. It describes the strength of the gravitational force between two masses.
The gravitational constant is approximately given as 6.67430x10
mkgs. The mathematical formula for finding the Gravitational Constant is given as follows −![Gravitational Constant](/scipy/images/gravitational_constant.jpg)
Where −

- **F:**It is the gravitational force between two masses m1 and m2.
- **r:**It is the distance between the centers of the two masses.
- **G:**It is the gravitational constant.
### Syntax

Following is the syntax of Gravitational Constant in Scipy −

```
scipy.constants.G
```

### Example

Below is the example which shows how to find the gravitational constant with the help of
**scipy.constants.G**method −
```
from scipy.constants import G
print(f"Gravitational constant (G): {G} m^3 kg^-1 s^-2")
```

Following is the output of the above program −

```
Gravitational constant (G): 6.6743e-11 m^3 kg^-1 s^-2
```

## Permeability of Free Space ()

The
**Permeability of free space**is also known as the magnetic constant which is a physical constant that describes the ability of a vacuum to support the formation of a magnetic field.
It is denoted by the symbol
**()**. It is a fundamental parameter in electromagnetism. The value of permeability of free space is approximately given by**4 x 10**. The mathematical formula for finding the Permeability is given as follows −
```
B=0H
```
H
Where −

- **B:**is the magnetic flux density (or magnetic field)
- **H:**is the magnetic field strength.
### Syntax

Following is the syntax of Permeability of Free Space in Scipy −

```
scipy.constants.mu_0
```

### Example

Below is the example which shows how to find the Permeability using the method
**scipy.constants.mu_0**which is availabe in Scipy −
```
from scipy.constants import mu_0
print(f"Permeability of free space (): {mu_0} H/m")
```

Following is the output of the above program −

```
Permeability of free space (): 1.25663706212e-06 H/m
```

## Permittivity of Free Space ()

The
**Permittivity of free space**is also known as the electric constant which is a fundamental physical constant that describes how electric fields interact with the vacuum. It is denoted by the symbol. It is a measure of the ability of a vacuum to permit electric field lines.
The value of Permittivity of free space is approximately given as 8.85418781710
F/m.  In Scipy it is represented as**epsilon_0**.Following is the formula for calculating the Permittivity mathematically −
```
B=0H
```
H
Where −

- **B:**It is the magnetic flux density (or magnetic field)
- **H:**It is the magnetic field strength.
### Syntax

Following is the syntax of Permittivity of Free Space in Scipy −

```
scipy.constants.mu_0
```

### Example

Here is the example of calculating the Permittivity of free space by using the
**scipy.constants.epsilon_0**method −
```
from scipy.constants import epsilon_0
print(f"Permittivity of free space (): {epsilon_0} F/m")
```

Following is the output of the above program −

```
Permittivity of free space (): 8.8541878128e-12 F/m
```

## Fine-Structure Constant ()

The
**Fine-Structure Constant**is a dimensionless fundamental physical constant characterizing the strength of the electromagnetic interaction between elementary charged particles. It is a key parameter in quantum electrodynamics (QED) and reflects the coupling strength of the electromagnetic force. It is denoted by the symbol.
The value of Fine-Structure Constant () is approximately given as 0.0072973525693. In Scipy it is represented as
**alpha**.Following is the formula for calculating the Fine-Structure Constant mathematically −![Fine Structure Constant](/scipy/images/fine_struct.jpg)
Where −

- **e:**It is the elementary charge.
- **:**It is the permittivity of free space.
- **(h-bar):**It is the reduced Planck's constant.
- **c:**It is the speed of light in a vacuum.
### Syntax

Following is the syntax of Fine-Structure Constant () in Scipy −

```
scipy.constants.alpha
```

### Example

Below is the example of calculating the Fine-Structure Constant () by using the
**scipy.constants.alpha**method −
```
from scipy.constants import alpha
print(f"Fine-Structure Constant (): {alpha}")
```

Following is the output of the above program −

```
Fine-Structure Constant (): 0.0072973525693
```

## List Of Physical Constants

Here in this tutorial we discussed about only few physical constants which can be used with the help of SciPy library. There are many other such as Stefan-Boltzmann Constant (), Reduced Planck's Constant (), Magnetic Constant (_0), Electric Constant (_0) and so on.

We can get all the constants available in the
**scipy.constants**module with the help of below code −
```
import scipy
from scipy import constants
print(dir(scipy.constants))
```

Following are the constants list −

```
['Avogadro', 'Boltzmann', 'Btu', 'Btu_IT', 'Btu_th', 'ConstantWarning', 'G', 'Julian_year', 'N_A', 'Planck', 'R', 'Rydberg', 'Stefan_Boltzmann', 'Wien', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_codata', '_constants', '_obsolete_constants', 'acre', 'alpha', 'angstrom', 'arcmin', 'arcminute', 'arcsec', 'arcsecond', 'astronomical_unit',.....................................
........................................................
 'u', 'unit', 'value', 'week', 'yard', 'year', 'yobi', 'yocto', 'yotta', 'zebi', 'zepto', 'zero_Celsius', 'zetta']
```

---

## 12. SciPy - Unit Conversion Constants

*Source: [https://www.tutorialspoint.com/scipy/scipy_unit_conversion_constants.htm](https://www.tutorialspoint.com/scipy/scipy_unit_conversion_constants.htm)*

---

---

## 13. SciPy - Astronomical Constants

*Source: [https://www.tutorialspoint.com/scipy/scipy_astronomical_constants.htm](https://www.tutorialspoint.com/scipy/scipy_astronomical_constants.htm)*

---

---
[Previous](/scipy/scipy_unit_conversion_constants.htm)[Quiz](/scipy/quiz_on_scipy_astronomical_constants.htm)[Next](/scipy/scipy_fftpack.htm)
## What are Astronomical Constants?
**Astronomical constants**are fundamental and fixed values that represent key physical quantities essential for understanding and describing celestial phenomena in fields such as astrophysics, astronomy and cosmology.
These constants provide standardized measures for various properties of celestial objects and events by facilitating precise and consistent scientific calculations.

Following are the purpose and importance of Astronomical Constants −

- **Uniformity:**Astronomical constants offer a uniform basis for comparing different measurements and phenomena across the universe. This standardization is crucial for ensuring consistency in scientific research and communication.
- **Precision:**By using well-defined constants the scientists can perform calculations with high precision. This is particularly important when dealing with vast distances, massive objects and intricate interactions in space.
- **Benchmarking:**Constants such as the astronomical unit (AU) and light year serve as benchmarks for measuring and comparing the sizes, distances and scales of celestial objects and events.
## Types of Astronomical Constants in SciPy

The
**scipy.constants**module provides several key types that are critical for various astronomical calculations. Let's see them in detail with examples −
## Astronomical Unit (AU)

The
**Astronomical Unit (AU)**is a fundamental unit of distance used in astronomy to describe the average distance between the Earth and the Sun. It provides a convenient way to express and compare distances within our solar system and beyond.
In SciPy the Astronomical Unit is represented by the constant
**scipy.constants.astronomical_unit**and has a value of approximately**1.495978707  10 meters**.
### Example

Following is the example which print the value of one Astronomical Unit in meters by providing a clear reference for scientific calculations involving solar system distances −

```
from scipy.constants import astronomical_unit

# Example usage
print(f"1 Astronomical Unit (AU) = {astronomical_unit} meters")
```

Following is the output of printing the Astronomical Unit −

```
1 Astronomical Unit (AU) = 149597870700.0 meters
```

## Light Year

The
**Light Year**is a unit of distance used in astronomy to measure the vast spaces between celestial objects. It represents the distance that light travels in one Julian year in a vacuum by making it an ideal unit for expressing astronomical distances due to the immense scales involved.
In SciPy the Light Year is represented by the constant
**scipy.constants.light_year**and has a value of approximately**9.460730472  10 meters**.
### Example

Below is the example which provides the output value of one Light Year in meters which helps to contextualize distances in a standard unit of measurement −

```
from scipy.constants import light_year

# Example usage
print(f"1 Light Year = {light_year} meters")
```

Below is the output of calculating the light_year −

```
1 Light Year = 9460730472580800.0 meters
```

## Parsec

The
**Parsec**is a unit of length used in astronomy to measure large distances to astronomical objects outside the Solar System. It is defined based on the method of parallax which involves the apparent shift in position of a nearby star against the background of more distant stars as observed from Earth.
In SciPy the parsec is represented by the constant
**scipy.constants.parsec**and has a value of approximately**3.085677581  10 meters**.
### Example

Here in this example we are calculating the distance to a star that is 4 parsecs away in light-years −

```
from scipy.constants import parsec, light_year

# Distance in parsecs
distance_pc = 4

# Convert to light-years
distance_ly = distance_pc * (parsec / light_year)
print(f"Distance to the star is {distance_ly:.2f} light-years")
```

The output of parsecs is given as follows −

```
Distance to the star is 13.05 light-years
```

## Standard Gravity

In SciPy
**scipy.constants.g**represents the standard acceleration due to gravity on Earth's surface. This is a commonly used constant in physics and engineering for calculations involving gravitational force, free-fall and related phenomena. Generally the value is given as**9.80665 m/s**.
### Example

Here in this example we show how to calculate the time it takes for an object to fall from a certain height under Earth's gravity −

```
from scipy.constants import g

# Given height (in meters)
height = 100  # e.g., 100 meters

# Time to fall (using the formula: t = sqrt(2 * height / g))
fall_time = (2 * height / g) ** 0.5

print(f"Time to fall from {height} meters: {fall_time:.2f} seconds")
```

The Standard Gravity output is given as follows −

```
Time to fall from 100 meters: 4.52 seconds
```

## Julian Year

A
**Julian year**is a unit of time used in astronomy and is defined to be exactly 365.25 days. This unit is used to simplify time-related calculations in celestial mechanics such as orbital periods where precise timekeeping is crucial.
In SciPy the
**Julian year**is represented as a constant in the**scipy.constants**module and generally the value of 1 Julian year is given as**3.15576  10 seconds**.
### Example

Following is the example which is used to calculate the number of seconds in 10 Julian years with the help of
**scipy.constants.Julian_year**−
```
from scipy.constants import Julian_year

# Calculate seconds in 10 Julian years
seconds_in_10_years = 10 * Julian_year

print(f"10 Julian Years = {seconds_in_10_years:.2e} seconds")
```

The output of the above program is as follows −

```
10 Julian Years = 3.16e+08 seconds
```

## List of Astronomical Constants

Here in this tutorial we discussed about only few Astronomical constants which can be used with the help of SciPy library. There are many other such as Day, Mean Radius of Earth and so on.

We can get all the constants available in the
**scipy.constants**module with the help of below code −
```
import scipy
from scipy import constants
print(dir(scipy.constants))
```

Following are the constants list −

```
['Avogadro', 'Boltzmann', 'Btu', 'Btu_IT', 'Btu_th', 'ConstantWarning', 'G', 'Julian_year', 'N_A', 'Planck', 'R', 'Rydberg', 'Stefan_Boltzmann', 'Wien', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_codata', '_constants', '_obsolete_constants', 'acre', 'alpha', 'angstrom', 'arcmin', 'arcminute', 'arcsec', 'arcsecond', 'astronomical_unit',.....................................
........................................................
 'u', 'unit', 'value', 'week', 'yard', 'year', 'yobi', 'yocto', 'yotta', 'zebi', 'zepto', 'zero_Celsius', 'zetta']
```

---

## 14. SciPy - FFT Pack

*Source: [https://www.tutorialspoint.com/scipy/scipy_fftpack.htm](https://www.tutorialspoint.com/scipy/scipy_fftpack.htm)*

---

---
[Previous](/scipy/scipy_astronomical_constants.htm)[Quiz](/scipy/quiz_on_scipy_fftpack.htm)[Next](/scipy/scipy_discrete_fourier_transform.htm)
SciPy's
**FFTpack**is a module in SciPy which provides efficient algorithms for computing the**Fast Fourier Transform (FFT)**and its inverse. It is a part of**scipy.fftpack**submodule and offers functions to perform 1D and multi-dimensional FFTs such as fft, ifft, fft2, and ifft2.
These functions are used to transform data between the time domain and the frequency domain by enabling the analysis of frequency components in signals.

FFTpack also includes functions for computing the discrete cosine and sine transforms which are useful in signal processing and data compression. Its designed for speed and accuracy by making it suitable for various scientific and engineering applications.

## Fast Fourier Transform (FFT)

Before learning about the Scipy
**FFTpack**we should know what is**Fast Fourier Transform(FFT)**. The**Fast Fourier Transform (FFT)**is a technique for efficiently calculating the**Discrete Fourier Transform (DFT)**of a signal and its inverse. It decomposes a signal into its various frequency components by simplifying the analysis process.
The FFT is significantly quicker than the traditional DFT method especially for large datasets because it minimizes the number of calculations required. This efficiency makes it highly valuable in fields such as signal processing, image analysis and audio compression, where understanding the frequency content of signals is essential.

In Fast Fourier Transform (FFT)there are several types of transforms based on the data being analyzed and the specific requirements of the application. Below are the few important types of FFT −

- **1D FFT**− The most common type which is used for analyzing a one-dimensional signal is**1D FFT**. It transforms a sequence of complex or real numbers into its frequency components.
- **2D FFT**− This is used for processing two-dimensional data such as images. It computes the FFT separately along each dimension.
- **n-D FFT**− This type generalizes the concept to n-dimensional data which allows for the transformation of arrays with more than two dimensions.
- **Real FFT**− This type is optimized for real-valued input data which avoids redundant calculations typically associated with complex inputs.
- **Inverse FFT(IFFT)**− This transform reconstructs the original signal from its frequency components essentially reversing the FFT process.
- **Fast Cosine Transform (DCT)**− A variant of the FFT used in applications like image compression (e.g., JPEG). It only transforms real-valued data and emphasizes certain frequencies.
- **Fast Sine Transform (DST)**− This is similar to the DCT but uses sine functions which is suitable for certain boundary conditions in signal processing.
These different types of FFTs help us to do the analysis of specific data characteristics and application needs by enhancing efficiency and performance in various fields.

## Key Functions in FFTpack
**FFTpack**is a collection of routines used for calculating Discrete Fourier Transforms (DFT) using the Fast Fourier Transform (FFT) algorithm. It is part of the SciPy library and offers a variety of functions to perform FFT, inverse FFT and other Fourier-related computations efficiently. Below are the key functions of FFTpack in detail −FunctionDescriptionUse Case**scipy.fftpack.fft(x)**Computes the one-dimensional n-point discrete Fourier Transform (DFT) of a real or complex sequence using the FFT algorithm.Converts a time-domain signal into its frequency components.**scipy.fftpack.ifft(x)**Calculates the inverse of the one-dimensional DFT effectively converting frequency-domain data back into the time domain.Recovers the original signal from its frequency components.**scipy.fftpack.rfft(x)**Computes the FFT of a real-valued input sequence by returning the positive frequency terms only.Optimized for real-valued signals and avoids unnecessary calculations for negative frequencies.**scipy.fftpack.irfft(x)**Calculates the inverse of the real FFT by converting frequency-domain data back to the time domain for real-valued input sequences.Reconstructs the original sequence from its positive frequency components.**scipy.fftpack.fft2(x)**Computes the two-dimensional FFT of a real or complex array.Analyzes frequency components in both horizontal and vertical directions which are commonly used in image processing.**scipy.fftpack.ifft2(x)**Performs the inverse two-dimensional FFT by converting frequency-domain data back to the spatial domain.Recovers a 2D signal, like an image, from its frequency representation.**scipy.fftpack.fftn(x)**Computes the n-dimensional FFT for data in any number of dimensions.Performs FFT on multi-dimensional data arrays beyond just 2D.**scipy.fftpack.ifftn(x)**Performs the inverse n-dimensional FFT by converting frequency data in n dimensions back to the original spatial or time domain.Recovers n-dimensional data from its frequency representation.**scipy.fftpack.dct(x, type=2)**Computes the Discrete Cosine Transform (DCT) of an array which are used in applications like image compression such as JPEG).Concentrates energy in fewer coefficients, useful in signal and image processing.**scipy.fftpack.idct(x, type=2)**Computes the inverse Discrete Cosine Transform by converting DCT coefficients back into the time or spatial domain.Reconstructs data from its compressed form such as in image decompression.**scipy.fftpack.dst(x, type=2)**Computes the Discrete Sine Transform (DST), used in solving partial differential equations and other mathematical tasks.Suitable for signals or systems that have certain boundary conditions.**scipy.fftpack.idst(x, type=2)**Computes the inverse Discrete Sine Transform, converting DST coefficients back to the original time or spatial domain.Recovers the original signal from its DST representation.
### Example

The
**FFTpack**enables quick and efficient frequency analysis of signals. In this example a sinusoidal signal is transformed into its frequency domain representation using fft and then reconstructed using ifft.
```
from scipy.fftpack import fft, ifft
import numpy as np

# Generate a simple signal
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Compute the FFT
y_fft = fft(y)

# Inverse FFT to retrieve original signal
y_ifft = ifft(y_fft)

# Display results
print("Original Signal: ", y)
print("FFT: ", y_fft)
print("Inverse FFT: ", y_ifft.real)  # Take real part to discard tiny imaginary components
```

#### Output

Following is the output of the simple example of FFTpack of SciPy −

```
Original Signal:  [ 0.00000000e+00  6.34239197e-02  1.26592454e-01  1.89251244e-01
  2.51147987e-01  3.12033446e-01  3.71662456e-01  4.29794912e-01
  4.86196736e-01  5.40640817e-01  5.92907929e-01  6.42787610e-01
  6.90079011e-01  7.34591709e-01  7.76146464e-01  8.14575952e-01
  8.49725430e-01  8.81453363e-01  9.09631995e-01  9.34147860e-01
  9.54902241e-01  9.71811568e-01  9.84807753e-01  9.93838464e-01
  ------------------------------------------------------------
  ------------------------------------------------------------
  ------------------------------------------------------------
 -8.14575952e-01 -7.76146464e-01 -7.34591709e-01 -6.90079011e-01
 -6.42787610e-01 -5.92907929e-01 -5.40640817e-01 -4.86196736e-01
 -4.29794912e-01 -3.71662456e-01 -3.12033446e-01 -2.51147987e-01
 -1.89251244e-01 -1.26592454e-01 -6.34239197e-02  3.55271368e-17]
```

---

## 15. SciPy - Discrete Fourier Transform

*Source: [https://www.tutorialspoint.com/scipy/scipy_discrete_fourier_transform.htm](https://www.tutorialspoint.com/scipy/scipy_discrete_fourier_transform.htm)*

---

---
[Previous](/scipy/scipy_fftpack.htm)[Quiz](/scipy/quiz_on_scipy_discrete_fourier_transform.htm)[Next](/scipy/scipy_fast_fourier_transform.htm)
## Discrete Fourier Transform

The
**Discrete Fourier Transform (DFT)**in SciPy is a numerical method for converting a sequence of time-domain data points into its frequency-domain representation. It reveals the amplitude and phase of different frequency components in a signal.
SciPy implements DFT through the
**scipy.fft**module which provides efficient computation using the Fast Fourier Transform (FFT) algorithm by reducing computation time significantly.
It includes functions such as fft and ifft for 1D signals and fft2, ifft2 for 2D signals. DFT is commonly used in signal processing, image analysis and audio analysis to analyze frequency characteristics and filter signals.

The Mathematical formula to calculate the
**Discrete Fourier Transform(DFT)**of a sequence x[n] of length N is given as follows −![DFT Formula](/scipy/images/dft.jpg)
Where −

- 
X[k] represents the DFT of [] at frequency index k.

- 
x[n] is the original time-domain signal at index n.

- 
The exponential value is complex exponential representing the basis functions of different frequencies.

- 
j is the imaginary value  = 0, 1, 2,...., N-1.

### Frequency Bins

The DFT transforms the signal into N frequency bins in which each bin representing a discrete frequency component of the original signal. The index k corresponds to a frequency of k/N times of the sampling rate. For real-valued signals the DFT output exhibits symmetry with the first half representing positive frequencies and the second half negative frequencies.

## Magnitude and Phase

In the Discrete Fourier Transform (DFT) magnitude and phase are key aspects used to describe the frequency content of a signal. Let's see them in detail −

### Magnitude in DFT

The
**magnitude**indicates the strength or amplitude of each frequency present in the signal. If the DFT of a signal x[n] is represented as X[k], where X[k] is a complex number then the magnitude is given as follows −![Magnitude Formula](/scipy/images/magnitude.jpg)
where, Re(X[k]) and Im(X[k]) represent the real and imaginary parts respectively. The magnitude shows how much energy is concentrated at each frequency.

### Phase in DFT

The
**phase**provides information about the timing or horizontal shift of each frequency component. The phase of X[k] is determined as follows −![Phase Formula](/scipy/images/phase.jpg)
The phase describes how the frequency components align in time and for real signals, the phase tends to be symmetric.

#### Significance of Magnitude and Phase

The significance of magnitude and phase are given as follows −

- 
The
**magnitude**helps understand the intensity of different frequencies in the signal.
- 
The
**phase**shows the alignment of each frequency component in time.
Both components magnitude and phase are essential for a full understanding of the signals frequency content. When using the inverse DFT (IDFT) to reconstruct the time-domain signal both magnitude and phase are required for accurate signal recovery.

### Example

Here's an example which shows how to compute and visualize the magnitude and phase of a signal using the Discrete Fourier Transform (DFT) with Python and SciPy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Create a sample signal
sampling_rate = 1000  # Sampling rate in Hz
T = 1.0 / sampling_rate  # Sampling interval
t = np.linspace(0.0, 1.0, sampling_rate, endpoint=False)  # Time vector
# Create a signal with two frequencies: 50 Hz and 120 Hz
signal = 0.5 * np.sin(2.0 * np.pi * 50.0 * t) + 0.3 * np.sin(2.0 * np.pi * 120.0 * t)

# Compute the DFT of the signal
X = fft(signal)
N = len(signal)  # Number of sample points
frequencies = fftfreq(N, T)[:N//2]  # Positive frequencies

# Compute the magnitude and phase
magnitude = np.abs(X)[:N//2]  # Magnitude of DFT
phase = np.angle(X)[:N//2]    # Phase of DFT

# Plot the signal
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Time-Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the magnitude spectrum
plt.subplot(2, 1, 2)
plt.plot(frequencies, magnitude)
plt.title("Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

# Show the plots
plt.tight_layout()
plt.show()

# Print phase information
for freq, ph in zip(frequencies, phase):
    print(f"Frequency: {freq:.2f} Hz, Phase: {ph:.2f} radians")
```

Following is the output of the Magnitude and Phase of the Discrete Fourier Transform(DFT) −

```
Frequency: 0.00 Hz, Phase: -3.14 radians
Frequency: 1.00 Hz, Phase: -1.08 radians
Frequency: 2.00 Hz, Phase: -1.85 radians
Frequency: 3.00 Hz, Phase: -0.19 radians
Frequency: 4.00 Hz, Phase: 1.18 radians
----------------------------------------
----------------------------------------
----------------------------------------
Frequency: 496.00 Hz, Phase: -0.43 radians
Frequency: 497.00 Hz, Phase: -2.43 radians
Frequency: 498.00 Hz, Phase: -0.15 radians
Frequency: 499.00 Hz, Phase: 2.54 radians
```

#### Output
![Magnitude and Phase Image](/scipy/images/magnitude_phase.jpg)
## Applications of DFT

The Discrete Fourier Transform (DFT) is widely used in various fields for analyzing the frequency components of discrete signals. Below are the common applications of DFT −

- **Signal Processing:**DFT helps in analyzing the frequency content of signals by enabling tasks such as filtering, noise reduction and modulation.
- **Audio Analysis:**DFT is used in applications such as speech recognition, music analysis and audio compression by breaking down audio signals into their frequency components.
- **Image Processing:**DFT is applied to images for tasks such as image filtering, compression and reconstruction especially for frequency domain analysis.
- **Communications:**In digital communications DFT is used in OFDM systems for efficient data transmission and channel estimation.
- **Spectral Analysis:**The DFT is used for analyzing the power spectrum of signals to identify periodicities and detect patterns in data.
These applications increase the ability of DFT to transform time-domain data into the frequency domain by providing valuable insights into the signal's characteristics.

### Example

Below is the example of calculating the
**Discrete Fourier Transform**in Scipy −
```
import numpy as np
import matplotlib.pyplot as plt

# Sample signal: A combination of two sine waves (5 Hz and 20 Hz)
sampling_rate = 100  # samples per second
T = 1.0 / sampling_rate  # sample spacing
t = np.linspace(0.0, 1.0, sampling_rate, endpoint=False)  # time vector
signal = np.sin(2.0 * np.pi * 5.0 * t) + 0.5 * np.sin(2.0 * np.pi * 20.0 * t)

# Compute the DFT of the signal
N = len(signal)  # number of sample points
dft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(N, T)

# Take only the positive half of the DFT (as it is symmetric)
positive_frequencies = frequencies[:N//2]
magnitude = np.abs(dft[:N//2])

# Plot the time-domain signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Time-Domain Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plot the magnitude of the DFT (Frequency-Domain Signal)
plt.subplot(2, 1, 2)
plt.stem(positive_frequencies, magnitude, basefmt=" ")
plt.title("Magnitude Spectrum (DFT)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()

# Show plots
plt.tight_layout()
plt.show()
```

#### Output

Following is the output of the calculating the Discrete Fourier Transform(DFT) in Scipy −
![DFT Example Image](/scipy/images/dft_example.jpg)
## Inverse Discrete Fast Fourier Transform

The
**Inverse Discrete Fast Fourier Transform (IDFT or IFFT)**is the mathematical process that transforms a frequency-domain signal back into its time-domain representation. In SciPy this is implemented using the ifft function which efficiently computes the inverse of the Discrete Fourier Transform (DFT).
The IFFT is an essential tool for a wide range of signal processing applications by allowing for efficient and accurate reconstruction of signals from their frequency representations.

SciPy provides the
**scipy.fftpack.ifft**or**scipy.fft.ifft**which is recommended function for calculating the inverse FFT.
### Applications of IDFT

Here are the applications of the Inverse Discrete Fourier Transform −

- **Signal Processing:**It is used for converting frequency-domain data back to time-domain for analysis or visualization.
- **Data Compression:**It is used in conjunction with FFT to manipulate and compress signals.
- **Filtering:**Applying filters in the frequency domain and then transforming the filtered signal back to the time domain.
### Example

Following is the example of using the Inverse Discrete Fast Fourier Transform with the help of Scipy −

```
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Sample signal: A combination of two sine waves (5 Hz and 20 Hz)
sampling_rate = 100  # samples per second
T = 1.0 / sampling_rate  # sample spacing
t = np.linspace(0.0, 1.0, sampling_rate, endpoint=False)  # time vector
original_signal = np.sin(2.0 * np.pi * 5.0 * t) + 0.5 * np.sin(2.0 * np.pi * 20.0 * t)

# Compute the FFT of the signal
fft_result = fft(original_signal)

# Compute the IFFT to recover the original signal
reconstructed_signal = ifft(fft_result)

# Verify if the reconstructed signal matches the original
assert np.allclose(original_signal, reconstructed_signal.real), "Mismatch between original and reconstructed signal."

# Plot the original and reconstructed signals
plt.figure(figsize=(12, 6))

# Original Signal
plt.subplot(2, 1, 1)
plt.plot(t, original_signal, label='Original Signal')
plt.title("Original Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()

# Reconstructed Signal
plt.subplot(2, 1, 2)
plt.plot(t, reconstructed_signal.real, label='Reconstructed Signal (IFFT)', linestyle='--')
plt.title("Reconstructed Signal Using IFFT")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Following is the output of the calculating the Discrete Fourier Transform(DFT) in Scipy −
![IDFFT Example Image](/scipy/images/idfft_example.jpg)
## Fast Fourier Transforms

SciPy's Fast Fourier Transform (FFT) routines are built to efficiently compute the Discrete Fourier Transform (DFT) and its inverse for one-, two-, and n-dimensional arrays. These routines find extensive use in signal processing, audio analysis, and image processing for analyzing data in the frequency domain.
Sr.No.Function & Description1[scipy.fft](/scipy/scipy_fft_function.htm)
Computes the one-dimensional discrete Fourier Transform (DFT) using the Fast Fourier Transform (FFT) algorithm.
2[scipy.ifft](/scipy/scipy_ifft_function.htm)
Computes the inverse one-dimensional discrete Fourier Transform.
3[scipy.fft2](/scipy/scipy_fft2_function.htm)
Computes the two-dimensional discrete Fourier Transform using FFT.
4[scipy.ifft2](/scipy/scipy_ifft2_function.htm)
Computes the inverse two-dimensional discrete Fourier Transform.
5[scipy.fftn](/scipy/scipy_fftn_function.htm)
Computes the n-dimensional discrete Fourier Transform.
6[scipy.ifftn](/scipy/scipy_ifftn_function.htm)
Computes the inverse n-dimensional discrete Fourier Transform.
7[scipy.rfft](/scipy/scipy_rfft_function.htm)
Computes the one-dimensional real-input Fourier Transform.
8[scipy.irfft](/scipy/scipy_irfft_function.htm)
Computes the inverse one-dimensional real-input Fourier Transform.
9[scipy.rfft2](/scipy/scipy_rfft2_function.htm)
Computes the two-dimensional real-input Fourier Transform.
10[scipy.irfft2](/scipy/scipy_irfft2_function.htm)
Computes the inverse two-dimensional real-input Fourier Transform.
11[scipy.rfftn](/scipy/scipy_rfftn_function.htm)
Computes the n-dimensional real-input Fourier Transform.
12[scipy.irfftn](/scipy/scipy_irfftn_function.htm)
Computes the inverse n-dimensional real-input Fourier Transform.
13[scipy.hfft](/scipy/scipy_hfft_function.htm)
Computes the FFT of a real-valued signal using the Hermitian symmetry property.
14[scipy.ihfft](/scipy/scipy_ihfft_function.htm)
Computes the inverse FFT of a real-valued signal assuming Hermitian symmetry.
15[scipy.hfft2](/scipy/scipy_hfft2_function.htm)
Computes the two-dimensional FFT for real-valued signals using Hermitian symmetry.
16[scipy.ihfft2](/scipy/scipy_ihfft2_function.htm)
Computes the inverse two-dimensional FFT assuming Hermitian symmetry.
17[scipy.hfftn](/scipy/scipy_hfftn_function.htm)
Computes the n-dimensional FFT for real-valued signals using Hermitian symmetry.
18[scipy.ihfftn](/scipy/scipy_ihfftn_function.htm)
Computes the inverse n-dimensional FFT assuming Hermitian symmetry.

## Discrete Sin and Cosine Transforms

SciPy offers functions for the Discrete Cosine Transform (DCT) and Discrete Sine Transform (DST), which play a crucial role in areas like signal compression, image processing, and solving differential equations. These transformations are essential for representing and processing data efficiently.
Sr.No.Function & Description1[scipy.dct](/scipy/scipy_dct_function.htm)
Computes the Discrete Cosine Transform (DCT), commonly used in signal and image processing.
2[scipy.idct](/scipy/scipy_idct_function.htm)
Computes the inverse Discrete Cosine Transform.
3[scipy.dst](/scipy/scipy_dst_function.htm)
Computes the Discrete Sine Transform (DST), useful in solving differential equations.
4[scipy.idst](/scipy/scipy_idst_function.htm)
Computes the inverse Discrete Sine Transform.

## Helper functions

The utility functions in SciPy help optimize Fourier Transform calculations by adjusting frequency components or managing parallel workers to improve processing speed. These tools significantly boost the efficiency and adaptability of Fourier analysis tasks.
Sr.No.Function & Description1[scipy.fhtoffset](/scipy/scipy_fhtoffset_function.htm)
Computes the optimal offset for a Fast Hartley Transform (FHT).
2[scipy.next.fast.len](/scipy/scipy_next_fast_len_function.htm)
Finds the next optimal input size for efficient FFT computation.
3**scipy.set.workers**
Sets the number of parallel workers for FFT computations.
4[scipy.get.workers](/scipy/scipy_get_workers_function.htm)
Gets the current number of workers used for FFT computations.
5[scipy.fftshift](/scipy/scipy_fftshift_function.htm)
Shifts the zero-frequency component of the Fourier Transform to the center.
6[scipy.ifftshift](/scipy/scipy_ifftshift_function.htm)
Reverses the effect of fftshift, moving the zero-frequency component back to the origin.
7[scipy.fftfreq](/scipy/scipy_fftfreq_function.htm)
Computes the sample frequencies for the discrete Fourier Transform.

---

## 16. SciPy - Fast Fourier Transform

*Source: [https://www.tutorialspoint.com/scipy/scipy_fast_fourier_transform.htm](https://www.tutorialspoint.com/scipy/scipy_fast_fourier_transform.htm)*

---

---
[Previous](/scipy/scipy_discrete_fourier_transform.htm)[Quiz](/scipy/quiz_on_scipy_fast_fourier_transform.htm)[Next](/scipy/scipy_integrate.htm)
The
**Fast Fourier Transform (FFT)**in SciPy is a powerful algorithm designed to compute the**Discrete Fourier Transform (DFT)**and its inverse with high efficiency, significantly reducing the computational cost compared to the standard DFT.
This allows for the conversion of signals between the time and frequency domains by enabling various types of signal and data analysis.

## Scipy Fast Fourier Transform in SciPy

SciPy's
**scipy.fft**module offers a suite of functions for performing one-dimensional, two-dimensional and even multi-dimensional FFTs. The primary function**scipy.fft.fft**is used for computing the one-dimensional FFT of an input array while**scipy.fft.ifft**calculates the**inverse FFT**by converting frequency data back to the time domain.
For data sets that consist of real numbers we can use
**scipy.fft.rfft**and**scipy.fft.irfft**are optimized versions that only handle positive frequencies by reducing both computational time and memory usage.
For more complex data the functions such as
**scipy.fft.fftn**and**scipy.fft.ifftn**extend these capabilities to multi-dimensional arrays which is particularly useful for tasks such as image processing.
SciPys FFT functions also provide flexibility with features such as normalization, axis specification and zero-padding. Moreover
**scipy.fft.fftshift**and**scipy.fft.ifftshift**are useful for shifting the zero-frequency component to the center or moving it back to the edges by aiding in frequency spectrum visualization.
## Key Features of Fast Fourier Transform(FFT)

The features make FFT a versatile and powerful tool for various applications such as signal processing, image analysis and audio processing. Following are the key features of Fast Fourier Transform(FFT) −

- **Efficiency**− FFT dramatically reduces the computational complexity of calculating the Discrete Fourier Transform (DFT) from O(N)to O(N log N) by making it much faster for large datasets.
- **Signal Analysis**− The FFT transforms time-domain signals into their frequency-domain representation by enabling the analysis of different frequency components in the signal.
- **Inverse FFT (IFFT)**− The inverse FFT function reconstructs the original time-domain signal from its frequency-domain components.
- **Support for Real and Complex Inputs**− FFT can handle both real-valued and complex-valued data. Specialized versions such as rfft for real-valued inputs further optimize performance.
- **Multi-Dimensional FFTs**− The Functions such as fftn and ifftn allow FFT to be applied to multi-dimensional data such as images or volumetric data by making it useful in image and 3D data analysis.
- **Zero-Padding and Truncation**− Zero-padding is used to increase the resolution of the frequency spectrum while truncation can be used to reduce computational costs for signals with significant zeros.
- **Frequency Shifting**− Functions such as fftshift and ifftshift shift the zero-frequency component to the center or back by facilitating better visualization of the frequency spectrum.
- **Normalization Options**− Various normalization options ensure accurate scaling of the FFT results according to different conventions.
### Example 1

Heres a simple example of using SciPy to compute the one-dimensional Fast Fourier Transform (FFT) of a signal along with its visualization −

```
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Create a sample signal
N = 600  # Number of sample points
T = 1.0 / 800.0  # Sample spacing
x = np.linspace(0.0, N*T, N, endpoint=False)
# Create a signal composed of two different frequencies
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

# Compute the FFT
yf = fft(y)

# Compute the corresponding frequencies
xf = np.fft.fftfreq(N, T)[:N//2]

# Plot the original signal
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the FFT (magnitude spectrum)
plt.subplot(2, 1, 2)
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.title("FFT - Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
```

#### Output

Following is the output of the 1-Dimensional Fast Fourier Transform(FFT) −
![One Dimensional FFT](/scipy/images/1d_fft.jpg)
### Example 2

Heres a simple example of performing a two-dimensional Fast Fourier Transform (2D FFT) using SciPy. This example shows how to transform a 2D image or any 2D array into the frequency domain and then back into the spatial domain using the inverse 2D FFT −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

# Create a sample 2D array (image)
image = np.zeros((256, 256))
image[100:150, 100:150] = 255  # Create a white square in the middle

# Compute the 2D FFT
fft_image = fft2(image)

# Shift the zero frequency component to the center of the spectrum
fft_image_shifted = fftshift(fft_image)

# Compute the magnitude spectrum
magnitude_spectrum = np.abs(fft_image_shifted)

# Inverse 2D FFT to reconstruct the original image
reconstructed_image = ifft2(fft_image).real

# Plot the original image, magnitude spectrum, and reconstructed image
plt.figure(figsize=(12, 4))

# Original image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Magnitude spectrum
plt.subplot(1, 3, 2)
plt.title("Magnitude Spectrum")
plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
plt.axis('off')

# Reconstructed image
plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')

plt.show()
```

#### Output

Here is the output of the 2d-Dimensional Fast Fourier Transform(FFT) −
![Two Dimensional FFT](/scipy/images/2d_fft.jpg)
## Applications of FFT

The
**Fast Fourier Transform (FFT)**in SciPy has a wide range of applications across various fields due to its ability to efficiently convert time-domain signals into their frequency-domain representation. Here are some key applications of FFT −
- **Signal Processing**− FFT is used to analyze the frequency content of signals, detect patterns, filter noise and design digital filters. It's fundamental in telecommunications, audio signal processing and radar signal analysis.
- **Image Processing**− FFT helps in image filtering, enhancement and compression by transforming images to the frequency domain by enabling operations such as high-pass and low-pass filtering.
- **Audio Analysis**− FFT is used for pitch detection, sound synthesis and music analysis. It helps in extracting features such as spectral content and rhythm from audio signals.
- **Vibration Analysis**− In mechanical engineering the FFT is used to analyze vibration data from machines to identify faults and predict maintenance needs.
- **Medical Imaging**− FFT is used in MRI and other imaging techniques to reconstruct images from raw data, improve resolution and filter noise.
- **Astronomy**− FFT is employed to analyze the frequency spectrum of astronomical signals by helping in the detection and characterization of celestial objects.

---

## 17. SciPy - Integrate

*Source: [https://www.tutorialspoint.com/scipy/scipy_integrate.htm](https://www.tutorialspoint.com/scipy/scipy_integrate.htm)*

---

---
[Previous](/scipy/scipy_fast_fourier_transform.htm)[Quiz](/scipy/quiz_on_scipy_integrate.htm)[Next](/scipy/scipy_single_integration.htm)
SciPy's
**Integrate**module provides functions for performing numerical integration, allowing users to compute both definite and indefinite integrals of mathematical functions. It includes various methods suitable for different types of integration tasks such as single-variable, double and triple integrals.
The Key functions of Integrate module such as
**quad**for single integrals,**dblquad**for double integrals,**tplquad**for triple integrals and**odeint**for solving ordinary differential equations.
This module is widely used in fields like physics, engineering, statistics and economics to analyze continuous data which compute areas under curves and solve dynamic systems.

## Key Functions in SciPy Integrate Module

The
**scipy.integrate**module offers several key functions for performing numerical integration and solving ordinary differential equations (ODEs). Here are some of the most important functions −
## SciPy Integration Module

The
**scipy.integrate**module provides various methods to perform the operation of numerical integration. Following are the list of methods to understand its functionality −Sr.No.Types & Description1[integrate.quad()](/scipy/scipy_integrate_quad_method.htm)**integrate.quad()**
This method is used to perform the task of definite integrals.
2[integrate.quad_vec()](/scipy/scipy_integrate_quad_vec_method.htm)**integrate.quad_vec()**
This method is used to calculate the definite integrals of vector-value function.
3[integrate.dblquad()](/scipy/scipy_integrate_dblquad_method.htm)**integrate.dblquad()**
This is used to calculate the double numerical integration.
4[integrate.tplquad()](/scipy/scipy_integrate_tplquad_method.htm)**integrate.tplquad()**
This method is used to calculate the triple numerical integration.
5[integrate.nquad()](/scipy/scipy_integrate_nquad_method.htm)**integrate.nquad()**
This method is used to find the integration of multiple variable.
6[integrate.fixed_quad()](/scipy/scipy_integrate_fixed_quad_method.htm)**integrate.fixed_quad()**
This method operates the fixed order of Gaussian quadrature for numerical integration.
7[integrate.quadrature()](/scipy/scipy_integrate_quadrature_method.htm)**integrate.quadrature()**
This method is used to calculate the numerical integration.
8[integrate.romberg()](/scipy/scipy_integrate_romberg_method.htm)**integrate.romberg()**
This method is used to calculate the numerical integration.
9[integrate.newton_cotes()](/scipy/scipy_integrate_newton_cotes_method.htm)**integrate.newton_cotes()**
This method is used to return the weights and error coefficient for Newton-Cotes integration.
10[integrate.trapezoid()](/scipy/scipy_integrate_trapezoid_method.htm)**integrate.trapezoid()**
This method is used to find the approximate value of integral function using trapezoid rule.
11[integrate.cumulative_trapezoid()](/scipy/scipy_integrate_cumulative_trapezoid_method.htm)**integrate.cumulative_trapezoid()**
This method is used to calculate the integral from the given set of points using trapezoidal rule.
12[integrate.simpson()](/scipy/scipy_integrate_simpson_method.htm)**integrate.simpson()**
This method is used to approximate the integral of a function using Simpson rule.
13[integrate.cumulative_simpson()](/scipy/scipy_integrate_cumulative_simpson_method.htm)**integrate.cumulative_simpson()**
This method is used to calculate the coordinates at every pairs
14[integrate.romb()](/scipy/scipy_integrate_romb_method.htm)**integrate.romb()**
This method is used to perform the task of numerical or romberg integration.

## Key Features of Scipy Integrate Module

The
**scipy.integrate**module provides various features for numerical integration and solving differential equations. Below are some key features −
- **Adaptive Quadrature**− Functions such as**quad**utilize adaptive algorithms to efficiently estimate integrals by dynamically adjusting the number of evaluation points based on the function's behavior.
- **Support for Multidimensional Integration**− Functions such as**dblquad**and**tplquad**are used to enable the computation of double and triple integrals over specified regions by making it easy to handle higher-dimensional problems.
- **Integration of Ordinary Differential Equations (ODEs)**− This module includes functions such as odeint and solve_ivp, which are designed to solve initial value problems for ordinary differential equations using various methods including adaptive algorithms.
- **Flexible Input**− Users can specify custom functions for integration and differential equations by allowing for a wide variety of mathematical models to be analyzed.
- **Time Integration**− The**solve_ivp**function provides various integration methods such as Runge-Kutta and implicit methods to handle stiff and non-stiff ODEs effectively.
- **Event Handling**− The**solve_ivp**function supports event detection by allowing users to specify conditions under which the integration should stop or change behavior.
- **Robust Error Handling**− These functions provide error estimates and warnings for integration problems by helping users identify potential issues with the integration process.
- **High Performance**− The underlying algorithms are optimized for speed and accuracy by making**scipy.integrate**suitable for both simple and complex numerical integration tasks.
These features make
**scipy.integrate**module a powerful and flexible tool for numerical analysis in scientific computing and engineering applications.
## Applications

Below are the applications of the Scipy Integrate module in various Fields −

- **Physics and Engineering**− This module is used for solving problems involving areas, volumes and forces.
- **Probability and Statistics**− The scipy Integrate module is used for computing probabilities and expected values in continuous distributions.
- **Economics**− This module helps in integrating utility functions and other models over continuous ranges.
### Example

Here's an example showing how to use the
**scipy.integrate**module for both numerical integration and solving ordinary differential equations (ODEs) −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Example 1: Numerical Integration
# Define the function to be integrated
def f(x):
    return np.sin(x)

# Perform the integration of f from 0 to p
integral_result, error = integrate.quad(f, 0, np.pi)
print(f"Numerical Integration Result: {integral_result}")
print(f"Estimated Error: {error}")

# Example 2: Solving an Ordinary Differential Equation
# Define the ODE dy/dt = -2y
def model(y, t):
    return -2 * y

# Initial condition
y0 = 1
# Time points where solution is computed
t = np.linspace(0, 5, 100)

# Solve the ODE
solution = integrate.odeint(model, y0, t)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(t, solution, label='y(t) = e^{-2t}', color='blue')
plt.title('Solution of the ODE dy/dt = -2y')
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()
```

#### Output

Below is the output of the scipy Integrate module which is used to calculate the numerical Integration −
![Integrate Module](/scipy/images/integrate.jpg)
```
Numerical Integration Result: 2.0
Estimated Error: 2.220446049250313e-14
```

---

## 18. SciPy - Single Integration

*Source: [https://www.tutorialspoint.com/scipy/scipy_single_integration.htm](https://www.tutorialspoint.com/scipy/scipy_single_integration.htm)*

---

---
[Previous](/scipy/scipy_integrate.htm)[Quiz](/scipy/quiz_on_scipy_single_integration.htm)[Next](/scipy/scipy_double_integration.htm)
## Single Integration in SciPy

SciPy's
**Single Integration**can be done with the help of**quad()**function which is a powerful tool used for one-dimensional numerical integration i.e. definite integrals. It is part of the**scipy.integrate**module and is based on adaptive quadrature methods.
This function computes the integral of a given function over a specified range by returning both the integral's result and an estimate of the error. It is especially useful for smooth functions over finite limits and handles both simple and complex integrands.

The quad() function is widely used in scientific computing for tasks involving the integration of continuous functions. Mathematically the
**quad()**function evaluates definite integrals of the form as follows −![Quad Mathathematical Formula](/scipy/images/quad_math.jpg)
### Syntax

Following is the syntax of the
**scipy.integrate.quad()**function which is used to calculate the Single Integration −
```
scipy.integrate.quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False)
```

### Parameters

Here are the parameters of the
**scipy.integrate.quad()**function −
- **func**− The function to integrate. It should accept a single argument and return a scalar.
- **a**− The lower limit of integration (float).
- **b**− The upper limit of integration (float).
- **args(Optional)**− A tuple of additional arguments to pass to the function.
- **full_output(Optional)**− If the value is set to 1 then the output includes error estimates and additional information.
- **epsabs(Optional)**− Absolute error tolerance. Default value is 1.49e-08.
- **epsrel(Optional)**− Relative error tolerance. Default value is 1.49e-08.
- **limit(Optional)**− The maximum number of subdivisions. Default value is 50.
- **points(Optional)**− A list of points at which to evaluate the integrand.
- **weight(Optional)**− A string specifying the type of weight function ('cauchy', 'cosine', 'chebyshev', etc.).
- **wvar(Optional)**− A variable for the weight function.
- **wopts(Optional)**− Options for the weight function.
- **maxp1(Optional)**− Maximum number of poles for the weight function. Default value is 50.
- **limlst(Optional)**− Maximum number of limit points. Default value is 50.
- **complex_func(Optional)**− If the value is set to True then it indicates that the function is complex.
## Example of Single Integration

Following is the example of finding the Single Integration of the function
**f(x) =  sin(x)**over the interval [o,] then the result will be 2 −
```
import numpy as np
from scipy import integrate

# Define the function to be integrated
def f(x):
    return np.sin(x)

# Perform the integration from 0 to p
result, error = integrate.quad(f, 0, np.pi)

# Display the results
print(f"Integral Result: {result}")
```

Below is the output of the Single Integration of the given function −

```
Integral Result: 2.0
```

## Handling Infinite Limits

In
**scipy.integrate.quad()**function, we can handle infinite limits for improper integrals by passing**np.inf**or**-np.inf**as the limits of integration.
### Example

Here in this example the function e
is integrated from 0 to infinity which converges to 1. The**quad()**function is capable of handling such improper integrals will automatically applying appropriate techniques to compute them.
```
from scipy import integrate
import numpy as np

# Define the function to integrate
def f(x):
    return np.exp(-x)

# Perform the integration with an infinite upper limit
result, error = integrate.quad(f, 0, np.inf)

print("Integral Result:", result)  # Output: 1.0
print("Estimated Error:", error)   # Output: Estimated error in the result
```

Here is the output of handling the infinite limits in single Integration in scipy −

```
Integral Result: 1.0000000000000002
Estimated Error: 5.842606701570796e-11
```

## Error Tolerances

In
**scipy.integrate.quad()**the error tolerances are controlled by two optional parameters namely, epsabs and epsrel. These parameters define the absolute and relative error tolerance by ensuring the result's precision.
- **epsabs**− Absolute error tolerance. The algorithm stops when the absolute error estimate is below this value. Default value is 1.49e-08.
- **epsrel**− Relative error tolerance. The algorithm stops when the relative error estimate is below this value. Default value is 1.49e-08.
The integration stops when the absolute error is less than epsabs or when the relative error is less than epsrel. We can adjust these values to increase precision or speed up computation.

### Example

In this example we set both epsabs and epsrel to 1e-6 which means the integration will continue until the error estimate is smaller than these tolerances −

```
from scipy import integrate
import numpy as np

# Define the function to integrate
def f(x):
    return np.exp(-x)

# Perform the integration with an infinite upper limit
result, error = integrate.quad(f, 0, np.inf,epsabs=1e-6, epsrel=1e-6)

print("Integral Result:", result)  # Output: 1.0
print("Estimated Error:", error)   # Output: Estimated error in the result
```

Below is the output of the Error Tolerance of single integration −

```
Integral Result: 0.9999999999995187
Estimated Error: 1.842981487432398e-07
```

## Complex Functions

In
**scipy.integrate.quad()**we can integrate complex-valued functions by setting the**complex_func**parameter to**True**. This allows**quad()**function to handle functions that return complex numbers. The integration is performed separately for the real and imaginary parts of the function.
### Example

In this example we will integrate e
from 0 to  by demonstrating how to handle complex-valued integrals using SciPy's quad() function.
```
from scipy import integrate
import numpy as np

# Define a complex function
def complex_function(x):
    return np.exp(1j * x)  # e^(ix)

# Perform the integration from 0 to p with complex function handling
result, error = integrate.quad(complex_function, 0, np.pi, limit=100, complex_func=True)

print("Integral Result:", result)
print("Estimated Error:", error)
```

Here is the output of the Complex functions single integration in scipy −

```
Integral Result: (4.9225526349740854e-17+2j)
Estimated Error: (2.2102239425853306e-14+2.220446049250313e-14j)
```

Finally we can conclude the SciPys
**quad()**function is highly versatile, capable of efficiently handling smooth functions, improper integrals, complex-valued functions and even providing detailed diagnostics. Its ideal for situations where analytic solutions to integrals are difficult or impossible to obtain by making it a fundamental tool in numerical computation.

---

## 19. SciPy - Double Integration

*Source: [https://www.tutorialspoint.com/scipy/scipy_double_integration.htm](https://www.tutorialspoint.com/scipy/scipy_double_integration.htm)*

---

---
[Previous](/scipy/scipy_single_integration.htm)[Quiz](/scipy/quiz_on_scipy_double_integration.htm)[Next](/scipy/scipy_triple_integration.htm)
## Double integration in SciPy

In SciPy
**Double Integration**is performed using the**dblquad()**function which allows us to compute the integral of a function of two variables over a specified region. This function integrates over two dimensions by evaluating the inner integral first with respect to one variable and then the outer integral with respect to the second variable.
The limits of integration can be constants or functions by providing flexibility in defining the integration region. Double integration is useful for calculating areas, volumes and solving partial differential equations.

In Mathematics,
**Double Integration**is a process of integrating a function of two variables over a specified region in the xy-plane. The double integral of a function f(x,y) over the rectangular region defined by a  x  b and g(x)  y h is given as follows −![Double Integration Mathathematical Formula](/scipy/images/double_integral_formula.jpg)
The inner integral is evaluated first with respect to y while holding x constant. The result is then integrated with respect to x,over the limits a and b.

### Syntax

Following is the syntax of
**dblquad()**function in**scipy.integrate**module to calculate the Double Integration −
```
scipy.integrate.dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)
```

### Parameters

Below are the parameters of the
**dblquad()**function in**scipy.integrate**module −
- **func**− The function to be integrated. It will take two arguments (x, y).
- **a**− The lower limit of integration for the outer integral with respect to x.
- **b**− The upper limit of integration for the outer integral with respect to x.
- **gfun**− A function that returns the lower limit of integration for the inner integral with respect to y.
- **hfun**− A function that returns the upper limit of integration for the inner integral with respect to y.
- **args(optional)**− Extra arguments to pass to func.
- **epsabs**− Absolute tolerance with the default value 1.4910
- **epsrel**− Relative tolerance with the default value 1.4910
### Example of Double Integration

Following is the example of calculating the Double Integartion of the given function f(x,y) = x.y over the regions, x ranges from 0 to 1 and y ranges from 0 to 2x.

```
import scipy.integrate as integrate

# Define the function to integrate: f(x, y) = x * y
def integrand(y, x):
    return x * y

# Perform double integration using scipy.integrate.dblquad
result, error = integrate.dblquad(integrand, 0, 1, lambda x: 0, lambda x: 2*x)

# Print the result and error
print("Result of the double integration:", result)
print("Estimated error:", error)
```

#### Output

Following is the output of the Double Integration of the given function using
**dblquad()**−
```
Result of the double integration: 0.5
Estimated error: 2.2060128823111155e-14
```

## Handling Limits

In
**scipy.integrate.dblquad()**we can handle infinite limits by passing**scipy.inf**or**-scipy.inf**for the integration bounds. This allows us to perform integrals over unbounded regions. The**dblquad()**function internally handles these infinite limits using numerical techniques suited for improper integrals.
### Example

Below is the example of handling the limits of the functions f(x,y) = e
over infinite region where x ranges from 0 to infinity () and y ranges from 0 to infinity () −-y
```
import numpy as np
import scipy.integrate as integrate

# Define the function to integrate: f(x, y) = exp(-x^2 - y^2)
def integrand(y, x):
    return np.exp(-x**2 - y**2)

# Perform double integration with infinite limits
result, error = integrate.dblquad(integrand, 0, np.inf, lambda x: 0, lambda x: np.inf)

# Print the result and error
print("Result of the double integration:", result)
print("Estimated error:", error)
```

#### Output

Following is the output of handling limits when using the
**dblquad()**function −
```
Result of the double integration: 0.7853981633973343
Estimated error: 1.4647640380321503e-08
```

## Error Tolerance

In the
**scipy.integrate.dblquad()**function the error tolerance is controlled through two parameters namely, epsabs and epsrel. These parameters help us to define the accuracy of the result by setting the absolute and relative error tolerances.
The integration routine tries to balance both absolute and relative error tolerance criteria. It ensures that the result is accurate either by meeting the absolute error or relative error conditions.

### Example

Here is the example of handling the error tolerance in calculating the double integration with the help of
**dblquad()**function −
```
import numpy as np
import scipy.integrate as integrate

# Define the function to integrate: f(x, y) = exp(-x^2 - y^2)
def integrand(y, x):
    return np.exp(-x**2 - y**2)

# Perform double integration with custom error tolerances
result, error = integrate.dblquad(integrand, 0, np.inf, lambda x: 0, lambda x: np.inf, epsabs=1e-10, epsrel=1e-10)

# Print the result and error
print("Result of the double integration:", result)
print("Estimated error:", error)
```

#### Output

Following is the output of error tolerance used in
**dblquad()**function −
```
Result of the double integration: 0.785398163397448
Estimated error: 9.688368546159476e-
```

## Complex Functions

Here is the example of calculating the double integration of the complex function f(x,y) = e
+i.sin(x+y) −+y)
### Example

```
import numpy as np
import scipy.integrate as integrate

# Define the real part of the function: exp(-x^2 - y^2)
def real_part(y, x):
    return np.exp(-x**2 - y**2)

# Define the imaginary part of the function: sin(x + y)
def imag_part(y, x):
    return np.sin(x + y)

# Define finite bounds to replace infinity, for instance (0, 100)
finite_bound = 100

# Perform double integration for real part
real_result, real_error = integrate.dblquad(real_part, 0, finite_bound, lambda x: 0, lambda x: finite_bound)

# Perform double integration for imaginary part
imag_result, imag_error = integrate.dblquad(imag_part, 0, finite_bound, lambda x: 0, lambda x: finite_bound)

# Combine the real and imaginary parts into a complex number
complex_result = real_result + 1j * imag_result

# Print the result
print("Result of the double integration (complex):", complex_result)
print("Estimated error (real part):", real_error)
print("Estimated error (imaginary part):", imag_error)
```

#### Output

Following is the output of calculating the double integration of a complex function using
**dblquad()**function −
```
Result of the double integration (complex): (0.7853981633971309-0.13943398500584472j)
Estimated error (real part): 1.1078396381028211e-08
Estimated error (imaginary part): 1.4892850109719667e-08
```

---

## 20. SciPy - Triple Integration

*Source: [https://www.tutorialspoint.com/scipy/scipy_triple_integration.htm](https://www.tutorialspoint.com/scipy/scipy_triple_integration.htm)*

---

---
[Previous](/scipy/scipy_double_integration.htm)[Quiz](/scipy/quiz_on_scipy_triple_integration.htm)[Next](/scipy/scipy_multiple_integration.htm)
## Triple Integration in SciPy

In SciPy
**Triple Integration**can be performed using the**tplquad()**function from the**scipy.integrate**module. This function allows us to compute the integral of a function of three variables over a specified region in three-dimensional space. The integral is computed iteratively for each variable following the specified bounds.
The triple integral of a function
*f(x, y, z)*over a three-dimensional region defined by the limits can be expressed mathematically given as follows −![Triple Integration Mathathematical Formula](/scipy/images/triple_integral_formula.jpg)
Where −

- **V**is the three-dimensional region over which the integration is performed.
- 
dv is the volume element which is in Cartesian coordinates is expressed as
**dv = dxdydz**and**f(x, y, z)**is the function being integrated.
The limits of integration can vary based on the region v. If the limits are constant then the triple integral can be written as follows −
![Triple Integral Constant Limits](/scipy/images/triple_limits_constant.jpg)
If the limits are functions of one another then the triple integral can be given as −
![Triple Integral function Limits](/scipy/images/triple_limits_other.jpg)
### Syntax

Following is the syntax for the
**tplquad()**function of**scipy.integrate**module can be given as follows −
```
scipy.integrate.tplquad(func, a, b, gfun, hfun, vfun, wfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)
```

### Parameters

Here are the parameters for the
**tplquad()**function of**scipy.integrate**module −
- **func**− The function to be integrated which takes three arguments (z, y, x).
- **a**− The lower limit of integration for the outer integral (x).
- **b**− The upper limit of integration for the outer integral (x).
- **gfun**− A function that returns the lower limit of integration for the middle integral (y) as a function of x.
- **hfun**− A function that returns the upper limit of integration for the middle integral (y) as a function of x.
- **vfun**− A function that returns the lower limit of integration for the inner integral (z) as a function of x and y.
- **wfun**− A function that returns the upper limit of integration for the inner integral (z) as a function of x and y.
- **args (optional)**− Extra arguments to pass to func.
- **epsabs**− Absolute tolerance for the integration where default value is 1.49e-08.
- **epsrel**− Relative tolerance for the integration where default value is 1.49e-08.
### Example of Triple Integration

Below is the example of calculating the triple integral of the function
*f(x, y, z) = x * y * z*over the region defined by
- *0  x  1*
- *0  y  x*
- *0  z  y*
```
import scipy.integrate as integrate

# Define the integrand function: f(x, y, z) = x * y * z
def integrand(z, y, x):
    return x * y * z

# Perform triple integration using scipy.integrate.tplquad
result, error = integrate.tplquad(
   integrand, 
   0, 1,              # Limits for x
   lambda x: 0,       # Lower limit for y
   lambda x: x,       # Upper limit for y
   lambda x, y: 0,    # Lower limit for z
   lambda x, y: y     # Upper limit for z
)
# Print the result and estimated error
print("Result of the triple integration:", result)
print("Estimated error:", error)
```

#### Output

Following is the output of the triple integration done using the function
**tplquad()**of**scipy.integrate**−
```
Result of the triple integration: 0.020833333333333336
Estimated error: 5.4672862306750106e-15
```

## Handling Infinite Limits

We can also handle infinite limits using
**scipy.inf**or**-scipy.inf**. This allows us to compute integrals over unbounded regions effectively.
### Example

Below is an example of integrating the function
*f(x, y, z) = e*over the region where+ y+ z)*x, y, z*range from 0 to infinity:
```
import numpy as np
import scipy.integrate as integrate

# Define the function to integrate: f(x, y, z) = exp(-(x^2 + y^2 + z^2))
def integrand(z, y, x):
    return np.exp(-(x**2 + y**2 + z**2))

# Perform triple integration with infinite limits
result, error = integrate.tplquad(integrand, 
                                   0, np.inf,               # Limits for x
                                   lambda x: 0,             # Lower limit for y
                                   lambda x: np.inf,        # Upper limit for y
                                   lambda x, y: 0,          # Lower limit for z
                                   lambda x, y: np.inf)     # Upper limit for z

# Print the result and estimated error
print("Result of the triple integration:", result)
print("Estimated error:", error)
```

#### Output

Following is the output of handling the Infinite limits when calculating the triple integration −

```
Result of the triple integration: 0.6960409996034802
Estimated error: 1.4884526702265109e-08
```

## Error Tolerance

The
**tplquad()**function allows us to control error tolerance with the**epsabs**and**epsrel**parameters. These parameters define the desired accuracy of the integration results.
### Example

Heres an example of calculating the triple integral with specified error tolerances −

```
import numpy as np
import scipy.integrate as integrate

# Define the function to integrate: f(x, y, z) = exp(-(x^2 + y^2 + z^2))
def integrand(z, y, x):
    return np.exp(-(x**2 + y**2 + z**2))

# Perform triple integration with custom error tolerances
result, error = integrate.tplquad(integrand, 
                                   0, np.inf,               # Limits for x
                                   lambda x: 0,             # Lower limit for y
                                   lambda x: np.inf,        # Upper limit for y
                                   lambda x, y: 0,          # Lower limit for z
                                   lambda x, y: np.inf,     # Upper limit for z
                                   epsabs=1e-10, epsrel=1e-10)  # Custom tolerances

# Print the result and estimated error
print("Result of the triple integration:", result)
print("Estimated error:", error)
```

#### Output

Following is the output of handling the error tolerance while performing the triple integration −

```
Result of the triple integration: 0.6960409996039614
Estimated error: 9.998852642787021e-11
```

## Complex Functions

When dealing with complex functions we have to split the real and imaginary parts and perform separate integrals for each. This is similar to how double integrals are handled.

### Example

Heres an example of calculating the triple integral of a complex function
*f(x, y, z) = e*−+ i.sin(x + y + z)+ y+ z)
```
import numpy as np
from scipy.integrate import nquad

# Define the real part of the function
def real_function(x, y, z):
    return np.exp(-(x**2 + y**2 + z**2))

# Define the imaginary part of the function
def imaginary_function(x, y, z):
    return np.sin(x + y + z)

# Define the limits for each variable (0 to 1 for this example)
limits = [[0, 1], [0, 1], [0, 1]]  # Integration limits for x, y, z

# Perform the triple integration for the real part
real_result, real_error = nquad(real_function, limits)

# Perform the triple integration for the imaginary part
imaginary_result, imaginary_error = nquad(imaginary_function, limits)

# Combine the results into a complex number
result = real_result + 1j * imaginary_result

# Print the results
print(f"Real Part Integral Result: {real_result}, Error Estimate: {real_error}")
print(f"Imaginary Part Integral Result: {imaginary_result}, Error Estimate: {imaginary_error}")
print(f"Combined Integral Result: {result}")
```

#### Output

Here is the output of calculating the triple integration of complex functions using
**tplquad()**in scipy −
```
Real Part Integral Result: 0.4165383858866382, Error Estimate: 8.291335287314424e-15
Imaginary Part Integral Result: 0.8793549306454008, Error Estimate: 1.0645376503904486e-14
Combined Integral Result: (0.4165383858866382+0.8793549306454008j)
```

Finally we can say the
**tplquad()**function in SciPy provides a powerful and flexible tool for performing triple integrals over various domains.
By specifying functions for the limits and integrating a wide range of functions we can efficiently compute complex integrals with high accuracy.

---

## 21. SciPy - Multiple Integration

*Source: [https://www.tutorialspoint.com/scipy/scipy_multiple_integration.htm](https://www.tutorialspoint.com/scipy/scipy_multiple_integration.htm)*

---

---
[Previous](/scipy/scipy_triple_integration.htm)[Quiz](/scipy/quiz_on_scipy_multiple_integration.htm)[Next](/scipy/scipy_differential_equations.htm)
## Multiple Integration in SciPy
**Multiple integration**in SciPy is used to calculating the integral of a function over more than one variable i.e., double, triple or higher-dimensional integrals. The**scipy.integrate**module which provides functions such as dblquad for double integration, tplquad for triple integration, quad for single integration and nquad for integration over multiple variables.
In multiple integration we can define the function to be integrated along with the integration limits for each variable and also any additional parameters. SciPy handles both finite and infinite limits and can integrate complex functions over specified ranges by providing an efficient solution for high-dimensional problems such as physics simulations and probability distributions.

Mathematical Formula for the n-dimensional Integration over a region of D −
![Multiple Integration Mathathematical Formula](/scipy/images/multiple_integral_formula.jpg)
The limits of the multiple integration depends on the specific region D in n-dimensional space.

### Syntax

Following is the syntax of the function
**scipy.integrate.nquad()**which is used to calculate the multiple integration −
```
scipy.integrate.nquad(func, ranges, args=None, opts=None, full_output=False)
```

### Parameters

Here are the parameters for the
**nquad()**function of**scipy.integrate**module −
- **func**− The function to be integrate. It should take the variables being integrated as individual arguments.
- **ranges**− A list of limits for each variable's integration given as tuples (a, b) where a and b are the bounds. Use -inf or inf for infinite limits in improper integrals.
- **args(optional)**− Additional arguments to pass to func.
- **opts(optional)**− Integration options for each dimension such as tolerance settings or other parameters for the integrator.
- **full_output(optional)**− If this parameter is True then the extra information about the integration process is returned in a dictionary.
## Multiple Integration

This example shows how to use
**nquad()**to perform double integration of a two-variable function over a specified rectangular region. We can modify the integrand and limits as needed for different applications −
### Example

```
import numpy as np
from scipy.integrate import nquad

# Define the function to integrate
def integrand(x, y):
    return np.sin(x) + np.cos(y)

# Define the limits for x and y
# x goes from 0 to pi
# y goes from 0 to pi/2
ranges = [[0, np.pi], [0, np.pi / 2]]

# Perform the double integration
result, error = nquad(integrand, ranges)

# Output the results
print(f"Double Integral Result: {result}")
print(f"Error Estimate: {error}")
```

#### Output

Following is the output of the calculating the dobule integration with the help of
**nquad()**function −
```
Double Integral Result: 6.283185307179586
Error Estimate: 6.975736996017264e-14
```

## Double Integration

This example shows how to use
**nquad()**to perform double integration of a two-variable function over a specified rectangular region. We can modify the integrand and limits as needed for different applications −
### Example

```
import numpy as np
from scipy.integrate import nquad

# Define the function to integrate
def integrand(x, y):
    return np.sin(x) + np.cos(y)

# Define the limits for x and y
# x goes from 0 to pi
# y goes from 0 to pi/2
ranges = [[0, np.pi], [0, np.pi / 2]]

# Perform the double integration
result, error = nquad(integrand, ranges)

# Output the results
print(f"Double Integral Result: {result}")
print(f"Error Estimate: {error}")
```

#### Output

Following is the output of the calculating the double integration with the help of
**nquad()**function −
```
Double Integral Result: 6.283185307179586
Error Estimate: 6.975736996017264e-14
```

## Triple Integration

Here's an example of triple integration using the
**nquad()**function from the**scipy.integrate**module in Python. This example will calculate the integral of the function f(x,y,z) = x+y+zover a specified range for each variable −
### Example

```
from scipy.integrate import nquad

# Define the function to integrate
def func(x, y, z):
    return x**2 + y**2 + z**2

# Specify the ranges for x, y, and z
ranges = [[0, 1], [0, 1], [0, 1]]

# Perform the triple integration
result, error = nquad(func, ranges)

# Output the result
print("Result of the triple integration:", result)
print("Estimated error:", error)
```

#### Output

Here is the output of the calculating the multiple integration with the help of
**nquad()**function −
```
Result of the triple integration: 1.0
Estimated error: 2.5808878251226036e-14
```

## Key Features of the nquad() Function

Following are the key features of nquad() of scipy.integrate module −

- 
The
**nquad()**function handles multiple Integrations of any dimension.
- 
This function supports variable limits for integration.

- 
It can pass additional arguments to the integrand function.

- 
This function perform customizable integration settings via the opts parameter.

---

## 22. SciPy - Differential Equations

*Source: [https://www.tutorialspoint.com/scipy/scipy_differential_equations.htm](https://www.tutorialspoint.com/scipy/scipy_differential_equations.htm)*

---

---
[Previous](/scipy/scipy_multiple_integration.htm)[Quiz](/scipy/quiz_on_scipy_differential_equations.htm)[Next](/scipy/scipy_integration_of_stochastic_differential_equations.htm)
SciPy's
**Differential Equations**module provides tools for solving ordinary differential equations (ODEs) and partial differential equations (PDEs). This module includes various functions such as**scipy.integrate.odeint()**and**scipy.integrate.solve_ivp()**which allow users to integrate ODEs using methods such as Runge-Kutta and BDF (Backward Differentiation Formulas).
These functions enable the specification of initial conditions and the handling of time-dependent problems. SciPy also supports the integration of systems of equations and provides options for adaptive step size control by making it a powerful resource for scientists and engineers modeling dynamic systems and analyzing their behaviors over time.

## Key Features of Scipy Differential Equations

SciPy offers several key features for solving differential equations by making it a powerful tool for scientific and engineering applications. Below are some of the main features, let's see them one by one −

- **ODE Solvers**− SciPy provides a range of ordinary differential equation (ODE) solvers through the**scipy.integrate.()**and**scipy.integrate.solve_ivp()**functions. These solvers support various methods such as Runge-Kutta and can handle stiff and non-stiff problems.
- **Event Handling**− The**solve_ivp()**function includes options to detect events such as reaching a specific threshold by allowing users to stop integration when certain conditions are met.
- **Integration of Initial Value Problems**− The solvers can handle initial value problems for systems of ODEs by making it easy to integrate complex dynamical systems.
- **Support for Time-dependent Parameters**− SciPy allows the incorporation of time-dependent parameters in the equations by enabling dynamic modeling of systems where parameters change over time.
- **Boundary Value Problems**− The**scipy.integrate.solve_bvp()**function can solve boundary value problems by providing a framework for problems with conditions specified at more than one point.
- **Numerical Integration**− In addition to solving differential equations, SciPy includes functions for numerical integration by enabling users to compute solutions to integrals related to differential equations.
- **Rich Documentation and Examples**− SciPys documentation includes extensive examples and explanations which makes it easier for users to understand how to apply its functions effectively.
- **Interoperability with NumPy**− SciPy is built on NumPy by ensuring seamless integration with arrays and numerical operations which is essential for efficient computation in differential equations.
These features make SciPy a comprehensive toolkit for tackling various differential equation problems in scientific computing.

## Applications of Scipy Differential Equations

SciPy's capabilities for solving differential equations have a wide range of applications across various fields. Here are some notable applications −

- **Physics**− Modeling physical systems such as harmonic oscillators, pendulums and wave equations. Differential equations describe the dynamics of particles and waves by enabling predictions about their behavior over time.
- **Engineering**− Analyzing systems in control theory, structural dynamics and fluid dynamics. Engineers use differential equations to model and control systems, optimize designs and ensure stability.
- **Biology**− This is used to simulate population dynamics, disease spread and biochemical reactions. Differential equations can model the growth of populations, the spread of infectious diseases or the rates of reaction in biological systems.
- **Finance**− The Differential Eqautions are used in option pricing and risk assessment often involve stochastic differential equations. These equations help model the behavior of financial instruments and assess risk in uncertain environments.
- **Chemistry**− This is used in modeling reaction kinetics and thermodynamics. Differential equations can describe the rate of chemical reactions and the changes in concentration of reactants and products over time.
- **Environmental Science**− Simulating environmental processes such as pollutant dispersion, resource management and climate modeling. Differential equations are used to predict how substances spread in ecosystems or how resources are utilized.
- **Robotics and Control Systems**− Used  in designing and analyzing robotic systems and control strategies. Differential equations describe the dynamics of robotic movements and are essential in creating feedback control systems.
- **Neuroscience**− Differential equations are used to simulate the electrical activity of neurons and the interactions between different brain regions.
- **Machine Learning**− The differential equations can be used in some algorithms especially in reinforcement learning, rely on differential equations to model continuous state transitions or dynamics of systems being learned.
- **Astrophysics**− These are used in simulating celestial mechanics and stellar dynamics. Differential equations help model the motion of planets, stars and galaxies as well as the evolution of stellar objects.
These applications demonstrate the versatility of differential equations in modeling and solving real-world problems across diverse domains.

### Example

Heres an example of how to solve a simple first-order ODE using the solve_ivp() function in scipy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODE as a function
def ode_function(t, y):
    return -2 * y

# Initial conditions
y0 = [1]  # Initial value of y
t_span = (0, 5)  # Time interval for the solution
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Points at which to store the solution

# Solve the ODE
solution = solve_ivp(ode_function, t_span, y0, t_eval=t_eval)

# Plot the results
plt.plot(solution.t, solution.y[0], label='y(t) = e^{-2t}')
plt.title('Solution of ODE')
plt.xlabel('Time t')
plt.ylabel('Function y')
plt.axhline(0, color='gray', lw=0.5, ls='--')
plt.axvline(0, color='gray', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()
```

#### Output

Following is the output of the calculating the simple first order ODE using solve_ivp() function in scipy −
![Differential Equation ODC Example](/scipy/images/odc_example.jpg)
We can say sciPys capabilities for solving differential equations make it a powerful tool for scientists and engineers by enabling them to model and analyze dynamic systems effectively. Whether working with simple first-order ODEs or more complex systems SciPy provides the necessary functions to find numerical solutions and visualize results.

---

## 23. SciPy - Integration of Stochastic Differential Equations

*Source: [https://www.tutorialspoint.com/scipy/scipy_integration_of_stochastic_differential_equations.htm](https://www.tutorialspoint.com/scipy/scipy_integration_of_stochastic_differential_equations.htm)*

---

---
[Previous](/scipy/scipy_differential_equations.htm)[Quiz](/scipy/quiz_on_scipy_integration_of_stochastic_differential_equations.htm)[Next](/scipy/scipy_integration_of_ordinary_differential_equations.htm)
The Integration of
**Stochastic Differential Equations (SDEs)**in SciPy is the process of solving differential equations that describe systems influenced by both deterministic and random components.
These equations are used to model various phenomena where uncertainty or noise is a fundamental characteristic, such as in finance, physics, biology and engineering.

Mathematically the
**Stochastic Differential Equations (SDEs)**can be given as follows −
```
dXt = f(Xt,t)dt + g(Xt,t)dWt
```
= f(X,t)dt + g(X,t)dW
Where −

- 
X
is the state variable.
- 
f(X
,t) represents the drift term i.e. deterministic part.
- 
g(X
,t) is the diffusion term i.e. stochastic part.
- 
g(X
,t) is the diffusion term i.e. stochastic part.
- 
W
is a Wiener process or Brownian motion.
- 
dt is a small time increment.

## Key Components of SDEs

Following are the key components of the Stochastic Differential Equations(SDEs) −

- **Stochastic Process**− A stochastic process introduces randomness into the system which often represented by Brownian motion, also called Wiener process. Brownian motion is a continuous-time random process characterized by unpredictable, continuous fluctuations.
- **Drift Term**− The deterministic part of an SDE which often denoted as f(X,t) represents the expected or average behavior of the system over time. This is similar to the rate of change in ordinary differential equations (ODEs).
- **Diffusion Term**− The stochastic part f(X,t) modulates the randomness in the system. This term is multiplied by the differential of Brownian motion dWwhich represents the random shocks that occur in the system.
## Brownian Motion and Wiener Process
**Brownian Motion**and the**Wiener Process**are foundational concepts in the study of stochastic processes and are crucial for modeling randomness in various scientific fields such as finance, physics and engineering. Let's see about these in detail −**Brownian motion**is also know as**Wiener Process**which describes the random movement of particles suspended in a fluid which was first observed by the botanist Robert Brown. Mathematically it has been formalized to represent continuous-time stochastic processes.
Following are the key properties of brownian motion −

- **Start at Zero**− Brownian motion starts at zero, i.e.,
```
B(0) = 0
```

This property establishes the starting point for the process and provides a reference for subsequent movement.

- **Independent Increments**− The increments of Brownian motion over non-overlapping intervals are independent.
```
B(t+s)-B(t)
```

It is independent of any past values B(u) for u < t which means that the future movement does not depend on the past trajectory of the process.

- **Normally Distributed Increments**− The increments of Brownian motion are normally distributed. Specifically for any value s > 0.
```
B(t+s)-B(t)N(0,s)
```

This indicates that the difference between the values of Brownian motion at two times t and t+s follows a normal distribution with a mean of 0 and variance equal to the length of the time interval s.

- **Continuous Paths**− The paths of Brownian motion are continuous which means that as time progresses, the function describing the motion does not have any jumps or discontinuities. However these paths are almost surely nowhere differentiable which signifies that they exhibit highly erratic behavior and cannot be described by a smooth function.
The mathematical representation of Brownian motion can be given as follows −
![Brownian Formula](/scipy/images/brownian_formula.jpg)
where B(0)=0 and dB(s) represents the infinitesimal increment of the Brownian motion.

### Example

Below is an example that shows how to generate and plot a simple Brownian motion path using SciPy and Matplotlib −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
T = 1.0          # Total time
N = 1000        # Number of steps
dt = T/N        # Time increment
t = np.linspace(0, T, N+1)  # Time array

# Generate increments of the Wiener process
# dW is normally distributed with mean 0 and variance dt
dW = norm.rvs(loc=0.0, scale=np.sqrt(dt), size=N)  # Using scipy to generate normal increments
# Create the Brownian motion path by taking the cumulative sum of the increments
W = np.concatenate(([0], np.cumsum(dW)))

# Plotting the Brownian motion
plt.figure(figsize=(10, 6))
plt.plot(t, W, label='Brownian Motion Path', color='blue')
plt.title('Simulated Brownian Motion')
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.grid()
plt.legend()
plt.show()
```

#### Output

Following is the output of the Brownian motion −
![Brownian Example](/scipy/images/brownian_example.jpg)
### Differences from Ordinary Differential Equations (ODEs)

In
**Ordinary Differential Equations (ODEs)**the system evolves deterministically which means future states depend only on current values and time. However SDEs account for random influences by making the evolution of the system uncertain. This introduces more complexity in solving SDEs because the solutions are not deterministic paths but probability distributions over possible paths.
## Integration of SDEs

The traditional methods used for integrating ODEs do not directly apply to SDEs due to the random component. There are specialized numerical techniques which are mentioned as below −

### Euler-Maruyama Method

This is a simple and first-order numerical approximation technique for solving SDEs. It generalizes the Euler method for ODEs by including a random term.

For an SDE of the form

```
dXt = f(Xt,t)dt + g(Xt,t)dWt
```
= f(X,t)dt + g(X,t)dW
which is Euler-Maruyama method approximates the solution as mentioned below −

```
Xt+tXt+f(Xt,t)t+g(Xt,t)tWt
```
X+f(X,t)+g(X,t)W
Where, W
N(0,)represents a random normal variable scaled by.
### Milstein Method

This method is an enhancement over the Euler-Maruyama method it improves accuracy by accounting for the derivative of the diffusion term.

The Milstein method for an SDE is given as follows −
![Milstein Formula](/scipy/images/milstein_formula.jpg)
where g(X
,t) is the derivative of the diffusion term with respect to X.
### Applications of SDEs

Below are the applications of the
**Stochastic Differential Equations (SDEs)**which are used in many fields to model systems where randomness plays a key role −
- **Finance**− In modeling stock prices such as Black-Scholes model for option pricing.
- **Physics**− For systems influenced by thermal noise such as particle motion in fluids.
- **Biology**− In population dynamics where birth and death processes involve randomness.
- **Control Theory**− For systems with random disturbances affecting control inputs.

---

## 24. SciPy - Integration of Ordinary Differential Equations

*Source: [https://www.tutorialspoint.com/scipy/scipy_integration_of_ordinary_differential_equations.htm](https://www.tutorialspoint.com/scipy/scipy_integration_of_ordinary_differential_equations.htm)*

---

---

## 25. SciPy - Discontinous Functions

*Source: [https://www.tutorialspoint.com/scipy/scipy_discontinuous_functions.htm](https://www.tutorialspoint.com/scipy/scipy_discontinuous_functions.htm)*

---

---
[Previous](/scipy/scipy_integration_of_ordinary_differential_equations.htm)[Quiz](/scipy/quiz_on_scipy_discontinuous_functions.htm)[Next](/scipy/scipy_oscillatory_functions.htm)
## What are Discontinuous Functions?

A
**Discontinuous**function is a type of mathematical function that does not have a well-defined limit at certain points in its domain which means that at least one point in the functions domain leads to a sudden jump or break in its value.
In the view of numerical analysis and scientific computing with libraries such as SciPy, handling discontinuous functions is important for accurate calculations especially in integration and optimization tasks.

### Characteristics of Discontinuous Functions
**Discontinuous**functions in SciPy exhibit several key characteristics that can complicate numerical computations especially integration.
Understanding these characteristics is crucial for effectively integrating and working with discontinuous functions in SciPy. Here are the features of the Discontinuous Functions −

- **Abrupt Changes**− Discontinuous functions have sudden jumps or breaks in value at specific points by making them non-smooth. This can lead to difficulties in defining their integrals.
- **Undefined Behavior**− At points of discontinuity this function may be undefined or infinite which can result in integration errors or warnings.
- **Non-differentiability**− Discontinuous functions are typically not differentiable at their points of discontinuity which affect methods that rely on derivative calculations.
- **Piecewise Definition**− Many discontinuous functions are defined piece wise with different expressions for different intervals. Handling these requires careful setup in integration routines.
- **Integration Challenges**− Traditional numerical integration methods may struggle with discontinuous functions which requires adaptive techniques or interval splitting to achieve accurate results.
- **Oscillation Around Discontinuities**− In some cases a discontinuous function may oscillate near its discontinuities by complicating the integration further due to rapid changes in value.
- **Special Handling Required**− Functions with known discontinuities often need special treatment such as using**quad()**with specified integration limits or splitting the intervals to avoid integrating across the discontinuity.
- **Sensitivity to Tolerance Settings**− The numerical results can be sensitive to the error tolerance parameters such as epsabs, epsrel which set in integration functions with necessary adjustments for accuracy.
## Handling Discontinuous Functions in SciPy

When dealing with discontinuous functions in SciPy it's important to ensure accurate integration and other numerical methods. Here are the important steps how to handle them effectively −

- **Identifying Discontinuities**− Identifying points of discontinuity is essential. This can be done by analyzing the function mathematically or graphically. If the discontinuities are known beforehand they can be incorporated into numerical methods.
- **Integration with SciPy**− The**scipy.integrate**module provides several tools for integrating functions which includes those with discontinuities. The key function for integration is**quad()**which can handle a variety of cases including discontinuities.
When we use the points parameter in the
**quad()**function then the points argument can specify known points of discontinuity by allowing the integrator to adjust its calculations accordingly.
If the function has multiple discontinuities then we can integrate the function in segments to avoid regions where the function is not continuous.

- **Optimization with SciPy**− When optimizing functions with discontinuities then the SciPy optimization module can be used but care must be taken. Some optimization algorithms might struggle with discontinuities so using methods like minimize_scalar with appropriate bounds is advisable.
### Simple Discontinuous Function

Following is the example of the simple Discontinuous Function −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define a discontinuous function
def discontinuous_function(x):
    return 1 if x < 0 else 0

# Plot the function
x = np.linspace(-1, 1, 100)
y = [discontinuous_function(val) for val in x]

plt.plot(x, y, label='Discontinuous Function')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.title("Discontinuous Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()

# Integrate the function
integral1, error1 = quad(discontinuous_function, -1, 0)
integral2, error2 = quad(discontinuous_function, 0, 1)
total_integral = integral1 + integral2
print(f"Integral from -1 to 0: {integral1}, Integral from 0 to 1: {integral2}, Total Integral: {total_integral}")
```

#### Output

Here is the output of the simple Discontinuous Function in scipy −
![Discontinuous Function Output](/scipy/images/discontinuous_function_ex.jpg)
## Different Discontinuous Functions

Following are the different types of discontinuous functions that we can use in numerical analysis and each with unique properties that can affect calculations and integrations in SciPy −
Function NameDescriptionStep FunctionThis function jumps from one value to another at a specific point. It is often used to model sudden changes.Piecewise FunctionDefined differently over different intervals with leading to discontinuities at transition points. Common in mathematical modeling.Dirichlet FunctionThis function is 1 at rational numbers and 0 at irrational numbers by making it highly discontinuous across its domain.Heaviside Step FunctionA variation of the step function used in control systems. It is 0 for negative inputs and 1 for positive inputs.Sine Function with JumpA modified sine function that introduces a jump at a specific point by demonstrating discontinuity in periodic functions.Removable DiscontinuityThis function is not defined at a specific point but can be defined to make it continuous. It illustrates the concept of removable discontinuity.
## Step function

A step function is a piecewise constant function that takes constant values on intervals and has discontinuities at certain points where it
**jumps**from one value to another. This behavior makes it useful for modeling systems that experience sudden changes or transitions. We can use the step function with the help of**scipy.interpolate**module.
Following are the characteristics of the step function −

- **Constant Value**− This function remains constant within specified intervals.
- **Discontinuity**− This function jump discontinuities at the points where it changes value.
- **Common Applications**− Step functions are often used in control systems, signal processing and mathematical modeling to represent thresholds or on/off conditions.
The Mathematical Representation of the simple Step function can be given as follows −
![Step Function Equation](/scipy/images/step_function_equa.jpg)
### Example

In this example we will define a step function that takes on different constant values in specified intervals −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the step function data points
x_points = [0, 1, 2, 3]
y_points = [1, 2, 1, 0]  # The function values at the step points

# Create a step function using scipy.interpolate
step_function = interp1d(x_points, y_points, kind='previous', fill_value='extrapolate')

# Generate x values for plotting
x_values = np.linspace(-1, 4, 100)
y_values = step_function(x_values)

# Plotting the step function
plt.plot(x_values, y_values, label='Step Function', color='blue')
plt.title('Step Function using SciPy')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
```

#### Output

Following is the output of the step function created in scipy −
![Step Function Output](/scipy/images/step_function_ex.jpg)
## Piecewise Function

A piecewise function is defined by multiple sub-functions with each applying to a certain interval of the input variable. These functions may have different rules in different intervals which leads to discontinuities at the boundaries between intervals.

Characteristics of the piecewise function are given as follows −

- **Multiple Definitions**− The function can take different forms based on the input interval.
- **Discontinuity**− It may exhibit discontinuities at the points where the definition changes.
- **Applications**− Used in various mathematical modeling scenarios including economics and engineering.
The Mathematical Representation of a piecewise function can be given as follows −
![Piecewise Function Equation](/scipy/images/piecewise_function_equa.jpg)
### Example

In this example, we will define a piecewise function with different expressions based on intervals using
**SciPy**−
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the piecewise function data points
x_points = [-2, 0, 1, 3]  # Interval boundaries
y_points = [0, 1, 0, -1]  # Function values at the specified points

# Create a piecewise function using scipy.interpolate
piecewise_function = interp1d(x_points, y_points, kind='linear', fill_value='extrapolate')

# Generate x values for plotting
x_values = np.linspace(-3, 4, 100)
y_values = piecewise_function(x_values)

# Plotting the piecewise function
plt.plot(x_values, y_values, label='Piecewise Function', color='green')
plt.title('Piecewise Function using SciPy')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
```

#### Output

Following is the output of the piecewise function created in SciPy −
![Piecewise Function Output](/scipy/images/piecewise_function_ex.jpg)
## Dirichlet Function

The Dirichlet function is defined to be 1 at rational numbers and 0 at irrational numbers. This function is not Riemann integrable by making it an interesting example in analysis.

Characteristics of the Dirichlet function are given as follows −

- **Rational vs Irrational**− It takes the value 1 at rational points and 0 at irrational points.
- **Discontinuity**− It is discontinuous everywhere in its domain.
- **Mathematical Interest**− Often discussed in real analysis and measure theory.
Let () be a function defined on a domain . The Dirichlet boundary condition specifies that the function  takes on prescribed values on the boundary  of the domain. This can be expressed mathematically as follows −

```
()= () for
```

Where, ()is the function whose behavior is being studied within the domain , () is a known function that specifies the values that  must take on the boundary  and  is the boundary of the domain  where the Dirichlet condition is applied.

### Example

In this example we will show how the Dirichlet function is used in
**SciPy**−
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

# Define alpha parameters (concentration parameters)
alpha = [0.5, 1.0, 2.0]

# Generate a random sample from the Dirichlet distribution
sample = dirichlet.rvs(alpha, size=1000)

# Plot the sample data for visualization
plt.figure(figsize=(8, 6))

# Plot 2D projection of the Dirichlet samples (first 2 components)
plt.scatter(sample[:, 0], sample[:, 1], s=10, color='purple', alpha=0.5)
plt.title('Random Samples from Dirichlet Distribution')
plt.xlabel('x1')
plt.ylabel('x2')

plt.grid(True)
plt.show()
```

#### Output

Following is the output of the Dirichlet function created in SciPy −
![Dirichlet Function Output](/scipy/images/dirichlet_function_ex.jpg)
## Heaviside Step Function

The Heaviside step function is a discontinuous function used in mathematics and engineering to represent the effect of turning on or off a switch. It is defined as 0 for negative inputs and 1 for non-negative inputs.

Following are the characteristics of the Heaviside step function −

- **Switching Behavior**− Represents the switch being off for negative inputs and on for non-negative inputs.
- **Discontinuity**− Has a jump discontinuity at zero.
- **Applications**− Used extensively in control systems and signal processing.
The Mathematical Representation of the Heaviside function can be given as follows −
![Heaviside Function Equation](/scipy/images/heaviside_function_equa.jpg)
### Example

In this example, we will define the Heaviside step function using
**SciPy**−
```
import numpy as np
import matplotlib.pyplot as plt

# Define a custom Heaviside step function
def heaviside(x):
    return np.where(x < 0, 0, 1)

# Define the x values
x_values = np.linspace(-5, 5, 100)

# Apply the custom Heaviside step function to the x values
y_values = heaviside(x_values)

# Plotting the Heaviside step function
plt.plot(x_values, y_values, label='Custom Heaviside Step Function', color='red')
plt.title('Custom Heaviside Step Function')
plt.xlabel('x')
plt.ylabel('H(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Here is the output of the Heaviside function created in SciPy −
![Heaviside Function Output](/scipy/images/heaviside_function_ex.jpg)
## Sine Function with Jump

A sine function with a jump is a modified version of the sine function that experiences a sudden shift in value at a certain point. This function combines oscillatory behavior with discontinuity.

Below are the characteristics of the sine function with a jump −

- **Oscillation**− Exhibits periodic oscillations like the standard sine function.
- **Jump Discontinuity**− Contains a sudden increase or decrease at a specified point.
- **Applications**− Useful in modeling situations where a sudden change occurs in an otherwise periodic behavior.
The mathematical representation of a sine function with a jump can be expressed as a piecewise function. For example consider a sine function that has a jump discontinuity at a certain point say at  = . Then the piecewise function can be defined as follows −
![Sine Function with Jump Equation](/scipy/images/sine_with_jump_function_equa.jpg)
Here represents the magnitude of the jump at.

### Example

In this example we will define a sine function with a jump discontinuity using
**SciPy**library −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the x values
x = np.linspace(0, 10, 500)

# Define the sine function with a jump at x=5
y = np.sin(x)
y[x > 5] += 1  # Create a jump in the function at x=5

# Create a piecewise function for better handling of the jump
x_points = [0, 5, 10]  # Points where the function changes
y_points = [np.sin(0), np.sin(5), np.sin(10) + 1]  # Corresponding function values

# Create an interpolation function
piecewise_func = interp1d(x_points, y_points, fill_value="extrapolate")

# Generate y values for the piecewise function
y_piecewise = piecewise_func(x)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Sine Function with Jump', color='blue')
plt.scatter(5, np.sin(5) + 1, color='red', label='Jump Point', zorder=5)  # Marking the jump
plt.title('Sine Function with a Jump Discontinuity')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Following is the output of the sine function with a jump created in SciPy −
![Sine Function with Jump Output](/scipy/images/sine_with_jump_function_ex.jpg)
## Removable Discontinuity

A removable discontinuity occurs in a function when it is not defined at a certain point, but it could be defined in such a way that the limit exists at that point. This makes it possible to "remove" the discontinuity by redefining the function at that point.

Characteristics of removable discontinuity −

- **Undefined Point**− The function is not defined at the discontinuous point.
- **Limit Exists**− The limit of the function approaches a finite value as the input approaches the discontinuous point.
- **Applications**− Useful in calculus and analysis for illustrating concepts of limits and continuity.
The Mathematical Representation of a removable discontinuity can be given as follows −
![Removable Discontinuity Equation](/scipy/images/removable_discontinuity_equa.jpg)
Where, g(x) is a function that is continuous at  =  and k is defined such that  = lim
().
### Example

In this example we will define a function with a removable discontinuity using
**scipy.interp1d**module −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the piecewise function
def piecewise_function(x):
    return np.where(x != 0, np.sin(x) / x, 1)  # sin(x)/x for x != 0, and 1 at x=0

# Generate x values
x_values = np.linspace(-10, 10, 1000)
y_values = piecewise_function(x_values)

# Plotting the function
plt.plot(x_values, y_values, label='f(x) = sin(x)/x (removable discontinuity at x=0)', color='blue')

# Highlight the discontinuity
plt.scatter([0], [1], color='red', label='Defined Value at x=0', zorder=5)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')

# Set plot limits and labels
plt.ylim(-0.5, 1.5)
plt.title('Removable Discontinuity Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()
plt.show()
```

#### Output

Following is the output of the function with removable discontinuity created in SciPy −
![Removable Discontinuity Output](/scipy/images/removable_discontinuity_ex.jpg)

---

## 26. SciPy - Oscillatory Functions

*Source: [https://www.tutorialspoint.com/scipy/scipy_oscillatory_functions.htm](https://www.tutorialspoint.com/scipy/scipy_oscillatory_functions.htm)*

---

---
[Previous](/scipy/scipy_discontinuous_functions.htm)[Quiz](/scipy/quiz_on_scipy_oscillatory_functions.htm)[Next](/scipy/scipy_partial_differential_equations.htm)
## What are Oscillatory Functions?
**Oscillatory**functions are functions that exhibit repeated fluctuations between certain values which often in a periodic manner. These functions are common in fields such as signal processing, physics and engineering.
In SciPy,
**oscillatory**functions are such as sine, cosine and other trigonometric functions can be handled effectively using various tools for numerical integration, optimization and signal processing.
In numerical computation the oscillatory functions can present challenges especially when dealing with integration over large intervals, as the frequent sign changes can lead to cancellation errors or difficulties in convergence.

SciPy provides specialized methods to handle such functions efficiently particularly in the context of integration and solving differential equations. Some common examples of oscillatory functions are sine and cosine functions, Bessel functions and other periodic signals.

## Characteristics of Oscillatory Functions

The below characteristics of Oscillatory Functions are essential in various fields such as signal processing, physics and electrical engineering where oscillatory functions model waves, vibrations and alternating currents −

- **Periodicity**− Oscillatory functions often repeat their values at regular intervals which is called as the period. For example when functions like sine and cosine exhibit periodic behavior with a fixed interval of 2.
- **Alternating Signs**− The Oscillatory functions fluctuate between positive and negative values. As they oscillate, they alternate between peaks i.e., maximum values and troughs i.e., minimum values.
- **Amplitude**− This refers to the maximum absolute value the function can reach. Oscillatory functions have peaks and troughs determined by this amplitude.
- **Frequency**− This defines how many oscillations or cycles occur within a unit interval. Higher frequency indicates more oscillations over the same interval.
- **Damping (Optional)**− Some oscillatory functions like damped oscillations exhibit a decrease in amplitude over time due to damping factors which can be modeled with exponential terms.
- **Symmetry**− Many oscillatory functions exhibit symmetry such as even i.e., symmetric around the y-axis or odd i.e., symmetric around the origin behavior especially trigonometric functions such as sine and cosine.
- **Phase Shift**− The phase shift of an oscillatory function refers to a horizontal displacement in the function's graph by shifting the position of its peaks and troughs.
## Handling Oscillatory Functions in SciPy

Oscillatory functions are common in various scientific computations particularly in physics and engineering. Due to their repetitive nature which accurately integrating or analyzing these functions can be challenging. SciPy offers tools and strategies to handle oscillatory behavior efficiently.

By adapting the integration approach or applying appropriate methods the SciPy effectively manages the challenges posed by oscillatory functions. Here the key approaches in detail −
**Quad Function for Integration**− SciPys quad function can handle oscillatory integrals. While quad automatically adapts to oscillations it can sometimes struggle with highly oscillatory functions so extra care or specific methods might be necessary. Here is the example of handling the oscillatory function using quad function −
### Example

```
import numpy as np
from scipy.integrate import quad

# Define an oscillatory function
def oscillatory_func(x):
    return np.sin(100 * x)

# Perform integration
result, error = quad(oscillatory_func, 0, np.pi)
print("Integral result:", result)
```

#### Output

Here is the output of the quad function for integration −

```
Integral result: 2.3480880169895062e-15
```

- **Oscillatory Weight Functions**− The**quad()**function allows specifying a weight for integration which can help in handling oscillatory functions more efficiently by compensating for rapid changes.
- **Avoiding Precision Loss**− For functions with very high frequencies the precision loss can occur due to the rapid oscillations. In such cases breaking the integration range into smaller sub-ranges may yield to better accuracy.
- **Specialized Methods**− In some cases of Highly Oscillatory Functions the methods such as Levin integration or other specialized algorithms are recommended. Though SciPy doesn't directly offer these but other packages or custom implementations may be used for high frequency oscillatory functions.
### Example - Simple Oscillatory function

In this example we define a simple sine wave function which is a classic example of an oscillatory function. We'll plot the sine wave using Matplotlib and then integrate it over a specific interval using SciPy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the oscillatory function (sine wave)
def oscillatory_func(x):
    return np.sin(10 * x)  # A sine wave with frequency 10

# Generate x values for plotting
x_values = np.linspace(0, 2 * np.pi, 100)
y_values = oscillatory_func(x_values)

# Plotting the oscillatory function
plt.plot(x_values, y_values, label="Sine wave", color='blue')
plt.title('Oscillatory Function (Sine Wave)')
plt.xlabel('x')
plt.ylabel('sin(10x)')
plt.grid(True)
plt.legend()
plt.show()

# Perform integration over the interval [0, 2p]
result, error = quad(oscillatory_func, 0, 2 * np.pi)
print("Integral of sin(10x) over [0, 2p]:", result)
```

#### Output

Here is the output of the simple oscillatory function using scipy −
![simple oscillatory function Output](/scipy/images/simple_osc_ex.jpg)
## Types of Oscillatory Functions in SciPy

Here are the different types of Oscillatory functions available in Scipy −
Function NameDescription**Sine Wave**A basic periodic oscillatory function that repeats at regular intervals.**Cosine Wave**Similar to sine wave but shifted by a phase of /2.**Fourier Series**A series of sine and cosine functions representing complex periodic signals.**Bessel Function**A type of oscillatory function that occurs in many physical problems such as waves.**Airy Function**Oscillates but decays for large values of the input, used in quantum mechanics.**Modified Bessel Function**Oscillates but grows exponentially for large values of the input.
## Sine Wave Oscillatory Function

In SciPy an
**oscillatory sine wave**function refers to a periodic function that fluctuates between a maximum and a minimum value over a specified interval. The sine function represented mathematically as follows −
```
f(x) = A . sin(Bx + C) + D
```

Where, A is the amplitude which determines the peak height of the wave, B affects the frequency of oscillation i.e., number of cycles in a unit interval, C is the phase shift which determines where the wave starts along the x-axis and D is the vertical shift which moves the entire function up or down.

### Example

Here's a simple example of how to generate and plot an oscillatory sine wave function using SciPy and Matplotlib −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define the sine wave function
def sine_wave(x, A=1, B=1, C=0, D=0):
    return A * np.sin(B * x + C) + D

# Set parameters
A = 1   # Amplitude
B = 2   # Frequency
C = 0   # Phase shift
D = 0   # Vertical shift

# Define the limits of integration
lower_limit = 0
upper_limit = 2 * np.pi

# Use scipy.integrate.quad to integrate the sine wave over one period
integral, error = integrate.quad(sine_wave, lower_limit, upper_limit, args=(A, B, C, D))

# Output the result of the integration
print(f"Integral of sine wave from {lower_limit} to {upper_limit} is: {integral:.5f}")
print(f"Estimated error: {error:.5f}")

# Generate points for plotting
x = np.linspace(0, 2 * np.pi, 1000)
y = sine_wave(x, A, B, C, D)

# Plot the sine wave
plt.plot(x, y, label=f'Sine Wave (A={A}, B={B})', color='blue')
plt.title('Oscillatory Sine Wave Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Following is the output of the Oscillatory Sine function using scipy −

```
Integral of sine wave from 0 to 6.283185307179586 is: -0.00000
Estimated error: 0.00000
```
![Oscillatory sine function Output](/scipy/images/sine_fun_example.jpg)
## Cosine Wave Oscillatory Function

An
**Oscillatory cosine**function in SciPy refers to a function that exhibits oscillatory behavior which is characterized by regular and repeated fluctuations in its values. Specifically the cosine function is a periodic function defined mathematically as follows −
```
f(x) = A . cos(Bx + C) + D
```

Where, A is the amplitude which determines the peak height of the wave, B affects the frequency of oscillation i.e., number of cycles in a unit interval, C is the phase shift which determines how much the function is shifted horizontally and D is the vertical shift which determines how much the function shifted vertically.

### Key Features of Oscillatory Cosine Wave

Below are the key features of the Oscillatory Cosine Wave in scipy −

- **Periodic Nature**− The cosine function repeats its values at regular intervals, with a period given by 2/.
- **Amplitude**− The amplitude determines how high and low the function oscillates. A larger amplitude results in greater peaks and troughs.
- **Frequency**− The frequency indicates how quickly the function oscillates. A higher frequency results in more cycles in a given interval.
- **Applications**− Oscillatory functions such as cosine functions are commonly used in various fields such as signal processing, physics and engineering to model waveforms, vibrations and periodic phenomena.
### Example

Following is the simple example of how to generate and plot an oscillatory cosine wave function using SciPy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define the cosine wave function
def cosine_wave(x, A=1, B=1, C=0, D=0):
    return A * np.cos(B * x + C) + D

# Set parameters
A = 1   # Amplitude
B = 2   # Frequency
C = 0   # Phase shift
D = 0   # Vertical shift

# Define the limits of integration
lower_limit = 0
upper_limit = 2 * np.pi

# Use scipy.integrate.quad to integrate the cosine wave over one period
integral, error = integrate.quad(cosine_wave, lower_limit, upper_limit, args=(A, B, C, D))

# Output the result of the integration
print(f"Integral of cosine wave from {lower_limit} to {upper_limit} is: {integral:.5f}")
print(f"Estimated error: {error:.5f}")

# Generate points for plotting
x = np.linspace(0, 2 * np.pi, 1000)
y = cosine_wave(x, A, B, C, D)

# Plot the cosine wave
plt.plot(x, y, label=f'Cosine Wave (A={A}, B={B})', color='red')
plt.title('Oscillatory Cosine Wave Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Following is the output of the Oscillatory Sine function using scipy −

```
Integral of cosine wave from 0 to 6.283185307179586 is: -0.00000
Estimated error: 0.00000
```
![Oscillatory Cosine function Output](/scipy/images/cosine_fun_ex.jpg)
## Fourier series  Oscillatory Function

The Fourier series is a way to represent a periodic function as a sum of sine and cosine functions. In SciPy we can analyze oscillatory functions using Fourier series by leveraging the
**Fast Fourier Transform (FFT)**.
### Example

Here's an example showing how to create a Fourier series for an oscillatory function using SciPy −

```
import numpy as np
import matplotlib.pyplot as plt

# Define the time variable and the oscillatory function
t = np.linspace(0, 2 * np.pi, 1000)  # Time variable
A = 1  # Amplitude
frequency = 1  # Frequency of the oscillatory function

# Create a composite function (e.g., a square wave)
square_wave = A * np.sign(np.sin(frequency * t))

# Compute the Fourier series coefficients
n_terms = 10  # Number of terms in the Fourier series
fourier_coefficients = np.zeros((n_terms, 2))

for n in range(1, n_terms + 1):
    # Calculate coefficients for sine and cosine terms
    a_n = (1 / np.pi) * np.trapz(square_wave * np.cos(n * t), t)  # Cosine coefficients
    b_n = (1 / np.pi) * np.trapz(square_wave * np.sin(n * t), t)  # Sine coefficients
    fourier_coefficients[n - 1] = [a_n, b_n]

# Reconstruct the function using the Fourier series
fourier_series = np.zeros_like(t)

for n in range(1, n_terms + 1):
    a_n, b_n = fourier_coefficients[n - 1]
    fourier_series += a_n * np.cos(n * t) + b_n * np.sin(n * t)

# Plot the original function and its Fourier series approximation
plt.figure(figsize=(10, 6))
plt.plot(t, square_wave, label='Square Wave', color='blue', linewidth=2)
plt.plot(t, fourier_series, label='Fourier Series Approximation', color='red', linestyle='--')
plt.title('Oscillatory Fourier Series Representation')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Here is the output of the Oscillatory FFT series function using scipy −
![Oscillatory FFT Series function Output](/scipy/images/fft_function_ex.jpg)
## Bessel Function

The
**Bessel functions**are a family of solutions to Bessel's differential equation and are commonly encountered in problems with cylindrical or spherical symmetry. In SciPy we can compute Bessel functions using the**scipy.special**module.
These functions are oscillatory in nature and they have applications in wave propagation, static potentials and signal processing among other fields.

### Example

This example shows how to compute and plot a Bessel function of the first kind
(x) which is one of the most common types of Bessel functions −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn  # Bessel function of the first kind

# Define the order of the Bessel function (n) and the x-values
n = 0  # Order of the Bessel function
x = np.linspace(0, 20, 1000)  # Range of x-values

# Compute the Bessel function of the first kind
bessel_function = jn(n, x)

# Plotting the Bessel function
plt.figure(figsize=(10, 6))
plt.plot(x, bessel_function, label=f'Bessel Function J_{n}(x)', color='blue')
plt.title(f'Oscillatory Bessel Function of the First Kind (J_{n}(x))')
plt.xlabel('x')
plt.ylabel(f'J_{n}(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Here is the output of the Oscillatory Bessel Function using scipy −
![Oscillatory Bessel Function Output](/scipy/images/bessel_fun_ex.jpg)
## Airy Function

The
**Airy function**is a special function used to solve differential equations and is characterized by oscillatory behavior for negative values of its argument. In SciPy we can compute the Airy function using the**scipy.special.airy()**function.
It returns the values of the Airy function along with its derivative and other related functions.

### Characteristics of the Airy Function

Here are the characteristics of the Airy Funtcion in scipy −

- **Oscillatory Behavior**− For negative values of x, the Airy function behaves like a damped oscillatory wave.
- **Decay**− For negative values of x, the Airy function behaves like a damped oscillatory wave.
- **Oscillatory Behavior**− For positive values of x, the Airy function decays rapidly.
- **Applications**− Airy functions are commonly used in quantum mechanics, optics and signal processing especially when solving differential equations with turning points.
### Example

Here is the example of working with the Oscillatory Airy Function in scipy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# Define the range of x values
x = np.linspace(-10, 10, 1000)

# Calculate the Airy function Ai(x) and its derivative
Ai, Aip, Bi, Bip = airy(x)

# Plotting the Airy Ai function (oscillatory for negative x)
plt.figure(figsize=(10, 6))
plt.plot(x, Ai, label='Ai(x) - Airy Function', color='blue')
plt.plot(x, Aip, label="Ai'(x) - Derivative of Airy Function", color='red', linestyle='--')

# Title and labels
plt.title('Oscillatory Airy Function and Its Derivative')
plt.xlabel('x')
plt.ylabel('Value')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
```

#### Output

Here is the output of the Oscillatory Bessel Function using SciPy −
![Airy Function Output](/scipy/images/airy_fun_ex.jpg)
## Modified Bessel Function

The
**Modified Bessel Function**is the First Kind which is often used in various fields including physics and engineering especially when dealing with oscillatory problems in cylindrical coordinates. In SciPy we can use the**scipy.special**module to compute these functions.
### Example

Following is the example of working with the Oscillatory Modified Bessel Function in SciPy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv  # Import the Modified Bessel function of the first kind

# Define the range for the x values
x = np.linspace(0, 20, 1000)

# Define the order of the Bessel function
order = 0  # You can change this to 1, 2, etc., for higher orders

# Compute the Modified Bessel function of the first kind
bessel_values = iv(order, x)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, bessel_values, label=f'Modified Bessel Function of the First Kind (order={order})', color='blue')
plt.title('Oscillatory Modified Bessel Function of the First Kind')
plt.xlabel('x')
plt.ylabel(f'I_{order}(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Here is the output of the Oscillatory Bessel Function using scipy −
![Modified Bessel Function Output](/scipy/images/modified_bessel_fun_ex.jpg)

---

## 27. SciPy - Partial Differential Equations

*Source: [https://www.tutorialspoint.com/scipy/scipy_partial_differential_equations.htm](https://www.tutorialspoint.com/scipy/scipy_partial_differential_equations.htm)*

---

---

## 28. SciPy - Interpolate

*Source: [https://www.tutorialspoint.com/scipy/scipy_interpolate.htm](https://www.tutorialspoint.com/scipy/scipy_interpolate.htm)*

---

---
[Previous](/scipy/scipy_partial_differential_equations.htm)[Quiz](/scipy/quiz_on_scipy_interpolate.htm)[Next](/scipy/scipy_1d_linear_interpolation.htm)**Interpolation**is a fundamental mathematical and computational technique used to estimate unknown values within the range of a set of known data points. Essentially it constructs a function that passes through or near the known points by allowing us to predict intermediate values where data is missing or sparse. It is widely used in areas like data analysis, signal processing, computer graphics and numerical simulations.
The SciPy library provides a comprehensive set of tools for interpolation through its
**scipy.interpolate**module. This module includes methods for 1-dimensional, multi-dimensional and spline interpolation by offering a wide range of algorithms to suit different types of data and smoothness requirements.
## Key Types of Interpolation in SciPy

SciPy provides a variety of interpolation methods for different kinds of data and applications. Below are the key types of interpolation available in the
**scipy.interpolate**module −
- **1-Dimensional Interpolation**− It allows for estimating unknown values based on known data points along a single variable. The**scipy.interpolate**module provides various functions for performing 1D interpolation by accommodating different data characteristics and requirements.
- **Cubic Spline Interpolation**− It is a powerful interpolation technique where the interpolating function is a piecewise cubic polynomial. It ensures smoothness at the data points by creating a series of cubic polynomials that connect the given data points smoothly with continuous first and second derivatives across intervals.
SciPys CubicSpline function enables this interpolation by producing smooth curves for data points with no sharp bends or breaks. Its widely used in scientific computing, data fitting and graphics because of its ability to create visually appealing curves.

- **Barycentric Interpolation**− This is an efficient form of polynomial interpolation where the interpolation polynomial is represented in terms of barycentric weights. It offers a numerically stable and efficient way to perform polynomial interpolation, especially when compared to methods like Lagrange interpolation.
The idea of this interpolation is to interpolate a set of given points by finding a polynomial that passes through them. The Lagrange interpolation which is computationally expensive and prone to numerical instability where barycentric interpolation is more stable and faster particularly for large datasets.

- **Piecewise Polynomial Interpolation**− This interpolation divides a functions domain into multiple sub-intervals and fits a separate polynomial in each of these intervals. This technique ensures that the interpolated function can follow the local behavior of the data more closely compared to a single global polynomial. Piecewise interpolation methods often provide better accuracy especially for functions that exhibit significant changes in behavior across different regions.
A common form of piecewise interpolation is spline interpolation particularly cubic spline interpolation, where third-degree polynomials are fitted between each pair of adjacent data points.

- **Multivariate Interpolation**− It is a generalization of interpolation to functions of multiple variables. Unlike univariate interpolation which involves a single independent variable where multivariate interpolation deals with multiple independent variables by making it essential for applications in higher-dimensional spaces. This technique is used when we want to approximate or estimate a function based on known values at specific points (data points) in multiple dimensions.
In multivariate interpolation the challenge is fitting a surface or higher-dimensional analog through the given data points.

- **Nearest Neighbor Interpolation**− This is a simple and fast interpolation method where the value of an unknown data point is estimated as the value of the nearest known data point. This method is most commonly used when speed is more critical than accuracy or when working with discrete data.
## Applications of Interpolation
**Interpolation**is a crucial technique in various fields by allowing users to estimate unknown values based on known data points. SciPy provides powerful tools for interpolation by enabling users to apply these techniques across diverse applications. Here are some key applications of interpolation in SciPy −
- **Data Resampling**− Interpolation is frequently employed to resample data, especially in time-series analysis. When dealing with irregularly spaced data points, interpolation aids in creating a regular grid, simplifying the analysis of trends and patterns.
- **Image Processing**− In image processing the interpolation methods are applied for resizing and transforming images. Techniques such as aside bilinear and bicubic interpolation facilitate smoother transitions when increasing or decreasing image sizes.
- **Numerical Solutions to Differential Equations**− Interpolation can be utilized within numerical methods to solve ordinary and partial differential equations. By estimating values at discrete points it helps create smooth solutions for problems like heat distribution and wave propagation.
- **Signal Processing**− In signal processing the interpolation is used to reconstruct signals from sampled data. It enhances the quality of audio and video signals by estimating values between sampled points.
- **Scientific Data Analysis**− Researchers often gather data at discrete intervals. Interpolation can fill gaps in experimental data by allowing for better visualization and analysis of trends.
- **Geographic Information Systems (GIS)**− In GIS applications interpolation techniques estimate values at unsampled locations based on available geographic data. This is essential for creating contour maps and analyzing spatial information.
- **Engineering and Manufacturing**− Interpolation is applied in engineering simulations and manufacturing to estimate material and component properties based on discrete measurements.
- **Financial Modeling**− In finance the interpolation helps to estimate asset prices and yields based on existing market data. It is beneficial for pricing derivatives and managing risk.
## Limitations of Interpolation in SciPy

While interpolation in SciPy offers powerful tools for estimating values between known data points it has several limitations and challenges that users must be aware of as mentioned below −

- **Accuracy Dependent on Data Distribution**− Interpolation assumes that the function between data points behaves smoothly. If the actual data is noisy, sparse or highly irregular so the interpolation may lead to inaccurate or misleading results.
- **Overfitting in Higher-Degree Interpolation**− Using higher-degree polynomial interpolation such as cubic splines,can sometimes result in overfitting where the interpolated curve oscillates excessively between data points especially when dealing with noisy data or a large number of points. This phenomenon is often referred to as Runge's phenomenon.
- **Limited to Within Known Data Range**− Standard interpolation methods such as linear and cubic splines are limited to interpolating values within the range of the known data points (extrapolation is often unreliable). Beyond the known range, the accuracy of the results can drop significantly and the behavior of the function becomes unpredictable.
- **Sensitive to Outliers**− If the dataset contains outliers or sudden spikes then interpolation methods may generate distorted results. The presence of such anomalies can lead to incorrect interpolated values as interpolation techniques are generally designed to work on smooth and continuous data.
- **High Computational Cost for Large Data Sets**− Interpolation methods particularly spline interpolation can become computationally expensive when applied to very large datasets. The complexity increases as more points are added especially for higher-dimensional interpolation which can lead to longer processing times.
- **Dimensionality Challenges**− In high-dimensional spaces such as 3D or 4D, the complexity of interpolation grows significantly. Managing multi-dimensional interpolation in SciPy like with griddata can be slow and often suffers from poor accuracy especially in sparse datasets.
- **Boundary Artifacts**− Interpolation methods such as cubic splines or other splines may show artifacts near the boundaries of the data where there are fewer points to influence the curve. These boundary effects can lead to inaccurate interpolated values near the edges of the dataset.
- **Not Suitable for Discontinuous Functions**− Interpolation methods assume smooth transitions between data points by making them unsuitable for discontinuous functions. In cases where there are sudden jumps or breaks in the data interpolation may fail to provide meaningful estimates.
- **Extrapolation Risks**− While interpolation is designed to work between known data points some users may attempt to use interpolation functions for extrapolation. However most interpolation methods in SciPy are unreliable for extrapolating beyond the original data range by leading to large errors or unpredictable results.
- **Data Requirements**− Some interpolation methods such as spline interpolation require a minimum number of data points to work properly. If the dataset is too small or if data points are unevenly spaced then the interpolation might not work as expected.
## Univariate Interpolation Functions

These functions are used for performing univariate interpolation, allowing the estimation of values between known data points.
S.NoFunction & Description1[scipy.interpolate.interp1d()](/scipy/scipy_interpolate_interp1d_function.htm)
Interpolate a 1-D function based on input data points.2[scipy.interpolate.BarycentricInterpolator()](/scipy/scipy_interpolate_BarycentricInterpolator_function.htm)
Interpolating polynomial for a set of points using Barycentric formulation.3[scipy.interpolate.KroghInterpolator()](/scipy/scipy_interpolate_kroghInterpolator_function.htm)
Interpolating polynomial for a set of points using Krogh's method.4[scipy.interpolate.barycentric_interpolate()](/scipy/scipy_interpolate_barycentric_interpolate_function.htm)
Convenience function for polynomial interpolation using Barycentric formulation.5[scipy.interpolate.krogh_interpolate()](/scipy/scipy_interpolate_krogh_interpolator_function.htm)
Convenience function for polynomial interpolation using Krogh's method.6[scipy.interpolate.pchip_interpolate()](/scipy/scipy_interpolate_pchip_interpolate_function.htm)
Convenience function for PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.7[scipy.interpolate.CubicHermiteSpline()](/scipy/scipy_interpolate_CubicHermiteSpline_function.htm)
Piecewise-cubic interpolator that matches values and first derivatives at given points.8[scipy.interpolate.PchipInterpolator()](/scipy/scipy_interpolate_pchipinterpolator_function.htm)
PCHIP 1-D monotonic cubic interpolation ensuring monotonicity.9[scipy.interpolate.Akima1DInterpolator()](/scipy/scipy_interpolate_akima1dinterpolator_function.htm)
Akima interpolator, which is a non-smoothing spline interpolation.10scipy.interpolate.CubicSpline()
Cubic spline data interpolator, providing smooth piecewise cubic polynomial interpolants.11[scipy.interpolate.PPoly()](/scipy/scipy_interpolate_ppoly_function.htm)
Piecewise polynomial interpolator defined in terms of coefficients and breakpoints.12[scipy.interpolate.BPoly()](/scipy/scipy_interpolate_bpoly_function.htm)
Piecewise polynomial interpolator defined in terms of coefficients and breakpoints (generalized).
## Multi-variant Interpolation Functions

These functions are used for performing interpolation on unstructured and structured N-dimensional data by enabling the estimation of values in multi-dimensional space.
S.NoFunction & Description1[scipy.interpolate.griddata()](/scipy/scipy_interpolate_griddata_function.htm)
Interpolate unstructured D-D data based on the specified method (linear, nearest, cubic).2[scipy.interpolate.LinearNDInterpolator()](/scipy/scipy_interpolate_linearndinterpolator_function.htm)
Piecewise linear interpolator for N dimensions, using the input points and values.3[scipy.interpolate.NearestNDInterpolator()](/scipy/scipy_interpolate_nearestndinterpolator_function.htm)
Nearest-neighbor interpolation for N-D data, finding the nearest point in the input data.4[scipy.interpolate.CloughTocher2DInterpolator()](/scipy/scipy_interpolate_cloughtocher2dinterpolator_function.htm)
Interpolates over a 2D domain using Clough-Tocher method, allowing for piecewise polynomial fitting.5[scipy.interpolate.RBFInterpolator()](/scipy/scipy_interpolate_rbfinterpolator_function.htm)
Radial basis function (RBF) interpolation in N dimensions for smooth surface fitting.6[scipy.interpolate.Rbf()](/scipy/scipy_interpolate_rbf_function.htm)
Radial basis function interpolator for multi-dimensional data, allowing for flexible interpolation methods.7[scipy.interpolate.interpn()](/scipy/scipy_interpolate_interpn_function.htm)
Multidimensional interpolation on regular or rectilinear grids, providing flexibility in interpolation methods.8[scipy.interpolate.RegularGridInterpolator()](/scipy/scipy_interpolate_regulargridinterpolator_function.htm)
Interpolator on a regular or rectilinear grid in arbitrary dimensions, enabling efficient interpolation on structured grids.9[scipy.interpolate.RectBivariateSpline()](/scipy/scipy_interpolate_rectbivariatespline_function.htm)
Piecewise spline interpolation for 2D data defined on a rectangular grid.
### Example

Heres an example of 1-D interpolation using SciPys
**interp1d()**function. This example shows how to interpolate between a set of data points and plot the results −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Given data points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])

# Create linear and cubic interpolation functions
linear_interp = interp1d(x, y, kind='linear')
cubic_interp = interp1d(x, y, kind='cubic')

# Generate new x values for interpolation
x_new = np.linspace(0, 5, 100)

# Interpolate the y values at the new x values
y_linear = linear_interp(x_new)
y_cubic = cubic_interp(x_new)

# Plot the original data points
plt.scatter(x, y, color='red', label='Data points')

# Plot the linear interpolation
plt.plot(x_new, y_linear, label='Linear interpolation', color='blue')

# Plot the cubic interpolation
plt.plot(x_new, y_cubic, label='Cubic interpolation', color='green')

# Adding labels and legend
plt.title('1-D Interpolation using SciPy')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
```

#### Output

Here is the output of the Interpolate example in scipy −
![Interpolate Example](/scipy/images/interpolate_example.jpg)

---

## 29. SciPy - Linear 1-D Interpolation

*Source: [https://www.tutorialspoint.com/scipy/scipy_1d_linear_interpolation.htm](https://www.tutorialspoint.com/scipy/scipy_1d_linear_interpolation.htm)*

---

---
[Previous](/scipy/scipy_interpolate.htm)[Quiz](/scipy/quiz_on_scipy_1d_linear_interpolation.htm)[Next](/scipy/scipy_polynomial_1d_interpolation.htm)
SciPy
**Linear 1-D Interpolation**is a method used to estimate unknown values between two known data points in one dimension by assuming a linear relationship between adjacent points. This is useful when we have discrete data and want to create a smooth function that approximates intermediate values. In SciPy the**interp1d()**function from the**scipy.interpolate**module is used to perform this interpolation.
Linear interpolation works by connecting two points with a straight line and then calculating any intermediate points along that line. This approach is computationally efficient but assumes a constant rate of change between data points which may not be accurate for more complex data.

## Working of Linear Interpolation
**Linear interpolation**works by estimating the value of a function between two known points by assuming that the function behaves linearly between those points. The given two known data points are (x,y) and (x,y) and now the linear interpolation finds an estimated value as**y**at a point**x**between**x**and**x**.
The formula to calculate the interpolated value y for a given x is given as follows −
![1-d Interpolation Equation](/scipy/images/1d_interpolation_equ.jpg)
Where, (x
,y) and (x,y) are the known data points, x is the point where we want to estimate the value and y is the interpolated value.
Linear interpolation assumes a straight-line relationship between adjacent points by making it simple and efficient but not suitable for data with nonlinear patterns.

For example if we have known data points as (1,3) and (4,7) then we can estimate the value at x = 2, the linear interpolation would be computed as 4.33.

## Syntax

Following is the syntax of generating the 1-d interpolation in scipy with the help of
**scipy.interpolate.inter1d()**function −
```
scipy.interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=False)
```

## Parameters

Here are the parameters of the
**scipy.interpolate.inter1d()**function −
- **x**− Array of independent data points.
- **y**− Array of dependent data points. It should have the same length as x.
- **kind**− This parameters specifies the type of interpolation to perform. Common options are 'linear'(default) which is linear interpolation between points and 'nearest', 'zero', 'slinear', 'quadratic', 'cubic' are for different kinds of interpolation.
- **axis**− This parameter specifies the axis of y along which to interpolate and the default value is -1 for the last axis.
- **copy**− If True then the arrays x and y are copied. Default value is True.
- **bounds_error**− If True then an error is raised if a value outside the range of x is requested. Default value is None which uses fill_value instead.
- **fill_value**− Value to return for x values outside the interpolation range. Default value is np.nan.
- **assume_sorted**− If True then the input arrays are assumed to be sorted. Default value is False.
## Linear Interpolation

### Example

Here is the example of the generating the linear interpolation with the use of scipy.interpolate.inter which connects the data points with straight lines.

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Known data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 2, 1, 3, 7])

# Linear interpolation
f_linear = interp1d(x, y, kind='linear')

# Interpolated points
x_new = np.linspace(0, 4, 100)
y_new = f_linear(x_new)

# Plotting
plt.plot(x, y, 'o', label='Data Points')
plt.plot(x_new, y_new, '-', label='Linear Interpolation')
plt.legend()
plt.title('Linear Interpolation')
plt.show()
```

#### Output

Below is the output of the linear 1-D interpolation in scipy −
![1-d Linear Interpolation Example](/scipy/images/linear_1d_example.jpg)
## Cubic Interpolation
**Cubic interpolation**is a more advanced interpolation method compared to linear interpolation. It fits a cubic polynomial between data points by resulting in a smoother curve that better captures the behavior of nonlinear data. This method is useful when we need smoother transitions between points as it avoids the sharp changes that can occur with linear interpolation.
SciPy provides cubic interpolation through the
**interp1d()**function in the**scipy.interpolate**module by specifying the parameter as kind='cubic'.
### Example

Here is the example of generating the cubic interpolation using the
**scipy.interpolate.inter1d()**function by passing the parameter kind = 'cubic' −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Known data points (nonlinear function)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])

# Create a cubic interpolation function
cubic_interp = interp1d(x, y, kind='cubic', fill_value='extrapolate')

# Generate new x values for interpolation
x_new = np.linspace(-1, 6, 100)

# Interpolate the y values at the new x values
y_new = cubic_interp(x_new)

# Plot the original data points
plt.scatter(x, y, color='red', label='Data points')

# Plot the cubic interpolation
plt.plot(x_new, y_new, label='Cubic interpolation', color='green')

# Adding labels and legend
plt.title('Cubic 1-D Interpolation using SciPy')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
```

#### Output

Here is the output of the Cubic 1-D interpolation in scipy library using the inter1d() function −
![1-d Cubic Interpolation Example](/scipy/images/cubic_1d_interpolation.jpg)
## Nearest-Neighbor Interpolation
**Nearest-neighbor interpolation**is the simplest form of interpolation where the value of an unknown point is assigned to the value of the nearest known data point. This method does not attempt to create a smooth curve or continuous transitions between data points instead it selects the value of the closest available point by making it fast and computationally inexpensive.
In SciPy the nearest-neighbor interpolation can be performed using the interp1d function from the
**scipy.interpolate**module by setting the kind parameter to**'nearest'**.
### Example

Here is the example of generating the Nearest-Neighbor Interpolation using the
**scipy.interpolate.inter1d()**function by passing the parameter kind = 'nearest' −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Known data points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])

# Create a nearest-neighbor interpolation function
nearest_interp = interp1d(x, y, kind='nearest', fill_value='extrapolate')

# Generate new x values for interpolation
x_new = np.linspace(-1, 6, 100)

# Interpolate the y values at the new x values
y_new = nearest_interp(x_new)

# Plot the original data points
plt.scatter(x, y, color='red', label='Data points')

# Plot the nearest-neighbor interpolation
plt.plot(x_new, y_new, label='Nearest-Neighbor Interpolation', color='blue')

# Adding labels and legend
plt.title('Nearest-Neighbor Interpolation using SciPy')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
```

#### Output

Following is the output of the Nearest Neighbour 1-D interpolation in scipy library using the inter1d() function −
![1-d Nearest Neighbour Interpolation Example](/scipy/images/1d_nearest_neighbour.jpg)
## Quadratic Interpolation
**Quadratic interpolation**is a method that fits a quadratic function i.e., a second-degree polynomial to three or more data points by allowing for a smooth curve to approximate the data. It is more accurate than linear interpolation because it captures curvature but it is less complex and computationally demanding than higher-order interpolation methods such as cubic interpolation.
In SciPy quadratic interpolation can be performed using the
**interp1d()**function from the**scipy.interpolate**module by setting the kind parameter to**quadratics**.
### Example

Following is the example of generating the Quadratic 1d Interpolation using the
**scipy.interpolate.inter1d()**function by passing the parameter kind = 'quadratic' −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Known data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 7, 13, 21])

# Create a quadratic interpolation function
quadratic_interp = interp1d(x, y, kind='quadratic', fill_value='extrapolate')

# Generate new x values for interpolation
x_new = np.linspace(-1, 5, 100)

# Interpolate the y values at the new x values
y_new = quadratic_interp(x_new)

# Plot the original data points
plt.scatter(x, y, color='red', label='Data points')

# Plot the quadratic interpolation
plt.plot(x_new, y_new, label='Quadratic Interpolation', color='blue')

# Adding labels and legend
plt.title('Quadratic Interpolation using SciPy')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
```

#### Output

Below is the output of the Quadratic 1-D interpolation in scipy library using the inter1d() function −
![1-d Quadratic Interpolation Example](/scipy/images/quadratic_1d_example.jpg)
## Benefits of Linear 1-D Interpolation

Here are the benefits of using the Linear 1-D Interpolation in scipy −

- **Simple and Efficient**− Linear interpolation is computationally inexpensive and straightforward to implement.
- **Approximation**− It provides a reasonable estimate when data points follow a near-linear trend.
## Limitations of Linear 1-D Interpolation

We can see some limitations of using the Linear 1-D Interpolation in Scipy −

- **Accuracy**− Linear interpolation may not be suitable for data with significant non-linearity as it doesn't capture curves or changes in gradient.
- **Sharp Corners**− The piecewise linear approach can introduce discontinuities in the derivative of the interpolated function.

---

## 30. SciPy - Polynomial 1-D Interpolation

*Source: [https://www.tutorialspoint.com/scipy/scipy_polynomial_1d_interpolation.htm](https://www.tutorialspoint.com/scipy/scipy_polynomial_1d_interpolation.htm)*

---

---
[Previous](/scipy/scipy_1d_linear_interpolation.htm)[Quiz](/scipy/quiz_on_scipy_polynomial_1d_interpolation.htm)[Next](/scipy/scipy_spline_1d_interpolation.htm)**Polynomial 1-D Interpolation**is a method used to estimate new data points within the range of a set of known data points by fitting a polynomial function through them. In this process a polynomial of degree -1 is constructed that passes through  given points. This polynomial can then be used to predict values for intermediate points.
Polynomial interpolation is widely used in numerical analysis and scientific computing for approximating smooth functions. However, high-degree polynomials can lead to inaccuracies due to oscillations near the boundaries and this phenomenon is called as
**Runge's phenomenon**. SciPy provides tools for polynomial interpolation including BarycentricInterpolator and KroghInterpolator which are numerically stable methods for performing this type of interpolation.
## Key Characteristics of Polynomial 1-D Interpolation

Following are the key characteristics of polynomial 1-D Interpolation in scipy −

- **Degree of Polynomial**− A polynomial of degree -1 is fitted through  data points.
- **Exact Fit to Data**− The polynomial passes through all the provided data points exactly.
- **Smoothness**− The interpolating polynomial is smooth and continuous across its range by making it suitable for approximating smooth functions.
- **Runge's Phenomenon**− For higher-degree polynomials the oscillations may occur especially near the edges of the interpolation range with reducing accuracy.
- **Global Nature**− Polynomial interpolation is a global method which change in any data point affects the entire interpolating polynomial.
- **Versatility**− The polynomial 1-D Interpolation works well for small to medium datasets but may struggle with large datasets due to overfitting or instability.
- **Efficient for Small Intervals**− This method works well over short intervals where lower-degree polynomials provide good approximations.
- **Error Sensitivity**− Errors from noisy data points can propagate and affect the entire polynomial by making it sensitive to data precision.
## What is Polynomial Degree?

The
**polynomial degree**refers to the highest exponent of the variable in a polynomial expression. It determines the shape and complexity of the polynomial function. The degree of a polynomial is defined as the largest integer  such that the polynomial can be expressed in the form as given below −
```
anxn+an1xn1++a1x+a0, where an, an1,...,a0 are coefficients and n0.
```
x+ax++ax+a, where a, a,...,aare coefficients and0.
### Types of Polynomials by Degree

Polynomials can be classified according to their degree which is defined as the highest power of the variable present in the polynomial expression. Each category of polynomial displays unique characteristics and behaviors by making them appropriate for different mathematical applications and modeling situations. Here are the types of polynomials by degree −
DegreeNameDescriptionGeneral FormExample0Constant PolynomialOutputs a constant value regardless of the input variable x.p(x) = cp(x) = 51Linear PolynomialDescribes a straight line defined by a slope and a y-intercept.p(x) = ax + bp(x) = 2x + 32Quadratic PolynomialForms a parabolic shape with a maximum or minimum point (the vertex).p(x) = ax + bx + cp(x) = 3x - 2x + 13Cubic PolynomialCan have two turning points and cross the x-axis up to three times.p(x) = ax + bx + cx + dp(x) = x + 2x - 3x + 44Quartic PolynomialCan exhibit three turning points and intersect the x-axis up to four times.p(x) = ax + bx + cx + dx + ep(x) = 2x - 3x + x + 15Quintic PolynomialExhibits complex behavior by allowing for up to four turning points and five x-axis crossings.p(x) = ax + bx + cx + dx + ex + fp(x) = x - 2x + x - 16Sextic PolynomialHigher degree can exhibit more complex oscillations and multiple intersections with the x-axis.p(x) = ax + bx + cx + dx + ex + fx + gp(x) = 4x + 3x - x + 7nnth Degree PolynomialPolynomials of degree n can display diverse behaviors based on the coefficients involved.p(x) = ax + bx + ... + kVaries based on n
### Example

Here's a simple example of 1-D polynomial interpolation using the
**BarycentricInterpolator**from the SciPy library. This example shows how to interpolate a set of known data points with a polynomial and visualize the results −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

# Step 1: Define known data points
# Known x values
x = np.array([0, 1, 2, 3, 4])
# Known y values (e.g., some function values)
y = np.array([1, 2, 0, 2, 1])

# Step 2: Create the polynomial interpolator
interpolator = BarycentricInterpolator(x, y)

# Generate new x values for interpolation
x_new = np.linspace(0, 4, 100)

# Interpolate the y values at new x values
y_new = interpolator(x_new)

# Step 3: Plot the original data points and the polynomial interpolation
plt.scatter(x, y, color='red', label='Data Points', zorder=5)
plt.plot(x_new, y_new, label='Polynomial Interpolation', color='blue', zorder=1)

# Adding labels and title
plt.title('Polynomial 1-D Interpolation Example')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Following is the output of the 1-D polynomial interpolation −
![1-d Polynomial Interpolation simple Example](/scipy/images/simple_polynomial_example.jpg)
## Scipy Functions for Polynomial Interpolation

In SciPy polynomial interpolation is primarily facilitated through the
**scipy.interpolate**module which provides several classes and functions to perform polynomial interpolation in one dimension (1-D). Let's see the key functions and classes related to polynomial interpolation in detail −
### Barycentric interpolation
**Barycentric interpolation**is an efficient method for polynomial interpolation that reduces the computational cost associated with evaluating interpolating polynomials at various points. In SciPy Barycentric interpolation can be performed using the**BarycentricInterpolator()**function.
#### Key Features of Barycentric Interpolation

Here are the key features of Barycentric Interpolation class in scipy −

- **Efficiency**− Barycentric interpolation is computationally efficient for evaluating polynomials especially for large datasets.
- **Stability**− It is numerically stable by reducing the risk of errors due to ill-conditioned polynomial interpolation.
- **Weight Calculation**− This method involves calculating weights based on the interpolation points by allowing for fast evaluation of the polynomial at any given point.
### Example

Following is the example of generating the Polynomial Interpolation by using the
**BarycentricInterpolator()**function in scipy −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

# Define the sample data points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.sin(x)  # Function values at the sample points

# Create the Barycentric Interpolator
interpolator = BarycentricInterpolator(x, y)

# Define new x values for interpolation
x_new = np.linspace(0, 5, 100)
y_new = interpolator(x_new)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data Points', zorder=5)
plt.plot(x_new, y_new, label='Barycentric Interpolated Curve', color='blue')
plt.title('Barycentric Interpolation Example')
plt.xlabel('x')
plt.ylabel('Interpolated Value')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
```

#### Output

Here is the output of the Barycentric Polynomial Interpolation −
![Barycentric Polynomial Interpolation Example](/scipy/images/barycentric_polynomial_example.jpg)
## Krogh interpolation
**Krogh interpolation**is a method for polynomial interpolation that allows for the specification of function values and their derivatives at given points. This technique is particularly useful when the data points contain not only function values but also information about the rates of change (derivatives) at those points. Krogh interpolation can provide a more accurate polynomial representation of the underlying function compared to standard polynomial interpolation especially when the derivatives at the interpolation points are known.
### Key Features of Krogh Interpolation

Following are the key features of Krogh Interpolation class in scipy −

- **Higher Accuracy**− By including derivatives the Krogh interpolation can yield more accurate results compared to traditional polynomial interpolation methods particularly for smooth functions.
- **Polynomial Degree**− The degree of the interpolating polynomial is determined by the total number of points used including both function values and derivative values. This can result in polynomials of varying degrees depending on the number of constraints applied.
- **Efficient Calculation**− Krogh interpolation uses divided differences to calculate the coefficients of the polynomial by making it computationally efficient.
### Example

Following is the example of generating the Polynomial Interpolation by using the
**KroghInterpolator()**function in scipy −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import KroghInterpolator

# Define sample data points (x) and their corresponding function values (y)
x = np.array([0, 1, 2, 3])
y = np.array([1, 2, 0, 1])  # Function values

# Create the Krogh Interpolator (without derivatives)
interpolator = KroghInterpolator(x, y)

# Define new x values for interpolation
x_new = np.linspace(0, 3, 100)
y_new = interpolator(x_new)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data Points', zorder=5)
plt.plot(x_new, y_new, label='Krogh Interpolated Curve', color='blue')
plt.title('Krogh Interpolation Example')
plt.xlabel('x')
plt.ylabel('Interpolated Value')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
```

#### Example

Below is the output of the Krogh Polynomial Interpolation −
![Krogh Polynomial Interpolation Example](/scipy/images/Kroghinterpolator_example.jpg)
## Limitations of Polynomial Interpolation

Here are the limitations of polynomial Interpolation in scipy −

- **Runge's Phenomenon**− High-degree polynomials can exhibit large oscillations, especially at the edges of the interpolation range by leading to poor approximations.
- **Overfitting**− A polynomial that is too high in degree can overfit the data by resulting in poor generalization to points outside the known range.
- **Computational Complexity**− Higher-degree polynomials increase the computational cost and complexity particularly for large datasets.

---

## 31. SciPy - Spline 1-D Interpolation

*Source: [https://www.tutorialspoint.com/scipy/scipy_spline_1d_interpolation.htm](https://www.tutorialspoint.com/scipy/scipy_spline_1d_interpolation.htm)*

---

---
[Previous](/scipy/scipy_polynomial_1d_interpolation.htm)[Quiz](/scipy/quiz_on_scipy_spline_1d_interpolation.htm)[Next](/scipy/scipy_grid_data_multi_dimensional_interpolation.htm)**Spline interpolation**in SciPy is a technique for creating smooth curves through a set of data points by fitting piecewise polynomials between the points. Splines are particularly cubic splines which are used to interpolate the data in a way that ensures both the function and its derivatives are continuous by providing a smooth and accurate representation of the data.
In SciPy the
**spline interpolation**is implemented in the**scipy.interpolate**module with functions such as CubicSpline() and InterpolatedUnivariateSpline(). These functions can generate smooth interpolating functions by making them suitable for data fitting and graphical applications.
## Key Features of Spline Interpolation

Below are the key features of the Spline Interpolation in Scipy −

- **Piecewise Polynomial Fit**− Spline interpolation divides the data range into intervals and fits a polynomial to each interval by ensuring a smooth transition between points.
- **Continuity of Derivatives**− For cubic splines the first and second derivatives are continuous at the boundaries of the intervals (called knots) by providing a smooth curve.
- **Flexibility**− Splines can be adapted to handle various boundary conditions, such as clamped or natural, giving more control over the behavior of the curve at the ends.
- **Accuracy**− Compared to higher-degree polynomials the splines are less prone to oscillations by making them more accurate for smoothly interpolating data.
- **Efficiency**− Spline interpolation is computationally efficient especially for large datasets due to its piecewise nature.
- **Applications**− Spline interpolation is widely used in data fitting, computer graphics (smooth animations) and engineering simulations where smooth curves are essential.
## Advantages of Spline Interpolation

Below are the advantages of using the Spline Interpolation in Scipy −

- **Smoothness**− Cubic splines provide a smooth curve without the oscillations seen in high-degree polynomial interpolations.
- **Local Control**− Adjusting one data point will only affect the nearby spline segments, not the entire curve.
- **Flexibility**− This method can be easily adapted for various boundary conditions.
### Example

This example shows how to use the spline interpolation in scipy, for smoothly connects the data points −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define sample data points (x) and their corresponding function values (y)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 0, 1, 0, 1])

# Create a cubic spline interpolator
cs = CubicSpline(x, y)

# Define new x values for interpolation
x_new = np.linspace(0, 5, 100)
y_new = cs(x_new)

# Plot the results
plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x_new, y_new, label='Cubic Spline Interpolated Curve', color='blue')
plt.title('Cubic Spline Interpolation Example')
plt.xlabel('x')
plt.ylabel('Interpolated Value')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Following is the output of the simple spline interpolation in scipy −
![Spline Interpolation simple Example](/scipy/images/spline_interpolation_example.jpg)
## Functions used for Spline Interpolation

SciPy provides several functions for
**spline interpolation**in which each catering to different types of interpolation needs. These functions are available under the**scipy.interpolate**module and can handle both 1-D and higher-dimensional spline interpolations. Here's are the key functions in SciPy for spline interpolation −FunctionPurposeKey Features**CubicSpline**Performs cubic spline interpolation- Smooth cubic polynomials between points
- Supports boundary conditions**splrep**and**splev**B-spline representation (**splrep**) and evaluation (**splev**)-**splrep**: Finds B-spline
-**splev**: Evaluates at given points**UnivariateSpline**Univariate spline interpolation of degree**k**- Smooth spline
- Optional smoothing factor**BSpline**More general B-spline with control over knots and coefficients- Full control over the spline
- Customizable knots and basis functions**make_interp_spline**Constructs a B-spline approximation- General spline orders
- Similar to**CubicSpline****PchipInterpolator**Piecewise cubic Hermite interpolation- Monotonic, piecewise cubic
- No overshooting
## CubicSpline() Function

The
**CubicSpline()**function in SciPy provides a powerful method for cubic spline interpolation which fits a piecewise cubic polynomial between given data points. A cubic spline ensures smoothness at the data points and the first and second derivatives of the polynomial are continuous across these points. This results in a smooth curve that passes through all data points by making it ideal for interpolation tasks where smoothness is required.
### Syntax

Following is the syntax of using the
**CubicSpline()**function in scipy −
```
scipy.interpolate.CubicSpline(x, y, bc_type='not-a-knot', extrapolate=True)
```

### Parameters

Below are the parameters of the
**CubicSpline()**function in scipy −
- **x**− 1-D array of independent variable data points.
- **y**−  1-D array of dependent variable data points.
- **bc_type**− Boundary condition type with default value as 'not-a-knot and can also be 'natural' or 'clamped'.
- **extrapolate**− Whether to extrapolate to out-of-bounds points and the default value is True.
### Example

In this example a cubic spline is fit through a set of data points and the resulting smooth curve is plotted. We can also adjust the boundary conditions by modifying the
**bc_type**parameter −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 0, 1, 0])

# Perform cubic spline interpolation
cs = CubicSpline(x, y)

# Generate new points for interpolation
x_new = np.linspace(0, 4, 100)
y_new = cs(x_new)

# Plot the results
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_new, y_new, label='Cubic Spline Interpolation')
plt.title('Cubic Spline Interpolation Example')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Below is the output of the Cubic Spline Interpolation in Scipy −
![Cubic Spline Interpolation Example](/scipy/images/cubic_spline_example.jpg)
## splrep() and splev() Functions

In SciPy the splrep and splev are two functions used together for B-spline interpolation. Heres a detailed explanation of each function −

### splrep() Function

The
**splrep()**function is abbrivated as Spline Representation which is used to compute the B-spline representation of a 1-D curve. It returns the knots, coefficients and degree of the spline that can be used to evaluate the spline later. We can provide the data points x and y to fit a spline curve. Optionally we can also specify the degree of the spline and a smoothing factor.
### Syntax

Following is the syntax of using the
**splrep()**function in scipy −
```
splrep(x, y, s=0, k=3)
```

### Parameters

Below are the parameters of the
**splrep()**function in scipy −
- **x**− array of x-coordinates of the data points.
- **y**− array of y-coordinates of the data points.
- **s(optional)**− smoothing factor.
- **k(optional)**− degree of the spline with default value is cubic spline k=3.
### Example

This example shows how to use splrep() to perform B-spline interpolation on a set of data points −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Define the data points (x) and corresponding values (y)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.5, 1.5, 2.0, 1.0, 0])

# Find the B-spline representation of the curve
# tck contains the knots (t), coefficients (c), and degree (k) of the spline
tck = interpolate.splrep(x, y)

# Generate new x values for interpolation
x_new = np.linspace(0, 5, 100)

# Evaluate the spline at the new x values
y_new = interpolate.splev(x_new, tck)

# Plot the original data points and the interpolated spline curve
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'o', label='Original Data Points')
plt.plot(x_new, y_new, label='B-spline Interpolation')
plt.title('B-spline Interpolation using splrep()')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

#### Output

Below is the output of the splrep() function in Scipy −
![splrep Interpolation Example](/scipy/images/splrep_example.jpg)
## splev Function

The
**splev()**function is defined as Spline Evaluation which is used to evaluates the spline or its derivatives at given points using the B-spline representation which is obtained from splrep() function. Once we have computed the spline representation using splrep() function then we can use splev() function to interpolate or evaluate the function at new points.
### Syntax

Below is the syntax of using the
**splev()**function in scipy −
```
splev(x_new, tck)
```

### Parameters

Here are the parameters of the
**splev()**function in scipy −
- **x_new**− array of x-coordinates of the data points.
- **tck**− The B-spline representation which is the output from splrep() function.
- **der(optional)**− Derivative order with default value as 0 for interpolation.
### Example

This example shows how the function splev() is used to generate the smooth B-spline curve based on the original data points −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

# Sample data points
x = np.linspace(0, 10, 10)
y = np.sin(x)

# Compute the B-spline representation of the data (tck = knots, coefficients, degree)
tck = splrep(x, y)

# New points for evaluating the spline
x_new = np.linspace(0, 10, 100)

# Use splev to evaluate the spline at the new points
y_new = splev(x_new, tck)

# Plot the original data points and the interpolated spline
plt.scatter(x, y, label='Data Points', color='red')
plt.plot(x_new, y_new, label='B-spline Interpolated Curve', color='blue')
plt.title('B-spline Interpolation using splrep and splev')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Here is the output of the splev() function in Scipy −
![splev Interpolation Example](/scipy/images/splev_example.jpg)
## UnivariateSpline

The
**UnivariateSpline**is a convenient class for performing one-dimensional spline interpolation. It fits a spline to a set of data points and allows for smooth interpolation while controlling the smoothing factor. The class also provides flexibility to adjust the degree of the spline and the level of smoothing by making it useful for noisy data or when a smooth and continuous approximation is needed.
### Syntax

Below is the syntax of using the
**UnivariateSpline()**class in scipy −
```
scipy.interpolate.UnivariateSpline(x, y, w=None, bbox=[None, None], k=3, s=None)
```

### Parameters

Following are the parameters of the
**UnivariateSpline()**class in scipy −
- **x,y**− The data points.
- **w**− Optional weights for the spline fit.
- **k**− The degree of the spline with the default value is 3 for cubic spl9ine.
- **s**− Smoothing factor, if s=0 then the spline will pass through all data points.
### Example

Here in this example we will generate the univariate spline interpolation with the help of
**UnivariateSpline()**class in scipy −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Generate sample data
x = np.linspace(0, 10, 10)
y = np.sin(x) + np.random.normal(0, 0.1, 10)  # Adding noise to the sine function

# Fit a cubic spline to the noisy data
spl = UnivariateSpline(x, y, s=1)  # s is the smoothing factor

# Generate new x values for the smooth spline curve
x_new = np.linspace(0, 10, 100)
y_new = spl(x_new)

# Plot the results
plt.scatter(x, y, color='red', label='Noisy Data')
plt.plot(x_new, y_new, color='blue', label='Spline Fit (smoothed)')
plt.title('Univariate Spline Interpolation with Smoothing')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

#### Output

Following is the output of the UnivariateSpline() class in Scipy −
![Univariate spline Interpolation Example](/scipy/images/univariate_spline_example.jpg)
## BSpline

In SciPy
**BSpline**is a class used for constructing and evaluating B-spline curves. B-splines can be given as Basis splines which are a generalization of Bzier curves and they provide a way to represent smooth curves through a set of control points. B-splines are widely used in computational geometry, numerical analysis and computer graphics for curve fitting and interpolation.
### Syntax

Below is the syntax of using the
**BSpline()**class in scipy −
```
scipy.interpolate.BSpline(t, c, k, extrapolate=True, axis=0)
```

### Parameters

Following are the parameters of the
**BSpline()**class in scipy −
- **t**− Array of knot points which sorted in non-decreasing order.
- **c**− Array of spline coefficients.
- **k**− Degree of the spline i.e., order.
- **extrapolate**− Whether to extrapolate for points outside the range of the knots.
### Example

Here in this example we will generate the BSpline interpolation with the help of
**BSpline()**class in scipy −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Define the knot vector and coefficients (control points)
knots = [0, 1, 2, 3, 4, 5, 6, 7]  # Knot vector (degree + control points + 1)
coefficients = [1, 2, 0, -1, 0, 2]  # Control points
degree = 3  # Cubic B-spline

# Create the B-spline object
spline = BSpline(knots, coefficients, degree)

# Generate x values for evaluating the spline
x = np.linspace(1, 6, 100)
y = spline(x)

# Plot the B-spline curve
plt.plot(x, y, label='B-Spline Curve')

# Mark the control points
plt.scatter([1, 2, 3, 4, 5], coefficients[:-1], color='red', label='Control Points')

# Add labels, title, grid, and legend
plt.title('Cubic B-Spline Example')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
```

#### Output

Following is the output of the BSpline() class in Scipy −
![Bspline Interpolation Example](/scipy/images/bspline_example.jpg)
## make_interp_spline

In SciPy the function
**make_interp_spline()**is used to create a B-spline representation for a given set of data points. It fits a smooth B-spline curve to the data points and it is commonly used for 1-D interpolation. Its primarily used when we need a smooth interpolation of data points especially when we want control over the degree of the spline or need to generate smooth curves for plotting.
### Syntax

Below is the syntax of using the
**make_interp_spline()**function in scipy −
```
scipy.interpolate.make_interp_spline(x, y, k=3, bc_type=None, axis=0)
```

### Parameters

Following are the parameters of the
**make_interp_spline()**function in scipy −
- **x**− The independent variable i.e., 1D array of the x-coordinates of the data points.
- **y**− The dependent variable i.e., y-values corresponding to x and must be the same shape.
- **k**− The degree of the spline with the default value is 3 i.e. cubic spline.
- **bc_type(optional)**− Boundary conditions for the spline where we can specify natural, clamped, etc.
- **axis**− Axis along which to interpolate.
### Example

Below is the example which shows how to use
**make_interp_spline()**function for cubic spline interpolation −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Define the data points (x, y)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])

# Create a new set of x values for interpolation (finer resolution)
x_new = np.linspace(x.min(), x.max(), 300)

# Create the cubic B-spline (k=3 for cubic)
spl = make_interp_spline(x, y, k=3)

# Compute the new y values using the spline
y_new = spl(x_new)

# Plot the original data points and the interpolated curve
plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x_new, y_new, label='Cubic Spline Interpolation', color='blue')
plt.title('Cubic Spline Interpolation using make_interp_spline')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

#### Output

Here is the output of the make_interp_spline() function in Scipy −
![make_interp_spline Interpolation Example](/scipy/images/make_interp_spline_example.jpg)
## PchipInterpolator

The
**PchipInterpolator**in SciPy is a class used for Piecewise Cubic Hermite Interpolating Polynomials (PCHIP). It is particularly advantageous for ensuring that the interpolation maintains the shape and monotonicity of the data by making it suitable for applications where preserving the original data's characteristics is crucial.
### Syntax

Below is the syntax of using the
**PchipInterpolator()**class in scipy −
```
PchipInterpolator(x, y, extrapolate=False)
```

### Parameters

Following are the parameters of the
**PchipInterpolator()**class in scipy −
- **x**− 1D array of the x-coordinates of the data points. Must be strictly increasing.
- **y**− 1D array of the y-coordinates corresponding to x.
- **extrapolate**− If this parameter set to True then the interpolator will allow extrapolation beyond the boundaries of x. The default value is False.
### Example

Following is the example which shows how to use
**PchipInterpolator()**class for generating the spline interpolation −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Define sample data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 0, 1, 0])  # Example data with peaks and valleys

# Create the PCHIP interpolator
pchip_interpolator = PchipInterpolator(x, y)

# Generate new x values for interpolation
x_new = np.linspace(0, 4, 100)
y_new = pchip_interpolator(x_new)

# Plot the original data points and the PCHIP interpolated curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data Points', zorder=5)
plt.plot(x_new, y_new, label='PCHIP Interpolation', color='blue')
plt.title('PCHIP Interpolation Example')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
```

#### Output

Here is the output of the PchipInterpolator() class in Scipy −
![PCHIP Interpolation Example](/scipy/images/pchip_example.jpg)

---

## 32. SciPy - Grid Data Multi-Dimensional Interpolation

*Source: [https://www.tutorialspoint.com/scipy/scipy_grid_data_multi_dimensional_interpolation.htm](https://www.tutorialspoint.com/scipy/scipy_grid_data_multi_dimensional_interpolation.htm)*

---

---
[Previous](/scipy/scipy_spline_1d_interpolation.htm)[Quiz](/scipy/quiz_on_scipy_grid_data_multi_dimensional_interpolation.htm)[Next](/scipy/scipy_rbf_multi_dimensional_interpolation.htm)
## Grid Data Multi-Dimensional Interpolation

SciPy
**Grid Data Multi-Dimensional Interpolation**is a technique used to estimate the values of a function at arbitrary points in multi-dimensional space based on data that is known only at a finite set of points. This method is particularly useful when dealing with irregularly spaced data in applications such as geographic data analysis, image processing and scientific simulations.
In SciPy the core function for this type of interpolation is
**scipy.interpolate.griddata()**function. It supports multiple interpolation methods including nearest-neighbor, linear and cubic interpolation by allowing users to balance between accuracy and computational efficiency. The**griddata()**function works for datasets of any dimensionality by making it a flexible tool for multi-dimensional data interpolation.
### Syntax

Following is the syntax of
**griddata()**function which is used to perform**Grid Data Mutli- Dimensional Interpolation**in scipy −
```
scipy.interpolate.griddata(
   points, 
   values, 
   xi, 
   method='linear', 
   fill_value=np.nan, 
   rescale=False
)
```

### Parameters

Here are the parameters of the function
**griddata()**−
- **points**− An array of shape (n, D) where n is the number of known data points and D is the number of dimensions. Each row represents the coordinates of a known data point.
- **values**− An array of length n containing the values corresponding to each point in points.
- **xi**− The coordinates at which to interpolate the data. This can be a single point or an array of points. Its shape can be (m, D) where m is the number of points at which interpolation is desired.
- **method**− The interpolation method to use such as linear, nearest, cubic. The default value is linear.
- **fill_value**− Value to return for points outside the convex hull of the input points. This is useful for managing out-of-bounds queries.
- **rescale**− If this parameter set to True then the input points will be re-scaled to fit in the unit box before interpolation. This is particularly useful when the ranges of the dimensions of the input points differ significantly.
## Key Grid Data Interpolation Methods in Scipy

In SciPy the
**griddata()**function offers three primary interpolation methods for grid data. These methods are useful for interpolating scattered data points over a grid in one or more dimensions. Let's see them one by one in detail −
## Linear Interpolation in griddata()
**Linear interpolation**in**griddata()**is a method that estimates unknown values within the bounds of known data points using linear equations. It is widely used when scattered data points are provided and an interpolated surface or function is desired.
n the view of multi-dimensional interpolation with SciPys
**griddata()**the linear interpolation is used to compute values at new points based on the values of surrounding points by forming a surface composed of triangular facets.
Following are the key features of Linear interpolation in griddata() −

- **Method**− For 2D interpolation, the linear interpolation involves creating triangles in 2D between neighboring points and for higher dimensions. It constructs simplified generalized triangles.
- **Interpolation**− The value at any new point is calculated based on a weighted average of the values at the vertices of the surrounding simplex. This provides a simple linear interpolation of the data.
- **Boundary Conditions**− Linear interpolation assumes the function is well-behaved within the bounds of the known data points and does not extrapolate outside the convex hull of the input points.
To use the linear method in
**Grid Data Multi Dimensional Interpolation**we have to pass the argument**method = 'linear'**for**griddata()**function. Here is the syntax −
```
griddata(points, values, xi, method='linear')
```

### Example

Heres a simple example of linear interpolation using SciPy's
**griddata()**function where we interpolate values at new points based on known data points in 2D space −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Define known data points (in 2D) and their corresponding values
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])  # [x, y] coordinates
values = np.array([0, 1, 1, 0, 0.5])  # Function values at the points

# Create a grid of new points where we want to interpolate values
xi = np.linspace(0, 1, 100)
yi = np.linspace(0, 1, 100)
xi, yi = np.meshgrid(xi, yi)  # Generate a grid of points

# Perform linear interpolation using griddata
zi = griddata(points, values, (xi, yi), method='linear')

# Plot the results
plt.figure(figsize=(6, 6))
plt.contourf(xi, yi, zi, levels=20, cmap='coolwarm')  # Interpolated surface
plt.scatter(points[:, 0], points[:, 1], c=values, s=100, edgecolor='k', label='Known Data Points')  # Data points
plt.title('Linear Interpolation using griddata()')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Interpolated Value')
plt.legend()
plt.grid(True)
plt.show()
```

#### Output

Here is the output of Linear Method in Grid Data Interpolation −
![Linear Grid Data Interpolation Example](/scipy/images/linear_griddata_example.jpg)
## Nearest Neighbour Interpolation in griddata()
**Nearest-Neighbor Interpolation**in**griddata()**is a simple interpolation method where the value at any given point is assigned based on the nearest known data point. This method doesn't perform any interpolation between points; instead it finds the closest data point and assigns its value to the target point. This is a quick method especially useful for categorical or discontinuous data where smooth transitions are not required.
Here are the key features of Nearest Neighbour interpolation in griddata() −

- **Speed**− Nearest-neighbor interpolation is fast since it only requires finding the closest data point.
- **Use Cases**− This method is ideal for scenarios where precision isn't as important or when working with non-continuous or sparse datasets.
- **Limitations**− It produces abrupt transitions between different regions in the output by making it unsuitable for smooth datasets.
When we want to use the Nearest Neighbour method in
**Grid Data Multi Dimensional Interpolation**then we have to pass the argument**method = 'nearest'**for**griddata()**function. Here is the syntax of it −
```
griddata(points, values, xi, method='nearest')
```

### Example

In this example the
**griddata()**function is used to assign the values of the nearest points to all points on the grid. The plot shows how values are filled based on the nearest data point by creating blocky transitions −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Define known data points and their values
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # coordinates of known points
values = np.array([0, 1, 2, 3])  # function values at these points

# Create a grid of points where interpolation is needed
xi = np.linspace(0, 1, 50)
yi = np.linspace(0, 1, 50)
xi, yi = np.meshgrid(xi, yi)

# Perform Nearest-Neighbor Interpolation
zi = griddata(points, values, (xi, yi), method='nearest')

# Plot the interpolated result
plt.contourf(xi, yi, zi, levels=20, cmap='viridis')
plt.scatter(points[:, 0], points[:, 1], c=values, s=100, edgecolor='red', label='Known Points')
plt.colorbar(label='Interpolated Value')
plt.title('Nearest-Neighbor Interpolation using griddata()')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

#### Output

Following is the output of Nearest Neighbor Method in Grid Data Interpolation −
![Nearest Neighbour Grid Data Interpolation Example](/scipy/images/nearest_griddata_example.jpg)
## Cubic Interpolation in griddata()

In SciPy's
**griddata()**the**Cubic Interpolation**is a method used to interpolate data over a grid when the data points are scattered. Cubic interpolation generates smooth surfaces by fitting cubic polynomials to the data points. This method provides a higher degree of smoothness compared to linear interpolation and is especially useful when the dataset contains smooth variations.
Below are the key features of Cubic interpolation in griddata() −

- **Smooth Surface**− Cubic interpolation produces smooth, continuous surfaces by making it ideal for datasets that require smooth transitions.
- **Higher Accuracy**− It provides better accuracy compared to nearest-neighbor and linear interpolation methods when smoothness is a priority.
- **2D Only**− This method is currently available for two-dimensional interpolation and cannot be applied to higher-dimensional data directly..
To use cubic interpolation in griddata() we should simply set the method argument to 'cubic' −

```
griddata(points, values, xi, method='cubic')
```

### Example

Following is the example which shows the usage of the method cubic in
**griddata()**function of scipy −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Define the known data points (coordinates)
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
values = np.array([0, 1, 2, 3, 1.5])  # The values at the known points

# Define the grid where interpolation will be performed
grid_x, grid_y = np.mgrid[0:1:50j, 0:1:50j]

# Perform cubic interpolation
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# Plot the result
plt.imshow(grid_z.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
plt.scatter(points[:, 0], points[:, 1], c=values, edgecolor='k', label='Data Points')
plt.title('Cubic Interpolation using griddata')
plt.colorbar(label='Interpolated Value')
plt.legend()
plt.show()
```

#### Output

Following is the output of Cubic Method in Grid Data Interpolation −
![Cubic Grid Data Interpolation Example](/scipy/images/cubic_griddata_example.jpg)
## Applications

Here are the applications of the Grid Data Mutli-dimensional Interpolation in scipy −

- **Geographic Information Systems (GIS)**− Interpolating elevation or climate data over a geographic grid.
- **Image Processing**− Resampling and transforming pixel grids in images.
- **Physics & Engineering**− Interpolating sensor readings, temperature or pressure fields in simulations.
- **Data Analysis**− Filling in missing data points in multi-dimensional datasets.
## Advantages

When we use the Grid Data Mutli-Dimensional Interpolation it posses the advantages, which are listed below −

- **Flexibility**− Works for scattered data points that are not regularly spaced.
- **Multi-dimensional Support**− Handles data of arbitrary dimensions by allowing it to be used in a variety of domains.
- **Choice of Methods**− Offers different interpolation methods to balance between computational efficiency and accuracy
## Limitations

As we know when we use a particular method or process, it has the pro's as well as the con's. Here are the limitations of the Grid Data −

- **Computational Cost**− Cubic interpolation while smoother can be computationally expensive especially for large datasets.
- **Boundary Extrapolation**− For points outside the convex hull of the known data the results can be less reliable unless an appropriate**fill_value**is provided.
Finally we can conclude the SciPy's grid data multi-dimensional interpolation particularly via the griddata() function, is a powerful tool for estimating values in high-dimensional spaces where data may be sparse or irregularly spaced.

With multiple interpolation methods available the users can select the right approach based on the smoothness, accuracy and computational requirements of their specific problem.

---

## 33. SciPy - Radial Basis Function(RBF) Multi-Dimensional Interpolation

*Source: [https://www.tutorialspoint.com/scipy/scipy_rbf_multi_dimensional_interpolation.htm](https://www.tutorialspoint.com/scipy/scipy_rbf_multi_dimensional_interpolation.htm)*

---

---

## 34. SciPy - Curve Fitting

*Source: [https://www.tutorialspoint.com/scipy/scipy_curve_fitting.htm](https://www.tutorialspoint.com/scipy/scipy_curve_fitting.htm)*

---

---
[Previous](/scipy/scipy_spline_1d_interpolation.htm)[Quiz](/scipy/quiz_on_scipy_curve_fitting.htm)[Next](/scipy/scipy_linear_curve_fitting.htm)**Curve fitting**is the process of constructing a mathematical function that best approximates a set of data points. In SciPy the**curve_fit()**function from the**scipy.optimize**module is commonly used to fit a given model which typically nonlinear to the data. The goal is to find the optimal parameters of the model that minimize the differences i.e. residuals between the data and the model's predicted values.
## How Curve Fitting Works?

As we know
**Curve fitting**is the process of finding a curve that best describes the relationship between a set of data points. In detail, heres how it works −
- **Data Points**− The data consists of independent variable(s)  and dependent variable . These points represent observations or measurements.
- **Model Selection**− Choose a mathematical model such as linear, polynomial, exponential that represents the underlying trend. The model has parameters that need to be determined to fit the data.
- **Objective**− The objective is to find the parameters of the model such that the curve minimizes the difference i.e. error between the predicted values() and the observed data .
- **Error Calculation**− The error can be calculated using a loss function which is typically the sum of squared errors (SSE) given as follows −
```
SSE = (observed - model)2
```
-)
This ensures that larger differences contribute more heavily to the total error.

- **Optimization**− An optimization technique such as least squares is applied to adjust the model parameters to minimize the error.
- **Curve Fitting Tools**− SciPys**curve_fit()**function is widely used. It takes the model function, data and an initial guess for the parameters and returns the optimal parameters that fit the data.
## Types of Curve fittings

There are several types of curve fitting techniques depending on the nature of the data and the model used. These can range from simple linear fits to more complex nonlinear models. Here are the main types of curve fitting used in SciPy −
Type of Curve FittingDefinitionModel EquationUse CaseLinearA method that fits a straight line to data points, showing a constant rate of change between variables.y = ax + bWhen the relationship is approximately linear.PolynomialA method that uses a polynomial equation to fit data points, capturing more complex relationships.y = ax+ ax+ ... + aFor smooth, nonlinear relationships (e.g., quadratic, cubic).ExponentialA fitting method used when data exhibits exponential growth or decay, commonly seen in natural phenomena.y = a e+ cData that grows or decays exponentially.LogarithmicA fitting method that models relationships where the rate of change decreases over time.y = a log(x) + bWhen growth slows down as the independent variable increases.Power LawA method that models data where one variable changes as a power of another variable, often in scale-invariant phenomena.y = axWhen a variable changes as a power of another variable.Sigmoidal (Logistic)A method that fits an S-shaped curve, commonly used for growth processes that saturate.y = a / (1 + e)For S-shaped curves in growth or saturation models.GaussianA fitting method that models data with a bell-shaped distribution, typical in statistical analyses.y = a e/ (2c)For bell-shaped data distributions (normal distribution).Rational FunctionA method that fits a ratio of polynomial equations to describe more complex relationships between variables.y = (ax+ ... + a) / (bx+ ... + b)For more complex relationships that require ratios of polynomials.
## Understanding the Results

Understanding the results of curve fitting involves interpreting the output from the fitting process and evaluating how well the model describes the data.

- **Fitted Parameters**− The estimated parameters minimize the difference between the actual data and the predicted values of the model function.
- **Covariance Matrix**− The covariance matrix gives insights into the accuracy of the parameter estimates. Diagonal elements represent the variance of each parameter by helping us understand the confidence in the fitted values.
## Curve Fitting vs Interpolation
**Curve fitting**focuses on modeling the relationship between variables for predictive purposes while interpolation is concerned with estimating values between known data points by ensuring precision within the observed range. The choice between the two techniques depends on the goals of the analysis and the nature of the data.
Here's a comparison of curve fitting and interpolation −
AspectCurve FittingInterpolationDefinitionFinds a curve that best represents the relationship between data points.Estimates values between known data points.PurposeModels underlying trends for predictions and insights.Provides precise estimates within the known range.MethodsLeast squares fitting, polynomial fitting, regression analysis.Linear interpolation, polynomial interpolation, spline interpolation.Mathematical ApproachMinimizes error between the fitted curve and data points.Constructs new points ensuring the function passes through known data.ApplicabilityUsed for understanding relationships and generalization (e.g., regression).Used for precise value estimation (e.g., numerical methods).ExtrapolationAllows extrapolation beyond the observed data range, but with caution.Does not involve extrapolation but estimates are only within known data range.OutputA smooth curve that may not pass through all data points.A curve that passes exactly through all known data points.
## SciPy Function for Curve Fitting

We have the function
**scipy.optimize.curve_fit()**to perform the curve fitting in scipy.S.NoFunction & Description1[scipy.optimize.curve_fit()](/scipy/scipy_optimize_curve_fit_function.htm)
Fits a function to data using non-linear least squares. Allows users to define custom models for fitting and returns the optimal parameters that best describe the data.
### Example

Following is the example which demonstrates how to fit an exponential decay function to data using curve_fit() function −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the model function (exponential decay)
def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate synthetic data (noisy exponential decay)
x_data = np.linspace(0, 4, 50)
y_data = model_func(x_data, 2.5, 1.3, 0.5) + np.random.normal(0, 0.2, size=x_data.shape)

# Perform curve fitting
# Initial guess for parameters a, b, c is optional but can speed up convergence
params, params_covariance = curve_fit(model_func, x_data, y_data, p0=[2, 1, 0])

# Generate data from the fitted curve
y_fitted = model_func(x_data, *params)

# Plot original noisy data and the fitted curve
plt.scatter(x_data, y_data, label='Data', color='red')
plt.plot(x_data, y_fitted, label='Fitted Curve', color='blue')

# Add labels and title
plt.title('Exponential Decay Curve Fitting using SciPy')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print the fitted parameters
print(f"Fitted parameters: a = {params[0]}, b = {params[1]}, c = {params[2]}")
```

#### Output

Following is the output of curve fitting in scipy −
![Curve fitting Example](/scipy/images/curve_fitting_example.jpg)

---

## 35. SciPy - Linear Curve Fitting

*Source: [https://www.tutorialspoint.com/scipy/scipy_linear_curve_fitting.htm](https://www.tutorialspoint.com/scipy/scipy_linear_curve_fitting.htm)*

---

---
[Previous](/scipy/scipy_curve_fitting.htm)[Quiz](/scipy/quiz_on_scipy_linear_curve_fitting.htm)[Next](/scipy/scipy_non_linear_curve_fitting.htm)**Linear Curve Fitting**is a fundamental statistical technique used to model the relationship between two variables by fitting a linear equation to the observed data. In the view of**SciPy linear curve fitting**typically involves in minimizing the differences i.e., residuals between observed data points and those predicted by a linear model.
Lets see in detail about
**Linear Curve Fitting**mathematical foundation and the tools provided by SciPy.
## Understanding Linear Curve Fitting

Understanding linear curve fitting involves grasping the fundamental concepts, methods and applications of fitting a linear model to a dataset. Here are the key aspects of linear curve fitting.

### Linear Relationship

A linear relationship implies that changes in one variable result in proportional changes in another variable. Mathematically it can be expressed as follows −

```
y = mx+b
```

Where −

- **y**is the dependent variable i.e. what we are trying to predict or explain.
- **x**is the independent variable i.e., the input or predictor.
- **m**is the slope of the line that indicates how much y changes for a one-unit change in x.
- **b**is the y-intercept i.e., the value of y when x=0.
### Objectives of Linear Fitting

The goal of linear curve fitting is to find the best-fitting line that minimizes the sum of the squared residuals which is given as follows −
![Objective Linear curve](/scipy/images/objective_linear_curve.jpg)
Where −

- **S**is the sum of squared differences (residuals).
- **y**are the observed values.
- **ax**are the predicted values from the linear model.+b
Here are the objectives to achieve the goal of linear curve fitting −

- **Understanding Relationships**− Linear fitting helps to determine how changes in an independent variable (x) affect a dependent variable (y). This understanding can reveal underlying trends and patterns in empirical data which can be critical for decision-making processes.
- **Prediction**− Once a linear relationship is established then the model can be used to predict the value of y for any given x. This is particularly useful in fields such as economics, finance and natural sciences where forecasting based on historical data is essential.
- **Modeling**− Linear models provide a simple yet effective way to approximate the behavior of a system. They serve as a foundation for more complex models by allowing researchers to build on them for more nuanced analyses.
- **Statistical Inference**− By analyzing the slope, intercept and goodness-of-fit statistics such asand p-values where one can draw conclusions about the reliability and validity of the model. This helps in understanding whether the observed relationships are statistically significant or likely due to chance.
- **Error Minimization**− The least squares method is commonly employed to find the optimal parameters i.e., slope and intercept that minimize the residuals which is the differences between actual and predicted values. This ensures the best possible fit of the linear model to the data.
- **Understanding Variability**− Understanding variability helps to assess the effectiveness of the linear model. By calculating metrics such asanalysts can determine the proportion of variance in y that is explained by x by providing insights into the model's explanatory power.
- **Diagnostic Insights**− Analyzing residuals and other diagnostic metrics can reveal whether the linear model is appropriate for the data or if assumptions such as linearity, homoscedasticity and normality are violated which may suggest the need for alternative modeling approaches.
## Methods for Linear Fitting in SciPy

SciPy provides several robust methods for performing linear fitting each with its unique features, advantages and use cases. Below is a detailed overview of the methods for linear fitting in SciPy −

### scipy.stats.linregress()
**scipy.stats.linregress()**is a function in the SciPy library used for performing simple linear regression analysis. It fits a linear model to a set of data points by providing various statistics that describe the relationship between the independent variable x and the dependent variable y. This function is particularly useful for quickly obtaining the slope, intercept, correlation coefficient, p-values and standard errors associated with the linear regression.
#### Syntax

Following is the syntax of using the
**scipy.stats.linregress()**function −
```
scipy.stats.linregress(x, y)
```

Following are the parameters of the
**scipy.stats.linregress()**function −
- **x(array)**− The independent variable data i.e., predictor. This should be a 1D array or list of values.
- **y(array)**− The dependent variable data i.e., response. This should also be a 1D array or list of values with the same length as x.
#### Example

Heres a simple example showing how to use
**scipy.stats.linregress()**function to perform linear regression and visualize the results −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.2, 6.1, 8.0, 10.3])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Print the results
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")  # R-squared value
print(f"P-value: {p_value}")
print(f"Standard Error: {std_err}")

# Create a line for plotting
x_fit = np.linspace(1, 5, 100)
y_fit = slope * x_fit + intercept

# Plot the data points and the fitted line
plt.scatter(x, y, label='Data Points', color='red')
plt.plot(x_fit, y_fit, label='Fitted Line', color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Example using scipy.stats.linregress')
plt.legend()
plt.grid()
plt.show()
```

##### Output

Below is the output of the
**scipy.stats.linregress()**function −
```
Slope: 2.0200000000000005
Intercept: 0.0799999999999983
R-squared: 0.9988250269264667
P-value: 1.709951883244442e-05
Standard Error: 0.039999999999992895
```
![Linear curve Regression](/scipy/images/linear_curve_linearreg.jpg)
### scipy.optimize.least_squares()
**scipy.optimize.least_squares()**is a function in the SciPy library that performs nonlinear least squares optimization. This function is used to minimize the sum of the squares of residuals between observed and modeled data by making it ideal for fitting models to data in various fields such as statistics, engineering and machine learning.
#### Syntax

Following is the syntax of using the
**scipy.optimize.least_squares()**function −
```
scipy.optimize.least_squares(fun, x0, args=(), jac='2-point', bounds=(-inf, inf), method='trf', 
                              x_scale='jac', ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=None, 
                              verbose=0, **options)
```

Following are the parameters of the
**scipy.optimize.least_squares()**function −
- **fun(callable)**− The objective function to minimize. It should return an array of residuals f(x) where x is the vector of parameters.
- **x0(array-like)**− Initial guess for the parameters to be optimized.
- **args(tuple, optional)**− Extra arguments passed to the objective function fun.
- **jac({'2-point, '3-point, 'cs, callable}, optional)**− The Jacobian matrix of the objective function. This can be provided explicitly or approximated using finite differences. The default is '2-point' which uses a two-point finite difference approximation.
- **bounds(sequence, optional)**− Bounds on the parameters. It should be provided as a tuple of two arrays, (min, max) where each array has the same length as x0.
- **method({'trf, 'dogbox, 'lm}, optional)**− The algorithm to use for optimization. The 'trf' and 'dogbox' methods are suitable for large problems while 'lm' is a Levenberg-Marquardt algorithm for smaller problems.
- **x_scale ({jac, linear, array-like}, optional)**− Scaling of the variables. This can improve convergence.
- **ftol, xtol, gtol (float, optional)**− The tolerances for termination. This algorithm will stop when these tolerances are met.
- **max_nfev (int, optional)**− Maximum number of function evaluations.
- **verbose (int, optional)**− Level of output. Use 1 for a summary, 2 for more detailed information.
- ****options (optional)**− Additional options specific to the chosen method.
#### Example

Here's an example which shows how to use
**scipy.optimize.least_squares()**to fit a nonlinear model to data −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Example data
x_data = np.linspace(0, 10, 100)
y_data = 3 * np.sin(x_data) + np.random.normal(size=x_data.size)

# Define the model function
def model(x, a, b):
    return a * np.sin(b * x)

# Define the residuals function
def residuals(params, x, y):
    return y - model(x, *params)

# Initial guess for parameters (a, b)
initial_params = [1.0, 1.0]

# Perform least squares fitting
result = least_squares(residuals, initial_params, args=(x_data, y_data))

# Get the optimal parameters
optimal_a, optimal_b = result.x

# Generate fitted values for plotting
y_fit = model(x_data, optimal_a, optimal_b)

# Plot the data and the fitted curve
plt.scatter(x_data, y_data, label='Data', color='red', alpha=0.5)
plt.plot(x_data, y_fit, label='Fitted Curve', color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Nonlinear Least Squares Fitting Example')
plt.legend()
plt.grid()
plt.show()

# Print the optimal parameters
print(f"Optimal parameters: a = {optimal_a}, b = {optimal_b}")
```

##### Output

Below is the output of the
**scipy.optimize.least_squares()**function −
```
Optimal parameters: a = 2.863938083609976, b = 0.9978215567742089
```
![Linear curve Least square](/scipy/images/linear_curve_leastsquare.jpg)
### scipy.optimize.minimize()

The
**scipy.optimize.minimize()**function is a versatile tool in the SciPy library used for minimizing scalar or multi-dimensional functions. It can handle a variety of optimization problems from simple unconstrained problems to complex constrained optimization tasks.
#### Syntax

Following is the syntax of using the
**scipy.optimize.minimize()**function −
```
scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, bounds=None, constraints=(), 
                         tol=None, options=None)
```

Following are the parameters of the
**scipy.optimize.minimize()**function −
- **fun(callable)**− The objective function to minimize.
- **x0(array-like)**− Initial guess for the parameters.
- **args(tuple, optional)**− Additional arguments to pass to fun.
- **bounds(sequence, optional)**− Bounds on the parameters.
- **method(optional)**− The optimization method to use such as 'BFGS', 'L-BFGS-B', etc.
- ****options (optional)**− Additional options specific to the chosen method.
#### Example

Below is a simple example of how to use
**scipy.optimize.minimize()**to minimize a quadratic function −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2  # f(x, y) = x^2 + y^2

# Initial guess
x0 = np.array([1, 1])

# Perform minimization
result = minimize(objective_function, x0)

# Print the results
print("Optimal value:", result.x)
print("Objective function value at optimal:", result.fun)

# Visualize the optimization process
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = objective_function([X1, X2])

plt.contour(X1, X2, Z, levels=20)
plt.scatter(result.x[0], result.x[1], color='red')  # Optimal point
plt.title('Contour plot of the objective function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
```

##### Output

Below is the output of the
**scipy.optimize.minimize()**function which is used to minimize the linear curve −
```
Optimal value: [-1.07505143e-08 -1.07505143e-08]
Objective function value at optimal: 2.311471135620994e-16
```
![Linear curve Minimize](/scipy/images/linear_curve_minimize.jpg)

---

## 36. SciPy - Non-Linear Curve Fitting

*Source: [https://www.tutorialspoint.com/scipy/scipy_non_linear_curve_fitting.htm](https://www.tutorialspoint.com/scipy/scipy_non_linear_curve_fitting.htm)*

---

---
[Previous](/scipy/scipy_linear_curve_fitting.htm)[Quiz](/scipy/quiz_on_scipy_non_linear_curve_fitting.htm)[Next](/scipy/scipy_input_output.htm)
SciPy's non-linear curve fitting is a powerful tool in Python for estimating the parameters of a non-linear model to best fit a given set of data. This method is commonly used to model data when the relationship between the independent variable x and the dependent variable y is not a straight line.

## Non-Linear Curve Fitting in SciPy

SciPy's
**optimize.curve_fit**function from the**scipy.optimize**module is the main tool for non-linear curve fitting. Heres how it works −
## Defining the Model Function

In non-linear curve fitting with SciPy the model function is a mathematical expression that represents the relationship between the independent variable(s) and the dependent variable(s) in our data. Its the function that we aim to fit to our data by adjusting its parameters to best match the observed values.

The model function should be defined explicitly in Python as a standard function. When using SciPys
**curve_fit()**function this model function is passed as an argument by allowing**curve_fit()**function to optimize the function's parameters to fit the data.
### Characteristics of the Model Function

Following are the characteristics of the Model Function −

- **Non-linear form**− The function is often non-linear with respect to its parameters. The most common non-linear models include exponential functions, power laws, logistic functions and other complex mathematical relationships.
- **Parameters**− This function must include parameters that**curve_fit()**function will optimize. These parameters are the variables that determine the shape of the function such as amplitude, decay rate, etc.
- **Input and Output**− This function must take at least two arguments namely, the**independent variables**which are typically given as arrays and the**parameters to be optimized**which are given as separate arguments.
## Using the Model Function in curve_fit

### Example

To perform non-linear curve fitting with
**curve_fit()**function in which we can pass this model function as an argument along with our data as follows −
```
from scipy.optimize import curve_fit
import numpy as np

def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Sample data
x_data = np.linspace(0, 4, 50)
y_data = model_func(x_data, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x_data))

# Fit the curve
initial_guesses = [1.0, 1.0, 1.0]
popt, pcov = curve_fit(model_func, x_data, y_data, p0=initial_guesses)
print(popt)
print(pcov)
```

#### Output

Following is the output of the above code −

```
[2.373493   1.3085931  0.55674186]
[[ 0.00991781  0.00433252 -0.00052527]
 [ 0.00433252  0.01414914  0.00379678]
 [-0.00052527  0.00379678  0.00191205]]
```

## Provide Initial Parameter Estimates

In non-linear curve fitting the initial parameter estimates or initial guesses are starting values for the model parameters that help the optimization algorithm converge to the best fit. Since non-linear fitting involves iterative optimization by having reasonable initial guesses can significantly affect the speed and success of the fitting process.

### Why Initial Estimates Are Important?

Here are the reasons why initial estimates are important −

- **Convergence**− Non-linear optimization methods like the one used by**curve_fit()**function may not converge to the correct solution without good starting points.
- **Efficiency**− Good estimates can reduce the number of iterations required by making the fitting faster.
- **Avoiding Local Minima**− Some models can have multiple solutions so an appropriate initial guess can help the algorithm avoid "getting stuck" in a local minimum instead of finding the global best fit.
### How to Choose Initial Parameter Estimates?

Choosing initial values can vary depending on the model and data but here are some general approaches −

- **Estimate from Data**− Use prior knowledge or approximate values based on our data. For instance let's consider −
- 
If fitting an exponential decay curve
**y = a.e**the initial value y at x = 0 could guide the choice for a.+c
- 
For linear or polynomial models then use the first few data points to estimate the slope or intercept.

- **Try Common Starting Values**− For some parameters typical values like 1.0, 0.0 or other neutral values can be a reasonable start if there is no strong prior information.
- **Trial and Error**− If a fit doesnt converge or gives a poor result, experiment with different starting values until we find a set that works well.
## Implementing Initial Estimates with curve_fit()

### Example

Here is the example of implementing initial estimates with the
**curve_fit()**function −
```
import numpy as np
from scipy.optimize import curve_fit

# Define the model function
def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate sample data
x_data = np.linspace(0, 4, 50)
y_data = model_func(x_data, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x_data))

# Initial guesses for parameters: a, b, c
initial_guesses = [2.0, 1.0, 0.5]  # Reasonable starting values based on the data

# Fit the model to the data
popt, pcov = curve_fit(model_func, x_data, y_data, p0=initial_guesses)

# Output the optimized parameters
print("Optimized parameters:", popt)
```

#### Output

Here is the output of implementing initial estimates with
**curve_fit()**function −
```
Optimized parameters: [2.44641902 1.25183589 0.51762444]
```

## Fit the Curve

Using
**curve_fit()**function we can fit the curve. This function tries to find the best parameters that minimize the difference between the model predictions and the actual data.
### Example

Here is the example of using
**curve_fit()**function to fit an exponential decay function −
```
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define a non-linear function (exponential decay)
def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate synthetic data (for illustration)
x_data = np.linspace(0, 4, 50)
y_data = model_func(x_data, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x_data))

# Fit the data to the model
initial_guesses = [1.0, 1.0, 1.0]  # Initial parameter guesses for a, b, c
popt, pcov = curve_fit(model_func, x_data, y_data, p0=initial_guesses)

# Extract the fitted parameters
a_fitted, b_fitted, c_fitted = popt

# Plot the data and the fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, model_func(x_data, *popt), color='red', label='Fitted curve')
plt.legend()
plt.show()
```

#### Output

Here is the output of fitting the non linear curve with the help of
**curve_fit()**function −![Non Linear Curve Fitting](/scipy/images/non_linear_curve_fitting.jpg)
## Obtain Fitted Parameters

The
**curve_fit()**function returns the optimal parameters along with a covariance matrix that provides information about the fit's accuracy.
### Example

Here is the example of getting the fitted parameters of the non linear curve with the help of
**curve_fit()**function −
```
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the model function
def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate synthetic data
x_data = np.linspace(0, 4, 50)
y_data = model_func(x_data, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x_data))

# Fit the model to the data
initial_guesses = [1.0, 1.0, 1.0]
popt, pcov = curve_fit(model_func, x_data, y_data, p0=initial_guesses)

# Retrieve the fitted parameters
a_fitted, b_fitted, c_fitted = popt

# Display the fitted parameters
print(f"Fitted parameters:\na = {a_fitted}\nb = {b_fitted}\nc = {c_fitted}")
```

#### Output

Here are the output of the fitted parameters obtained with the help of
**curve_fit()**function −
```
Fitted parameters:
a = 2.5673199143371517
b = 1.3548808337833609
c = 0.5102248520438042
```

---

## 37. SciPy - Input and Output

*Source: [https://www.tutorialspoint.com/scipy/scipy_input_output.htm](https://www.tutorialspoint.com/scipy/scipy_input_output.htm)*

---

---
[Previous](/scipy/scipy_non_linear_curve_fitting.htm)[Quiz](/scipy/quiz_on_scipy_input_output.htm)[Next](/scipy/scipy_reading_writing_files.htm)
SciPy is a Python library used for scientific and technical computing and it includes a range of input and output functions especially for handling various data formats.

The
**SciPy input and output (I/O)**functions which enable reading and writing data in various scientific formats. The**scipy.io**module supports loading and saving MATLAB files with loadmat and savemat, reading and writing text files using NumPys loadtxt and savetxt and handling WAV audio files through**wavfile.read**and**wavfile.write**.
This also supports Fortran binary files and IDL
**.sav**files with FortranFile and readsav. These I/O functions make SciPy versatile for data exchange across different formats commonly used in scientific computing and analysis.
Heres an overview of the main features and functions −

## MATLAB Files (.mat files)

SciPy provides functionality for reading from and writing to MATLAB
**.mat**files through the**scipy.io**module which is especially useful for users working with data in both Python and MATLAB.**.mat**files are binary files that store MATLAB variables and SciPys loadmat and savemat functions handle these files efficiently.
Here are the commonly used SciPy functions to use the .mat files −
S.NoFunctionDescription1[scipy.io.loadmat](/scipy/scipy_loadmat_function.htm)Reads**.mat**files and returns data in the form of a Python dictionary2[scipy.io.savemat](/scipy/scipy_savemat_function.htm)Saves Python data structures (like dictionaries) to a**.mat**file.3[scipy.io.whosmat](/scipy/scipy_whosmat_function.htm)Lists the variables stored in a**.mat**file without loading the actual data.
## Text and Binary File I/O

SciPy provides Functions for reading and writing data in text and binary formats such as .txt, .dat, and .csv files. These Functions rely on NumPys loadtxt, savetxt and genfromtxt methods.

- **numpy.loadtxt:**Loads data from a text file.
- **numpy.savetxt:**Saves an array to a text file.
- **numpy.genfromtxt:**Similar to loadtxt but allows for more flexible parsing such as handling missing values.
## Wav Files

In SciPy WAV files can be read and written using Functions in the
**scipy.io.wavfile**module which is ideal for simple audio data processing tasks. WAV files can contain multiple channels such as mono, stereo and various bit depths but SciPys Functions handle these with basic data type conversion.
Here are the SciPy Functions for handling the
**.wav**files −S.No.FunctionDescription1[scipy.io.wavfile.write](/scipy/scipy_write_function.htm)Writes data to a WAV file with a specified sample rate.2[scipy.io.wavfile.read](/scipy/scipy_read_function.htm)Reads a WAV file and returns the sample rate in samples per second and the audio data as a NumPy array.
## Fortran and IDL Files
**Fortran and IDL files**are types of data files commonly used in scientific computing especially in fields like physics, engineering and remote sensing. They originate from the programming languages Fortran and IDL (Interactive Data Language) respectively.
Fortran binary files can be read and written with the FortranFile class in
**scipy.io**module. This class allows for low-level binary I/O in the format commonly used by Fortran programs.
SciPy also supports reading IDL
**.sav files**which are used to store data in the IDL (Interactive Data Language) format. The**scipy.io.readsav()**Function reads IDL files and returns a dictionary with variable names as keys.S.No.FunctionDescription1[scipy.io.FortranFile](/scipy/scipy_fortranfile_function.htm)Used for reading and writing binary data in Fortran.2[scipy.io.readsav](/scipy/scipy_readsav_function.htm)Reads IDL**.sav**files into Python.
## Image Files

In SciPy basic image file I/O is possible through the misc module and but for more advanced image processing libraries like imageio, PIL (Pillow) or OpenCV are often preferred. SciPys Functions allow basic reading and writing for formats like .png or .jpg.
S.No.FunctionDescription1scipy.misc.faceLoads an image of a "face" that comes with the SciPy library for testing and demonstration purposes.

---

## 38. SciPy - Reading and Writing Files

*Source: [https://www.tutorialspoint.com/scipy/scipy_reading_writing_files.htm](https://www.tutorialspoint.com/scipy/scipy_reading_writing_files.htm)*

---

---
[Previous](/scipy/scipy_input_output.htm)[Quiz](/scipy/quiz_on_scipy_reading_writing_files.htm)[Next](/scipy/scipy_working_with_different_file_formats.htm)
SciPy is primarily used for scientific and mathematical computing but it also offers functionalities that can help with reading and writing certain file formats especially in scientific data.

In SciPy
**reading and writing**files is handled primarily through NumPy's file I/O functions. To save data we use**np.savetxt()**for text files or**np.save()**for binary .npy files which efficiently store arrays. Loading these files is as simple as**np.loadtxt()**and**np.load()**respectively.
For MATLAB files the SciPy provides
**scipy.io.savemat**and**scipy.io.loadmat**. For other formats like .wav for audio**scipy.io.wavfile.read()**and**scipy.io.wavfile.write()**are available. SciPy supports working with sparse matrices via scipy.sparse, with**scipy.io.mmread()**and**scipy.io.mmwrite()**for Matrix Market formats by facilitating efficient file handling across formats.
Below are some of the common ways SciPy is used to read and write files which focuses on working with data formats often encountered in scientific computing −

## Working with .mat Files (MATLAB files)
**MAT files**are data files used by MATLAB which is a high-level programming language and environment for numerical computation and visualization. The**.mat**file format is designed to store variables, arrays and other data structures in a way that MATLAB can easily read and write.
MAT files can be read and written by MATLAB as well as by other programming languages like Python uses libraries like SciPy and R.

SciPy provides tools for working with MATLAB
**.mat**files by enabling Python users to exchange data with MATLAB users. These files are handled through the**scipy.io**module.
### Loading .mat Files

To load
**.mat**files in Python we can use the loadmat function from SciPys io module. This function reads**..mat**files and converts them into a Python dictionary by allowing us to access MATLAB variables in Python.
Here is the step by step guide to load the .mat file with the help of scipy −

- First we have to import the**loadmat**function.
- After that we have to specify the path of the**.mat**file.**loadmat**reads the file and returns a dictionary where the keys are the variable names and the values are the data.
- Next access variables in the**.mat**file by referring to the dictionary keys.
### Example

Here is the example which loads the
**.mat**file with the help of**loadmat()**function of the Scipy library −
```
from scipy.io import loadmat

# loading the .mat file from the local drive
data = loadmat('/files/array_file.mat')

# Access a variable named 'my_array' in the MATLAB file
my_array = data['my_array']
print(my_array)
```

Here is the output after loading the
**.mat**file with the help of**loadmat()**function −
```
[[1 2 3]
 [4 5 6]]
```

### Writing into .mat files

To write into
**.mat**files in Python we can use the savemat function from SciPys io module. This function saves data as a**.mat**file by allowing Python data to be shared with MATLAB or other programs that support this format.
Following are the steps to be followed to write the data into the
**.mat**file −
- **Import the function:**First we have to import the**savemat**function from the**scipy.io**module.
- **Prepare the Data:**The data to be saved should be in the form of a dictionary, where each key is the variable name (as it should appear in MATLAB) and each value is the data to be saved (usually as a NumPy array or other serializable object).
- **Save the Data to a .mat File:**Use**savemat**to specify the filename and data dictionary.
```
from scipy.io import savemat
import numpy as np

# Data to save
data = {
    'array1': np.array([1, 2, 3]),
    'matrix1': np.array([[1, 2], [3, 4]])
}
# Save data to a .mat file
savemat('/files/written_matfile.mat', {'my_array': data})
print("Data written into the Mat file")
```

Here is the output after writing into the
**.mat**file with the help of**savemat()**function −
```
Data written into the Mat file
```

## Reading and Writing .npz and .npy Files
**.npy**and**.npz**are file formats used by NumPy to store arrays efficiently in binary format. They are commonly used for saving and loading data in Python particularly for handling large arrays in a compact, fast-access format.
The
**.npy**file include metadata such as data type and shape, to enable efficient and accurate reconstruction of the array when loaded and In**.npz**file each array is stored as a separate .npy file within the archive with keys for access.
### Writing into .npy files

To write into
**.npy**files in Python using NumPy we can use the**np.save()**function. This function stores a single NumPy array in a binary format with metadata by making it efficient for saving large datasets.
Here are the steps that to be followed to write the data into the
**.npy**file −
- **Import Numpy:**First we need to import NumPy.
- **Prepare the Data:**We need to have a NumPy array that we want to save.
- **Save the Array:**We have to**np.save()**function to save the array to a**.npy**file.
```
import numpy as np

# Create a NumPy array
array_data = np.array([1, 2, 3, 4, 5])

# Save the array to a .npy file
np.save('/files/written_npyfile.npy', array_data)
print("Data saved to the npy file")
```

Here is the output after writing into the
**.npy**file with the help of**np.save()**function −
```
Data saved to the npy file
```

### Reading the .npy files

To read
**.npy**files in Python we have to use the**np.load()**function from NumPy. This function loads a**.npy**file by restoring the array with its original shape and data type.
Below are the steps that to be followed to read the data from the
**.npy**file −
- **Import Numpy:**First we need to import NumPy.
- **Load the Data:**We have to use**np.load()**function to read the .npy file by specifying the filename.
```
import numpy as np

# Load a .npy file
array = np.load('/files/written_npyfile.npy')
print(array)
```

Here is the output after Reading the
**.npy**file with the help of**np.load()**function −
```
[1 2 3 4 5]
```

### Reading the .npz files

To read
**.npz**files in Python we can use the**np.load()**function from NumPy. An**.npz**file is essentially a compressed archive containing multiple**.npy**files with each corresponding to a separate array. When we load an**.npz**file it returns a NpzFile object which behaves like a dictionary. Each array inside the**.npz**file can be accessed by its corresponding key.
Below are the steps that to be followed to read the data from the
**.npz**file −
- **Import NumPy:**First we must import NumPy library.
- **Load the .npz file:**For loading the .npz file we have to use the function
- **Access the Arrays:**Access the individual arrays inside the**.npz**file by their keys which are the variable names used when the**.npz**file was saved.
```
import numpy as np

# Load the .npz file
data = np.load('/files/data_arrays.npz')

# Access individual arrays using their keys
array1 = data['array1']
array2 = data['array2']

# Print the arrays
print(array1)
print(array2)

# Print all keys in the .npz file
print(data.files)
```

Following is the output after Reading the
**.npz**file with the help of**np.load()**function −
```
[1 2 3]
[[4 5]
 [6 7]]
['array1', 'array2']
```

### Writing the .npz files

To read
**.npz**files in Python we can use the**np.load()**function from NumPy. An**.npz**file is essentially a compressed archive containing multiple**.npy**files with each corresponding to a separate array. When we load an**.npz**file it returns a NpzFile object which behaves like a dictionary. Each array inside the**.npz**file can be accessed by its corresponding key.
Below are the steps that to be followed to read the data from the
**.npz**file −
- **Import NumPy:**First we should make sure that we have NumPy imported in our script
- **Prepare the data:**We can store multiple arrays in a**.npz**file. Each array is stored with a unique name which are similar to dictionary key-value pairs.
- **Save the Data to a .npz File:**We can use the**np.savez()**function to save multiple arrays to a**.npz**file.
> Alternatively we can use np.savez_compressed() function to compress the .npz file to reduce the file size
**np.savez_compressed()**function to compress the**.npz**file to reduce the file size
```
import numpy as np

# Create two NumPy arrays
array1 = np.array([1, 2, 3])
array2 = np.array([[4, 5], [6, 7]])

# Save arrays into a .npz file
np.savez('/files/data_arrays.npz', array1=array1, array2=array2)

# Alternatively, to save with compression
np.savez_compressed('/files/data_arrays_compressed.npz', array1=array1, array2=array2)
print("Files saved as with compression and without compression")
```

Following is the output of writing into the
**.npz**files without file compression and with compression −
```
Files saved as with compression and without compression
```

## Working with Sparse Matrices
**Working with sparse matrices**is an important concept when dealing with large datasets where most of the elements are zero. Sparse matrices are memory-efficient as they only store the non-zero elements and their positions rather than the entire matrix. SciPy provides various functions for working with sparse matrices.
### Sparse Matrix Formats in SciPy

SciPy offers several sparse matrix formats each optimized for different types of operations −

- **CSR (Compressed Sparse Row):**Efficient for row slicing and matrix-vector products.
- **CSC (Compressed Sparse Column):**Efficient for column slicing and matrix-vector products.
- **COO (Coordinate List):**Efficient for constructing sparse matrices and for quick insertions of elements.
- **LIL (List of Lists):**Efficient for constructing sparse matrices incrementally.
- **DIA (Diagonal):**Efficient for diagonal matrices.
- **BSR (Block Sparse Row):**Efficient for block-sparse matrices.
### Saving and Loading Sparse Matrices

We can save and load sparse matrices using
**scipy.sparse**module and the**.npz**format since .npz files can store multiple arrays including sparse matrix formats.
#### Saving Sparse Matrices

To save sparse matrices we use
**scipy.sparse.save_npz()**which saves a sparse matrix to a**.npz**file. The function takes two arguments namely the filename and the sparse matrix to save. Following is the example which saves the sparse matrix −
```
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

# Create a sparse matrix (CSR format)
data = np.array([1, 2, 3, 4])
row_indices = np.array([0, 1, 2, 3])
col_indices = np.array([0, 1, 2, 3])
sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(4, 4))

# Save the sparse matrix to a .npz file
save_npz('/files/sparse_matrix.npz', sparse_matrix)
print("Done saving the sparse matrices")
```

Following is the output of saving the sparse matrix using the scipy library −

```
Done saving the sparse matrices
```

#### Loading Sparse Matrices

To load a sparse matrix we use
**scipy.sparse.load_npz()**. This function loads a sparse matrix stored in**.npz**format and returns it in the correct sparse matrix format such as CSR, CSC, etc. Here is the example of loading the sparse matrix −
```
from scipy.sparse import load_npz

# Load the sparse matrix from a .npz file
loaded_sparse_matrix = load_npz('/files/sparse_matrix.npz')

# Print the loaded sparse matrix
print(loaded_sparse_matrix)
```

Following is the output of loading the sparse matrix using the scipy library −

```
<Compressed Sparse Row sparse matrix of dtype 'int32'
        with 4 stored elements and shape (4, 4)>
  Coords        Values
  (0, 0)        1
  (1, 1)        2
  (2, 2)        3
  (3, 3)        4
```

Let's see about the other file types which we can use in Scipy in the next chapter.

---

## 39. SciPy - Working With Different File Formats

*Source: [https://www.tutorialspoint.com/scipy/scipy_working_with_different_file_formats.htm](https://www.tutorialspoint.com/scipy/scipy_working_with_different_file_formats.htm)*

---

---

## 40. Scipy - Efficient Data Storage with HDF5

*Source: [https://www.tutorialspoint.com/scipy/scipy_efficient_data_storage_with_hdf5.htm](https://www.tutorialspoint.com/scipy/scipy_efficient_data_storage_with_hdf5.htm)*

---

---
[Previous](/scipy/scipy_working_with_different_file_formats.htm)[Quiz](/scipy/quiz_on_scipy_efficient_data_storage_with_hdf5.htm)[Next](/scipy/scipy_data_serialization.htm)**Efficient data storage and management**are indeed essential for handling large datasets particularly in scientific computing and data analysis. The**Hierarchical Data Format version 5 (HDF5)**is a widely-used solution in this context by offering powerful capabilities for organizing, storing and retrieving large datasets.
Within the SciPy ecosystem the h5py is the library that provides a Pythonic interface to HDF5 by facilitating the storage and retrieval of extensive numerical data in a user-friendly way.

Now let's see the overview of how HDF5 works with SciPy and why it's so useful for data storage −

## What is HDF5?
**HDF5**or**Hierarchical Data Format version 5**is a widely-used file format designed to store and organize large amounts of data. It is popular in scientific computing and data-intensive fields because it can efficiently handle complex datasets and provides tools for managing large-scale data in a way that is both flexible and high-performing.
Following are the components of HDF5 which make it highly flexible by enabling the storage of large, structured and hierarchical datasets with metadata −

- **File:**The HDF5 file itself is the container for all stored data. It can hold multiple groups and datasets with a flexible structure that enables efficient storage and retrieval.
- **Groups:**Similar to directories in a file system the groups can contain datasets or other groups by allowing a hierarchical organization of data. They help structure complex datasets.
- **Datasets:**These are multidimensional arrays that store the actual data such as numerical arrays, images or tables. Datasets can have arbitrary dimensions and data types by making them versatile for various types of data.
- **Attributes:**Key-value metadata pairs associated with datasets or groups with attributes provide descriptive information about data like units, descriptions or settings used for data collection.
- **Datatypes:**HDF5 supports multiple datatypes including integers, floats, strings and even complex data structures. These are defined at the dataset level and ensure data consistency.
### Why Use HDF5?
**HDF5**is especially useful for scientific applications, data science, machine learning and any field requiring high-performance data storage and retrieval. With support for large, hierarchical and complex datasets in which HDF5 is used in fields as follows −
- **Physics and Astronomy:**Handling data from simulations or large-scale experiments.
- **Genomics and Bioinformatics:**Storing complex datasets from genetic studies.
- **Machine Learning:**Organizing and managing large datasets used in training and testing.
## Key Features of HDF5

The key features of HDF5 that make it highly suitable for managing large and complex datasets as follows −

- **Hierarchical Structure:**HDF5 organizes data in a tree-like structure of groups which are similar to folders and datasets which are similar to files by making it easy to navigate and manage complex data structures.
- **High Performance and Scalability:**HDF5 is optimized for efficient data access by allowing for fast read and write operations which is crucial for handling large datasets in high-performance computing environments.
- **Support for Large Datasets:**HDF5 can store datasets that are much larger than available memory by allowing users to work with large datasets that cannot fit entirely in RAM.
- **Compression and Storage Efficiency:**HDF5 supports multiple compression algorithms such as gzip, SZIP by reducing storage requirements and improving I/O performance. This compression is especially useful when storing large scientific datasets.
- **Data Type Flexibility:**HDF5 supports various data types such as integers, floats, strings, compound data types and user-defined types. This versatility enables it to handle complex data structures.
- **Metadata Storage:**Each dataset and group in HDF5 can store metadata in the form of attributes. This feature is essential for storing descriptive information about the data which aids in data interpretation and documentation.
- **Cross-Platform Compatibility:**HDF5 files are binary, platform-independent and self-describing by ensuring they can be used on different systems and programming environments. This portability makes HDF5 ideal for collaborative and long-term storage.
- **Data Integrity:**HDF5 includes error-detection mechanisms such as checksums to ensure data integrity during storage and retrieval.
- **Parallel I/O:**HDF5 supports parallel I/O by enabling multiple processes to read from and write to the same file simultaneously. This is particularly useful in high-performance computing (HPC) applications.
- **Partial I/O and Data Chunking:**HDF5 allows partial I/O operations which means that users can load specific sections of a dataset without loading the entire dataset into memory. Combined with data chunking this feature allows efficient access to subsets of large datasets.
## Using HDF5 in SciPy with h5py

In SciPy we can use the
**h5py**library to work with**HDF5**files. This library provides a Pythonic interface to the HDF5 format by enabling us to efficiently store and retrieve large numerical datasets which is especially useful in data science, machine learning and scientific computing.
### Installing h5py

To use h5py we can install it with the help of pip command as follows −

```
pip install h5py
```

### Creating and Writing to an HDF5 File

Let's start by creating an HDF5 file and saving a dataset to it by using the below reference code −

```
import h5py
import numpy as np

# Create an HDF5 file
with h5py.File('/files/example.h5', 'w') as f:
    # Create a dataset within the file
    data = np.random.random((1000, 1000))  # Generate some sample data
    dset = f.create_dataset('my_dataset', data=data)  # Save data in the dataset

    # Add metadata as an attribute
    dset.attrs['description'] = 'This is a 1000x1000 array of random numbers'
```

After executing the above code the HDF5 file will be created in the prescribed location with the file name
**example.h5**.
Reading Data from an HDF5 File

Following is the example of reading the data and metedata stored in a HDF5 file −

```
import h5py
import numpy as np

with h5py.File('/files/example.h5', 'r') as f:
    # Access the dataset
    dset = f['my_dataset']
    data = dset[:]  # Load data into memory

    # Access dataset attributes
    description = dset.attrs['description']
    print(description)
```

Following is the output of reading the data from HDF5 file using h5py module −

```
This is a 1000x1000 array of random numbers
```

### Organizing Data with Groups

HDF5 allows data to be stored in a hierarchical structure using groups which can contain datasets or other groups. Here is the output of organizing the data with groups in scipy −

```
import h5py
import numpy as np

with h5py.File('/files/example_grouped.h5', 'w') as f:
    # Create groups
    grp1 = f.create_group('group1')
    grp2 = f.create_group('group2')

    # Create datasets within groups
    grp1.create_dataset('dataset1', data=np.arange(10))
    grp2.create_dataset('dataset2', data=np.linspace(0, 1, 100))
```

When the above code executed then the scipy data is organized as groups.

### Using Compression and Chunking

HDF5 supports data compression which reduces file size and can improve I/O performance. Chunking divides the dataset into smaller blocks by optimizing access. Following is the example which compresses the data and makes into chunks −

```
import h5py
import numpy as np

# Create a large random dataset
data = np.random.random((1000, 1000))

with h5py.File('/files/compressed_data.h5', 'w') as f:
    dset = f.create_dataset(
        'compressed_dataset',
        data=data,
        compression="gzip",       # Apply gzip compression
        compression_opts=9,       # Maximum compression level
        chunks=(100, 100)         # Chunk size of 100x100
    )
```

The above code creates a compressed data file with the help of
**HDF5**.
### Working with a Large Dataset

This example avoids loading all data into memory at once which is helpful when working with very large datasets −

```
import h5py
import numpy as np

with h5py.File('/files/large_data.h5', 'w') as f:
    dset = f.create_dataset('large_dataset', shape=(10000, 10000), dtype='float32')
    for i in range(10000):
        dset[i] = np.random.random(10000)  # Writing row by row
```

## Advantages of Using HDF5

Following are the advantages of using the HDF5 while dealing with the scipy data −

- **Efficiency:**HDF5s optimized I/O operations make data retrieval faster.
- **Compression:**Store large datasets while reducing file size.
- **Hierarchical Structure:**Organize complex data with groups and sub-groups which are suitable for organizing experiment results.
- **Data Integrity:**HDF5 files have built-in error-checking mechanisms.
- **Scalability:**HDF5 scales well for large datasets.

---

## 41. SciPy - Data Serialization

*Source: [https://www.tutorialspoint.com/scipy/scipy_data_serialization.htm](https://www.tutorialspoint.com/scipy/scipy_data_serialization.htm)*

---

---
[Previous](/scipy/scipy_efficient_data_storage_with_hdf5.htm)[Quiz](/scipy/quiz_on_scipy_data_serialization.htm)[Next](/scipy/scipy_linalg.htm)**SciPy data serialization**refers to the process of converting complex Python objects or datasets into a format that can be easily stored, transferred or reconstructed later.
In the view of SciPy the serialization is commonly used to save large scientific datasets such as NumPy arrays, matrices or other data structures to a file and load them back efficiently for future use.

This process is essential for preserving data between sessions by sharing data across systems or optimizing performance when working with large datasets.

## Common Methods for Data Serialization in Python

As we already know that in SciPy
**data serialization**involves converting complex data structures such as NumPy arrays, SciPy sparse matrices or other Python objects into a format that can be stored, transferred and later reconstructed. Here are the key methods for data serialization commonly used in SciPy −
## Serialization in SciPy Using Pickle
**Pickle**is the standard library in Python used for serializing and deserializing Python objects. While it is flexible and works with any Python object and it has certain limitations particularly when dealing with large datasets or numerical data. Additionally the security risks exist when loading untrusted data so caution should be exercised.
This method is suitable for general Python object serialization but not ideal for very large datasets due to its inefficiency in terms of storage and speed.

Following is the example of using the
**pickle**method to perform data serialization −
```
import pickle
import numpy as np

# Create some data
data = np.random.rand(1000, 1000)

# Serialize to file
with open('/files/data.pkl', 'wb') as f:
    pickle.dump(data, f)

# Deserialize from file
with open('/files/data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
```

## Serialization in SciPy Using HDF5
**Data serialization using HDF5**is the process of storing complex data structures in an HDF5 file format so they can be easily saved, shared and reloaded later. HDF5s ability to handle large and hierarchical datasets with different data types and metadata which makes it an ideal format for data serialization especially in scientific computing and machine learning applications.
In Python the h5py is the primary library used for HDF5 serialization. Using h5py we can serialize multidimensional arrays, complex datasets and metadata by storing them in an organized, efficient and portable way.

### Why to Use HDF5 for Data Serialization?

Following are the reasons why we can choose the
**HDF5**method for Data Serialization −
- **Efficient Storage:**HDF5 compresses and organizes data efficiently by making it suitable for large datasets that don't fit into memory.
- **Portability:**HDF5 files are platform-independent which allow data to be shared and reused across different computing environments.
- **Metadata Support:**Each dataset and group can store attributes by providing additional context for serialized data.
- **Hierarchical Structure:**HDF5s hierarchical format helps organize complex data relationships, making it ideal for structured data serialization.
Following are the steps to be followed to perform Data Serilization with the help of
**HDF5**.
- **Create a File:**Open an HDF5 file in write mode.
- **Store Data:**Use datasets to store data and groups to organize complex data hierarchies.
- **Add Metadata:**Store metadata as attributes for additional context.
### Serializing Data in HDF5 with h5py

Here is the example to serialize the Data in HdF5 file with the help of h5py method −

```
import h5py
import numpy as np

# Create a file for serialization
with h5py.File('/files/serialized_data.h5', 'w') as f:
    # Serialize a dataset
    data = np.random.rand(1000, 1000)
    dset = f.create_dataset('my_dataset', data=data, compression="gzip", compression_opts=4)
    
    # Serialize additional information as metadata
    dset.attrs['description'] = 'Random data for serialization example'
    dset.attrs['data_source'] = 'Simulated data'
    
    # Organize data in a group (like a directory)
    group = f.create_group("experiment_1")
    group.create_dataset('measurements', data=np.arange(100))

    # Add metadata to the group
    group.attrs['experiment_date'] = '2024-11-12'
    group.attrs['experiment_notes'] = 'Test run with random values'
```

### Deserializing Data with HDF5

To deserialize open the HDF5 file in read mode and access the datasets and metadata.

```
import h5py
import numpy as np

# Deserialize the data
with h5py.File('/files/serialized_data.h5', 'r') as f:
    # Load the dataset
    data = f['my_dataset'][:]
    description = f['my_dataset'].attrs['description']
    source = f['my_dataset'].attrs['data_source']
    
    # Load data from a group
    measurements = f['experiment_1/measurements'][:]
    exp_date = f['experiment_1'].attrs['experiment_date']
    notes = f['experiment_1'].attrs['experiment_notes']
    
    print(description, source, exp_date, notes)
```

Here is the output of the deserialized data of the HDF5 file −

```
Random data for serialization example Simulated data 2024-11-12 Test run with random values
```

## Numpy's np.load/np.save

NumPy provides straightforward functions for data serialization and deserialization namely
**np.save**and**np.load**which are useful for saving and loading arrays in a binary format. This functionality is part of the SciPy ecosystem and is often used in scientific computing when data structures are simpler and do not require the full capabilities of HDF5.
Here are the key features of np.save and np.load −

- **Simplicity:**np.save and np.load are easy to use for saving individual arrays or basic data structures without the need for hierarchical or complex data storage.
- **Binary Format:**Data is saved in a**.npy**binary format which is optimized for NumPy arrays and includes metadata such as data shape and data type for fast loading.
- **Portability:**The**.npy**files are cross-platform and can be shared between systems as long as they use compatible NumPy versions.
### Saving Data with np.save

The
**np.save**function writes a single NumPy array to a file in .npy format. Following is the example which shows how to save the data using np.save() −
```
import numpy as np

# Create a sample array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Save array to 'array.npy'
np.save('/files/array.npy', array)
```

We can also save multiple arrays using
**np.savez**which stores them in a single .npz file as a compressed archive of multiple .npy files. Here is the example which saves multiple arrays −
```
import numpy as np

# Create two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# Save multiple arrays to 'arrays.npz'
np.savez('/files/arrays.npz', array1=array1, array2=array2)
```

### Loading Data with np.load

The
**np.load**function reads an .npy or .npz file and loads the data into memory as a NumPy array. Below is the example of loading data −
```
import numpy as np

# Create two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# Save multiple arrays to 'arrays.npz'
np.savez('/files/arrays.npz', array1=array1, array2=array2)
```

Following is the output of the loaded array from the .npy file −

```
[[1 2 3]
 [4 5 6]]
```

For .npz files, np.load returns a dictionary-like object that allows access to each array by name. Below is the example of loading multiple arrays −

```
import numpy as np

# Load a single array from 'arrays.npy'
loaded_data = np.load('/files/arrays.npz')
# Load multiple arrays from 'arrays.npz'
print(loaded_data['array1'])
print(loaded_data['array2'])
```

Following is the output of the loaded data from the .npz file −

```
[1 2 3]
[4 5 6]
```

## Serialization in SciPy Using JSON
**JSON**is a lightweight text-based format that can be used for serializing simple Python objects such as dictionaries, lists and arrays.
It's not as efficient or suitable for large datasets as Pickle, HDF5 or .npy but it is human-readable and ideal for small datasets or transferring simple data structures over the web.
Following is the example of using the JSON method for Data Serialization −

```
import json
import numpy as np

data = np.random.rand(1000, 1000).tolist()  # Convert to list for JSON serialization

# Serialize to a JSON file
with open('/files/data.json', 'w') as f:
    json.dump(data, f)

# Deserialize from JSON file
with open('/files/data.json', 'r') as f:
    loaded_data = json.load(f)
```

## Serialization in SciPy Using SQLite
**SQLite**is a lightweight, disk-based database that can store data in a structured format like tables, rows and columns. It is useful for applications that require relational data structures and can handle small to medium-sized datasets.
Here is the example which shows how to use the SQLite method for Data Serialization −

```
import sqlite3
import numpy as np

# Create a SQLite database and table
conn = sqlite3.connect('/files/data.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS data (id INTEGER PRIMARY KEY, value REAL)')

# Insert data into the table
data = np.random.rand(1000)
for i, value in enumerate(data):
    c.execute('INSERT INTO data (id, value) VALUES (?, ?)', (i, value))
conn.commit()

# Query the data
c.execute('SELECT * FROM data')
rows = c.fetchall()
conn.close()
```

Finally we can conclude that each method for data serialization in SciPy serves different purposes, depending on the nature of the data and the use case.

- **Pickle**is flexible but not optimal for large scientific datasets.
- **HDF5**and**NumPy's**.npy formats are highly efficient for large numerical datasets with HDF5 offering additional features like compression and chunking.
- **JSON**is human-readable but less efficient for large datasets.
- **SQLite**is suitable for structured relational data.
For most scientific and data analysis tasks the HDF5 (via h5py) and NumPy's .npy are typically the best choices due to their efficiency and support for large datasets.

---

## 42. SciPy - linalg

*Source: [https://www.tutorialspoint.com/scipy/scipy_linalg.htm](https://www.tutorialspoint.com/scipy/scipy_linalg.htm)*

---

---
[Previous](/scipy/scipy_data_serialization.htm)[Quiz](/scipy/quiz_on_scipy_linalg.htm)[Next](/scipy/scipy_matrix_creation_basic_operations.htm)
In SciPy the
**linalg**module is abbrivated as linear algebra which provides a comprehensive set of functions for performing various linear algebra operations such as solving linear systems, computing matrix factorizations and handling eigenvalue problems.
It is a highly optimized and efficient module built on top of BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACKage) libraries which are widely used in scientific computing.

## When to use scipy.linalg vs numpy.linalg?
**scipy.linalg**and**numpy.linalg**both provide functions for linear algebra operations but they differ in terms of functionality, performance and specific use cases. Heres a comparison to help us to decide when to use each deping upon the requirement we have −Criteriascipy.linalgnumpy.linalg**Functionality**Extensive linear algebra routines, including additional functions for matrix decompositions such as QR, LU, Cholesky and matrix exponentials.
Supports specialized matrix types such as sparse matrices.Basic linear algebra routines like matrix inversion, determinants, eigenvalues, SVD.
Lacks advanced decompositions and specialized matrix support.**Performance**Often faster especially for larger matrices or specialized tasks due to optimized routines and bindings to BLAS and LAPACK libraries.Sufficient for smaller matrices and general-purpose tasks.
Can be slower than**scipy.linalg**for certain operations.**Use Case**Ideal for advanced linear algebra tasks, large matrices and scientific computing applications requiring high performance and specialized functions.Suitable for basic linear algebra tasks and simpler applications.**Dependency**Requires SciPy.Part of the NumPy package.
## Key Features of scipy.linalg

The
**scipy.linalg**module offers a wide range of linear algebra functions many of which extend the capabilities of NumPy's**numpy.linalg**module. Some of the most important functions  are as mentioned below −
### Matrix Decompositions

Matrix decomposition or matrix factorization which is indeed a powerful tool in linear algebra with broad applications across various fields. Following is the overview of some commonly used types of matrix decompositions, their forms and primary applications −
S.NoDecomposition TypeFunction and Description1**LU Decomposition**[scipy.linalg.lu()](/scipy/scipy_linalg_lu_function.htm)
Decomposes a matrix into lower and upper triangular matrices, (A = LU).2**QR Decomposition**[scipy.linalg.qr()](/scipy/scipy_linalg_qr_function.htm)
Decomposes a matrix into an orthogonal matrix (Q) and an upper triangular matrix (R).3**Cholesky Decomposition****scipy.linalg.cholesky()**
Decomposes a positive-definite matrix into a lower triangular matrix, ( A = LL).4**Eigen Decomposition****scipy.linalg.eig()**
Computes the eigenvalues and eigenvectors of a square matrix.5**Singular Value Decomposition (SVD)****scipy.linalg.svd()**
Computes the singular value decomposition, ( A = UV).6**Schur Decomposition****scipy.linalg.schur()**
Computes the Schur decomposition, breaking a matrix into quasi-triangular form.7**Hessenberg Decomposition****scipy.linalg.hessenberg()**
Decomposes a matrix into Hessenberg form, which has zero entries below the first sub-diagonal.8**Polar Decomposition****scipy.linalg.polar()**
Decomposes a matrix into a product of a unitary and positive semi-definite matrix.9**Jordan Decomposition**Not directly available
Jordan decomposition is not directly available in SciPy and custom implementations may be used.
## Solving Linear Systems

SciPy provides several efficient functions for
**solving linear systems**of equations which is suitable for different types of matrices such as dense, sparse, symmetric, etc. Heres a are the overview of some key functions available in scipy.linalg and scipy.sparse.linalg for solving linear systems −S.NoFunction & Description1[scipy.linalg.solve()](/scipy/scipy_solve_function.htm)
Solves a linear matrix equation (Ax = b) for dense matrices using LU decomposition. General-purpose solver for dense matrices.
2**scipy.linalg.lu_solve()**
Solves (Ax = b) using LU decomposition from**lu_factor**. Useful when solving multiple systems with the same matrix ( A ).3[scipy.linalg.lstsq()](/scipy/scipy_lstsq_function.htm)
Solves linear least-squares problems for over-determined systems. Suitable for systems where (Ax = b) has no exact solution.
4**scipy.linalg.cho_solve()**
Solves (Ax = b) using Cholesky factorization from**cho_factor**. Efficient for symmetric positive-definite matrices.5[scipy.linalg.solve_triangular()](/scipy/scipy_solve_triangular.htm)
Solves (Ax = b) for triangular matrices (upper or lower). Optimized for triangular matrices, used in back-substitution.
6**scipy.sparse.linalg.spsolve()**
Solves (Ax = b) for sparse matrices using LU decomposition. Ideal for sparse matrices, conserving memory.7**scipy.sparse.linalg.cg()**
Conjugate gradient solver for large, sparse, symmetric positive-definite matrices. Efficient for large symmetric positive-definite matrices.8**scipy.sparse.linalg.gmres()**
Generalized minimal residual method for sparse linear systems. Effective for non-symmetric sparse matrices in iterative solutions.9**scipy.sparse.linalg.lsmr()**
Iterative least-squares solver for large-scale sparse problems. Suitable for large, sparse, and over-determined systems.10**scipy.sparse.linalg.minres()**
Minimum residual method for symmetric matrices. Used for symmetric, indefinite matrices.
## Function to Solve Eigenvalue Problems
**Eigenvalues and Eigenvectors**are used to compute the eigenvalues and eigenvectors of a square matrix. Eigenvalues are important in many areas such as stability analysis and principal component analysis. Below are some important functions in Scipy which are used to compute the Eigenvalues and Eigenvectors −S.NoFunction & Description1**scipy.linalg.eig()**
Computes the eigenvalues and right eigenvectors of a square matrix. Suitable for general eigenvalue problems.2**scipy.linalg.eigh()**
Computes the eigenvalues and eigenvectors of a symmetric or Hermitian matrix. Optimized for symmetric matrices.3**scipy.linalg.eigvals()**
Computes only the eigenvalues of a square matrix. Useful when only eigenvalues are needed.4**scipy.linalg.eigvalsh()**
Computes only the eigenvalues of a symmetric or Hermitian matrix. Efficient for symmetric matrices.5**scipy.sparse.linalg.eigs()**
Computes a few eigenvalues and eigenvectors of a sparse matrix using iterative methods. Suitable for large sparse matrices.6**scipy.sparse.linalg.eigsh()**
Computes a few eigenvalues and eigenvectors of a sparse symmetric or Hermitian matrix using iterative methods. Optimized for large symmetric matrices.
## Matrix Inversion and Determinant

In SciPy
**matrix inversion and determinant**calculation are key operations in linear algebra which often used to solve systems of linear equations, analyze matrix properties and perform various transformations in scientific computing. SciPy provides functions in the**scipy.linalg**module that are optimized for these tasks.
Here's an overview of the key functions for matrix inversion and determinant calculation in SciPy −
S.No.Function & Description1[scipy.linalg.inv()](/scipy/scipy_inv_function.htm)
Computes the inverse of a matrix
*A*. This function is useful when you need to find the matrix that, when multiplied by*A*, results in the identity matrix.2[scipy.linalg.det()](/scipy/scipy_det_function.htm)
Calculates the determinant of a matrix
*A*. The determinant is a scalar value that provides insight into the matrix's invertibility and other properties. If the determinant is zero, the matrix is singular (non-invertible).
## Matrix Functions

In SciPy
**matrix functions**provide efficient methods for handling and operating on matrices, particularly when working with large datasets, linear algebra and scientific computing. These matrix functions are built on top of the NumPy library and provide higher-level operations for advanced linear algebra tasks. Here are the key matrix functions that are available in SciPy −S.No.Function & Description1[scipy.linalg.expm()](/scipy/scipy_expm_function.htm)
Compute the matrix exponential of an array.
2[scipy.linalg.logm()](/scipy/scipy_logm_function.htm)
Compute matrix logarithm.
3[scipy.linalg.cosm()](/scipy/scipy_cosm_function.htm)
Compute the matrix cosine.
4[scipy.linalg.sinm()](/scipy/scipy_sinm_function.htm)
Compute the matrix sine.
5[scipy.linalg.tanm()](/scipy/scipy_tanm_function.htm)
Compute the matrix tangent.
6**scipy.linalg.coshm()**
Compute the hyperbolic matrix cosine.
7[scipy.linalg.sinhm()](/scipy/scipy_sinhm_function.htm)
Compute the hyperbolic matrix sine.
8[scipy.linalg.tanhm()](/scipy/scipy_tanhm_function.htm)
Compute the hyperbolic matrix tangent.
9[scipy.linalg.signm()](/scipy/scipy_signm_function.htm)
Matrix sign function.
10[scipy.linalg.sqrtm()](/scipy/scipy_sqrtm_function.htm)
Matrix square root.
11[scipy.linalg.funm()](/scipy/scipy_funm_function.htm)
Evaluate a matrix function specified by a callable.
12[scipy.linalg.expm_frechet()](/scipy/scipy_expm_frechet_function.htm)
Frechet derivative of the matrix exponential of A in the direction E.
13[scipy.linalg.expm_cond()](/scipy/scipy_expm_cond_function.htm)
Relative condition number of the matrix exponential in the Frobenius norm.
14[scipy.linalg.fractional_matrix_power()](/scipy/scipy_fractional_matrix_power_function.htm)
Compute the fractional power of a matrix.

The
**scipy.linalg**module is an essential part of the SciPy ecosystem for linear algebra which provide a wide range of optimized functions for matrix decompositions, eigenvalue problems and solving linear systems, among others. It is an indispensable tool for scientists, engineers and data analysts working with large, complex datasets.
## Linear Algebra Methods and Operations

Linear algebra is crucial in domains including machine learning, data science, and scientific computing. The procedures and techniques listed above are applied to solve tough problems with matrices and vectors and have numerous applications across various disciplines.
Sr.No.Function & Description1[scipy.solve.banded](/scipy/scipy_solve_function.htm)
Solves a linear system with a banded matrix, often used for sparse systems.
2**scipy.solveh.banded**
Solves a Hermitian banded system, applicable in complex matrix problems.
3[scipy.solve.circulant](/scipy/scipy_solve_circulant_function.htm)
Solves a system with a circulant matrix, common in signal processing.
4[scipy.solve.toeplitz](/scipy/scipy_solve_toeplitz_function.htm)
Solves a system with a Toeplitz matrix, used in time series and signal analysis.
5[scipy.matmul.toeplitz](/scipy/scipy_matmul_toeplitz_function.htm)
Performs matrix multiplication with a Toeplitz matrix.
6[scipy.norm](/scipy/scipy_norm_function.htm)
Computes various norms (e.g., Euclidean, Frobenius) for matrices or vectors.
7[scipy.pinv](/scipy/scipy_pinv_function.htm)
Computes the Moore-Penrose pseudo-inverse of a matrix.
8[scipy.pinvh](/scipy/scipy_pinvh_function.htm)
Computes the pseudo-inverse of a Hermitian matrix.
9[scipy.kron](/scipy/scipy_kron_function.htm)
Computes the Kronecker product of two matrices.
10[scipy.khatri.rao](/scipy/scipy_khatri_rao_function.htm)
Computes the Khatri-Rao product of two matrices.
11[scipy.orthogonal.procrustes](/scipy/scipy_orthogonal_procrustes_function.htm)
Finds the orthogonal matrix that best aligns two matrices.
12[scipy.matrix.balance](/scipy/scipy_matrix_balance_function.htm)
Balances a matrix to improve numerical stability.
13[scipy.subspace.angles](/scipy/scipy_subspace_angles_function.htm)
Computes the angles between subspaces of two matrices.
14[scipy.bandwidth](/scipy/scipy_bandwidth_function.htm)
Computes the bandwidth of a matrix (distance from the main diagonal to the farthest non-zero element)
15[scipy.issymmetric](/scipy/scipy_issymmetric_function.htm)
Checks if a matrix is symmetric.
16[scipy.ishermitian](/scipy/scipy_ishermitian_function.htm)
Checks if a matrix is Hermitian (equal to its conjugate transpose).
17[scipy.solve.sylvester](/scipy/scipy_solve_sylvester_function.htm)
Solves the Sylvester equation AX+XB=C, where A, B, and C are given matrices. This method is commonly used in control theory and system analysis.

---

## 43. SciPy - Matrix Creation & Basic Operations

*Source: [https://www.tutorialspoint.com/scipy/scipy_matrix_creation_basic_operations.htm](https://www.tutorialspoint.com/scipy/scipy_matrix_creation_basic_operations.htm)*

---

---
[Previous](/scipy/scipy_linalg.htm)[Quiz](/scipy/quiz_on_scipy_matrix_creation_basic_operations.htm)[Next](/scipy/scipy_matrix_lu_decomposition.htm)
SciPy offers many convenient functions for creating and manipulating matrices by extending the capabilities of NumPy. The
**scipy.linalg**module in SciPy provides functions for creating and performing basic operations on matrices such as linear algebraic computations like solving systems of linear equations, finding determinants and performing matrix factorization.
It is similar to the NumPy library but SciPys linalg module extends NumPys capabilities with more advanced linear algebra tools.

In this chapter let's discuss in detail about the matrix creation and basic operations using
**scipy.linalg**−
## Creating a Matrix in SciPy
**Matrix creation**in SciPy and NumPy involves using a variety of functions to initialize matrices with different values, structures and properties. Here are various ways to create matrices in SciPy which focus on common techniques, special matrices and properties −
### Example

To create matrices SciPy uses the same array creation techniques as NumPy with an additional focus on linear algebra functions. Here is the example of it −

```
import numpy as np
import scipy.linalg as la

# Creating a matrix (2x2)
A = np.array([[1, 2], [3, 4]])
print("The 2x2 matrix:",A)
# Creating a square matrix of zeros (3x3)
B = np.zeros((3, 3))
print("The 3x3 square matrix:,"B)
# Creating an identity matrix (4x4)
I = np.eye(4)
print("The 4X4 Identity matrix:,"I)
# Creating a random matrix
R = np.random.rand(3, 3)
print("The Random matrix:,"R)
```

#### Output

Following is the output of creating the different type of matrices −

```
The 2x2 matrix: [[1 2]
 [3 4]]
The 3x3 square matrix: [[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
The 4X4 Identity matrix: [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
The Random matrix: [[0.93209861 0.44003913 0.94137284]
 [0.25650407 0.59441862 0.49890997]
 [0.65446923 0.38596346 0.92084365]]
```

## Basic Matrix Operations in SciPy
**scipy.linalg**supports numerous matrix operations such as addition, subtraction, multiplication, transpose, etc. Some functions are more efficient than using basic NumPy operations.
## Matrix Multiplication

### Example

Scipy does not have functions to perform the matrix multiplication, in such case we can use the numpy library
**np.matmul()**function. Following is the example of performing the matrix multiplication using**np.matmul()**and**@**decorator −
```
import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
import scipy.linalg as la

# Matrix multiplication
F = A @ B  # Using @ operator
G = np.matmul(A, B)  # Using np.matmul

print("Matrix Multiplication using @:\n", F)
print("Matrix Multiplication using np.matmul:\n", G)
```

#### Output

Following is the output of matrix multiplication performed using
**@**and**np.transpose()**−
```
Matrix Multiplication using @:
 [[19 22]
 [43 50]]
Matrix Multiplication using np.matmul:
 [[19 22]
 [43 50]]
```

## Matrix Transpose in SciPy

### Example

Since
**scipy.linalg**module doesnt provide a direct transpose function its common to use**numpy.transpose()**or the**.T**attribute for this operation. Both approaches are efficient and work well with scipy for other matrix operations if we need them. Following is the example of performing matrix transpose −
```
import numpy as np

A = np.array([[1, 2], [3, 4]])
A_transpose = A.T
print("Matrix transpose using .T",A_transpose)
transpose_matrix = np.transpose(A)
print("Matrix transpose using transpose()",transpose_matrix)
```

#### Output

Following is the output of Performing matrix transpose using numpy −

```
Matrix transpose using .T [[1 3]
 [2 4]]
Matrix transpose using transpose() [[1 3]
 [2 4]]
```

## Matrix Addition and Subtraction in SciPy

### Example

Matrix addition and subtraction are not directly available in scipy.linalg either as these are basic operations typically handled by numpy. However we can use scipy in combination with numpy to perform these operations. Below is the example which shows how to perform matrix addition and subtraction using numpy −

```
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition
C_add = A + B
print("Addition:\n", C_add)

# Matrix subtraction
C_subtract = A - B
print("Subtraction:\n", C_subtract)
```

#### Output

Following is the output of Performing matrix addition and subtraction using numpy −

```
Addition:
 [[ 6  8]
 [10 12]]
Subtraction:
 [[-4 -4]
 [-4 -4]]
```

## Element wise Matrix Multiplication in SciPy

### Example

Element-wise multiplication in Python can be easily done with numpy using either the * operator or numpy.multiply. While scipy.linalg does not provide a specific function for element-wise multiplication we can still use numpy alongside scipy. Here is the example of it −

```
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
C = A * B
print("Element-wise Multiplication using *:\n", C)
D = np.multiply(A, B)
print("Element-wise Multiplication using multiply():\n", D)
```

#### Output

Following is the output of Performing matrix element wise multiplication −

```
Element-wise Multiplication using *:
 [[ 5 12]
 [21 32]]
Element-wise Multiplication using multiply():
 [[ 5 12]
 [21 32]]
```

## Matrix Inversion in SciPy

Matrix inversion is the process of finding a matrix that, when multiplied by the original matrix, yields the identity matrix. In
**scipy.linalg**matrix inversion can be done using the**scipy.linalg.inv()**function.
Following are the key points to be considered while performing inversion of the matrix −

- **scipy.linalg.inv(A)**computes the inverse of matrix A.
- 
The matrix must be square i.e., it has the same number of rows and columns and non-singular i.e., it must have a non-zero determinant.

- 
After computing the inverse we can verify the result by multiplying the original matrix with its inverse which should yield the identity matrix.

### Example - Check for Singular Matrix

We can check the determinant using
**np.linalg.det(A)**or**scipy.linalg.det(A)**to finrd whether the given matrix is singular or not. If the matrix is singular  then it has a determinant of 0 and the inversion will fail and we will get an error or a warning. Here is the example which checks for the singular matrix −
```
import numpy as np
from scipy import linalg

# Define a square matrix
A = np.array([[1, 2], [3, 4]])

if linalg.det(A) == 0:
    print("Matrix is singular, so it doesn't have an inverse.")
else:
    A_inv = linalg.inv(A)
    print("Inverse of A:", A_inv)
```

#### Output

Following is the output which checks whether the given matrix is singular or not −

```
Inverse of A: [[-2.   1. ]
 [ 1.5 -0.5]]
```

### Example - Matrix Inversion

Here is the example which shows how we can perform matrix inversion using
**scipy.linalg.inv()**−
```
import numpy as np
from scipy import linalg

# Define a square matrix
A = np.array([[1, 2], [3, 4]])

# Compute the inverse using scipy.linalg.inv
A_inv = linalg.inv(A)

print("Matrix A:")
print(A)

print("\nInverse of A:")
print(A_inv)

# To verify the result, multiply the matrix with its inverse and check if it gives the identity matrix
identity_matrix = np.dot(A, A_inv)
print("\nA * A_inv (should be identity matrix):")
print(identity_matrix)
```

#### Output

Following is the output of Performing matrix inversion using the scipy.linalg.inv() −

```
Inverse of A: [[-2.   1. ]
 [ 1.5 -0.5]]
PS D:\Tutorialspoint> python sample.py
Matrix A:
[[1 2]
 [3 4]]

Inverse of A:
[[-2.   1. ]
 [ 1.5 -0.5]]

A * A_inv (should be identity matrix):
[[1.0000000e+00 0.0000000e+00]
 [8.8817842e-16 1.0000000e+00]]
```

## Matrix Determinant in SciPy

The
**determinant**of a square matrix is a scalar value that provides significant information about the matrix. In linear algebra the determinant can be used to answer questions about the invertibility of a matrix among other things.
### Example

In this example to compute the determinant of a matrix we are using SciPy where we can use the function
**scipy.linalg.det()**−
```
import numpy as np
from scipy import linalg

# Define a square matrix
A = np.array([[1, 2], [3, 4]])

# Compute the determinant using scipy.linalg.det
det_A = linalg.det(A)

print("Matrix A:")
print(A)

print("\nDeterminant of A:")
print(det_A)
```

#### Output

Following is the output of Performing matrix determinant using the scipy.linalg.det() −

```
Matrix A:
[[1 2]
 [3 4]]

Determinant of A:
-2.0
```

---

## 44. SciPy - Matrix LU Decomposition

*Source: [https://www.tutorialspoint.com/scipy/scipy_matrix_lu_decomposition.htm](https://www.tutorialspoint.com/scipy/scipy_matrix_lu_decomposition.htm)*

---

---

## 45. SciPy - Matrix QU Decomposition

*Source: [https://www.tutorialspoint.com/scipy/scipy_matrix_qu_decomposition.htm](https://www.tutorialspoint.com/scipy/scipy_matrix_qu_decomposition.htm)*

---

---
[Previous](/scipy/scipy_matrix_lu_decomposition.htm)[Quiz](/scipy/quiz_on_scipy_matrix_qu_decomposition.htm)[Next](/scipy/scipy_singular_value_decomposition.htm)**Matrix QU Decomposition**is a factorization method that decomposes a matrix**A**into the product of two matrices namely, a unitary matrix**Q**and an upper triangular matrix**U**. This method is closely related to QR Decomposition but with some distinct differences.
When we give a matrix
**A**of size**m x n**then the QU Decomposition factorizes it into two matrices as −
```
A=QU
```

Where −

- **Q**is an  orthogonal or unitary matrix which means**Q**for real matrices orQ = I**Q**for complete matrices whereQ = I**Q**denotes the transpose of**Q**and**Q**denotes conjugate transpose.
- **U**is an triangular matrix which means all the elements below the diagonal are zero. The matrix**U**has the same shape as**A**.
The QU decomposition is particularly useful when working with least squares problems or in cases where we need to solve linear systems efficiently. It can also be seen as an alternative to QR decomposition when working with different matrix properties.

## Properties of QU Decomposition

The matrix
**Q**in QU decomposition is orthogonal if**A**is a real matrix or unitary if**A**is a complex matrix. This ensures that multiplying by**Q**preserves the length of vectors i.e., it is a rotation or reflection.
The matrix
**U**is upper triangular which means it has non-zero entries only on and above the diagonal.
## Steps for QU Decomposition

While Scipy does not have a direct implementation of QU decomposition but we can achieve the equivalent decomposition by using QR decomposition
**scipy.linalg.qr()**. The**R**matrix from QR decomposition corresponds to the**U**in QU decomposition and**Q**remains orthogonal or unitary.
Following are the steps to be followed to implement the QU Decomposition in scipy −

- **Initialization**− Start with the matrix**A**of size**mn**.
- **Apply Orthogonalization**− By using an orthogonalization method such as Gram-Schmidt which is used to generate the orthogonal matrix**Q**.
- **Construct Upper Triangular Matrix**− From the orthogonalized vectors we can form the upper triangular matrix**U**. This can be done by solving for the coefficients that ensure the matrix product**QU**matches the original matrix**A**.
- **Matrix Factorization**− The matrix**A**is now decomposed into**Q**and**U**where**A = QU**.
### Example - Implementation of QU Decomposition in Scipy

As we know that the Scipy provides methods to perform matrix decompositions like QR decomposition but the QU decomposition is not directly available in Scipy. However since QR decomposition can be seen as a special case of QU decomposition when the matrix is square we can use the QR decomposition function
**scipy.linalg.qr()**for a similar purpose and the results of QR decomposition will give us**Q**and**R**where**R**is upper triangular.
Here is the example which we can use QR Decomposition in scipy −

```
import numpy as np
from scipy.linalg import qr

# Define a matrix A
A = np.array([[1, 2], [3, 4], [5, 6]])

# Perform QR decomposition
Q, R = qr(A)

print("Matrix Q:")
print(Q)

print("Matrix R (equivalent to U in QU decomposition):")
print(R)
```

#### Output

Here is the output of the QU Decomposition computed using the
**scipy.linalg.qr()**function −
```
Matrix Q:
[[-0.16903085  0.89708523  0.40824829]
 [-0.50709255  0.27602622 -0.81649658]
 [-0.84515425 -0.34503278  0.40824829]]
Matrix R (equivalent to U in QU decomposition):
[[-5.91607978 -7.43735744]
 [ 0.          0.82807867]
 [ 0.          0.        ]]
```

### Example

Here is an another example which computes the QU Decomposition for the square matrix  of the shape 3 x 3 −

```
import numpy as np
from scipy.linalg import qr

# Define a 3x3 square matrix
A = np.array([[2, 4, 1],
              [6, 5, 3],
              [1, 2, 7]])

# Perform QR decomposition
Q, R = qr(A)

# Interpret R as U in QU decomposition
U = R

print("Original Matrix A:")
print(A)

print("\nOrthogonal Matrix Q:")
print(Q)

print("\nUpper Triangular Matrix U:")
print(U)

print("\nReconstructed A (Q * U):")
print(Q @ U)
```

#### Output

Following is the output of the QU Decomposition computed for the square using the
**scipy.linalg.qr()**function −
```
Original Matrix A:
[[2 4 1]
 [6 5 3]
 [1 2 7]]

Orthogonal Matrix Q:
[[-3.12347524e-01  8.38116355e-01 -4.47213595e-01]
 [-9.37042571e-01 -3.49215148e-01 -1.11022302e-16]
 [-1.56173762e-01  4.19058177e-01  8.94427191e-01]]

Upper Triangular Matrix U:
[[-6.40312424 -6.24695048 -4.21669157]
 [ 0.          2.44450604  2.72387815]
 [ 0.          0.          5.81377674]]

Reconstructed A (Q * U):
[[2. 4. 1.]
 [6. 5. 3.]
 [1. 2. 7.]]
```

## Applications of QU Decomposition

Following are the applications of the QU Decomposition −

- **Solving Linear Systems**−  QU decomposition is often used in solving over-determined or under-determined systems of linear equations.
- **Least Squares Problems**− In optimization and regression problems the QU decomposition helps in finding the best fit line or hyperplane by solving the least squares problem.
- **Eigenvalue Problems**− It can be used to find eigenvalues and eigenvectors for certain types of matrices.

---

## 46. SciPy - Singular Value Decomposition

*Source: [https://www.tutorialspoint.com/scipy/scipy_singular_value_decomposition.htm](https://www.tutorialspoint.com/scipy/scipy_singular_value_decomposition.htm)*

---

---
[Previous](/scipy/scipy_matrix_qu_decomposition.htm)[Quiz](/scipy/quiz_on_scipy_singular_value_decomposition.htm)[Next](/scipy/scipy_cholesky_decomposition.htm)
## Singular Value Decomposition in SciPy

SciPy's
**Singular Value Decomposition (SVD)**is a computational method provided by the**scipy.linalg**module for decomposing a matrix into three components namely, two orthogonal matrices and a diagonal matrix of singular values. It is a cornerstone in numerical linear algebra with applications in data analysis, signal processing and machine learning.
The Singular Value Decomposition(SVD) is defined for a matrix
**A**of size**m x n**as follows −
```
A = UVT
```

Where −

- **U**is an m x n orthogonal matrix.
- is an mn diagonal matrix with non-negative real numbers on the diagonal i.e., singular values.
- **V**is an n x n orthogonal matrix.
### Syntax

In Scipy the Singular Value Decomposition (SVD) is computed with the help of the
**svd()**function of the**linalg**module available in scipy library. Below is the syntax of the**scipy.linalg.svd()**function −
```
scipy.linalg.svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
```

Where −

- **a**− Input matrix to be decomposed.
- **full_matrices(default : True)**− If True then computes**U**and**V**as full size matrices.
- **compute_uv(default : True)**− If false then only computes the singular values without**U**and**V**.
This functions returns upto three outputs depending on the
**compute_uv**flag as follows −
- **u((ndarray, shape (M, M) or (M, K)))**− The left singular vectors. The shape depends on full_matrices.
- **s(ndarray, shape (K,))**− The singular values which are always sorted in descending order.
- **Vt(ndarray, shape (N, N) or (K, N))**− The transpose of the right singular vectors. hape depends on full_matrices.
## Applications of SVD

Below are the applications of the Singular Value Decomposition(SVD) −

- **Dimensionality Reduction**− In machine learning SVD is used to reduce the dimensions of large datasets like in PCA, Principal Component Analysis.
- **Noise Reduction**− SVD can help in identifying and removing noise in the data.
- **Data Compression**− SVD is used in applications like image compression.
- **Matrix Inversion**− SVD can be used to find the pseudoinverse of a matrix.
### Example of Basic Full SVD

Following is the example which uses the function
**scipy.linalg.svd()**to compute a full SVD using SciPywith full_matrices=True the default value −
```
import numpy as np
from scipy.linalg import svd

# Input matrix
A = np.array([[1, 2], [3, 4], [5, 6]])

# Compute Full SVD
U, s, Vt = svd(A, full_matrices=True)

# Diagonalize the singular values for full SVD reconstruction
Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, s)

# Print results
print("Matrix A:")
print(A)

print("\nU (left singular vectors):")
print(U)

print("\nSigma (diagonal matrix of singular values):")
print(Sigma)

print("\nVt (right singular vectors):")
print(Vt)

# Verify reconstruction
A_reconstructed = np.dot(U, np.dot(Sigma, Vt))
print("\nReconstructed A:")
print(A_reconstructed)
```

#### Output

Here is the output of the basic full SVD −

```
Matrix A:
[[1 2]
 [3 4]
 [5 6]]

U (left singular vectors):
[[-0.2298477   0.88346102  0.40824829]
 [-0.52474482  0.24078249 -0.81649658]
 [-0.81964194 -0.40189603  0.40824829]]

Sigma (diagonal matrix of singular values):
[[9.52551809 0.        ]
 [0.         0.51430058]
 [0.         0.        ]]

Vt (right singular vectors):
[[-0.61962948 -0.78489445]
 [-0.78489445  0.61962948]]

Reconstructed A:
[[1. 2.]
 [3. 4.]
 [5. 6.]]
```

### Example of Reduced SVD

Reduced Singular Value Decomposition (Reduced SVD) is a simplified version of full SVD that only computes the essential components corresponding to the rank of the matrix A. It avoids computing unnecessary singular vectors by making it computationally efficient especially for large matrices. Following is the example which use reduced - size
**U**and**V**matrices to save memory for large matrices −
```
import numpy as np
from scipy.linalg import svd

# Input matrix
A = np.array([[1, 2], [3, 4], [5, 6]])

# Perform reduced SVD
U, s, Vt = svd(A, full_matrices=False)

print("Reduced U:")
print(U)

print("\nReduced Vt:")
print(Vt)

# Notice that U and Vt are smaller compared to full SVD
```

#### Output

Here is the output of the basic full SVD −

```
Reduced U:
[[-0.2298477   0.88346102]
 [-0.52474482  0.24078249]
 [-0.81964194 -0.40189603]]

Reduced Vt:
[[-0.61962948 -0.78489445]
 [-0.78489445  0.61962948]]
```

## Example - Reconstructing the Original Matrix

Here is the example of reconstructing the original matrix A from its SVD components involves multiplying the matrices U, and V
together −
```
import numpy as np
from scipy.linalg import svd

# Example matrix
A = np.array([[1, 2], [3, 4], [5, 6]])

# Compute SVD
U, s, Vt = svd(A)

# Rebuild the Sigma matrix (m x n)
Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, s)

# Reconstruct the original matrix
A_reconstructed = np.dot(U, np.dot(Sigma, Vt))

# Output
print("Original Matrix A:")
print(A)
print("\nReconstructed Matrix A from SVD components:")
print(A_reconstructed)
```

#### Output

Following is the example of reconstructing the original matrix from the svd components −

```
Reduced U:
[[-0.2298477   0.88346102]
 [-0.52474482  0.24078249]
 [-0.81964194 -0.40189603]]

Reduced Vt:
[[-0.61962948 -0.78489445]
 [-0.78489445  0.61962948]]
PS D:\Tutorialspoint> python sample.py
Reduced U:
[[-0.2298477   0.88346102]
 [-0.52474482  0.24078249]
 [-0.81964194 -0.40189603]]

Reduced Vt:
[[-0.61962948 -0.78489445]
 [-0.78489445  0.61962948]]
```

---

## 47. SciPy - Cholesky Decomposition

*Source: [https://www.tutorialspoint.com/scipy/scipy_cholesky_decomposition.htm](https://www.tutorialspoint.com/scipy/scipy_cholesky_decomposition.htm)*

---

---
[Previous](/scipy/scipy_singular_value_decomposition.htm)[Quiz](/scipy/quiz_on_scipy_cholesky_decomposition.htm)[Next](/scipy/scipy_solving_linear_systems.htm)
## Cholesky Decomposition in SciPy
**Cholesky decomposition**is a numerical technique used to decompose a positive-definite matrix into the product of a lower triangular matrix and its transpose. This is particularly useful in numerical computations such as solving systems of linear equations, optimizing algorithms or performing Monte Carlo simulations.
In SciPy the
**Cholesky decomposition**can be performed using the**scipy.linalg.cholesky()**function. We can define**Cholesky decomposition**for a lower triangular matrix L, for a given symmeytric positive - definite matrix A as follows −
```
A = LL^T
```

Where −

- **L**is a lower triangular matrix.
- **L**is the transpose of L.
Alternatively we can also compute
**Cholesky decomposition**for the upper triangular matrix U as follows −
```
A = U^TU
```

Where −

- **U**is a Upper triangular matrix.
- **U**is the transpose of U.
## Syntax of scipy.linalg.cholesky()

Following is the syntax of using the
**scipy.linalg.cholesky()**function to compute**Cholesky Decomposition**−
```
scipy.linalg.cholesky(a, lower=False, overwrite_a=False, check_finite=True)
```

Where −

- **a(array_like)**− The matrix A to decompose. It must be symmetric and positive-definite.
- **lower(bool, optional)**− If True then computes the lower triangular matrix L and if False(default) then computes the upper triangular matrix U.
- **overwrite_a(bool, optional)**− If True then allows modification of the input matrix to save memory. Default value is False.
- **check_finite(bool, optional)**− If True(default) then checks if the input matrix contains only finite numbers. This may be skipped for performance reasons.
The
**scipy.linalg.cholesky()**function returns the cholesky factor of the input matrix A. If this factor is**lower = True**then it is lower triangular otherwise it is upper triangular.
### Basic Cholesky Decomposition

Following is the example of computing the basic
**Cholesky Decomposition**using the**scipy.linalg.cholesky()**function in scipy −
```
import numpy as np
from scipy.linalg import cholesky

# Define a symmetric, positive-definite matrix
A = np.array([[4, 2], 
              [2, 3]])

# Perform Cholesky decomposition to get the lower triangular matrix
L = cholesky(A, lower=True)

# Print the results
print("Matrix A:")
print(A)

print("\nLower Triangular Matrix L:")
print(L)

# Verify: Reconstruct the original matrix using L
A_reconstructed = L @ L.T
print("\nReconstructed Matrix A (L @ L.T):")
print(A_reconstructed)
```

#### Output

Here is the output of the basic cholesky decomposition computed using the
**scipy.linalg.cholesky()**function −
```
Matrix A:
[[4 2]
 [2 3]]

Lower Triangular Matrix L:
[[2.         0.        ]
 [1.         1.41421356]]

Reconstructed Matrix A (L @ L.T):
[[4. 2.]
 [2. 3.]]
```

### Setting lower parameter to True

When the lower parameter in
**scipy.linalg.cholesky()**is set to True then the function computes the lower triangular matrix 
L. Following is the example of copmuting it −
```
import numpy as np
from scipy.linalg import cholesky

# Define a symmetric, positive-definite matrix
A = np.array([[6, 3, 4], 
              [3, 6, 5], 
              [4, 5, 10]])

# Perform Cholesky decomposition with lower=True
L = cholesky(A, lower=True)

# Print the results
print("Matrix A:")
print(A)

print("\nLower Triangular Matrix L (lower=True):")
print(L)

# Verify: Reconstruct the original matrix using L
A_reconstructed = L @ L.T
print("\nReconstructed Matrix A (L @ L.T):")
print(A_reconstructed)
```

#### Output

Here is the output of the basic cholesky decomposition computed using the
**scipy.linalg.cholesky()**function −
```
Matrix A:
[[4 2]
 [2 3]]

Lower Triangular Matrix L:
[[2.         0.        ]
 [1.         1.41421356]]

Reconstructed Matrix A (L @ L.T):
[[4. 2.]
 [2. 3.]]
PS D:\Tutorialspoint> python sample.py
Matrix A:
[[ 6  3  4]
 [ 3  6  5]
 [ 4  5 10]]

Lower Triangular Matrix L (lower=True):
[[2.44948974 0.         0.        ]
 [1.22474487 2.12132034 0.        ]
 [1.63299316 1.41421356 2.30940108]]

Reconstructed Matrix A (L @ L.T):
[[ 6.  3.  4.]
 [ 3.  6.  5.]
 [ 4.  5. 10.]]
```

## Key Points

Here are the key points of the Cholesky Decomposition in scipy −

- **Matrix Requirements**− The matrix A must be symmetric and positive-definite. If A is not positive-definite then the decomposition will fail.
- **Efficiency**− Cholesky decomposition is faster and more stable than other methods like LU decomposition for positive-definite matrices because it takes advantage of the symmetry and positive-definiteness.
- **Error Handling**− If the matrix is not positive-definite then a**LinAlgError**is raised.
## Applications
**Cholesky decomposition**is a powerful numerical tool with applications in various fields particularly in computational mathematics, physics, engineering and machine learning. Below are the key areas where it is commonly used −
- **Solving Systems of Linear Equations**− For symmetric positive-definite matrices the Cholesky decomposition provides a more efficient way to solve Ax=b compared to methods like LU decomposition.
- **Optimization Problems**− Many optimization problems involve minimizing quadratic objective functions subject to linear constraints. Cholesky decomposition is used to efficiently solve the systems arising in QP and also used in Machine Learning to optimization algorithms for training models like ridge regression and kernel methods.
- **Monte Carlo Simulations**− To generate correlated random variables the Cholesky decomposition is used to transform uncorrelated normal random variables.
- **Numerical Stability**− Cholesky decomposition is more numerically stable than LU decomposition for symmetric positive-definite matrices because it avoids pivoting and exploits matrix symmetry and used in numerical libraries and algorithms to improve the accuracy of results.

---

## 48. SciPy - Solving Linear Systems

*Source: [https://www.tutorialspoint.com/scipy/scipy_solving_linear_systems.htm](https://www.tutorialspoint.com/scipy/scipy_solving_linear_systems.htm)*

---

---
[Previous](/scipy/scipy_cholesky_decomposition.htm)[Quiz](/scipy/quiz_on_scipy_solving_linear_systems.htm)[Next](/scipy/scipy_eigenvalues_eigenvectors.htm)
In linear algebra
**solving linear systems**refers to finding the solution(s) to a system of linear equations. A linear system can be represented as follows −
```
A . x = b
```

Where −

- **A**is a matrix of coefficients).
- **x**is the vector of variables (unknowns)
- **b**is the vector of constants.
In SciPy
**solving linear systems**is done using several methods depending on the type and properties of the matrix A. SciPy provides highly optimized functions to solve linear systems directly or via matrix decompositions such as LU, Cholesky, QR and others.
## SciPy Function to Solve Linear Systems

The primary function in SciPy to solve a linear system is
**scipy.linalg.solve()**. This function is used to compute the solution x of the system Ax=b where A is the matrix of coefficients and b is the vector of constants.
### Syntax

Below is the syntax of the function
**scipy.linalg.solve()**which is used to Solve Linear systems −
```
scipy.linalg.solve(
   a, 
   b, 
   sym_pos=False, 
   lower=False, 
   overwrite_a=False, 
   overwrite_b=False, 
   check_finite=True
)
```

- **a(array_like)**− The coefficient matrix A which should be a square matrix i.e., number of rows = number of columns and in most cases non-singular i.e., invertible.
- **b(array_like)**− The right-hand side vector or matrix i.e., b which must have the same number of rows as the matrix A.
- **sym_pos(bool, optional)**− If True then it indicates that the matrix A is symmetric and positive-definite. In this case the function uses more efficient algorithms such as Cholesky decomposition.
- **lower(bool, optional)**− If True then it indicates that the matrix A is lower triangular. This will reduce the computational cost.
- **overwrite_a(bool, optional)**− If True then allows overwriting the input matrix A in memory to save space which perform faster computations but changes the original matrix.
- **overwrite_b(bool, optional)**− If True then allows overwriting the input vector b in memory.
- **check_finite(bool, optional)**− If True then the function checks whether A and b contain only finite numbers. This helps  to avoid numerical issues but may incur some overhead.
This function returns the solution vector or matrix i.e., the values of the variables that satisfy the linear system Ax=b.

## Example of Solving a Linear System

### Example - usage of solve() function

Following is the example of
**scipy.linalg.solve()**function which is used to solve a system of linear equations with the shape 2 x 2 −
```
import numpy as np
from scipy.linalg import solve

# Define matrix A (2x2)
A = np.array([[3, 2],
              [1, 2]])

# Define vector b (2x1)
b = np.array([5, 5])

# Solve for x in Ax = b
x = solve(A, b)

print("Solution x:", x)
```

#### Output

Following is the output of the above code −

```
Solution x: [0.  2.5]
```

## Solving Linear Systems for Non-Square Matrices

### Example

In this example we have an over-determined system i.e., more equations than unknowns and we use the least-squares method to solve it −

```
import numpy as np
from scipy.linalg import solve

# Define the overdetermined matrix A (3x2 matrix)
A = np.array([[1, 2],
              [2, 3],
              [3, 4]])

# Define the right-hand side vector b
b = np.array([5, 6, 7])

# Solve using least-squares method
x = solve(A.T @ A, A.T @ b)

print("Least-squares solution x:", x)
```

#### Output

Following is the output of the above code −

```
Least-squares solution x: [-3.  4.]
```

## Solving a Singular System

### Example

Following is the example in which we have a singular matrix i.e., the system does not have a unique solution
**x + y = 1**and**x + y = 2**−
```
import numpy as np
from scipy.linalg import solve

# Define a singular matrix A (rows are linearly dependent)
A = np.array([[1, 1],
              [1, 1]])

# Define the right-hand side vector b
b = np.array([1, 2])

# Attempt to solve the system A * x = b
try:
    x = solve(A, b)
    print("Solution x:", x)
except Exception as e:
    print(f"Error: {e}")
```

#### Output

Following is the output of the above code −

```
Error: Matrix is singular.
```

---

## 49. SciPy - Eigenvalues and Eigenvectors

*Source: [https://www.tutorialspoint.com/scipy/scipy_eigenvalues_eigenvectors.htm](https://www.tutorialspoint.com/scipy/scipy_eigenvalues_eigenvectors.htm)*

---

---
[Previous](/scipy/scipy_solving_linear_systems.htm)[Quiz](/scipy/quiz_on_scipy_eigenvalues_eigenvectors.htm)[Next](/scipy/scipy_ndimage.htm)
In
**SciPy Eigenvalues and Eigenvectors**are computed as part of solving problems involving linear transformations specifically for square matrices. These values and vectors provide insight into the properties of a matrix and are widely used in various scientific and engineering domains.
Before understanding how scipy is used to work with the
**Eigenvalues and Eigenvectors**, let's understand about**Eigenvalues and Eigenvectors**as follows −
## What are Eigenvalues and Eigenvectors?
**Eigenvalues and Eigenvectors**are mathematical concepts used in linear algebra to analyze transformations represented by square matrices. They provide insights into how a matrix transforms a vector.
### What is Eigenvector (v)?

An
**Eigenvector**of a matrix**A**is a non-zero vector that only changes in scale i.e., not direction when the matrix is applied to it. Mathematically for a square matrix A the Eigenvector is given as follows −
```
Av = v
```

Where,
**v**is the Eigenvector,**A**is a square matrix of size**n x n**andis the corresponding eigenvalue.
### What is Eigenvalue()?

The eigenvalue is the scalar by which the eigenvector
**v**is stretched or compressed during the transformation.can be positive, negative or even complex.
### Finding Eigenvalues and Eigenvectors

When we want to find the
**Eigenvalues and Eigenvectors**of a square matrix**A**then we have to follow the below steps −
#### Step 1: Eigenvalues

The Eigenvalues of a sqaure matrix
**A**are the scalarsthat satisfy the eqaution as follows −
```
det(A - I) = 0
```

Now we have to rewrite the Egienvalue equation
**Av = v**as follows −
```
(A - I)v = 0
```

Where,
**I**is the identity matrix of the same size as A and**A - I**is the  new matrix obtained by subtracting  times the identity matrix from A.
This equation has non-zero
**v**only if**det(A - I) = 0**. This is called the**characteristic equation**and solving it gives the eigenvalues.
To solve the equation
**det(A - I) = 0**first we have to substituteinto the natrix**A - I**. Next we have to compute the determinant of the resulting matrix and then solve the resulting polynomial equation for.
The roots of the polynomial are the eigenvalues of
**A**.
#### Step 2: Eigenvectors

Once when we found the Eigenvalues then we can compute the Eigenvectors by solving the beloow equation −

```
(A - I)v = 0
```

This equation is a system of linear equations. It has infinitely many solutions because (AI) is singular i.e., its determinant is zero. Typically these solutions are expressed as a scaled version of one eigenvector.

## Computing Eigenvalues and Eigenvectors

In
**SciPy Eigenvalues and Eigenvectors**can be calculated using the**scipy.linalg.eig()**function which is part of the**scipy.linalg**module. This function is used for both general matrices and with specific optimizations, Hermitian or symmetric matrices.
### Example - Computing Eigenvalues and Eigenvectors of General matrix

Here is the example of computing the eigenvalues and eigenvectors of a general
**n x n**matrix using scipy −
```
import numpy as np
from scipy.linalg import eig

# Define a 3x3 matrix
A = np.array([[6, 2, 1],
              [2, 3, 1],
              [1, 1, 1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)

# Print results
print("Matrix A:")
print(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors (columns):")
print(eigenvectors)
```

#### Output

Here is the output of computing the eigenvalues and eigenvectors using scipy library −

```
Matrix A:
[[6 2 1]
 [2 3 1]
 [1 1 1]]

Eigenvalues:
[7.28799214+0.j 2.13307448+0.j 0.57893339+0.j]

Eigenvectors (columns):
[[ 0.86643225  0.49742503 -0.0431682 ]
 [ 0.45305757 -0.8195891  -0.35073145]
 [ 0.20984279 -0.28432735  0.9354806 ]]
```

### Hermitian and Symmetric Matrices
**Hermitian Matrix**is a square matrix that is equal to its own conjugate transpose, which is defined as follows −
```
A = A^H
```

Where, A
is the conjugate transpose of**A**.**Symmetric Matrix**is a special case of a Hermitian matrix where all elements are real so it satisfies and it is given as follows −
```
A = A^T
```

Where A
is the transpose of A.
### Example

Here's the example which shows how to compute the eigenvalues and eigenvectors of Hermitian matrix using scipy library −

```
import numpy as np
from scipy.linalg import eigh  

# Define a Hermitian matrix
A = np.array([[2, 1j],
              [-1j, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(A)

print("Hermitian Matrix A:")
print(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)
```

#### Output

Here is the output of computing the eigenvalues and eigenvectors of Hermitian matrix using scipy library −

```
Hermitian Matrix A:
[[ 2.+0.j  0.+1.j]
 [-0.-1.j  3.+0.j]]

Eigenvalues:
[1.38196601 3.61803399]

Eigenvectors:
[[-0.85065081+0.j          0.52573111+0.j        ]
 [ 0.        -0.52573111j  0.        -0.85065081j]]
```

### Example

Following is the example of computing the eigenvalues and eigenvectors of symmetric matrix using scipy library −

```
import numpy as np
from scipy.linalg import eigh  

# Define a Symmetric matrix
A = np.array([[4, 2],
              [2, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(A)

print("Symmetric Matrix A:")
print(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)
```

#### Output

Here is the output of computing the eigenvalues and eigenvectors of Symmetric matrix using scipy library −

```
Symmetric Matrix A:
[[4 2]
 [2 3]]

Eigenvalues:
[1.43844719 5.56155281]

Eigenvectors:
[[ 0.61541221 -0.78820544]
 [-0.78820544 -0.61541221]]
```

---

## 50. SciPy - Ndimage

*Source: [https://www.tutorialspoint.com/scipy/scipy_ndimage.htm](https://www.tutorialspoint.com/scipy/scipy_ndimage.htm)*

---

---

## 51. SciPy - Reading and Writing Images

*Source: [https://www.tutorialspoint.com/scipy/scipy_reading_writing_images.htm](https://www.tutorialspoint.com/scipy/scipy_reading_writing_images.htm)*

---

---
[Previous](/scipy/scipy_ndimage.htm)[Quiz](/scipy/quiz_on_scipy_reading_writing_images.htm)[Next](/scipy/scipy_image_transformation.htm)
In
**SciPy Reading and Writing**images is typically done using the**scipy.ndimage**module in combination with other libraries such as PIL (Python Imaging Library) or imageio. While scipy.ndimage is powerful for image processing it doesn't naively handle image I/O (input/output).
For reading and writing images the SciPy relies on the imageio and PIL libraries which allow handling different image formats such as PNG, JPEG, TIFF, etc. Heres an in-depth explanation of how to read and write images in Python using the tools available through SciPy and related libraries.

## Image Formats

There are several common image formats are supported for reading and writing operations. Following is an overview of the image formats and related functionalities in scipy −

- **JPEG (.jpg, .jpeg):**Joint Photographic Experts Group (JPEG) is used for compressing photographic images and  the lossy compression supports varying quality levels.
- **PNG (.png):**Portable Network Graphics ideal for lossless compression. This supports transparency i.e., alpha channel.
- **BMP (.bmp):**Bitmap is used for uncompressed or minimally compressed image format. This is simple and widely supported.
- **TIFF (.tif, .tiff):**Tagged Image File Format is highly flexible format which supports lossless compression and suitable for high-quality images.
- **GIF (.gif):**Graphics Interchange Format is often used for simple animations or static images and limited to 256 colors.
- **PPM/PGM (.ppm, .pgm):**Portable pixmap format is simple, raw image format which is primarily used in academic and research contexts.
## Reading Images in SciPy

Scipy uses the
**scipy.ndimage.imread()**function for reading the images but now it is deprecated so we can use Pillow or imageio to read color images along with the scipy library.
### Reading Images Using PIL
**Pillow**is another popular library for image I/O in Python which is a fork of the original Python Imaging Library (PIL). As we all know that SciPy doesn't have built-in functions for reading and writing images so Pillow is often used for these tasks.
Here is an example of reading the image with the help of
**PIL**library along with the scipy library −
```
from PIL import Image
import numpy as np

# Load the image using Pillow
img = Image.open("/Images/images.jpeg")

# Convert to a NumPy array
img_array = np.array(img)

print(f"Image shape: {img_array.shape}")  # e.g., (height, width, 3) for RGB
```

Following is the output of reading an image with the help op
**PIL**library −
```
Image shape: (162, 311)
```

### Reading Images Using imageio
**imageio**is a Python library that supports reading and writing images in various formats such as PNG, JPEG, TIFF and others. It is easy to use and works well with SciPys image processing tools and which is the most recommended one.
Following is the example which uses the
**imageio**library for reading the image along the scipy library −
```
import imageio.v2 as imageio

# Reading an image
image = imageio.imread("/Images/images.jpeg")
print(f"Image shape: {image.shape}")  # e.g., (height, width, 3) for RGB
```

Following is the output of reading an image with the help op
**PIL**library −
```
Image shape: (162, 311, 3)
```

## Writing Images in Scipy

Once the image has been processed or transformed then we can write the result to a new file. As with reading we will typically use imageio or Pillow to save images.

### Writing Images Using imageio

The
**imwrite()**function from imageio writes the NumPy array image to a file in the specified format such as PNG, JPEG.
Here is the example of writing the image with the help of
**imageio**library**imwrite()**function −
```
import imageio.v2 as imageio
import numpy as np

# Reading an image
img = imageio.imread("/Images/images.jpeg")
 
# Convert the image to a numpy array for further processing
image = np.array(img)

# Write image to a file
imageio.imwrite("/Images/output_image.jpg", image)
print("Writing image is completed")
```

Here is the output after writing the image in the defined file −

```
Writing image is completed
```

### Writing Images Using PIL(pillow)

With Pillow library we can easily convert the NumPy array back into a
**PIL.Image**object and save it using the**save()**function. Following is the example which shows how to write the image with the help of pillow library −
```
from PIL import Image
import numpy as np

# Load the image using Pillow
img = Image.open("/Images/images.jpeg")

# Convert to a NumPy array
img_array = np.array(img)

# Convert the NumPy array back to a PIL Image object
img = Image.fromarray(img_array)

# Save the image to a file
img.save('/Images/output_image.png')
print("Writing into the file is completed")
```

Here is the output after writing the image in the defined file using the
**PIL**−
```
Writing into the file is completed
```

## Handling Color Images
**Handling color images**in SciPy involves working with image data in the form of multi-dimensional arrays where each dimension corresponds to image attributes such as height, width and color channels. Here's a guide on how to process color images using SciPy and related libraries −
### Structure of Color Images

A color image is typically represented as a 3D NumPy array which can be defined as follows −

- **Shape:**This consists of three attributes namely height, width and channels.
- **Channels:**The channels are of two types such as**RGB(Red, Green, Blue)**which is the common format for color image and**RGBA**which includes an additional alpha channel for transparency.
- **Grayscale:**It is a single channel typically with shape (height, width)
### Converting Color Formats
**Converting color formats**is a fundamental task in image processing. It involves changing the representation of image colors between different models or formats such as RGB to grayscale, RGBA to RGB or converting images for specific processing tasks.
While scipy itself doesnt directly provide utilities for advanced color format conversion it can be achieved easily using libraries such as Pillow (PIL) and OpenCV often in conjunction with
**scipy.ndimage**module for further processing.
Let's see the common color format conversions using scipy along with other libraries −

#### RGB to Grayscale

Grayscale reduces the color channels from three namely, Red, Green, Blue to one which represent the intensity. Here is the example of converting the RGB image to grayscale with the help of pillow library −

```
from PIL import ImageOps
from PIL import Image

# Load an RGB image
img = Image.open("/Images/images.jpeg")

# Convert to grayscale
gray_img = ImageOps.grayscale(img)

# Save or display
gray_img.save("/Images/gray_example.jpg")
gray_img.show()
```

Following is the output of the Greyscale image −
![Greyscale image](/scipy/images/gray_example.jpg)
#### RGBA to RGB

If an image includes an alpha channel i.e., transparency then converting it to RGB removes this channel. Below is the example of converting the RGBA image to RGB image using the Pillow library −

```
from PIL import ImageOps
from PIL import Image

# Load an RGB image
img = Image.open("/Images/images.jpeg")

# Convert RGBA to RGB
rgb_img = img.convert("RGB")
rgb_img.save("/Images/rgb_example.jpg")
rgb_img.show()
```

Here is the output of the RGB image after converting from RGBA image −
![RGB image](/scipy/images/rgb_example.jpg)
### Converting Formats for Specific Libraries

When working with image data in Python different libraries have unique requirements for image formats. To ensure compatibility we often need to convert images into the specific formats required by these libraries. Here's an overview of how to handle converting formats for specific libraries like SciPy, Pillow, OpenCV and others.

#### SciPy (NumPy Array Format)

SciPy primarily works with images as NumPy arrays where we can convert between image file formats and NumPy arrays using libraries like Pillow or imageio.

Following is the example which convert an image to Numpy array using pillow library −

```
from PIL import Image
import numpy as np

# Load the image using Pillow
img = Image.open("/Images/gray_example.jpeg")

# Convert to NumPy array
img_array = np.array(img)
print(img_array.shape)
```

Here is the output of the converting the image into numpy array −

```
(162, 311)
```

Here is the another example which convert Numpy array into image using pillow library −

```
from PIL import Image
import numpy as np

# Load the image using Pillow
img = Image.open("/Images/gray_example.jpg")

# Convert to NumPy array
img_array = np.array(img)

# Convert NumPy array back to an image
img_reconstructed = Image.fromarray(img_array)

# Save the image
img_reconstructed.save("/Images/reconstructed.jpg")
img_reconstructed.show()
```

Here is the output of the converting the image into numpy array −
![reconstructed image](/scipy/images/reconstructed.jpg)
#### Pillow (Image Object Format)

Pillow uses its own Image object format to ensure compatibility with other libraries where we might need to convert formats.

Below is the example which converts the numpy array to pillow −

```
from PIL import Image
import numpy as np

# Create a dummy NumPy array
array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Convert to Pillow Image
img = Image.fromarray(array)
img.show()
```

Here is the output of the converting the array into pillow image −
![reconstructed image](/scipy/images/reconstructed.jpg)
Following is the example which shows how to convert the pillow image into array −

```
from PIL import Image
import numpy as np

# Create a dummy NumPy array
array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Convert to Pillow Image
img = Image.fromarray(array)
# Convert Pillow Image to NumPy array
array_from_pillow = np.array(img)
print(array_from_pillow.shape)
```

Here is the output of the converting the pillow image into array −

```
(100, 100, 3)
```

## Image Processing Libraries in Python

Here are the different common libraries where we can use on images −
S.NoLibrary & Description1[SciPy](/scipy/index.htm)
Works with images as NumPy arrays (H  W  C). Use Pillow or imageio for conversion.2[Pillow](/python_pillow/index.htm)
Uses Image objects. Convert to/from NumPy using Image.fromarray() and np.array().3[OpenCV](/opencv_python/index.htm)
Uses NumPy arrays in BGR format (H  W  C). Convert to/from RGB using cv2.cvtColor().4[matplotlib](/matplotlib/index.htm)
Expects NumPy arrays in RGB format for visualization. Convert using cv2.cvtColor() if needed.5**imageio**
Supports multi-format reading/writing. Outputs images as NumPy arrays directly.6[TensorFlow](/tensorflow/index.htm)/[Keras](/keras/index.htm)
Expects images as tensors (H  W  C). Convert from NumPy using tf.convert_to_tensor().7[PyTorch](/pytorch/index.htm)
Expects images as tensors in C  H  W format. Use .permute() for channel order conversion.

---

## 52. SciPy - Image Transformation

*Source: [https://www.tutorialspoint.com/scipy/scipy_image_transformation.htm](https://www.tutorialspoint.com/scipy/scipy_image_transformation.htm)*

---

---
[Previous](/scipy/scipy_reading_writing_images.htm)[Quiz](/scipy/quiz_on_scipy_image_transformation.htm)[Next](/scipy/scipy_filtering_edge_detection.htm)**SciPy's Image Transformation**which is a functionality provides a set of tools for performing various transformations on images such as scaling, rotating, translating and warping. These operations are useful in applications such as image processing, computer vision and data augmentation for machine learning models.
SciPy uses its
**scipy.ndimage**module to handle image transformations. This module is part of the broader SciPy library and is designed specifically for multidimensional image processing.
## Key Features of Image Transformation in SciPy

Below are the key features that make SciPy's image transformation capabilities powerful and adaptable −

- **Affine Transformations:**This include operations such as scaling, rotating, shearing and translating an image while maintaining straight lines and parallelism. To perform this feature we can use the function**scipy.ndimage.affine_transform()**.
- **Geometric Transformations:**These transformations involve remapping pixels using a mathematical function that specifies new pixel coordinates. These are useful for more general, custom transformations such as warping or bending the image and this can be achieved with the help of**scipy.ndimage.affine_transform()**function.
- **Rotation:**Rotating an image around a point by a specified angle can be achieved with this rotation feature. This feature supports interpolation for smoother rotations and the option to preserve or change the image shape i.e., bounding box. This can be achived with the help of**scipy.ndimage.rotate()**
- **Zoom (Scaling):**Scaling the image either uniformly or non-uniformly along each axis. This feature is mainly used for resizing images or performing zooming effects for images. To achieve this scaling we can use the function**scipy.ndimage.zoom()**.
- **Shifting (Translation):**Translating (shifting) an image by a specified number of pixels along each axis. This feature is used for translating an image by arbitrary amounts without affecting its content. The**scipy.ndimage.shift()**function is used to perform shift operation.
- **Warping:**When applying a non-linear transformation that remaps the coordinates of an image to custom positions. This allows for more complex image distortions or adjustments. Here this is used for tasks such as perspective corrections or non-linear adjustments to the image. It uses the function**scipy.ndimage.map_coordinates()**.
- **Interpolation Methods:**Different interpolation methods control how pixel values are estimated during transformations. The Interpolation methods are such as Nearest neighbor, Bilinear, Bicubic. Many transformation functions such as rotate, affine_transform, zoom allow users to select interpolation methods.
- **Boundary Handling:**It control how pixel values are handled at the edges of the image where transformation might go beyond the original boundaries. Here in this we have the different options such as constant, nearest, wrap and mirror. We can use parameter**mode='constant'**and**mode='nearest'**to perform this boundary handling.
- **Custom Transformation Functions:**Users can specify their own transformation functions to remap pixels in a customized manner. The function**scipy.ndimage.geometric_transform()**is used to perform Custom Transformation.
- **Multi-dimensional Support:**SciPy supports transformations not just for 2D images but also for higher-dimensional arrays such as 3D, 4D, etc by making it versatile for volumetric data and time-series data. All the above transformation functions work for n-dimensional arrays.
Let's see commonly used transformation techniques with examples in brief −

## Rotation of Image
**Rotation**is a fundamental image transformation that turns an image around a specific point which is usually the center by a given angle. SciPy provides the**scipy.ndimage.rotate()**function for this purpose which is flexible and supports a range of options like interpolation and boundary handling.
### Syntax

Following is the syntax of the function
**scipy.ndimage.rotate()**−
```
scipy.ndimage.rotate(input, angle, axes=(1, 0), reshape=True, order=3, mode='constant', cval=0.0, prefilter=True)
```

Here are the parameters of the function
**scipy.ndimage.rotate()**−
- **input:**The input array (image).
- **angle:**The rotation angle in degrees.
- **axes:**The axes defining the plane of rotation with default value as (1, 0) for 2D images.
- **reshape:**Whether to adjust the output shape to fit the rotated image.
- **order:**The interpolation order with the default value is 3 for cubic.
- **mode:**Boundary mode such as 'constant', 'reflect'.
- **cval:**Constant value to fill when mode = 'constant'.
- **prefilter:**Pre-filtering for higher-order interpolation i.e., usually left as True.
Following is the example to perform the rotation of the image with the help of
**scipy.ndimage.rotate()**function −
```
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

# Create a simple 2D array as an image (gradient)
image = np.arange(100).reshape(10, 10)

# Rotate the image by 45 degrees
rotated_image = rotate(image, angle=45, reshape=True)

# Display the original and rotated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Rotated Image (45)")
plt.imshow(rotated_image, cmap='gray')
plt.colorbar()
plt.show()
```

Here is the output of the roatated image using the
**scipy.ndimage.rotate()**function −![Rotated Image](/scipy/images/roatation_image.jpg)
Here is another example which rotates the image by defining the boundary modes to the
**scipy.ndimage.rotate()**function −
```
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

# Create a simple 2D array as an image (gradient)
image = np.arange(100).reshape(10, 10)

rotated_constant = rotate(image, angle=45, mode='constant', cval=255)
rotated_nearest = rotate(image, angle=45, mode='nearest')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Mode: Constant (cval=255)")
plt.imshow(rotated_constant, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Mode: Nearest")
plt.imshow(rotated_nearest, cmap='gray')
plt.colorbar()

plt.show()
```

Here is the output of the roatated image using the
**scipy.ndimage.rotate()**function within the defined boundaries −![Rotated Image with boundary](/scipy/images/rotated_boundary.jpg)
## Scaling an Image
**Scaling**is the process of resizing an image by increasing or decreasing its dimensions along one or more axes. SciPy provides the**scipy.ndimage.zoom()**function for this purpose which allows users to scale images with control over interpolation and boundary handling.
### Syntax

Following is the syntax for using the
**scipy.ndimage.zoom()**function to perform scaling on the image −
```
scipy.ndimage.zoom(input, zoom, order=3, mode='constant', cval=0.0, prefilter=True)
```

Here are the parameters of the function
**scipy.ndimage.rotate()**−
- **input:**The input array (image).
- **zoom:**The zoom factor(s). A scalar or a sequence of factors for each axis.
- **order:**Interpolation order (default is 3 for cubic).
- **mode:**Boundary mode such as 'constant', 'reflect'.
- **cval:**Constant value to fill for mode='constant'.
- **prefilter:**Whether to prefilter the input for higher-order interpolation.
Following is an example which perform scaling to an image uniformly with the help of the function
**scipy.ndimage.zoom()**−
```
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Create a simple 2D array (gradient image)
image = np.arange(100).reshape(10, 10)

# Scale the image by a factor of 2
scaled_image = zoom(image, zoom=2)

# Display the original and scaled images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Scaled Image (Zoom=2)")
plt.imshow(scaled_image, cmap='gray')
plt.colorbar()

plt.show()
```

Here is the output of the image which had scaling uniformly −
![Scaling Uniformly](/scipy/images/scaling_uniformly.jpg)
Here is the example of scaling an image with the non-uniform factors by using the function
**scipy.ndimage.zoom()**−
```
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Create a simple 2D array as an image (gradient)
image = np.arange(100).reshape(10, 10)

# Scale with different factors for each axis
scaled_non_uniform = zoom(image, zoom=(1, 2))  # Scale rows by 1, columns by 2

plt.figure(figsize=(6, 6))
plt.title("Scaled Non-Uniformly (Zoom=(1, 2))")
plt.imshow(scaled_non_uniform, cmap='gray')
plt.colorbar()
plt.show()
```

Here is the output of the image scaled with non uniform factors −
![Scaling Non Uniform](/scipy/images/scaling_nonuniform.jpg)
Below is another example which specifies the boundaries to the function
**scipy.ndimage.zoom()**to perform the scaling on the image −
```
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Create a simple 2D array as an image (gradient)
image = np.arange(100).reshape(10, 10)

scaled_constant = zoom(image, zoom=2, mode='constant', cval=255)
scaled_reflect = zoom(image, zoom=2, mode='reflect')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Mode: Constant (cval=255)")
plt.imshow(scaled_constant, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Mode: Reflect")
plt.imshow(scaled_reflect, cmap='gray')
plt.colorbar()

plt.show()
```

Here is the output of the scaling image within specified boundary modes −
![Scaling Boundary](/scipy/images/scaling_boundary.jpg)
## Translation of Image
**Translation**refers to shifting an image by a specified distance along one or more axes. SciPy provides the**scipy.ndimage.shift()**function for performing translation operations on images or multidimensional arrays. This can be useful for image augmentation, alignment or spatial transformations.
### Syntax

Here is the syntax for using the
**scipy.ndimage.shift()**function to perform the image translation −
```
scipy.ndimage.shift(input, shift, order=3, mode='constant', cval=0.0, prefilter=True)
```

Here are the parameters of the function
**scipy.ndimage.shift()**−
- **input:**The input array (image).
- **shift:**The shift values, specified as a scalar or a sequence of shifts for each axis.
- **order:**Interpolation order with the default value 3 for cubic).
- **mode:**Boundary mode such as 'constant', 'reflect', 'nearest' or 'wrap'.
- **cval:**Constant value to fill for mode='constant'.
- **prefilter:**Whether to prefilter the input for higher-order interpolation.
Here is the example which performs the basic translation of the image by using the function
**scipy.ndimage.shift()**−
```
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt

# Create a simple 2D array (gradient image)
image = np.arange(100).reshape(10, 10)

# Translate the image by 2 pixels down and 3 pixels to the right
shifted_image = shift(image, shift=(2, 3))

# Display the original and shifted images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Shifted Image (2 down, 3 right)")
plt.imshow(shifted_image, cmap='gray')
plt.colorbar()

plt.show()
```

Here is the output of the simple image translation with the help of
**scipy.ndimage.shift()**function −![Simple Translation](/scipy/images/simple_translation.jpg)
Following is another example which performs the image translation within the specified boundary modes passed to the
**scipy.ndimage.shift()**function −
```
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt

# Create a simple 2D array (gradient image)
image = np.arange(100).reshape(10, 10)

# Translate with constant boundary (fill with 255)
shifted_constant = shift(image, shift=(2, 3), mode='constant', cval=255)

# Translate with reflect boundary mode
shifted_reflect = shift(image, shift=(2, 3), mode='reflect')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Mode: Constant (cval=255)")
plt.imshow(shifted_constant, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Mode: Reflect")
plt.imshow(shifted_reflect, cmap='gray')
plt.colorbar()

plt.show()
```

Here is the output of the image in which translation was done within the specified boundary modes using the
**scipy.ndimage.shift()**function −![Translation with Boundary Modes](/scipy/images/translation_boundary.jpg)
Here is the example of performing Non - uniform translation −

```
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt

# Create a simple 2D array (gradient image)
image = np.arange(100).reshape(10, 10)

# Shift by different amounts along rows and columns
shifted_non_uniform = shift(image, shift=(0, 5))  # No vertical shift, 5-pixel horizontal shift

plt.figure(figsize=(6, 6))
plt.title("Non-Uniform Translation (0, 5)")
plt.imshow(shifted_non_uniform, cmap='gray')
plt.colorbar()
plt.show()
```

Here is the output of the image with non uniform translation −
![Non-uniformTranslation](/scipy/images/translation_nonuniform.jpg)
## Affine Transformations
**Affine transformations**are a class of geometric transformations that preserve points, straight lines and planes. They can involve operations such as translation, rotation, scaling and shearing while maintaining the basic structure of the image such as parallel lines remain parallel and ratios of distances are preserved. In SciPy affine transformations can be performed using the**scipy.ndimage.affine_transform()**function.
### Syntax

Following is the syntax for using the
**scipy.ndimage.affine_transform()**function to perform the geometric transformations −
```
scipy.ndimage.affine_transform(input, matrix, offset=0, output_shape=None, order=3, mode='constant', cval=0.0, prefilter=True)
```

Here are the parameters of the function
**scipy.ndimage.affine_transform()**−
- **input:**The input array (image).
- **matrix:**The linear transformation matrix which is typically a 2x2 matrix for 2D images or 3x3 for 3D images.
- **offset:**The translation vector specify the amount to shift the image after applying the matrix transformation.
- **output_shape:**The shape of the output array. If not specified then it is inferred from the input.
- **order:**Interpolation order with the default value 3 for cubic interpolation.
- **mode:**Boundary mode such as 'constant', 'reflect', 'nearest' or 'wrap'.
- **cval:**Constant value to fill for mode='constant'.
- **prefilter:**Whether to prefilter the input for higher-order interpolation.
Here is the example of simple Affine transformation which performs scaling and translation of the image together by using the function
**scipy.ndimage.affine_transform()**−
```
import numpy as np
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt

# Create a simple 2D image (gradient)
image = np.arange(100).reshape(10, 10)

# Define a scaling matrix (scale by 2 in x and y)
matrix = np.array([[2, 0], [0, 2]])  # Scaling by 2 in both axes
offset = [0, 0]  # No translation

# Apply affine transformation (scaling)
scaled_image = affine_transform(image, matrix, offset=offset, order=1)

# Display original and scaled image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Scaled Image (2x)")
plt.imshow(scaled_image, cmap='gray')
plt.colorbar()

plt.show()
```

Here is the output of the simple affline transformation using the
**scipy.ndimage.affline_transform()**function −![Simple Affline transformation](/scipy/images/simple_affline.jpg)
Following is the example which performs the Affine Transformation with Rotation and Translation using the function
**scipy.ndimage.affine_transform()**−
```
import numpy as np
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt

# Create a simple 2D image (gradient)
image = np.arange(100).reshape(10, 10)

# Define a scaling matrix (scale by 2 in x and y)
matrix = np.array([[2, 0], [0, 2]])  # Scaling by 2 in both axes
offset = [0, 0]  # No translation

# Define a rotation matrix (rotate by 45 degrees)
theta = np.radians(45)  # Convert to radians
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
offset = [0, 0]  # No translation

# Apply affine transformation (rotation)
rotated_image = affine_transform(image, rotation_matrix, offset=offset, order=1)

# Display original and rotated image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Rotated Image (45)")
plt.imshow(rotated_image, cmap='gray')
plt.colorbar()

plt.show()
```

Here is the output of the affline transformation along with rotation and translation using the
**scipy.ndimage.affline_transform()**function −![Affline transformation with rotation](/scipy/images/affline_rotation.jpg)
In this example we will perform the Affline Transformation with shearing by using the function
**scipy.ndimage.affine_transform()**−
```
import numpy as np
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt

# Create a simple 2D image (gradient)
image = np.arange(100).reshape(10, 10)

# Define a scaling matrix (scale by 2 in x and y)
matrix = np.array([[2, 0], [0, 2]])  # Scaling by 2 in both axes
offset = [0, 0]  # No translation

# Define a shearing matrix (shear along x-axis)
shear_matrix = np.array([[1, 1], [0, 1]])  # Shear along x-axis
offset = [0, 0]  # No translation

# Apply affine transformation (shearing)
sheared_image = affine_transform(image, shear_matrix, offset=offset, order=1)

# Display original and sheared image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Sheared Image")
plt.imshow(sheared_image, cmap='gray')
plt.colorbar()

plt.show()
```

Following is the output of the affline transformation which performs shearing of the image with the help of
**scipy.ndimage.affline_transform()**function −![Affline shearing](/scipy/images/affline_sheering.jpg)

---

## 53. SciPy - Filtering and Edge Detection

*Source: [https://www.tutorialspoint.com/scipy/scipy_filtering_edge_detection.htm](https://www.tutorialspoint.com/scipy/scipy_filtering_edge_detection.htm)*

---

---
[Previous](/scipy/scipy_image_transformation.htm)[Quiz](/scipy/quiz_on_scipy_filtering_edge_detection.htm)[Next](/scipy/scipy_top_hat_filters.htm)
## What is Filtering in SciPy?
**Filtering in image processing**is a fundamental technique used for a variety of tasks such as noise reduction, image enhancement and feature extraction. Image filters work by modifying or processing an image's pixel values based on its neighbors or applying mathematical transformations.
In SciPy filters can be applied to images to perform operations like smoothing, sharpening and edge detection. In this chapter let's see about the key concepts, types of filters and how to implement filters using SciPy.

## What is Edge Detection?
**Edge detection**is a crucial technique in image processing used to identify points in an image where there is a sharp contrast in pixel intensity which is often corresponding to boundaries or transitions between different regions of the image. In the view of SciPy image processing the edge detection is typically achieved by using filters that highlight areas of rapid intensity change usually via the computation of gradients or second-order derivatives.
In SciPy, edge detection can be performed using several techniques such as Sobel, Prewitt, Scharr, Roberts and Laplacian filters. These filters are applied to an image to highlight areas with significant intensity gradients which typically correspond to edges.

Before getting deep into the Filtering and Edge Detection techniques we need to understand some basics concepts for filtering as follows −

## Kernel

A
**kernel**is also known as a mask, filter or convolution matrix which is a small, matrix-shaped set of numerical weights used in image filtering. It is applied to an image by performing a mathematical operation i.e., typically convolution between the kernel and small sections of the image. This operation produces a new value for each pixel by depending on the kernel's purpose.
For a kernel
**K**and an image**I**the operation at a pixel**(x,y)**can be expressed as follows −![Kernel Formula](/scipy/images/kernel_formula.jpg)
Where −

- **I'(x,y):**Output pixel value at position (x,y).
- **K(i,j):**Kernel weight at offset (i,j).
- **n,m:**Half the width and height of the kernel e.g., for a 33 kernel n=m=1.
### Role of Kernels in Filtering

Kernels are designed to perform specific operations such as mentioned below −

- **Blurring:**Reducing noise and smoothing images.
- **Edge Detection:**Highlighting boundaries and transitions.
- **Enhancing Details:**Sharpening fine image features.
### Key Properties of Kernels

Here are the key properties of Kernels in SciPy Image Processing −

- **Symmetry:**Some kernels such as Gaussian are symmetric which ensure equal effects in all direction
- **Normalization:**Kernels like the mean filter often normalize values i.e., divide by the sum of weights to maintain intensity balance.
- **Edge Effects:**Handling edges of the image means where the kernel extends beyond the image boundary requires padding techniques such as zero-padding or reflection.
### Types of Kernels

Following are different types of kernel which determines the effect it has on the image.

- **Smoothing Kernels:**These kernels reduce noise and smooth an image by averaging the intensity of neighboring pixels.
- **Sharpening Kernels:**It enhance the fine details and edges in an image.
- **Edge Detection Kernels:**These kernels are designed to find edges in an image by identifying areas with high intensity gradients.
- **Embossing Kernels:**These kernels hHighlights edges and gives a 3D effect to the image.
- **Gradient Kernels:**These type of kernels are used to calculate intensity gradients in specific directions.
- **Specialized Kernels:**The specialized kernels are used to focus on circular regions or to simulate motion blur along a specific direction.
- **custom Kernels:**Kernels can be designed for specific purposes by customizing weights.
### Convolution

In image processing
**convolution**is a mathematical operation used to apply a filter or kernel to an image. It involves sliding the kernel over the image by performing element-wise multiplication between the kernel and the image pixels under the kernel and then summing the results to get a new pixel value.
In SciPy convolution can be done using the
**scipy.ndimage.convolve()**or**scipy.signal.convolve2d()**functions. These functions allow us to apply a kernel to an image, time series or multi-dimensional data.
The convolution operation between an image
**I**and a kernel**K**is define d as follows −![Convolution Formula](/scipy/images/convolution_formula.jpg)
Where −

- **I**is the input image.
- **K**is the kernel or filter.
- **I'**is the output image or the result of convolution.
- The kernel slides over the image and the sum of element-wise products at each position gives the resulting pixel value.
### Key properties of Convolution

Following are the properties which make convolution a versatile and powerful operation in image and signal processing −

- **Commutativity:**The order of convolution does not matter which is given as**I*K = K*I**.
- **Associativity:**The grouping of convolutions does not matter which can be refered as**I*(K*L) = (I*K)*L**.
- **Distributivity:**Convolution distributes over addition which can represented as**I*(K+L) = (I*K)+(I*L)**.
- **Identity Element:**The identity kernel for example a kernel with a 1 in the center and 0s elsewhere, leaves the image unchanged when convolved.
- **Shift Invariance:**Convolution is unaffected by shifting the image. Shifting the image before or after convolution gives the same result.
- **Linearity:**Convolution is linear so we can scale images or kernels and the result will scale accordingly.
- **Separable:**If a 2D kernel can be separated into two 1D kernels then the convolution can be done more efficiently by applying the 1D kernels in sequence.
## Types of Filters

Filters are crucial tools in image processing for enhancing, transforming or extracting features from images. There are different types of filters designed for a specific effect which depends on the desired outcome for the image processing task. Following are the types of filters −

- Low Pass Filters (Smoothing Filters)
- High Pass Filters
- Morphological Filters

---

## 54. SciPy - Top-Hat Filters

*Source: [https://www.tutorialspoint.com/scipy/scipy_top_hat_filters.htm](https://www.tutorialspoint.com/scipy/scipy_top_hat_filters.htm)*

---

---
[Previous](/scipy/scipy_filtering_edge_detection.htm)[Quiz](/scipy/quiz_on_scipy_top_hat_filters.htm)[Next](/scipy/scipy_morphological_filters.htm)
The
**Top-Hat filter**in SciPy is a morphological operation designed to highlight specific intensity features in an image. It is particularly useful for detecting and enhancing small, localized features that are either brighter or darker than their surroundings. There are two primary types of top-hat filters as follows −
- White Top-Hat Filter
- Black Top-Hat Filter
Now, let's see each type of the
**Top-Hat Filters**in detail −
## White Top-Hat Filter

The
**White Top-Hat Filter**is a morphological operation that extracts small bright features from an image that are smaller than the structuring element.
It is often used to enhance bright details, remove uneven background illumination or highlight specific regions of interest. The White Top-Hat filter is mathematically defined as follows −

```
WhiteTop-HatImage = OriginalImageOpenedImage
```

A morphological operation that consists of erosion followed by dilation. It smooths the image by removing small bright regions that do not fit within the structuring element.

Subtracting the opened image from the original image enhances the bright regions that were removed during the opening process.

### Creating/Implementing White Top-Hat Filter

The structuring element i.e., a kernel defines the size and shape of the features to extract. The bright regions smaller than the structuring element are isolated because the opening operation removes bright regions smaller than the structuring element and subtraction restores these regions while suppressing the larger-scale structures.

In SciPy the
**White Top-Hat Filter**can be implemented by the function**scipy.ndimage.white_tophat()**.
#### Syntax

Following is the syntax of the function
**scipy.ndimage.white_tophat()**which is used to extract small bright features from an image −
```
scipy.ndimage.white_tophat(
   input, 
   size=None, 
   footprint=None, 
   structure=None, 
   output=None, 
   mode='reflect',
   cval=0.0, 
   origin=0
)
```

Following are the parameters of the
**scipy.ndimage.white_tophat()**function −
- **input:**The input image or array i.e., grayscale or binary.
- **size:**This parameter specifies the size of a square structuring element.
- **footprint:**A binary array that defines the shape of the structuring element.
- **structure:**It specifies the exact structuring element and overrides size and footprint.
- **mode:**This determines how image boundaries are handled.
- **cval:**The constant value used when mode='constant'.
- **origin:**This parameter controls the position of the structuring element relative to the current pixel.
### White Top-Hat Filter with Default Structuring Element

Following is the basic example of the function
**scipy.ndimage.white_tophat()**, in which we will apply the white top-hat filter using the default structuring element i.e., a 3x3 square to a sample image (camera) from skimage.
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import white_tophat
from skimage.data import camera

# Load the sample image
image = camera()

# Apply the white top-hat filter with the default structuring element (3x3 square)
result = white_tophat(image, size=3)

# Plot the original and filtered images side by side
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# White Top-Hat Filter Result
plt.subplot(1, 2, 2)
plt.title("White Top-Hat Filter Result")
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```

Here is the output of the function
**scipy.ndimage.white_tophat()**−![White tophat basic example](/scipy/images/white_tophat_basic.jpg)
### White Top-Hat Filter with Varying the Size Parameter

The size parameter in the white top-hat filter controls the size of the structuring element used for the morphological opening operation. By changing this parameter we can adjust the scale of the features that we want to enhance i.e., the size of the small bright features where we want to isolate. Here is the example which varies the size parameter as per requirement −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import white_tophat
from skimage.data import camera

# Load the example image
image = camera()

# Apply white top-hat filter with small structuring element (size=5)
result_small = white_tophat(image, size=5)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("White Top-Hat (size=5)")
plt.imshow(result_small, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
```

Here is the output of the function
**scipy.ndimage.white_tophat()**which varies the size −![White tophat size example](/scipy/images/white_tophat_size.jpg)
## Black Top-Hat Filter

The
**Black Top-Hat Filter**also called as**Bottom Hat Filter**, which is a morphological operation that is used to extract small dark features from an image. It works in the opposite manner of the**White Top-Hat Filter**, in which it highlights small bright regions.
The Black Top-Hat filter helps to isolate small dark regions i.e., dark spots against a brighter background. The mathematical definition for the
**Black Top-Hat Filter**can be given as follows −
```
BlackTop-Hat = ClosedImage  OriginalImage
```

Where −

- **Closing:**The closing operation consists of dilation followed by erosion which smoothens the image by filling small holes in bright regions.
- **Subtracting the original image from the closed image**helps to isolate small dark regions that were enhanced by the closing operation.
In Scipy we have a function namely,
**scipy.ndimage.black_tophat()**to implement the**Black Top-Hat**Operation −
### Syntax

Following is the syntax of the function
**scipy.ndimage.black_tophat()**to perform the Black Top-Hat Operation on the image −
```
scipy.ndimage.black_tophat(
   input, 
   size=None, 
   footprint=None, 
   structure=None, 
   output=None, 
   mode='reflect', 
   cval=0.0, 
   origin=0
)
```

Here are the parameters of
**scipy.ndimage.black_tophat()**function −
- **input:**The input image or array, i.e., grayscale or binary.
- **size:**This parameter specifies the size of a square structuring element.
- **footprint:**A binary array that defines the shape of the structuring element.
- **structure:**It specifies the exact structuring element and overrides 'size' and 'footprint'.
- **output:**The array to store the result of the filter. If not specified then a new array will be created.
- **mode:**This determines how image boundaries are handled and the modes are such as 'reflect', 'constant', 'nearest', etc.
- **cval:**The constant value used when mode='constant' to pad the boundaries.
- **origin:**This parameter controls the position of the structuring element relative to the current pixel.
### Black Top-Hat Filter with Default Structuring Element

Following is the example of the function
**scipy.ndimage.black_tophat()**in which we'll apply the Black Top-Hat filter using the default structuring element i.e., a 3x3 square to the sample image (camera from skimage) −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import black_tophat
from skimage.data import camera

# Load the sample image
image = camera()

# Apply the black top-hat filter with the default structuring element (3x3 square)
result = black_tophat(image, size=3)

# Plot the original and filtered images side by side
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Black Top-Hat Filter Result
plt.subplot(1, 2, 2)
plt.title("Black Top-Hat Filter Result")
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```

Following is the output of the basic example implemented using
**scipy.ndimage.black_tophat()**function −![Black tophat basic example](/scipy/images/black_top_hat_basic.jpg)
### Black Top-Hat Filter with a Custom Structuring Element

In this example we'll use a disk-shaped structuring element with a radius of 10 pixels to extract dark spots using a custom shape −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import black_tophat
from skimage.morphology import disk
from skimage.data import camera

# Load the sample image
image = camera()

# Define a disk-shaped structuring element with radius 10
structuring_element = disk(10)

# Apply the black top-hat filter with the custom structuring element
result = black_tophat(image, structure=structuring_element)

# Plot the original and filtered images side by side
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Black Top-Hat Filter Result
plt.subplot(1, 2, 2)
plt.title("Black Top-Hat Filter Result")
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```

Following is the output of the example implemented using
**scipy.ndimage.black_tophat()**function with disk structuring element −![Black tophat disk example](/scipy/images/black_top_hat_disk.jpg)

---

## 55. SciPy - Morphological Filters

*Source: [https://www.tutorialspoint.com/scipy/scipy_morphological_filters.htm](https://www.tutorialspoint.com/scipy/scipy_morphological_filters.htm)*

---

---
[Previous](/scipy/scipy_top_hat_filters.htm)[Quiz](/scipy/quiz_on_scipy_morphological_filters.htm)[Next](/scipy/scipy_low_pass_filters.htm)
## Morphological Filters
**Morphological filters**in image processing are a set of operations that process images based on their shapes and structures. These operations are particularly useful in the context of binary or grayscale images where the primary focus is on the shapes and structures of objects in the image such as boundaries, holes and small objects.
SciPy provides morphological operations through the
**scipy.ndimage**module which includes several functions to perform basic morphological operations. These operations are typically applied to binary or grayscale images.
## Common Morphological Operations
**Morphological operations**manipulate an image based on its shapes and structures which is typically for binary images or grayscale images. These operations rely on a structuring element i.e., a small kernel that defines the operation's effect on the image. Here are the most common morphological operations −
- **Dilation**− This operation expands the boundaries of bright regions (foreground objects).
- **Erosion**− It shrinks the boundaries of bright regions.
- **Opening**− This is the erosion followed by dilation which removes small objects i.e., noise.
- **Closing**− This is dilation followed by erosion which fills small holes.
- **Morphological Gradient**− This performs the difference between dilation and erosion by highlighting the edges.
## Structuring Element

In
**Morphological Filtering**a structuring element or kernel which is a small matrix or array that defines the neighborhood over which a morphological operation is applied. The choice of the structuring element significantly affects the results of morphological operations such as dilation, erosion, opening and closing.
In
**scipy.ndimage**structuring elements can be customized and are often defined using functions such as**generate_binary_structure()**or manually created as NumPy arrays.
### Syntax

Following is the syntax for the function
**generate_binary_structure()**to create a structuring element −
```
scipy.ndimage.generate_binary_structure(rank, connectivity)
```

Here are the parameters of the function
**scipy.ndimage.generate_binary_structure()**−
- **rank(int)**− The dimensionality of the structuring element such as 2 for 2D, 3 for 3D, etc.
- **connectivity(int)**− This parameter determines the type of connectivity or neighborhood.**1**includes only the nearest neighbors in each dimension i.e., cross-shaped neighborhood in 2D and**2**includes diagonal neighbors i.e., full square or cube neighborhood in 2D or 3D.
### Return Value

This function returns a NumPy array of shape (3,) * rank containing True and False by representing the structuring element.

### Example - 2D Structuring Element with Connectivity 1

A 2D structuring element with connectivity 1 is a small matrix that defines a cross-shaped neighborhood. It includes the center pixel and the immediate horizontal and vertical neighbors but excludes the diagonal neighbors.

Here is the example which shows how we can create such a structuring element using
**scipy.ndimage.generate_binary_structure()**with rank=2 (for 2D) and connectivity=1 −
```
from scipy.ndimage import generate_binary_structure

# Create a 2D structuring element with connectivity=1
structure_2d = generate_binary_structure(rank=2, connectivity=1)
print("2D Structuring Element (Connectivity=1):\n", structure_2d)
```

#### Output

Following is the output of the 2D sturcturing element with connectivity 1 −

```
2D Structuring Element (Connectivity=1):
 [[False  True False]
 [ True  True  True]
 [False  True False]]
```

## Dilation with generate_binary_structure() Function

As we discussed before the structuring element created by generate_binary_structure() is often used in binary morphological operations such as dilation, erosion, opening and closing. The choice of connectivity affects how neighboring pixels are considered during these operations.

### Example

Following is the example of the Structuring element which is used in the Dilation −

```
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure

# Create a binary image
image = np.zeros((10, 10), dtype=int)
image[4:6, 4:6] = 1  # Small square in the center

# Generate a structuring element with connectivity=1
structure = generate_binary_structure(rank=2, connectivity=1)

# Apply dilation
dilated_image = binary_dilation(image, structure=structure)

# Plot the results
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Dilated image
plt.subplot(1, 2, 2)
plt.title("Dilated Image (Connectivity=1)")
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')

plt.show()
```

#### Output

Following is the output of the 2D structuring element with connectivity 1 −
![Structuring Element](/scipy/images/structuring_element.jpg)
## Enlarging of Structuring Element
**Enlarging a structuring element**in morphological operations increases the area of influence by allowing the operation to act on a broader region in the image. This is useful in tasks like filling larger gaps, expanding shapes or removing noise from larger regions.
In SciPy the function
**scipy.ndimage.iterate_structure()**is used to enlarge structuring elements. It**dilates**the original structuring element by a specified number of iterations effectively by increasing its size.
### Syntax

Following is the syntax of the function
**scipy.ndimage.iterate_structure()**which is used to enlarge the structuring element −
```
scipy.ndimage.iterate_structure(structure, iterations)
```

Following are the parameters of the function
**scipy.ndimage.iterate_structure()**−
- **structure(ndarray)**− The input structuring element which is a binary (boolean) array. It defines the shape that will be used for morphological operations.
- **iterations(int)**−  The number of times the structuring element should be dilated and increasing the number of iterations then it will enlarge the structuring element.
This function returns the dilated structuring element after iterations dilations.

### Example

Following is the example which shows how to use the
**scipy.ndimage.iterate_structure()**function to the enlarge the structuring element generated using the**scipy.ndimage.generate_binary_structure()**function −
```
from scipy.ndimage import generate_binary_structure, iterate_structure
import matplotlib.pyplot as plt

# Step 1: Create a 2D cross-shaped structuring element
struct_2d = generate_binary_structure(rank=2, connectivity=1)
print("Original Structuring Element:\n", struct_2d)

# Step 2: Enlarge the structuring element by 2 iterations
enlarged_struct = iterate_structure(struct_2d, iterations=2)
print("Enlarged Structuring Element (2 iterations):\n", enlarged_struct)

# Step 3: Visualize the structuring elements
plt.figure(figsize=(10, 5))

# Original structuring element
plt.subplot(1, 2, 1)
plt.title("Original Structuring Element")
plt.imshow(struct_2d, cmap='gray')
plt.axis('off')

# Enlarged structuring element
plt.subplot(1, 2, 2)
plt.title("Enlarged Structuring Element")
plt.imshow(enlarged_struct, cmap='gray')
plt.axis('off')

plt.show()
```

#### Output

Following is the output of the enlarged structuring element with the help of
**scipy.ndimage.iterate_structure()**function −
```
Original Structuring Element:
 [[False  True False]
 [ True  True  True]
 [False  True False]]
Enlarged Structuring Element (2 iterations):
 [[False False  True False False]
 [False  True  True  True False]
 [ True  True  True  True  True]
 [False  True  True  True False]
 [False False  True False False]]
```
![Enlarging Structuring Element](/scipy/images/enlarge_structuring.jpg)
## Applications of Morphological Filters
**Morphological filters**are widely used in image processing for analyzing and modifying the geometric structure of objects in an image. They are particularly effective in tasks involving shape, structure and segmentation. Here are the few applications of the Morphological Filters −
- **Noise Removal**− By using opening and closing operations, small noise elements can be removed or small holes can be filled.
- **Edge Detection**− The morphological gradient can be used to highlight edges in the image.
- **Shape Analysis**− These operations are often used in object recognition, segmentation and feature extraction.
- **Pre-processing**− In many computer vision tasks the morphological operations helps to prepare the image for further analysis.
## Implementing Morphological Filters in SciPy
**Morphological filters**are powerful tools in image processing which are used for shape analysis, noise removal and enhancing image structures. In SciPy library, these operations can be performed using the**scipy.ndimage**module.
Following are the functions available in
**scipy.ndimage**module to perform the Morphological Operations in image processing −S.No.Function & Description1**scipy.ndimage.binary_erosion()**
Perform erosion on a binary image (shrinking).2**scipy.ndimage.binary_dilation()**
Perform dilation on a binary image (expanding).3**scipy.ndimage.binary_opening()**
Perform binary opening i.e., erosion followed by dilation.4**scipy.ndimage.binary_closing()**
Perform binary closing i.e., dilation followed by erosion.5**scipy.ndimage.grey_erosion()**
Shrinks bright regions in the image.6**scipy.ndimage.grey_dilation()**
Expands bright regions in the image.7**scipy.ndimage.grey_opening()**
Perform grayscale opening, removing small bright spots.8**scipy.ndimage.grey_closing()**
Perform grayscale closing, filling small dark holes.9**scipy.ndimage.label()**
Label connected components in a binary image or multi-dimensional array.10**scipy.ndimage.find_objects()**
Return slice objects corresponding to the labeled regions in an array.

---

## 56. SciPy - Low Pass Filters

*Source: [https://www.tutorialspoint.com/scipy/scipy_low_pass_filters.htm](https://www.tutorialspoint.com/scipy/scipy_low_pass_filters.htm)*

---

---
[Previous](/scipy/scipy_morphological_filters.htm)[Quiz](/scipy/quiz_on_scipy_low_pass_filters.htm)[Next](/scipy/scipy_high_pass_filters.htm)
## SciPy - Low Pass Filters
**Low-pass filters**are also called as**Smoothing filters**which are used in image processing to smooth or blur an image by reducing high-frequency components such as noise or rapid intensity changes. These filters preserve low-frequency information i.e., smooth variations while attenuating high-frequency details like edges or noise.
Mathematically we can give a low-pass filter which modifies the image
**I(x,y)**by convolving it with a kernel**K**as follows −![Low pass filter Formula](/scipy/images/low_pass_filter.jpg)
Following are the different types of Low pass filters available in scipy −

## Box Filter in SciPy

A
**Box filter**is also called as**Uniform filter**which computes the average intensity of all pixels in a neighborhood defined by a kernel size. It is simple and fast but may blur edges. In scipy we have the function**scipy.ndimage.uniform_filter()**to apply the box filter to an image. The**size**parameter defines the neighborhood size and large sizes result in more blur but reduce image details.
Following is an example which applies the uniform filter on an image with the help of the function
**scipy.ndimage.uniform_filter()**−
```
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from skimage import data

# Load an example image
image = data.camera()

# Apply a box filter (uniform filter) with a size of 5x5
filtered_image = uniform_filter(image, size=5)

# Display original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Box Filter (5x5)")
plt.imshow(filtered_image, cmap="gray")
plt.show()
```

Below is the output of the box filter −
![Box Filter Example](/scipy/images/box_filter_example.jpg)
## Gaussian Filter in SciPy

The
**Gaussian filter**is one of the most popular low-pass filters. It uses a Gaussian kernel where pixels closer to the center of the kernel have higher weights. We have the function**scipy.ndimage.guassian_filter()**to use the Gaussian filter on an image.
The Gaussian kernel is defined as follows −
![Gaussian Formula](/scipy/images/gaussian_formula.jpg)
Where,  is the Standard deviation of the Gaussian which controls the amount of smoothing.

is used to control the degree of smoothing where larger values result in greater blur and Gaussian smoothing is ideal for removing noise without severely blurring edges.

Below is an example which shows how to use the gaussian filter on an image with the help of the function
**scipy.ndimage.guassian_filter()**−
```
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage import data

# Load an example image
image = data.camera()

# Apply Gaussian filter with standard deviation sigma=2
gaussian_blurred = gaussian_filter(image, sigma=2)

# Display original and blurred images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Gaussian Filter (sigma=2)")
plt.imshow(gaussian_blurred, cmap="gray")
plt.show()
```

Here is the output of the Gaussian filter −
![Guassian Filter Example](/scipy/images/guassian_filter_example.jpg)
## Mean Filter in SciPy

The
**mean filter**replaces each pixel with the average of its neighbors. While effective for noise reduction it does not preserve edges well.
The kernel averages the pixel intensities which lead to a simple smoothing effect and larger kernels increase the smoothing effect but at the cost of edge detail.

In this example we are applying the mean filter to the given input image by using the function
**scipy.ndimage.convolve()**−
```
from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load an example image
image = data.camera()

# Define a mean filter kernel (3x3)
mean_kernel = np.ones((3, 3)) / 9

# Apply mean filter using convolution
mean_filtered = convolve(image, mean_kernel)

# Display original and mean-filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Mean Filter (3x3)")
plt.imshow(mean_filtered, cmap="gray")
plt.show()
```

Following is the output of the Mean filter −
![Mean Filter Example](/scipy/images/mean_filter_example.jpg)
Here is an example which compares all the three different types of Low pass filters that we can use on the given input image −

```
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load an example image
image = data.camera()

# Define a mean filter kernel (3x3)
mean_kernel = np.ones((3, 3)) / 9

# Apply different filters and compare
gaussian = ndimage.gaussian_filter(image, sigma=2)
mean = ndimage.convolve(image, mean_kernel)
box = ndimage.uniform_filter(image, size=5)

# Display results
plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(2, 2, 2)
plt.title("Gaussian Filter")
plt.imshow(gaussian, cmap="gray")

plt.subplot(2, 2, 3)
plt.title("Mean Filter")
plt.imshow(mean, cmap="gray")

plt.subplot(2, 2, 4)
plt.title("Box Filter")
plt.imshow(box, cmap="gray")

plt.tight_layout()
plt.show()
```

Following is the output of the above code for comparing the different low pass filters −
![Comparision of Low pass filters](/scipy/images/comparision_lowpass.jpg)

---

## 57. SciPy - High Pass Filters

*Source: [https://www.tutorialspoint.com/scipy/scipy_high_pass_filters.htm](https://www.tutorialspoint.com/scipy/scipy_high_pass_filters.htm)*

---

---
[Previous](/scipy/scipy_low_pass_filters.htm)[Quiz](/scipy/quiz_on_scipy_high_pass_filters.htm)[Next](/scipy/scipy_bilateral_filter.htm)
## High Pass Filters

A
**High-pass Filter**is used in image processing to emphasize the high-frequency components of an image such as edges, fine details and rapid intensity changes while suppressing the low-frequency components like smooth areas or gradual intensity variations. This makes high-pass filters especially useful for tasks like edge detection, image sharpening and feature enhancement.
## Key concepts of High pass Filters

Following are the key concepts of High Pass Filters in SciPy −

- **High Frequencies:**These represent sharp intensity transitions such as edges, noise or fine textures.
- **Low Frequencies:**These represent smooth transitions such as uniform regions or gradual shading.
- **Purpose:**The purpose of these filters is to retain high-frequency details while attenuating low-frequency background information.
## Types of High-Pass Filters

In
**High Pass Filters**, we have different types and each filter is with its unique characteristics suited for specific tasks. They are −
- Gradient-Based Filters
- Sobel Filter
- Prewitt Filter
- Roberts Cross Filter
- Charr Filter
- Central Difference Filter
Let's go through the each filter in detail −

## Spatial Domain High-Pass Filters

In image processing the
**spatial domain high-pass filters**work by directly modifying pixel intensities to emphasize high-frequency components such as edges, fine details and rapid intensity transitions. These filters use convolution with specific kernels or operations to achieve the desired effect.
In
**spatial domain high-pass filters**we have different types of filters which are as follows −
## Gradient-Based Filters
**Gradient-based filters**are edge-detection techniques that emphasize intensity changes by calculating gradients i.e., first-order derivatives in an image. They are widely used in image processing for tasks like feature extraction, edge detection and object recognition.
The gradient of an image at a point is a vector representing the rate and direction of intensity changes where we can represent as magnitude (G) and Direction ().

- **Magnitude(G):**This indicates the strength of intensity change i.e., edge strength.![Magnitude Gradient Formula](/scipy/images/magnitude_gradient.jpg)
- **Direction():**This represents the orientation of the edge.![Direction Gradient Formula](/scipy/images/direction_gradient.jpg)
Where G
and Gare gradients in the horizontal and vertical directions, respectively.
In Gradient Filters, again we have different types which can be given as follows −

## Sobel Filter

The
**Sobel filter**is a gradient-based edge detection technique used in image processing to detect edges and highlight regions with high spatial frequency. It calculates the gradient of image intensity in the horizontal**G**and vertical**G**directions, which combines the results to produce an edge map.
In scipy, we have the
**scipy.ndimage.sobel()**function to apply the sobel filter on the given image.
### Syntax

Following is the syntax of using the
**scipy.ndimage.sobel()**function to apply sobel filter on the image −
```
scipy.ndimage.sobel(input, axis=-1, output=None, mode='reflect', cval=0.0)
```

Following are the parameters of the scipy.ndimage.sobel() function −

- **input (array_like):**The input image or array on which the Sobel filter is applied and typically this is a 2d grayscale image.
- **axis (int, optional):**The axis along which to compute the derivative, when 1 applies the Sobel filter in the horizontal direction(G) and 0 applies the Sobel filter in the vertical direction(G)
- **output (array or dtype, optional):**This specifies where the output will be stored and if**None**then a new array is created and returned.
- **mode (str, optional)::**This defines how the input array is extended beyond its boundaries. The modes are such as reflect, constant, nearest, mirror and wrap.
- **cval (scalar, optional):**The constant value to use when mode='constant'. The default value is 0.0.
Following is the example of using the
**scipy.ndimage.sobel()**function on the given image −
```
from scipy.ndimage import sobel
from skimage import data
import matplotlib.pyplot as plt

# Load a sample grayscale image
image = data.camera()

# Apply Sobel filter
sobel_x = sobel(image, axis=0)  # Horizontal edges
sobel_y = sobel(image, axis=1)  # Vertical edges
sobel_combined = sobel_x + sobel_y

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.title("Original")
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2), plt.title("Sobel X (Horizontal Edges)")
plt.imshow(sobel_x, cmap='gray')
plt.subplot(1, 3, 3), plt.title("Sobel Y (Vertical Edges)")
plt.imshow(sobel_y, cmap='gray')
plt.tight_layout()
plt.show()
```

Following is the output of the image on which the sobel filter is applied−
![Sobel Filter](/scipy/images/sobel_output.jpg)
## Prewitt Filter

The
**Prewitt filter**is an edge detection operator used in image processing to highlight edges in an image by computing the gradient of pixel intensities. It is similar to the Sobel filter but uses different coefficients for its convolution masks. The**Prewitt filter**emphasizes the rate of intensity change in the horizontal and vertical directions.
This filter typically involves two convolution kernels (masks) used to calculate the gradients in the horizontal and vertical directions.

In scipy
**scipy.ndimage**module does not have a built-in function for the**Prewitt filter**like the Sobel filter it is easy to implement using the**convolve()**function by directly applying the Prewitt kernels.
Following is the syntax of the function
**scipy.ndimage.convolve()**used to apply the**Prewitt filter**−
```
scipy.ndimage.convolve(
   input, 
   weights, 
   output=None, 
   mode='reflect', 
   cval=0.0, 
   origin=0
)
```

Here are the parameters of the
**scipy.ndimage.convolve()**function −
- **input (array_like):**The input image or array on which the convolution operation is applied. Typically this is a 2D grayscale image or a multidimensional array.
- **weights (array_like):**The filter or kernel to apply during the convolution. This should have the same number of dimensions as the input or be broadcastable.
- **output (array or dtype, optional):**This parameter specifies where the output will be stored. If**None**then a new array is created and returned.
- **mode (str, optional):**This parameter defines how the input array is extended beyond its boundaries. The available modes are such as**'reflect'**,**'constant'**,**'nearest'**,**'mirror'**and**'wrap'**. The default value is**'reflect'**.
- **cval (scalar, optional):**The constant value used for padding when**mode='constant'**. The default value is**0.0**.
- **origin (int or tuple of ints, optional):**This parameter controls the placement of the kernel relative to the input elements. Default value is**0**which centers the kernel.
Following is an example which uses the
**scipy.ndimage.convolve()**function to apply the**Prewitt filter**to perform the edge detection in images both in horizontal and vertical edges based on intensity gradients −
```
import numpy as np
from scipy.ndimage import convolve
from skimage import data
import matplotlib.pyplot as plt

# Load a sample grayscale image
image = data.camera()

# Define Prewitt kernels for detecting vertical and horizontal edges
prewitt_vertical = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])

prewitt_horizontal = np.array([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]])

# Apply the Prewitt filter in the vertical direction (horizontal edges)
edges_vertical = convolve(image, prewitt_vertical)

# Apply the Prewitt filter in the horizontal direction (vertical edges)
edges_horizontal = convolve(image, prewitt_horizontal)

# Combine the results by taking the magnitude of the gradient
edges_magnitude = np.sqrt(edges_vertical**2 + edges_horizontal**2)

# Display the results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Vertical edges (horizontal gradient)
plt.subplot(1, 3, 2)
plt.title("Prewitt - Vertical Edges")
plt.imshow(edges_vertical, cmap='gray')
plt.axis('off')

# Magnitude of edges
plt.subplot(1, 3, 3)
plt.title("Prewitt - Edge Magnitude")
plt.imshow(edges_magnitude, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```

Following is the output of the
**convolve()**function which is used to apply the**prewitt filter**−![prewitt Filter example](/scipy/images/prewitt_example.jpg)
## Roberts Cross Filter

The
**Roberts Cross filter**is an edge detection operator used in image processing to highlight edges in an image by calculating the diagonal gradients of pixel intensities. It works by using small 2x2 convolution kernels by making it computationally efficient but sensitive to noise. The**Roberts Cross filter**emphasizes intensity changes along diagonals for detecting edges effectively in those directions.
This filter uses two convolution kernels (masks) to compute gradients in the diagonal directions.

In
**scipy.ndimage**module we do not have a built-in function for the**Roberts Cross filter**. However it can be implemented using the**convolve()**function by directly applying the Roberts Cross kernels.
Following is an example which uses the
**scipy.ndimage.convolve()**function to apply the**Roberts Cross filter**to detect edges in an image −
```
import numpy as np  
from scipy.ndimage import convolve  
from skimage import data  
import matplotlib.pyplot as plt  

# Load a sample grayscale image  
image = data.camera()  

# Define Roberts Cross kernels for diagonal gradients  
roberts_cross_vertical = np.array([[1, 0],  
                                    [0, -1]])  

roberts_cross_horizontal = np.array([[0, 1],  
                                      [-1, 0]])  

# Apply the Roberts Cross filter for diagonal gradients  
edges_vertical = convolve(image, roberts_cross_vertical)  
edges_horizontal = convolve(image, roberts_cross_horizontal)  

# Combine the results by taking the magnitude of the gradient  
edges_magnitude = np.sqrt(edges_vertical**2 + edges_horizontal**2)  

# Display the results  
plt.figure(figsize=(15, 5))  

# Original Image  
plt.subplot(1, 3, 1)  
plt.title("Original Image")  
plt.imshow(image, cmap='gray')  
plt.axis('off')  

# Vertical edges (horizontal gradient)  
plt.subplot(1, 3, 2)  
plt.title("Roberts Cross - Vertical Edges")  
plt.imshow(edges_vertical, cmap='gray')  
plt.axis('off')  

# Magnitude of edges  
plt.subplot(1, 3, 3)  
plt.title("Roberts Cross - Edge Magnitude")  
plt.imshow(edges_magnitude, cmap='gray')  
plt.axis('off')  

plt.tight_layout()  
plt.show()
```

Following is the output of the
**convolve()**function which is used to apply the**Roberts Cross filter**−![Roberts Cross Filter example](/scipy/images/roberts_cross_example.jpg)
## Scharr Filter

The
**Scharr filter**is an edge detection operator used in image processing which is a variant of the**Sobel filter**designed to achieve better rotational symmetry and accuracy. It computes the gradient of pixel intensities in horizontal and vertical directions to highlight edges in an image. The**Scharr filter**is especially useful for applications requiring precise edge detection.
In SciPy we does not have any function to apply the scharr filter but we can implement it by using the
**scipy.ndimage.convolve(()**function.
Following is an example that uses the
**scipy.ndimage.convolve()**function to apply the Scharr filter and detect edges in an image −
```
import numpy as np 
from scipy.ndimage import convolve 
from skimage import data 
import matplotlib.pyplot as plt
#Load a sample grayscale image
image = data.camera()

#Define Scharr kernels for detecting vertical and horizontal edges
scharr_vertical = np.array([[-3, 0, 3],
[-10, 0, 10],
[-3, 0, 3]])

scharr_horizontal = np.array([[-3, -10, -3],
[0, 0, 0],
[3, 10, 3]])

#Apply the Scharr filter in the vertical direction (horizontal edges)
edges_vertical = convolve(image, scharr_vertical)

#Apply the Scharr filter in the horizontal direction (vertical edges)
edges_horizontal = convolve(image, scharr_horizontal)

#Combine the results by taking the magnitude of the gradient
edges_magnitude = np.sqrt(edges_vertical + edges_horizontal)

#Display the results
plt.figure(figsize=(15, 5))

#Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

#Vertical edges (horizontal gradient)
plt.subplot(1, 3, 2)
plt.title("Scharr - Vertical Edges")
plt.imshow(edges_vertical, cmap='gray')
plt.axis('off')

#Magnitude of edges
plt.subplot(1, 3, 3)
plt.title("Scharr - Edge Magnitude")
plt.imshow(edges_magnitude, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```

Following is the output of the
**convolve()**function used to apply the**Scharr filter**−![Scharr Filter example](/scipy/images/scharr_example_output.jpg)
## Central Difference Filter

The
**Central Difference filter**is a gradient operator used in image processing to approximate the derivative of pixel intensities. It computes the difference between neighboring pixels symmetrically which helps detect changes in intensity by making it useful for edge detection tasks. This filter is simple and often used in applications where a basic gradient approximation is sufficient.
This filter uses a small convolution kernel to calculate gradients in the horizontal and vertical directions.

In
**scipy.ndimage**the central difference filter can be implemented using the**convolve()**function by applying the appropriate central difference kernels.
Following is an example which uses the
**scipy.ndimage.convolve()**function to apply the**Central Difference filter**to detect edges in an image −
```
import numpy as np  
from scipy.ndimage import convolve  
from skimage import data  
import matplotlib.pyplot as plt  

# Load a sample grayscale image  
image = data.camera()  

# Define Central Difference kernels for horizontal and vertical gradients  
central_diff_horizontal = np.array([[-1, 0, 1]])  
central_diff_vertical = np.array([[-1], [0], [1]])  

# Apply the Central Difference filter in the horizontal direction  
edges_horizontal = convolve(image, central_diff_horizontal)  

# Apply the Central Difference filter in the vertical direction  
edges_vertical = convolve(image, central_diff_vertical)  

# Combine the results by taking the magnitude of the gradient  
edges_magnitude = np.sqrt(edges_horizontal**2 + edges_vertical**2)  

# Display the results  
plt.figure(figsize=(15, 5))  

# Original Image  
plt.subplot(1, 3, 1)  
plt.title("Original Image")  
plt.imshow(image, cmap='gray')  
plt.axis('off')  

# Horizontal Edges  
plt.subplot(1, 3, 2)  
plt.title("Central Difference - Horizontal Edges")  
plt.imshow(edges_horizontal, cmap='gray')  
plt.axis('off')  

# Magnitude of Edges  
plt.subplot(1, 3, 3)  
plt.title("Central Difference - Edge Magnitude")  
plt.imshow(edges_magnitude, cmap='gray')  
plt.axis('off')  

plt.tight_layout()  
plt.show()
```

Following is the output of the
**convolve()**function which is used to apply the**Central Difference filter**−![Central Difference Filter example](/scipy/images/central_difference_example.jpg)

---

## 58. SciPy - Bilateral Filter

*Source: [https://www.tutorialspoint.com/scipy/scipy_bilateral_filter.htm](https://www.tutorialspoint.com/scipy/scipy_bilateral_filter.htm)*

---

---
[Previous](/scipy/scipy_high_pass_filters.htm)[Quiz](/scipy/quiz_on_scipy_bilateral_filter.htm)[Next](/scipy/scipy_median_filter.htm)
A
**Bilateral Filter**is a nonlinear, edge-preserving and noise-reducing smoothing filter. It smooths images while preserving edges by weighting pixel values with a spatial Gaussian filter and an intensity Gaussian filter.
SciPy does not have a built-in bilateral filter but it can be implemented using the
**scipy.ndimage**module or other libraries like OpenCV.
## Key Features of Bilateral Filter

Following are the key features of the Bilateral Filter which is used in SciPy Image Processing −

- **Edge Preservation:**The**Bilateral filter**is designed to preserve edges while reducing noise. It smooths regions that are spatially close and have similar intensity values while keeping sharp edges intact. This makes it especially useful for tasks where edge information is critical such as in image denoising and pre-processing for segmentation.
- **Non-Linear Filtering:**Unlike linear filters such as Gaussian filters, the bilateral filter is non-linear because it combines both the spatial distance and intensity difference between neighboring pixels to compute a weighted average. This non-linearity helps to preserve the edges of objects in the image.
- **Dual Gaussian Filters:**The bilateral filter uses two Gaussian functions namely,**Spatial Gaussian**which determines the spatial distance i.e., the Euclidean distance between pixels to decide how much influence neighboring pixels have and**Intensity Guassian**which considers the intensity i.e., color or grayscale difference between neighboring pixels to determine how much influence a pixel will have based on its similarity to the center pixel.
- **Edge-Aware Smoothing:**This filter performs smoothing by adjusting the weight of each pixel based not only on its spatial distance but also its intensity difference from the target pixel. This makes it ideal for applications where we want to reduce noise while preserving the edges as the filter will not smooth across edges with high intensity differences.
- **Parameter Sensitivity:**There are two methods namely,**sigma spatial(_spatial)**which is used to how much spatial distance influences the filter. Larger values means that pixels farther from the center pixel will have a higher weight and**sigma Intensity(_Intensity)**which controls how much the intensity difference influences the filter. Larger values reduce the edge-preserving effect and increase smoothing even across intensity differences.
- **Local Adaptive Filtering:**The Bilateral filter adapts to the local features of the image. It adjusts the amount of filtering applied based on both the pixels spatial location and its intensity similarity to neighboring pixels. This leads to a more context-sensitive result compared to traditional linear filters.
- **Computational Complexity:**The Bilateral filter is computationally expensive because it requires the calculation of pixel similarity for each pixel in a neighborhood by making it slower than linear filters. The complexity increases with the size of the filter kernel and the image.
- **Noise Reduction:**Bilateral filters are especially effective at reducing various types of noise such as Gaussian noise or salt-and-pepper noise without blurring the edges of objects in the image.
## Advantages of Bilateral Filter

Here are the advantages of the Scipy Bilateral Filter used in Image processing −

- **Edge Preservation:**The traditional filters that smooth across edges where the bilateral filter keeps edges sharp.
- **Noise Removal:**This filter is effective in removing various types of noise especially in high-resolution images.
- **Local Adaptability:**The filter adapts to local image features by considering both spatial proximity and intensity similarity.
## Disadvantages of Bilateral Filter

Bilateral Filter have the advantages as well as the disadvantages which can be defined as follows −

- **Computationally Intensive:**In the Bilateral Filter the need of computing the intensity and spatial weights for each pixel makes the bilateral filter slower than other traditional smoothing filters, especially for large images.
- **Parameter Tuning:**In this filter the parameter tuning for choosing appropriate values for the spatial and intensity sigma values can be tricky and requires experimentation for different images.
## Implementing a Bilateral Filter in SciPy

Implementing a bilateral filter in SciPy requires combining spatial and intensity domain Gaussian weighting to smooth images while preserving edges.

SciPy itself does not have a direct bilateral filter implementation but we can create one using its utilities such as
**scipy.ndimage.gaussian_filter()**and NumPy for computations.
### Syntax

Following is the syntax of the
**scipy.ndimage.gaussian_filter()**function which can be used to apply the Bilateral filter in scipy −
```
scipy.ndimage.gaussian_filter(
   input, 
   sigma, 
   order=0, 
   output=None, 
   mode='reflect', 
   cval=0.0, 
   truncate=4.0
)
```

Here are the parameters of the function
**scipy.ndimage.gaussian_filter()**−
- **input:**The input array such as an image as a NumPy array.
- **sigma:**Standard deviation for the Gaussian kernel. It can be a single number or a sequence for each axis.
- **order:**This is the derivative order where**0**for a regular Gaussian blur and higher values compute derivatives.
- **output(optional):**This is the output array to store the result.
- **mode:**This parameter determines how the array borders are handled and the modes are such as reflect, constant, nearest, mirror and wrap.
- **cval:**This is the constant value to fill edges if mode='constant'.
- **truncate:**This parameter is used to truncate the filter at this many standard deviations.
### Example

Following is the example of implementing the
**Bilateral filter**using the**scipy.ndimage.guassian_filter()**function of scipy −
```
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import data
import matplotlib.pyplot as plt
 
# load an image
image = data.camera()
sigma_s = 5
sigma_r = 20
def bilateral_filter(image, sigma_s, sigma_r):
   # Ensure the image is a floating-point array
   image = image.astype(np.float64)
   
   # Create a grid of spatial coordinates
   radius = int(3 * sigma_s)  # Typically, 3*sigma_s is used as the filter radius
   y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
   
   # Spatial Gaussian weights
   spatial_gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))
   
   # Initialize the output image
   output = np.zeros_like(image)
   normalizer = np.zeros_like(image)
   
   # Iterate over each pixel
   for i in range(radius, image.shape[0] - radius):
      for j in range(radius, image.shape[1] - radius):
         # Extract the local patch
         patch = image[i-radius:i+radius+1, j-radius:j+radius+1]
         
         # Compute the intensity Gaussian weights
         intensity_gaussian = np.exp(-((patch - image[i, j])**2) / (2 * sigma_r**2))
         
         # Combine the spatial and intensity weights
         weights = spatial_gaussian * intensity_gaussian
         
         # Normalize weights
         weights /= weights.sum()
         
         # Compute the filtered pixel value
         output[i, j] = np.sum(patch * weights)
         normalizer[i, j] = weights.sum()
   
   return output
    
bilateral_image  = bilateral_filter(image, sigma_s, sigma_r)
# Display original and Filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Bilateral Filter (sigma_s=2)")
plt.imshow(bilateral_image)
plt.show()
```

Here is the output of the Bilateral filter with the help of
**scipy.ndimage.guassian_filter()**function −![Bilateral Filter using scipy](/scipy/images/bilateral_scipy.jpg)
## Implementing Bilateral Filter using OpenCV
**OpenCV**provides a highly optimized implementation of bilateral filtering through the**cv2.bilateralFilter()**function. This function is much faster than a custom implementation and is well-suited for real-time applications.
Here is the example of implementing the
**Bilateral Filtering**through the**cv2.bilateralFilter()**function in OpenCV −
```
import cv2
import matplotlib.pyplot as plt

# Load a color image
image = cv2.imread("\images\images.jpeg") 
# Check if the image was loaded correctly
if image is None:
    print("Error: Unable to load image. Check the file path.")
else:
   # Convert from BGR to RGB (OpenCV loads images in BGR by default)
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
   # Apply bilateral filter
   filtered_image = cv2.bilateralFilter(image, d=15, sigmaColor=80, sigmaSpace=80)
   
   # Display the original and filtered images
   plt.figure(figsize=(12, 6))
   plt.subplot(1, 2, 1)
   plt.title("Original Image")
   plt.imshow(image)
   plt.axis('off')
   
   plt.subplot(1, 2, 2)
   plt.title("Bilateral Filtered Image")
   plt.imshow(filtered_image)
   plt.axis('off')
   
   plt.show()
```

Following is the output of the Bilateral Function through OpenCV −
![Bilateral Filter using OpenCV](/scipy/images/bilateral_opencv.jpg)

---

## 59. SciPy - Median Filter

*Source: [https://www.tutorialspoint.com/scipy/scipy_median_filter.htm](https://www.tutorialspoint.com/scipy/scipy_median_filter.htm)*

---

---
[Previous](/scipy/scipy_bilateral_filter.htm)[Quiz](/scipy/quiz_on_scipy_median_filter.htm)[Next](/scipy/scipy_non_linear_filters_image_processing.htm)
## Median Filter in SciPy

The
**Median Filter**in SciPy is a non-linear image processing technique used to remove noise especially salt-and-pepper noise while preserving edges. It works by replacing each pixel in an image with the median of the values in its surrounding neighborhood which can be a square or rectangular region.
This filter is effective in smoothing images without blurring sharp edges as it is less sensitive to outliers compared to linear filters. The neighborhood size is adjustable  such as 3x3, 5x5 and it is commonly used in preprocessing stages of image analysis for denoising.

The
**scipy.ndimage**module of scipy library have a function namely,**median_filter()**to apply the median filter on the given images to remove salt-and-pepper noise by preserving the details of the edges.
### Syntax

Following is the syntax of the
**scipy.ndimage.median_filter()**function to apply the median filter −
```
scipy.ndimage.median_filter(input, size, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
```

Following are the parameters of the
**scipy.ndimage.median_filter**−
- **input**− The input image or array to which the filter will be applied.
- **size**− The size of the neighborhood i.e., a tuple or scalar. If scalar then a square neighborhood is used ie., 3 means a 3x3 neighborhood.
- **footprint(optional)**− A binary array that defines the neighborhood shape.
- **output(optional)**− The array to store the result.
- **mode**− The mode used to handle borders such as 'reflect', 'constant', 'nearest', etc.
- **cval(optional)**− The constant value used when mode='constant'
- **origin**− The offset for the neighborhood with default value as 0.
## How Median Filter Works?

Following are the steps to be followed while implementing the
**Median Filter**in removing the salt-and-pepper noise from the image −
- **Neighborhood Selection**− For each pixel there will be a neighborhood of surrounding pixels is considered. For example a 3x3 or 5x5 grid.
- **Median Calculation**− The median value of the pixels in that neighborhood is calculated.
- **Pixel Replacement**− The central pixel is replaced with the median value.
- **Preserving Edges**− The median filter is particularly effective at removing noise without blurring sharp edges in an image.
## Advantages of Median filter

Here are the advantages of using the Median filter in Image processing −

- **Effective for Salt-and-Pepper Noise:**Removes noise without blurring the edges as much as linear filters.
- **Non-linear**− It does not rely on weighted sums by making it resistant to outliers.
## Limitations of Median Filter

The Median filter not only have the benefits while using it but also have some limitations, which can be mentioned as follows −

- **Computational Cost**− The median filter can be computationally more expensive than linear filters for larger neighborhoods.
- **Edge Effects**− This filter may not perform as well at the edges of the image where the neighborhood size is constrained by the image boundary.
## Basic Median Filter

### Example

Following is an example shows how to apply a simple
**Median Filter**on a 2D image or array to remove noise with the help of**scipy.ndimage.median_filter()**function −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Sample 2D image (5x5 grid)
image = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25]])

# Apply median filter with a 3x3 neighborhood
filtered_image = ndimage.median_filter(image, size=3)

# Plot original and filtered images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(filtered_image, cmap='gray')
axes[1].set_title('Median Filtered Image')
axes[1].axis('off')

plt.show()
```

#### Output

Following is the output of the basic median filter −
![Median filter example](/scipy/images/median_filter_basic.jpg)
## Median Filter with a Larger Neighborhood

### Example

This example uses a larger neighborhood for the median filter such as 5x5 to show the effect of smoothing on a larger area −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data

# Load sample image
image = data.camera()

# Apply median filter with a 5x5 neighborhood
filtered_image = ndimage.median_filter(image, size=5)

# Plot original and filtered images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(filtered_image, cmap='gray')
axes[1].set_title('Median Filtered (5x5) Image')
axes[1].axis('off')

plt.show()
```

#### Output

Following is the output of the median filter applied on the Larger neighborhood−
![Median filter with neighboorhood](/scipy/images/median_larger_neighbor.jpg)
## Applying Median Filter to 3D Image

Applying a median filter to a 3D image is similar to applying it to a 2D image but in 3D, the filter works across a volume of data where each voxel (3D pixel) is replaced by the median of its neighbors within a defined neighborhood.

### Example

This can be useful for denoising 3D data such as medical imaging, volumetric data from simulations or 3D scanning data. Here is the example where a 3D image is processed with a median filter −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Create a synthetic 3D image (10x10x10 array)
image_3d = np.random.random((10, 10, 10))

# Apply median filter with a 3x3x3 neighborhood
filtered_image_3d = ndimage.median_filter(image_3d, size=3)

# Plot original and filtered slices (3D visualization)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_3d[5, :, :], cmap='gray')  # Show middle slice
axes[0].set_title('Original 3D Slice')
axes[0].axis('off')

axes[1].imshow(filtered_image_3d[5, :, :], cmap='gray')  # Show middle slice of filtered image
axes[1].set_title('Median Filtered 3D Slice')
axes[1].axis('off')

plt.show()
```

#### Output

Following is the output of the median filter applied 3D Image −
![Median filter on 3d image](/scipy/images/median_3d_image.jpg)

---

## 60. SciPy - Non-Linear Filters in Image processing

*Source: [https://www.tutorialspoint.com/scipy/scipy_non_linear_filters_image_processing.htm](https://www.tutorialspoint.com/scipy/scipy_non_linear_filters_image_processing.htm)*

---

---
[Previous](/scipy/scipy_median_filter.htm)[Quiz](/scipy/quiz_on_scipy_non_linear_filters_image_processing.htm)[Next](/scipy/scipy_high_boost_filter.htm)
## Non-Linear Filters in Image processing

In image processing,
**Non-linear filters**refer to filters whose output is not a weighted sum or linear combination of the input values. Unlike linear filters such as Gaussian filters, Sobel operators, non-linear filters process the pixel values in a local neighborhood in a non-linear manner based on operations like sorting, thresholding or other mathematical operations.**Non-linear filters**are often used in situations where linear filters fail to preserve important features like edges or texture or when an image is corrupted with noise i.e., salt-and-pepper noise. They can provide better performance in noise removal, edge detection and preserving the structure of the image.
## Key Features of Non-Linear Filters

Following are the features which make non-linear filters essential tools in image processing, especially for tasks like noise removal, edge preservation, shape manipulation and contrast enhancement.−

- **Edge Preservation:**Non-linear filters such as median filters and bilateral filters are known for their ability to preserve edges while removing noise. When compared to linear filters like Gaussian filter the non-linear filters do not blur edges. This makes them especially useful for applications like denoising where edge detail is crucial.
- **Effective Noise Removal:**Non-linear filters can efficiently remove different types of noise especially salt-and-pepper noise and Gaussian noise. For instance, the median filter is highly effective for removing salt-and-pepper noise without significantly affecting the underlying image details.
- **Non-Linear Neighborhood Operations::**Non-linear filters operate by applying non-linear operations such as median, maximum, minimum within a neighborhood of each pixel rather than using a weighted sum of pixel values. This allows non-linear filters to perform tasks such as object boundary detection, feature enhancement and noise suppression in a more flexible manner compared to linear filters.
- **Adaptability to Image Content:**Non-linear filters such as the bilateral filter are used to adjust their behavior based on the content of the image like spatial distance and pixel intensity differences. This adaptability allows them to preserve important structures such as edges while still filtering out unwanted noise.
- **Shape and Structure Preservation:**Morphological operations such as erosion and dilation are used to modify the shapes and structures of objects in binary and grayscale images. These filters can be used for tasks such as object extraction, segmentation and shape-based noise removal, all while preserving the overall structure of objects.
- **Non-Blurring Smoothing:**Non-linear filters like the bilateral filter smooth the image while preserving sharp edges and important details.
- **Enhancement of Specific Image Features:**Filters like Top-Hat and Bottom-Hat are used to enhance specific image features such as bright or dark objects against a relatively uniform background.
- **Handling of Complex Noise Patterns:**Non-linear filters are robust to complex noise patterns including salt-and-pepper and Gaussian noise by making them ideal for real-world images with various types of disturbances.
- **Flexible Structuring Elements:**In morphological filtering the structuring element such as square, disk, rectangle are used to define how the filter interacts with the image.
- **Local Operation Based on Pixel Neighborhood:**Non-linear filters process pixels based on their local neighborhood and the operation depends on the values of neighboring pixels. This makes non-linear filters powerful in situations where the local context such as surrounding pixels is important such as edge detection, noise removal or feature extraction.
- **Non-Monotonic Behavior:**Non-linear filters especially morphological filters exhibit non-monotonic behavior which means they can produce results that differ significantly from linear filters for the same input.
- **Customizable Filter Behavior:**In many cases non-linear filters can be customized with different parameters such as filter size, thresholds and structuring elements. This flexibility allows users to fine-tune the filter to better suit their specific application whether it's denoising, edge detection or feature extraction.
## Types of Non-Linear Filters

Following are the different types of Non-Linear filters used in Image processing in such a way to address specific tasks such as noise removal, edge preservation and shape extraction.
S.NoFilter TypeCategorySpecific Operation1Median FilterNoise RemovalReplaces central pixel with median2Bilateral FilterEdge-preserving SmoothingUses spatial and intensity differences3Morphological FiltersShape-based OperationsErosion, Dilation, Opening, Closing4Top-Hat FilterContrast EnhancementHighlights bright features5Bottom-Hat FilterContrast EnhancementHighlights dark features
### Example

This example shows how a non-linear filter such as the Median Filter can effectively clean up noise while maintaining important features in the image.

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data

# Load an example image (camera image from skimage.data)
image = data.camera()

# Add salt-and-pepper noise to the image
noise_image = np.copy(image)
num_salt = 0.02 * image.size
salt_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
noise_image[salt_coords[0], salt_coords[1]] = 255  # Salt (white) noise
num_pepper = 0.02 * image.size
pepper_coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
noise_image[pepper_coords[0], pepper_coords[1]] = 0  # Pepper (black) noise

# Apply Median filter using scipy.ndimage.median_filter
filtered_image = ndimage.median_filter(noise_image, size=3)

# Plot original and filtered images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(noise_image, cmap='gray')
axes[0].set_title('Noisy Image')
axes[0].axis('off')

axes[1].imshow(filtered_image, cmap='gray')
axes[1].set_title('Median Filtered Image')
axes[1].axis('off')

plt.show()
```

Here is the output of the Non linear filter −
![Non Linear filter example](/scipy/images/non_linear_example.jpg)

---

## 61. SciPy - High Boost Filter

*Source: [https://www.tutorialspoint.com/scipy/scipy_high_boost_filter.htm](https://www.tutorialspoint.com/scipy/scipy_high_boost_filter.htm)*

---

---
[Previous](/scipy/scipy_non_linear_filters_image_processing.htm)[Quiz](/scipy/quiz_on_scipy_high_boost_filter.htm)[Next](/scipy/scipy_laplacian_filter.htm)
## High-Boost Filter in SciPy

A
**High-boost filter**is an image sharpening technique that enhances the high-frequency components such as edges and fine details, while retaining the original image's low-frequency content. It is often used to emphasize subtle details in images or restore blurred images.
We don't have a specified function in
**scipy.ndimage**module of SciPy library but we can to implement this filter in SciPy with the help of low pass filters such as Gaussian filter. The high-boost filtered image can be calculated as follows −
```
Hb = A . I - G
```
= A . I - G
Where −

- **I:**Original Image
- **G:**Smoothed version of the image which is usually obtained using a low-pass filter like Gaussian blur.
- **A:**Amplification factor i.e., boosting constant when A=1, it reduces to a high-pass filter.
We also have the an alternative representation of the High - Boost filter as follows −

```
Hb = (A - 1) . I + (I - G)
```
= (A - 1) . I + (I - G)
Where −

- **(I - G):**High-frequency components i.e., details and edges.
- **(A - 1).I:**Low-frequency content amplified by A1.
## Properties of the High - Boost Filter

The High - Boost Filter exhibits some properties which are mentioned as follows −

- When**A > 1**then the filter enhances the high-frequency components while retaining the low-frequency components of the original image.
- When**A = 1**then the filter reduces to a standard high-pass filter by emphasizing only edges and details.
- High-boost filters provide a flexible sharpening mechanism where we can control the degree of sharpening through A.
## Steps to Apply High Boost Filter

To apply the High - Boost Filter to an image we have to follow certain steps. Here are the steps to be followed −

- **Smooth the image:**Firstly we have to smooth the image by using a low pass filter such as Guassian filter to extract low-frequency components.
- **Subtract the smoothed image:**Next we have to subtract the smoothed image from the original image to extract the high-frequency components.
- **Add back the original image:**Finally we have to add back the original image multiplied by the amplification factor to retain and amplify the low-frequency content.
## Basic High-Boost Filtering Example

Following is the example of the basic high boost filter applied to the given input image by using the function
**scipy.ndimage.guassian()**−
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data, color

# Load a sample image (e.g., astronaut image from skimage)
image = color.rgb2gray(data.astronaut())  # Convert to grayscale

# Define the amplification factor
A = 1.5  # Adjust this value for more or less sharpening

# Create a smoothed version of the image using Gaussian filter
smoothed = ndimage.gaussian_filter(image, sigma=3)

# Compute the high-boost filtered image
high_boost = A * image - smoothed

# Plot the original, smoothed, and high-boost filtered images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Smoothed Image (Low-Pass Filter)")
plt.imshow(smoothed, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"High-Boost Filtered Image (A = {A})")
plt.imshow(high_boost, cmap='gray')
plt.axis('off')

plt.show()
```

Here is the output of the basic high boost filter applied on the input image −
![Basic High boost filter](/scipy/images/high_boost_basic.jpg)
## Varying Amplification Factor A

The high-boost filter is a popular image sharpening technique that enhances details by amplifying high-frequency components while retaining some of the original image's low-frequency content. The amplification factor
**A**in the high-boost filter plays a key role in determining the intensity of the sharpening effect. The High boost filter formula in terms of Amplification factor can be given as follows −
```
High-Boost Image = A . I - Blurred Image = I + (A . 1) . Mask
```

Where −

- **I:**Original Image
- **BlurredImage:**Result of applying a smoothing (low-pass) filter to I.
- **A:**Amplification factor which is typically A  1.
- **Mask:**Difference between the original image and the blurred image (IBlurredImage).
For A = 1, the result is equivalent to standard image sharpening. For A > 1, the high-boost effect becomes stronger.

Below is an example of applying a high-boost filter to an image with different amplification factors.−

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data, color

# Load and preprocess the image
image = color.rgb2gray(data.astronaut())  # Convert to grayscale

# Define a Gaussian blur for low-pass filtering
blurred = ndimage.gaussian_filter(image, sigma=2)

# Compute the mask (high-frequency components)
mask = image - blurred

# Apply high-boost filtering with different A values
A_values = [1, 1.5, 2, 3]  # Amplification factors
high_boost_images = [image + (A - 1) * mask for A in A_values]

# Plot original, mask, and high-boost images
plt.figure(figsize=(15, 8))

# Original image
plt.subplot(2, len(A_values) + 1, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Mask
plt.subplot(2, len(A_values) + 1, 2)
plt.title("Mask (High-Freq Components)")
plt.imshow(mask, cmap='gray')
plt.axis('off')

# High-boost images
for i, (A, hb_image) in enumerate(zip(A_values, high_boost_images), start=3):
    plt.subplot(2, len(A_values) + 1, i)
    plt.title(f"High-Boost A={A}")
    plt.imshow(hb_image, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
```

Here is the output of the high boost filter with varying Amplification factor A −
![Basic High boost filter Amplification](/scipy/images/high_boost_amplification_a.jpg)
## Comparing High-Pass & High-Boost Filters

The high-pass filter and high-boost filter are closely related but they serve slightly different purpose. The High pass filter enhances high-frequency components by removing low-frequency components where as the High Boost filter enhances high-frequency components while retaining some of the original image's low-frequency content, controlled by an amplification factor A.

Following example highlights the difference between a high-pass filter (A=1) and a high-boost filter (A > 1).

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data, color

# Load and preprocess the image
image = color.rgb2gray(data.astronaut())  # Convert to grayscale

# Apply a Gaussian blur for low-pass filtering
blurred = ndimage.gaussian_filter(image, sigma=2)

# Compute the high-pass filter result
high_pass = image - blurred

# Compute the high-boost filter result with varying amplification factors
A = 2  # Amplification factor for high-boost
high_boost = image + (A - 1) * high_pass

# Plot original, high-pass, and high-boost images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("High-Pass Filter")
plt.imshow(high_pass, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"High-Boost Filter (A={A})")
plt.imshow(high_boost, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```

Here is the output image, which shows the comparision of the High pass and High Boost filters −
![High boost & High pass filters comparision](/scipy/images/high_pass_boost_comparision.jpg)

---

## 62. SciPy - Laplacian Filter

*Source: [https://www.tutorialspoint.com/scipy/scipy_laplacian_filter.htm](https://www.tutorialspoint.com/scipy/scipy_laplacian_filter.htm)*

---

---
[Previous](/scipy/scipy_high_boost_filter.htm)[Quiz](/scipy/quiz_on_scipy_laplacian_filter.htm)[Next](/scipy/scipy_morphological_operations.htm)
## Laplacian Filter in SciPy

The
**Laplacian filter**is a second-order derivative filter used to highlight regions of rapid intensity change in an image such as edges. It calculates the Laplacian which is the sum of the second derivatives in the x and y directions.
For a 2D image I(x,y), mathematically the Laplacian is given as follows −
![Laplacian Formula](/scipy/images/laplacian_formula.jpg)
This operation captures regions where the intensity changes abruptly. When we want to implement the
**Laplacian filter**using scipy library then we can use the function**scipy.ndimage.laplace()**.
### Syntax

Following is the syntax of the
**scipy.ndimage.laplace()**function −
```
scipy.ndimage.laplace(input, output=None, mode='reflect', cval=0.0)
```

Following are the parameters of the function
**scipy.ndimage.laplace()**−
- **input**− The input array i.e., image to which the filter is applied.
- **output(optional)**− An array to store the result. If not provided then a new array is created.
- **mode**− This parameter defines how the input array is extended at its boundaries. The modes can be 'reflect', 'constant', 'nearest', 'mirror' and 'wrap'.
- **cval**− This is a value to fill past edges if mode='constant'. Default value is 0.0.
### Basic Laplacian Filter on a Synthetic Image

Following is an example which shows how to apply the basic
**Laplacian Filter**on a synthetic image using the function**scipy.ndimage.laplace()**−
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Create a synthetic checkerboard pattern
x = np.indices((100, 100)).sum(axis=0) % 2
image = x.astype(float)

# Apply the Laplacian filter
laplacian = ndimage.laplace(image)

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Checkerboard Pattern")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Laplacian Filter")
plt.imshow(laplacian, cmap='gray')
plt.axis('off')
plt.show()
```

#### Output

Following is the output of the basic laplacian filter applied on the given image −
![Laplacian Basic](/scipy/images/laplacian_basic.jpg)
## Laplacian Filter with Pre-Smoothing

The Laplacian filter with pre-smoothing is a technique in image processing where a Gaussian filter is applied to reduce noise before applying the Laplacian filter. This combination is effective for edge detection while minimizing noise artifacts. Following is the example applying a Gaussian filter before the Laplacian filter on a noisy image −

### Example

```
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Create a noisy image
np.random.seed(42)
image = np.random.random((100, 100))

# Apply Gaussian smoothing before Laplacian filter
smoothed = ndimage.gaussian_filter(image, sigma=2)
laplacian = ndimage.laplace(smoothed)

# Plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Noisy Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Smoothed Image")
plt.imshow(smoothed, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Laplacian Filter (Post-Smoothing)")
plt.imshow(laplacian, cmap='gray')
plt.axis('off')
plt.show()
```

#### Output

Following is the output of the applying the Laplace Filter with Pre - smoothing −
![Laplacian with pre - smoothing](/scipy/images/laplace_presmooth.jpg)
## Laplacian Filter with Boundary Modes

In the Laplacian filter and Gaussian smoothing the boundary modes control how the edges of the image are handled during the convolution. This is important because the filter operates beyond the original image's boundaries. Here in this example we are passing the mode parameter to the
**scipy.ndimage.laplace()**function −
### Example

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data, color

# Step 1: Load image
image = color.rgb2gray(data.astronaut())  # Convert to grayscale

# Step 2: Pre-smoothing with Gaussian filter
sigma = 2
smoothed_image = ndimage.gaussian_filter(image, sigma=sigma)

# Step 3: Apply Laplacian filter with different modes
modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
results = {}

for mode in modes:
    laplacian = ndimage.laplace(smoothed_image, mode=mode, cval=0)  # cval=0 for 'constant'
    results[mode] = laplacian

# Step 4: Visualize the results
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

for i, mode in enumerate(modes, start=2):
    plt.subplot(2, 3, i)
    plt.title(f"Mode: {mode}")
    plt.imshow(results[mode], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

Following is the output of the applying the Laplace Filter with Pre - smoothing −
![Laplacian with boundary modes](/scipy/images/laplace_modes.jpg)

---

## 63. SciPy - Morphological Operations

*Source: [https://www.tutorialspoint.com/scipy/scipy_morphological_operations.htm](https://www.tutorialspoint.com/scipy/scipy_morphological_operations.htm)*

---

---
[Previous](/scipy/scipy_laplacian_filter.htm)[Quiz](/scipy/quiz_on_scipy_morphological_operations.htm)[Next](/scipy/scipy_image_segmentation.htm)
Morphological operations are a set of image processing techniques that process images based on their shapes. They are primarily applied to binary and gray-scale images and rely on a structuring element to probe, transform and manipulate the shapes in the image.

The core concept behind the Morphological operations is set theory, where the image is treated as a set of pixels and the transformations depend on the interaction between the image and the structuring element.

## Key concepts in Morphological Operations

Before proceeding with the Morphological Operations in detail, first we need to get to know the key concepts of them. Here are the key concepts of Morphological Operations in SciPy −

## Image Representation

In the view of morphological operations the images are represented as arrays of pixel values. This array-based representation enables mathematical manipulations using structuring elements. SciPy and similar libraries handle images as NumPy arrays which allow efficient computations.

The images in morphological Operations can be represented in two ways as mentioned below −

- **Binary Images:**A binary image consists of only two pixel values as 0 and 1 or 0 and 255 in some systems. 0 represents the background i.e., black while 1 represents the foreground i.e., white.
- **Grayscale Images:**A grayscale image contains pixel values ranging from 0 i.e., black to 255 i.e., white. Each pixel intensity represents the brightness level of that point in the image.
### Structuring Element (Kernel)

The structuring element is a small matrix or pattern that defines the neighborhood used to perform morphological operations. Its shapes like  square, disk, cross determines how the operation interacts with the image and the size of the structuring element affects the scale of the transformation.

### Origin

The center of the structuring element determines its alignment with the image pixels during the operation.

## Core Morphological Operations

Morphological operations are fundamental techniques used in image processing for shape analysis, noise removal and feature extraction. The core operations are Erosion, Dilation, Opening and Closing. These operations rely on the use of a structuring element to probe the image. Let's see all the available Morphological operations in SciPy Image processing −

## Erosion Using SciPy

Erosion shrinks the foreground i.e., bright regions of an image by removing pixels at the boundaries of objects. This Operation works as, if a pixel in the output is set to 1 or remains part of the foreground only then all pixels in the neighborhood defined by the structuring element are 1.

When we apply the Erosion operation then it reduces the size of objects, removes small white noise and disconnects joined components. We have a function as
**binary_erosion()**in**scipy.ndimage**module to perform**Erosion**Morphological Operation in SciPy.
Following is the mathematical formula of Erosion −

```
AB={z(B)z A}
```
A}
where A is the input image, B is the structuring element and (B)
represents the structuring element translated to position z.
## Dilation Ui

Dilation grows or expands the foreground, i.e., bright regions of an image, by adding pixels to the boundaries of objects. This operation works as, if a pixel in the output is set to 1, at least one pixel in the neighborhood defined by the structuring element must be 1.

When we apply the Dilation operation, it increases the size of objects, fills small holes and connects nearby components. We have a function as
**binary_dilation()**in**scipy.ndimage**module to perform**Dilation**Morphological Operation in SciPy.
Following is the mathematical formula of Dilation −

```
AB={z(B)z  A  }
```
A  }
where A is the input image, B is the structuring element and (B)
represents the structuring element translated to position z.
### Opening

Opening is a combination of Erosion followed by Dilation. It works by first performing Erosion to remove small bright regions or thin details and then applying Dilation to restore the main shape of the objects.

The Opening operation smoothens the boundaries of objects, removes small bright regions and eliminates noise. We have a function as
**binary_opening()**in**scipy.ndimage**module to perform**Opening**Morphological Operation in SciPy.
Following is the mathematical formula of Opening −

```
AB = (AB)B
```

where A is the input image and B is the structuring element.

### Closing

Closing is a combination of Dilation followed by Erosion. It works by first performing Dilation to expand the boundaries of objects and then applying Erosion to restore the shape of the objects.

The Closing operation smoothens object boundaries, fills small black holes and connects broken components. We have a function as
**binary_closing()**in**scipy.ndimage**module to perform**Closing**Morphological Operation in SciPy.
Following is the mathematical formula of Closing −

```
AB = (AB)B
```

where A is the input image and B is the structuring element.

### Morphological Gradient

The Morphological Gradient highlights the boundaries of objects by subtracting the Eroded image from the Dilated image. It is useful for detecting edges or outlines of objects in an image.

The Morphological Gradient operation produces an outline of the object boundaries by comparing the effects of Dilation and Erosion. It can be implemented using functions like
**grey_dilation()**and**grey_erosion()**in**scipy.ndimage**module.
Following is the mathematical formula of Morphological Gradient −

```
Gradient = (AB)  (AB)
```

where A is the input image and B is the structuring element.

### Top-Hat Transform

The Top-Hat Transform is used to extract small bright or dark regions in an image. It comes in two variations as mentioned below −

- **White Top-Hat**: This extracts bright regions smaller than the structuring element.
- **Black Top-Hat**: It extracts dark regions smaller than the structuring element.
The White Top-Hat Transform is performed by subtracting the result of Opening from the original image, while the Black Top-Hat Transform is performed by subtracting the original image from the result of Closing. We have functions as
**white_tophat()**and**black_tophat()**in**scipy.ndimage**module.
Following are the mathematical formulas of Top-Hat Transform −
**White Top-Hat:**
```
White Top-Hat = A  (AB)
```
**Black Top-Hat:**
```
Black Top-Hat = (AB)  A
```

where A is the input image and B is the structuring element.

Following are the functions available in
**scipy.ndimage**module to perform the Morphological Operations in image processing −S.No.Function & Description1[scipy.ndimage.binary_erosion()](/scipy/scipy_ndimage_binary_erosion_function.htm)
Perform erosion on a binary image (shrinking).2[scipy.ndimage.binary_dilation()](/scipy/scipy_ndimage_binary_dilation_function.htm)
Perform dilation on a binary image (expanding).3[scipy.ndimage.binary_opening()](/scipy/scipy_ndimage_binary_opening_function.htm)
Perform binary opening i.e., erosion followed by dilation.4[scipy.ndimage.binary_closing()](/scipy/scipy_ndimage_binary_closing_function.htm)
Perform binary closing i.e., dilation followed by erosion.5[scipy.ndimage.grey_erosion()](/scipy/scipy_ndimage_grey_erosion_function.htm)
Shrinks bright regions in the image.6[scipy.ndimage.grey_dilation()](/scipy/scipy_ndimage_grey_dilation_function.htm)
Expands bright regions in the image.7[scipy.ndimage.grey_opening()](/scipy/scipy_ndimage_grey_opening_function.htm)
Perform grayscale opening, removing small bright spots.8[scipy.ndimage.grey_closing()](/scipy/scipy_ndimage_grey_closing_function.htm)
Perform grayscale closing, filling small dark holes.9[scipy.ndimage.white_tophat_function()](/scipy/scipy_ndimage_white_tophat_function.htm)
Enhances and extracts small bright features in images.10[scipy.ndimage.black_tophat_function()](/scipy/scipy_ndimage_black_tophat_function.htm)
Enhances and extract small dark features from an image.
## Applications of morphological operations

Morphological operations are widely used in image processing and computer vision tasks. They provide effective tools for analyzing, pre-processing and enhancing image features by manipulating shapes and structures within images. Below are the key applications of morphological operations −

- **Noise Removal:**Removes small objects or artifacts using Opening i.e.,erosion followed by dilation.
- **Image Preprocessing:**Enhances images for segmentation and edge detection using Erosion and Dilation.
- **Boundary Extraction:**Extracts object boundaries by subtracting an eroded image from the original image.
- **Object Detection and Segmentation:**Identifies and isolates objects of interest using Closing and Opening.
- **Skeletonization:**Removes small objects or artifacts using Opening i.e.,erosion followed by dilation.
- **Hole Filling:**Fills small gaps or holes within objects using Closing.
- **Edge Detection:**Highlights object edges using Morphological Gradient (dilation - erosion).
- **Shape Analysis:**Analyzes and measures object properties like size, shape, and connectivity.
- **Text Extraction:**Extracts and enhances text regions in document images.

---

## 64. SciPy Image Segmentation

*Source: [https://www.tutorialspoint.com/scipy/scipy_image_segmentation.htm](https://www.tutorialspoint.com/scipy/scipy_image_segmentation.htm)*

---

---
[Previous](/scipy/scipy_morphological_operations.htm)[Quiz](/scipy/quiz_on_scipy_image_segmentation.htm)[Next](/scipy/scipy_thresholding_image_segmentation.htm)
## Image Segmentation in SciPy
**Image segmentation**in SciPy is the process of dividing an image into distinct regions or objects based on certain characteristics such as pixel intensity, color, texture or boundaries. The goal of Image Segmentation is to simplify the image for further analysis by identifying meaningful structures or separating the foreground from the background.
In SciPy the
**scipy.ndimage**module provides a variety of tools and algorithms for image segmentation. These are primarily used in pre-processing, feature extraction and analysis tasks in scientific and engineering applications.
## Key concepts in Image segmentation
**Image segmentation**is an essential process in image analysis and computer vision which involves in the division of an image into meaningful regions. Below are the key concepts that underpin this technique −
### Segmentation Objective

The main goal of the Image segmentation is to simplify an image by identifying and isolating regions or objects of interest. This mainly focus on pixels within a segment share similar properties like intensity, color or texture and distinct segments differ significantly.

## Homogeneity & Discontinuity

In image segmentation, the goal is to partition an image into regions that are meaningful for further analysis or processing. There are two key concepts that often guide this segmentation process are
**homogeneity**and**discontinuity**. These concepts help in defining the criteria for how pixels or regions of an image should be grouped or separated.
### Homogeneity
**Homogeneity**refers to the similarity or uniformity within a region of an image. In the view of image segmentation it is the property of pixels or regions that share similar characteristics such as color, intensity, texture or other features.
The role of Homogeneity in Segmentation is to seek the group pixels with similar characteristics into the same region. For example pixels that have similar intensity or color values are considered part of the same homogeneous region and a region is considered homogeneous if its pixels are within a defined range of similarity.

### Discontinuity

Discontinuity refers to abrupt changes or differences in pixel values that separate different regions in an image. These discontinuities can occur in terms of intensity, color or texture and are typically found at object boundaries.

Discontinuity-based segmentation techniques attempt to identify edges or boundaries where there is a sharp contrast in pixel values. These discontinuities are typically used to divide an image into separate regions or objects.

Segmentation based on discontinuity often focuses on detecting edges or boundaries between different regions, which are areas where the pixel intensities or features change abruptly.

## Types of Image Segmentation Techniques

Following different types of Image Segmentation techniques available in SciPy −
Segmentation TechniqueDescriptionImplementationApplications**Thresholding**Segments the image based on pixel intensity, classifying pixels above or below a threshold as different regions.No separate function is availableGlobal Thresholding, Adaptive Thresholding**Region-Based Segmentation**Groups pixels into regions based on shared properties like intensity, color, or texture.ndimage.label(), measurements.find_objects()Connected Component Labeling, Watershed Segmentation**Edge-Based Segmentation**Detects boundaries between regions by identifying rapid changes in intensity (edges).ndimage.sobel(), ndimage.laplace()Sobel Edge Detection, Laplacian of Gaussian for edge finding**Morphological Operations**Refines segmentation by modifying the shape of regions (erosion, dilation, opening, closing).ndimage.binary_erosion(), ndimage.binary_dilation()Noise removal, Filling gaps, Object merging**Connected Component Labeling**Labels each connected component (region) in the image, commonly used for identifying distinct regions.ndimage.label(), ndimage.find_objects()Object counting, Region labeling for further analysis or processing
## Applications of Image Segmentation

Image segmentation using SciPy is widely applied in various domains. Below are some of the primary applications of image segmentation −

- **Medical Imaging:**Segmentation helps in detecting and delineating regions of interest such as tumors, organs and lesions in medical scans e.g., MRI, CT.
- **Object Recognition & Detection:**Image Segmentation is used to identify and classify objects or regions within an image based on characteristics such as color, texture or intensity.
- **Image Preprocessing for Computer Vision:**Segmentation is often used as a preprocessing step for tasks like feature extraction, object tracking or classification.
- **Remote Sensing:**Applied to analyze and interpret images from satellites or drones for land use classification, vegetation detection or water bodies segmentation.
- **Robotics & Automation:**Segmentation aids robots in understanding their environment by separating obstacles and free space for navigation.
- **Surveillance & Security:**Applied in surveillance cameras for monitoring, detecting people, and recognizing specific objects or regions.
- **Face Recognition & Emotion Detection:**Segmentation helps in identifying key facial features and regions for emotion or identity recognition.
- **Art and Historical Preservation:**It is used to segment and analyze old or damaged artwork for restoration purposes.

---

## 65. SciPy - Thresholding in Image Segmentation

*Source: [https://www.tutorialspoint.com/scipy/scipy_thresholding_image_segmentation.htm](https://www.tutorialspoint.com/scipy/scipy_thresholding_image_segmentation.htm)*

---

---
[Previous](/scipy/scipy_image_segmentation.htm)[Quiz](/scipy/quiz_on_scipy_thresholding_image_segmentation.htm)[Next](/scipy/scipy_region_based_segmentation.htm)**Thresholding in image segmentation**in SciPy is a fundamental technique used in image processing to separate different regions in an image based on pixel intensity values.
The basic idea is to apply a threshold to an image so that pixels with values above a certain threshold are classified into one group which often called
**foreground**and those below are classified into another group which often called**background**.
In SciPy library the thresholding image segmentation is often implemented using the
**scipy.ndimage**module along with the other libraries such as Matplotlib and Numpy for image processing. Here in this chapter we will see, thresholding image segmentation in detail.
## Types of Thresholding in Image Segmentation

Thresholding is a key method in image processing used to segment an image by converting grayscale values into binary values. There different types of thresholding techniques that cater to various requirements based on the nature of the image as mentioned below −

## Global Thresholding
**Global thresholding**is a simple and effective image segmentation technique where a single threshold value is used to classify pixels into two categories namely,**foreground**and**background**. In SciPy global thresholding can be achieved using basic array operations and tools from the**scipy.ndimage**module.
Following are the steps used to implement the Global thresholding in SciPy −

- 
Load or create an image.

- 
Define a global threshold value.

- 
Apply the threshold by comparing the image pixel values to the threshold.

- 
Finally, generate a binary mask where pixels above the threshold are classified as foreground
**True**and pixels below are classified as background**False**.
### Example

Here is a complete example of
**Global Thresholding**using NumPy and Matplotlib to implement a Global thresholding operation −
```
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Create a synthetic grayscale image
np.random.seed(42)
image = np.random.random((100, 100))  # Values between 0 and 1

# Define a global threshold value
threshold_value = 0.5

# Apply global thresholding manually
binary_image = image > threshold_value

# Display the original and binary thresholded images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

# Binary thresholded image
axes[1].imshow(binary_image, cmap='gray')
axes[1].set_title(f"Binary Thresholded Image (T = {threshold_value})")
axes[1].axis("off")

plt.tight_layout()
plt.show()
```

#### Output

Below is the output of the Global thresholding of the image segmentation implemented with the help Numpy and Matplotlib libraries −
![Global Thresholding](/scipy/images/global_threshold.jpg)
## Otsu's Thresholding

Otsu's Thresholding is a global thresholding technique that determines the optimal threshold value for an image by maximizing the variance between the foreground and background classes. This method assumes that the image has a bimodal histogram i.e., two distinct peaks representing background and foreground.

The main objective of this method is to find the threshold T that minimizes the intra-class variance or equivalently maximizes the inter-class variance.

### Example
**Otsus Thresholding**is not directly implemented in SciPy but we can achieve it by using a combination of SciPy's histogram functionality and NumPy for calculations. Below is the example which gives step-by-step approach of how to implement Otsu's Thresholding in SciPy −
```
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def otsu_threshold(image):
   """
   Computes Otsu's threshold for a grayscale image.
   
   Parameters:
       image (ndarray): Input grayscale image (2D array).
   
   Returns:
       threshold (float): Optimal threshold value.
   """
   # Compute the histogram of the image
   hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))
   bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
   
   # Total number of pixels
   total_pixels = image.size
   
   # Cumulative sums and cumulative means
   cumulative_sum = np.cumsum(hist)
   cumulative_mean = np.cumsum(hist * bin_centers)
   
   # Global mean
   global_mean = cumulative_mean[-1] / total_pixels
   
   # Between-class variance
   numerator = (global_mean * cumulative_sum - cumulative_mean) ** 2
   denominator = cumulative_sum * (total_pixels - cumulative_sum)
   between_class_variance = numerator / (denominator + 1e-10)  # Add epsilon to avoid division by zero
   
   # Find the maximum of between-class variance
   optimal_idx = np.argmax(between_class_variance)
   optimal_threshold = bin_centers[optimal_idx]
   
   return optimal_threshold
   
# Generate a synthetic grayscale image (random noise for demonstration)
np.random.seed(0)
image = np.random.random((100, 100))

# Apply Otsu's thresholding
optimal_threshold = otsu_threshold(image)
thresholded_image = image > optimal_threshold

# Display the original and thresholded images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(thresholded_image, cmap='gray')
ax[1].set_title(f"Thresholded Image\n(Otsu's Threshold = {optimal_threshold:.2f})")
ax[1].axis('off')

plt.tight_layout()
plt.show()
```

#### Output

Below is the output of implementing the
**Otsu thresholding**−![Ostu Thresholding](/scipy/images/otsu_thresholding.jpg)
## Adaptive Thresholding
**Adaptive Thresholding**is a thresholding technique where the threshold value varies across the image by depending on local image properties. This method is particularly useful for images with uneven lighting or varying intensity levels where a single global threshold would fail to segment the image properly.
Following are the steps involved in implementing the Adaptive Thresholding −

- **Divide the Image into Local Regions:**The image is analyzed in small neighborhoods or regions instead of globally.
- **Calculate Local Statistics:**Compute a local threshold for each pixel based on the statistics e.g., mean or Gaussian-weighted mean of the surrounding region.
- **Threshold Each Pixel:**Classify pixels as foreground or background based on the locally computed threshold.
### Example

Here is an example using SciPy's tools to implement adaptive thresholding using a local mean filter −

```
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Create a synthetic image with uneven lighting
np.random.seed(0)
x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
image = np.sin(10 * x * y) + np.random.random((100, 100)) * 0.5

# Apply a local mean filter
window_size = 15  # Size of the local region
local_mean = ndi.uniform_filter(image, size=window_size)

# Perform adaptive thresholding
thresholded_image = image > local_mean

# Display the results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(local_mean, cmap='gray')
ax[1].set_title("Local Mean")
ax[1].axis('off')

ax[2].imshow(thresholded_image, cmap='gray')
ax[2].set_title("Adaptive Thresholded Image")
ax[2].axis('off')

plt.tight_layout()
plt.show()
```

#### Output
![Adaptive Thresholding](/scipy/images/adaptive_thresholding.jpg)
## Advanced Adaptive Thresholding Technique

### Example

For more advanced adaptive thresholding technique we can use a Gaussian-weighted mean instead of a simple mean. Below is the example of the advanced adaptive thresholding technique −

```
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Create a synthetic image with uneven lighting
np.random.seed(0)
x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
image = np.sin(10 * x * y) + np.random.random((100, 100)) * 0.5

# Apply a local Gaussian filter
sigma = 5  # Standard deviation for Gaussian kernel
local_gaussian_mean = ndi.gaussian_filter(image, sigma=sigma)

# Perform adaptive thresholding
thresholded_image_gaussian = image > local_gaussian_mean

# Display results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(local_gaussian_mean, cmap='gray')
ax[0].set_title("Local Gaussian Mean")
ax[0].axis('off')

ax[1].imshow(thresholded_image_gaussian, cmap='gray')
ax[1].set_title("Gaussian Adaptive Thresholded Image")
ax[1].axis('off')

plt.tight_layout()
plt.show()
```

#### Output

Below is the output of the advanced adapative filtering −
![Advanced Adaptive Thresholding](/scipy/images/adaptive_thresholding_advanced.jpg)

---

## 66. SciPy - Region-based Segmentation

*Source: [https://www.tutorialspoint.com/scipy/scipy_region_based_segmentation.htm](https://www.tutorialspoint.com/scipy/scipy_region_based_segmentation.htm)*

---

---
[Previous](/scipy/scipy_thresholding_image_segmentation.htm)[Quiz](/scipy/quiz_on_scipy_region_based_segmentation.htm)[Next](/scipy/scipy_connected_component_labeling.htm)
Region-based segmentation is a key approach in image processing for dividing an image into meaningful regions based on pixel intensity, color, texture or other features.

In the view of SciPy, region-based segmentation often leverages libraries such as
**scipy.ndimage**and other Python libraries like skimage (scikit-image).
## Approaches of Region-Based Segmentation
**Region-based segmentation**approaches aim to divide an image into meaningful regions based on certain criteria such as intensity, color or texture. Below are the primary approaches and let's see them one by one −
### Region Growing
**Region Growing**is a pixel-based image segmentation method that starts from one or more seed points and grows regions by adding neighboring pixels that satisfy a similarity criterion. This approach is intuitive, simple and widely used in image processing particularly when prior knowledge about the region of interest (ROI) is available.
## Key Concepts

Here are the key concepts that we have to learn before proceeding with the Region growing approach −

- **Seed points**are the foundation of the region-growing algorithm. They are the starting locations from which the segmentation process begins. The choice and placement of seed points are crucial as they determine the quality and extent of the segmented region.
- **Growth criteria**define the rules for adding new pixels to a region in the region-growing algorithm. These criteria ensure that the region remains consistent and homogeneous based on specific properties like intensity, color or texture. It is important to define growth criteria properly for obtaining meaningful and accurate segmentation results.
- **Connectivity**is a critical concept in region-based segmentation and image analysis. It defines how pixels (or voxels in 3D) are considered connected to one another based on their spatial arrangement and/or similarity in values. Connectivity determines how regions are formed by grouping pixels into connected components.
- **Stopping conditions**are essential in region-based segmentation algorithms to define when the segmentation process should halt. These conditions ensure that the algorithm terminates at an appropriate point, preventing over-segmentation or unnecessary computations.
## Region Growing Algorithm

The Region Growing algorithm iteratively adds neighboring pixels to an existing region based on a set of criteria such as intensity, color, texture. Below is a detailed step-by-step breakdown of the Region Growing algorithm −

- **Initialize**− Choose the seed points and define a similarity threshold and Connectivity.
- **Region Growing**− Add neighboring pixels to the region if they meet the growth criteria and update the region's properties such as mean intensity as pixels are added.
- **Repeat**− Continue until no new pixels can be added.
- **Output**− Segmented regions representing different parts of the image.
### Example

Following is the example which shows how to perform
**Region Growing**approach on an image. In this example we are implementing the region growing for a greyscale image −
```
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seed_point, threshold):
   """
   Perform region growing on a grayscale image.
   
   Parameters:
      image (2D numpy array): Input grayscale image.
      seed_point (tuple): Starting point (row, col) for region growing.
      threshold (float): Intensity difference threshold for region inclusion.
   
   Returns:
      region (2D numpy array): Binary mask of the segmented region.
   """
   rows, cols = image.shape
   region = np.zeros_like(image, dtype=bool)
   visited = np.zeros_like(image, dtype=bool)
   intensity = image[seed_point]
   
   # Initialize the region with the seed point
   region[seed_point] = True
   visited[seed_point] = True
   to_process = [seed_point]
   
   while to_process:
      current_point = to_process.pop(0)
      r, c = current_point
      
      # Check 4-connectivity neighbors
      for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
         nr, nc = r + dr, c + dc
         if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
            visited[nr, nc] = True
            # Check similarity condition
            if abs(image[nr, nc] - intensity) <= threshold:
               region[nr, nc] = True
               to_process.append((nr, nc))
            
   return region

# Example usage
if __name__ == "__main__":
   # Create a synthetic grayscale image
   image = np.array([[1, 1, 2, 2, 3],
                     [1, 1, 2, 3, 3],
                     [1, 2, 2, 3, 4],
                     [2, 2, 3, 4, 4],
                     [3, 3, 4, 4, 5]], dtype=float)
   
   # Define seed point and threshold
   seed_point = (2, 2)  # Starting point in the image
   threshold = 1.0      # Intensity difference threshold
   
   # Apply region growing
   segmented_region = region_growing(image, seed_point, threshold)
   
   # Plot results
   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.title("Original Image")
   plt.imshow(image, cmap="gray", interpolation="none")
   plt.scatter(seed_point[1], seed_point[0], color="red")  # Mark seed point
   
   plt.subplot(1, 2, 2)
   plt.title("Segmented Region")
   plt.imshow(segmented_region, cmap="gray", interpolation="none")
   plt.show()
```

#### Output

Following is the output of
**Region Growing**approach on a greyscale image −![Region Growing Example](/scipy/images/region_growing_example.jpg)
## Region Splitting & Merging
**Region Splitting and Merging**is a hybrid image segmentation approach that combines top-down and bottom-up techniques. It starts by splitting an image into smaller regions and then merges adjacent regions that satisfy a similarity criterion. This method is particularly useful for hierarchical segmentation and efficiently balances detail and computational effort.
### Key Concepts in Region Splitting & Merging

Here are the key concepts that we have to know before proceeding with the Region Splitting and Merging approach −

- **Homogeneity criteria**define whether a region is uniform based on specific properties such as intensity, color or texture. Proper criteria ensure meaningful and accurate segmentation results.
- **Quadtree decomposition**is a common splitting method where an image is divided into four quadrants recursively until regions become homogeneous or reach a minimum size.
- **Merging criteria**determine how adjacent regions are combined. Regions with similar properties are merged to form larger homogeneous areas.
- **Stopping conditions**are essential to define when splitting or merging should stop by ensuring efficient computation and meaningful segmentation.
### Region Splitting & Merging Algorithm

The Region Splitting and Merging algorithm alternates between dividing and combining regions based on defined criteria. Below is a detailed step-by-step breakdown −

- **Initialize**− Treat the entire image as a single region. Define homogeneity and merging criteria.
- **Region Splitting**− Recursively divide non-homogeneous regions, typically using quadtree decomposition.
- **Region Merging**− Combine adjacent homogeneous regions that meet the merging criteria.
- **Repeat**− Continue splitting and merging until no further changes occur.
- **Output**− Homogeneous regions representing different parts of the image.
### Example

Following is the example which shows how to perform
**Region Splitting and Merging**approach on an image. In this example we are implementing the algorithm using quad-tree decomposition for splitting and simple merging criteria −
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def is_homogeneous(region, threshold=10):
   """Check if a region is homogeneous based on intensity variance."""
   return np.var(region) < threshold

def quadtree_split(image, x, y, size, threshold, segmentation, label_id):
   """Recursively split a region into quadrants if not homogeneous."""
   if size <= 1 or is_homogeneous(image[y:y+size, x:x+size], threshold):
       segmentation[y:y+size, x:x+size] = label_id
       return label_id + 1
   
   half = size // 2
   label_id = quadtree_split(image, x, y, half, threshold, segmentation, label_id)  # Top-left
   label_id = quadtree_split(image, x + half, y, half, threshold, segmentation, label_id)  # Top-right
   label_id = quadtree_split(image, x, y + half, half, threshold, segmentation, label_id)  # Bottom-left
   label_id = quadtree_split(image, x + half, y + half, half, threshold, segmentation, label_id)  # Bottom-right
   
   return label_id

def region_merge(segmentation, image, threshold):
   """Merge adjacent regions based on similarity."""
   labeled_regions, _ = label(segmentation)
   region_means = {region: np.mean(image[segmentation == region]) for region in np.unique(segmentation)}
   
   for region_a in region_means:
      for region_b in region_means:
          if region_a != region_b and abs(region_means[region_a] - region_means[region_b]) < threshold:
              segmentation[segmentation == region_b] = region_a
      
   return segmentation

def split_and_merge(image, split_threshold, merge_threshold):
    """Perform region splitting and merging."""
    h, w = image.shape
    segmentation = np.zeros_like(image, dtype=int)

    # Perform quadtree splitting
    label_id = 1
    label_id = quadtree_split(image, 0, 0, min(h, w), split_threshold, segmentation, label_id)

    # Perform merging of regions
    segmentation = region_merge(segmentation, image, merge_threshold)

    return segmentation

# Example usage
if __name__ == "__main__":
    # Create a synthetic grayscale image
    image = np.array([[1, 1, 2, 2, 3],
                      [1, 1, 2, 3, 3],
                      [1, 2, 2, 3, 4],
                      [2, 2, 3, 4, 4],
                      [3, 3, 4, 4, 5]], dtype=float)

    # Define thresholds
    split_threshold = 1.5  # Variance threshold for splitting
    merge_threshold = 0.5  # Intensity difference threshold for merging

    # Apply region splitting and merging
    segmented_image = split_and_merge(image, split_threshold, merge_threshold)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray", interpolation="none")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image, cmap="tab20", interpolation="none")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
```

#### Output

Following is the output of
**Region Splitting and Merging**approach on a greyscale image −![Region Splitting and Merging Example](/scipy/images/region_splitting_and_merging.jpg)
## Watershed Algorithm

The
**Watershed Algorithm**is a segmentation technique in image processing that is inspired by the concept of watershed in topography. It views the image as a topographic surface where pixel intensities represent heights.
This algorithm simulates the process of flooding regions from seed points, with boundaries forming where waters from different seeds meet. The watershed algorithm is commonly used to segment touching or overlapping objects in an image especially in cases where other segmentation methods fail.

### Key Concepts of Watershed Algorithm

Here are the key concepts that you need to know before understanding the Watershed Algorithm −

- **Topographic Surface**represents the image where pixel intensity corresponds to the elevation or height of the surface.
- **Gradient Image**is derived from the original image and highlights boundaries between regions, essentially simulating the slopes of a topographic map.
- **Markers**are predefined regions or seed points that are used to guide the flood process. Markers are often placed manually or detected automatically.
- **Flooding Process**refers to the algorithm's approach of spreading the region from each marker until boundaries are formed.
- **Watershed Lines**are the boundaries formed where floods from different markers meet, segmenting the image into distinct regions.
### Steps in the Watershed Algorithm

The Watershed Algorithm follows a sequence of steps to segment an image. Below is a detailed breakdown −

- **Preprocessing**− Compute the gradient or edge map of the image to highlight the boundaries between regions.
- **Marker Initialization**− Mark the initial regions i.e., foreground and background, to define where the flooding process will begin.
- **Flooding**− Simulate the flooding of regions from each marker. The process continues until the flooded regions meet by forming watershed lines.
- **Segmentation**− The boundaries formed by the meeting of floods define distinct regions in the image.
### Example

Following is an example that shows how the Watershed Algorithm is applied to an image. In this example we compute the gradient, place markers and then apply the watershed algorithm using scipy −

```
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import data, filters, segmentation

# Step 1: Load the sample image (e.g., a synthetic image of coins)
image = data.coins()

# Step 2: Convert the image to grayscale (if it is not already)
# Skimage provides the image in grayscale, so we can skip this step.

# Step 3: Compute the gradient of the image using a Sobel filter
# The gradient will highlight the boundaries (edges) between regions
gradient = np.sqrt(ndi.sobel(image, axis=0)**2 + ndi.sobel(image, axis=1)**2)

# Step 4: Apply a threshold to create markers for the watershed algorithm
# Here we use a simple method: identifying regions with low gradient as markers
threshold = filters.threshold_otsu(gradient)
markers = gradient < threshold

# Step 5: Label markers (background and foreground)
# Background will be labeled as 0 and the foreground as 1
markers = ndi.label(markers)[0]

# Step 6: Perform the watershed transformation using the gradient image
# Watershed labels the regions based on the gradient and the markers
labels = segmentation.watershed(gradient, markers)

# Step 7: Visualize the results
# Plot original image, gradient, and segmented (labeled) result
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')

ax[1].imshow(gradient, cmap='hot')
ax[1].set_title('Gradient')

ax[2].imshow(labels, cmap='tab20b')
ax[2].set_title('Segmented (Watershed)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

Following is the output of
**Watershed Algorithm**applied to an image −![Watershed Algorithm Example](/scipy/images/watershed_example.jpg)

---

## 67. SciPy - Connected Component Labeling

*Source: [https://www.tutorialspoint.com/scipy/scipy_connected_component_labeling.htm](https://www.tutorialspoint.com/scipy/scipy_connected_component_labeling.htm)*

---

---
[Previous](/scipy/scipy_region_based_segmentation.htm)[Quiz](/scipy/quiz_on_scipy_connected_component_labeling.htm)[Next](/scipy/scipy_optimize.htm)**Connected Component Labeling**in SciPy refers to a technique often used in image processing and computer vision to identify and label connected regions or components in binary or gray-scale images.
This labeling helps in analyzing and processing distinct regions of an image such as identifying separate objects, blobs or clusters.

## Key Concepts

Following are the key concepts that we need to know before proceeding with the Connected Component Labeling in Image segmentation −

- **Binary Image**− Connected component labeling typically starts with a binary image where pixels are either foreground  as**1 or True**or background**0 or False**. Gray scale images may need thresholding to convert them to binary.
- **Connectivity**− Pixels are considered connected based on a predefined connectivity rule −
- **4-connectivity**− A pixel is connected to its immediate neighbors such as top, bottom, left, right.
- **8-connectivity**− A pixel is connected to its neighbors diagonally as well such as top-left, top-right, bottom-left, bottom-right.
- **Labels**− Each distinct connected region in the image is assigned a unique label.
## How SciPy Handles Connected Component Labeling?

In SciPy library, we have a
**scipy.ndimage.label()**function which is used to perform connected component labeling.
### Syntax

Following is the syntax of the function
**scipy.ndimage.label()**to perform Connected Component Labeling −
```
scipy.ndimage.label(input, structure=None, output=None)
```

Following are the parameters of the function
**scipy.ndimage.label()**−
- **input**− The binary input array which is a NumPy array.
- **structure(optional)**− Defines the connectivity of the elements i.e., pixels or voxels.
- **output (optional)**− A pre-allocated array where the labeled output will be stored.
## Basic 4-connectivity

### Example
**4-connectivity**refers to a method of defining how pixels in a 2D grid are considered "connected" to each other. In 4-connectivity, a pixel is connected to its immediate horizontal and vertical neighbors i.e., top, bottom, left and right.
```
import numpy as np
from scipy.ndimage import label

# Input binary image
binary_image = np.array([[0, 1, 1, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 1, 1, 0]])

# Perform labeling with default (4-connectivity)
labeled_array, num_features = label(binary_image)

print("Labeled Array:")
print(labeled_array)
print("Number of Features:", num_features)
```

#### Output

Following is the output of the function
**scipy.ndimage.label()**which is used to perform 4-Connectivity −
```
Labeled Array:
[[0 1 1 0]
 [1 1 0 0]
 [0 0 2 2]
 [0 2 2 0]]
Number of Features: 2
```

## Using 8-Connectivity

### Example

In
**8-connectivity**pixels are considered connected if they share an edge or a corner with the current pixel. This includes all 8 neighbors in a 2D grid such as top, bottom, left, right and the four diagonal neighbors.
```
import numpy as np
from scipy.ndimage import label

# Input binary image
binary_image = np.array([[0, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 1, 0]])

# Define an 8-connectivity structure
structure = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])

# Perform connected component labeling
labeled_array, num_features = label(binary_image, structure=structure)

print("Labeled Array:")
print(labeled_array)
print("Number of Features:", num_features)
```

#### Output

Following is the output of the function
**scipy.ndimage.label()**which is used to perform 4-Connectivity −
```
Labeled Array:
[[0 1 0 0]
 [1 1 0 0]
 [0 0 1 1]
 [0 0 1 0]]
Number of Features: 1
```

## 4-Connectivity Vs 8-Connectivity

The primary difference between 4-connectivity and 8-connectivity lies in how they define the neighborhood of a pixel in an image and determine which pixels are considered "connected" to each other.
Aspect4-Connectivity8-Connectivity**Connected Neighbors**Only horizontal and vertical neighbors (top, bottom, left, right).Includes horizontal, vertical, and diagonal neighbors (top-left, top-right, bottom-left, bottom-right).**Total Number of Neighbors**4 (top, bottom, left, right)8 (including diagonal neighbors)**Connectivity Behavior**Only pixels sharing edges (not corners) are connected.Pixels sharing edges or corners are connected.**Resulting Regions**May result in more disconnected components, as diagonal connections are ignored.May result in fewer disconnected components, as diagonal connections are considered.

---

## 68. SciPy - Optimize

*Source: [https://www.tutorialspoint.com/scipy/scipy_optimize.htm](https://www.tutorialspoint.com/scipy/scipy_optimize.htm)*

---

---
[Previous](/scipy/scipy_connected_component_labeling.htm)[Quiz](/scipy/quiz_on_scipy_optimize.htm)[Next](/scipy/scipy_special_matrices_functions.htm)
## SciPy Optimize
**SciPy's optimize**module is a collection of tools for solving mathematical optimization problems. It helps minimize or maximize functions, find function roots, and fit models to data. This makes it useful for tasks like data analysis, engineering, and scientific research.
The scipy.optimize package provides several commonly used optimization algorithms. This module contains the following aspects −

- 
Unconstrained and constrained minimization of multivariate scalar functions using minimize() function. Supports various algorithms like BFGS, Nelder-Mead simplex, Newton Conjugate Gradient, COBYLA or SLSQP

- 
Global (Brute-Force) Optimization Routines. Examples: anneal(), basinhopping()

- 
Least-Squares Minimization and Curve Fitting. Examples: leastsq()) and curve fitting (curve_fit()) algorithms

- 
Scalar univariate functions minimizers. Examples: minimize_scalar()) and root finders − newton()

- 
Multivariate equation system solvers using root() function. Examples: Hybrid Powell, Levenberg-Marquardt or large-scale methods such as Newton-Krylov

### Unconstrained and Constrained Minimization

The minimize() function provides a common interface to unconstrained and constrained minimization algorithms for multivariate scalar functions in scipy.optimize. To demonstrate the minimization function, consider the problem of minimizing the Rosenbrock function of the NN variables −

We can use scipy.optimize.minimize() function to minimize the above Rosenbrock function

The minimize() function takes the following arguments:

- **fun**: The objective function that you want to be minimize in our case it is Rosenbrock function.
- **x(0)**: The x0 argument is an array-like structure that represents the initial guess for the variables
- **method**: Which optimization algorithm to use. Supporting algorithms include.
- **Unconstrained**: Nelder-Mead, BFGS( Quasi-Newton method), CG(Conjugate Gradient).
- **Constrained**: L-BFGS-B, TNC, trust-constr
By default the method is BFGS.

Let us minimize the Rosenbrock function using the Nelder-Mead simplex algorithm (method = 'Nelder-Mead')

```
import numpy as np
from scipy.optimize import minimize
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosenbrock, x0, method='nelder-mead')

print(res.x)
```

The above program will generate the following output.

```
[0.99910115 0.99820923 0.99646346 0.99297555 0.98600385]
```

The minimum value of this function is 0, which is achieved when xi = 1.

## Global (Brute-Force) Optimization Routines

The basinhopping algorithm is a global optimization technique that successfully avoids local minima while locating the global minimum of complex multimodal functions by combining random perturbations with local minimizations.

Let us minimize the quadratic equation using the basinhopping algorithm −

```
import numpy as np
from scipy.optimize import basinhopping
def quadratic(x):
    return (x - 3)**2 + 1

# Initial guess
x0 = np.array([0])

res = basinhopping(quadratic, x0)

# Print the result
print("global minimum:", res.x)
```

The above program will generate the following output.

```
global minimum: [2.99999999]
```

## Least Squares Minimization

Solve a nonlinear least-squares problem with bounds on the variables. Given the residuals f(x) (an m-dimensional real function of n real variables) and the loss function rho(s) (a scalar function), least_squares find a local minimum of the cost function F(x). Let us consider the following example.

In this example, we find a minimum of the Rosenbrock function without bounds on the independent variables.

```
import numpy as np
def fun_rosenbrock(x):
   return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
   
from scipy.optimize import least_squares
input = np.array([2, 2])
res = least_squares(fun_rosenbrock, input)
print(res)
```

Notice that, we only provide the vector of the residuals. The algorithm constructs the cost function as a sum of squares of the residuals, which gives the Rosenbrock function. The exact minimum is at x = [1.0,1.0].

Following is the output of the above code −

```
message: `gtol` termination condition is satisfied.
     success: True
      status: 1
         fun: [ 4.441e-15  1.110e-16]
           x: [ 1.000e+00  1.000e+00]
        cost: 9.866924291084687e-30
         jac: [[-2.000e+01  1.000e+01]
               [-1.000e+00  0.000e+00]]
        grad: [-8.893e-14  4.441e-14]
  optimality: 8.892886493421953e-14
 active_mask: [ 0.000e+00  0.000e+00]
        nfev: 3
        njev: 3
```

## Root Finding

Let us understand how root finding helps in SciPy.

### Scalar Functions

If one has a single-variable equation, there are four different root-finding algorithms, which can be tried. Each of these algorithms require the endpoints of an interval in which a root is expected (because the function changes signs). In general, brentq is the best choice, but the other methods may be useful in certain circumstances or for academic purposes.

### Fixed-point solving

A problem closely related to finding the zeros of a function is the problem of finding a fixed point of a function. A fixed point of a function is the point at which evaluation of the function returns the point: g(x) = x. Clearly the fixed point of gg is the root of f(x) = g(x)x. Equivalently, the root of ff is the fixed_point of g(x) = f(x)+x. The routine fixed_point provides a simple iterative method using the Aitkens sequence acceleration to estimate the fixed point of gg, if a starting point is given.

### Sets of equations

Finding a root of a set of non-linear equations can be achieved using the root() function. Several methods are available, amongst which hybr (the default) and lm, respectively use the hybrid method of Powell and the Levenberg-Marquardt method from the MINPACK.

The following example considers the single-variable transcendental equation.

```
x2 + 2cos(x) = 0
```
**x**+ 2cos(x) = 0
A root of which can be found as follows −

```
import numpy as np
from scipy.optimize import root
def func(x):
   return x*2 + 2 * np.cos(x)
sol = root(func, 0.3)
print(sol)
```

The above program will generate the following output.

```
message: The solution converged.
 success: True
  status: 1
     fun: [ 2.220e-16]
       x: [-7.391e-01]
    nfev: 10
    fjac: [[-1.000e+00]]
       r: [-3.347e+00]
     qtf: [-2.777e-12]
```

## Multivariate Equation System Solvers

The root() function in scipy.optimize solves multivariate equations. Here, we solve a system of nonlinear equations using the Hybrid Powell method.

Consider the following equations −

```
x2 + y2 - 4  = 0
x * y - 1 = 0
```
**x**+ y- 4  = 0**x * y - 1 = 0**
Following is an example −

```
import numpy as np
from scipy.optimize import root

def equations(vars):
    x, y = vars
    return [x**2 + y**2 - 4, x * y - 1]

x0 = [1.5, 1.5]  # Initial guess
res = root(equations, x0, method='hybr')

print(res.x)
```

The above program will generate the following output.

```
[1.93185165 0.51763809]
```

---

## 69. SciPy - Special Matrices and Functions

*Source: [https://www.tutorialspoint.com/scipy/scipy_special_matrices_functions.htm](https://www.tutorialspoint.com/scipy/scipy_special_matrices_functions.htm)*

---

---

## 70. SciPy - Unconstrained Optimization

*Source: [https://www.tutorialspoint.com/scipy/scipy_unconstrained_optimization.htm](https://www.tutorialspoint.com/scipy/scipy_unconstrained_optimization.htm)*

---

---
[Previous](/scipy/scipy_special_matrices_functions.htm)[Quiz](/scipy/quiz_on_scipy_unconstrained_optimization.htm)[Next](/scipy/scipy_constrained_optimization.htm)
## Unconstrained Optimization in SciPy
**Unconstrained optimization**in SciPy refers to the process of finding the minimum or maximum of an objective function without any restrictions or constraints on the variables. Unconstrained optimization is typically performed using the**scipy.optimize.minimize()**function which provides a wide range of algorithms suited to different types of optimization problems. Lets go into detail about how this works along with different aspects.
### Syntax

Following is the syntax of the function
**scipy.optimize.minimize()**which is used to find the minimum or maximum of an objective function −
```
scipy.optimize.minimize(
   fun, 
   x0, 
   args=(), 
   method=None, 
   jac=None, 
   hess=None, 
   hessp=None, 
   bounds=None, 
   constraints=(), 
   tol=None, 
   callback=None, 
   options=None
)
```

### Parameters

Here are the Parameters of
**scipy.optimize.minimize()**function −
- **fun**− Objective function to minimize
- **x0**− Initial guess for the variables.
- **method**− Optimization algorithm. There are different methods such as 'BFGS', 'Nelder-Mead', 'L-BFGS-B', 'trust-constr'.
- **jac(optional)**− Gradient of objetcive function.
- **hess(optional)**− Hessian of objective function.
- **bounds**− Variable bounds for constrained problems
- **constraints**− Equality or inequality constraints.
- **options**− Solver-specific settings.
## Basic Minimization

### Example

Following is the basic Minimization example in which we will minimize a simple quadratic function
**f(x) = x**−+x+2
```
from scipy.optimize import minimize

# Objective function
def objective(x):
    return x**2 + x + 2

# Initial guess
x0 = [0]

# Minimize the function
result = minimize(objective, x0)

# Display results
print("Optimal solution:", result.x)
print("Function value at optimum:", result.fun)
```

#### Output

Here is the output of the basic minimization by using the function
**scipy.optimize.minimize()**−
```
Optimal solution: [-0.50000001]
Function value at optimum: 1.75
```

## Minimization with Variable Bounds

### Example

When variables have specific bounds we can define them using the bounds parameter in
**scipy.optimize.minimize()**function. This is useful for constrained problems. The following example shows how to minimize a function with variable bounds −
```
from scipy.optimize import minimize

# Objective function
def objective(x):
    return x[0]**2 + x[1]**2

# Bounds on the variables
bounds = [(0, 1), (-1, 1)]  # x in [0, 1], y in [-1, 1]

# Initial guess
x0 = [0.5, 0]  # A point within the bounds

# Minimization
result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

# Output
print("Optimal solution:", result.x)
print("Function value at optimum:", result.fun)
```

#### Output

Here is the output of the function
**scipy.optimize.minimize()**used with bounds parameter −
```
Optimal solution: [ 0.00000000e+00 -1.11022301e-08]
Function value at optimum: 1.2325951233541654e-16
```

## Optimization methods in minimize() Function

The
**scipy.optimize.minimize()**function supports a variety of optimization methods in which each tailored for specific types of problems such as unconstrained or constrained optimization, gradient-based or gradient-free methods and problems with bounds.MethodDescriptionTypeGradient Needed?Hessian Needed?Use Case**Nelder-Mead**Simplex algorithm that minimizes based only on function values.Gradient-freeNoNoNon-smooth functions, small problems**Powell**Directional search algorithm optimizing along chosen directions.Gradient-freeNoNoNon-smooth functions, high dimensions**CG**Conjugate Gradient method minimizing quadratic approximation of functions.Gradient-basedYesNoSmooth functions, large-scale problems**BFGS**Quasi-Newton method approximating the inverse Hessian.Gradient-basedYesApproximationSmooth functions, efficient optimization**Newton-CG**Newtons method with conjugate gradient to improve efficiency.Gradient-basedYesOptionalLarge-scale problems with Hessian-vector products**trust-ncg**Trust-region Newton-Conjugate Gradient method.Trust-region, Newton'sYesApproximationLarge-scale problems**trust-krylov**Trust-region method using Krylov subspace approximation of Hessians.Trust-region, Krylov-basedYesApproximationLarge-scale problems**trust-exact**Trust-region method leveraging exact Hessians for precise solutions.Trust-region, Newton'sYesYesSmall-scale problems with exact Hessians**dogleg**Trust-region method that uses a dogleg step for solving sub-problems.Trust-region, Newton'sYesYesMedium-scale problems with exact Hessians

---

## 71. SciPy - Constrained Optimization

*Source: [https://www.tutorialspoint.com/scipy/scipy_constrained_optimization.htm](https://www.tutorialspoint.com/scipy/scipy_constrained_optimization.htm)*

---

---
[Previous](/scipy/scipy_unconstrained_optimization.htm)[Quiz](/scipy/quiz_on_scipy_constrained_optimization.htm)[Next](/scipy/scipy_matrix_norms.htm)
The SciPy
**Constrained optimization**involves finding the optimal value of an objective function**f(x)**subject to constraints. These constraints can be equality constraints as**h(x)=0**, inequality constraints as**g(x)0)**or simple 
bounds**lxu**.
SciPy's
**scipy.optimize**module provides powerful tools for solving constrained optimization problems. In this chapter we are going to see in detail, how the**Constrained Optimization**works.
## Components of a Constrained Optimization

A constrained optimization problem consists of several key components that define its structure and behavior. These components are as follows −

- **Objective Function**− This is the scalar function to minimize or maximize.
- **Decision Variables**− These are the variables that are optimized to achieve the objective.
- **Equality Constraints**− These are the conditions that must be satisfied exactly.
- **Inequality Constraints**− These are the conditions that impose upper or lower limits.
- **Bounds**− They are the Lower and upper limits for individual variables.
- **Feasible Region**− This is the region where set of all points that satisfy the constraints.
- **Optimization Direction**− This indicates whether to minimize or maximize the constrained optimization.
## Optimization methods in SciPy

SciPy provides several optimization methods through the
**scipy.optimize**module by catering to various problem types which include unconstrained, constrained and large-scale optimization. Heres a list of commonly used optimization methods and the function that we can use to atchieve it in SciPy −S.NoOptimization MethodFunction and Description1Sequential Least Squares Programming (SLSQP)[scipy.optimize.minimize()](/scipy/scipy_optimize_minimize_function.htm)
A method for solving small- to medium-scale nonlinear constrained problems.2Linear Programming[scipy.optimize.linprog()](/scipy/scipy_optimize_linprog_function.htm)
Used for solving linear objective functions with linear constraints.3Nonlinear Least Squares[scipy.optimize.least_squares()](/scipy/scipy_optimize_least_squares_function.htm)
Suitable for curve fitting and parameter estimation under bounds.4Root Finding (Nonlinear Equations)[scipy.optimize.root()](/scipy/scipy_optimize_root_function.htm)
Used for solving systems of nonlinear equations with equality constraints.5Simulated Annealing[scipy.optimize.dual_annealing()](/scipy/scipy_optimize_dual_annealing_function.htm)
A global optimization technique for solving non-convex problems under bounds.
## Applications of Constrained Optimization

Constrained optimization is widely used across various fields to solve real-world problems where certain conditions or limitations must be satisfied. Below are some of the main applications −

- **Engineering Design**− We can use the Constrained optimization for structural and mechanical designs.
- **Economics**− Used in Portfolio optimization with constraints on risk and return.
- **Machine Learning**− Used in regularization in training models with bounded parameters.
### Example

Let's consider a business optimization problem where a company wants to produce two products with limited resources such as labor and materials. The goal is to maximize the profit which subject to constraints on the available resources. In this example we are going to solve this problem by using constrained optimization techniques such as Linear Programming (LP) with the help of
**scipy.optimize.linprog()**function −
```
import numpy as np
from scipy.optimize import linprog

# Coefficients of the objective function (profit from A and B)
c = [-5, -4]  # Negative because we are maximizing

# Coefficients of the inequality constraints (Labor and Material)
A = [[2, 4], [3, 2]]

# Right-hand side values of the constraints (100 labor, 80 material)
b = [100, 80]

# Bounds for the variables (Product A and B cannot be negative)
x_bounds = (0, None)
y_bounds = (0, None)

# Solve the optimization problem
result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='highs')

# Output the result
print("Optimal number of Product A:", result.x[0])
print("Optimal number of Product B:", result.x[1])
print("Maximum Profit:", -result.fun)
```

#### Output

Following is the output of the Constrained optimization performed using the function
**scipy.optimize.linprog()**−
```
Optimal number of Product A: 15.0
Optimal number of Product B: 17.5
Maximum Profit: 145.0
```

---

## 72. SciPy - Matrix Norms

*Source: [https://www.tutorialspoint.com/scipy/scipy_matrix_norms.htm](https://www.tutorialspoint.com/scipy/scipy_matrix_norms.htm)*

---

---
[Previous](/scipy/scipy_constrained_optimization.htm)[Quiz](/scipy/quiz_on_scipy_matrix_norms.htm)[Next](/scipy/scipy_sparse_matrix.htm)
## What are Matrix Norms?
**SciPy Matrix norms**are mathematical functions that assign a non-negative scalar to a matrix by providing a measure of the size or**magnitude**of the matrix. These norms are widely used in linear algebra, optimization and numerical analysis for tasks such as measuring distances between matrices, solving linear systems and evaluating the condition of a matrix.
SciPy provides several functions to compute different types of matrix norms in the
**scipy.linalg**and**scipy.sparse.linalg**modules. The most commonly used norms are Frobenius norm, L2 norm and L1 norm.
## What are Sparse Matrices?

Sparse matrices are matrices that are primarily composed of zeros. These types of matrices are common in fields such as large-scale scientific computing, machine learning and graph theory where the matrix size may be very large but only a small proportion of the elements are non-zero.

SciPy provides efficient ways to work with sparse matrices through its
**scipy.sparse**module. When dealing with sparse matrices for computing matrix norms can be crucial for tasks such as optimization, regularization and evaluating the quality of matrix approximations.
## Common Norms available in Scipy

In SciPy both matrix norms and sparse matrix norms are available through the
**scipy.linalg.norm**and**scipy.sparse**modules. Below is the list of the available norms for matrices and sparse matrices in SciPy −S.NoNorm & Description1Frobenius Norm (fro)
Square root of the sum of the absolute squares of the matrix elements.2L2 Norm (Spectral Norm
Largest singular value of the matrix.3L1 Norm
Sum of the absolute values of the matrix elements.4Max Norm (Infinity Norm)
Maximum absolute row sum of the matrix.5L0 Norm
Count of non-zero elements in the matrix.6Nuclear Norm (Trace Norm)
Sum of the singular values of the matrix.7Operator Norm
Largest eigenvalue in absolute value of the matrix.8p-Norm
Generalization where the sum of absolute values raised to the power p is taken.9Sparse Matrix Frobenius Norm
Frobenius norm for sparse matrices, calculated using non-zero elements.10Sparse L2 Norm
Largest singular value for sparse matrices.11Sparse L1 Norm
Sum of absolute values of non-zero elements in the sparse matrix.12Sparse Max Norm
Maximum absolute row sum in sparse matrices.13Sparse L0 Norm
Count of non-zero elements in a sparse matrix.
## Applications of Scipy Norms

As We discussed above sciPy provides a wide range of norms through the
**scipy.linalg.norm()**and**scipy.sparse.linalg.norm()**functions which are used to evaluate the magnitude or size of matrices or vectors. These norms are crucial in various mathematical, computational and engineering fields. Below are some of the key applications of SciPy norms −
- **Optimization:**Norms like L1 and L2 are used in optimization problems such as Lasso and Ridge regression to regularize models which helps to prevent overfitting and promoting sparsity.
- **Machine Learning:**Norms measure distances between data points in algorithms like k-NN and SVM where the Euclidean (L2) norm is often used for classification and clustering.
- **Signal Processing:**Norms assess errors in signal reconstruction by using the Frobenius norm and help in filter design and compressed sensing.
- **Numerical Linear Algebra:**Norms are used to determine matrix conditioning by indicating the sensitivity of a matrix to computational errors, essential in solving linear systems.
- **Data Analysis and Clustering:**In clustering algorithms like k-means, norms especially L2 calculate distances between points and centroids to group data.
- **Control Theory:**Norms such as the L2 norm are used to evaluate system performance and stability in control systems, especially for tracking and energy analysis.
- **Graph Theory:**Norms such as the Frobenius norm are applied to analyze the structure of graphs by aiding in algorithms for graph traversal and spectral clustering.

---

## 73. SciPy - Sparse Matrix

*Source: [https://www.tutorialspoint.com/scipy/scipy_sparse_matrix.htm](https://www.tutorialspoint.com/scipy/scipy_sparse_matrix.htm)*

---

---

## 74. SciPy - Frobenius Norm(fro)

*Source: [https://www.tutorialspoint.com/scipy/scipy_frobenius_norm.htm](https://www.tutorialspoint.com/scipy/scipy_frobenius_norm.htm)*

---

---

## 75. SciPy - L2 Norm(Spectral Norm)

*Source: [https://www.tutorialspoint.com/scipy/scipy_spectral_norm.htm](https://www.tutorialspoint.com/scipy/scipy_spectral_norm.htm)*

---

---
[Previous](/scipy/scipy_frobenius_norm.htm)[Quiz](/scipy/quiz_on_scipy_spectral_norm.htm)[Next](/scipy/scipy_condition_numbers.htm)
## Spectral Norm in SciPy (L2)

The
**L2 Norm**(also known as the spectral norm) in SciPy is a matrix norm that corresponds to the**largest singular value**of the matrix. It is widely used in various fields such as numerical optimization, machine learning, and stability analysis due to its significance in measuring the largest scaling factor that the matrix can apply to any vector.
The L2 norm of a matrix A quantifies the maximum amount by which the matrix stretches a vector. In essence, it provides a measure of how much the matrix "expands" vectors in its domain. The norm is equal to the largest singular value of A, which can be found by computing the singular value decomposition (SVD) of the matrix.

If a matrix A has dimensions
**m x n**, the mathematical formula for the**L2 Norm**(Spectral Norm)**||A||**is:
$&bsol;mathrm{|A|_2 = σ^{max}}$

Where −

- **σ**denotes the largest singular value of matrix A.
- 
The spectral norm is the largest eigenvalue of the matrix
**A**whereA**A**is the transpose of A.
## Properties of the L2 Norm (Spectral Norm)

The L2 norm has a number of useful mathematical properties that make it important in many applications, especially in numerical analysis. These are some key properties −

### Non-Negativity

The L2 norm is always non-negative and it is zero if and only if the matrix is a zero matrix i.e., all elements of A are zero.

```
A2  0
```
0
### Homogeneity (Scaling)

The L2 norm is homogeneous with respect to scalar multiplication. That is, scaling a matrix by a constant factor scales its norm by the absolute value of the constant.

```
cA2 = |c|  A2
```
= |c|  A
where
**c**is a scalar, and**A**is a matrix. This means that multiplying a matrix by a constant factor scales its L2 norm by the absolute value of that constant.
### Subadditivity (Triangle Inequality)

The L2 norm satisfies the triangle inequality which means the norm of the sum of two matrices is less than or equal to the sum of their norms.

```
A + B2  A2 + B2
```
A+ B
for matrices A and B of the same size. This property ensures that the L2 norm behaves like a distance metric.

### Invariance under Orthogonal/Unitary Transformations

The L2 norm remains unchanged when the matrix is multiplied by an orthogonal (real) or unitary (complex) matrix.

```
UA2 = A2 and AV2 = A2
```
= Aand AV= A
where
**U**and**V**are orthogonal or unitary matrices. This property is significant in many numerical methods such as Singular Value Decomposition (SVD) where transformations do not alter the L2 norm.
### Example 1

The following example demonstrates how to compute the
**L2 Norm**(Spectral Norm) using SciPy's**scipy.linalg.norm()**function by passing the argument**ord=2**. We can calculate the L2 Norm for a small 2D matrix −
```
import numpy as np
from scipy.linalg import norm

# Define the matrix
A = np.array([[1, 2],
              [3, 4]])

# Compute the L2 norm (spectral norm)
l2_norm = norm(A, ord=2)

print("Matrix A:")
print(A)
print("L2 Norm (Spectral Norm) of A:", l2_norm)
```

#### Output

Following is the output for the 2D matrix's L2 Norm −

```
Matrix A:
[[1 2]
 [3 4]]
L2 Norm (Spectral Norm) of A: 5.464985704219043
```

### Example 2

In this example we calculate the L2 Norm or Spectral Norm for a complex matrix using the
**scipy.linalg.norm()**function −
```
import numpy as np
from scipy.linalg import norm

# Define the complex matrix
A = np.array([[1 + 1j, 2 - 1j],
              [-1j, 3 + 2j]])

# Compute the L2 norm (spectral norm)
l2_norm = norm(A, ord=2)

print("Complex Matrix A:")
print(A)
print("L2 Norm (Spectral Norm) of A:", l2_norm)
```

#### Output

Heres the result of the L2 Norm (Spectral Norm) for a complex matrix −

```
Complex Matrix A:
[[ 1.+1.j  2.-1.j]
 [-0.-1.j  3.+2.j]]
L2 Norm (Spectral Norm) of A: 4.25045561972017
```

### Example 3

Here in this example let's calculate the L2 Norm (Spectral Norm) for a zero matrix −

```
import numpy as np
from scipy.linalg import norm

# Define the zero matrix
A = np.zeros((2, 2))

# Compute the L2 norm (spectral norm)
l2_norm = norm(A, ord=2)

print("Zero Matrix A:")
print(A)
print("L2 Norm (Spectral Norm) of A:", l2_norm)
```

#### Output

Here is the output for the L2 Norm (Spectral Norm) of a zero matrix −

```
Zero Matrix A:
[[0. 0.]
 [0. 0.]]
L2 Norm (Spectral Norm) of A: 0.0
```

---

## 76. SciPy - Condition Numbers

*Source: [https://www.tutorialspoint.com/scipy/scipy_condition_numbers.htm](https://www.tutorialspoint.com/scipy/scipy_condition_numbers.htm)*

---

---
[Previous](/scipy/scipy_spectral_norm.htm)[Quiz](/scipy/quiz_on_scipy_condition_numbers.htm)[Next](/scipy/scipy_linear_least_squares.htm)
In SciPy the
**Condition Number**quantifies the sensitivity of a matrix to small changes in its input or its elements. It is a key metric for assessing the numerical stability of linear systems, eigenvalue problems and matrix computations. The condition number of a matrix A is defined as follows −
```
κ(A) = &Vert; A &Vert; &cdot; &Vert; A-1 &Vert;
```
&Vert;
where A is a matrix norm e.g., 2-norm, Frobenius norm and A
is the norm of the inverse matrix. A higher condition number indicates that the matrix is ill-conditioned which means small changes in the input may cause large errors in the output. For singular matrices, the condition number is infinite because the inverse does not exist.
## Key Concepts

Following are the key concepts of the Conditional Numbers −

- **Condition number:**A value that indicates the sensitivity of the matrix. A large condition number e.g., > 10indicates that the matrix is ill-conditioned and small errors in the matrix or input could lead to large errors in the computed solution.
- **Norm:**The condition number depends on the norm used. Common norms include the 2-norm (spectral norm), 1-norm and Frobenius norm.
## Interpretation of Condition Numbers

Here are the Interpretations of the Condition Numbers in Scipy −

- **Small Condition Number (close to 1)**− The matrix is well-conditioned which means that small changes in the input data result in small changes in the output. The system of equations is numerically stable.
- **Large Condition Number**− The matrix is ill-conditioned. Small changes or errors in the input can lead to large errors in the solution. This typically occurs when the matrix is close to singular which means it does not have a full rank or has nearly zero singular values.
### Example 1

The condition number of a matrix is the ratio of the largest singular value to the smallest singular value. As we don't have a predefined function to implement the condition number, we can use the
**scipy.linalg.svd()**function to implement the computation of the condition number manually. Here is the example of implementing the condition number −
```
import numpy as np
from scipy.linalg import svd

# Example matrix (well-conditioned)
A_well = np.array([[2, 1], [1, 3]])

# Compute Singular Value Decomposition (SVD)
U, s, Vh = svd(A_well)

# Compute the condition number: max singular value / min singular value
condition_number = s[0] / s[-1]

print(f"Condition number (well-conditioned matrix): {condition_number}")

# Example matrix (ill-conditioned)
A_ill = np.array([[1, 2], [2, 4]])

# Compute Singular Value Decomposition (SVD)
U, s, Vh = svd(A_ill)

# Compute the condition number
condition_number_ill = s[0] / s[-1]

print(f"Condition number (ill-conditioned matrix): {condition_number_ill}")
```

#### Output

Here is the output of the function
**scipy.linalg.svd()**which is used to implement the condition number −
```
Condition number (well-conditioned matrix): 2.6180339887498953
Condition number (ill-conditioned matrix): 2.5175887275607884e+16
```

### Example 2

Here is the example of the well-conditioned and ill-conditioned Matrices together implemented by using the function
**scipy.linalg.svd()**−
```
import numpy as np
from scipy.linalg import svd

# Well-conditioned matrix
A_well = np.array([[2, 1], [1, 3]])

# Ill-conditioned matrix
A_ill = np.array([[1, 2], [2, 4]])

# Compute singular values and condition numbers for both matrices

def compute_condition_number(A):
    # Compute singular values using SVD
    U, S, Vt = svd(A)
    # Condition number is the ratio of the largest to the smallest singular value
    return S[0] / S[-1]

# Condition numbers
condition_number_well = compute_condition_number(A_well)
condition_number_ill = compute_condition_number(A_ill)

print(f"Singular values (well-conditioned matrix): {svd(A_well)[1]}")
print(f"Condition number (well-conditioned matrix): {condition_number_well}")

print(f"Singular values (ill-conditioned matrix): {svd(A_ill)[1]}")
print(f"Condition number (ill-conditioned matrix): {condition_number_ill}")
```

#### Output

Here is the output of the function
**scipy.linalg.svd()**which is used to implement the well-conditioned and ill-conditioned matrices −
```
Singular values (well-conditioned matrix): [3.61803399 1.38196601]
Condition number (well-conditioned matrix): 2.6180339887498953
Singular values (ill-conditioned matrix): [5.00000000e+00 1.98602732e-16]
Condition number (ill-conditioned matrix): 2.5175887275607884e+16
```

## Condition Number and Solving Linear Systems

When solving a system of linear equations
**Ax=b**, the condition number (A) of the matrix A gives an estimate of how much the solution**x**will change in response to small changes in the matrix**A**or the vector**b**.
- 
If the condition number is low, then solving the system is stable and small changes in
**A**or**b**will only cause small changes in**x**.
- 
If the condition number is high then the solution can be highly sensitive to errors. Small changes in
**A**or**b**could result in large changes in**x**.

---

## 77. SciPy - Linear Least Squares

*Source: [https://www.tutorialspoint.com/scipy/scipy_linear_least_squares.htm](https://www.tutorialspoint.com/scipy/scipy_linear_least_squares.htm)*

---

---
[Previous](/scipy/scipy_condition_numbers.htm)[Quiz](/scipy/quiz_on_scipy_linear_least_squares.htm)[Next](/scipy/scipy_non_linear_least_squares.htm)
The
**linear least squares (LLS)**is a method for finding the best approximation to an over-determined system of linear equations. An over-determined system has more equations than unknowns and is typically inconsistent, so an exact solution does not exist.
Mathematical equation of the Linear Least squares given for a system
**Ax=b**by minimize the residual is given as follows −
```
Axb22
```

Where A is an mn matrix (mn), b is an m-dimensional vector, x is an n-dimensional vector.

## Implementation of Linear Least Squares

SciPy provides several tools to solve the linear least-squares problem efficiently. The primary function for implementing the Linear Least Squares is with the function
**scipy.linalg.lstsq()**.
This function solves the least-squares problem using Singular Value Decomposition (SVD) internally by ensuring numerical stability and robustness. It works for both full-rank and rank-deficient matrices.

### How Linear Least Squares Method Works

Here are the steps how the Linear Least squares works in scipy −

- **Normal Equations**− These equations while theoretically valid, solving via normal equations can lead to numerical instability. The equation can be given as −
```
ATAx = ATb
```
Ax = Ab
- **Singular Value Decomposition (SVD)**− The function internally uses SVD to decompose A −
```
A = UVT
```

The least-squares solution is computed as −

```
x = V-1UTb
```
Ub
The above approach avoids the numerical issues of normal equations.

### Syntax

Following is the syntax for using the function
**scipy.linalg.lstsq()**−
```
scipy.linalg.lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None)
```

### Parameters

Here are the parameters of the function
**scipy.linalg.lstsq()**−
- **a**− The**m x n**coefficient matrix A.
- **b**− The m-dimensional vector or matrix with multiple columns for multiple right-hand sides.
- **cond(optional)**− Cutoff for small singular values. Singular values smaller than cond * max(singular_values) are treated as zero.
- **overwrite_a**− If True then overwrites matrix A to save memory.
- **overwrite_b**− If True then overwrites vector b to save memory.
- **check_finite**− If True then checks that input matrices contain only finite numbers.
- **lapack_driver**− This specifies the LAPACK driver used to solve the problem. Options include gelsd, gelsy or gelss.
## Over-determined System (m>n)

An over-determined system occurs when there are more equations than unknowns (m>n) by making the system likely inconsistent. The linear least squares method finds the solution x that minimizes the residual i.e., the sum of squared differences between the observed values (b) and the values predicted by the model (Ax).

### Example

Following is the example which helps us to find the best fit solution for the given overdetermined system  with the help of the function
**scipy.linalg.lstsq()**−
```
import numpy as np
from scipy.linalg import lstsq

# Define the overdetermined system
A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([1, 2, 2])

# Solve using scipy.linalg.lstsq
x, residuals, rank, singular_values = lstsq(A, b)

# Output results
print("Matrix A:")
print(A)
print("\nVector b:")
print(b)
print("\nSolution (x):", x)
print("Residuals:", residuals)
print("Rank of A:", rank)
print("Singular values of A:", singular_values)
```

#### Output

Following is the output of the function
**scipy.linalg.lstsq()**function which is used for overdetermined system −
```
Matrix A:
[[1 1]
 [1 2]
 [1 3]]

Vector b:
[1 2 2]

Solution (x): [0.66666667 0.5       ]
Residuals: 0.16666666666666677
Rank of A: 2
Singular values of A: [4.07914333 0.60049122]
```

## Underdetermined System (m<n)

An underdetermined system occurs when there are fewer equations than unknowns (m<n). This means the system has infinitely many solutions as there are not enough constraints to uniquely determine all variables. In such cases the least-squares solution minimizes the residual and provides the solution with the smallest Euclidean norm.

### Example

Following is the example which helps us to find the best fit solution for the given underdetermined system with the help of the function
**scipy.linalg.lstsq()**−
```
import numpy as np
from scipy.linalg import lstsq

# Define the underdetermined system
A = np.array([[1, 1, 1], [1, 2, 3]])
b = np.array([1, 2])

# Solve using scipy.linalg.lstsq
x, residuals, rank, singular_values = lstsq(A, b)

# Output results
print("Matrix A:")
print(A)
print("\nVector b:")
print(b)
print("\nSolution (x):", x)
print("Residuals:", residuals)
print("Rank of A:", rank)
print("Singular values of A:", singular_values)
```

#### Output

Here is the output of the function
**scipy.linalg.lstsq()**function which is used for underdetermined system −
```
Matrix A:
[[1 1 1]
 [1 2 3]]

Vector b:
[1 2]

Solution (x): [0.33333333 0.33333333 0.33333333]
Residuals: []
Rank of A: 2
Singular values of A: [4.07914333 0.60049122]
```

---

## 78. SciPy - Non-Linear Least Squares

*Source: [https://www.tutorialspoint.com/scipy/scipy_non_linear_least_squares.htm](https://www.tutorialspoint.com/scipy/scipy_non_linear_least_squares.htm)*

---

---
[Previous](/scipy/scipy_linear_least_squares.htm)[Quiz](/scipy/quiz_on_scipy_non_linear_least_squares.htm)[Next](/scipy/scipy_finding_roots_of_scalar_functions.htm)
The
**Non-Linear least squares (NLLS)**is a method for fitting a model to data where the model's parameters are non-linear. It minimizes the sum of squared residuals between the observed values and the model's predictions.
Mathematical equation of Non-Linear Least Squares for a set of residuals
**r(x)**given a model**f(x, t)**is given as follows −
```
r(x)22
```

Where r(x) is the vector of residuals r_i(x) = y_i - f(x, t_i) and x is the vector of parameters to be estimated.

## Implementation of Non-Linear Least Squares

SciPy provides the function
**scipy.optimize.least-squares()**to solve non-linear least-squares problems efficiently. This function is flexible and supports various optimization methods and robust loss functions.
### How Non-Linear Least Squares Method Works

Here are the steps how the Non-Linear least squares works in scipy −

- **Residual Function**− Defines the difference between the observed and predicted values for the model.
- **Optimization Algorithms**− SciPy uses methods like Trust Region Reflective ('trf'), Levenberg-Marquardt ('lm') and Dogleg ('dogbox') to find the optimal parameters.
- **Robust Loss Functions**− Functions such as 'soft_l1', 'huber' and others are used to reduce the influence of outliers in the fitting process.
### Syntax

Following is the syntax for using the function
**scipy.optimize.least_squares()**−
```
scipy.optimize.least_squares(
   fun, 
   x0, 
   jac='2-point', 
   bounds=(-inf, inf), 
   method='trf', 
   loss='linear', ...
)
```

### Parameters

Here are the parameters of the function
**scipy.optimize.least_squares()**−
- **fun**− The residual function that computes the difference between the observed data and the model.
- **x0**− Initial guess for the parameters to be optimized.
- **jac (optional)**− Jacobian matrix or a function to compute it.
- **bounds(optional)**− Constraints on the parameters given as**lower_bounds, upper_bounds**.
- **method**− Optimization algorithm ('trf', 'dogbox', 'lm').
- **loss**− Robust loss function to handle outliers ('linear', 'soft_l1', 'huber').
## Fitting a Nonlinear Function

### Example

Lets solve a simple example where we fit a non-linear model
**y=aexp(bx)**to some noisy data.
```
import numpy as np
from scipy.optimize import least_squares

# Generate synthetic data
x_data = np.linspace(0, 1, 10)
true_params = [2.5, -1.3]
y_data = true_params[0] * np.exp(true_params[1] * x_data) + 0.1 * np.random.randn(len(x_data))

# Define the residual function
def residuals(params, x, y):
    a, b = params
    return y - a * np.exp(b * x)

# Initial guess
x0 = [1.0, -1.0]

# Solve the least-squares problem
result = least_squares(residuals, x0, args=(x_data, y_data))

# Output results
print("Optimal parameters:", result.x)
print("Cost:", result.cost)
print("Residuals:", result.fun)
```

#### Output

Following is the output of the above example which shows the optimal parameters, cost and residuals after fitting the model to the data −

```
Optimal parameters: [ 2.46335211 -1.22610415]
Cost: 0.0631726759601759
Residuals: [-0.14739481  0.13317732  0.12567623 -0.0355416  -0.12790527  0.19364081
 -0.04241043 -0.11512075 -0.02253795  0.02025557]
```

## Constrained Nonlinear Fitting

### Example

Here is the example in which we can solve the same problem as mentioned above with bounds on the parameters such as
**a0, b-2**−
```
import numpy as np
from scipy.optimize import least_squares

# Generate synthetic data
x_data = np.linspace(0, 1, 10)
true_params = [2.5, -1.3]
y_data = true_params[0] * np.exp(true_params[1] * x_data) + 0.1 * np.random.randn(len(x_data))

# Define the residual function
def residuals(params, x, y):
    a, b = params
    return y - a * np.exp(b * x)

# Initial guess
x0 = [1.0, -1.0]

# Solve with bounds on the parameters
result = least_squares(residuals, x0, bounds=([0, -2], [5, 0]), args=(x_data, y_data))

# Output results
print("Optimal parameters with bounds:", result.x)
```

#### Output

Following is the output of solving the non linear least squares with the bound parameter of the function
**scipy.optimize.least-squares()**−
```
Optimal parameters with bounds: [ 2.4754014  -1.26138182]
```

## Robust Nonlinear Fitting

### Example

Sometimes the data may contain outliers and we can use a robust loss function to reduce their impact. Lets solve the above example but introduce an outlier and use the 'soft_l1' loss function in it −

```
import numpy as np
from scipy.optimize import least_squares

# Generate synthetic data
x_data = np.linspace(0, 1, 10)
true_params = [2.5, -1.3]
y_data = true_params[0] * np.exp(true_params[1] * x_data) + 0.1 * np.random.randn(len(x_data))

# Define the residual function
def residuals(params, x, y):
    a, b = params
    return y - a * np.exp(b * x)

# Initial guess
x0 = [1.0, -1.0]

# Add an outlier to the data
y_data_with_outliers = y_data.copy()
y_data_with_outliers[2] += 1  # Introduce an outlier

# Solve using a robust loss function
result = least_squares(residuals, x0, args=(x_data, y_data_with_outliers), loss='soft_l1')

# Output results
print("Optimal parameters (robust):", result.x)
```

#### Output

Here is the output of the robust Non-Linear least square fitting −

```
Optimal parameters (robust): [ 2.63552355 -1.16439492]
```

---

## 79. SciPy - Finding Roots of Scalar Functions

*Source: [https://www.tutorialspoint.com/scipy/scipy_finding_roots_of_scalar_functions.htm](https://www.tutorialspoint.com/scipy/scipy_finding_roots_of_scalar_functions.htm)*

---

---
[Previous](/scipy/scipy_non_linear_least_squares.htm)[Quiz](/scipy/quiz_on_scipy_finding_roots_of_scalar_functions.htm)[Next](/scipy/scipy_finding_roots_of_multivariate_functions.htm)
In numerical analysis, finding the roots of scalar functions is a fundamental task. A root of a scalar function
**f(x)**is a value**x**such that**f(x)=0**. This is commonly used in solving equations where the goal is to find the value of**x**that makes the function zero. SciPy a powerful scientific computing library in Python provides several tools to solve for the roots of scalar functions using various numerical methods.
## Key features of Finding root

Here are the key features of finding root of scalar functions in SciPy −

- **Multiple Algorithms**− SciPy offers a variety of algorithms to solve root-finding problems as mentioned below −
- **Bisection Method**− This method is ideal for continuous functions with opposite signs at the interval endpoints.
- **Newton's Method**− It is used when the function is differentiable and its derivative is available.
- **Hybrid Methods**− These methods which combine several techniques for robust root-finding.
- **Handling Nonlinear Equations**− Suitable for finding roots of nonlinear scalar functions whether smooth or with multiple roots.
- **Initial Guess**− Suitable for finding roots of nonlinear scalar functions whether smooth or with multiple roots.
- **Initial Guess**− Provides flexibility with initial guesses for root location, helping guide the algorithm toward convergence.
- **Customizable Tolerance**− Control the accuracy of the solution with customizable tolerance values e.g., tol.
- **Flexible Constraints**− Some methods support constraints or boundaries on variables during root search, especially in optimization contexts.
- **Convergence Monitoring**− Returns detailed information about the convergence status and messages by allowing users to check if the solution has been found.
- **Support for Derivative-Free Methods**− Methods such as Nelder-Mead are useful when the derivative of the function is not available or the function is non-smooth.
- **Multi-Root Capabilities**− Some algorithms can be adapted to find multiple roots or solutions for a set of equations.
## Functions to find roots of Scalar functions

Here are the key functions which are used to find the roots of Scalar Functions −
S.No.Function & Description1[scipy.optimize.root()](/scipy/scipy_optimize_root_function.htm)
Finds the roots of a scalar or system of nonlinear equations.2scipy.optimize.newton()
Finds the root of a function using the Newton-Raphson method.3scipy.optimize.bisect()
Finds a root of a function within a specified interval using the bisection method.4scipy.optimize.brentq()
Finds a root of a function within a specified interval using Brent's method.5scipy.optimize.fsolve()
Finds the roots of a function using a numerical method for systems of nonlinear equations.6scipy.optimize.minimize_scalar()
Minimizes a scalar function using various optimization algorithms.7scipy.optimize.ridder()
Finds the root of a function using Ridder's method, a root-finding algorithm based on bracketing.
## Using the Bisection Method

### Example

The
**Bisection method**is a simple and robust root-finding method that requires an interval in which the function changes sign. It's particularly useful when we have a continuous function and know that the root exists within the given interval. Here is the using the Bisection method with the help of**scipy.optimize.bisect()**method for finding the roots of Scalar functions −
```
from scipy.optimize import bisect

# Define the objective function
def objective_function(x):
    return x**2 - 4  # root at x = 2 or x = -2

# Perform the root finding in the interval [1, 3]
root = bisect(objective_function, 1, 3)

# Display the result
print("Root found at:", root)
```

#### Output

Following is the output of using the
**scipy.optimize.bisect()**function −
```
Root found at: 2.0
```

## Using the Newton-Raphson Method

### Example

The
**Newton-Raphson**method is an efficient root-finding technique that requires the first derivative of the function. It is often faster than bisection but requires an initial guess near the root. In Scipy we have the function**scipy.optimize.newton()**to perform the Newton-Raphson method. Following is the example which shows how to use the**scipy.optimize.newton()**function to find the roots of scalar functions −
```
from scipy.optimize import newton

# Define the objective function and its derivative
def objective_function(x):
    return x**2 - 4

def derivative_function(x):
    return 2*x

# Perform the root finding with an initial guess of 1.5
root = newton(objective_function, x0=1.5, fprime=derivative_function)

# Display the result
print("Root found at:", root)
```

#### Output

Following is the output of using the
**scipy.optimize.newton()**function −
```
Root found at: 2.0
```

## Using the scipy.optimize.root for general equation

### Example

The root function in SciPy can be used for more complex root-finding tasks including systems of equations. In SciPy we have the function
**scipy.optimize.root()**function to find the roots for a general non linear equation. Below is the example which shows how to use the**scipy.optimize.root()**function −
```
from scipy.optimize import root

# Define the objective function
def objective_function(x):
    return [x[0]**2 - 4, x[1] - x[0]]

# Initial guess
initial_guess = [1, 1]

# Find the root of the system
solution = root(objective_function, initial_guess)

# Display the result
print("Root found at:", solution.x)
```

#### Output

Following is the output of using the
**scipy.optimize.root()**function −
```
Root found at: [2. 2.]
```

---

## 80. SciPy - Finding Roots of Multivariate Functions

*Source: [https://www.tutorialspoint.com/scipy/scipy_finding_roots_of_multivariate_functions.htm](https://www.tutorialspoint.com/scipy/scipy_finding_roots_of_multivariate_functions.htm)*

---

---
[Previous](/scipy/scipy_finding_roots_of_scalar_functions.htm)[Quiz](/scipy/quiz_on_scipy_finding_roots_of_multivariate_functions.htm)[Next](/scipy/scipy_signal_filtering_smoothing.htm)**Finding Roots of Multivariate Functions**in SciPy refers to the process of finding the values of multiple variables, typically represented as a vector that satisfy a system of nonlinear equations i.e., solving for the root or zero of a multivariate function. In other words we can say it involves finding the points where a vector-valued function**F(x)=0**where**F(x)**is a set of equations and**x**a vector of variables.
## How SciPy Helps in Finding Roots?

In SciPy we have the function
**scipy.optimize.root()**which is used to find the roots of systems of multivariate nonlinear equations. It is a part of the**scipy.optimize**module which provides optimization and root-finding routines. The root function allows us to solve these systems by providing an initial guess and choosing from a variety of methods for solving the system.
## Why to Use SciPy to find Multivariate Root?

Here are the reasons why to use SciPy for Multivariate Root Finding −

- **Robust and Flexible Algorithms**− SciPy provides various solvers such as 'hybr', 'lm', 'broyden1', etc which tailored for different nonlinear systems by making it versatile for diverse problems.
- **Handles Complex Systems**− It can solve coupled, nonlinear equations that lack analytical solutions.
- **Ease of Use**− A simple interface lets users define equations, provide initial guesses, and solve efficiently.
- **Automatic or Custom Jacobians**− SciPy can approximate derivatives or accept user-provided Jacobians for faster convergence.
- **Scalable for Large Systems**− Solvers such as 'broyden1' handle large systems without needing explicit Jacobians.
- **Reliable and Optimized**− Built on trusted libraries like MINPACK by ensuring accuracy and performance.
- **Integration**− Works seamlessly with NumPy, Matplotlib and SymPy for end-to-end workflows.
- **Diagnostic Tools**− Provides detailed outputs for convergence, residuals, and solution validation.
### Syntax

Following is the Syntax of the function
**scipy.optimize.root()**which is used to find the roots of multivariate functions −
```
scipy.optimize.root(
   fun, 
   x0, 
   args=(), 
   method='hybr', 
   jac=None, 
   tol=None, 
   options={}
)
```

### Parameters

- **fun(callable)**− The function whose roots are sought. It should take an array**x**and return an array of the same shape.
- **x0(array_like)**− Initial guess for the solution.
- **args(tuple, optional)**− Extra arguments to pass to the function**fun**.
- **method(str, optional)**− The algorithm to use. Options include 'hybr', 'lm', 'broyden1', 'anderson', etc.
- **jac(callable, optional)**− A function to compute the Jacobian matrix of**fun**. If not provided then finite differences are used.
- **tol(float, optional)**− Tolerance for termination.
- **options(dict, optional)**− Solver-specific options like max iterations or verbosity level.
### Return Value
**scipy.optimize.root()**returns an OptimizeResult object containing the below values −
- **x**− The solution array.
- **success**− A boolean indicating if the solver succeeded.
- **message**− Description of the cause of termination.
- **fun**− The residuals at the solution.
## Finding the Root Multiple functions

### Example

Here in this example we will solve for the roots of multiple function involving multiple variables for the equations
**f**and(x,y) = e+y-1 = 0**f**−(x,y) = x+3x+1 = 0
```
import numpy as np
from scipy.optimize import root

# Define the system of nonlinear equations
def fun_system(vars):
    x, y = vars
    # f1(x, y) = e^x + y^2 - 1
    # f2(x, y) = x^3 - 3x + 1
    f1 = np.exp(x) + y**2 - 1
    f2 = x**3 - 3*x + 1
    return [f1, f2]  # Return both residuals

# Initial guess for [x, y]
initial_guess = [0.0, 0.0]

# Use root to solve the system of equations
result = root(fun_system, initial_guess)

# Display the result
if result.success:
    print(f"Roots found: x = {result.x[0]}, y = {result.x[1]}")
else:
    print(f"Root finding failed: {result.message}")
```

#### Output

Here is the output of the function
**scipy.optimize.root()**which is used to find the root of multiple functions −
```
Root finding failed: The iteration is not making good progress, as measured by the
  improvement from the last ten iterations.
```

## Solving a System with More Equations & Variables

### Example

Here is an example which shows how to solve a system with more equations and variables with the help of the function
**scipy.optimize.root()**−
```
import numpy as np
from scipy.optimize import root

# Define the system of equations
def system_of_equations(x):
   return [
      x[0]**2 + x[1]**2 + x[2]**2 - 9,  # x + x + x = 9
      x[0] + x[1] - x[2] - 1,            # x + x - x = 1
      x[0] - x[1] - x[2] - 2             # x - x - x = 2
   ]

# Initial guess for the solution
x0 = [1, 1, 1]

# Solve the system using the 'hybr' method (default)
result = root(system_of_equations, x0, method='hybr')

# Check the result and display the solution
if result.success:
    print("Solution found:", result.x)
else:
    print("Solver failed:", result.message)
```

#### Output

Following is the output of the function
**scipy.optimize.root()**which is used to solve a system with more equations and variables −
```
Solution found: [ 2.70256242 -0.5         1.20256242]
```

---

## 81. SciPy - Signal Filtering and Smoothing

*Source: [https://www.tutorialspoint.com/scipy/scipy_signal_filtering_smoothing.htm](https://www.tutorialspoint.com/scipy/scipy_signal_filtering_smoothing.htm)*

---

---
[Previous](/scipy/scipy_finding_roots_of_multivariate_functions.htm)[Quiz](/scipy/quiz_on_scipy_signal_filtering_smoothing.htm)[Next](/scipy/scipy_short_time_fourier_transform.htm)
In SciPy, the
**signal**module provides a comprehensive set of tools for signal processing, including functions for filtering and smoothing. These tools are widely used for removing noise, improving signal clarity and analyzing data in fields like audio processing, communications and sensor data. This module includes methods for creating, applying and analyzing filters, as well as smoothing data to reduce fluctuations or noise.
## Signal Filtering in SciPy

Filtering is a common technique in signal processing to modify a signal by removing or emphasizing certain frequency components. SciPy provides various types of filters including FIR (Finite Impulse Response) and IIR (Infinite Impulse Response) filters. Below are some key functions available for filtering −

## FIR Filters in SciPy
**FIR filters**are a class of digital filters that use a finite number of sample values. These filters have a fixed, non-recursive structure and are often used for linear-phase filtering. They are typically used to filter out unwanted noise or frequencies from a signal.S.No.Function & Description1[scipy.signal.firwin()](/scipy/scipy_signal_firwin_function.htm)
Creates an FIR filter by specifying the desired filter type (low-pass, high-pass, etc.), the cutoff frequency and other filter parameters.2[scipy.signal.firwin2()](/scipy/scipy_signal_firwin2_function.htm)
Designs an FIR filter with a specified frequency response using piecewise linear interpolation between the desired frequency points.3[scipy.signal.lfilter()](/scipy/scipy_signal_lfilter_function.htm)
Applies a 1D FIR filter to a signal using the filter coefficients obtained from firwin() or similar methods.4[scipy.signal.remez()](/scipy/scipy_signal_remez_function.htm)
Designs an FIR filter using the Parks-McClellan algorithm, which optimizes the filter's frequency response based on the desired specifications.5[scipy.signal.kaiserord()](/scipy/scipy_signal_kaiserord_function.htm)
Calculates the order of a Kaiser window given the desired stopband attenuation and normalized cutoff frequency, useful for FIR filter design.6scipy.signal.firls()
Designs an FIR filter using least squares optimization, where the error between the desired and actual frequency response is minimized.7scipy.signal.firwin2()
Designs an FIR filter using a piecewise linear interpolation of frequency response and corresponding amplitudes.8scipy.signal.dlti()
Creates a discrete-time linear time-invariant system for use with FIR filter designs.9scipy.signal.filtfilt()
Applies a zero-phase FIR filter to a signal using forward and reverse filtering to avoid phase shift.10scipy.signal.hamming()
Creates a Hamming window, often used for FIR filter design to minimize side lobes in the frequency response.11scipy.signal.hann()
Generates a Hanning window, useful for reducing spectral leakage in FIR filter design.12scipy.signal.blackman()
Generates a Blackman window to reduce side lobes for FIR filter design.13scipy.signal.bartlett()
Generates a Bartlett (triangular) window for FIR filter design.14scipy.signal.tukey()
Generates a Tukey window, a combination of a Hanning window and a rectangular window.15scipy.signal.peak_widths()
Measures the widths of peaks in the filters frequency response, useful for analyzing filter characteristics.
## IIR Filters in SciPy
**IIR filters**use feedback which means their output depends on both current and previous input and output values. These filters are more efficient in terms of computation but can introduce phase distortion. They are used for tasks like smoothing and noise reduction.S.No.Function & Description1scipy.signal.butter()
Designs an IIR Butterworth filter with a specified order and cutoff frequency. It is commonly used for low-pass, high-pass, band-pass and band-stop filters.2scipy.signal.cheby1()
Designs a Chebyshev Type I IIR filter with a specified ripple in the passband. Suitable for applications requiring a sharper cutoff than the Butterworth filter.3scipy.signal.cheby2()
Designs a Chebyshev Type II IIR filter with a specified ripple in the stopband. This filter provides a steeper roll-off in the stopband than the Butterworth filter.4scipy.signal.ellip()
Designs an Elliptic (Cauer) IIR filter which provides the sharpest cutoff for a given order and ripple in both the passband and stopband.5scipy.signal.iirfilter()
Generates the coefficients of an IIR filter using various design methods such as Butterworth, Chebyshev or Elliptic.6scipy.signal.lfilter()
Applies a 1D IIR filter to a signal using the filter coefficients obtained from butter(), cheby1() or other IIR filter design functions.7scipy.signal.bilinear()
Converts an analog filter design to a digital filter using the bilinear transform.8scipy.signal.sosfreqz()
Computes the frequency response of a filter in second-order sections (SOS) form, which is commonly used for numerically stable filter designs.9scipy.signal.group_delay()
Calculates the group delay of a filter, which is important in signal processing for understanding how different frequency components are delayed by the filter.10scipy.signal.freqs()
Computes the frequency response of an analog filter, useful for analyzing continuous-time filters.11scipy.signal.resample()
Resamples a signal to a different number of samples, often used in downsampling or upsampling.12scipy.signal.causal()
Generates a causal IIR filter from a given transfer function.13scipy.signal.dlti()
Creates a discrete-time linear time-invariant system to apply IIR filter coefficients.14scipy.signal.sosfreqz()
Calculates the frequency response of a filter in second-order section (SOS) format, a numerical stability technique.15scipy.signal.sosfilt()
Applies a second-order section (SOS) filter to a signal for more stable results in IIR filtering.
## Signal Smoothing
**Smoothing**is another signal processing technique used to reduce noise or fluctuations in a signal. SciPy provides several methods for smoothing signals such as moving averages, Gaussian smoothing and Savitzky-Golay filters. These methods can be applied to both 1D and 2D signals.
## Moving Average

A
**moving average**is a simple method for smoothing data by averaging values within a sliding window. SciPy provides the**uniform_filter1d()**function to compute a moving average −S.No.Function & Description1scipy.ndimage.uniform_filter1d()
Applies a 1D moving average filter to a signal. It replaces each data point with the average of its neighboring points within a specified window size.2scipy.ndimage.uniform_filter()
Applies a uniform filter to an array, commonly used for moving average smoothing in multi-dimensional data.3scipy.ndimage.median_filter1d()
Applies a 1D median filter to smooth the data, replacing each value with the median of its neighbors.4scipy.ndimage.median_filter()
Applies a median filter to multi-dimensional arrays to remove noise while preserving edges.5scipy.ndimage.gaussian_filter1d()
Applies a 1D Gaussian filter to smooth the signal by giving higher weights to values near the center of the window.6scipy.ndimage.gaussian_filter()
Applies a Gaussian filter to smooth multi-dimensional signals, useful for blurring and noise reduction.7scipy.signal.savgol_filter()
Applies the Savitzky-Golay filter to smooth data by fitting successive polynomials to small windows of data.8scipy.signal.wiener()
Applies a Wiener filter to reduce noise in a signal. It adjusts the filtering parameters based on local variance in the data.9scipy.signal.medfilt()
Applies a median filter to a 1D signal for noise reduction, often used to preserve edges in the signal.10scipy.ndimage.correlate1d()
Computes a 1D correlation of an input signal with a specified filter kernel, commonly used for smoothing operations.11scipy.ndimage.convolve1d()
Applies a 1D convolution with a filter kernel, useful for various smoothing and filtering tasks.12scipy.ndimage.laplace()
Computes the Laplacian of a signal, which can be used for edge detection and noise reduction.13scipy.signal.boxcar()
Generates a boxcar window, useful for simple averaging or smoothing of signals.14scipy.ndimage.zoom()
Resizes an array by a specified factor, which can be used to upsample or downsample a signal and smooth it during the process.15scipy.ndimage.prewitt()
Applies the Prewitt filter to a signal, primarily used for edge detection but can also be used for basic smoothing purposes.
Signal filtering and smoothing are fundamental techniques in signal processing and SciPy provides a rich set of tools for performing these operations. Whether we need to remove noise, emphasize specific frequency components or smooth our data and the
**scipy.signal**module offers efficient and easy-to-use functions for achieving these tasks.

---

## 82. SciPy - Short-Time Fourier Transform (STFT)

*Source: [https://www.tutorialspoint.com/scipy/scipy_short_time_fourier_transform.htm](https://www.tutorialspoint.com/scipy/scipy_short_time_fourier_transform.htm)*

---

---
[Previous](/scipy/scipy_signal_filtering_smoothing.htm)[Quiz](/scipy/quiz_on_scipy_short_time_fourier_transform.htm)[Next](/scipy/scipy_discrete_wavelet_transform.htm)
## Short-Time Fourier Transform in SciPy

The
**Short-Time Fourier Transform (STFT)**in SciPy is a tool to analyze signals in both the time and frequency domains. It works by dividing a signal into small, overlapping segments by using a sliding window and then performing a Fourier Transform on each segment.
The result is a time-frequency representation of the signal by showing how its frequency content changes over time. Mathematically, the STFT of a signal
**x(t)**is defined as follows −
```
X(t, f) = - x() w( - t)  e -j 2 f d
```

X(t, f) =
x() w( - t)  ed
Where −

- **x()**is the signal
- **w(t)**is the window function centered at time
- **e**represents the Fourier Transform.
- 
and  are the time and frequency variables respectively.

The window function  ensures that only a small segment of the signal around  contributes to the Fourier Transform at that point.

### Steps of STFT

Following are the steps to perform the Short Time Fourier Transform in SciPy −

- **Windowing**− This divides the signal into overlapping segments using a window function. The window function is typically chosen to minimize spectral leakage e.g., Hann, Hamming, Blackman.
- **Fourier Transform**− Then apply the Fourier Transform to each windowed segment.
- **Time-Frequency Representation**− Finally combine the results for all segments to form a 2D representation with time and frequency axes.
## Short-Time Fourier Transform (STFT) in SciPy

In SciPy we have the function
**scipy.signal.stft()**to perform the**Short-Time Fourier Transform**which provides flexibility in parameters such as the window type, segment length, overlap and FFT size.
### Syntax

Following is the syntax of
**scipy.signal.stft()**function which is used to perform**Short-Time Fourier Transform**−
```
scipy.signal.stft(x,fs=1.0,window='hann',nperseg=256,noverlap=None,nfft=None,detrend=False,return_onesided=True,boundary='zeros',padded=True,axis=-1,scaling='spectrum')
```

### Parameters

Here are the parameters of the function
**scipy.signal.stft()**−
- **x(array_like)**− Input signal. The data to be transformed.
- **fs(float, optional)**− Sampling frequency of the input signal. Default is 1.0.
- **window(str or tuple or array_like, optional)**− Desired window function to apply to each segment. Defaults to 'hann'.
- **nperseg(int, optional)**− Length of each segment for the STFT. Default is 256.
- **noverlap(int, optional)**− Number of points to overlap between segments. If not specified, it defaults to half of**nperseg**.
- **nfft(int, optional)**− Number of points for the FFT computation. If not provided, defaults to**nperseg**.
- **detrend(str or function or bool, optional)**− Specifies how to detrend each segment. Default is False (no detrending).
- **return_onesided(bool, optional)**− If True, returns a one-sided spectrum for real signals. Default is True.
- **boundary(str or None, optional)**− Specifies how to handle the signal boundaries. Default is 'zeros'.
- **padded(bool, optional)**− If True, pads each segment to the nearest power of two. Default is True.
- **axis(int, optional)**− Axis along which the STFT is computed. Default is -1 (last axis).
- **scaling(str, optional)**− Determines the scaling of the STFT output. Options are 'spectrum' (default) and 'density'.
### Basic Example

Following is the example which computes and plots the STFT of a signal composed of two sine waves by using the function
**scipy.signal.stft()**with default parameters −
```
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

# Create a simple signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 2, 2 * fs, endpoint=False)  # 2-second time vector
x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # Two sine waves

# Compute the STFT
f, t, Zxx = stft(x, fs=fs, window='hann', nperseg=256, noverlap=128)

# Plot the STFT Magnitude
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.show()
```

#### Output

Below is the output of the STFT basic example −
![Basic Example of Stft](/scipy/images/basic_example_stft.jpg)
### Adjusting Overlap

This example shows how to change the overlap between segments to 75% resulting in smoother time-frequency resolution −

```
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

# Create a simple signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 2, 2 * fs, endpoint=False)  # 2-second time vector
x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # Two sine waves

# Compute the STFT
f, t, Zxx = stft(x, fs=fs, window='hann', nperseg=256, noverlap=128)

# Plot the STFT Magnitude
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.show()
```

#### Output

Below is the output of the STFT which adjusts the overlap using the function
**scipy.signal.stft()**−![Adjust Overlap Stft](/scipy/images/adjust_overlap_stft.jpg)
## Applications of STFT

Here are the applications of SciPy Short Time Fourier Transform −

- **Speech and Audio Processing**− Analyze time-varying frequency content of speech signals.
- **Music Analysis**− Identify harmonic components and beats in music.
- **Biomedical Signals**− Analyze non-stationary signals like EEG or ECG.
- **Vibration Analysis**− Detect faults in rotating machinery.
- **Communications**− Demodulate time-varying frequency signals.

---

## 83. SciPy - Discrete Wavelet Transform (DWT)

*Source: [https://www.tutorialspoint.com/scipy/scipy_discrete_wavelet_transform.htm](https://www.tutorialspoint.com/scipy/scipy_discrete_wavelet_transform.htm)*

---

---
[Previous](/scipy/scipy_short_time_fourier_transform.htm)[Quiz](/scipy/quiz_on_scipy_discrete_wavelet_transform.htm)[Next](/scipy/scipy_continuous_wavelet_transform.htm)
## Discrete Wavelet Transform in SciPy

The
**Discrete Wavelet Transform (DWT)**is a powerful tool for analyzing signals by decomposing them into different frequency components with a discrete scale. Unlike the Continuous Wavelet Transform (CWT), DWT uses a fixed set of wavelet functions which makes it computationally more efficient and appropriate for real-time signal processing applications.
DWT is commonly used for signal compression, denoising and feature extraction in various fields such as image processing, audio processing and bio-signal analysis.

The Discrete Wavelet Transform of a signal
**x(t)**can be represented as follows −
```
$&bsol;mathrm{W(j, k) = &bsol;int_{-&bsol;infty}^{&bsol;infty} x(t) &bsol;psi^* &bsol;left( &bsol;frac{t - k}{2^j} &bsol;right) dt}$
```

Where −

- **x(t)**is the input signal.
- **(t)**is the mother wavelet.
- **= 2**is the scale factor.
- **= k**is the translation parameter (discrete shifts of the signal).
- **W(j, k)**represents the wavelet coefficients at scale**j**and shift**k**.
## Key Properties of DWT

Following are the key properties of the Discrete Wavelet Transform −

- **Multi-Resolution Analysis**− DWT allows the signal to be analyzed at different scales (resolutions) which helps capture both the low-frequency and high-frequency components.
- **Efficient Computation**− DWT is computationally efficient because it operates with a discrete set of wavelet functions by making it suitable for real-time applications.
- **Downsampling**− DWT performs downsampling at each level, reducing the size of the coefficients and thus the data storage requirements.
- **Time-Frequency Localization**− DWT provides both time and frequency localization of the signal by making it useful for analyzing non-stationary signals.
In SciPy versions 1.15.1 and later the cwt function has been removed from the scipy.signal module. Therefore, if you're using SciPy 1.15.1 or higher, we should use alternative methods to perform the Continuous Wavelet Transform (CWT) such as the PyWavelets (pywt) library.

## Using PyWavelets (pywt)

If We're facing issues with SciPy's dwt function then we have an alternative approach which is to use PyWavelets (pywt), a powerful library for wavelet analysis in Python. It provides extensive functionality for Continuous Wavelet Transform (CWT), Discrete Wavelet Transform (DWT) and more.

### Basic Example of DWT using PyWavelets

In this example we will perform a 1-level DWT on a simple signal using the Haar wavelet which one of the simplest wavelets −

```
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Create a simple signal (a step function for demonstration)
signal = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])

# Perform 1-level Discrete Wavelet Transform (DWT) using the Haar wavelet
coeffs = pywt.dwt(signal, 'haar')

# Extract the approximation (cA) and detail (cD) coefficients
cA, cD = coeffs

# Plot the original signal, approximation coefficients, and detail coefficients
plt.figure(figsize=(10, 6))

# Plot the original signal
plt.subplot(3, 1, 1)
plt.plot(signal, label="Original Signal", color='blue')
plt.title("Original Signal")
plt.legend()

# Plot the approximation coefficients (cA)
plt.subplot(3, 1, 2)
plt.plot(cA, label="Approximation Coefficients (cA)", color='green')
plt.title("Approximation Coefficients (cA)")
plt.legend()

# Plot the detail coefficients (cD)
plt.subplot(3, 1, 3)
plt.plot(cD, label="Detail Coefficients (cD)", color='red')
plt.title("Detail Coefficients (cD)")
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Following is the output of the Basic Discrete Wavelet Transform using PyWavelets −
![PyWavelets Basic DWT](/scipy/images/dwt_basic_pywavelets.jpg)
## Multi-Level DWT Decomposition using PyWavelets

### Example

This example shows how to perform multi-level DWT decomposition on a signal using PyWavelets. We will decompose the signal into multiple levels −

```
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate a simple signal (a combination of two sine waves)
t = np.linspace(0, 1, 500, endpoint=False)  # Time vector
signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)  # Signal

# Perform Multi-Level DWT decomposition using 'db1' (Daubechies wavelet)
coeffs = pywt.wavedec(signal, 'db1', level=3)  # Decompose to 3 levels

# coeffs contains the approximation and detail coefficients at each level
cA3, cD3, cD2, cD1 = coeffs  # cA3: Approximation at level 3, cD3: Detail at level 3, etc.

# Plot the original signal and the decomposition results
plt.figure(figsize=(10, 10))

# Plot the original signal
plt.subplot(5, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the approximation coefficients at level 3
plt.subplot(5, 1, 2)
plt.plot(cA3)
plt.title("Approximation Coefficients at Level 3")

# Plot the detail coefficients at level 3
plt.subplot(5, 1, 3)
plt.plot(cD3)
plt.title("Detail Coefficients at Level 3")

# Plot the detail coefficients at level 2
plt.subplot(5, 1, 4)
plt.plot(cD2)
plt.title("Detail Coefficients at Level 2")

# Plot the detail coefficients at level 1
plt.subplot(5, 1, 5)
plt.plot(cD1)
plt.title("Detail Coefficients at Level 1")

plt.tight_layout()
plt.show()
```

#### Output

Below is the output of the Multi-Level Discrete Wavelet Transform using PyWavelets −
![PyWavelets Multi-Level DWT](/scipy/images/dwt_multi_level_wavelets.jpg)
## Applications of DWT

The Discrete Wavelet Transform (DWT) is widely used in various fields due to its computational efficiency and ability to provide both time and frequency information. Some key applications are as follows −

- **Signal Compression**− DWT is used in applications like JPEG 2000 for image compression and in audio and video compression formats.
- **Denoising**− DWT helps remove noise from signals while preserving the important features by making it useful in fields like speech processing and biomedical signal analysis.
- **Feature Extraction**− DWT is used to extract features from signals for classification or qattern recognition in machine learning tasks.
- **Data Fusion**− DWT can fuse information from multiple data sources useful in sensor networks and medical diagnostics.
## Choosing the Right Wavelet for DWT

Just like in CWT choosing the right wavelet for DWT is important for capturing the features of the signal being analyzed. Some common wavelets for DWT are as follows −
WaveletBest ForExample Use Cases**Daubechies (db)**Smooth signals, multiresolution analysisSignal denoising, image compression**Symlet**Symmetric signals, efficient approximationAudio processing, compression**Coiflet**Signals with smooth features, multiresolutionFeature extraction, denoising**Haar**Discontinuous signals, edge detectionCompression, real-time signal analysis

---

## 84. SciPy - Continuous Wavelet Transform (CWT)

*Source: [https://www.tutorialspoint.com/scipy/scipy_continuous_wavelet_transform.htm](https://www.tutorialspoint.com/scipy/scipy_continuous_wavelet_transform.htm)*

---

---
[Previous](/scipy/scipy_discrete_wavelet_transform.htm)[Quiz](/scipy/quiz_on_scipy_continuous_wavelet_transform.htm)[Next](/scipy/scipy_discrete_wavelet_transform.htm)
## Continuous Wavelet Transform in SciPy

The
**Continuous Wavelet Transform (CWT)**is a powerful signal processing technique used to analyze signals in both the time and frequency domains simultaneously. Unlike the, Fourier Transform which provides global frequency content, CWT allows for multi-resolution analysis by making it particularly useful for signals with time-varying frequency content (non-stationary signals).
The Continuous Wavelet Transform of a signal
**x(t)**is defined by the following integral −
```
W(a, b) = - x(t) * ( (t - b) / a ) dt
```

W(a, b) =
x(t)( (t - b) / a ) dt
Where −

- **x(t)**is the input signal.
- **(t)**is the mother wavelet and a function localized in both time and frequency.
- is the complex conjugate of the wavelet function.
- is the scale parameter i.e.,controls the dilation or compression of the wavelet, related to frequency.
- **b**is the translation parameter i.e., controls the shifting in time.
- **W(a,b)**represents the wavelet coefficients that provide a measure of similarity between the signal and the wavelet at different scales and time positions.
## Key Properties of CWT

Following are the key properties of the Continuous Wavelet Transform −

- **Time-Frequency Localization:**CWT provides information about when and at what frequency a particular event occurs.
- **Multi-Resolution Analysis:**Small scales (low a) capture high-frequency details i.e., sharp features and Large scales (high a) capture low-frequency components i.e., smooth variations.
- **Redundancy:**CWT is highly redundant compared to Discrete Wavelet Transform (DWT) by making it computationally expensive but useful for detailed analysis.**Note:**
In SciPy versions 1.15.1 and later the cwt function has been removed from the scipy.signal module. Therefore, if you're using SciPy 1.15.1 or higher, we should use alternative methods to perform the Continuous Wavelet Transform (CWT) such as the PyWavelets (pywt) library.

## Using PyWavelets (pywt)

If We're facing issues with SciPy's cwt function, an alternative approach is to use PyWavelets (pywt), a powerful library for wavelet analysis in Python. It provides extensive functionality for Continuous Wavelet Transform (CWT), Discrete Wavelet Transform (DWT) and more.

### Installing PyWavelets

To install PyWavelets in our working environment we have to run the following command in the command prompt −

```
pip install pywavelets
```

Following is the output after executing the above command −

```
Successfully installed pywavelets-1.8.0
```

Verifying the installation with the help of below command −

```
import pywt
print(pywt.__version__)
```

Following is the output of the installed version of PyWavelets −

```
1.8.0
```

## Basic Example of CWT using PyWavelets

Following is a basic example of performing a Continuous Wavelet Transform (CWT) using the PyWavelets (pywt) library. This example analyzes a simple signal using the Morlet wavelet which is commonly used for time-frequency analysis −

```
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Generate a simple signal: combination of two sine waves (10 Hz and 20 Hz)
t = np.linspace(0, 1, 500, endpoint=False)  # Time vector (1 second, 500 samples)
signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)  # Signal

# Define the range of scales for wavelet transform
scales = np.arange(1, 50)

# Perform Continuous Wavelet Transform using the Morlet wavelet
coefficients, frequencies = pywt.cwt(signal, scales, 'morl')

# Plot the original signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the CWT coefficients as a heatmap
plt.subplot(2, 1, 2)
plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 50], cmap='jet', aspect='auto',
           vmax=abs(coefficients).max(), vmin=0)
plt.colorbar(label="Magnitude")
plt.title("Continuous Wavelet Transform (CWT)")
plt.xlabel("Time (s)")
plt.ylabel("Scale")
plt.tight_layout()
plt.show()
```

Following is the output of the Continuous Wavelet Transform using PyWavelets −
![PyWavelets Basic CWT](/scipy/images/pywavelet_cwt_basic.jpg)
## Detecting a Transient Event (Gaussian Pulse)

In this example we will see how to use PyWavelets to detect a transient event such as a Gaussian pulse, within a time series signal. The Continuous Wavelet Transform (CWT) can help identify the presence and timing of such transient events −

```
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Generate a time vector
t = np.linspace(0, 1, 1000, endpoint=False)  # 1 second, 1000 samples

# Create a signal with a Gaussian pulse embedded in noise
signal = np.sin(2 * np.pi * 10 * t)  # Background sine wave (10 Hz)
gaussian_pulse = np.exp(-((t - 0.5)**2) / (2 * 0.01**2))  # Gaussian pulse at t = 0.5s
noise = np.random.normal(0, 0.5, t.shape)  # Random noise

# Final signal (sine wave + transient pulse + noise)
signal += gaussian_pulse + noise

# Define wavelet scales for analysis
scales = np.arange(1, 128)

# Perform Continuous Wavelet Transform using the Ricker (Mexican Hat) wavelet
coefficients, frequencies = pywt.cwt(signal, scales, 'mexh')

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Signal with Transient Gaussian Pulse")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the CWT coefficients as a heatmap
plt.subplot(2, 1, 2)
plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='jet', aspect='auto',
           vmax=abs(coefficients).max(), vmin=0)
plt.colorbar(label="Magnitude")
plt.title("Continuous Wavelet Transform (CWT)")
plt.xlabel("Time (s)")
plt.ylabel("Scale")
plt.tight_layout()
plt.show()
```

Following is the output of the Detecting Transient event by using the CWT in PyWavelets −
![PyWavelets Guassian CWT](/scipy/images/guassian_cwt.jpg)
## Choosing the Right Wavelet

Choosing the right wavelet for performing Continuous Wavelet Transform (CWT) depends on the specific characteristics of the signal we're analyzing and the type of features we want to capture. Different wavelets have different properties that make them more suitable for certain types of signals. Heres a guide to help we choose the right wavelet −
WaveletBest ForExample Use Cases**Ricker (Mexican Hat)**Transient pulses, sharp featuresDetecting spikes, Gaussian pulses**Morlet**Oscillatory signals, time-frequency analysisEEG, audio signals, periodic signals**Gaussian**Smooth, non-oscillatory signalsGradual transitions, denoising**Haar**Discontinuous signals, edge detectionSignal compression, sharp transitions**Morse**Time-frequency analysis of broad signalsSeismic, bio-signals (EEG, ECG), multiresolution analysis

---

## 85. SciPy - Wavelet Packet Transform

*Source: [https://www.tutorialspoint.com/scipy/scipy_wavelet_packet_transform.htm](https://www.tutorialspoint.com/scipy/scipy_wavelet_packet_transform.htm)*

---

---

## 86. SciPy - Multi-Resolution Analysis (MRA)

*Source: [https://www.tutorialspoint.com/scipy/scipy_multi_resolution_analysis.htm](https://www.tutorialspoint.com/scipy/scipy_multi_resolution_analysis.htm)*

---

---
[Previous](/scipy/scipy_wavelet_packet_transform.htm)[Quiz](/scipy/quiz_on_scipy_multi_resolution_analysis.htm)[Next](/scipy/scipy_stationary_wavelet_transform.htm)
## Multi-Resolution Analysis in SciPy

The
**Multi-Resolution Analysis (MRA)**is a fundamental concept in wavelet analysis that enables the decomposition of signals into different resolution levels. It provides a systematic way to analyze signals at multiple scales by capturing both coarse and fine details. MRA is widely used in signal processing applications such as denoising, compression and feature extraction.
The Multi-Resolution Analysis is based on the scaling function
**(t)**and the wavelet function**(t)**which are used to represent a signal**f(t)**at different resolution levels. Mathematically, MRA is defined as follows −
```
f(t) = ∑j ∑k cj, k φ(t - k 2j) + ∑j ∑k dj, k ψ(t - k 2j)
```

f(t) = ∑
∑cφ(t - k 2) + ∑∑dψ(t - k 2)
Where −

- **f(t)**is the original signal.
- **(t)**is the scaling function that represents the approximation at each level.
- **(t)**is the wavelet function that captures the detail components.
- **c**are the approximation coefficients.
- **d**are the detail coefficients.
## Key Properties of Multi-Resolution Analysis

The Multi-Resolution Analysis provides the following key properties in SciPy −

- **Hierarchical Signal Representation:**MRA decomposes a signal into different levels of approximation and detail components.
- **Localization:**MRA provides time-frequency localization by making it useful for analyzing transient and non-stationary signals.
- **Data Reduction:**MRA reduces the amount of data needed to represent a signal by focusing on relevant features at each resolution.
- **Smooth and Detail Separation:**MRA separates the low-frequency (approximation) and high-frequency (detail) components effectively.
## Multi-Resolution Decomposition of a signal

Multi-Resolution Decomposition (MRD) using
**PyWavelets**allows breaking down a signal into different resolution levels by applying Discrete Wavelet Transform (DWT). This helps analyze both coarse and fine details of the signal efficiently. MRD is widely used in applications such as signal compression, feature extraction and denoising.
In PyWavelets, the
**wavedec()**function is used to perform multi-level decomposition of a signal. It returns the approximation and detail coefficients at each decomposition level.
Following is the example which shows how to decompose a signal into multiple levels using PyWavelets −

```
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Generate a sample signal (a combination of sine waves)
t = np.linspace(0, 1, 500, endpoint=False)  # Time vector
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)  # Combination of frequencies

# Perform Multi-Level DWT decomposition using Daubechies wavelet ('db4')
coeffs = pywt.wavedec(signal, 'db4', level=3)

# Extract approximation and detail coefficients
cA3, cD3, cD2, cD1 = coeffs  # cA3: Approximation at level 3, cD3: Detail at level 3, etc.

# Plot the original signal and decomposition results
plt.figure(figsize=(10, 10))

# Plot the original signal
plt.subplot(5, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Plot approximation coefficients at level 3
plt.subplot(5, 1, 2)
plt.plot(cA3)
plt.title("Approximation Coefficients at Level 3")

# Plot detail coefficients at level 3
plt.subplot(5, 1, 3)
plt.plot(cD3)
plt.title("Detail Coefficients at Level 3")

# Plot detail coefficients at level 2
plt.subplot(5, 1, 4)
plt.plot(cD2)
plt.title("Detail Coefficients at Level 2")

# Plot detail coefficients at level 1
plt.subplot(5, 1, 5)
plt.plot(cD1)
plt.title("Detail Coefficients at Level 1")

plt.tight_layout()
plt.show()
```

Below is the output of the Multi-Level Wavelet Decomposition −
![Multi-Resolution Decomposition](/scipy/images/basic_multi_level_dwt.jpg)
## Multi-Resolution Decomposition of multiple signals

This example shows how multi-resolution decomposition works with a different signal and wavelet by giving a different perspective on how this technique can be applied to various types of signals −

```
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Generate a signal (combination of a sine wave and a high-frequency pulse)
t = np.linspace(0, 1, 1000, endpoint=False)  # Time vector
signal = np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)  # Combination of low and high frequencies

# Perform Multi-Level DWT decomposition using the Symlet wavelet (sym2)
coeffs = pywt.wavedec(signal, 'sym2', level=4)

# Extract approximation and detail coefficients for each level
cA4, cD4, cD3, cD2, cD1 = coeffs  # cA4: Approximation at level 4, cD4: Detail at level 4, etc.

# Plot the original signal and the decomposition results
plt.figure(figsize=(10, 12))

# Plot the original signal
plt.subplot(6, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Plot approximation coefficients at level 4
plt.subplot(6, 1, 2)
plt.plot(cA4)
plt.title("Approximation Coefficients at Level 4")

# Plot detail coefficients at level 4
plt.subplot(6, 1, 3)
plt.plot(cD4)
plt.title("Detail Coefficients at Level 4")

# Plot detail coefficients at level 3
plt.subplot(6, 1, 4)
plt.plot(cD3)
plt.title("Detail Coefficients at Level 3")

# Plot detail coefficients at level 2
plt.subplot(6, 1, 5)
plt.plot(cD2)
plt.title("Detail Coefficients at Level 2")

# Plot detail coefficients at level 1
plt.subplot(6, 1, 6)
plt.plot(cD1)
plt.title("Detail Coefficients at Level 1")

plt.tight_layout()
plt.show()
```

Below is the output of the Multi-Level Wavelet Decomposition for multi signals −
![Multi-Resolution Decomposition for multiple signals](/scipy/images/multiple_signals__multilevel_dwt.jpg)
## Understanding the Decomposed Components

After performing the decomposition the signal is broken down into the following components −

- **Approximation Coefficients:**Represent the low-frequency (coarse) part of the signal.
- **Detail Coefficients:**Capture the high-frequency (fine) components at each level.
## Applications of Multi-Resolution Decomposition

The Multi-Resolution Decomposition technique is useful in various fields such as −

- **Image Compression:**Used in JPEG2000 for progressive image reconstruction.
- **Biomedical Signal Processing:**Analyzing ECG and EEG signals for diagnosis.
- **Fault Detection:**Identifying faults in machinery through vibration analysis.
- **Time-Series Forecasting:**Extracting key features from financial or environmental data.
## Choosing the Right Wavelet for MRA

The choice of wavelet function is critical for effective Multi-Resolution Analysis. Some commonly used wavelets and their applications are listed below −
WaveletBest ForExample Use Cases**Daubechies (db)**General-purpose smooth signalsSignal compression, denoising**Coiflet**High vanishing moments, feature detectionBiomedical analysis**Haar**Simple step-like signalsEdge detection, compression**Note:**Choosing the appropriate wavelet depends on the characteristics of the signal and the application requirements.

---

## 87. SciPy - Stationary Wavelet Transform

*Source: [https://www.tutorialspoint.com/scipy/scipy_stationary_wavelet_transform.htm](https://www.tutorialspoint.com/scipy/scipy_stationary_wavelet_transform.htm)*

---

---

## 88. SciPy - Stats

*Source: [https://www.tutorialspoint.com/scipy/scipy_stats.htm](https://www.tutorialspoint.com/scipy/scipy_stats.htm)*

---

---

## 89. SciPy - Descriptive Statistics

*Source: [https://www.tutorialspoint.com/scipy/scipy_descriptive_statistics.htm](https://www.tutorialspoint.com/scipy/scipy_descriptive_statistics.htm)*

---

---
[Previous](/scipy/scipy_stats.htm)[Quiz](/scipy/quiz_on_scipy_descriptive_statistics.htm)[Next](/scipy/scipy_continuous_probability_distributions.htm)**Descriptive statistics**is a branch of statistics that focuses on summarizing and organizing data to reveal meaningful insights. It helps in understanding the distribution, central tendency and variability of data. The Python library SciPy, particularly its stats module provides various functions to compute descriptive statistics efficiently.
## Key Measures in Descriptive Statistics

Descriptive statistics are used to summarize and describe the main features of a dataset. These measures fall into three main categories as follows −

## Measures of Central Tendency in SciPy

Measures of central tendency summarize a dataset by identifying a single value that represents the center or "typical" value of the data. The three main measures of central tendency as mentioned below −

### Mean (Arithmetic Average)

The mean is calculated by summing all data points and dividing by the total number of points. It is sensitive to outliers which can significantly affect its value. The formula for Mean is given as below −

Mean = ½ (∑ X) / N

Below is the example of finding Mean by the function with the help of
**scipy.stats.tmean()**function −
```
from scipy import stats

data = [10, 20, 30, 40, 50]

# Calculate mean using SciPy
mean_value = stats.tmean(data)
print("Mean:", mean_value)
```

Here is the output of Mean with the help of
**scipy.stats.tmean()**function −
```
Mean: 30.0
```

### Median

The median is the value that falls in the center of a sorted dataset. When there is an even number of data points then the median is calculated as the average of the two middle values. Unlike the mean, the median is less affected by outliers.

Here is the example which calculates the median with the help of
**scipy.stats.scoreatpercentile()**function −
```
from scipy import stats

# Sample data
data = [10, 20, 30, 40, 50]

# Calculate median using SciPy's scoreatpercentile
median_value = stats.scoreatpercentile(data, 50)
print("Median:", median_value)
```

Below is the output of the median calculated using the function
**scipy.stats.scoreatpercentile()**−
```
Median: 30.0
```

### Mode

The mode is the value that occurs most frequently in the dataset. If there is more than one mode, it is referred to as multimodal.

Following is the example which calculates the Mode with the help of
**scipy.stats.mode()**function −
```
from scipy import stats

# Sample data
data = [10, 20, 20, 30, 40]

# Calculate mode using SciPy
mode_value = stats.mode(data)

# Access mode and count correctly
print("Mode:", mode_value.mode, "Frequency:", mode_value.count)
```

Below is the output of the Mode calculated using the function
**scipy.stats.mode()**−
```
Mode: 20 Frequency: 2
```

## Measures of Dispersion in SciPy

Measures of dispersion indicate how data values are spread out or dispersed within a dataset. They help determine the variability or consistency of data points relative to each other. The key measures of dispersion are described below −

### Range

The range is the simplest way to measure dispersion, calculated by subtracting the smallest value from the largest value in the dataset. Although it gives a quick sense of data spread, it is highly influenced by outliers.

Here is an example that shows how to compute the range using the
**numpy.ptp()**function −
```
# Sample data
data = [10, 20, 20, 30, 40]

range_value = max(data) - min(data)
print("Range:", range_value)
```

Here is the output of the range calculation −

```
Range: 30
```

### Variance

Variance measures how much the data values deviate from the mean. It is computed by averaging the squared differences between each data point and the mean value. A higher variance indicates more spread-out data.

The mathematical representation of variance is given below −

```
Variance = ½ (∑ (X - Mean)2) ÷ N
```
) ÷ N
The following example calculates variance using the
**scipy.stats.tvar()**function −
```
from scipy import stats

data = [10, 20, 30, 40, 50]

# Calculate variance using SciPy
variance_value = stats.tvar(data)
print("Variance:", variance_value)
```

Here is the output of the variance calculation using
**scipy.stats.tvar()**function −
```
Variance: 250.0
```

### Standard Deviation

Standard deviation is derived from the variance and provides a measure of data dispersion in the same units as the original dataset. It indicates how much the values differ from the mean.

Below example shows how to compute the standard deviation using the
**scipy.stats.tstd()**function −
```
from scipy import stats

data = [10, 20, 30, 40, 50]

# Calculate standard deviation using SciPy
std_deviation = stats.tstd(data)
print("Standard Deviation:", std_deviation)
```

Below is the output of the standard deviation calculation using
**scipy.stats.tstd()**function −
```
Standard Deviation: 15.811388300841896
```

### Skewness

Skewness measures the asymmetry of a dataset's distribution around its mean. If the skewness is positive, it indicates that the data has a long right tail (positive skew) whereas a negative skew indicates a long left tail (negative skew). The formula for calculating skewness is given below −

```
Skewness = (n ∑i (Xi - X)3) / ((n - 1) s3)
```
(X- X)) / ((n - 1) s)
Below is an example of how to calculate Skewness using the
**scipy.stats.skew()**function −
```
from scipy import stats

data = [10, 20, 20, 30, 40, 50, 60]

# Calculate skewness using SciPy
skewness_value = stats.skew(data)
print("Skewness:", skewness_value)
```

Here is the output when calculating Skewness using the function
**scipy.stats.skew()**−
```
Skewness: 0.28372927689018057
```

### Kurtosis

Kurtosis measures the heaviness of the tails of a data distribution. High kurtosis suggests the presence of outliers or extreme values while low kurtosis indicates a distribution with fewer outliers. The formula for calculating kurtosis is given below −

```
Kurtosis = &frac{n ∑ (Xi - X)4}{(n - 1) · s4}
```
- X)}{(n - 1) · s}
Below is an example of calculating Kurtosis using the
**scipy.stats.kurtosis()**function −
```
from scipy import stats

data = [10, 20, 20, 30, 40, 50, 60]

# Calculate kurtosis using SciPy
kurtosis_value = stats.kurtosis(data)
print("Kurtosis:", kurtosis_value)
```

Here is the output when calculating Kurtosis using the function
**scipy.stats.kurtosis()**−
```
Kurtosis: -1.2208044982698956
```

---

## 90. SciPy - Continous Probability Distributions

*Source: [https://www.tutorialspoint.com/scipy/scipy_continuous_probability_distributions.htm](https://www.tutorialspoint.com/scipy/scipy_continuous_probability_distributions.htm)*

---

---
[Previous](/scipy/scipy_descriptive_statistics.htm)[Quiz](/scipy/quiz_on_scipy_continuous_probability_distributions.htm)[Next](/scipy/scipy_discrete_probability_distributions.htm)**Continuous probability distributions**refer to statistical models where the random variable can take any value within a specified range or interval. These distributions are fundamental in many scientific fields such as physics, engineering and economics, as they can model real-world scenarios like measurements or time intervals.
The
**scipy.stats**library in Python provides an extensive collection of tools for working with these distributions by allowing us to calculate important statistical measures such as probability density functions (PDF), cumulative distribution functions (CDF) and more.
## Key Continuous Distributions in SciPy

In SciPy continuous distributions represent random variables that can take any value within a range. SciPy provides a wide variety of continuous probability distributions and methods for working with them.

## Normal Distribution

The
**Normal Distribution**which often referred to as the Gaussian distribution, is one of the most commonly used continuous distributions in statistics. It has a symmetric bell-shaped curve, with the center of the distribution defined by its mean and the spread determined by its standard deviation. This distribution is widely applied in various fields like quality control, finance and natural sciences.
### Example

In SciPy the normal distribution is represented by the
**scipy.stats.norm**object. Heres an example of calculating and visualizing the probability density and cumulative distribution of a normal distribution −
```
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Define mean and standard deviation
mean = 0
std_dev = 1

# Generate an array of values for x between -5 and 5
x_values = np.linspace(-5, 5, 100)

# Calculate the probability density function (PDF) and cumulative distribution function (CDF)
pdf_values = norm.pdf(x_values, mean, std_dev)
cdf_values = norm.cdf(x_values, mean, std_dev)

# Plot the results
plt.figure(figsize=(12, 6))

# PDF plot
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, label='PDF')
plt.title('Normal Distribution - PDF')
plt.legend()

# CDF plot
plt.subplot(1, 2, 2)
plt.plot(x_values, cdf_values, label='CDF', color='red')
plt.title('Normal Distribution - CDF')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Here is the output of the normal distribution calculated using
**scipy.stats.norm.pdf()**and**scipy.stats.norm.cdf()**function −![Normal Distribution](/scipy/images/normal_distribution.jpg)
## Exponential Distribution

The
**Exponential Distribution**is often used to model the time between events in a Poisson process, where the events occur independently and at a constant average rate. The distribution has a single parameter,  (lambda) which represents the rate at which events happen. This distribution is useful for processes that involve waiting times.
### Example

In SciPy the exponential distribution can be handled with the
**scipy.stats.expon**object. Heres an example of calculating and plotting the PDF and CDF for the exponential distribution −
```
from scipy.stats import expon
import numpy as np
import matplotlib.pyplot as plt

# Set the rate (lambda)
rate = 1

# Create an array of x values from 0 to 10
x_values = np.linspace(0, 10, 100)

# Compute the PDF and CDF
pdf_values = expon.pdf(x_values, scale=1/rate)
cdf_values = expon.cdf(x_values, scale=1/rate)

# Plot the distributions
plt.figure(figsize=(12, 6))

# PDF plot
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, label='PDF')
plt.title('Exponential Distribution - PDF')
plt.legend()

# CDF plot
plt.subplot(1, 2, 2)
plt.plot(x_values, cdf_values, label='CDF', color='red')
plt.title('Exponential Distribution - CDF')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Following is the output of the Exponential distribution calculated using
**scipy.stats.expon.pdf()**and**scipy.stats.expon.cdf()**function −![Exponential Distribution](/scipy/images/exponential_distribution.jpg)
## Gamma Distribution

The
**Gamma Distribution**is a generalization of the exponential distribution that includes an additional parameter, the shape parameter which allows for a wider variety of distribution shapes. This distribution is frequently used in queuing theory and reliability analysis.
### Example

In SciPy the gamma distribution is represented by the
**scipy.stats.gamma**object. Below is an example of calculating the PDF and CDF for the gamma distribution −
```
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the gamma distribution
shape_param = 2
scale_param = 1

# Generate an array of x values
x_values = np.linspace(0, 10, 100)

# Compute the PDF and CDF
pdf_values = gamma.pdf(x_values, shape_param, scale=scale_param)
cdf_values = gamma.cdf(x_values, shape_param, scale=scale_param)

# Plot the results
plt.figure(figsize=(12, 6))

# PDF plot
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, label='PDF')
plt.title('Gamma Distribution - PDF')
plt.legend()

# CDF plot
plt.subplot(1, 2, 2)
plt.plot(x_values, cdf_values, label='CDF', color='red')
plt.title('Gamma Distribution - CDF')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Below is the output of the Gamma distribution calculated using
**scipy.stats.gamma.pdf()**and**scipy.stats.gamma.cdf()**function −![Gamma Distribution](/scipy/images/gamma_distribution.jpg)
## Beta Distribution

The
**Beta Distribution**is a versatile distribution used to model random variables that are constrained to a fixed interval, typically between 0 and 1. It is often applied in scenarios where probabilities and proportions are involved such as in Bayesian statistics.
### Example

The beta distribution is represented in SciPy by
**scipy.stats.beta**. Here's an example of plotting the PDF and CDF of a beta distribution −
```
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

# Set the shape parameters for the beta distribution
alpha = 2
beta_param = 5

# Generate values for x in the range [0, 1]
x_values = np.linspace(0, 1, 100)

# Calculate the PDF and CDF
pdf_values = beta.pdf(x_values, alpha, beta_param)
cdf_values = beta.cdf(x_values, alpha, beta_param)

# Plot the results
plt.figure(figsize=(12, 6))

# PDF plot
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, label='PDF')
plt.title('Beta Distribution - PDF')
plt.legend()

# CDF plot
plt.subplot(1, 2, 2)
plt.plot(x_values, cdf_values, label='CDF', color='red')
plt.title('Beta Distribution - CDF')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Below is the output of the Beta distribution calculated using
**scipy.stats.beta.pdf()**and**scipy.stats.beta.cdf()**function −![Beta Distribution](/scipy/images/beta_distribution.jpg)
## Working with Continuous Distributions in SciPy

SciPy provides numerous methods for manipulating and working with continuous distributions which are mentioned as below −

- **PDF (Probability Density Function)**−**distribution.pdf(x, params)**computes the likelihood of a given value**x**.
- **CDF (Cumulative Distribution Function)**−**distribution.cdf(x, params)**calculates the cumulative probability up to the point**x**.
- **PPF (Percent-Point Function)**−**distribution.ppf(p, params)**returns the value corresponding to a specified cumulative probability**p**.
- **Random Sampling**−**distribution.rvs(params, size=N)**generates**N**random values from the distribution.
- **Mean and Variance**−**distribution.mean()**and**distribution.var()**calculate the mean and variance of the distribution.
### Example

For instance we can calculate the mean and variance of a normal distribution as follows −

```
from scipy.stats import norm

# Calculate the mean and variance of the normal distribution
mean = norm.mean(loc=0, scale=1)
variance = norm.var(loc=0, scale=1)

print("Mean of Normal Distribution:", mean)
print("Variance of Normal Distribution:", variance)
```

#### Output

Here is the output of Mean and Variance of a normal distribution −

```
Mean of Normal Distribution: 0.0
Variance of Normal Distribution: 1.0
```

SciPys
**scipy.stats**module offers a powerful suite of tools for working with continuous probability distributions. Whether we're analyzing simple distributions like the normal and exponential distributions or more complex models like the beta and gamma distributions, SciPy provides the necessary functions to calculate key statistical measures and perform in-depth analysis of continuous data.

---

## 91. SciPy - Discrete Probability Distributions

*Source: [https://www.tutorialspoint.com/scipy/scipy_discrete_probability_distributions.htm](https://www.tutorialspoint.com/scipy/scipy_discrete_probability_distributions.htm)*

---

---
[Previous](/scipy/scipy_continuous_probability_distributions.htm)[Quiz](/scipy/quiz_on_scipy_discrete_probability_distributions.htm)[Next](/scipy/scipy_statistical_tests_interference.htm)**Discrete probability distributions**refer to statistical models where the random variable can take a finite or countable set of values, often integers. These distributions are widely used in various fields like computer science, engineering and operations research to model phenomena such as successes in trials, event occurrences or sampling outcomes.
The
**scipy.stats**library in Python provides an extensive collection of tools for working with these distributions, enabling us to calculate probability mass functions (PMF), cumulative distribution functions (CDF) and perform random sampling.
## Key Discrete Distributions in SciPy

In SciPy, discrete distributions model random variables that can take specific values. SciPy provides a variety of discrete probability distributions along with methods for analyzing them.

## Binomial Distribution

The
**Binomial Distribution**describes the number of successes in a specified number of independent trials where each trial has the same probability of success. It is frequently applied in scenarios such as coin toss experiments or quality control assessments.
### Example

In SciPy, the binomial distribution is represented by the
**scipy.stats.binom**object. Below is an example which shows how to calculate and visualize the Probability Mass Function (PMF) and Cumulative Distribution Function (CDF) for a binomial distribution −
```
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# Parameters: n = trials, p = probability of success
n, p = 10, 0.5

# Generate an array of outcomes
x_values = np.arange(0, n + 1)

# Compute the PMF and CDF
pmf_values = binom.pmf(x_values, n, p)
cdf_values = binom.cdf(x_values, n, p)

# Plot the results
plt.figure(figsize=(12, 6))

# PMF plot
plt.subplot(1, 2, 1)
plt.bar(x_values, pmf_values, label='PMF', alpha=0.7, color='blue')
plt.title('Binomial Distribution - PMF')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.legend()

# CDF plot
plt.subplot(1, 2, 2)
plt.step(x_values, cdf_values, label='CDF', color='red', where='mid')
plt.title('Binomial Distribution - CDF')
plt.xlabel('Number of Successes')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Here is the output of the Binomial Distribution computed using the functions
**scipy.stats.binom.pmf()**and**scipy.stats.binom.cdf()**−![Binomial Distribution](/scipy/images/binomial_discrete.jpg)
## Poisson Distribution

The
**Poisson Distribution**represents the count of events happening within a specific time or space interval by assuming the events occur independently and at a consistent average rate. It is commonly applied in areas like queueing systems, telecommunications and traffic analysis.
### Example

In SciPy the Poisson distribution is available through the
**scipy.stats.poisson()**module. The following example shows how to compute and visualize the Probability Mass Function (PMF) and the Cumulative Distribution Function (CDF) for a Poisson distribution −
```
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

# Parameter: lambda (mean rate of events)
mu = 3

# Generate an array of outcomes
x_values = np.arange(0, 15)

# Compute the PMF and CDF
pmf_values = poisson.pmf(x_values, mu)
cdf_values = poisson.cdf(x_values, mu)

# Plot the results
plt.figure(figsize=(12, 6))

# PMF plot
plt.subplot(1, 2, 1)
plt.bar(x_values, pmf_values, label='PMF', alpha=0.7, color='blue')
plt.title('Poisson Distribution - PMF')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.legend()

# CDF plot
plt.subplot(1, 2, 2)
plt.step(x_values, cdf_values, label='CDF', color='red', where='mid')
plt.title('Poisson Distribution - CDF')
plt.xlabel('Number of Events')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Below is the output of the Poisson distribution calculated using
**scipy.stats.poisson.pmf()**and**scipy.stats.poisson.cdf()**function −![Poisson Distribution](/scipy/images/poisson_distribution.jpg)
## Geometric Distribution

The
**Geometric Distribution**models the number of trials needed to get the first success in a series of independent Bernoulli trials in which each with a constant probability of success. This distribution is often used in areas like reliability testing and survival analysis.
### Example

In SciPy, the geometric distribution is represented by the
**scipy.stats.geom**module. The following example illustrates how to compute and plot the Probability Mass Function (PMF) and Cumulative Distribution Function (CDF) for the geometric distribution −
```
from scipy.stats import geom
import numpy as np
import matplotlib.pyplot as plt

# Parameter: probability of success
p = 0.3

# Generate an array of outcomes
x_values = np.arange(1, 11)

# Compute the PMF and CDF
pmf_values = geom.pmf(x_values, p)
cdf_values = geom.cdf(x_values, p)

# Plot the results
plt.figure(figsize=(12, 6))

# PMF plot
plt.subplot(1, 2, 1)
plt.bar(x_values, pmf_values, label='PMF', alpha=0.7, color='blue')
plt.title('Geometric Distribution - PMF')
plt.xlabel('Trials')
plt.ylabel('Probability')
plt.legend()

# CDF plot
plt.subplot(1, 2, 2)
plt.step(x_values, cdf_values, label='CDF', color='red', where='mid')
plt.title('Geometric Distribution - CDF')
plt.xlabel('Trials')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Output

Here is the output of the geometric distribution calculated using
**scipy.stats.geom.pmf()**and**scipy.stats.geom.cdf()**function −![Geometric Distribution](/scipy/images/geometric_distribution.jpg)
## Working with Discrete Distributions in SciPy

SciPy provides powerful methods to work with discrete distributions such as −

- **PMF (Probability Mass Function)**− distribution.pmf(x, params) gives the probability of observing a specific outcome x.
- **CDF (Cumulative Distribution Function)**− distribution.cdf(x, params) calculates the cumulative probability up to x.
- **Random Sampling**− distribution.rvs(params, size=N) generates N random values from the distribution.
- **Mean and Variance**− distribution.mean() and distribution.var() compute the mean and variance of the distribution.
### Example

For example the mean and variance of a Poisson distribution can be calculated as follows −

```
from scipy.stats import poisson

# Calculate the mean and variance of the Poisson distribution
mu = 3
mean = poisson.mean(mu)
variance = poisson.var(mu)

print("Mean of Poisson Distribution:", mean)
print("Variance of Poisson Distribution:", variance)
```

#### Output

Below is the output of the mean and variance of the Poisson distribution −

```
Mean of Poisson Distribution: 3.0
Variance of Poisson Distribution: 3.0
```

With the
**scipy.stats**module we can efficiently analyze and work with discrete probability distributions to solve a variety of real-world problems by ranging from modeling successes in trials to analyzing event occurrences.

---

## 92. SciPy - Statistical and Tests Inference

*Source: [https://www.tutorialspoint.com/scipy/scipy_statistical_tests_interference.htm](https://www.tutorialspoint.com/scipy/scipy_statistical_tests_interference.htm)*

---

---
[Previous](/scipy/scipy_discrete_probability_distributions.htm)[Quiz](/scipy/quiz_on_scipy_statistical_tests_interference.htm)[Next](/scipy/scipy_generating_random_samples.htm)**Statistical tests and inference**involve deriving conclusions about a population from sample data. These methodologies are fundamental for validating hypotheses, analyzing data trends, and making informed decisions in research, economics, engineering and many other fields. SciPys**scipy.stats**module offers a comprehensive set of tools to perform various statistical tests and data inferences.
## Important Statistical Tests in SciPy

The
**scipy.stats**library in Python includes a variety of functions to execute tests such as t-tests, chi-square tests and ANOVA, helping you validate assumptions and test hypotheses in different applications.
SciPy provides several statistical tests designed to assess different types of data and determine if observed differences or relationships are statistically significant. These tests play a critical role in hypothesis testing and analysis.

## t-Test

A
**t-test**is used to assess whether the means of two groups are different from one another typically applied in situations like comparing the results of two sample groups. The function**scipy.stats.ttest_ind()**can be used to perform a t-test on two independent samples.
### Example

The following example demonstrates how to perform a t-test on two datasets −

```
from scipy.stats import ttest_ind
import numpy as np

# Generate sample data
group1 = np.random.normal(0, 1, 100)
group2 = np.random.normal(0.5, 1, 100)

# Conduct the t-test
stat, p_value = ttest_ind(group1, group2)

print(f"t-statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

#### Output

Here is the result of the t-test showing the t-statistic and p-value which help us to determine if the differences between the two groups are statistically significant −

```
t-statistic: -3.1020
p-value: 0.0022
```

## Chi-Squared Test

The
**Chi-Squared Test**is typically used to analyze categorical data, determining whether there is an association between two categorical variables. It's useful in situations like contingency tables where data is grouped into categories.
### Example

To perform Chi-Squared Test, SciPy provides the
**scipy.stats.chi2_contingency()**function −
```
from scipy.stats import chi2_contingency
import numpy as np

# Example data in a contingency table
data = np.array([[10, 20], [20, 30]])

# Run the chi-squared test
chi2_stat, p_val, dof, expected = chi2_contingency(data)

print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"p-value: {p_val:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Expected values: \n{expected}")
```

#### Output

Below is the output of the Chi-squared test showing the statistic, p-value, degrees of freedom, and expected values:

```
Chi-squared statistic: 0.1280
p-value: 0.7205
Degrees of freedom: 1
Expected values:
[[11.25 18.75]
 [18.75 31.25]]
```

## ANOVA (Analysis of Variance)
**ANOVA**tests whether there are significant differences among the means of three or more groups. It's useful when comparing multiple datasets to determine if at least one of them is different from the others.
### Example

To perform a one-way ANOVA we can use the
**scipy.stats.f_oneway()**function, following is the example which performs the Annova test −
```
from scipy.stats import f_oneway
import numpy as np

# Example data from three groups
group1 = np.random.normal(0, 1, 100)
group2 = np.random.normal(1, 1, 100)
group3 = np.random.normal(2, 1, 100)

# Run one-way ANOVA
f_stat, p_value = f_oneway(group1, group2, group3)

print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

#### Output

Here is the result of the ANOVA test showing the F-statistic and p-value, which help us assess whether the group means are statistically different:

```
F-statistic: 75.5012
p-value: 0.0000
```

## Normality Tests

### Example

To determine if a dataset follows a normal distribution we can use normality tests like the
**Shapiro-Wilk Test**or**D'Agostino and Pearson's Test**available in SciPy. The**scipy.stats.shapiro()**function conducts the Shapiro-Wilk test to check normality −
```
from scipy.stats import shapiro
import numpy as np

# Example data
data = np.random.normal(0, 1, 100)

# Perform Shapiro-Wilk normality test
stat, p_value = shapiro(data)

print(f"Test statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

#### Output

Following is the output of the Shapiro-Wilk test helps to evaluate if the sample data is consistent with a normal distribution −

```
Test statistic: 0.9878
p-value: 0.4939
```

## Using Statistical Inference in SciPy

SciPy provides essential tools for making inferences about a population from sample data, such as −

- **p-value**− This is used to determine the statistical significance of test results. A p-value below a threshold (commonly 0.05) suggests a significant result.
- **Confidence Intervals**− Estimate the range in which a population parameter (such as the mean) lies based on sample data.
- **Effect Size**− Quantifies the magnitude of an observed effect or difference.
Using these methods the researchers can perform thorough statistical analyses and make decisions backed by solid evidence from their data.

---

## 93. SciPy - Generating Random Samples

*Source: [https://www.tutorialspoint.com/scipy/scipy_generating_random_samples.htm](https://www.tutorialspoint.com/scipy/scipy_generating_random_samples.htm)*

---

---
[Previous](/scipy/scipy_statistical_tests_interference.htm)[Quiz](/scipy/quiz_on_scipy_generating_random_samples.htm)[Next](/scipy/scipy_kaplan_meier_estimator_survival_analysis.htm)**Generating random samples**in SciPy refers to the process of drawing values from predefined probability distributions using the**scipy.stats**module. SciPy provides a wide range of continuous and discrete probability distributions such as the normal, uniform, binomial and exponential distributions. The**.rvs()**(random variates) method is used to generate these samples while maintaining the statistical properties of the chosen distribution.
Mathematically, a random sample
**X**is drawn from a probability distribution**f(x)**where,
```
X  f(x,)
```

Here
represents the parameters of the distribution such as mean and standard deviation for a normal distribution.
Random sampling is essential in simulations, statistical modeling and machine learning. It allows researchers to approximate real-world uncertainties, perform Monte Carlo simulations and conduct hypothesis testing.

Here are the key parameters in include −

- **loc**− Represents the mean or starting value of the distribution.
- **scale**− Controls the spread or range.
- **size**− Specifies the number of samples to generate.
- **random_state**− Ensures reproducibility by fixing the random seed.
For example
**norm.rvs(loc=0, scale=1, size=10)**generates 10 random numbers from a standard normal distribution. This functionality makes SciPy a powerful tool for probabilistic data analysis and simulations.
## Generating Random Samples from Different Distributions

SciPy provides powerful tools for generating random samples from various probability distributions through the scipy.stats module. This is widely used in statistical analysis, simulations and machine learning.

## Normal (Gaussian) Distribution

The Normal distribution is also known as the Gaussian distribution which is one of the most important probability distributions in statistics. It is widely used in real-world applications such as finance, physics, biology and machine learning.

SciPy provides the
**norm.rvs()**function within**scipy.stats**module to generate random samples from a normal distribution.
### Syntax

Following is the syntax of SciPy's norm.rvs() function which is used to generate random samples from a normal distribution −

```
scipy.stats.norm.rvs(loc=mean, scale=std_dev, size=n, random_state=seed)
```

### parameters

Here are the parameters of the function
**scipy.stats.norm.rvs()**−
- **loc**− The Mean () of the distribution.
- **scale**− The Standard deviation () of the distribution.
- **size**− The number of random samples to generate.
- **random_state**− An optional seed for reproducibility.
### Example

Following is the example in which we generate 1,000 random samples from a normal distribution with a mean of 0 and a standard deviation of 1. The histogram of the samples is plotted alongside the theoretical probability density function (PDF) to illustrate the distribution −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
 = 0    # Mean
 = 1    # Standard deviation
n = 1000 # Number of samples

# Generate random samples
samples = norm.rvs(loc=, scale=, size=n, random_state=42)

# Plot histogram
plt.hist(samples, bins=30, density=True, alpha=0.6, color='b')

# Plot the theoretical PDF
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, loc=, scale=), 'r', lw=2)

plt.title("Normal Distribution Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

#### Output

Following is the output of the random samples generated from Normal Distribution −
![Normal Distribution Samples](/scipy/images/normal_distribution_samples.jpg)
## Uniform Distribution

The Uniform distribution is a probability distribution where all outcomes are equally likely within a given range. It is commonly used in simulations, random sampling and statistical modeling.

SciPy provides the
**uniform.rvs()**function within the**scipy.stats**module to generate random samples from a uniform distribution.
### Syntax

Following is the syntax of SciPy's uniform.rvs() function which is used to generate random samples from a uniform distribution −

```
scipy.stats.uniform.rvs(loc=a, scale=b-a, size=n, random_state=seed)
```

### parameters

Here are the parameters of the function
**scipy.stats.uniform.rvs()**−
- **loc**− The lower bound (a) of the distribution.
- **scale**− The range (b - a) of the distribution.
- **size**− The number of random samples to generate.
- **random_state**− An optional seed for reproducibility.
### Example

Following is the example in which we generate 1,000 random samples from a uniform distribution between 0 and 10. The histogram of the samples is plotted alongside the theoretical probability density function (PDF) to illustrate the distribution −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Parameters
a = 0    # Lower bound
b = 10   # Upper bound
n = 1000 # Number of samples

# Generate random samples
samples = uniform.rvs(loc=a, scale=b-a, size=n, random_state=42)

# Plot histogram
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')

# Plot the theoretical PDF
x = np.linspace(a, b, 100)
plt.plot(x, uniform.pdf(x, loc=a, scale=b-a), 'r', lw=2)

plt.title("Uniform Distribution Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

#### Output

Below is the output of the random samples generated from Uniform Distribution −
![Uniform Distribution Samples](/scipy/images/uniform_distribution_samples.jpg)
## Exponential Distribution

The Exponential distribution is a continuous probability distribution that describes the time between events in a Poisson process where events occur independently at a constant average rate. It is widely applied in reliability engineering, queuing systems and survival analysis.

SciPy provides the
**expon.rvs()**function within the**scipy.stats**module to generate random samples following an exponential distribution.
### Syntax

The following is the syntax for SciPy's
**expon.rvs()**function, which is used to generate random samples from an exponential distribution −
```
scipy.stats.expon.rvs(scale=1/lambda, size=n, random_state=seed)
```

### Parameters

The
**scipy.stats.expon.rvs()**function accepts the following parameters −
- **scale**− The reciprocal of the rate parameter (1/), which defines the mean time between events.
- **size**− The number of random samples to be generated.
- **random_state**− Optional seed value to ensure reproducibility of results.
### Example

Below is the example which helps, how to generate 1,000 random samples from an exponential distribution with a rate parameter () of 1. The histogram of the samples is plotted along with the probability density function (PDF) to visualize the distribution −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Define parameters
 = 1    # Rate parameter
n = 1000 # Number of samples

# Generate random samples
samples = expon.rvs(scale=1/, size=n, random_state=42)

# Plot histogram
plt.hist(samples, bins=30, density=True, alpha=0.6, color='purple')

# Plot the theoretical PDF
x = np.linspace(0, 8, 100)
plt.plot(x, expon.pdf(x, scale=1/), 'r', lw=2)

plt.title("Exponential Distribution Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

#### Output

Here is the output which gives the random samples generated from the Exponential Distribution −
![Exponential Distribution Samples](/scipy/images/exponential_distribution_samples.jpg)
## Binomial Distribution

The Binomial distribution is a probability distribution that represents the number of successful outcomes in a fixed number of independent trials. Each trial results in one of two possible outcomes: success or failure, with a constant probability of success.

This distribution is commonly applied in real-world scenarios where outcomes are binary such as evaluating product defects in a manufacturing process or counting how many times a coin lands on heads in multiple tosses.

In SciPy, the
**binom.rvs()**function, available within the**scipy.stats**module, allows users to generate random values that follow a binomial pattern.
### Syntax

The following is the syntax for the
**binom.rvs()**function, which is used to create random values that follow a binomial pattern −
```
scipy.stats.binom.rvs(n=trials, p=probability, size=samples, random_state=seed)
```

### Parameters

The function
**binom.rvs()**includes several input parameters as mentioned follows −
- **n**− Represents the number of trials or attempts.
- **p**− The probability of achieving success in a single trial.
- **size**− Specifies the number of random values to generate.
- **random_state**− An optional parameter to set a fixed seed for reproducibility.
### Example

In the following example we create 1,000 random values from a binomial distribution where there are 10 trials, and the probability of success in each trial is 0.5. The generated values are then displayed using a histogram to show the distribution pattern −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Define the parameters
num_trials = 10   # Number of experiments
success_prob = 0.5  # Probability of success per trial
num_samples = 1000  # Total samples to generate

# Generate the binomially distributed random values
data = binom.rvs(n=num_trials, p=success_prob, size=num_samples, random_state=42)

# Plot a histogram of the generated values
plt.hist(data, bins=10, density=True, alpha=0.6, color='g')

plt.title("Generated Binomial Distribution Data")
plt.xlabel("Number of Successful Outcomes")
plt.ylabel("Probability Density")
plt.show()
```

#### Output

Following is the output which represents the binomially distributed random samples −
![Binomial Distribution Samples](/scipy/images/binomial_distribution_samples.jpg)
## Poisson distribution

The Poisson distribution is a statistical model that represents the frequency of an event occurring within a fixed period of time or space. It is applicable in scenarios where occurrences are random, independent and happen at a constant average rate.

This distribution is commonly used in real-world cases such as estimating the number of customer calls received by a support center per hour by tracking the footfall at a store within a set duration or analyzing the frequency of emails arriving in an inbox daily.

In SciPy, the
**poisson.rvs()**function from the**scipy.stats**module allows users to generate random samples that follow a Poisson distribution.
### Syntax

The following is the syntax for the
**poisson.rvs()**function which generates random values following a Poisson distribution −
```
scipy.stats.poisson.rvs(mu=rate, size=samples, random_state=seed)
```

### Parameters

The function
**poisson.rvs()**includes several input parameters as mentioned follows −
- **mu**− The expected number of occurrences (mean rate of events).
- **size**− Specifies the number of random values to generate.
- **random_state**− An optional parameter to set a fixed seed for reproducibility.
### Example

Here in this example we generate 1,000 random values from a Poisson distribution with an average event rate of 5 per unit time. The generated values are then displayed using a histogram to illustrate the distribution pattern −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define the parameters
event_rate = 5   # Average occurrences per time unit
num_samples = 1000  # Total samples to generate

# Generate Poisson-distributed random values
data = poisson.rvs(mu=event_rate, size=num_samples, random_state=42)

# Plot a histogram of the generated values
plt.hist(data, bins=15, density=True, alpha=0.6, color='b')

plt.title("Generated Poisson Distribution Data")
plt.xlabel("Number of Events")
plt.ylabel("Probability Density")
plt.show()
```

#### Output

The following graph represents the Poisson-distributed random samples −
![Poisson Distribution Samples](/scipy/images/poisson_distribution_samples.jpg)
## Setting a Seed for Reproducibility

When generating random numbers in Python then the output changes with each execution. This can be problematic for debugging, testing or sharing results. To ensure consistency we use a seed value to initialize the random number generator. This makes the random output reproducible across multiple runs.

In SciPy, functions like binom.rvs() for generating binomially distributed random values include the random_state parameter. Setting this parameter to a fixed integer ensures that the generated values remain the same each time the code is executed.

### Example

The following example demonstrates how setting a seed ensures that the same random values are generated in multiple runs:

```
import numpy as np
from scipy.stats import binom

# Define parameters
num_trials = 10    # Number of experiments
success_prob = 0.5  # Probability of success per trial
num_samples = 10   # Total samples to generate

# Generate random values with a fixed seed
seed_value = 42
samples1 = binom.rvs(n=num_trials, p=success_prob, size=num_samples, random_state=seed_value)
samples2 = binom.rvs(n=num_trials, p=success_prob, size=num_samples, random_state=seed_value)

# Display the results
print("First Run:", samples1)
print("Second Run:", samples2)

# Verify if both runs produce the same output
print("Are both runs identical?", np.array_equal(samples1, samples2))
```

#### Output

The output will confirm that setting a seed generates the same values every time −

```
First Run: [4 8 6 5 3 3 3 7 5 6]
Second Run: [4 8 6 5 3 3 3 7 5 6]
Are both runs identical? True
```

---

## 94. SciPy - Kaplan-Meier Estimator Survival Analysis

*Source: [https://www.tutorialspoint.com/scipy/scipy_kaplan_meier_estimator_survival_analysis.htm](https://www.tutorialspoint.com/scipy/scipy_kaplan_meier_estimator_survival_analysis.htm)*

---

---
[Previous](/scipy/scipy_generating_random_samples.htm)[Quiz](/scipy/quiz_on_scipy_kaplan_meier_estimator_survival_analysis.htm)[Next](/scipy/scipy_cox_proportional_hazards_model.htm)
The
**Kaplan-Meier estimator**is a statistical method used to estimate the probability of survival over time, especially when dealing with censored data, where some subjects' event times are unknown due to loss of follow-up. Its commonly applied in survival analysis in medical research, reliability engineering and other fields.
SciPy doesn't directly implement the Kaplan-Meier estimator but we can use Python's lifelines library to perform the estimation and visualize the survival function. Below is an explanation of the Kaplan-Meier method and how we can apply it.

## Basics of Kaplan-Meier Estimator

The Kaplan-Meier estimator calculates the survival function which represents the probability that a subject survives past a given time. It does so by considering both observed event times and censored data points i.e., when an event hasn't occurred but the subject is lost to follow-up.

The estimator is defined by a step-function curve that steps down each time an event occurs. The survival probability decreases when a failure or event happens and stays flat between events.

## Implementation Using lifelines library

Though the SciPy library provides robust numerical and statistical functions, for survival analysis like Kaplan-Meier, lifelines is more efficient and designed specifically for this purpose. Heres how we can implement the Kaplan-Meier estimator using lifelines library−

### Install lifelines

First we need to install the
**lifelines**library with the help of command prompt by using the below command −
```
pip install lifelines
```

## Manual Implementation of the Kaplan-Meier Estimator

The Kaplan-Meier estimator is used to estimate the survival function from lifetime data. It is commonly used in survival analysis and is a non-parametric method to estimate the probability of survival over time.

### Example

Heres how we can compute the Kaplan-Meier estimator using Python, without any external library but optionally using lifelines for simplicity −

```
import numpy as np
import pandas as pd

# Define function to compute Kaplan-Meier estimator
def kaplan_meier_estimator(event_times, events_observed):
    # Sort the event times
    event_times_sorted = np.sort(event_times)
    
    # Initialize the number at risk (initially everyone is at risk)
    n_risk = len(event_times_sorted)
    
    # Initialize the survival probabilities
    survival_probs = []
    previous_survival_prob = 1.0
    
    # Iterate through the event times to calculate survival probabilities
    for time in event_times_sorted:
        # Number of events at this time (number of deaths)
        deaths = np.sum((event_times == time) & (events_observed == 1))
        
        # Number of individuals at risk just before this time
        risk_set = np.sum(event_times >= time)
        
        # Update the survival probability
        survival_prob = previous_survival_prob * (1 - deaths / risk_set)
        survival_probs.append(survival_prob)
        
        # Update previous survival probability
        previous_survival_prob = survival_prob
        
    # Create a DataFrame for the Kaplan-Meier estimate
    km_df = pd.DataFrame({
        'Event Time': event_times_sorted,
        'Survival Probability': survival_probs
    })
    
    return km_df

# Example Data
event_times = np.array([5, 6, 6, 2, 4, 3, 8, 10, 7])  # Time to event (or censoring)
events_observed = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1])  # 1 = event observed, 0 = censored

# Compute the Kaplan-Meier estimator
km_df = kaplan_meier_estimator(event_times, events_observed)

print(km_df)
```

#### Output

Following is the output of the manual implementation of the Kaplan-Meier estimator without using the lifelines library −

```
Event Time  Survival Probability
0           2              0.888889
1           3              0.777778
2           4              0.777778
3           5              0.648148
4           6              0.518519
5           6              0.414815
6           7              0.276543
7           8              0.138272
8          10              0.138272
```

## Using lifelines library

### Example

If we want to use a more straightforward library to compute the Kaplan-Meier estimator, we can use the lifelines library which we installed before −

```
from lifelines import KaplanMeierFitter
import numpy as np

# Example Data
event_times = np.array([5, 6, 6, 2, 4, 3, 8, 10, 7])  # Time to event (or censoring)
events_observed = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1])  # 1 = event observed, 0 = censored

# Instantiate the KaplanMeierFitter
kmf = KaplanMeierFitter()

# Fit the model to the data
kmf.fit(event_times, event_observed=events_observed)

# Plot the Kaplan-Meier estimator
kmf.plot()

# Display the survival probabilities at each time point
print(kmf.survival_function_)
```

#### Output

Here is the output of the Kaplan-Meier estimator computed using the lifelines library −

```
KM_estimate
timeline
0.0          1.000000
2.0          0.888889
3.0          0.777778
4.0          0.777778
5.0          0.648148
6.0          0.518519
7.0          0.345679
8.0          0.172840
10.0         0.172840
```

## Customizing the Plot

### Example

After computing the Kaplan-Meier we can adjust the plot by customizing the title, labels and line style. For example we can change the color or make the line dashed to highlight different parts of the curve. Below is the code −

```
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Example data: event times and censoring indicators
event_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
censored = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0])  # 1: event, 0: censored

# Initialize the Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Fit the model to the data
kmf.fit(event_times, event_observed=censored)

# Customize the Kaplan-Meier plot
kmf.plot_survival_function(color='blue', linestyle='-', label='Survival Curve')

# Adding titles and axis labels
plt.title('Customized Kaplan-Meier Survival Curve')
plt.xlabel('Time (Months)')
plt.ylabel('Survival Probability')

# Display the plot
plt.show()
```

#### Output

Following is the output of the Customized Kaplan-Meier estimator plot computed using the lifelines library −
![Kaplan-Meier customized plot](/scipy/images/customized_kaplan_meier.jpg)

---

## 95. SciPy - Cox Proportional Hazards Model Survival Analysis

*Source: [https://www.tutorialspoint.com/scipy/scipy_cox_proportional_hazards_model.htm](https://www.tutorialspoint.com/scipy/scipy_cox_proportional_hazards_model.htm)*

---

---
[Previous](/scipy/scipy_kaplan_meier_estimator_survival_analysis.htm)[Quiz](/scipy/quiz_on_scipy_cox_proportional_hazards_model.htm)[Next](/scipy/scipy_spatial.htm)
The
**Cox Proportional Hazards Model**is a popular statistical method used for survival analysis. It helps estimate the effect of various variables on the time it takes for an event such as failure or death to occur. This model is particularly valuable when dealing with censored data where some individuals may not have experienced the event by the end of the study.
Although SciPy doesn't have a built-in Cox Proportional Hazards model, the lifelines library which is based on SciPy, offers comprehensive support for survival analysis, including the Cox Proportional Hazards model.

## Steps of Cox Proportional Hazards in Python using lifelines

Following are the steps that need to be followed to implement the Cox Proportional Hazards in Python using lifelines −

### Install Lifeline Library

First we have to install the lifeline library by executing the below command in the command prompt, if haven't installed before −

```
pip install lifelines
```

### Importing the Libraries

After that we have to import all the necessary libraries −

```
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
```

### Prepare our Data

For the Cox Proportional Hazards model our dataset should include at least two components which are mentioned as follows −

- **Duration,**which is the time until the event or censoring occurs.
- **Event/Censoring Indicator,**1 if the event occurred, 0 if the observation was censored.
Below is the example dataset to implement the
**Cox Proportional Hazards Model**−
```
# Example dataset
data = {
    'age': [60, 65, 70, 80, 85],
    'sex': [1, 0, 1, 1, 0],  # 1 for male, 0 for female
    'duration': [5, 6, 7, 8, 9],  # Duration in years
    'event': [1, 0, 1, 1, 0]  # 1 for event (e.g., death), 0 for censored
}
# Create a DataFrame
df = pd.DataFrame(data)
```

### Fit the ox Proportional Hazards Model

Here we will display a summary of the model with estimated coefficients, standard errors, z-scores and p-values for each predictor variable which helps us to understand their effects on the hazard ratio.

```
# Instantiate the Cox Proportional Hazards model
cph = CoxPHFitter()

# Fit the model with the dataset
cph.fit(df, duration_col='duration', event_col='event')

# Print the model summary
cph.print_summary()
```

#### Making Predictions

Once the model is fitted we can predict survival functions for new data or calculate the cumulative hazard.

```
# Predict the survival function for a new individual
new_data = pd.DataFrame({
    'age': [75],
    'sex': [1],
})

# Predict survival for the new individual
survival_function = cph.predict_survival_function(new_data)
print(survival_function)
```

### Example

Following is the example which simulates a dataset with different risk factors such as age, gender, smoking status and analyzes their effect on survival time −

```
import pandas as pd
from lifelines import CoxPHFitter

# Simulated dataset: Survival data with age, sex, and smoking status
data = {
    'age': [60, 62, 65, 70, 72, 75, 80, 85],      # Age of individuals
    'sex': [1, 0, 1, 1, 0, 1, 0, 1],              # 1 for male, 0 for female
    'smoking_status': [1, 0, 1, 0, 1, 0, 1, 0],   # 1 for smoker, 0 for non-smoker
    'duration': [5, 6, 7, 8, 9, 5, 6, 7],         # Survival time in years
    'event': [1, 1, 0, 1, 1, 0, 1, 1]             # 1 for event (e.g., death), 0 for censored
}

# Create DataFrame
df = pd.DataFrame(data)

# Instantiate the Cox Proportional Hazards model
cph = CoxPHFitter()

# Fit the model with the dataset
cph.fit(df, duration_col='duration', event_col='event')

# Print the model summary to analyze the results
cph.print_summary()

# Predict the survival function for a new individual (e.g., 70-year-old smoker)
new_data = pd.DataFrame({
    'age': [70],
    'sex': [1],          # Male
    'smoking_status': [1],  # Smoker
})

# Predict survival for the new individual
survival_function = cph.predict_survival_function(new_data)

# Print the survival function for the new individual
print(survival_function)
```

Here is the output of the above example −

```
< lifelines.CoxPHFitter: fitted with 8 total observations, 2 right-censored observations >
             duration col = 'duration'
                event col = 'event'
      baseline estimation = breslow
   number of observations = 8
number of events observed = 6
   partial log-likelihood = -7.39
         time fit was run = 2025-01-31 12:06:31 UTC

---
                coef exp(coef)  se(coef)  coef lower 95%  coef upper 95% exp(coef) lower 95% exp(coef) upper 95%
covariate
age            -0.02      0.98      0.06           -0.14            0.11                0.87                1.12
sex            -0.23      0.79      1.06           -2.31            1.84                0.10                6.32
smoking_status -0.55      0.58      1.03           -2.57            1.47                0.08                4.33

                cmp to     z    p  -log2(p)
covariate
age               0.00 -0.26 0.80      0.33
sex               0.00 -0.22 0.83      0.28
smoking_status    0.00 -0.54 0.59      0.76
---
Concordance = 0.47
Partial AIC = 20.79
log-likelihood ratio test = 0.33 on 3 df
-log2(p) of ll-ratio test = 0.07

            0
5.0  0.918415
6.0  0.734865
7.0  0.610552
8.0  0.435392
9.0  0.191870
```

---

## 96. SciPy - Spatial

*Source: [https://www.tutorialspoint.com/scipy/scipy_spatial.htm](https://www.tutorialspoint.com/scipy/scipy_spatial.htm)*

---

---
[Previous](/scipy/scipy_cox_proportional_hazards_model.htm)[Quiz](/scipy/quiz_on_scipy_spatial.htm)[Next](/scipy/scipy_special_package.htm)
The
**scipy.spatial package**can compute Triangulations, Voronoi Diagrams and Convex Hulls of a set of points, by leveraging the**Qhull library**. Moreover, it contains**KDTree implementations**for nearest-neighbor point queries and utilities for distance computations in various metrics.
## Delaunay Triangulations

Let us understand what Delaunay Triangulations are and how they are used in SciPy.

### What are Delaunay Triangulations?

In mathematics and computational geometry, a Delaunay triangulation for a given set
**P**of discrete points in a plane is a triangulation**DT(P)**such that no point in**P**is inside the circumcircle of any triangle in DT(P).
We can the compute the same through SciPy. Let us consider the following example.

```
from scipy.spatial import Delaunay
points = np.array([[0, 4], [2, 1.1], [1, 3], [1, 2]])
tri = Delaunay(points)
import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
```

#### Output

The above program will generate the following output.
![Delaunay Triangulations](/scipy/images/delaunay_triangulations.jpg)
## Coplanar Points

Let us understand what Coplanar Points are and how they are used in SciPy.

### What are Coplanar Points?

Coplanar points are three or more points that lie in the same plane. Recall that a plane is a flat surface, which extends without end in all directions. It is usually shown in math textbooks as a four-sided figure.

Let us see how we can find this using SciPy. Let us consider the following example.

```
from scipy.spatial import Delaunay
points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
tri = Delaunay(points)
print tri.coplanar
```

#### Output

The above program will generate the following output.

```
array([[4, 0, 3]], dtype = int32)
```

This means that point 4 resides near triangle 0 and vertex 3, but is not included in the triangulation.

## Convex hulls

Let us understand what convex hulls are and how they are used in SciPy.

### What are Convex Hulls?

In mathematics, the
**convex hull**or**convex envelope**of a set of points X in the Euclidean plane or in a Euclidean space (or, more generally, in an affine space over the reals) is the smallest**convex set**that contains X.
Let us consider the following example to understand it in detail.

```
from scipy.spatial import ConvexHull
points = np.random.rand(10, 2) # 30 random points in 2-D
hull = ConvexHull(points)
import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
plt.plot(points[simplex,0], points[simplex,1], 'k-')
plt.show()
```

#### Output

The above program will generate the following output.
![Convex Hulls](/scipy/images/convex_hulls.jpg)

---

## 97. SciPy - Special Packages

*Source: [https://www.tutorialspoint.com/scipy/scipy_special_package.htm](https://www.tutorialspoint.com/scipy/scipy_special_package.htm)*

---

---
[Previous](/scipy/scipy_spatial.htm)[Quiz](/scipy/quiz_on_scipy_special_package.htm)[Next](/scipy/scipy_csgraph.htm)
The functions available in the special package are universal functions, which follow broadcasting and automatic array looping.

Let us look at some of the most frequently used special functions −

- 
Cubic Root Function

- 
Exponential Function

- 
Relative Error Exponential Function

- 
Log Sum Exponential Function

- 
Lambert Function

- 
Permutations and Combinations Function

- 
Gamma Function

Let us now understand each of these functions in brief.

### Cubic Root Function

The syntax of this cubic root function is  scipy.special.cbrt(x). This will fetch the element-wise cube root of
**x**.
Let us consider the following example.

```
from scipy.special import cbrt
res = cbrt([10, 9, 0.1254, 234])
print(res)
```

#### Output

The above program will generate the following output.

```
[ 2.15443469 2.08008382 0.50053277 6.16224015]
```

### Exponential Function

The syntax of the exponential function is  scipy.special.exp10(x). This will compute 10**x element wise.

Let us consider the following example.

```
from scipy.special import exp10
res = exp10([2, 9])
print(res)
```

#### Output

The above program will generate the following output.

```
[1.e+02 1.e+09]
```

### Relative Error Exponential Function

The syntax for this function is  scipy.special.exprel(x). It generates the relative error exponential, (exp(x) - 1)/x.

When
**x**is near zero, exp(x) is near 1, so the numerical calculation of exp(x) - 1 can suffer from catastrophic loss of precision. Then exprel(x) is implemented to avoid the loss of precision, which occurs when**x**is near zero.
Let us consider the following example.

```
from scipy.special import exprel
res = exprel([-0.25, -0.1, 0, 0.1, 0.25])
print(res)
```

#### Output

The above program will generate the following output.

```
[0.88479687 0.95162582 1.   1.05170918 1.13610167]
```

### Log Sum Exponential Function

The syntax for this function is  scipy.special.logsumexp(x). It helps to compute the log of the sum of exponentials of input elements.

Let us consider the following example.

```
from scipy.special import logsumexp
import numpy as np
a = np.arange(10)
res = logsumexp(a)
print(res)
```

#### Output

The above program will generate the following output.

```
9.45862974443
```

### Lambert Function

The syntax for this function is  scipy.special.lambertw(x). It is also called as the Lambert W function. The Lambert W function W(z) is defined as the inverse function of w * exp(w). In other words, the value of W(z) is such that z = W(z) * exp(W(z)) for any complex number z.

The Lambert W function is a multivalued function with infinitely many branches. Each branch gives a separate solution of the equation z = w exp(w). Here, the branches are indexed by the integer k.

Let us consider the following example. Here, the Lambert W function is the inverse of w exp(w).

```
from scipy.special import lambertw
import numpy as np
w = lambertw(1)
print(w)
print(w * np.exp(w))
```

#### Output

The above program will generate the following output.

```
(0.56714329041+0j)
(1+0j)
```

### Permutations & Combinations

Let us discuss permutations and combinations separately for understanding them clearly.
**Combinations**− The syntax for combinations function is  scipy.special.comb(N,k). Let us consider the following example −
```
from scipy.special import comb
res = comb(10, 3, exact = False,repetition=True)
print(res)
```

#### Output

The above program will generate the following output.

```
220.0
```
**Note**− Array arguments are accepted only for exact = False case. If k > N, N < 0, or k < 0, then a 0 is returned.**Permutations**− The syntax for combinations function is  scipy.special.perm(N,k). Permutations of N things taken k at a time, i.e., k-permutations of N. This is also known as partial permutations.
Let us consider the following example.

```
from scipy.special import perm
res = perm(10, 3, exact = True)
print(res)
```

#### Output

The above program will generate the following output.

```
720
```

### Gamma Function

The gamma function is often referred to as the generalized factorial since z*gamma(z) = gamma(z+1) and gamma(n+1) = n!, for a natural number n.

The syntax for combinations function is  scipy.special.gamma(x). Permutations of N things taken k at a time, i.e., k-permutations of N. This is also known as partial permutations.

The syntax for combinations function is  scipy.special.gamma(x). Permutations of N things taken k at a time, i.e., k-permutations of N. This is also known as partial permutations.

```
from scipy.special import gamma
res = gamma([0, 0.5, 1, 5])
print(res)
```

#### Output

The above program will generate the following output.

```
[inf  1.77245385  1.  24.]
```

---

## 98. SciPy - CSGraph

*Source: [https://www.tutorialspoint.com/scipy/scipy_csgraph.htm](https://www.tutorialspoint.com/scipy/scipy_csgraph.htm)*

---

---
[Previous](/scipy/scipy_special_package.htm)[Quiz](/scipy/quiz_on_scipy_csgraph.htm)[Next](/scipy/scipy_odr.htm)
CSGraph stands for
**Compressed Sparse Graph**, which focuses on Fast graph algorithms based on sparse matrix representations.
## Graph Representations

To begin with, let us understand what a sparse graph is and how it helps in graph representations.

### What exactly is a sparse graph?

A graph is just a collection of nodes, which have links between them. Graphs can represent nearly anything − social network connections, where each node is a person and is connected to acquaintances; images, where each node is a pixel and is connected to neighboring pixels; points in a high-dimensional distribution, where each node is connected to its nearest neighbors; and practically anything else you can imagine.

One very efficient way to represent graph data is in a sparse matrix: let us call it G. The matrix G is of size N x N, and G[i, j] gives the value of the connection between node i' and node j. A sparse graph contains mostly zeros − that is, most nodes have only a few connections. This property turns out to be true in most cases of interest.

The creation of the sparse graph submodule was motivated by several algorithms used in scikit-learn that included the following −

- **Isomap**− A manifold learning algorithm, which requires finding the shortest paths in a graph.
- **Hierarchical clustering**− A clustering algorithm based on a minimum spanning tree.
- **Spectral Decomposition**− A projection algorithm based on sparse graph laplacians.
As a concrete example, imagine that we would like to represent the following undirected graph −
![Undirected Graph](/scipy/images/undirected_graph.jpg)
This graph has three nodes, where node 0 and 1 are connected by an edge of weight 2, and nodes 0 and 2 are connected by an edge of weight 1. We can construct the dense, masked and sparse representations as shown in the following example, keeping in mind that an undirected graph is represented by a symmetric matrix.

```
import numpy as np

G_dense = np.array([ [0, 2, 1],
                     [2, 0, 0],
                     [1, 0, 0] ])
                     
G_masked = np.ma.masked_values(G_dense, 0)
from scipy.sparse import csr_matrix

G_sparse = csr_matrix(G_dense)
print(G_sparse.data)
```

The above program will generate the following output.

```
array([2, 1, 2, 1])
```
![Undirected Graph Using Symmetric Matrix](/scipy/images/undirected_graph_using_symmetric_matrix.jpg)
This is identical to the previous graph, except nodes 0 and 2 are connected by an edge of zero weight. In this case, the dense representation above leads to ambiguities − how can non-edges be represented, if zero is a meaningful value. In this case, either a masked or a sparse representation must be used to eliminate the ambiguity.

### Word ladders using sparse graphs

Word ladders is a game invented by Lewis Carroll, in which words are linked by changing a single letter at each step. For example −
**APE → APT → AIT → BIT → BIG → BAG → MAG → MAN**
Here, we have gone from "APE" to "MAN" in seven steps, changing one letter each time. The question is - Can we find a shorter path between these words using the same rule? This problem is naturally expressed as a sparse graph problem. The nodes will correspond to individual words, and we will create connections between words that differ by at the most  one letter.

## Obtaining a List of Words

First, of course, we must obtain a list of valid words. I am running Mac, and Mac has a word dictionary at the location given in the following code block. If you are on a different architecture, you may have to search a bit to find your system dictionary.

```
wordlist = open('/usr/share/dict/words').read().split()
print len(wordlist)
```

The above program will generate the following output.

```
235886
```

We now want to look at words of length 3, so let us select just those words of the correct length. We will also eliminate words, which start with upper case (proper nouns) or contain non-alpha-numeric characters such as apostrophes and hyphens. Finally, we will make sure everything is in lower case for a comparison later on.

```
wordlist = open('/usr/share/dict/words').read().split()

word_list = [word for word in word_list if len(word) == 3]
word_list = [word for word in word_list if word[0].islower()]
word_list = [word for word in word_list if word.isalpha()]
word_list = map(str.lower, word_list)
print len(word_list)
```

The above program will generate the following output.

```
1135
```

Now, we have a list of 1135 valid three-letter words (the exact number may change depending on the particular list used). Each of these words will become a node in our graph, and we will create edges connecting the nodes associated with each pair of words, which differs by only one letter.

```
import numpy as np
wordlist = open('/usr/share/dict/words').read().split()
word_list = np.asarray(word_list)

word_list.dtype
word_list.sort()

word_bytes = np.ndarray((word_list.size, word_list.itemsize),
   dtype = 'int8',
   buffer = word_list.data)
print(word_bytes.shape)
```

The above program will generate the following output.

```
(1135, 3)
```

We will use the Hamming distance between each point to determine, which pairs of words are connected. The Hamming distance measures the fraction of entries between two vectors, which differ: any two words with a hamming distance equal to 1/N1/N, where NN is the number of letters, which are connected in the word ladder.

```
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
hamming_dist = pdist(word_bytes, metric = 'hamming')
graph = csr_matrix(squareform(hamming_dist < 1.5 / word_list.itemsize))
```

When comparing the distances, we do not use equality because this can be unstable for floating point values. The inequality produces the desired result as long as no two entries of the word list are identical. Now, that our graph is set up, we will use the shortest path search to find the path between any two words in the graph.

```
wordlist = open('/usr/share/dict/words').read().split()
i1 = word_list.searchsorted('ape')
i2 = word_list.searchsorted('man')
print(word_list[i1],word_list[i2])
```

The above program will generate the following output.

```
ape, man
```

We need to check that these match, because if the words are not in the list there will be an error in the output. Now, all we need is to find the shortest path between these two indices in the graph. We will use
**dijkstras**algorithm, because it allows us to find the path for just one node.
```
from scipy.sparse.csgraph import dijkstra
distances, predecessors = dijkstra(graph, indices = i1, return_predecessors = True)
print(distances[i2])
```

The above program will generate the following output.

```
5.0
```

Thus, we see that the shortest path between ape and man contains only five steps. We can use the predecessors returned by the algorithm to reconstruct this path.

```
path = []
i = i2

while i != i1:
   path.append(word_list[i])
   i = predecessors[i]
   
path.append(word_list[i1])
print(path[::-1]i2])
```

The above program will generate the following output.

```
['ape', 'ope', 'opt', 'oat', 'mat', 'man']
```

---

## 99. SciPy - ODR

*Source: [https://www.tutorialspoint.com/scipy/scipy_odr.htm](https://www.tutorialspoint.com/scipy/scipy_odr.htm)*

---

---
[Previous](/scipy/scipy_csgraph.htm)[Quiz](/scipy/quiz_on_scipy_odr.htm)[Next](/scipy/scipy_reference.htm)
ODR stands for
**Orthogonal Distance Regression**, which is used in the regression studies. Basic linear regression is often used to estimate the relationship between the two variables**y**and**x**by drawing the line of best fit on the graph.
The mathematical method that is used for this is known as
**Least Squares**, and aims to minimize the sum of the squared error for each point. The key question here is how do you calculate the error (also known as the residual) for each point?
In a standard linear regression, the aim is to predict the Y value from the X value  so the sensible thing to do is to calculate the error in the Y values (shown as the gray lines in the following image). However, sometimes it is more sensible to take into account the error in both X and Y (as shown by the dotted red lines in the following image).

For example − When you know your measurements of X are uncertain, or when you do not want to focus on the errors of one variable over another.
![Orthogonal Distance linear regression](/scipy/images/orthogonal_distance_linear_regression.jpg)
Orthogonal Distance Regression (ODR) is a method that can do this (orthogonal in this context means perpendicular  so it calculates errors perpendicular to the line, rather than just vertically).

### scipy.odr Implementation for Univariate Regression

The following example demonstrates scipy.odr implementation for univariate regression.

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
import random

# Initiate some data, giving some randomness using random.random().
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([i**2 + random.random() for i in x])

# Define a function (quadratic in our case) to fit the data with.
def linear_func(p, x):
   m, c = p
   return m*x + c

# Create a model for fitting.
linear_model = Model(linear_func)

# Create a RealData object using our initiated data from above.
data = RealData(x, y)

# Set up ODR with the model and data.
odr = ODR(data, linear_model, beta0=[0., 1.])

# Run the regression.
out = odr.run()

# Use the in-built pprint method to give us results.
out.pprint()
```

#### Output

The above program will generate the following output.

```
Beta: [ 5.51846098 -4.25744878]
Beta Std Error: [ 0.7786442 2.33126407]

Beta Covariance: [
   [ 1.93150969 -4.82877433]
   [ -4.82877433 17.31417201
]]

Residual Variance: 0.313892697582
Inverse Condition #: 0.146618499389
Reason(s) for Halting:
   Sum of squares convergence
```

---

## 100. SciPy - Reference

*Source: [https://www.tutorialspoint.com/scipy/scipy_reference.htm](https://www.tutorialspoint.com/scipy/scipy_reference.htm)*

---

---
[Previous](/scipy/scipy_odr.htm)[Next](/scipy/scipy_quick_guide.htm)
The
**SciPy**is interconnected with[NumPy](/numpy/index.htm)which defines two libraries of[Python](/python/index.htm). These libraries create the foundation of data science and are utilized for analyzing datasets to address real-world problems. SciPy works on the built-in top of NumPy which provide advanced mathematics and scientific calculation.
Here, we provide the SciPy references that explain how to use the libraries. When you look at our list of methods in the table below, you will find details of all references that cover the proper introduction of specific methods by including introduction, syntax, parameters, return type, and various examples. In addition to this, we add the
[Matplotlib](/matplotlib/index.htm)library to SciPy code for plotting the graph.
## SciPy Constants Module

This module provide the physical and mathematical constants. Following are the methods of the SciPy
**Constants**−Sr.No.Types & Description1**nu2lambda()**
This method is used to convert the optical frequency into wavelength.
2**lambda2nu()**
This method is used to convert the wavelength into optical frequency.
3**convert_temperature()**
This method is used to calculate the temperature scale in various form.
4**value()**
This method defines the physical_constants dictionary which is indexed by a key.
5**unit()**
This method is defined by retrieving the specific unit from the physical constant using the dictionary.
6**precision()**
This method is defined by accessing the information of physical constants that includes values and units.
7**find()**
This method is defined by an array of elements indices which satisfy the given condition.

## SciPy Cluster Module

The cluster module provide the functionality related to cluster algorithm. Following are the methods of the SciPy
**Cluster**−Sr.No.Types & Description1**fcluster()**
This method is a part of hierarchical algorithm which group the data points into a specified number of cluster.
2**fclusterdata()**
This method grouped the similar data into cluster.
3**leaders()**
This method is used to identify the cluster center.
4**linkage()**
This method works on hierarchical cluster which can be used to perform the task of linkage matrix.
5**single()**
This method performs the task of single/minimimum/nearest linkage on a condensed matrix.
6**complete()**
This Method perform the task of complete linkage(largest point) on a condensed distance matrix.
7**average()**
This method is used to perform the task of arithmetic mean on a distance matrix.
8**weighted()**
This method depends on other functions which user can perform such as weighted means, weighted sums, and weighted operations.
9**centroid()**
This method define an one-dimensional array in which data values are calculated with the help of average weight and these weights itself represent a value.
10**median()**
This method is used to find the median value of an array.
11**ward()**
This method is a part of agglomerative cluster which minimize the total cluster variance within its control.
12**cophenet()**
This method calculates the cophenetic distance between each observation of the hierarchical cluster.
13**from_mlab_linkage()**
This method is used to work with clustering algorithm(mlab.linkage) and converts it into a format that can be used for the references of other scipy clustering functions.
14**inconsistent()**
This method is used to perform the calculation of inconsistency statistics on a linkage matrix.
15**maxinconsts()**
This method is used to calculate the distances between two datasets.
16**maxdists()**
This method calculate the pairwise distances between the points from the given set.
17**maxRstat()**
This method perform the task of maximum value obtained by a column R for each non-singleton cluster and its children.
18**to_mlab_linkage()**
This method is used to convert the clustering output into MATLAB compatible format.
19**dendrogram()**
This method determine its functionality by cutting clusters at a particular height.
20**set_link_color_palette()**
This method perform the task of matplotlib color codes while representing different level of clusters.
21**DisjointSet()**
This method is used to manage the data partition set into a disjoint subsets.

## SciPy Misc Module

This
**scipy.misc**module help us to print the images in gray-scale mode. Following are the methods of the SciPy**Misc**−Sr.No.Types & Description1**ascent()**
This method is used to get the 8-bit grayscale derieved image
2**face()**
This method is used to get the images of a racoon.
3**electrocardiogram()**
This method is used to represent the electrical activity of the heart.

## SciPy Integration Module

The
**scipy.integrate**module provides various methods to perform the operation of numerical integration. Following are the list of methods to understand its functionality −Sr.No.Types & Description1**integrate.quad()**
This method is used to perform the task of definite integrals.
2**integrate.quad_vec()**
This method is used to calculate the definite integrals of vector-value function.
3**integrate.dblquad()**
This is used to calculate the double numerical integration.
4**integrate.tplquad()**
This method is used to calculate the triple numerical integration.
5**integrate.nquad()**
This method is used to find the integration of multiple variable.
6**integrate.fixed_quad()**
This method operates the fixed order of gaussian quadrature for numerical integration.
7**integrate.quadrature()**
This method is used to calculate the numerical integration.
8**integrate.romberg()**
This method is used to calculate the numerical integration.
9**integrate.newton_cotes()**
This method is used to return the weights and error coefficient for Newton-Cotes integration.
10**integrate.trapezoid()**
This method is used to find the approximate value of integral function using trapezoid rule.
11**integrate.cumulative_trapezoid()**
This method is used to calculate the integral from the given set of points using trapezoidal rule.
12**integrate.simpson()**
This method is used to approximate the integral of a function using simpson rule.
13**integrate.cumulative_simpson()**
This method is used to calculate the coordinates at every pairs
14**integrate.romb()**
This method is used to perform the task of numerical or romberg integration.

## SciPy Datasets Module

SciPy Datasets module enables you to access and work with datasets used in scientific computation and research. It offers straightforward means of loading and clearing cached datasets for use in a wide range of projects and investigations.
Sr.No.Function & Description1[scipy.download.all](/scipy/scipy_datasets_download_all_function.htm)
This method is used to download all available datasets in the SciPy dataset module.
2[scipy.clear.cache](/scipy/scipy_datasets_clear_cache_function.htm)
This method is used to clear the cached datasets that have been previously downloaded using the SciPy dataset module.

---


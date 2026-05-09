# Machine Learning With Python

## Table of Contents

1. [Machine Learning - Basic Concepts](#machine-learning---basic-concepts)
2. [Machine Learning - Ecosystem](#machine-learning---ecosystem)
3. [Machine Learning - Models](#machine-learning---models)
4. [Machine Learning - Data Loading](#machine-learning---data-loading)
5. [Machine Learning - Statistics](#machine-learning---statistics)
6. [Machine Learning - Data Visualization](#machine-learning---data-visualization)
7. [Data Preparation in Machine Learning](#data-preparation-in-machine-learning)
8. [Machine Learning - Feature Selection](#machine-learning---feature-selection)
9. [Machine Learning - Classification Algorithms](#machine-learning---classification-algorithms)
10. [Logistic Regression in Machine Learning](#logistic-regression-in-machine-learning)
11. [Support Vector Machine (SVM) in Machine Learning](#support-vector-machine-svm-in-machine-learning)
12. [Machine Learning - Decision Tree Algorithm](#machine-learning---decision-tree-algorithm)
13. [Nave Bayes Algorithm in Machine Learning](#nave-bayes-algorithm-in-machine-learning)
14. [Random Forest Algorithm in Machine Learning](#random-forest-algorithm-in-machine-learning)
15. [Regression Analysis in Machine Learning](#regression-analysis-in-machine-learning)
16. [Linear Regression in Machine Learning](#linear-regression-in-machine-learning)
17. [Clustering Algorithms in Machine Learning](#clustering-algorithms-in-machine-learning)
18. [Machine Learning - K-Means Clustering Algorithm](#machine-learning---k-means-clustering-algorithm)
19. [Machine Learning - Mean-Shift Clustering Algorithm](#machine-learning---mean-shift-clustering-algorithm)
20. [Machine Learning - Hierarchical Clustering](#machine-learning---hierarchical-clustering)
21. [K-Nearest Neighbors (KNN) in Machine Learning](#k-nearest-neighbors-knn-in-machine-learning)
22. [Performance Metrics in Machine Learning](#performance-metrics-in-machine-learning)
23. [Machine Learning - Automatic Workflows](#machine-learning---automatic-workflows)
24. [Machine Learning - Boost Model Performance](#machine-learning---boost-model-performance)
25. [Improving Performance of ML Model (Contd)](#improving-performance-of-ml-model-contd)

---

## 1. Machine Learning - Basic Concepts

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_basics.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_basics.htm)*

---

---
[Previous](/machine_learning/machine_learning_getting_started.htm)[Quiz](/machine_learning/quiz_on_machine_learning_basics.htm)[Next](/machine_learning/machine_learning_ecosystem.htm)
Machine learning, as we know, is a subset of artificial intelligence that involves training computer algorithms to automatically learn patterns and relationships in data. Here are some basic concepts of machine learning −

## Data

Data is the foundation of machine learning. Without data, there would be nothing for the algorithm to learn from. Data can come in many forms, including structured data (such as spreadsheets and databases) and unstructured data (such as text and images). The quality and quantity of the data used to train the machine learning algorithm are crucial factors that can significantly impact its performance.

## Feature

In machine learning, features are the variables or attributes used to describe the input data. The goal is to select the most relevant and informative features that will allow the algorithm to make accurate predictions or decisions. Feature selection is a crucial step in the machine learning process because the performance of the algorithm is heavily dependent on the quality and relevance of the features used.

## Model

A machine learning model is a mathematical representation of the relationship between the input data (features) and the output (predictions or decisions). The model is created using a training dataset and then evaluated using a separate validation dataset. The goal is to create a model that can accurately generalize to new, unseen data.

## Training

Training is the process of teaching the machine learning algorithm to make accurate predictions or decisions. This is done by providing the algorithm with a large dataset and allowing it to learn from the patterns and relationships in the data. During training, the algorithm adjusts its internal parameters to minimize the difference between its predicted output and the actual output.

## Testing

Testing is the process of evaluating the performance of the machine learning algorithm on a separate dataset that it has not seen before. The goal is to determine how well the algorithm generalizes to new, unseen data. If the algorithm performs well on the testing dataset, it is considered to be a successful model.

## Overfitting

Overfitting occurs when a machine learning model is too complex and fits the training data too closely. This can lead to poor performance on new, unseen data because the model is too specialized to the training dataset. To prevent overfitting, it is important to use a validation dataset to evaluate the model's performance and to use regularization techniques to simplify the model.

## Underfitting

Underfitting occurs when a machine learning model is too simple and cannot capture the patterns and relationships in the data. This can lead to poor performance on both the training and testing datasets. To prevent underfitting, we can use several techniques such as increasing model complexity, collect more data, reduce regularization, and feature engineering.

It is important to note that preventing underfitting is a balancing act between model complexity and the amount of data available. Increasing model complexity can help prevent underfitting, but if there is not enough data to support the increased complexity, overfitting may occur instead. Therefore, it is important to monitor the model's performance and adjust the complexity as necessary.

## Why & When to Make Machines Learn?

We have already discussed the need for machine learning, but another question arises that in what scenarios we must make the machine learn? There can be several circumstances where we need machines to take data-driven decisions with efficiency and at a huge scale. The followings are some of such circumstances where making machines learn would be more effective −

### Lack of human expertise

The very first scenario in which we want a machine to learn and take data-driven decisions, can be the domain where there is a lack of human expertise. The examples can be navigations in unknown territories or spatial planets.

### Dynamic scenarios

There are some scenarios which are dynamic in nature i.e. they keep changing over time. In case of these scenarios and behaviors, we want a machine to learn and take data-driven decisions. Some of the examples can be network connectivity and availability of infrastructure in an organization.

### Difficulty in translating expertise into computational tasks

There can be various domains in which humans have their expertise,; however, they are unable to translate this expertise into computational tasks. In such circumstances we want machine learning. The examples can be the domains of speech recognition, cognitive tasks etc.

## Machine Learning Model

Before discussing the machine learning model, we must need to understand the following formal definition of ML given by professor Mitchell −

A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

The above definition is basically focusing on three parameters, also the main components of any learning algorithm, namely Task(T), Performance(P) and experience (E). In this context, we can simplify this definition as −

ML is a field of AI consisting of learning algorithms that −

- 
Improve their performance (P)

- 
At executing some task (T)

- 
Over time with experience (E)

Based on the above, the following diagram represents a Machine Learning Model −
![Machine Learning Model](/machine_learning/images/machine_learning_model.jpg)
Let us discuss them more in detail now −

### Task(T)

From the perspective of problem, we may define the task T as the real-world problem to be solved. The problem can be anything like finding best house price in a specific location or to find best marketing strategy etc. On the other hand, if we talk about machine learning, the definition of task is different because it is difficult to solve ML based tasks by conventional programming approach.

A task T is said to be a ML based task when it is based on the process and the system must follow for operating on data points. The examples of ML based tasks are Classification, Regression, Structured annotation, Clustering, Transcription etc.

### Experience (E)

As name suggests, it is the knowledge gained from data points provided to the algorithm or model. Once provided with the dataset, the model will run iteratively and will learn some inherent pattern. The learning thus acquired is called experience(E). Making an analogy with human learning, we can think of this situation as in which a human being is learning or gaining some experience from various attributes like situation, relationships etc. Supervised, unsupervised and reinforcement learning are some ways to learn or gain experience. The experience gained by out ML model or algorithm will be used to solve the task T.

### Performance (P)

An ML algorithm is supposed to perform task and gain experience with the passage of time. The measure which tells whether ML algorithm is performing as per expectation or not is its performance (P). P is basically a quantitative metric that tells how a model is performing the task, T, using its experience, E. There are many metrics that help to understand the ML performance, such as accuracy score, F1 score, confusion matrix, precision, recall, sensitivity etc.

---

## 2. Machine Learning - Ecosystem

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_ecosystem.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_ecosystem.htm)*

---

---
[Previous](/machine_learning/machine_learning_basics.htm)[Quiz](/machine_learning/quiz_on_machine_learning_ecosystem.htm)[Next](/machine_learning/machine_learning_python_libraries.htm)
Python has become one of the most popular programming languages for machine learning due to its simplicity, versatility, and extensive ecosystem of libraries and tools. There are various programming languages such as Java, C++, Lisp, Julia, Python, etc., that can be used in machine learning. Among them, Python programming language has gained a huge popularity.

Here, we will explore the Python ecosystem for machine learning and highlight some of the most popular libraries and frameworks.

## Python Machine Learning Ecosystem

The machine learning ecosystem refers to the collection of tools and technologies that are used to develop the machine learning applications. Python provides various libraries and tools that form the components of Python machine learning ecosystem. These useful components make Python an important language for Machine Learning & Data Science. Though there are many such components, let us discuss some of the importance components of Python ecosystem here −

- Programming Language: Python
- Integrated Development Environment
- Python Libraries
## Programming Language: Python

The programming languages such are the important components of any development ecosystem. Python programming language is extensively used in machine learning and data science.

Let's discuss why Python is the best choice for machine learning.

### Why Python for Machine Learning?

According to Stack OverFlow Developer Survey 2023, Python is third most popular programming language as well as the most popular language for machine learning and data science. The following are the features of Python that makes it the preferred choice of language for data science −

#### Extensive set of packages

Python has an extensive and powerful set of packages which are ready to be used in various domains. It also has packages like
**numpy, scipy, pandas, scikit-learn**etc. which are required for machine learning and data science.
#### Easy prototyping

Another important feature of Python that makes it the choice of language for data science is the easy and fast prototyping. This feature is useful for developing new algorithm.

#### Collaboration feature

The field of data science basically needs good collaboration and Python provides many useful tools that make this extremely.

#### One language for many domains

A typical data science project includes various domains like data extraction, data manipulation, data analysis, feature extraction, modelling, evaluation, deployment and updating the solution. As Python is a multi-purpose language, it allows the data scientist to address all these domains from a common platform.

### Strengths and Weaknesses of Python

Every programming language has some strengths as well as weaknesses, so does Python too.

#### Strengths

According to studies and surveys, Python is the fifth most important language as well as the most popular language for machine learning and data science. It is because of the following strengths that Python has −
**Easy to learn and understand**− The syntax of Python is simpler; hence it is relatively easy, even for beginners also, to learn and understand the language.**Multi-purpose language**− Python is a multi-purpose programming language because it supports structured programming, object-oriented programming as well as functional programming.**Huge number of modules**− Python has huge number of modules for covering every aspect of programming. These modules are easily available for use hence making Python an extensible language.**Support of open source community**− As being open source programming language, Python is supported by a very large developer community. Due to this, the bugs are easily fixed by the Python community. This characteristic makes Python very robust and adaptive.**Scalability**− Python is a scalable programming language because it provides an improved structure for supporting large programs than shell-scripts.
#### Weakness

Although Python is a popular and powerful programming language, it has its own weakness of slow execution speed.

The execution speed of Python is slow as compared to compiled languages because Python is an interpreted language. This can be the major area of improvement for Python community.

### Installing Python

For working in Python, we must first have to install it. You can perform the installation of Python in any of the following two ways −

- 
Installing Python individually

- 
Using Pre-packaged Python distribution − Anaconda

Let us discuss these each in detail.

#### Installing Python Individually

If you want to install Python on your computer, then then you need to download only the binary code applicable for your platform. Python distribution is available for Windows, Linux and Mac platforms.

The following is a quick overview of installing Python on the above-mentioned platforms −
**On Unix and Linux platform**
With the help of following steps, we can install Python on Unix and Linux platform −

- 
First, go to
[www.python.org/downloads/](https://www.python.org/downloads/).
- 
Next, click on the link to download zipped source code available for Unix/Linux.

- 
Now, Download and extract files.

- 
Next, we can edit the Modules/Setup file if we want to customize some options.

- 
Next, write the command
**run ./configure script**
- 
make

- 
make install
**On Windows platform**
With the help of following steps, we can install Python on Windows platform −

- 
First, go to
[www.python.org/downloads/](https://www.python.org/downloads/).
- 
Next, click on the link for Windows installer python-XYZ.msi file. Here XYZ is the version we wish to install.

- 
Now, we must run the file that is downloaded. It will take us to the Python install wizard, which is easy to use. Now, accept the default settings and wait until the install is finished.
**On Macintosh platform**
For Mac OS X, Homebrew, a great and easy to use package installer is recommended to install Python 3. In case if you don't have Homebrew, you can install it with the help of following command −

```
$ ruby -e "$(curl -fsSL
https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

It can be updated with the command below −

```
$ brew update
```

Now, to install Python3 on your system, we need to run the following command −

```
$ brew install python3
```

#### Using Pre-packaged Python Distribution: Anaconda

Anaconda is a packaged compilation of Python which have all the libraries widely used in Data science. We can follow the following steps to setup Python environment using Anaconda −

- **Step 1**− First, we need to download the required installation package from Anaconda distribution. The link for the same is[www.anaconda.com/distribution/](https://www.anaconda.com/products/individual). You can choose from Windows, Mac and Linux OS as per your requirement.
- **Step 2**− Next, select the Python version you want to install on your machine. The latest Python version is 3.7. There you will get the options for 64-bit and 32-bit Graphical installer both.
- **Step 3**− After selecting the OS and Python version, it will download the Anaconda installer on your computer. Now, double click the file and the installer will install Anaconda package.
- **Step 4**− For checking whether it is installed or not, open a command prompt and type Python.
## Integrated Development Environment

An Integrated Development Environment (IDE) is a software tool that combines standard developer tools into a single user-friendly interface (Graphical User interface). There are many popular IDEs that are used in machine learning and data science related development. Some of them are as follow −

- Jupyter Notebook
- PyCharm
- Visual Studio Code
- Spyder
- Sublime Text
- Atom
- Thonny
- Google Colab Notebook
Here, we will discuss in detail about the Jupyter notebook. You can visit to the respective official websites for the particular IDEs for more details such how to download, install and use them.

### Jupyter Notebook

Jupyter notebooks basically provides an interactive computational environment for developing Python based Data Science applications. They are formerly known as ipython notebooks. The following are some of the features of Jupyter notebooks that makes it one of the best components of Python ML ecosystem −

- 
Jupyter notebooks can illustrate the analysis process step by step by arranging the stuff like code, images, text, output etc. in a step by step manner.

- 
It helps a data scientist to document the thought process while developing the analysis process.

- 
One can also capture the result as the part of the notebook.

- 
With the help of jupyter notebooks, we can share our work with a peer also.

#### Installation and Execution

If you are using Anaconda distribution, then you need not install jupyter notebook separately as it is already installed with it. You just need to go to Anaconda Prompt and type the following command −

```
C:\>jupyter notebook
```

After pressing enter, it will start a notebook server at localhost:8888 of your computer. It is shown in the following screen shot −
![Jupyter Notebook](/machine_learning/images/jupyter_notebook.jpg)
Now, after clicking the New tab, you will get a list of options. Select Python 3 and it will take you to the new notebook for start working in it. You will get a glimpse of it in the following screenshots −
![Python Table](/machine_learning/images/python_table.jpg)![Search Bar](/machine_learning/images/search_bar.jpg)
On the other hand, if you are using standard Python distribution then jupyter notebook can be installed using popular python package installer, pip.

```
pip3 install jupyter
```

#### Types of Cells in Jupyter Notebook

The following are the three types of cells in a jupyter notebook −
**Code cells**− As the name suggests, we can use these cells to write code. After writing the code/content, it will send it to the kernel that is associated with the notebook.**Markdown cells**− We can use these cells for notating the computation process. They can contain the stuff like text, images, Latex equations, HTML tags etc.**Raw cells**− The text written in them is displayed as it is. These cells are basically used to add the text that we do not wish to be converted by the automatic conversion mechanism of jupyter notebook.
For more detailed study of jupyter notebook, you can go to the link
[www.tutorialspoint.com/jupyter/index.htm](/jupyter/index.htm).
## Python Libraries and Packages

Python ecosystem has a huge collection of libraries and packages that help developers to build easily and quickly machine learning models. We have discussed here some of them as follows −

### NumPy

NumPy is a fundamental library for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on them.

NumPy is a critical component of the Python machine learning ecosystem, as it provides the underlying data structure and numerical operations required for many machine learning algorithms. Below is the command to install NumPy −

```
pip3 install numpy
```

### Pandas

Pandas is a powerful library for data manipulation and analysis. It provides a range of functions for importing, cleaning, and transforming data, along with powerful tools for grouping and aggregating data.

Pandas is particularly useful for data preprocessing in machine learning, as it allows for efficient data handling and manipulation. Below is the command to install Pandas −

```
pip3 install pandas
```

### Scikit-learn

Scikit-learn is a popular machine learning library in Python, providing a range of algorithms for classification, regression, clustering, and more. It also includes tools for data preprocessing, feature selection, and model evaluation. Scikit-learn is widely used in the machine learning community due to its ease of use, performance, and extensive documentation.

Below is the command to install Scikit-learn −

```
pip3 install scikit-learn
```

### TensorFlow

TensorFlow is an open-source library for machine learning developed by Google. It provides support for building and training deep learning models, along with tools for distributed computing and deployment. TensorFlow is a powerful tool for building complex machine learning models, particularly in the areas of computer vision and natural language processing. Below is the command to install TensorFlow −

```
pip install tensorflow
```

### PyTorch

PyTorch is another popular deep learning library in Python. Developed by Facebook, it provides a range of tools for building and training neural networks, along with support for dynamic computation graphs and GPU acceleration.

PyTorch is particularly useful for researchers and developers who need a flexible and powerful deep learning framework. Below is the command to install PyTorch −

```
pip install torch
```

### Keras

Keras is a high-level neural network library that runs on top of TensorFlow and other lower-level frameworks. It provides a simple and intuitive API for building and training deep learning models, making it an excellent choice for beginners and researchers who need to quickly prototype and experiment with different models. Below is the command to install Keras −

```
pip3 install keras
```

### OpenCV

OpenCV is a computer vision library that provides tools for image and video processing, along with support for machine learning algorithms. It is widely used in the computer vision community for tasks such as object detection, image segmentation, and facial recognition. Below is the command to install OpenCV −

```
pip3 install opencv-python
```

In addition to these libraries, there are many other tools and frameworks in the Python ecosystem for machine learning, including
**XGBoost, LightGBM, spaCy,**and**NLTK**.
The Python ecosystem for machine learning is constantly evolving, with new libraries and tools being developed all the time.

Whether you are a beginner or an experienced machine learning practitioner, Python provides a rich and flexible environment for developing and deploying machine learning models.

Here, it is also important to note that some libraries may require additional dependencies or system-specific requirements. In such cases, it is recommended to consult the library's documentation for installation instructions and requirements.

---

## 3. Machine Learning - Models

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_models.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_models.htm)*

---

---
[Previous](/machine_learning/machine_learning_preparing_data.htm)[Quiz](/machine_learning/quiz_on_machine_learning_models.htm)[Next](/machine_learning/machine_learning_supervised.htm)
There are various Machine Learning algorithms, techniques and methods that can be used to build
**models**for solving real-life problems by using data. In this chapter, we are going to discuss such different kinds of methods.
There are four main types of machine learning methods classified based on human supervision −

- [Supervised Learning](#supervised_learning)
- [Unsupervised Learning](#unsupervised_learning)
- [Semi-supervised Learning](#semi-supervised_learning)
- [Reinforcement Learning](#reinforcement_learning)
In the next four chapters, we will discuss each of these machine learning models in detail. Here, let's have a brief overview of these methods:

## Supervised Learning
[Supervised learning](/machine_learning/machine_learning_supervised.htm)algorithms or methods are the most commonly used ML algorithms. This method or learning algorithm takes the data sample i.e. the training data and its associated output i.e. labels or responses with each data sample during the training process.
The main objective of supervised learning algorithms is to learn an association between input data samples and corresponding outputs after performing multiple training data instances.

For example, we have
**x**: Input variables and**Y**: Output variable
Now, apply an algorithm to learn the mapping function from the input to output as follows −

Y=f(x)

Now, the main objective would be to approximate the mapping function so well that even when we have new input data (x), we can easily predict the output variable (Y) for that new input data.

It is called supervised because the whole process of learning can be thought as it is being supervised by a teacher or supervisor. Examples of supervised machine learning algorithms includes
**Decision tree, Random Forest, KNN, Logistic Regression**etc.
Based on the ML tasks, supervised learning algorithms can be divided into the following two broad classes −

- Classification
- Regression
### Classification

The key objective of classification-based tasks is to predict categorial output labels or responses for the given input data. The output will be based on what the model has learned in the training phase. As we know the categorial output responses means unordered and discrete values, hence each output response will belong to a specific class or category. We will discuss Classification and associated algorithms in detail in the upcoming chapters also.

#### Classification Models

Followings are some common classification models −

- [Logistic Regression](/machine_learning/machine_learning_logistic_regression.htm)
- [Decision Trees](/machine_learning/machine_learning_decision_tree_algorithm.htm)
- [Random Forest](/machine_learning/machine_learning_random_forest_classification.htm)
- [K-nearest Neighbor](/machine_learning/machine_learning_knn_nearest_neighbors.htm)
- [Support Vector Machine](/machine_learning/machine_learning_support_vector_machine.htm)
- [Naive Bayes](/machine_learning/machine_learning_naive_bayes_algorithms.htm)
- Linear Discriminant Analysis
- [Neural Networks](/machine_learning/machine_learning_artificial_neural_networks.htm)
### Regression

The key objective of regression-based tasks is to predict output labels or responses, which are continuous numeric values, for the given input data. The output will be based on what the model has learned in its training phase. Basically, regression models use the input data features (independent variables) and their corresponding continuous numeric output values (dependent or outcome variables) to learn specific associations between inputs and corresponding outputs. We will discuss regression and associated algorithms in detail in further chapters.

#### Regression Models

Followings are some common regression models −

- [Linear Regression](/machine_learning/machine_learning_linear_regression.htm)
- Ridge regression
- [Decision Trees](/machine_learning/machine_learning_decision_tree_algorithm.htm)
- [Random Forest](/machine_learning/machine_learning_random_forest_classification.htm)
- [K-nearest Neighbor](/machine_learning/machine_learning_knn_nearest_neighbors.htm)
- [Neural Network Regression](/machine_learning/machine_learning_artificial_neural_networks.htm)
## Unsupervised Learning

As the name suggests,
[unsupervised learning](/machine_learning/machine_learning_unsupervised.htm)is opposite to supervised ML methods or algorithms in which we do not have any supervisor to provide any sort of guidance. Unsupervised learning algorithms are handy in the scenario in which we do not have the liberty, like in supervised learning algorithms, of having pre-labeled training data and we want to extract useful pattern from input data.
For example, it can be understood as follows −

Suppose we have −
**x: Input variables**, then there would be no corresponding output variable and the algorithms need to discover the interesting pattern in data for learning.
Examples of unsupervised machine learning algorithms includes K-means clustering,
**K-nearest neighbors**etc.
Based on the ML tasks, unsupervised learning algorithms can be divided into the following broad classes −

- Clustering
- Association
- Dimensionality Reduction
### Clustering

Clustering methods are one of the most useful unsupervised ML methods. These algorithms used to find similarity as well as relationship patterns among data samples and then cluster those samples into groups having similarity based on features. The real-world example of clustering is to group the customers by their purchasing behavior.

#### Clustering Models

Followings are some common clustering models −

- [K-Means Clustering](/machine_learning/machine_learning_k_means_clustering.htm)
- [Hierarchical Clustering](/machine_learning/machine_learning_hierarchical_clustering.htm)
- [Mean-shift Clustering](/machine_learning/machine_learning_mean_shift_clustering.htm)
- [DBSCAN Clustering](/machine_learning/machine_learning_dbscan_clustering.htm)
- [HDBSCAN Clustering](/machine_learning/machine_learning_hdbscan_clustering.htm)
- [BIRCH Clustering](/machine_learning/machine_learning_birch_clustering.htm)
- [Affinity Propagation](/machine_learning/machine_learning_affinity_propagation.htm)
- [Agglomerative Clustering](/machine_learning/machine_learning_agglomerative_clustering.htm)
### Association

Another useful unsupervised ML method is
**Association**which is used to analyze large dataset to find patterns which further represents the interesting relationships between various items. It is also termed as**Association Rule Mining**or**Market basket analysis**which is mainly used to analyze customer shopping patterns.
#### Association Models

Followings are some common association models −

- [Apriori Algorithm](/machine_learning/machine_learning_apriori_algorithm.htm)
- Eclat algorithm
- FP-growth algorithm
### Dimensionality Reduction

This unsupervised ML method is used to reduce the number of feature variables for each data sample by selecting set of principal or representative features. A question arises here is that why we need to reduce the dimensionality? The reason behind is the problem of feature space complexity which arises when we start analyzing and extracting millions of features from data samples. This problem generally refers to curse of dimensionality. PCA (Principal Component Analysis), K-nearest neighbors and discriminant analysis are some of the popular algorithms for this purpose.

#### Dimensionality Reduction Models

Followings are some common dimensionality Reduction models −

- [Principal Component Analysis(PCA)](/machine_learning/machine_learning_principal_component_analysis.htm)
- Autoencoders
- Singular value decomposition (SVD)
### Anomaly Detection

This unsupervised ML method is used to find out the occurrences of rare events or observations that generally do not occur. By using the learned knowledge, anomaly detection methods would be able to differentiate between anomalous or a normal data point. Some of the unsupervised algorithms like clustering, KNN can detect anomalies based on the data and its features.

## Semi-supervised Learning
[Semi-supervised learning](/machine_learning/machine_learning_semi_supervised_learning.htm)algorithms or methods are neither fully supervised nor fully unsupervised. They basically fall between the two i.e. supervised and unsupervised learning methods. These kinds of algorithms generally use small supervised learning component i.e. small amount of pre-labeled annotated data and large unsupervised learning component i.e. lots of unlabeled data for training. We can follow any of the following approaches for implementing semi-supervised learning methods −
- The first and simple approach is to build the supervised model based on small amount of labeled and annotated data and then build the unsupervised model by applying the same to the large amounts of unlabeled data to get more labeled samples. Now, train the model on them and repeat the process.
- The second approach needs some extra efforts. In this approach, we can first use the unsupervised methods to cluster similar data samples, annotate these groups and then use a combination of this information to train the model.
## Reinforcement Learning
[Reinforcement learning](/machine_learning/machine_learning_reinforcement_learning.htm)methods are different from previously studied methods and very rarely used also. In this kind of learning algorithms, there would be an agent that we want to train over a period of time so that it can interact with a specific environment. The agent will follow a set of strategies for interacting with the environment and then after observing the environment it will take actions regards the current state of the environment. The following are the main steps of reinforcement learning methods −
- **Step 1**− First, we need to prepare an agent with some initial set of strategies.
- **Step 2**− Then observe the environment and its current state.
- **Step 3**− Next, select the optimal policy regards the current state of the environment and perform important action.
- **Step 4**− Now, the agent can get corresponding reward or penalty as per accordance with the action taken by it in previous step.
- **Step 5**− Now, we can update the strategies if it is required so.
- **Step 6**− At last, repeat steps 2-5 until the agent got to learn and adopt the optimal policies.
#### Reinforcement Learning Models

Following are some common reinforcement learning algorithms −

- Q-learning
- Markov Decision Process (MDP)
- SARSA
- DQN
- DDPG
We will discuss each of the above machine learning models in detail in upcoming chapters.

---

## 4. Machine Learning - Data Loading

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_data_loading.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_data_loading.htm)*

---

---
[Previous](/machine_learning/machine_learning_categorical_data.htm)[Quiz](/machine_learning/quiz_on_machine_learning_data_loading.htm)[Next](/machine_learning/machine_learning_data_understanding.htm)
Suppose if you want to start a ML project then what is the first and most important thing you would require? It is the data that we need to load for starting any of the ML project.

In machine learning, data loading refers to the process of importing or reading data from external sources and converting it into a format that can be used by the machine learning algorithm. The data is then preprocessed to remove any inconsistencies, missing values, or outliers. Once the data is preprocessed, it is split into training and testing sets, which are then used for model training and evaluation.

The data can come from various sources such as CSV files, databases, web APIs, cloud storage, etc. The most common file formats for machine learning projects is CSV (Comma Separated Values).

## Consideration While Loading CSV data

CSV is a plain text format that stores tabular data, where each row represents a record, and each column represents a field or attribute. It is widely used because it is simple, lightweight, and can be easily read and processed by programming languages such as Python, R, and Java.

In Python, we can load CSV data into ML projects with different ways but before loading CSV data we must have to take care about some considerations.

In this chapter, let's understand the main parts of a CSV file, how they might affect the loading and analysis of data, and some consideration we should take care before loading CSV data into ML projects.

## File Header

This is the first row of the CSV file, and it typically contains the names of the columns in the table. When loading CSV data into an ML project, the file header (also known as column headers or variable names) can play an important role in data analysis and model training. Here are some considerations to keep in mind regarding the file header −

- **Consistency**− The header row should be consistent across the entire CSV file. This means that the number of columns and their names should be the same for each row. Inconsistencies can cause issues with parsing and analysis.
- **Meaningful names**− Column names should be meaningful and descriptive. This can help with understanding the data and building more accurate models. Avoid using generic names like "column1", "column2", etc.
- **Case sensitivity**− Depending on the tool or library being used to load the CSV file, the column names may be case sensitive. It's important to ensure that the case of the header row matches the expected case sensitivity of the tool or library being used.
- **Special characters**− Column names should not contain any special characters, such as spaces, commas, or quotation marks. These characters can cause issues with parsing and analysis. Instead, use underscores or camelCase to separate words.
- **Missing header**− If the CSV file does not have a header row, it's important to specify the column names manually or provide a separate file or documentation that includes the column names.
- **Encoding**− The encoding of the header row can affect its interpretation when loading the CSV file. It's important to ensure that the encoding of the header row is compatible with the tool or library being used to read the file.
## Comments

These are optional lines that begin with a specified character, such as "#" or "//", and are ignored by most programs that read CSV files. They can be used to provide additional information or context about the data in the file.

Comments in a CSV file are not typically used to represent data that would be used in a machine learning project. However, if comments are present in a CSV file, it's important to consider how they might affect the loading and analysis of the data. Here are some considerations −

- **Comment markers**− In a CSV file, comments can be indicated using a specific marker, such as "#" or "//". It's important to know what marker is being used, so that the loading process can ignore comments properly.
- **Placement**− Comments should be placed in a separate line from the actual data. If a comment is included in a line with actual data, it may cause issues with parsing and analysis.
- **Consistency**− If comments are used in a CSV file, it's important to ensure that the comment marker is used consistently throughout the entire file. Inconsistencies can cause issues with parsing and analysis.
- **Handling comments**− Depending on the tool or library being used to load the CSV file, comments may be ignored by default or may require a specific parameter to be set. It's important to understand how comments are handled by the tool or library being used.
- **Effect on analysis**− If comments contain important information about the data, it may be necessary to process them separately from the data itself. This can add complexity to the loading and analysis process.
## Delimiter

This is the character that separates the fields in each row. While the name suggests that a comma is used as the delimiter, other characters such as tabs, semicolons, or pipes can also be used depending on the file.

The delimiter used in a CSV file can significantly affect the accuracy and performance of a machine learning model, so it is important to consider the following while loading data into an ML project −

- **Delimiter choice**− The delimiter used in a CSV file should be carefully chosen based on the data being used. For example, if the data contains commas within the values (e.g. "New York, NY"), then using a comma as a delimiter may cause issues.
In this case, a different delimiter, such as a tab or semicolon, may be more appropriate.

- **Consistency**− The delimiter used in the CSV file should be consistent throughout the entire file. Mixing different delimiters or using whitespace inconsistently can lead to errors and make it difficult to parse the data accurately.
- **Encoding**− The delimiter can also be affected by the encoding of the CSV file. For example, if the CSV file uses a non-ASCII delimiter and is encoded in UTF-8, it may not be correctly read by some machine learning libraries or tools. It is important to ensure that the encoding and delimiter are compatible with the machine learning tools being used.
- **Other considerations**− In some cases, the delimiter may need to be customized based on the machine learning tool being used. For example, some libraries may require a specific delimiter or may not support certain delimiters. It is important to check the documentation of the machine learning tool being used and customize the delimiter as needed.
## Quotes

These are optional characters that can be used to enclose fields that contain the delimiter character or newlines. For example, if a field contains a comma, enclosing the field in quotes ensures that the comma is treated as part of the field and not as a delimiter. When loading CSV data into an ML project, there are several considerations to keep in mind regarding the use of quotes −

- **Quote character**− The quote character used in a CSV file should be consistent throughout the file. The most commonly used quote character is the double quote (") but some files may use single quotes or other characters. It's important to make sure that the quote character used is consistent with the tool or library being used to read the CSV file.
- **Quoted values**− In some cases, values in a CSV file may be enclosed in quotes to differentiate them from other values. For example, if a field contains a comma, it may be enclosed in quotes to prevent it from being interpreted as a new field. It's important to make sure that quoted values are properly handled when loading the data into an ML project.
- **Escaping quotes**− If a field contains the quote character used to enclose values, it must be escaped. This is typically done by doubling the quote character. For example, if the quote character is double quote (") and a field contains the value "John "the Hammer" Smith", it would be enclosed in quotes and the internal quotes would be escaped like this: "John ""the Hammer"" Smith".
- **Use of quotes**− The use of quotes in CSV files can vary depending on the tool or library being used to generate the file. Some tools may use quotes around every field, while others may only use quotes around fields that contain special characters. It's important to make sure that the quote usage is consistent with the tool or library being used to read the file.
- **Encoding**− The use of quotes can also be affected by the encoding of the CSV file. If the file is encoded in a non-standard way, it may cause issues when loading the data into an ML project. It's important to make sure that the encoding of the CSV file is compatible with the tool or library being used to read the file.
## Various Methods of Loading a CSV Data File

While working with ML projects, the most crucial task is to load the data properly into it. As told earlier, the most common data format for ML projects is CSV and it comes in various flavors and varying difficulties to parse.

In this section, we are going to discuss some common approaches in Python to load CSV data file into machine learning project −

### Using the CSV Module

This is a built-in module in Python that provides functionality for reading and writing CSV files. You can use it to read a CSV file into a list or dictionary object. Below is its implementation example in Python −

```
import csv
with open('mydata.csv', 'r') as file:
   reader = csv.reader(file)
   for row in reader:
      print(row)
```

This code reads a CSV file called
**mydata.csv**and prints each row in the file.
### Using the Pandas Library

This is a popular data manipulation library in Python that provides a read_csv() function for reading CSV files into a pandas DataFrame object. This is a very convenient way to load data and perform various data manipulation tasks. Below is its implementation example in Python −

```
import pandas as pd

data = pd.read_csv('mydata.csv')
```

This code reads a CSV file called
**mydata.csv**and loads it into a pandas DataFrame object called data.
### Using the Numpy Library

This is a numerical computing library in Python that provides a
**genfromtxt()**function for loading CSV files into a**numpy**array. Below is its implementation example in Python −
```
import numpy as np

data = np.genfromtxt('mydata.csv', delimiter=',')
```

This code reads a CSV file called
**mydata.csv**and loads it into a numpy array called 'data'.
### Using the Scipy Library

This is a scientific computing library in Python that provides a
**loadtxt()**function for loading text files, including CSV files, into a numpy array. Below is its implementation example in Python −
```
import numpy as np

from scipy import loadtxt
data = loadtxt('mydata.csv', delimiter=',')
```

This code reads a CSV file called
**mydata.csv**and loads it into a numpy array called 'data'.
### Using the Sklearn Library

This is a popular machine learning library in Python that provides a load_iris() function for loading the iris dataset, which is a commonly used dataset for classification tasks. Below is its implementation example in Python −

```
from sklearn.datasets import load_iris

data = load_iris().data
```

This code loads the iris dataset, which is included in the
**sklearn**library, and loads it into a numpy array called data.

---

## 5. Machine Learning - Statistics

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_statistics.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_statistics.htm)*

---

---

## 6. Machine Learning - Data Visualization

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_data_visualization.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_data_visualization.htm)*

---

---
[Previous](/machine_learning/machine_learning_supervised_vs_unsupervised.htm)[Quiz](/machine_learning/quiz_on_machine_learning_data_visualization.htm)[Next](/machine_learning/machine_learning_histograms.htm)
Data visualization is an important aspect of machine learning (ML) as it helps to analyze and communicate patterns, trends, and insights in the data. Data visualization involves creating graphical representations of the data, which can help to identify patterns and relationships that may not be apparent from the raw data.

## What is Data Visualization?

Data visualization is a graphical representation of data and information. With the help of data visualization, we can see how the data looks like and what kind of correlation is held by the attributes of the data. It is the fastest way to see if the features correspond to the output.

## Importance of Data Visualization in Machine Learning

The data visualization play a significant role in machine learning. We can use it in many ways in machine learning. Here are some of the ways data visualization is used in machine learning −

- **Exploring Data**− Data visualization is an essential tool for exploring and understanding data. Visualization can help to identify patterns, correlations, and outliers and can also help to detect data quality issues such as missing values and inconsistencies.
- **Feature Selection**− Data visualization can help to select relevant features for the ML model. By visualizing the data and its relationship with the target variable, you can identify features that are strongly correlated with the target variable and exclude irrelevant features that have little predictive power.
- **Model Evaluation**− Data visualization can be used to evaluate the performance of the ML model. Visualization techniques such as ROC curves, precision-recall curves, and[confusion matrices](/machine_learning/machine_learning_confusion_matrix.htm)can help to understand the accuracy,[precision, recall](/machine_learning/machine_learning_precision_and_recall.htm), and F1 score of the model.
- **Communicating Insights**− Data visualization is an effective way to communicate insights and results to stakeholders who may not have a technical background. Visualizations such as scatter plots, line charts, and bar charts can help to convey complex information in an easily understandable format.
## Popular Python Libraries for Data Visualization

Following are the most popular Python libraries for data visualization in Machine learning. These libraries provide a wide range of visualization techniques and customization options to suit different needs and preferences.

### 1. Matplotlib
[Matplotlib](/matplotlib/index.htm)is one of the most popular Python packages used for data visualization. It is a cross-platform library for making 2D plots from data in arrays. It provides an object-oriented API that helps in embedding plots in applications using Python GUI toolkits such as PyQt, WxPython, or Tkinter. It can be used in Python and IPython shells, Jupyter notebook and web application servers also.
### 2. Seaborn
[Seaborn](/seaborn/index.htm)is an open source, BSD-licensed Python library providing high level API for visualizing the data using Python programming language.
### 3. Plotly
[Plotly](/plotly/plotly_introduction.htm)is a Montreal based technical computing company involved in development of data analytics and visualisation tools such as Dash and Chart Studio. It has also developed open source graphing Application Programming Interface (API) libraries for Python, R, MATLAB, Javascript and other computer programming languages.
### 4. Bokeh
[Bokeh](/bokeh/bokeh_introduction.htm)is a data visualization library for Python. Unlike Matplotlib and Seaborn, they are also Python packages for data visualization, Bokeh renders its plots using HTML and JavaScript. Hence, it proves to be extremely useful for developing web based dashboards.
## Types of Data Visualization

Data visualization for machine learning data can be classified into two different categories as follows -

- Univariate Plots
- Multivariate Plots![Data Visualization Techniques](/machine_learning/images/data_visualization_techniques.jpg)
Let's understand each of the above two type of data visualization plots in detail.

## Univariate Plots: Understanding Attributes Independently

The simplest type of visualization is single-variable or univariate visualization. With the help of univariate visualization, we can understand each attribute of our dataset independently. The following are some techniques in Python to implement univariate visualization −

- [Histograms](/machine_learning/machine_learning_histograms.htm)
- [Density Plots](/machine_learning/machine_learning_density_plots.htm)
- [Box and Whisker Plots](/machine_learning/machine_learning_box_and_whisker_plots.htm)
We will learn the above techniques in detail in their respective chapters. Let's look at these techniques in brief.

### Example - Histograms
[Histograms](/machine_learning/machine_learning_histograms.htm)group the data in bins and is the fastest way to get an idea about the distribution of each attribute in the dataset. The following are some of the characteristics of histograms −
- It provides us a count of the number of observations in each bin created for visualization.
- From the shape of the bin, we can easily observe the distribution, i.e., whether it is Gaussian, skewed, or exponential.
- Histograms also help us to see possible outliers.
The code below is an example of a Python script creating the histogram. Here, we will be using hist() function on NumPy Array to generate histograms and
**matplotlib**for plotting them.
```
import matplotlib.pyplot as plt
import numpy as np
# Generate some random data
data = np.random.randn(1000)
# Create the histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()
```

#### Output
![ML Histograms Plot](/machine_learning/images/ml_data_visualization_histograms.jpg)
Because of random number generation, you may notice a slight difference between the outputs when you execute the above program.

### Example - Density Plots
[Density Plot](/machine_learning/machine_learning_density_plots.htm)is another quick and easy technique for getting each attribute distribution. It is also like histogram but having a smooth curve drawn through the top of each bin. We can call them as abstracted histograms.
In the following example, the Python script will generate Density Plots for the distribution of attributes of the iris dataset.

```
import seaborn as sns
import matplotlib.pyplot as plt
# Load a sample dataset
df = sns.load_dataset("iris")
# Create the density plot
sns.kdeplot(data=df, x="sepal_length", fill=True)
# Add labels and title
plt.xlabel("Sepal Length")
plt.ylabel("Density")
plt.title("Density Plot of Sepal Length")
# Show the plot
plt.show()
```

#### Output
![Density Plot](/machine_learning/images/ml_data_visualization_density_plot.jpg)
From the above output, the difference between Density plots and Histograms can be easily understood.

### Example - Box and Whisker Plots
[Box and Whisker Plots](/machine_learning/machine_learning_box_and_whisker_plots.htm), also called boxplots in short, is another useful technique to review the distribution of each attributes distribution. The following are the characteristics of this technique −
- It is univariate in nature and summarizes the distribution of each attribute.
- It draws a line for the middle value i.e. for median.
- It draws a box around the 25% and 75%.
- It also draws whiskers which will give us an idea about the spread of the data.
- The dots outside the whiskers signifies the outlier values. Outlier values would be 1.5 times greater than the size of the spread of the middle data.
In the following example, the Python script will generate a Box and Whisker Plot for the distribution of attributes of the Iris dataset.

```
import matplotlib.pyplot as plt
# Sample data
data = [10, 15, 18, 20, 22, 25, 28, 30, 32, 35]
# Create a figure and axes
fig, ax = plt.subplots()
# Create the boxplot
ax.boxplot(data)
# Set the title
ax.set_title('Box and Whisker Plot')
# Show the plot
plt.show()
```

#### Output
![Box Plot](/machine_learning/images/ml_data_visualization_box_plot.jpg)
## Multivariate Plots: Interaction Among Multiple Variables

Another type of visualization is multi-variable or multivariate visualization. With the help of multivariate visualization, we can understand the interaction between multiple attributes of our dataset. The following are some techniques in Python to implement multivariate visualization −

- [Correlation Matrix Plot](/machine_learning/machine_learning_correlation_matrix_plot.htm)
- [Scatter Matrix Plot](/machine_learning/machine_learning_scatter_matrix_plot.htm)
### Example - Correlation Matrix Plot

Correlation is an indication of the changes between two variables. We can plot
[correlation matrix plot](/machine_learning/machine_learning_correlation_matrix_plot.htm)to show which variable is having a high or low correlation in respect to another variable.
In the following example, the Python script will generate a correlation matrix plot. It can be generated with the help of corr() function on Pandas DataFrame and plotted with the help of Matplotlib pyplot.

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Sample data
data = {'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 1, 4, 5]}
df = pd.DataFrame(data)
# Calculate the correlation matrix
c_matrix = df.corr()
# Create a heatmap
sns.heatmap(c_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

#### Output
![Correlation Matrix Plot](/machine_learning/images/ml_data_visualization_correlation_matrix_plot.jpg)
From the above output of the correlation matrix, we can see that it is symmetrical i.e. the bottom left is same as the top right.

### Example - Scatter Matrix Plot
[Scatter matrix plot](/machine_learning/machine_learning_scatter_matrix_plot.htm)shows how much one variable is affected by another or the relationship between them with the help of dots in two dimensions. Scatter plots are very much like line graphs in the concept that they use horizontal and vertical axes to plot data points.
In the following example, the Python script will generate and plot the Scatter matrix for the Iris dataset. It can be generated with the help of scatter_matrix() function on Pandas DataFrame and plotted with the help of pyplot.

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# Load the iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Create the scatter matrix plot
pd.plotting.scatter_matrix(df, diagonal='hist', figsize=(8, 7))
plt.show()
```

#### Output
![Scatter Matrix Plot](/machine_learning/images/ml_data_visualization_scatter_matrix_plot.jpg)
In the next few chapters, we will look at some of the popular and widely used visualization techniques available in machine learning.

---

## 7. Data Preparation in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_preparing_data.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_preparing_data.htm)*

---

---
[Previous](/machine_learning/machine_learning_data_understanding.htm)[Quiz](/machine_learning/quiz_on_machine_learning_preparing_data.htm)[Next](/machine_learning/machine_learning_models.htm)
Data preparation is a critical step in the machine learning process, and can have a significant impact on the accuracy and effectiveness of the final model. It requires careful attention to detail and a thorough understanding of the data and the problem at hand.

Let's discuss how data should be prepared in order to fit right with the model for better accuracy and outcome.

## What is Data Preparation?

Data preparation is the process of dealing with raw data i.e, cleaning, organizing and transforming it to align with the machine learning algorithms. Data preparation is a continuous process, and has a huge impact on the performance of machine learning model. Clean and structured data would result in better outcomes.

## Importance of Data Preparation

In Machine learning, the model learns from the data that is fed. So, the algorithm can learn efficiently only if the data is organized and perfect. The quality of the data you use for your model can have a significant impact on the performance of the model.

Few aspects that define the importance of data preparation in machine learning are −

- **Improves model accuracy**− Machine learning algorithms reply completely on data. When you provide clean and structured data to models, the outcomes are accurate.
- **Facilitates Feature Engineering**− Data preparation often includes the process of selecting or creating new features to train the model. Hence, data preparation would make feature engineering easy.
- **Data Quality**− Collected data most often would contain inconsistencies, errors and irrelevant information. Hence when tasks like data cleaning, transformation are applied, the data is formatted and neat. This can be used for gaining insights and patterns.
- **Enables rate of prediction**− Prepared data makes it easier to analyze results and would yield accurate outcomes.
## Data Preparation Process Steps

Data preparation process involves a sequence of steps that is required to make data suitable for analysis and modeling. The goal of data preparation is to make sure that the data is accurate, complete, and relevant for the analysis.

The following are some of the key steps involved in data preparation −

- Data Collection
- Data Cleaning
- Data Transformation
- Data Reduction
- Data Splitting![ML Data Preparation Steps](/machine_learning/images/machine_learning_data_preparation.jpg)
> The process shown is not always sequential. You might, for example, split your data before you transform it. You might need to collect more data.

Let's understand each of the above steps in detail −

## Data Collection

Data collection is the first step in the process of machine learning, where data from different sources is gathered to make decisions, answer research questions and statistical planning. Different sources such as databases, text files, pictures, sound files, or web scraping may be used for data collection. Once the data is selected, the data has to be preprocessed in order to gain insights. This process is carried out to put the data in an appropriate format that would be useful for problem solving. Some time data collection follows the data integration step.
**Data integration**involves combining data from multiple sources into a single dataset for analysis. This may involve matching or linking records across different datasets, or merging datasets based on common variables.
After selecting the raw data, the most important task is
**data preprocessing**. In broad sense, data preprocessing will convert the selected data into a form we can work with or can feed to ML algorithms. We always need to preprocess our data so that it can be as per the expectation of machine learning algorithm. The data preprocessing includes data cleaning, transformation and reduction. Let's discuss each of these three in detail.
## Data Cleaning
**Data cleaning**is the process of identifying and correcting errors, missing values, duplicate values and outliers, etc. in the data. This step is crucial in the process of machine learning as it ensures that the data is accurate, relevant and error free.
Common techniques used for data cleaning include imputation, outlier detection and removal, etc. The following is a sequence of steps for data cleaning −

### 1. Handling duplicate values

Duplicates in the dataset means that there is repeated data, which might occur due to data entry errors or issues while collecting data. The technique used to remove duplicates is first they are identified and then deleted using
**drop_duplicates function in Pandas**.
### 2. Fixing syntax errors

In this step, structural errors like inconsistencies in data format or naming conventions should be addressed. Standardizing formats and fixing errors would ensure data consistence and accurate analysis.

### 3. Dealing outliers

Outliers are values that are unusual and differ greatly with the data. The techniques used to detect outliers include statistical methods like
**z-score**or**IQR method**and machine learning methods like**clustering**and**SVM's**.
### 4. Handling Missing Values

Missing values are the values or data that is not stored for some values in the dataset. There are several ways to handle missing data like:

- **Imputation**− In this process the missing values are substituted with different value, which can be a central tendency measure like mean, median or mode for numeric values and most frequency category for categorical data. Some other methods in imputation include regression imputation and multiple imputation.
- **Deletion**− In this process the entire instances with missing values are removed. Well, this is not a reliable method since there is loss of data.
### 5. Validating the data
**Data Validation**is another stage that makes sure that the data aligns perfectly with the requirements so that the predicted outcome is accurate. Some common data validation procedures it the correctness of data before storing them in databases are:
- Data type check
- Code Check
- Format check
- Range check
## Data Transformation
**Data transformation**is the process of converting the data from its original format into a format that is suitable for analysis and modeling. This could include defining the structure, aligning the data, extracting data from source, and then storing it in an appropriate form.
There are many techniques available to transorm data into a sutable format. Some commonly used
**data transformation techniques**are as follows −
- Scaling
- Normalization − L1 & L2 Normalizations
- Standardization
- Binarization
- Encoding
- Log Transformation
Lets discuss each of the above data transformation techniques in detail −

### 1. Scaling

In most cases, the data we collected consists of attributes with varying scale, but we cannot provide such data to ML algorithm hence it requires rescaling.
**Data scaling**makes sure that attributes are at the same scale i.e, usually range of 0 to 1.
We can rescale the data with the help of MinMaxScaler class of scikit-learn Python library.

#### Example

In this example we will rescale the data of Pima Indians Diabetes dataset which we used earlier. First, the CSV data will be loaded (as done in the previous chapters) and then with the help of MinMaxScaler class, it will be rescaled in the range of 0 and 1.

The first few lines of the following script are same as we have written in previous chapters while loading CSV data.

```
from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing
path = r'C:\pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values
```

Now, we can use MinMaxScaler class to rescale the data in the range of 0 and 1.

```
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_rescaled = data_scaler.fit_transform(array)
```

We can also summarize the data for output as per our choice. Here, we are setting the precision to 1 and showing the first 10 rows in the output.

```
set_printoptions(precision=1)
print ("\nScaled data:\n", data_rescaled[0:10])
```

##### Output

```
Scaled data:
[
   [0.4 0.7 0.6 0.4 0.  0.5 0.2 0.5 1. ]
   [0.1 0.4 0.5 0.3 0.  0.4 0.1 0.2 0. ]
   [0.5 0.9 0.5 0.  0.  0.3 0.3 0.2 1. ]
   [0.1 0.4 0.5 0.2 0.1 0.4 0.  0.  0. ]
   [0.  0.7 0.3 0.4 0.2 0.6 0.9 0.2 1. ]
   [0.3 0.6 0.6 0.  0.  0.4 0.1 0.2 0. ]
   [0.2 0.4 0.4 0.3 0.1 0.5 0.1 0.1 1. ]
   [0.6 0.6 0.  0.  0.  0.5 0.  0.1 0. ]
   [0.1 1.  0.6 0.5 0.6 0.5 0.  0.5 1. ]
   [0.5 0.6 0.8 0.  0.  0.  0.1 0.6 1. ]
]
```

From the above output, all the data got rescaled into the range of 0 and 1.

### 2. Normalization

Normalization is used to rescale the data with a distribution value between 0 and 1. For every feature, the minimum value is set to 0 and the maximum value is set to 1.

This is used to rescale each row of data to have a length of 1. It is mainly useful in Sparse dataset where we have lots of zeros. We can rescale the data with the help of Normalizer class of scikit-learn Python library.

In machine learning, there are
**two types**of normalization preprocessing techniques as follows −
#### L1 Normalization

It may be defined as the normalization technique that modifies the dataset values in a way that in each row the sum of the absolute values will always be up to 1. It is also called Least Absolute Deviations.

##### Example

In this example, we use L1 Normalize technique to normalize the data of Pima Indians Diabetes dataset which we used earlier. First, the CSV data will be loaded and then with the help of Normalizer class it will be normalized.

The first few lines of following script are same as we have written in previous chapters while loading CSV data.

```
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer
path = r'C:\pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv (path, names=names)
array = dataframe.values
```

Now, we can use Normalizer class with L1 to normalize the data.

```
Data_normalizer = Normalizer(norm='l1').fit(array)
Data_normalized = Data_normalizer.transform(array)
```

We can also summarize the data for output as per our choice. Here, we are setting the precision to 2 and showing the first 3 rows in the output.

```
set_printoptions(precision=2)
print ("\nNormalized data:\n", Data_normalized [0:3])
```

###### Output

```
Normalized data:
[
   [0.02 0.43 0.21 0.1  0. 0.1  0. 0.14 0. ]
   [0.   0.36 0.28 0.12 0. 0.11 0. 0.13 0. ]
   [0.03 0.59 0.21 0.   0. 0.07 0. 0.1  0. ]
]
```

#### L2 Normalization

It may be defined as the normalization technique that modifies the dataset values in a way that in each row the sum of the squares will always be up to 1. It is also called least squares.

##### Example

In this example, we use L2 Normalization technique to normalize the data of Pima Indians Diabetes dataset which we used earlier. First, the CSV data will be loaded (as done in previous chapters) and then with the help of Normalizer class it will be normalized.

The first few lines of following script are same as we have written in previous chapters while loading CSV data.

```
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer
path = r'C:\pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv (path, names=names)
array = dataframe.values
```

Now, we can use Normalizer class with L1 to normalize the data.

```
Data_normalizer = Normalizer(norm='l2').fit(array)
Data_normalized = Data_normalizer.transform(array)
```

We can also summarize the data for output as per our choice. Here, we are setting the precision to 2 and showing the first 3 rows in the output.

```
set_printoptions(precision=2)
print ("\nNormalized data:\n", Data_normalized [0:3])
```

###### Output

```
Normalized data:
[
   [0.03 0.83 0.4  0.2  0. 0.19 0. 0.28 0.01]
   [0.01 0.72 0.56 0.24 0. 0.22 0. 0.26 0.  ]
   [0.04 0.92 0.32 0.   0. 0.12 0. 0.16 0.01]
]
```

### 3. Standardization

Standardization is used to transform data attributes to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1. This technique is useful in ML algorithms like linear regression, logistic regression that assumes a Gaussian distribution in input dataset and produce better results with rescaled data.

We can standardize the data (mean = 0 and SD =1) with the help of StandardScaler class of scikit-learn Python library.

#### Example

In this example, we will rescale the data of Pima Indians Diabetes dataset which we used earlier. First, the CSV data will be loaded and then with the help of StandardScaler class it will be converted into Gaussian Distribution with mean = 0 and SD = 1.

The first few lines of following script are same as we have written in previous chapters while loading CSV data.

```
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
path = r'C:\pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values
```

Now, we can use StandardScaler class to rescale the data.

```
data_scaler = StandardScaler().fit(array)
data_rescaled = data_scaler.transform(array)
```

We can also summarize the data for output as per our choice. Here, we are setting the precision to 2 and showing the first 5 rows in the output.

```
set_printoptions(precision=2)
print ("\nRescaled data:\n", data_rescaled [0:5])
```

##### Output

```
Rescaled data:
[
   [ 0.64  0.85  0.15  0.91 -0.69  0.2   0.47  1.43  1.37]
   [-0.84 -1.12 -0.16  0.53 -0.69 -0.68 -0.37 -0.19 -0.73]
   [ 1.23  1.94 -0.26 -1.29 -0.69 -1.1   0.6  -0.11  1.37]
   [-0.84 -1.   -0.16  0.15  0.12 -0.49 -0.92 -1.04 -0.73]
   [-1.14  0.5  -1.5   0.91  0.77  1.41  5.48 -0.02  1.37]
]
```

### 4. Binarization

As the name suggests, this is the technique with the help of which we can make our data binary. We can use a binary threshold for making our data binary. The values above that threshold value will be converted to 1 and below that threshold will be converted to 0. For example, if we choose threshold value = 0.5, then the dataset value above it will become 1 and below this will become 0. That is why we can call it
**binarizing**the data or**thresholding**the data. This technique is useful when we have probabilities in our dataset and want to convert them into crisp values.
We can binarize the data with the help of Binarizer class of scikit-learn Python library.

#### Example

In this example, we will rescale the data of Pima Indians Diabetes dataset which we used earlier. First, the CSV data will be loaded and then with the help of Binarizer class it will be converted into binary values i.e. 0 and 1 depending upon the threshold value. We are taking 0.5 as threshold value.

The first few lines of following script are same as we have written in previous chapters while loading CSV data.

```
from pandas import read_csv
from sklearn.preprocessing import Binarizer
path = r'C:\pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values
```

Now, we can use Binarize class to convert the data into binary values.

```
binarizer = Binarizer(threshold=0.5).fit(array)
Data_binarized = binarizer.transform(array)
```

Here, we are showing the first 5 rows in the output.

```
print ("\nBinary data:\n", Data_binarized [0:5])
```

##### Output

```
Binary data:
[
   [1. 1. 1. 1. 0. 1. 1. 1. 1.]
   [1. 1. 1. 1. 0. 1. 0. 1. 0.]
   [1. 1. 1. 0. 0. 1. 1. 1. 1.]
   [1. 1. 1. 1. 1. 1. 0. 1. 0.]
   [0. 1. 1. 1. 1. 1. 1. 1. 1.]
]
```

### 5. Encoding

This technique is used to convert categorical variables into numerical representations. Some common encoding techniques include
**one-hot encoding**,**label encoding**and**target encoding**.
#### Label Encoding

Most of the sklearn functions expect that the data with number labels rather than word labels. Hence, we need to convert such labels into number labels. This process is called label encoding. We can perform label encoding of data with the help of LabelEncoder() function of scikit-learn Python library.

##### Example

In the following example, Python script will perform the label encoding.

First, import the required Python libraries as follows −

```
import numpy as np
from sklearn import preprocessing
```

Now, we need to provide the input labels as follows −

```
input_labels = ['red','black','red','green','black','yellow','white']
```

The next line of code will create the label encoder and train it.

```
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
```

The next lines of script will check the performance by encoding the random ordered list −

```
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
```

We can get the list of encoded values with the help of following python script −

```
print("\nEncoded values =", encoded_values)
print("\nDecoded labels =", list(decoded_list))
```

###### Output

```
Labels = ['green', 'red', 'black']
Encoded values = [1, 2, 0]
Encoded values = [3, 0, 4, 1]
Decoded labels = ['white', 'black', 'yellow', 'green']
```

### 6. Log Transformation

This technique is usually used in handling skewed data. It involves apply natural logarithmic function for all values in the dataset to modify the scale of numeric values.

## Data Reduction

Data Reduction is a technique to reduce the size of the dataset by selecting a subset of features or observations that are most relevant for the analysis. This can help to reduce noise and improve the accuracy of the model.

This is useful when the dataset is very large or when a dataset contains large amount of irrelevant data.

One of the most common technique used is
**Dimensionality Reduction**, which reduces the size of the dataset without loosing the important information. Other method is the**Discretization**, where continuous values like time and temperature are converted to discrete categories which simplifies the data.
## Data Splitting
**Data Splitting**is the last step in the preparation of data for machine learning, where the data is split into different sets -
- **Training**− subset which is used by the machine learning model for learning patterns.
- **Validation**−  subset used to evaluate the performance of machine learning model while training.
- **Testing**−  subset used to evaluate the performance and efficiency of the trained model.
## Python Example

Let's check an example of data preparation using the breast cancer dataset −

```
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the dataset
data = load_breast_cancer()

# separate the features and target
X = data.data
y = data.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

In this example, we first load the breast cancer dataset using load_breast_cancer function from scikit-learn. Then we separate the features and target, and split the data into training and testing sets using train_test_split function.

Finally, we normalize the data using StandardScaler from scikit-learn, which subtracts the mean and scales the data to unit variance. This helps to bring all the features to a similar scale, which is particularly important for models like SVM and neural networks.

## Data Preparation and Feature Engineering
**Feature engineering**involves creating new features from the existing data that may be more informative or useful for the analysis. It can involve combining or transforming existing features, or creating new features based on domain knowledge or insights. Both data preparation and feature engineering go hand-in-hand in the overall data preprocessing pipeline.

---

## 8. Machine Learning - Feature Selection

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_feature_selection.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_feature_selection.htm)*

---

---
[Previous](/machine_learning/machine_learning_dimensionality_reduction.htm)[Quiz](/machine_learning/quiz_on_machine_learning_feature_selection.htm)[Next](/machine_learning/machine_learning_feature_extraction.htm)
Feature selection is an important step in machine learning that involves selecting a subset of the available features to improve the performance of the model. The following are some commonly used feature selection techniques −

## Filter Methods

This method involves evaluating the relevance of each feature by calculating a statistical measure (e.g., correlation, mutual information, chi-square, etc.) and ranking the features based on their scores. Features that have low scores are then removed from the model.

To implement filter methods in Python, you can use the SelectKBest or SelectPercentile functions from the sklearn.feature_selection module. Below is a small code snippet to implement Feature selection.

```
from sklearn.feature_selection import SelectPercentile, chi2
selector = SelectPercentile(chi2, percentile=10)
X_new = selector.fit_transform(X, y)
```

## Wrapper Methods

This method involves evaluating the model's performance by adding or removing features and selecting the subset of features that yields the best performance. This approach is computationally expensive, but it is more accurate than filter methods.

To implement wrapper methods in Python, you can use the RFE (Recursive Feature Elimination) function from the sklearn.feature_selection module. Below is a small code snippet to implement Wrapper method.

```
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=5)
selector = selector.fit(X, y)
X_new = selector.transform(X)
```

## Embedded Methods

This method involves incorporating feature selection into the model building process itself. This can be done using techniques such as Lasso regression, Ridge regression, or Decision Trees. These methods assign weights to each feature and features with low weights are removed from the model.

To implement embedded methods in Python, you can use the Lasso or Ridge regression functions from the sklearn.linear_model module. Below is a small code snippet for implementing embedded methods −

```
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
coef = pd.Series(lasso.coef_, index = X.columns)
important_features = coef[coef != 0]
```

## Principal Component Analysis (PCA)

This is a type of unsupervised learning method that involves transforming the original features into a set of uncorrelated principal components that explain the maximum variance in the data. The number of principal components can be selected based on a threshold value, which can reduce the dimensionality of the dataset.

To implement PCA in Python, you can use the PCA function from the sklearn.decomposition module. For example, to reduce the number of features you can use PCA as given the following code −

```
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_new = pca.fit_transform(X)
```

## Recursive Feature Elimination (RFE)

This method involves recursively eliminating the least significant features until a subset of the most important features is identified. It uses a model-based approach and can be computationally expensive, but it can yield good results in high-dimensional datasets.

To implement RFE in Python, you can use the RFECV (Recursive Feature Elimination with Cross Validation) function from the sklearn.feature_selection module. For example, below is a small code snippet with the help of which we can implement to use Recursive Feature Elimination −

```
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
X_new = selector.transform(X)
```

These feature selection techniques can be used alone or in combination to improve the performance of machine learning models. It is important to choose the appropriate technique based on the size of the dataset, the nature of the features, and the type of model being used.

### Example

In the below example, we will implement three feature selection methods − univariate feature selection using the chi-square test, recursive feature elimination with cross-validation (RFECV), and principal component analysis (PCA).

We will use the Breast Cancer Wisconsin (Diagnostic) Dataset, which is included in scikit-learn. This dataset contains 569 samples with 30 features, and the task is to classify whether a tumor is malignant or benign based on these features.

Here is the Python code to implement these feature selection methods on the Breast Cancer Wisconsin (Diagnostic) Dataset −

```
# Import necessary libraries and dataset
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
diabetes = pd.read_csv(r'C:\Users\Leekha\Desktop\diabetes.csv')

# Split the dataset into features and target variable
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

# Apply univariate feature selection using the chi-square test
selector = SelectKBest(chi2, k=4)
X_new = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Fit a logistic regression model on the selected features
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = clf.score(X_test, y_test)
print("Accuracy using univariate feature selection: {:.2f}".format(accuracy))

# Recursive feature elimination with cross-validation (RFECV)
estimator = LogisticRegression()
selector = RFECV(estimator, step=1, cv=5)
selector.fit(X, y)
X_new = selector.transform(X)
scores = cross_val_score(LogisticRegression(), X_new, y, cv=5)
print("Accuracy using RFECV feature selection: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# PCA implementation
pca = PCA(n_components=5)
X_new = pca.fit_transform(X)
scores = cross_val_score(LogisticRegression(), X_new, y, cv=5)
print("Accuracy using PCA feature selection: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

#### Output

When you execute this code, it will produce the following output on the terminal −

```
Accuracy using univariate feature selection: 0.74
Accuracy using RFECV feature selection: 0.77 (+/- 0.03)
Accuracy using PCA feature selection: 0.75 (+/- 0.07)
```

---

## 9. Machine Learning - Classification Algorithms

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_classification_algorithms.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_classification_algorithms.htm)*

---

---
[Previous](/machine_learning/machine_learning_polynomial_regression.htm)[Quiz](/machine_learning/quiz_on_machine_learning_classification_algorithms.htm)[Next](/machine_learning/machine_learning_logistic_regression.htm)
## Classification in Machine Learning

Classification may be defined as the process of predicting class or category from observed values or given data points. The categorized output can have the form such as "Black" or "White" or "spam" or "no spam".

Classification in machine learning is a
[supervised learning](/machine_learning/machine_learning_supervised.htm)technique where an algorithm is trained with labeled data to predict the category of new data.
Mathematically, classification is the task of approximating a mapping function (f) from input variables (X) to output variables (Y). It is basically belongs to the supervised machine learning in which targets are also provided along with the input data set.

An example of classification problem can be the spam detection in emails. There can be only two categories of output, "spam" and "no spam"; hence this is a binary type classification.

To implement this classification, we first need to train the classifier. For this example, "spam" and "no spam" emails would be used as the training data. After successfully train the classifier, it can be used to detect an unknown email.

## Types of Learners in Classification

We have two types of learners in respective to classification problems −

- **Lazy Learners**− As the name suggests, such kind of learners waits for the testing data to be appeared after storing the training data. Classification is done only after getting the testing data. They spend less time on training but more time on predicting. Examples of lazy learners are K-nearest neighbor and case-based reasoning.
- **Eager Learners**− As opposite to lazy learners, eager learners construct classification model without waiting for the testing data to be appeared after storing the training data. They spend more time on training but less time on predicting. Examples of eager learners are Decision Trees, Nave Bayes and Artificial Neural Networks (ANN).
## Classification Algorithms in Machine Learning

The classification algorithm is a type of supervised learning technique that involves predicting a categorical target variable based on a set of input features. It is commonly used to solve problems such as spam detection, fraud detection, image recognition, sentiment analysis, and many others.

The goal of a classification model is to learn a mapping function (f) between the input features (X) and the target variable (Y). This mapping function is often represented as a decision boundary, which separates different classes in the input feature space. Once the model is trained, it can be used to predict the class of new, unseen examples.

The followings are some important ML classification algorithms −

- [Logistic Regression](#logistic_regression)
- [K-Nearest Neighbors (KNN)](#k_nearest_neighbors)
- [Support Vector Machine (SVM)](#support_vector_machine)
- [Decision Tree](#decision_tree)
- [Nave Bayes](#naive_bayes)
- [Random Forest](#random_forest)
We will be discussing all these classification algorithms in detail in further chapters. However let's discuss these algorithms in brief as follows −

### Logistic Regression
[Logistic Regression](/machine_learning/machine_learning_logistic_regression.htm)is a popular algorithm used for binary classification problems, where the target variable is categorical with two classes. It models the probability of the target variable given the input features and predicts the class with the highest probability.
Logistic regression is a type of generalized linear model, where the target variable follows a Bernoulli distribution. The model consists of a linear function of the input features, which is transformed using the logistic function to produce a probability value between 0 and 1.

### K-Nearest Neighbors (KNN)
[K-Nearest Neighbors (KNN)](/machine_learning/machine_learning_knn_nearest_neighbors.htm)is a supervised learning algorithm that can be used for both classification and regression problems. The main idea behind KNN is to find the k-nearest data points to a given test data point and use these nearest neighbors to make a prediction. The value of k is a hyperparameter that needs to be tuned, and it represents the number of neighbors to consider.
For classification problems, the KNN algorithm assigns the test data point to the class that appears most frequently among the k-nearest neighbors. In other words, the class with the highest number of neighbors is the predicted class.

For regression problems, the KNN algorithm assigns the test data point the average of the k-nearest neighbors' values.

### Support Vector Machine (SVM)
[Support Vector Machines (SVMs)](/machine_learning/machine_learning_support_vector_machine.htm)are powerful yet flexible supervised machine learning algorithm which is used for both classification and regression. But generally, they are used in classification problems. In 1960s, SVMs were first introduced but later they got refined in 1990 also. SVMs have their unique way of implementation as compared to other machine learning algorithms. Now a days, they are extremely popular because of their ability to handle multiple continuous and categorical variables.
### Decision Tree

The
[Decision Tree algorithm](/machine_learning/machine_learning_decision_tree_algorithm.htm)is a hierarchical tree-based algorithm that is used to classify or predict outcomes based on a set of rules. It works by splitting the data into subsets based on the values of the input features. The algorithm recursively splits the data until it reaches a point where the data in each subset belongs to the same class or has the same value for the target variable. The resulting tree is a set of decision rules that can be used to make predictions or classify new data.
### Nave Bayes

The
[Nave Bayes algorithm](/machine_learning/machine_learning_naive_bayes_algorithms.htm)is a classification algorithm based on Bayes' theorem. The algorithm assumes that the features are independent of each other, which is why it is called "naive." It calculates the probability of a sample belonging to a particular class based on the probabilities of its features. For example, a phone may be considered as smart if it has touch-screen, internet facility, good camera, etc. Even if all these features are dependent on each other, but all these features independently contribute to the probability of that the phone is a smart phone.
### Random Forest
[Random Forest](/machine_learning/machine_learning_random_forest_classification.htm)is a machine learning algorithm that uses an ensemble of decision trees to make predictions. The algorithm was first introduced by Leo Breiman in 2001. The key idea behind the algorithm is to create a large number of decision trees, each of which is trained on a different subset of the data. The predictions of these individual trees are then combined to produce a final prediction.
## Applications of Classification in Machine Learning

Some of the most important applications of classification algorithms are as follows −

- Speech Recognition
- Handwriting Recognition
- Biometric Identification
- Document Classification
- Image Classification
- Spam Filtering
- Fraud Detection
- Facial Recognition
## Building a Classication Model in Machine Learning

Let us now take a look at the steps involved in building a classification model −

### 1. Data Preparation

The first step is to collect and preprocess the data. This involves cleaning the data, handling missing values, and converting categorical variables to numerical values.

### 2. Feature Extraction/Selection

The next step is to extract or select relevant features from the data. This is an important step because the quality of the features can greatly impact the performance of the model. Some common feature selection techniques include correlation analysis, feature importance ranking, and principal component analysis.

### 3. Model Selection

Once the features are selected, the next step is to choose an appropriate classification algorithm. There are many different algorithms to choose from, each with its own strengths and weaknesses. Some popular algorithms include logistic regression, decision trees, random forests, support vector machines, and neural networks

### 4. Model Training

After selecting a suitable algorithm, the next step is to train the model on the labeled training data. During training, the model learns the mapping function between the input features and the target variable. The model parameters are adjusted iteratively to minimize the difference between the predicted outputs and the actual outputs.

### 5. Model Evaluation

Once the model is trained, the next step is to evaluate its performance on a separate set of validation data. This is done to estimate the model's accuracy and generalization performance. Common evaluation metrics include accuracy, precision, recall, F1-score, and area under the receiver operating characteristic (ROC) curve.

### 5. Hyperparameter Tuning

In many cases, the performance of the model can be further improved by tuning its hyperparameters. Hyperparameters are settings that are chosen before training the model and control aspects such as the learning rate, regularization strength, and the number of hidden layers in a neural network. Grid search, random search, and Bayesian optimization are some common techniques used for hyperparameter tuning.

### 6. Model Deployment

Once the model has been trained and evaluated, the final step is to deploy it in a production environment. This involves integrating the model into a larger system, testing it on realworld data, and monitoring its performance over time.

## Building a Classification Model with Python

Scikit-learn, a Python library for machine learning can be used to build a classifier in Python. The steps for building a classifier in Python are as follows −

### Step 1: Importing necessary python package

For building a classifier using scikit-learn, we need to import it. We can import it by using following script −

```
import sklearn
```

### Step 2: Importing dataset

After importing necessary package, we need a dataset to build classification prediction model. We can import it from sklearn dataset or can use other one as per our requirement. We are going to use sklearns Breast Cancer Wisconsin Diagnostic Database. We can import it with the help of following script −

```
from sklearn.datasets import load_breast_cancer
```

The following script will load the dataset;

```
data = load_breast_cancer()
```

We also need to organize the data and it can be done with the help of following scripts −

```
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
```

The following code will print the name of the labels,
**malignant**and**'benign'**in case of our database.
```
import sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

print(label_names)
```

The output of the above command is the names of the labels −

```
['malignant' 'benign']
```

These labels are mapped to binary values 0 and 1.
**Malignant**cancer is represented by 0 and**Benign**cancer is represented by 1.
The feature names and feature values of these labels can be seen with the help of following commands −

```
print(feature_names[0])
```

The output of the above command is the names of the features for label 0 i.e.
**Malignant**cancer −
```
mean radius
```

Similarly, names of the features for label can be produced as follows −

```
print(feature_names[1])
```

The output of the above command is the names of the features for label 1 i.e. Benign cancer −

```
mean texture
```

We can print the features for these labels with the help of following command −

```
print(features[0])
```

This will give the following output −

```
[1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01
 1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02
 6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01
 1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01
 4.601e-01 1.189e-01]
```

We can print the features for these labels with the help of following command −

```
print(features[1])
```

This will give the following output −

```
[2.057e+01 1.777e+01 1.329e+02 1.326e+03 8.474e-02 7.864e-02 8.690e-02
7.017e-02  1.812e-01 5.667e-02 5.435e-01 7.339e-01 3.398e+00 7.408e+01
5.225e-03  1.308e-02 1.860e-02 1.340e-02 1.389e-02 3.532e-03 2.499e+01
2.341e+01  1.588e+02 1.956e+03 1.238e-01 1.866e-01 2.416e-01 1.860e-01
2.750e-01  8.902e-02]
```

### Step 3: Organizing data into training & testing sets

As we need to test our model on unseen data, we will divide our dataset into two parts: a training set and a test set. We can use
*train_test_split()*function of*sklearn*python package to split the data into sets. The following command will import the function −
```
from sklearn.model_selection import train_test_split
```

Now, next command will split the data into training & testing data. In this example, we are using taking 40 percent of the data for testing purpose and 60 percent of the data for training purpose −

```
train, test, train_labels, test_labels = 
   train_test_split(features,labels,test_size = 0.40, random_state = 42)
```

### Step 4: Model evaluation

After dividing the data into training and testing we need to build the model. We will be using
*Nave Bayes*algorithm for this purpose. The following commands will import the*GaussianNB*module −
```
from sklearn.naive_bayes import GaussianNB
```

Now, initialize the model as follows −

```
gnb = GaussianNB()
```

Next, with the help of following command we can train the model −

```
model = gnb.fit(train, train_labels)
```

Now, for evaluation purpose we need to make predictions. It can be done by using predict() function as follows −

```
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels=train_test_split(features,labels,test_size = 0.40, random_state = 42)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)
```

This will give the following output −

```
[1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
 1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0
 1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0
 1 1 0 0 0 1 1 1 0 0 1 1 0 1 0 0 1 1 0 0 0 1 1 1 0 1 1 0 0 1 0 1 1 0 1 0 0
 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 0
 0 1 1 0 1 0 1 1 1 1 0 1 1 0 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1
 0 0 1 1 0 1]
```

The above series of 0s and 1s in output are the predicted values for the
**Malignant**and**Benign**tumor classes.
### Step 5: Finding accuracy

We can find the accuracy of the model build in previous step by comparing the two arrays namely
*test_labels*and*preds*. We will be using the*accuracy_score()*function to determine the accuracy.
```
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,preds))
```

```
0.951754385965
```

The above output shows that
*NaveBayes*classifier is 95.17% accurate.
## Evaluation Metrics for Classification Model

The job is not done even if you have finished implementation of your Machine Learning application or model. We must have to find out how effective our model is? There can be different
[evaluation/ performance metrics](/machine_learning/machine_learning_performance_metrics.htm), but we must choose it carefully because the choice of metrics influences how the performance of a machine learning algorithm is measured and compared.
The following are some of the important classification evaluation metrics among which you can choose based upon your dataset and kind of problem −

### Confusion Matrix

The
[confusion matrix](/machine_learning/machine_learning_confusion_matrix.htm)is the easiest way to measure the performance of a classification problem where the output can be of two or more type of classes. A confusion matrix is nothing but a table with two dimensions viz. "Actual" and "Predicted" and furthermore, both the dimensions have "True Positives (TP)", "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)" as shown below −![Confusion Matrix](/machine_learning/images/confusion_matrix.jpg)
The explanation of the terms associated with confusion matrix are as follows −

- **True Positives (TP)**− It is the case when both actual class & predicted class of data point is 1.
- **True Negatives (TN)**− It is the case when both actual class & predicted class of data point is 0.
- **False Positives (FP)**− It is the case when actual class of data point is 0 & predicted class of data point is 1.
- **False Negatives (FN)**− It is the case when actual class of data point is 1 & predicted class of data point is 0.
We can find the confusion matrix with the help of confusion_matrix() function of sklearn. With the help of the following script, we can find the confusion matrix of above built binary classifier −

```
from sklearn.metrics import confusion_matrix
preds = gnb.predict(test)
cm = confusion_matrix(test, preds)
print(cm)
```

### Output

```
[
   [ 73   7]
   [  4 144]
]
```

### Accuracy

It may be defined as the number of correct predictions made by our ML model. We can easily calculate it by confusion matrix with the help of following formula −

$$\mathrm{Accuracy=\frac{TP+TN}{TP+FP+FN+TN}}$$

For above built binary classifier, TP + TN = 73+144 = 217 and TP+FP+FN+TN = 73+7+4+144=228.

Hence, Accuracy = 217/228 = 0.951754385965 which is same as we have calculated after creating our binary classifier.

### Precision

Precision, used in document retrievals, may be defined as the number of correct documents returned by our ML model. We can easily calculate it by confusion matrix with the help of following formula −

$$\mathrm{Precision=\frac{TP}{TP+FP}}$$

For the above built binary classifier, TP = 73 and TP+FP = 73+7 = 80.

Hence, Precision = 73/80 = 0.915

### Recall or Sensitivity

Recall may be defined as the number of positives returned by our ML model. We can easily calculate it by confusion matrix with the help of following formula −

$$\mathrm{Recall=\frac{TP}{TP+FN}}$$

For above built binary classifier, TP = 73 and TP+FN = 73+4 = 77.

Hence, Precision = 73/77 = 0.94805

### Specificity

Specificity, in contrast to recall, may be defined as the number of negatives returned by our ML model. We can easily calculate it by confusion matrix with the help of following formula −

$$\mathrm{Specificity=\frac{TN}{TN+FP}}$$

For the above built binary classifier, TN = 144 and TN+FP = 144+7 = 151.

Hence, Precision = 144/151 = 0.95364

In the subsequent chapters, we will discuss some of the most popular classification algorithms in machine learning in detail.

---

## 10. Logistic Regression in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_logistic_regression.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_logistic_regression.htm)*

---

---
[Previous](/machine_learning/machine_learning_classification_algorithms.htm)[Quiz](/machine_learning/quiz_on_machine_learning_logistic_regression.htm)[Next](/machine_learning/machine_learning_knn_nearest_neighbors.htm)
## Introduction to Logistic Regression

Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. The nature of target or dependent variable is dichotomous, which means there would be only two possible classes.

In simple words, the dependent variable is binary in nature having data coded as either 1 (stands for success/yes) or 0 (stands for failure/no).

Mathematically, a logistic regression model predicts P(Y=1) as a function of X. It is one of the simplest ML algorithms that can be used for various classification problems such as spam detection, Diabetes prediction, cancer detection etc.

## Types of Logistic Regression

Generally, logistic regression means binary logistic regression having binary target variables, but there can be two more categories of target variables that can be predicted by it. Based on those number of categories, Logistic regression can be divided into following types −

### Binary or Binomial

In such a kind of classification, a dependent variable will have only two possible types either 1 and 0. For example, these variables may represent success or failure, yes or no, win or loss etc.

### Multinomial

In such a kind of classification, dependent variable can have 3 or more possible unordered types or the types having no quantitative significance. For example, these variables may represent "Type A" or "Type B" or "Type C".

### Ordinal

In such a kind of classification, dependent variable can have 3 or more possible ordered types or the types having a quantitative significance. For example, these variables may represent "poor" or "good", "very good", "Excellent" and each category can have the scores like 0,1,2,3.

## Logistic Regression Assumptions

Before diving into the implementation of logistic regression, we must be aware of the following assumptions about the same −

- 
In case of binary logistic regression, the target variables must be binary always and the desired outcome is represented by the factor level 1.

- 
There should not be any multi-collinearity in the model, which means the independent variables must be independent of each other .

- 
We must include meaningful variables in our model.

- 
We should choose a large sample size for logistic regression.

## Binary Logistic Regression Model

The simplest form of logistic regression is binary or binomial logistic regression in which the target or dependent variable can have only 2 possible types either 1 or 0. It allows us to model a relationship between multiple predictor variables and a binary/binomial target variable. In case of logistic regression, the linear function is basically used as an input to another function such as  in the following relation −
$$h_{\theta}{(x)}=g(\theta^{T}x)  0h_{\theta}1$$
Here,  is the logistic or sigmoid function which can be given as follows −
$$g(z)= \frac{1}{1+e^{-z}}  =\theta  ^{T}$$
To sigmoid curve can be represented with the help of following graph. We can see the values of y-axis lie between 0 and 1 and crosses the axis at 0.5.
![sigmoid curve](/machine_learning/images/flow.jpg)
The classes can be divided into positive or negative. The output comes under the probability of positive class if it lies between 0 and 1. For our implementation, we are interpreting the output of hypothesis function as positive if it is 0.5, otherwise negative.

We also need to define a loss function to measure how well the algorithm performs using the weights on functions, represented by theta as follows −

$$=()$$

$$J(\theta) = \frac{1}{m}.(-y^{T}log(h) - (1 -y)^Tlog(1-h))$$

Now, after defining the loss function our prime goal is to minimize the loss function. It can be done with the help of fitting the weights which means by increasing or decreasing the weights. With the help of derivatives of the loss function w.r.t each weight, we would be able to know what parameters should have high weight and what should have smaller weight.

The following gradient descent equation tells us how loss would change if we modified the parameters −
$$\frac{()}{\theta_{j}}=\frac{1}{m}X^{T}(())$$
### Implementation of Binary Logistic Regression Model in Python

Now we will implement the above concept of binomial logistic regression in Python. For this purpose, we are using a multivariate flower dataset named iris which have 3 classes of 50 instances each, but we will be using the first two feature columns. Every class represents a type of iris flower.

First, we need to import the necessary libraries as follows −

```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
```

Next, load the iris dataset as follows −

```
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1
```

We can plot our training data s follows −

```
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='y', label='1')
plt.legend();
```
![Iris Training Data](/machine_learning/images/logistic_regression_iris_training_data.jpg)
Next, we will define sigmoid function, loss function and gradient descend as follows −

```
class LogisticRegression:
   def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
      self.lr = lr
      self.num_iter = num_iter
      self.fit_intercept = fit_intercept
      self.verbose = verbose
   def __add_intercept(self, X):
      intercept = np.ones((X.shape[0], 1))
      return np.concatenate((intercept, X), axis=1)
   def __sigmoid(self, z):
      return 1 / (1 + np.exp(-z))
   def __loss(self, h, y):
      return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
   def fit(self, X, y):
      if self.fit_intercept:
         X = self.__add_intercept(X)
```

Now, initialize the weights as follows −

```
self.theta = np.zeros(X.shape[1])
   for i in range(self.num_iter):
      z = np.dot(X, self.theta)
      h = self.__sigmoid(z)
      gradient = np.dot(X.T, (h - y)) / y.size
      self.theta -= self.lr * gradient
      z = np.dot(X, self.theta)
      h = self.__sigmoid(z)
      loss = self.__loss(h, y)
      if(self.verbose ==True and i % 10000 == 0):
         print(f'loss: {loss} \t')
```

With the help of the following script, we can predict the output probabilities −

```
def predict_prob(self, X):
   if self.fit_intercept:
      X = self.__add_intercept(X)
   return self.__sigmoid(np.dot(X, self.theta))
def predict(self, X):
   return self.predict_prob(X).round()
```

Next, we can evaluate the model and plot it as follows −

```
model = LogisticRegression(lr=0.1, num_iter=300000)
preds = model.predict(X)
(preds == y).mean()

plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='y', label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='red');
```
![Model Evaluation](/machine_learning/images/binary_logistic_regression_classification.jpg)
## Multinomial Logistic Regression Model

Another useful form of logistic regression is multinomial logistic regression in which the target or dependent variable can have 3 or more possible unordered types i.e. the types having no quantitative significance.

### Implementation of Multinomial Logistic Regression Model in Python

Now we will implement the above concept of multinomial logistic regression in Python. For this purpose, we are using a dataset from sklearn named digit.

First, we need to import the necessary libraries as follows −

```
Import sklearn
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
```

Next, we need to load digit dataset −

```
digits = datasets.load_digits()
```

Now, define the feature matrix(X) and response vector(y)as follows −

```
X = digits.data
y = digits.target
```

With the help of next line of code, we can split X and y into training and testing sets −

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
```

Now create an object of logistic regression as follows −

```
digreg = linear_model.LogisticRegression()
```

Now, we need to train the model by using the training sets as follows −

```
digreg.fit(X_train, y_train)
```

Next, make the predictions on testing set as follows −

```
y_pred = digreg.predict(X_test)
```

Next print the accuracy of the model as follows −

```
print("Accuracy of Logistic Regression model is:",
metrics.accuracy_score(y_test, y_pred)*100)
```

### Output

```
Accuracy of Logistic Regression model is: 95.6884561891516
```

From the above output we can see the accuracy of our model is around 96 percent.

---

## 11. Support Vector Machine (SVM) in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_support_vector_machine.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_support_vector_machine.htm)*

---

---
[Previous](/machine_learning/machine_learning_decision_tree_algorithm.htm)[Quiz](/machine_learning/quiz_on_machine_learning_support_vector_machine.htm)[Next](/machine_learning/machine_learning_random_forest_classification.htm)
## What is Support Vector Machine (SVM)

Support vector machines (SVMs) are powerful yet flexible supervised machine learning algorithm which is used for both classification and regression. But generally, they are used in classification problems. In 1960s, SVMs were first introduced but later they got refined in 1990 also. SVMs have their unique way of implementation as compared to other machine learning algorithms. Now a days, they are extremely popular because of their ability to handle multiple continuous and categorical variables.

## Working of SVM

The goal of SVM is to find a hyperplane that separates the data points into different classes. A hyperplane is a line in 2D space, a plane in 3D space, or a higher-dimensional surface in n-dimensional space. The hyperplane is chosen in such a way that it maximizes the margin, which is the distance between the hyperplane and the closest data points of each class. The closest data points are called the support vectors.

The distance between the hyperplane and a data point "x" can be calculated using the formula −

```
distance = (w . x + b) / ||w||
```

where "w" is the weight vector, "b" is the bias term, and "||w||" is the Euclidean norm of the weight vector. The weight vector "w" is perpendicular to the hyperplane and determines its orientation, while the bias term "b" determines its position.

The optimal hyperplane is found by solving an optimization problem, which is to maximize the margin subject to the constraint that all data points are correctly classified. In other words, we want to find the hyperplane that maximizes the margin between the two classes while ensuring that no data point is misclassified. This is a convex optimization problem that can be solved using quadratic programming.

If the data points are not linearly separable, we can use a technique called kernel trick to map the data points into a higher-dimensional space where they become separable. The kernel function computes the inner product between the mapped data points without computing the mapping itself. This allows us to work with the data points in the higherdimensional space without incurring the computational cost of mapping them.

Let's understand it in detail with the help of following diagram −
![Working Of Svm](/machine_learning/images/working_of_svm.jpg)
Given below are the important concepts in SVM −

- **Support Vectors**− Datapoints that are closest to the hyperplane is called support vectors. Separating line will be defined with the help of these data points.
- **Hyperplane**− As we can see in the above diagram it is a decision plane or space which is divided between a set of objects having different classes.
- **Margin**− It may be defined as the gap between two lines on the closet data points of different classes. It can be calculated as the perpendicular distance from the line to the support vectors. Large margin is considered as a good margin and small margin is considered as a bad margin.
## Implementing SVM Using Python

For implementing SVM in Python we will start with the standard libraries import as follows −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
```

Next, we are creating a sample dataset, having linearly separable data, from sklearn.dataset.sample_generator for classification using SVM −

```
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.50)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer');
```

The following would be the output after generating sample dataset having 100 samples and 2 clusters −
![SVM Plotting blobs of datapoints](/machine_learning/images/svm_blobs_datapoints.jpg)
We know that SVM supports discriminative classification. it divides the classes from each other by simply finding a line in case of two dimensions or manifold in case of multiple dimensions. It is implemented on the above dataset as follows −

```
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
plt.plot([0.6], [2.1], 'x', color='black', markeredgewidth=4, markersize=12)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
   plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5);
```

The output is as follows −
![SVM plotting line/ hyperplane](/machine_learning/images/svm_line_hyperplane.jpg)
We can see from the above output that there are three different separators that perfectly discriminate the above samples.

As discussed, the main goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane (MMH) hence rather than drawing a zero line between classes we can draw around each line a margin of some width up to the nearest point. It can be done as follows −

```
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
  yfit = m * xfit + b
  plt.plot(xfit, yfit, '-k')
  plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
        color='#AAAAAA', alpha=0.4)
  plt.xlim(-1, 3.5);
```
![Plotting Maximum Marginal Hyperplane](/machine_learning/images/svm_maximum_marginal_hyperplane.jpg)
From the above image in output, we can easily observe the "margins" within the discriminative classifiers. SVM will choose the line that maximizes the margin.

Next, we will use Scikit-Learn's support vector classifier to train an SVM model on this data. Here, we are using linear kernel to fit SVM as follows −

```
from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)
```

The output is as follows −

```
SVC(C=10000000000.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
kernel='linear', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False)
```

Now, for a better understanding, the following will plot the decision functions for 2D SVC −

```
def decision_function(model, ax=None, plot_support=True):
   if ax is None:
      ax = plt.gca()
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()
```

For evaluating model, we need to create grid as follows −

```
x = np.linspace(xlim[0], xlim[1], 30)
y = np.linspace(ylim[0], ylim[1], 30)
Y, X = np.meshgrid(y, x)
xy = np.vstack([X.ravel(), Y.ravel()]).T
P = model.decision_function(xy).reshape(X.shape)
```

Next, we need to plot decision boundaries and margins as follows −

```
ax.contour(X, Y, P, colors='k',
   levels=[-1, 0, 1], alpha=0.5,
   linestyles=['--', '-', '--'])
```

Now, similarly plot the support vectors as follows −

```
if plot_support:
   ax.scatter(model.support_vectors_[:, 0],
      model.support_vectors_[:, 1],
      s=300, linewidth=1, facecolors='none');
ax.set_xlim(xlim)
ax.set_ylim(ylim)
```

Now, use this function to fit our models as follows −

```
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
decision_function(model);
```
![SVM Best Fit Hyperplane](/machine_learning/images/svm_best_fit_hyperplane.jpg)
We can observe from the above output that an SVM classifier fit to the data with margins i.e. dashed lines and support vectors, the pivotal elements of this fit, touching the dashed line. These support vector points are stored in the support_vectors_ attribute of the classifier as follows −

```
model.support_vectors_
```

The output is as follows −

```
array([[0.5323772 , 3.31338909],
   [2.11114739, 3.57660449],
   [1.46870582, 1.86947425]])
```

## SVM Kernels

In practice, SVM algorithm is implemented with kernel that transforms an input data space into the required form. SVM uses a technique called the kernel trick in which kernel takes a low dimensional input space and transforms it into a higher dimensional space. In simple words, kernel converts non-separable problems into separable problems by adding more dimensions to it. It makes SVM more powerful, flexible and accurate. The following are some of the types of kernels used by SVM −

### Linear Kernel

It can be used as a dot product between any two observations. The formula of linear kernel is as below −

k(x,x
) = sum(x*x)
From the above formula, we can see that the product between two vectors say  &  is the sum of the multiplication of each pair of input values.

### Polynomial Kernel

It is more generalized form of linear kernel and distinguish curved or nonlinear input space. Following is the formula for polynomial kernel −

K(x, xi) = 1 + sum(x * xi)^d

Here d is the degree of polynomial, which we need to specify manually in the learning algorithm.

### Radial Basis Function (RBF) Kernel

RBF kernel, mostly used in SVM classification, maps input space in indefinite dimensional space. Following formula explains it mathematically −

K(x,xi) = exp(-gamma * sum((x  xi^2))

Here, gamma ranges from 0 to 1. We need to manually specify it in the learning algorithm. A good default value of gamma is 0.1.

As we implemented SVM for linearly separable data, we can implement it in Python for the data that is not linearly separable. It can be done by using kernels.

### Example

The following is an example for creating an SVM classifier by using kernels. We will be using iris dataset from scikit-learn −

We will start by importing following packages −

```
import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
```

Now, we need to load the input data −

```
iris = datasets.load_iris()
```

From this dataset, we are taking first two features as follows −

```
X = iris.data[:, :2]
y = iris.target
```

Next, we will plot the SVM boundaries with original data as follows −

```
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
   np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]
```

Now, we need to provide the value of regularization parameter as follows −

```
C = 1.0
```

Next, SVM classifier object can be created as follows −

Svc_classifier = svm.SVC(kernel='linear', C=C).fit(X, y)

```
Z = svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')
```

### Output

```
Text(0.5, 1.0, 'Support Vector Classifier with linear kernel')
```
![Curve](/machine_learning/images/svm_curve.jpg)
For creating SVM classifier with
**rbf**kernel, we can change the kernel to**rbf**as follows −
```
Svc_classifier = svm.SVC(kernel='rbf', gamma ='auto',C=C).fit(X, y)
Z = svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with rbf kernel')
```

### Output

```
Text(0.5, 1.0, 'Support Vector Classifier with rbf kernel')
```
![Classifier](/machine_learning/images/svm_classifier.jpg)
We put the value of gamma to 'auto' but you can provide its value between 0 to 1 also.

## Tuning SVM Parameters

In practice, SVMs often require tuning of their parameters to achieve optimal performance. The most important parameters to tune are the kernel, the regularization parameter C, and the kernel-specific parameters.

The kernel parameter determines the type of kernel to use. The most common kernel types are linear, polynomial, radial basis function (RBF), and sigmoid. The linear kernel is used for linearly separable data, while the other kernels are used for non-linearly separable data.

The regularization parameter C controls the trade-off between maximizing the margin and minimizing the classification error. A higher value of C means that the classifier will try to minimize the classification error at the expense of a smaller margin, while a lower value of C means that the classifier will try to maximize the margin even if it means more misclassifications.

The kernel-specific parameters depend on the type of kernel being used. For example, the polynomial kernel has parameters for the degree of the polynomial and the coefficient of the polynomial, while the RBF kernel has a parameter for the width of the Gaussian function.

We can use cross-validation to tune the parameters of the SVM. Cross-validation involves splitting the data into several subsets and training the classifier on each subset while using the remaining subsets for testing. This allows us to evaluate the performance of the classifier on different subsets of the data and choose the best set of parameters.

### Example

```
from sklearn.model_selection import GridSearchCV
# define the parameter grid
param_grid = {
   'C': [0.1, 1, 10, 100],
   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
   'degree': [2, 3, 4],
   'coef0': [0.0, 0.1, 0.5],
   'gamma': ['scale', 'auto']
}

# create an SVM classifier
svm = SVC()

# perform grid search to find the best set of parameters
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)
# print the best set of parameters and their accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)
```

We start by importing the
**GridSearchCV**module from scikit-learn, which is a tool for performing grid search on a set of parameters. We define a parameter grid that contains the possible values for each parameter we want to tune.
We create an SVM classifier using
**SVC()**and then pass it to**GridSearchCV**along with the parameter grid and the number of cross-validation folds (cv=5). We then call**grid_search.fit(X_train, y_train)**to perform the grid search.
Once the grid search is complete, we print the best set of parameters and their accuracy using
**grid_search.best_params_**and**grid_search.best_score_,**respectively.
#### Output

On executing this program, you will get the following output −

```
Best parameters: {'C': 0.1, 'coef0': 0.5, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
Best accuracy: 0.975
```

This means that the best set of parameters found by the grid search are:
**C=0.1, coef0=0.5, degree=3, gamma=scale, and kernel=poly**. The accuracy achieved by this set of parameters on the training set is 97.5%.
You can now use these parameters to create a new SVM classifier and test its performance on the testing set.

## Pros and Cons of SVM Classifiers

### Pros of SVM classifiers

SVM classifiers offers great accuracy and work well with high dimensional space. SVM classifiers basically use a subset of training points hence in result uses very less memory.

### Cons of SVM classifiers

They have high training time hence in practice not suitable for large datasets. Another disadvantage is that SVM classifiers do not work well with overlapping classes.

---

## 12. Machine Learning - Decision Tree Algorithm

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_decision_tree_algorithm.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_decision_tree_algorithm.htm)*

---

---
[Previous](/machine_learning/machine_learning_naive_bayes_algorithms.htm)[Quiz](/machine_learning/quiz_on_machine_learning_decision_tree_algorithm.htm)[Next](/machine_learning/machine_learning_support_vector_machine.htm)
## Decision Tree Algorithm

The decision tree algorithm is a hierarchical tree-based algorithm that is used to classify or predict outcomes based on a set of rules. It works by splitting the data into subsets based on the values of the input features. The algorithm recursively splits the data until it reaches a point where the data in each subset belongs to the same class or has the same value for the target variable. The resulting tree is a set of decision rules that can be used to make predictions or classify new data.

The Decision Tree algorithm works by selecting the best feature to split the data at each node. The best feature is the one that provides the most information gain or the most reduction in entropy. Information gain is a measure of the amount of information gained by splitting the data at a particular feature, while entropy is a measure of the randomness or disorder in the data. The algorithm uses these measures to determine the best feature
to split the data at each node.

The example of a binary tree for predicting whether a person is fit or unfit providing various information like age, eating habits and exercise habits, is given below −
![Decision Tree Algorithm](/machine_learning/images/decision_tree_algorithm.jpg)
In the above decision tree, the question are decision nodes and final outcomes are leaves.

## Types of Decision Tree Algorithm

There are two main types of Decision Tree algorithm −

- **Classification Tree**− A classification tree is used to classify data into different classes or categories. It works by splitting the data into subsets based on the values of the input features and assigning each subset to a different class.
- **Regression Tree**− A regression tree is used to predict numerical values or continuous variables. It works by splitting the data into subsets based on the values of the input features and assigning each subset a numerical value.
## Implementing Decision Tree Algorithm

### Gini Index

It is the name of the cost function that is used to evaluate the binary splits in the dataset and works with the categorial target variable Success or Failure.

Higher the value of Gini index, higher the homogeneity. A perfect Gini index value is 0 and worst is 0.5 (for 2 class problem). Gini index for a split can be calculated with the help of following steps −

- 
First, calculate Gini index for sub-nodes by using the formula p^2+q^2 , which is the sum of the square of probability for success and failure.

- 
Next, calculate Gini index for split using weighted Gini score of each node of that split.

Classification and Regression Tree (CART) algorithm uses Gini method to generate binary splits.

### Split Creation

A split is basically including an attribute in the dataset and a value. We can create a split in dataset with the help of following three parts −

- **Part1: Calculating Gini Score**− We have just discussed this part in the previous section.
- **Part2: Splitting a dataset**− It may be defined as separating a dataset into two lists of rows having index of an attribute and a split value of that attribute. After getting the two groups - right and left, from the dataset, we can calculate the value of split by using Gini score calculated in first part. Split value will decide in which group the attribute will reside.
- **Part3: Evaluating all splits**− Next part after finding Gini score and splitting dataset is the evaluation of all splits. For this purpose, first, we must check every value associated with each attribute as a candidate split. Then we need to find the best possible split by evaluating the cost of the split. The best split will be used as a node in the decision tree.
## Building a Tree

As we know that a tree has root node and terminal nodes. After creating the root node, we can build the tree by following two parts −

### Part1: Terminal node creation

While creating terminal nodes of decision tree, one important point is to decide when to stop growing tree or creating further terminal nodes. It can be done by using two criteria namely maximum tree depth and minimum node records as follows −

- **Maximum Tree Depth**− As name suggests, this is the maximum number of the nodes in a tree after root node. We must stop adding terminal nodes once a tree reached at maximum depth i.e. once a tree got maximum number of terminal nodes.
- **Minimum Node Records**− It may be defined as the minimum number of training patterns that a given node is responsible for. We must stop adding terminal nodes once tree reached at these minimum node records or below this minimum.
Terminal node is used to make a final prediction.

### Part2: Recursive Splitting

As we understood about when to create terminal nodes, now we can start building our tree. Recursive splitting is a method to build the tree. In this method, once a node is created, we can create the child nodes (nodes added to an existing node) recursively on each group of data, generated by splitting the dataset, by calling the same function again and again.

### Prediction

After building a decision tree, we need to make a prediction about it. Basically, prediction involves navigating the decision tree with the specifically provided row of data.

We can make a prediction with the help of recursive function, as did above. The same prediction routine is called again with the left or the child right nodes.

### Assumptions

The following are some of the assumptions we make while creating decision tree −

- 
While preparing decision trees, the training set is as root node.

- 
Decision tree classifier prefers the features values to be categorical. In case if you want to use continuous values then they must be done discretized prior to model building.

- 
Based on the attributes values, the records are recursively distributed.

- 
Statistical approach will be used to place attributes at any node position i.e.as root node or internal node.

## Implementation in Python

Let's implement the Decision Tree algorithm in Python using a popular dataset for classification tasks named Iris dataset. It contains 150 samples of
**iris**flowers, each with four features: sepal length, sepal width, petal length, and petal width. The flowers belong to three classes: setosa, versicolor, and virginica.
First, we will import the necessary libraries and load the dataset −

```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data,
iris.target, test_size=0.3, random_state=0)
```

We then create an instance of the Decision Tree classifier and train it on the training set −

```
# Create a Decision Tree classifier
dtc = DecisionTreeClassifier()

# Fit the classifier to the training data
dtc.fit(X_train, y_train)
```

We can now use the trained classifier to make predictions on the testing set −

```
# Make predictions on the testing data
y_pred = dtc.predict(X_test)
```

We can evaluate the performance of the classifier by calculating its accuracy −

```
# Calculate the accuracy of the classifier
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)
```

We can visualize the Decision Tree using Matplotlib library −

```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualize the Decision Tree using Matplotlib
plt.figure(figsize=(20,10))
plot_tree(dtc, filled=True, feature_names=iris.feature_names,
class_names=iris.target_names)
plt.show()
```

The
**plot_tree**function from the**sklearn.tree**module can be used to plot the Decision Tree. We can pass in the trained Decision Tree classifier, the filled argument to fill the nodes with color, the**feature_names**argument to label the features, and the**class_names**argument to label the target classes. We also specify the**figsize**argument
to set the size of the figure and call the show function to display the plot.
### Complete Implementation Example

Given below is the complete implementation example of Decision Tree Classification algorithm in python using the iris dataset −

```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

# Create a Decision Tree classifier
dtc = DecisionTreeClassifier()

# Fit the classifier to the training data
dtc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dtc.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)

# Visualize the Decision Tree using Matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(dtc, filled=True, feature_names=iris.feature_names,
class_names=iris.target_names)
plt.show()
```

#### Output

This will create a plot of the Decision Tree that looks like this −
![Plot Of Decision Tree](/machine_learning/images/plot_of_decision_tree.jpg)
```
Accuracy: 0.9777777777777777
```

As you can see, the plot shows the structure of the Decision Tree, with each node representing a decision based on the value of a feature, and each leaf node representing a class or numerical value. The color of each node indicates the majority class or value of the samples in that node, and the numbers at the bottom indicate the number of samples that reach that node.

---

## 13. Nave Bayes Algorithm in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_naive_bayes_algorithms.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_naive_bayes_algorithms.htm)*

---

---
[Previous](/machine_learning/machine_learning_knn_nearest_neighbors.htm)[Quiz](/machine_learning/quiz_on_machine_learning_naive_bayes_algorithms.htm)[Next](/machine_learning/machine_learning_decision_tree_algorithm.htm)
## What is Nave Bayes Algorithm?

The Naive Bayes algorithm is a classification algorithm based on Bayes' theorem. The algorithm assumes that the features are independent of each other, which is why it is called "naive." It calculates the probability of a sample belonging to a particular class based on the probabilities of its features. For example, a phone may be considered as smart if it has touch-screen, internet facility, good camera, etc. Even if all these features are dependent on each other, but all these features independently contribute to the probability of that the phone is a smart phone.

In Bayesian classification, the main interest is to find the posterior probabilities i.e. the probability of a label given some observed features, P(L | features). With the help of Bayes theorem, we can express this in quantitative form as follows −

$$P\left ( L| features\right )=\frac{P\left ( L \right )P\left (features| L\right )}{P\left (features\right )}$$

Here,

- 
$P\left ( L| features\right )$ is the posterior probability of class.

- 
$P\left ( L \right )$ is the prior probability of class.

- 
$P\left (features| L\right )$ is the likelihood which is the probability of predictor given class.

- 
$P\left (features\right )$ is the prior probability of predictor.

In the Naive Bayes algorithm, we use Bayes' theorem to calculate the probability of a sample belonging to a particular class. We calculate the probability of each feature of the sample given the class and multiply them to get the likelihood of the sample belonging to the class. We then multiply the likelihood with the prior probability of the class to get the posterior probability of the sample belonging to the class. We repeat this process for each class and choose the class with the highest probability as the class of the sample.

## Types of Naive Bayes Algorithm

There are many types of Naive Bayes Algorithm. Here we discuss the following three types −

### Gaussian Nave Bayes

Gaussian Nave Bayes is the simplest Nave Bayes classifier having the assumption that the data from each label is drawn from a simple Gaussian distribution. It is used when the features are continuous variables that follow a normal distribution.

### Multinomial Nave Bayes

Another useful Nave Bayes classifier is Multinomial Nave Bayes in which the features are assumed to be drawn from a simple Multinomial distribution. Such kind of Nave Bayes are most appropriate for the features that represents discrete counts. It is commonly used in text classification tasks where the features are the frequency of words in a document.

### Bernoulli Nave Bayes

Another important model is Bernoulli Nave Bayes in which features are assumed to be binary (0s and 1s). Text classification with 'bag of words' model can be an application of Bernoulli Nave Bayes.

## Implementation of Nave Bayes Algorithm in Python

Depending on our data set, we can choose any of the Nave Bayes model explained above. Here, we are implementing Gaussian Nave Bayes model in Python −

We will start with required imports as follows −

```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
```

Now, by using make_blobs() function of Scikit learn, we can generate blobs of points with Gaussian distribution as follows −

```
from sklearn.datasets import make_blobs
X, y = make_blobs(300, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer');
```
![Blobs of Points with Gaussian Distribution](/machine_learning/images/blobs_points_gaussian_distribution.jpg)
Next, for using GaussianNB model, we need to import and make its object as follows −

```
from sklearn.naive_bayes import GaussianNB
model_GNB = GaussianNB()
model_GNB.fit(X, y);
```

Now, we have to do prediction. It can be done after generating some new data as follows −

```
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model_GNB.predict(Xnew)
```

Next, we are plotting new data to find its boundaries −

```
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='summer', alpha=0.1)
plt.axis(lim);
```
![Plotting the prediction with new data](/machine_learning/images/naive_bayes_model_prediction.jpg)
Now, with the help of following line of codes, we can find the posterior probabilities of first and second label −

```
yprob = model_GNB.predict_proba(Xnew)
yprob[-10:].round(3)
```

### Output

```
array([[0.998, 0.002],
   [1.   , 0.   ],
   [0.987, 0.013],
   [1.   , 0.   ],
   [1.   , 0.   ],
   [1.   , 0.   ],
   [1.   , 0.   ],
   [1.   , 0.   ],
   [0.   , 1.   ],
   [0.986, 0.014]]
)
```

## Pros & Cons of Nave Bayes classification

Let's discuss some of the advantages and limitations of Naive Bayes classification algorithm.

### Pros

The followings are some pros of using Nave Bayes classifiers −

- 
Nave Bayes classification is easy to implement and fast.

- 
It will converge faster than discriminative models like logistic regression.

- 
It requires less training data.

- 
It is highly scalable in nature, or they scale linearly with the number of predictors and data points.

- 
It can make probabilistic predictions and can handle continuous as well as discrete data.

- 
Nave Bayes classification algorithm can be used for binary as well as multi-class classification problems both.

### Cons

The followings are some cons of using Nave Bayes classifiers −

- 
One of the most important cons of Nave Bayes classification is its strong feature independence because in real life it is almost impossible to have a set of features which are completely independent of each other.

- 
Another issue with Nave Bayes classification is its 'zero frequency' which means that if a categorial variable has a category but not being observed in training data set, then Nave Bayes model will assign a zero probability to it and it will be unable to make a prediction.

## Applications of Nave Bayes classification

The following are some common applications of Nave Bayes classification −
**Real-time prediction**− Due to its ease of implementation and fast computation, it can be used to do prediction in real-time.**Multi-class prediction**− Nave Bayes classification algorithm can be used to predict posterior probability of multiple classes of target variable.**Text classification**− Due to the feature of multi-class prediction, Nave Bayes classification algorithms are well suited for text classification. That is why it is also used to solve problems like spam-filtering and sentiment analysis.**Recommendation system**− Along with the algorithms like collaborative filtering, Nave Bayes makes a Recommendation system which can be used to filter unseen information and to predict weather a user would like the given resource or not.

---

## 14. Random Forest Algorithm in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_random_forest_classification.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_random_forest_classification.htm)*

---

---
[Previous](/machine_learning/machine_learning_support_vector_machine.htm)[Quiz](/machine_learning/quiz_on_machine_learning_random_forest_classification.htm)[Next](/machine_learning/machine_learning_confusion_matrix.htm)
Random Forest is a machine learning algorithm that uses an ensemble of decision trees to make predictions. The algorithm was first introduced by Leo Breiman in 2001. The key idea behind the algorithm is to create a large number of decision trees, each of which is trained on a different subset of the data. The predictions of these individual trees are then combined to produce a final prediction.

## Working of Random Forest Algorithm

We can understand the working of Random Forest algorithm with the help of following steps −

- **Step 1**− First, start with the selection of random samples from a given dataset.
- **Step 2**− Next, this algorithm will construct a decision tree for every sample. Then it will get the prediction result from every decision tree.
- **Step 3**− In this step, voting will be performed for every predicted result.
- **Step 4**− At last, select the most voted prediction result as the final prediction result.
The following diagram illustrates how the Random Forest Algorithm works −
![Random Forest Algorithm](/machine_learning/images/random_forest_algorithm.jpg)
Random Forest is a flexible algorithm that can be used for both classification and regression tasks. In classification tasks, the algorithm uses the mode of the predictions of the individual trees to make the final prediction. In regression tasks, the algorithm uses the mean of the predictions of the individual trees.

## Advantages of Random Forest Algorithm

Random Forest algorithm has several advantages over other machine learning algorithms. Some of the key advantages are −

- **Robustness to Overfitting**− Random Forest algorithm is known for its robustness to overfitting. This is because the algorithm uses an ensemble of decision trees, which helps to reduce the impact of outliers and noise in the data.
- **High Accuracy**− Random Forest algorithm is known for its high accuracy. This is because the algorithm combines the predictions of multiple decision trees, which helps to reduce the impact of individual decision trees that may be biased or inaccurate.
- **Handles Missing Data**− Random Forest algorithm can handle missing data without the need for imputation. This is because the algorithm only considers the features that are available for each data point and does not require all features to be present for all data points.
- **Non-Linear Relationships**− Random Forest algorithm can handle non-linear relationships between the features and the target variable. This is because the algorithm uses decision trees, which can model non-linear relationships.
- **Feature Importance**− Random Forest algorithm can provide information about the importance of each feature in the model. This information can be used to identify the most important features in the data and can be used for feature selection and feature engineering.
## Implementation of Random Forest Algorithm in Python

Let's take a look at the implementation of Random Forest Algorithm in Python. We will be using the scikit-learn library to implement the algorithm. The scikit-learn library is a popular machine learning library that provides a wide range of algorithms and tools for machine learning.

### Step 1 − Importing the Libraries

We will begin by importing the necessary libraries. We will be using the pandas library for data manipulation, and the scikit-learn library for implementing the Random Forest algorithm.

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
```

### Step 2 − Loading the Data

Next, we will load the data into a pandas dataframe. For this tutorial, we will be using the famous Iris dataset, which is a classic dataset for classification tasks.

```
# Loading the iris dataset

iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learningdatabases/iris/iris.data', header=None)

iris.columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width', 'species']
```

### Step 3 − Data Preprocessing

Before we can use the data to train our model, we need to preprocess it. This involves separating the features and the target variable and splitting the data into training and testing sets.

```
# Separating the features and target variable
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
```

### Step 4 − Training the Model

Next, we will train our Random Forest classifier on the training data.

```
# Creating the Random Forest classifier object
rfc = RandomForestClassifier(n_estimators=100)

# Training the model on the training data
rfc.fit(X_train, y_train)
```

### Step 5 − Making Predictions

Once we have trained our model, we can use it to make predictions on the test data.

```
# Making predictions on the test data
y_pred = rfc.predict(X_test)
```

### Step 6 − Evaluating the Model

Finally, we will evaluate the performance of our model using various metrics such as accuracy, precision, recall, and F1-score.

```
# Importing the metrics library
from sklearn.metrics import accuracy_score, precision_score,
recall_score, f1_score

# Calculating the accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```

### Complete Implementation Example

Below is the complete implementation example of Random Forest Algorithm in python using the iris dataset −

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Loading the iris dataset
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learningdatabases/iris/iris.data', header=None)

iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Separating the features and target variable
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.35, random_state=42)

# Creating the Random Forest classifier object
rfc = RandomForestClassifier(n_estimators=100)

# Training the model on the training data
rfc.fit(X_train, y_train)
# Making predictions on the test data
y_pred = rfc.predict(X_test)
# Importing the metrics library
from sklearn.metrics import accuracy_score, precision_score,
recall_score, f1_score

# Calculating the accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```

#### Output

This will give us the performance metrics of our Random Forest classifier as follows −

```
Accuracy: 0.9811320754716981
Precision: 0.9821802935010483
Recall: 0.9811320754716981
F1-score: 0.9811157396063056
```

## Pros and Cons of Random Forest

### Pros

The following are the advantages of Random Forest algorithm −

- 
It overcomes the problem of overfitting by averaging or combining the results of different decision trees.

- 
Random forests work well for a large range of data items than a single decision tree does.

- 
Random forest has less variance then single decision tree.

- 
Random forests are very flexible and possess very high accuracy.

- 
Scaling of data does not require in random forest algorithm. It maintains good accuracy even after providing data without scaling.

- 
Scaling of data does not require in random forest algorithm. It maintains good accuracy even after providing data without scaling.

### Cons

The following are the disadvantages of Random Forest algorithm −

- 
Complexity is the main disadvantage of Random forest algorithms.

- 
Construction of Random forests are much harder and time-consuming than decision trees.

- 
More computational resources are required to implement Random Forest algorithm.

- 
It is less intuitive in case when we have a large collection of decision trees .

- 
The prediction process using random forests is very time-consuming in comparison with other algorithms.

---

## 15. Regression Analysis in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_regression_analysis.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_regression_analysis.htm)*

---

---
[Previous](/machine_learning/machine_learning_hypothesis.htm)[Quiz](/machine_learning/quiz_on_machine_learning_regression_analysis.htm)[Next](/machine_learning/machine_learning_linear_regression.htm)
## What is Regression Analysis?

In machine learning, regression analysis is a statistical technique that predicts continuous numeric values based on the relationship between independent and dependent variables. The main goal of regression analysis is to plot a line or curve that best fit the data and to estimate how one variable affects another.

Regression analysis is a fundamental concept in machine learning and it is used in many applications such as forecasting, predictive analytics, etc.

In machine learning, regression is a type of
[supervised learning](/machine_learning/machine_learning_supervised.htm). The key objective of regression-based tasks is to predict output labels or responses, which are continuous numeric values, for the given input data. The output will be based on what the model has learned in the training phase.
Regression models use the input data features (independent variables) and their corresponding continuous numeric output values (dependent or outcome variables) to learn specific associations between inputs and corresponding outputs.

## Terminologies Used In Regression Analysis

Let us understand some basic terminologies used in regression analysis before going into further detail. The following are some important terminologies −

- **Independent Variables**− These variables are used to predict the value of the dependent variable. These are also called predictors. In dataset, these are represented as**features**.
- **Dependent Variables**− These are the variables whose values we want to predict. These are the main factors in regression analysis. In dataset, these are represented as**target variables**
- **Regression line**− It is a straight line or curve that a regressor plots to fit the data points best.
- **Overfitting and underfitting**− Overfitting is when the regression model works well with the training dataset but not with the testing dataset. It's also referred to as the problem of high variance. Underfitting is when the model doesn't work well with training datasets. It's also referred to as the problem of high bias.
- **Outliers**− These are data points that don't fit the pattern of the rest of the data. They are the extremely high or extremely low values in the data set.
- **Multicollinearity**− multicollinearity occurs when independent variables (features) have dependency among them.
## How Does Regression Work?

Regression in machine learning is a supervised learning. Basically, regression is a statistical technique that finds a relationship between dependent and independent variables. To implement regression in machine learning, a regression algorithm is trained with a labeled dataset. The dataset contains features (independent variables) and target values (dependent variable).

During the training phase, the regression algorithm learns the relation between independent variables (predictors) and dependent variables (target).

The regression models predict new values based on the learned relation between predictors and targets during the training.

## Types of Regression in Machine Learning

Generally, the classification of regression methods is done based on the three metrics − the number of independent variables, type of dependent variables, and shape of the regression line.

There are numerous regression techniques used in machine learning. However, the following are commonly used types of regression −

- Linear Regression
- Logistic Regression
- Polynomial Regression
- Lasso Regression
- Ridge Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression
Let's discuss each type of regression in machine learning in detail.

### 1. Linear Regression
[Linear regression](/machine_learning/machine_learning_linear_regression.htm)is the most commonly used regression model in machine learning. It may be defined as the statistical model that analyzes the linear relationship between a dependent variable with a given set of independent variables. A linear relationship between variables means that when the value of one or more independent variables changes (increase or decrease), the value of the dependent variable will also change accordingly (increase or decrease).
Linear regression is further divided into two subcategories: simple linear regression and multiple linear regression (also known as multivariate linear regression).

In simple linear regression, a single independent variable (or predictor) is used to predict the dependent variable.

Mathematically, the simple linear regression can be represented as follows −

$$Y=mX+b$$

Where,

- ${Y}$ is the dependent variable we are trying to predict.
- ${X}$ is the dependent variable we are using to make predictions.
- ${m}$ is the slope of the regression line, which represents the effect ${X}$ has on ${Y}$.
- ${b}$ is a constant known as the Y-intercept. If ${X = 0}$, ${Y}$ would be equal to ${b}$.
In multi-linear regression, multiple independent variables are used to predict the dependent variables.

We will learn linear regression in more detail in upcoming chapters.

### 2. Logistic Regression
[Logistic regression](/machine_learning/machine_learning_logistic_regression.htm)is a popular machine learning algorithm used for predicting the probability of an event occurring.
Logistic regression is a generalized linear model where the target variable follows a Bernoulli distribution. Logistic regression uses a logistic function or logit function to learn a relationship between the independent variables (predictors) and dependent variables (target).

It maps the dependent variable as a sigmoid function of independent variables. The sigmoid function produces a probability between 0 and 1. The probability value is used to estimate the dependent variable's value.

It is mostly used in binary classification problems, where the target variable is categorical with two classes. It models the probability of the target variable given the input features and predicts the class with the highest probability.

### 3. Polynomial Regression
[Polynomial Linear Regression](/machine_learning/machine_learning_polynomial_regression.htm)is a type of regression analysis in which the relationship between the independent variable and the dependent variable is modeled as an n-th degree polynomial function. Polynomial regression allows for a more complex relationship between the variables to be captured, beyond the linear relationship in Simple and Multiple Linear Regression.
Polynomial regression is one of the most widely used non-linear regressions. It is very useful because it can model non-linear relationships between predictors and targets, and also it is more sensitive to outliers.

### 4. Lasso Regression

Lasso regression is a regularization technique that uses a penalty to prevent overfitting and improve the accuracy of regression models. It performs
[L1 regularization](/machine_learning/machine_learning_regularization.htm#l1_regularization). It modifies the loss function by adding the penalty (shrinkage quantity) equivalent to the summation of the absolute value of coefficients.
Lasso regression is often used to handle high dimensional and high correlation data.

### 5. Ridge Regression

Ridge regression is a statistical technique used in machine learning to prevent overfitting in linear regression models. It is used as a regularization technique that performs
[L2 regularization](/machine_learning/machine_learning_regularization.htm#l2_regularization). It modifies the loss or cost function by adding the penalty (shrinkage quantity) equivalent to the square of the magnitude of coefficients.
Ridge regression helps to reduce model complexity and improve prediction accuracy. It is useful in developing many parameters with high weights. It is also well suited to datasets with more feature variables than a number of observations.

It also corrects the multicollinearity in regression analysis. Multicollinearity occurs when independent variables are dependent on each other.

### 6. Decision Tree Regression

Decision tree regression uses the
[decision tree algorithm](/machine_learning/machine_learning_decision_tree_algorithm.htm)to predict numerical values. The decision tree algorithm is a supervised machine learning algorithm that can be used for both classification and regression.
It is used to predict numerical values or continuous variables. It works by splitting the data into smaller subsets based on the values of the input features and assigning each subset a numerical value. So incrementally, it develops a decision tree

The tree fits local linear regressions that approximate a curve, and each leaf represents a numeric value. The algorithm tries to reduce the mean square error at each child node, which measures how much the predictions deviate from the original target.

The decision tree regression can be used in predicting stock prices or customer behavior etc.

### 7. Random Forest Regression

Random forest regression is a supervised machine learning algorithm that uses an ensemble of decision trees to predict continuous target variables. It uses a bagging technique that involves randomly selecting subsets of training data to build smaller decision trees. These smaller models are combined to form a random forest model that outputs a single prediction value.

The technique helps improve accuracy and reduce variance by combining the predictions from multiple decision trees.

### 8. Support Vector Regression

Support vector regression (SVR) is a machine learning algorithm that uses
[support vector machine](/machine_learning/machine_learning_support_vector_machine.htm)to solve regression problems. It can learn non-linear relationships between the input data (feature variables) and output data (target values).
Support vector regression has many advantages. It can handle linear as well as non-linear relationships in datasets. It is resistant to outliers. It has high prediction accuracy.

## Types of Regression Models

Regression models are of following two types −
**Simple regression model**− This is the most basic regression model in which predictions are formed from a single, univariate feature of the data.**Multiple regression model**− As the name implies, in this regression model, the predictions are formed from multiple features of the data.![Types of Regression Models](/machine_learning/images/types_of_regression_models.jpg)
## How to Select Best Regression Model?

You can consider factors like performance metrics, model complexity, interpretability, etc., to select the best regression model. Evaluate the model performance using metrics such as Mean Squared Error (MSE), Mean absolute error (MAE), R-squared, etc. Compare the performance of different models, such as linear regression, decision trees, random forests, etc., and choose a model that has the highest performance metrics, the lowest complexity, and the best interpretability.

## Evaluation Metrics for Regression

Common
[evaluation/ performance metrics](/machine_learning/machine_learning_performance_metrics.htm)for regression models −
- **Mean Absolute error (MAE)**− It is the average of the absolute difference between predicted values and true values.
- **Mean Squared error (MSE)**− It is the average of the square of the difference between actual and estimated values.
- **Median Absolute error**− It is the median value of the absolute difference between predicted values and true values.
- **Root mean square error (RMSE)**− It is the square root value of the mean squared error (MSE).
- **R**− the best possible score is 1.0, and it can be negative (because the model can be arbitrarily worse).(coefficient of determination) Score
- **Mean absolute percentage error(MAPE)**− It is the percentage equivalent of mean absolute error (MAE).
## Applications of Regression in Machine Learning

The applications of ML regression algorithms are as follows −
**Forecasting or Predictive analysis**− One of the important uses of regression is forecasting or predictive analysis. For example, we can forecast GDP, oil prices, or, in simple words, the quantitative data that changes with the passage of time.**Optimization**− We can optimize business processes with the help of regression. For example, a store manager can create a statistical model to understand the peak time of coming customers.**Error correction**− In business, making correct decisions is equally important as optimizing the business process. Regression can help us to make correct decision as well as correct the already implemented decision.**Economics**− It is the most used tool in economics. We can use regression to predict supply, demand, consumption, inventory investment, etc.**Finance**− A financial company is always interested in minimizing the risk portfolio and wants to know the factors that affect the customers. All these can be predicted with the help of a regression model.
## Building a Regressor in Python

Regressor model can be constructed from scratch in Python. Scikit-learn, a Python library for machine learning, can also be used to build a regressor in Python.

In the following example, we will be building a basic regression model that will fit a line to the data, i.e., linear regressor. The necessary steps for building a regressor in Python are as follows −

### Step 1: Importing necessary python package

For building a regressor using scikit-learn, we need to import it along with other necessary packages. We can import the by using following script −

```
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
```

### Step 2: Importing dataset

After importing necessary package, we need a dataset to build regression prediction model. We can import it from sklearn dataset or can use other one as per our requirement. We are going to use our saved input data. We can import it with the help of following script −

```
input = r'C:\linear.txt'
```

Next, we need to load this data. We are using
*np.loadtxt*function to load it.
```
input_data = np.loadtxt(input, delimiter=',')
X, y = input_data[:, :-1], input_data[:, -1]
```

### Step 3: Organizing data into training & testing sets

As we need to test our model on unseen data hence, we will divide our dataset into two parts: a training set and a test set. The following command will perform it −

```
training_samples = int(0.6 * len(X))
testing_samples = len(X) - num_training
X_train, y_train = X[:training_samples], y[:training_samples]
X_test, y_test = X[training_samples:], y[training_samples:]
```

### Step 4: Model evaluation & prediction

After dividing the data into training and testing we need to build the model. We will be using LineaRegression() function of Scikit-learn for this purpose. Following command will create a linear regressor object.

```
reg_linear = linear_model.LinearRegression()
```

Next, train this model with the training samples as follows −

```
reg_linear.fit(X_train, y_train)
```

Now, at last we need to do the prediction with the testing data.

```
y_test_pred = reg_linear.predict(X_test)
```

### Step 5: Plot & visualization

After prediction, we can plot and visualize it with the help of following script −

```
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_test_pred, color = 'black', linewidth = 2)
plt.xticks(())
plt.yticks(())
plt.show()
```

#### Output
![Plot Visualization](/machine_learning/images/plot_visualization.jpg)
In the above output, we can see the regression line between the data points.

### Step 6: Performance computation

We can also compute the performance of our regression model with the help of various performance metrics as follows.

```
print("Regressor model performance:")
print("Mean absolute error(MAE) =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error(MSE) =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
```

#### Output

```
Regressor model performance:
Mean absolute error(MAE) = 1.78
Mean squared error(MSE) = 3.89
Median absolute error = 2.01
Explain variance score = -0.09
R2 score = -0.09
```

---

## 16. Linear Regression in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_linear_regression.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_linear_regression.htm)*

---

---
[Previous](/machine_learning/machine_learning_regression_analysis.htm)[Quiz](/machine_learning/quiz_on_machine_learning_linear_regression.htm)[Next](/machine_learning/machine_learning_simple_linear_regression.htm)
Linear regression in machine learning is defined as a statistical model that analyzes the linear relationship between a dependent variable and a given set of independent variables. The linear relationship between variables means that when the value of one or more independent variables will change (increase or decrease), the value of the dependent variable will also change accordingly (increase or decrease).

In machine learning, linear regression is used for predicting continuous numeric values based on learned linear relation for new and unseen data. It is used in predictive modeling, financial forecasting, risk assessment, etc.

In this chapter, we will discuss the following topics in detail −

- [What is Linear Regression?](#what_is_linear_regression)
- [Types of Linear Regression](#types_of_linear_regression)
- [How Does Linear Regression Work?](#how_does_linear_regression_work)
- [Hypothesis Function For Linear Regression](#hypothesis_function_for_linear_regression)
- [Finding the Best Fit Line](#finding_the_best_fit_line)
- [Loss Function For Linear Regression](#loss_function_for_linear_regression)
- [Gradient Descent for Optimization](#gradient_descent_for_optimization)
- [Assumptions of Linear Regression](#assumptions_of_linear_regression)
- [Evaluation Metrics for Linear Regression](#evaluation_metrics_for_linear_regression)
- [Applications of Linear Regression](#applications_of_linear_regression)
- [Advantages of Linear Regression](#advantages_of_linear_regression)
- [Common Challenges with Linear Regression](#common_challenges_with_linear_regression)
## What is Linear Regression?

Linear regression is a statistical technique that estimates the linear relationship between a dependent and one or more independent variables. In machine learning, linear regression is implemented as a
[supervised learning](/machine_learning/machine_learning_supervised.htm)approach. In machine learning, labeled datasets contain input data (features) and output labels (target values). For linear regression in machine learning, we represent features as independent variables and target values as the dependent variable.
For the simplicity, take the following data (Single feature and single target)
Square Feet (X)House Price (Y)1300240150032017003301830295155025623504091450319
In the above data, the target House Price is the dependent variable represented by X, and the feature, Square Feet, is the independent variable represented by Y. The input features (X) are used to predict the target label (Y). So, the independent variables are also known as predictor variables, and the dependent variable is known as the response variable.

So lets define linear regression in machine learning as follows:

In machine learning, linear regression uses a linear equation to model the relationship between a dependent variable (Y) and one or more independent variables (Y).

The main goal of the linear regression model is to find the best-fitting straight line (often called a regression line) through a set of data points.

### Line of Regression

A straight line that shows a relation between the dependent variable and independent variables is known as the line of regression or regression line.
![ML Regression Line](/machine_learning/images/regression_line_linear_regression.jpg)
Furthermore, the linear relationship can be positive or negative in nature as explained below −

#### 1. Positive Linear Relationship

A linear relationship will be called positive if both independent and dependent variable increases. It can be understood with the help of the following graph −
![Positive Linear Relationship](/machine_learning/images/positive_linear_relationship.jpg)
#### 2. Negative Linear Relationship

A linear relationship will be called positive if the independent increases and the dependent variable decreases. It can be understood with the help of the following graph −
![Negative Linear Relationship](/machine_learning/images/negative_linear_relationship.jpg)
Linear regression is of two types, "simple linear regression" and "multiple linear regression", which we are going to discuss in the next two chapters of this tutorial.

## Types of Linear Regression

Linear regression is of the following two types −

- Simple Linear Regression
- Multiple Linear Regression
### 1. Simple Linear Regression
[Simple linear regression](/machine_learning/machine_learning_simple_linear_regression.htm)is a type of regression analysis in which a single independent variable (also known as a predictor variable) is used to predict the dependent variable. In other words, it models the linear relationship between the dependent variable and a single independent variable.![ML Simple Linear Regression](/machine_learning/images/simple_regression_regression.jpg)
In the above image, the straight line represents the simple linear regression line where &Ycirc; is the predicted value, and X is the input value.

Mathematically, the relationship can be modeled as a linear equation −

$$\mathrm{ Y = w_0 + w_1 X + \epsilon }$$

Where

- Y  is the dependent variable (target).
- X  is the independent variable (feature).
- wis the y-intercept of the line.
- wis the slope of the line, representing the effect of X on Y.
- ε is the error term, capturing the variability in Y not explained by X.
### 2. Multiple Linear Regression
[Multiple linear regression](/machine_learning/machine_learning_multiple_linear_regression.htm)is basically the extension of simple linear regression that predicts a response using two or more features.
When dealing with more than one independent variable, we extend simple linear regression to multiple linear regression. The model is expressed as:

Multiple linear regression extends the concept of simple linear regression to multiple independent variables. The model is expressed as:

$$\mathrm{Y = w_0 + w_1 X_1 + w_2 X_2 + \dots + w_p X_p + \epsilon}$$

Where

- X, X, ..., Xare the independent variables (features).
- w, w, ..., ware the coefficients for these variables.
- ε is the error term.
## How Does Linear Regression Work?

The main goal of linear regression is to find the best-fit line through a set of data points that minimizes the difference between the actual values and predicted values. So it is done? This is done by estimating the parameters w
, wetc.
The working of linear regression in machine learning can be broken down into many steps as follows −

- **Hypothesis**− We assume that there is a linear relation between input and output.
- **Cost Function**− Define a loss or cost function. The cost function quantifies the model's prediction error. The cost function takes the model's predicted values and actual values and returns a single scaler value that represents the cost of the model's prediction.
- **Optimization**− Optimize (minimize) the model's cost function by updating the model's parameters.
It continues updating the model's parameters until the cost or error of the model's prediction is optimized (minimized).

Let's discuss the above three steps in more detail −

## Hypothesis Function For Linear Regression

In linear regression problems, we assume that there is a linear relationship between input features (X) and predicted value (&Ycirc;).

The
[hypothesis function](/machine_learning/machine_learning_hypothesis.htm)returns the predicted value for a given input value. Generally we represent a hypothesis by h(X) and it is equal to &Ycirc;.
Hypothesis function for simple linear regression −

$$\mathrm{\hat{Y} = w_0 + w_1 X}$$

Hypothesis function for multiple linear regression −

$$\mathrm{\hat{Y} = w_0 + w_1 X_1 + w_2 X_2 + \dots + w_p X_p}$$

For different values of parameters (weights), we can find many regression lines. The main goal is to find the best-fit lines. Let's discuss it as below −

## Finding the Best Fit Line

We discussed above that different set of parameters will provide different regression lines. However, each regression line does not represent the optimal relation between the input and output values. The main goal is to find the best-fit line.

A regression line is said to be the best fit if the error between actual and predicted values is minimal.

Below image shows a regression line with error (ε) at input data point X. The error is calculated for all data points and our goal is to minimize the average error/ loss. We can use different types of loss functions such as mean square error (MSE), mean average error (MAE), L
loss, LLoss, etc.![ML Best Fit Line Representation](/machine_learning/images/linear_regression_best_fit_line.jpg)
So, how can we minimize the error between the actual and predicted values? Let's discuss the important concept, which is cost function or loss function.

## Loss Function for Linear Regression

The error between actual and predicted values can be quantified using a loss function of the cost function. The
[cost function](/machine_learning/machine_learning_cost_function.htm)takes the model's predicted values and actual values and returns a single scaler value that represents the cost of the model's prediction. Our main goal is to minimize the cost function.
The most commonly used cost function is the mean squared error function.

$$\mathrm{J(w_0, w_1) = \frac{1}{2n} \sum_{i=1}^{n} \left( Y_i - \hat{Y}_i \right)^2}$$

Where,

- n is the number of data points.
- Yis the observed value for the i-th data point.
- \( \hat{Y}_i = w_0 + w_1 X_i \) is the predicted value for the i-th data point.
## Gradient Descent for Optimization

Now we have defined our loss function. The next step is to minimize it and find the optimized values of the parameters or weights. This process of finding optimal values of parameters such that the loss or error is minimal is known as model optimization.
[Gradient Descent](/machine_learning/machine_learning_stochastic_gradient_descent.htm)is one of the most used optimization techniques for linear regression.
To find the optimal values of parameters, gradient descent is often used, especially in cases with large datasets. Gradient descent iteratively adjusts the parameters in the direction of the steepest descent of the cost function.

The parameter updates are given by

$$\mathrm{w_0 = w_0 - \alpha \frac{\partial J}{\partial w_0}}$$

$$\mathrm{w_1 = w_1 - \alpha \frac{\partial J}{\partial w_1}}$$

Where α is the learning rate, and the partial derivatives are:

$$\mathrm{\frac{\partial J}{\partial w_0} = -\frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \hat{Y}_i \right)}$$

$$\mathrm{\frac{\partial J}{\partial w_1} = -\frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \hat{Y}_i \right) X_i}$$

These gradients are used to update the parameters until convergence is reached (i.e., when the changes in \( w_0 \) and \( w_1 \) become negligible).

## Assumptions of Linear Regression

The following are some assumptions about the dataset that are made by the Linear Regression model −
**Multi-collinearity**− Linear regression model assumes that there is very little or no multi-collinearity in the data. Basically, multi-collinearity occurs when the independent variables or features have a dependency on them.**Auto-correlation**− Another assumption the Linear regression model assumes is that there is very little or no auto-correlation in the data. Basically, auto-correlation occurs when there is dependency between residual errors.**Relationship between variables**− Linear regression model assumes that the relationship between response and feature variables must be linear.
Violations of these assumptions can lead to biased or inefficient estimates. It is essential to validate these assumptions to ensure model accuracy.

## Evaluation Metrics for Linear Regression

To assess the performance of a linear regression model, several evaluation metrics are used −
**R-squared (R**− It measures the proportion of the variance in the dependent variable that is predictable from the independent variables.)
$$\mathrm{ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} }$$
**Mean Squared Error (MSE)**− It measures an average of the sum of the squared difference between the predicted values and the actual values.
$$\mathrm{ \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 }$$
**Root Mean Squared Error (RMSE)**− It measures the square root of the MSE.
$$\mathrm{ \text{RMSE} = \sqrt{\text{MSE}} }$$
**Mean Absolute Error (MAE)**− It measures the average of the sum of the absolute values of the difference between the predicted values and the actual values.
$$\mathrm{ \text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| }$$

## Applications of Linear Regression

### 1. Predictive Modeling

Linear regression is widely used for predictive modeling. For instance, in real estate, predicting house prices based on features such as size, location, and number of bedrooms can help buyers, sellers, and real estate agents make informed decisions.

### 2. Feature Selection

In multiple linear regression, analyzing the coefficients can help in feature selection. Features with small or zero coefficients might be considered less important and can be dropped to simplify the model.

### 3. Financial Forecasting

In finance, linear regression models predict stock prices, economic indicators, and market trends. Accurate forecasts can guide investment strategies and financial planning.

### 4. Risk Management

Linear regression helps in risk assessment by modeling the relationship between risk factors and financial metrics. For example, in insurance, it can model the relationship between policyholder characteristics and claim amounts.

## Advantages of Linear Regression

- **Interpretability**− Linear regression is easy to understand, which is useful when explaining how a model makes decisions.
- **Speed**− Linear regression is faster to train than many other machine learning algorithms.
- **Predictive analytics**− Linear regression is a fundamental building block for predictive analytics.
- **Linear relationships**− Linear regression is a powerful statistical method for finding linear relationships between variables.
- **Simplicity**− Linear regression is simple to implement and interpret.
- **Efficiency**− Linear regression is efficient to compute.
## Common Challenges with Linear Regression

### 1. Overfitting

Overfitting occurs when the regression model performs well on training data but lacks generalization on test data. Overfitting leads to poor prediction on new, unseen data.

### 2. Multicollinearity

When the dependent variables (predictor or feature variables) correlate, the situation is known as mutilcolinearty. In this, the estimates of the parameters (coefficients) can be unstable.

### 3. Outliers and Their Impact

Outliers can cause the regression line to be a poor fit for the majority of data points.

## Polynomial Regression: An Alternate to Linear Regression
[Polynomial Linear Regression](/machine_learning/machine_learning_polynomial_regression.htm)is a type of regression analysis in which the relationship between the independent variable and the dependent variable is modeled as an n-th degree polynomial function. Polynomial regression allows for a more complex relationship between the variables to be captured beyond the linear relationship in Simple and Multiple Linear Regression.

---

## 17. Clustering Algorithms in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_clustering_algorithms.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_clustering_algorithms.htm)*

---

---
[Previous](/machine_learning/machine_learning_stochastic_gradient_descent.htm)[Quiz](/machine_learning/quiz_on_machine_learning_clustering_algorithms.htm)[Next](/machine_learning/machine_learning_centroid_based_clustering.htm)
Clustering Algorithms are one of the most useful
[unsupervised machine learning](/machine_learning/machine_learning_unsupervised.htm)methods. These methods are used to find similarity as well as the relationship patterns among data samples and then cluster those samples into groups having similarity based on features.
Clustering is important because it determines the intrinsic grouping among the present unlabeled data. They basically make some assumptions about data points to constitute their similarity. Each assumption will construct different but equally valid clusters.

For example, below is the diagram which shows clustering system grouped together the similar kind of data in different clusters −
![clustering system grouped](/machine_learning/images/clustering.jpg)
## Cluster Formation Methods

It is not necessary that clusters will be formed in spherical form. Followings are some other cluster formation methods −

### Density-based

In these methods, the clusters are formed as the dense region. The advantage of these methods is that they have good accuracy as well as good ability to merge two clusters. Ex. Density-Based Spatial Clustering of Applications with Noise (DBSCAN), Ordering Points to identify Clustering structure (OPTICS) etc.

### Hierarchical-based

In these methods, the clusters are formed as a tree type structure based on the hierarchy. They have two categories namely, Agglomerative (Bottom up approach) and Divisive (Top down approach). Ex. Clustering using Representatives (CURE), Balanced iterative Reducing Clustering using Hierarchies (BIRCH) etc.

### Partitioning

In these methods, the clusters are formed by portioning the objects into k clusters. Number of clusters will be equal to the number of partitions. Ex. K-means, Clustering Large Applications based upon randomized Search (CLARANS).

### Grid

In these methods, the clusters are formed as a grid like structure. The advantage of these methods is that all the clustering operation done on these grids are fast and independent of the number of data objects. Ex. Statistical Information Grid (STING), Clustering in Quest (CLIQUE).

## Clustering Algorithms in Machine Learning

The following are the most important and useful machine learning clustering algorithms −

- [K-Means Clustering](#k_means_clustering)
- [K-Medoids Clustering](#k_medoids_clustering)
- [Mean-Shift Clustering](#mean_shift_clustering)
- [DBSCAN Clustering](#dbscan_clustering)
- [OPTICS Clustering](#optics_clustering)
- [HDBSCAN Clustering](#hdbscan_clustering)
- [BIRCH algorithm](#birch_algorithm)
- [Affinity Propagation Clustering](#affinity_propagation_clustering)
- [Agglomerative Clustering](#agglomerative_clustering)
- [Gaussian Mixture Model](#gaussian_mixture_model)
### K-Means Clustering

The
[K-Means clustering](/machine_learning/machine_learning_k_means_clustering.htm)algorithm computes the centroids and iterates until we it finds optimal centroid. It assumes that the number of clusters are already known. It is also called flat clustering algorithm. The number of clusters identified from data by algorithm is represented by 'K' in K-means.
### K-Medoids Clustering

The
[K-Methoids Clustering](/machine_learning/machine_learning_k_medoids_clustering.htm)is an improved version of K-means clustering algorithm. Working is as follows
- Select k random data points from the dataset as the initial medoids.
- Assign each data point to the nearest medoid.
- For each cluster, select the data point that minimizes the sum of distances to all the other data points in the cluster, and set it as the new medoid.
- Repeat steps 2 and 3 until convergence or a maximum number of iterations is reached.
### Mean-Shift Clustering
[Mean-Shift Clustering](/machine_learning/machine_learning_mean_shift_clustering.htm)It is another powerful clustering algorithm used in unsupervised learning. Unlike K-means clustering, it does not make any assumptions hence it is a non-parametric algorithm.
### DBSCAN Clustering

The
[DBSCAN](/machine_learning/machine_learning_dbscan_clustering.htm)(Density-Based Spatial Clustering of Applications with Noise) algorithm is one of the most common density-based clustering algorithms. The DBSCAN algorithm requires two parameters: the minimum number of neighbors (minPts) and the maximum distance between core data points (eps).
### OPTICS Clustering
[OPTICS](/machine_learning/machine_learning_optics_clustering.htm)(Ordering Points to Identify the Clustering Structure) is like DBSCAN, another popular density-based clustering algorithm. However, OPTICS has several advantages over DBSCAN, including the ability to identify clusters of varying densities, the ability to handle noise, and the ability to produce a hierarchical clustering structure.
### HDBSCAN Clustering
[HDBSCAN](/machine_learning/machine_learning_hdbscan_clustering.htm)(Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that is based on density clustering. It is a newer algorithm that builds upon the popular DBSCAN algorithm and offers several advantages over it, such as better handling of clusters of varying densities and the ability to detect clusters of different shapes and sizes.
### BIRCH algorithm
[BIRCH](/machine_learning/machine_learning_birch_clustering.htm)(Balanced Iterative Reducing and Clustering hierarchies) is a hierarchical clustering algorithm that is designed to handle large datasets efficiently. The algorithm builds a treelike structure of clusters by recursively partitioning the data into subclusters until a stopping criterion is met.
### Affinity Propagation Clustering
[Affinity Propagation](/machine_learning/machine_learning_affinity_propagation.htm)is a clustering algorithm that identifies "exemplars" in a dataset and assigns each data point to one of these exemplars. It is a type of clustering algorithm that does not require a pre-specified number of clusters, making it a useful tool for exploratory data analysis. Affinity Propagation was introduced by Frey and Dueck in 2007 and has since been widely used in many fields such as biology, computer vision, and social network analysis.
### Agglomerative Clustering
[Agglomerative clustering](/machine_learning/machine_learning_agglomerative_clustering.htm)is a hierarchical clustering algorithm that starts with each data point as its own cluster and iteratively merges the closest clusters until a stopping criterion is reached. It is a bottom-up approach that produces a dendrogram, which is a tree-like diagram that shows the hierarchical relationship between the clusters. The algorithm can be implemented using the scikit-learn library in Python.
### Gaussian Mixture Model
[Gaussian Mixture Models](/machine_learning/machine_learning_distribution_based_clustering.htm)(GMM) is a popular clustering algorithm used in machine learning that assumes that the data is generated from a mixture of Gaussian distributions. In other words, GMM tries to fit a set of Gaussian distributions to the data, where each Gaussian distribution represents a cluster in the data.
## Measuring Clustering Performance

One of the most important consideration regarding ML model is assessing its performance or you can say model's quality. In case of supervised learning algorithms, assessing the quality of our model is easy because we already have labels for every example.

On the other hand, in case of unsupervised learning algorithms we are not that much blessed because we deal with unlabeled data. But still we have some metrics that give the practitioner an insight about the happening of change in clusters depending on algorithm.

Before we deep dive into such metrics, we must understand that these metrics only evaluates the comparative performance of models against each other rather than measuring the validity of the model's prediction. Followings are some of the metrics that we can deploy on clustering algorithms to measure the quality of model −
1. Silhouette Analysis
2. Davis-Bouldin Index
3. Dunn Index
### 1. Silhouette Analysis

Silhouette analysis used to check the quality of clustering model by measuring the distance between the clusters. It basically provides us a way to assess the parameters like number of clusters with the help of
**Silhouette score**. This score measures how close each point in one cluster is to points in the neighboring clusters.
#### Analysis of Silhouette Score

The range of Silhouette score is [-1, 1]. Its analysis is as follows −

- **+1 Score**− Near +1**Silhouette score**indicates that the sample is far away from its neighboring cluster.
- **0 Score**− 0**Silhouette score**indicates that the sample is on or very close to the decision boundary separating two neighboring clusters.
- **-1 Score**&minusl -1**Silhouette score**indicates that the samples have been assigned to the wrong clusters.
The calculation of Silhouette score can be done by using the following formula −

=()/ (,)

Here,  = mean distance to the points in the nearest cluster

And,  = mean intra-cluster distance to all the points.

### 2. Davis-Bouldin Index

DB index is another good metric to perform the analysis of clustering algorithms. With the help of DB index, we can understand the following points about clustering model −

- 
Weather the clusters are well-spaced from each other or not?

- 
How much dense the clusters are?

We can calculate DB index with the help of following formula −
$$DB=\frac{1}{n}\displaystyle\sum\limits_{i=1}^n max_{j\neq{i}}\left(\frac{\sigma_{i}+\sigma_{j}}{d(c_{i},c_{j})}\right)$$
Here,  = number of clusters

σ
= average distance of all points in cluster  from the cluster centroid .
Less the DB index, better the clustering model is.

### 3. Dunn Index

It works same as DB index but there are following points in which both differs −

- 
The Dunn index considers only the worst case i.e. the clusters that are close together while DB index considers dispersion and separation of all the clusters in clustering model.

- 
Dunn index increases as the performance increases while DB index gets better when clusters are well-spaced and dense.

We can calculate Dunn index with the help of following formula −
$$D=\frac{min_{1\leq i <{j}\leq{n}}P(i,j)}{mix_{1\leq i < k \leq n}q(k)}$$
Here, ,, = each indices for clusters

= inter-cluster distance

q = intra-cluster distance

## Applications of Clustering

We can find clustering useful in the following areas −
**Data summarization and compression**− Clustering is widely used in the areas where we require data summarization, compression and reduction as well. The examples are image processing and vector quantization.**Collaborative systems and customer segmentation**− Since clustering can be used to find similar products or same kind of users, it can be used in the area of collaborative systems and customer segmentation.**Serve as a key intermediate step for other data mining tasks**− Cluster analysis can generate a compact summary of data for classification, testing, hypothesis generation; hence, it serves as a key intermediate step for other data mining tasks also.**Trend detection in dynamic data**− Clustering can also be used for trend detection in dynamic data by making various clusters of similar trends.**Social network analysis**− Clustering can be used in social network analysis. The examples are generating sequences in images, videos or audios.**Biological data analysis**− Clustering can also be used to make clusters of images, videos hence it can successfully be used in biological data analysis.

---

## 18. Machine Learning - K-Means Clustering Algorithm

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_k_means_clustering.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_k_means_clustering.htm)*

---

---
[Previous](/machine_learning/machine_learning_centroid_based_clustering.htm)[Quiz](/machine_learning/quiz_on_machine_learning_k_means_clustering.htm)[Next](/machine_learning/machine_learning_k_medoids_clustering.htm)
## K-Means Clustering Algorithm

K-means clustering algorithm computes the centroids and iterates until we it finds optimal centroid. It assumes that the number of clusters are already known. It is also called
**flat clustering**algorithm. The number of clusters identified from data by algorithm is represented by 'K' in K-means.
In this algorithm, the data points are assigned to a cluster in such a manner that the sum of the squared distance between the data points and centroid would be minimum. It is to be understood that less variation within the clusters will lead to more similar data points within same cluster.

## Working of K-Means Algorithm

We can understand the working of K-Means clustering algorithm with the help of following steps −

- **Step 1**− First, we need to specify the number of clusters, K, need to be generated by this algorithm.
- **Step 2**− Next, randomly select K data points and assign each data point to a cluster. In simple words, classify the data based on the number of data points.
- **Step 3**− Now it will compute the cluster centroids.
- **Step 4**− Next, keep iterating the following until we find optimal centroid which is the assignment of data points to the clusters that are not changing any more −**4.1**− First, the sum of squared distance between data points and centroids would be computed.**4.2**− Now, we have to assign each data point to the cluster that is closer than other cluster (centroid).**4.3**− At last compute the centroids for the clusters by taking the average of all data points of that cluster.
K-means follows
**Expectation-Maximization**approach to solve the problem. The Expectation-step is used for assigning the data points to the closest cluster and the Maximization-step is used for computing the centroid of each cluster.
While working with K-means algorithm we need to take care of the following things −

- While working with clustering algorithms including K-Means, it is recommended to standardize the data because such algorithms use distance-based measurement to determine the similarity between data points.
- Due to the iterative nature of K-Means and random initialization of centroids, K-Means may stick in a local optimum and may not converge to global optimum. That is why it is recommended to use different initializations of centroids.
The K-Means algorithm is a straightforward and efficient algorithm, and it can handle large datasets. However, it has some limitations, such as its sensitivity to the initial centroids, its tendency to converge to local optima, and its assumption of equal variance for all clusters.

## Objective of K-means Clustering

The main goals of cluster analysis are −

- To get a meaningful intuition from the data we are working with.
- Cluster-then-predict where different models will be built for different subgroups.
## Implementation of K-Means Algorithm Using Python

Python has several libraries that provide implementations of various machine learning algorithms, including K-Means clustering. Let's see how to implement the K-Means algorithm in Python using the scikit-learn library.

### Example - Understanding K-Means Algorithm

It is a simple example to understand how k-means works. In this example, we generate 300 random data points with two features. And apply K-means algorithm to generate clusters.

#### Step 1 − Import Required Libraries

To implement the K-Means algorithm in Python, we first need to import the required libraries. We will use the numpy and matplotlib libraries for data processing and visualization, respectively, and the scikit-learn library for the K-Means algorithm.

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

#### Step 2 − Generate Data

To test the K-Means algorithm, we need to generate some sample data. In this example, we will generate 300 random data points with two features. We will visualize the data also.

```
X = np.random.rand(300,2)

plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:, 0], X[:, 1], s=20);
plt.show()
```

#### Complete Code

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.random.rand(300,2)

plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:, 0], X[:, 1], s=20);
plt.show()
```

##### Output
![K-Means Clustering](/machine_learning/images/k_means_clustering.jpg)
#### Step 3 − Initialize K-Means

Next, we need to initialize the K-Means algorithm by specifying the number of clusters (K) and the maximum number of iterations.

```
kmeans = KMeans(n_clusters=3, max_iter=100)
```

#### Step 4 − Train the Model

After initializing the K-Means algorithm, we can train the model by fitting the data to the algorithm.

```
kmeans.fit(X)
```

#### Step 5 − Visualize the Clusters

To visualize the clusters, we can plot the data points and color them based on their assigned cluster.

```
plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, s=20)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
marker='x', c='r', s=50, alpha=0.9)
plt.show()
```

#### Complete Code

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.random.rand(300,2)

kmeans = KMeans(n_clusters=3, max_iter=100)
kmeans.fit(X)

plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, s=20)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
marker='x', c='r', s=50, alpha=0.9)
plt.show()
```

#### Output

The output of the above code will be a plot with the data points colored based on their assigned cluster, and the centroids marked with an 'x' symbol in red color.
![K-Means Clustering Plot](/machine_learning/images/k_means_clustering_plot.jpg)
### Example - Using 2D Datasets

In this example, we are going to first generate 2D dataset containing 4 different blobs and after that will apply k-means algorithm to see the result.

First, we will start by importing the necessary packages −

```
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
```

The following code will generate the 2D, containing four blobs −

```
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
```

Next, the following code will help us to visualize the dataset −

```
plt.scatter(X[:, 0], X[:, 1], s=20);
plt.show()
```

#### Complete Code

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=20);
plt.show()
```

#### Output
![Visualizing 2D Blog](/machine_learning/images/k_means_clustering_dataset_2d_blog.jpg)
Next, make an object of KMeans along with providing number of clusters, train the model and do the prediction as follows −

```
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
```

Now, with the help of following code we can plot and visualize the cluster's centers picked by k-means Python estimator −

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9);
plt.show()
```
![Visualizing Clusters Ceters](/machine_learning/images/k_means_clustering_clusters_centers.jpg)
### Example - Using Single Digit Dataset

Let us move to another example in which we are going to apply K-means clustering on simple digits dataset. K-means will try to identify similar digits without using the original label information.

First, we will start by importing the necessary packages −

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
```

Next, load the digit dataset from sklearn and make an object of it. We can also find number of rows and columns in this dataset as follows −

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape
```

#### Output

```
(1797, 64)
```

The above output shows that this dataset is having 1797 samples with 64 features.

We can perform the clustering as we did in Example 1 above −

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape
```

#### Output

```
(10, 64)
```

The above output shows that K-means created 10 clusters with 64 features.

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
   axi.set(xticks=[], yticks=[])
   axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
```

#### Output

As output, we will get following image showing clusters centers learned by k-means.
![Visualizing Digits Clusters Centers](/machine_learning/images/k_means_clustering_digit_clusters_centers.jpg)
The following lines of code will match the learned cluster labels with the true labels found in them −

```
from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
   mask = (clusters == i)
   labels[mask] = mode(digits.target[mask])[0]
```

Next, we can check the accuracy as follows −

```
from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)
```

#### Output

```
0.7935447968836951
```

The above output shows that the accuracy is around 80%.

## Advantages of K-Means Clustering Algorithm

The following are some advantages of K-Means clustering algorithms −

- It is very easy to understand and implement.
- If we have large number of variables then, K-means would be faster than Hierarchical clustering.
- On re-computation of centroids, an instance can change the cluster.
- Tighter clusters are formed with K-means as compared to Hierarchical clustering.
## Disadvantages of K-Means Clustering Algorithm

The following are some disadvantages of K-Means clustering algorithms −

- It is a bit difficult to predict the number of clusters i.e. the value of k.
- Output is strongly impacted by initial inputs like number of clusters (value of k).
- Order of data will have strong impact on the final output.
- It is very sensitive to rescaling. If we will rescale our data by means of normalization or standardization, then the output will completely change.final output.
- It is not good in doing clustering job if the clusters have a complicated geometric shape.
## Applications of K-Means Clustering

K-Means clustering is a versatile algorithm with various applications in several fields. Here we have highlighted some of the important applications −

### Image Segmentation

K-Means clustering can be used to segment an image into different regions based on the color or texture of the pixels. This technique is widely used in computer vision applications, such as object recognition, image retrieval, and medical imaging.

### Customer Segmentation

K-Means clustering can be used to segment customers into different groups based on their purchasing behavior or demographic characteristics. This technique is widely used in marketing applications, such as customer retention, loyalty programs, and targeted advertising.

### Anomaly Detection

K-Means clustering can be used to detect anomalies in a dataset by identifying data points that do not belong to any cluster. This technique is widely used in fraud detection, network intrusion detection, and predictive maintenance.

### Genomic Data Analysis

K-Means clustering can be used to analyze gene expression data to identify different groups of genes that are co-regulated or co-expressed. This technique is widely used in bioinformatics applications, such as drug discovery, disease diagnosis, and personalized medicine.

---

## 19. Machine Learning - Mean-Shift Clustering Algorithm

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_mean_shift_clustering.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_mean_shift_clustering.htm)*

---

---
[Previous](/machine_learning/machine_learning_k_medoids_clustering.htm)[Quiz](/machine_learning/quiz_on_machine_learning_mean_shift_clustering.htm)[Next](/machine_learning/machine_learning_hierarchical_clustering.htm)
## Mean-Shift Clustering Algorithm

The Mean-Shift clustering algorithm is a non-parametric clustering algorithm that works by iteratively shifting the mean of a data point towards the densest area of the data. The densest area of the data is determined by the kernel function, which is a function that assigns weights to the data points based on their distance from the mean. The kernel function used in Mean-Shift clustering is usually a Gaussian function.

The Mean-Shift clustering algorithm is a powerful clustering algorithm used in unsupervised learning. Unlike K-means clustering, it does not make any assumptions; hence it is a non-parametric algorithm.

The difference between K-Means algorithm and Mean-Shift is that later one does not need to specify the number of clusters in advance because the number of clusters will be determined by the algorithm w.r.t data.

## Working of Mean-Shift Algorithm

We can understand the working of Mean-Shift clustering algorithm with the help of following steps −

- **Step 1**− First, start with the data points assigned to a cluster of their own.
- **Step 2**− Next, this algorithm will compute the centroids.
- **Step 3**− In this step, location of new centroids will be updated.
- **Step 4**− Now, the process will be iterated and moved to the higher density region.
- **Step 5**− At last, it will be stopped once the centroids reach at position from where it cannot move further.
The Mean-Shift clustering algorithm is a density-based clustering algorithm, which means that it identifies clusters based on the density of the data points rather than the distance between them. In other words, the algorithm identifies clusters based on the areas where the density of the data points is highest.

## Implementation of Mean-Shift Clustering in Python

The Mean-Shift clustering algorithm can be implemented in Python programming language using the scikit-learn library. The scikit-learn library is a popular machine learning library in Python that provides various tools for data analysis and machine learning. The following steps are involved in implementing the Mean-Shift clustering algorithm in Python using the scikit-learn library −

### Step 1 − Import the necessary libraries

The
**numpy**library is used for scientific computing in Python, while the matplotlib library is used for data visualization. The**sklearn.cluster**library contains the**MeanShift**class, which is used for implementing the Mean-Shift clustering algorithm in Python.
The
**estimate_bandwidth**function is used to estimate the bandwidth of the kernel function, which is an important parameter in the Mean-Shift clustering algorithm.
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
```

### Step 2 − Generate the data

In this step, we generate a random dataset with 500 data points and 2 features. We use the
**numpy.random.randn**function to generate the data.
```
# Generate the data
X = np.random.randn(500,2)
```

### Step 3 − Estimate the bandwidth of the kernel function

In this step, we estimate the bandwidth of the kernel function using the
**estimate_bandwidth**function. The bandwidth is an important parameter in the Mean-Shift clustering algorithm, which determines the width of the kernel function.
```
# Estimate the bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
```

### Step 4 − Initialize the Mean-Shift clustering algorithm

In this step, we initialize the Mean-Shift clustering algorithm using the
**MeanShift**class. We pass the bandwidth parameter to the class to set the width of the kernel function.
```
# Initialize the Mean-Shift algorithm
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
```

### Step 5 − Train the model

In this step, we train the Mean-Shift clustering algorithm on the dataset using the fit method of the MeanShift class.

```
# Train the model
ms.fit(X)
```

### Step 6 − Visualize the results

```
# Visualize the results
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

# Plot the data points and the centroids
plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='*', s=300, c='r')
plt.show()
```

In this step, we visualize the results of the Mean-Shift clustering algorithm. We extract the cluster labels and the cluster centers from the trained model. We then print the number of estimated clusters. Finally, we plot the data points and the centroids using the matplotlib library.

### Example - Mean-Shift Clustering Algorithm

Here is the complete implementation example of Mean-Shift Clustering Algorithm in python −

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Generate the data
X = np.random.randn(500,2)

# Estimate the bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)

# Initialize the Mean-Shift algorithm
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# Train the model
ms.fit(X)

# Visualize the results
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

# Plot the data points and the centroids
plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='summer')
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='*',
s=200, c='r')
plt.show()
```

#### Output

When you execute the program, it will produce the following plot as the output −
![Mean Shift Clustering](/machine_learning/images/mean_shift_clustering.jpg)
### Example - Mean-Shift Clustering Algorithm using 2D dataset

It is a simple example to understand how Mean-Shift algorithm works. In this example, we are going to first generate 2D dataset containing 4 different blobs and after that will apply Mean-Shift algorithm to see the result.

```
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.datasets import make_blobs
centers = [[3,3,3],[4,5,5],[3,10,10]]
X, _ = make_blobs(n_samples = 700, centers = centers, cluster_std = 0.5)
plt.scatter(X[:,0],X[:,1])
plt.show()
```

#### Output
![2d data points with 4 blobs](/machine_learning/images/mean_shift_clustering_generate_2d_data.jpg)
### Example - Mean-Shift Clustering Algorithm using Clusters

```
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)
colors = 10*['r.','g.','b.','c.','k.','y.','m.']
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 3)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
    marker=".",color='k', s=20, linewidths = 5, zorder=10)
plt.show()
```

#### Output

```
[[ 4.03457771  5.03063843  4.92928409]
 [ 3.01124859  2.9957586   2.981767  ]
 [ 2.94969928 10.00712673 10.01575558]]
Estimated clusters: 3
```
![Visualizing Clusters](/machine_learning/images/mean_shift_clustering_visualizing_clusters.jpg)
## Applications of Mean-Shift Clustering

The Mean-Shift clustering algorithm has several applications in various fields. Some of the applications of Mean-Shift clustering are as follows −

- **Computer vision**− Mean-Shift clustering is widely used in computer vision for object tracking, image segmentation, and feature extraction.
- **Image processing**− Mean-Shift clustering is used for image segmentation, which is the process of dividing an image into multiple segments based on the similarity of the pixels.
- **Anomaly detection**− Mean-Shift clustering can be used for detecting anomalies in data by identifying the areas with low density.
- **Customer segmentation**− Mean-Shift clustering can be used for customer segmentation in marketing by identifying groups of customers with similar behavior and preferences.
- **Social network analysis**− Mean-Shift clustering can be used for clustering users in social networks based on their interests and interactions.
## Advantages and Disadvantages

Let's discuss some advantages and disadvantages of the means-shift clustering algorithm.

### Advantages

The following are some advantages of Mean-Shift clustering algorithm −

- It does not need to make any model assumption as like in K-means or Gaussian mixture.
- It can also model the complex clusters which have nonconvex shape.
- It only needs one parameter named bandwidth which automatically determines the number of clusters.
- There is no issue of local minima as like in K-means.
- No problem generated from outliers.
### Disadvantages

The following are some disadvantages of Mean-Shift clustering algorithm −

- Mean-shift algorithm does not work well in case of high dimension, where number of clusters changes abruptly.
- We do not have any direct control on the number of clusters but in some applications, we need a specific number of clusters.
- It cannot differentiate between meaningful and meaningless modes.

---

## 20. Machine Learning - Hierarchical Clustering

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_hierarchical_clustering.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_hierarchical_clustering.htm)*

---

---

## 21. K-Nearest Neighbors (KNN) in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_knn_nearest_neighbors.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_knn_nearest_neighbors.htm)*

---

---
[Previous](/machine_learning/machine_learning_logistic_regression.htm)[Quiz](/machine_learning/quiz_on_machine_learning_knn_nearest_neighbors.htm)[Next](/machine_learning/machine_learning_naive_bayes_algorithms.htm)
## K-Nearest Neighbors (KNN) Algorithm

K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems.  However, it is mainly used for classification predictive problems in industry. The main idea behind KNN is to find the k-nearest data points to a given test data point and use these nearest neighbors to make a prediction. The value of k is a hyperparameter that needs to be tuned, and it represents the number of neighbors to consider.

For classification problems, the KNN algorithm assigns the test data point to the class that appears most frequently among the k-nearest neighbors. In other words, the class with the highest number of neighbors is the predicted class.

For regression problems, the KNN algorithm assigns the test data point the average of the k-nearest neighbors' values.

The distance metric used to measure the similarity between two data points is an essential factor that affects the KNN algorithm's performance. The most commonly used distance metrics are Euclidean distance, Manhattan distance, and Minkowski distance.

The following two properties would define KNN well −

- **Lazy learning algorithm**− KNN is a lazy learning algorithm because it does not have a specialized training phase and uses all the data for training while classification.
- **Non-parametric learning algorithm**− KNN is also a non-parametric learning algorithm because it doesn't assume anything about the underlying data.
## How Does K-Nearest Neighbors Algorithm Work?

K-nearest neighbors (KNN) algorithm uses 'feature similarity' to predict the values of new datapoints which further means that the new data point will be assigned a value based on how closely it matches the points in the training set. We can understand its working with the help of following steps −

- **Step 1**− For implementing any algorithm, we need dataset. So during the first step of KNN, we must load the training as well as test data.
- **Step 2**− Next, we need to choose the value of K i.e. the nearest data points. K can be any integer.
- **Step 3**− For each point in the test data do the following −**3.1**− Calculate the distance between test data and each row of training data with the help of any of the method namely: Euclidean, Manhattan or Hamming distance. The most commonly used method to calculate distance is Euclidean.**3.2**− Now, based on the distance value, sort them in ascending order.**3.3**− Next, it will choose the top K rows from the sorted array.**3.4**− Now, it will assign a class to the test point based on most frequent class of these rows.
- **Step 4**− End
### Example

The following is an example to understand the concept of K and working of KNN algorithm −

Suppose we have a dataset which can be plotted as follows −
![Violate](/machine_learning/images/violate.jpg)
Now, we need to classify new data point with black dot (at point 60,60) into blue or red class. We are assuming K = 3 i.e. it would find three nearest data points. It is shown in the next diagram −
![Circle](/machine_learning/images/circle.jpg)
We can see in the above diagram the three nearest neighbors of the data point with black dot. Among those three, two of them lies in Red class hence the black dot will also be assigned in red class.

## Building a K Nearest Neighbors Model

We can follow the below steps to build a KNN model −

- **Load the data**− The first step is to load the dataset into memory. This can be done using various libraries such as pandas or numpy.
- **Split the data**− The next step is to split the data into training and test sets. The training set is used to train the KNN algorithm, while the test set is used to evaluate its performance.
- **Normalize the data**− Before training the KNN algorithm, it is essential to normalize the data to ensure that each feature contributes equally to the distance metric calculation.
- **Calculate distances**− Once the data is normalized, the KNN algorithm calculates the distances between the test data point and each data point in the training set.
- **Select k-nearest neighbors**− The KNN algorithm selects the k-nearest neighbors based on the distances calculated in the previous step.
- **Make a prediction**− For classification problems, the KNN algorithm assigns the test data point to the class that appears most frequently among the k-nearest neighbors. For regression problems, the KNN algorithm assigns the test data point the average of the k-nearest neighbors' values.
- **Evaluate performance**− Finally, the KNN algorithm's performance is evaluated using various metrics such as accuracy, precision, recall, and F1-score.
## Implementation of KNN Algorithm in Python

As we know K-nearest neighbors (KNN) algorithm can be used for both classification as well as regression. The following are the recipes in Python to use KNN as classifier as well as regressor −

### KNN as Classifier

First, start with importing necessary python packages −

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

Next, download the iris dataset from its weblink as follows −

```
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
```

Next, we need to assign column names to the dataset as follows −

```
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
```

Now, we need to read dataset to pandas dataframe as follows −

```
dataset = pd.read_csv(path, names=headernames)
dataset.head()
```
slno.sepal-lengthsepal-widthpetal-lengthpetal-widthClass05.13.51.40.2Iris-setosa14.93.01.40.2Iris-setosa24.73.21.30.2Iris-setosa34.63.11.50.2Iris-setosa45.03.61.40.2Iris-setosa
Data Preprocessing will be done with the help of following script lines −

```
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
```

Next, we will divide the data into train and test split. Following code will split the dataset into 60% training data and 40% of testing data −

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
```

Next, data scaling will be done as follows −

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Next, train the model with the help of KNeighborsClassifier class of sklearn as follows −

```
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)
```

At last we need to make prediction. It can be done with the help of following script −

```
y_pred = classifier.predict(X_test)
```

Next, print the results as follows −

```
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
```

### Output

```
Confusion Matrix:
[[21 0 0]
[ 0 16 0]
[ 0 7 16]]
Classification Report:
            precision      recall       f1-score       support
Iris-setosa       1.00        1.00         1.00          21
Iris-versicolor   0.70        1.00         0.82          16
Iris-virginica    1.00        0.70         0.82          23
micro avg         0.88        0.88         0.88          60
macro avg         0.90        0.90         0.88          60
weighted avg      0.92        0.88         0.88          60

Accuracy: 0.8833333333333333
```

### KNN as Regressor

First, start with importing necessary Python packages −

```
import numpy as np
import pandas as pd
```

Next, download the iris dataset from its weblink as follows −

```
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
```

Next, we need to assign column names to the dataset as follows −

```
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
```

Now, we need to read dataset to pandas dataframe as follows −

```
data = pd.read_csv(url, names=headernames)
array = data.values
X = array[:,:2]
Y = array[:,2]
data.shape

output:(150, 5)
```

Next, import KNeighborsRegressor from sklearn to fit the model −

```
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors=10)
knnr.fit(X, y)
```

At last, we can find the MSE as follows −

```
print ("The MSE is:",format(np.power(y-knnr.predict(X),2).mean()))
```

### Output

```
The MSE is: 0.12226666666666669
```

## Pros and Cons of KNN

### Pros

- 
It is very simple algorithm to understand and interpret.

- 
It is very useful for nonlinear data because there is no assumption about data in this algorithm.

- 
It is a versatile algorithm as we can use it for classification as well as regression.

- 
It has relatively high accuracy but there are much better supervised learning models than KNN.

### Cons

- 
It is computationally a bit expensive algorithm because it stores all the training data.

- 
High memory storage required as compared to other supervised learning algorithms.

- 
Prediction is slow in case of big N.

- 
It is very sensitive to the scale of data as well as irrelevant features.

## Applications of KNN

The following are some of the areas in which KNN can be applied successfully −

### Banking System

KNN can be used in banking system to predict weather an individual is fit for loan approval? Does that individual have the characteristics similar to the defaulters one?

### Calculating Credit Ratings

KNN algorithms can be used to find an individual's credit rating by comparing with the persons having similar traits.

### Politics

With the help of KNN algorithms, we can classify a potential voter into various classes like "Will Vote", "Will not Vote", "Will Vote to Party 'Congress', "Will Vote to Party 'BJP'.

Other areas in which KNN algorithm can be used are Speech Recognition, Handwriting Detection, Image Recognition and Video Recognition.

---

## 22. Performance Metrics in Machine Learning

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_performance_metrics.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_performance_metrics.htm)*

---

---

## 23. Machine Learning - Automatic Workflows

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_automatic_workflows.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_automatic_workflows.htm)*

---

---
[Previous](/machine_learning/machine_learning_performance_metrics.htm)[Quiz](/machine_learning/quiz_on_machine_learning_automatic_workflows.htm)[Next](/machine_learning/machine_learning_boost_model_performance.htm)
## Introduction

In order to execute and produce results successfully, a machine learning model must automate some standard workflows. The process of automate these standard workflows can be done with the help of Scikit-learn Pipelines. From a data scientists perspective, pipeline is a generalized, but very important concept. It basically allows data flow from its raw format to some useful information. The working of pipelines can be understood with the help of following diagram −
![Data](/machine_learning/images/data.jpg)
The blocks of ML pipelines are as follows −
**Data ingestion**− As the name suggests, it is the process of importing the data for use in ML project. The data can be extracted in real time or batches from single or multiple systems. It is one of the most challenging steps because the quality of data can affect the whole ML model.**Data Preparation**− After importing the data, we need to prepare data to be used for our ML model. Data preprocessing is one of the most important technique of data preparation.**ML Model Training**− Next step is to train our ML model. We have various ML algorithms like supervised, unsupervised, reinforcement to extract the features from data, and make predictions.**Model Evaluation**− Next, we need to evaluate the ML model. In case of AutoML pipeline, ML model can be evaluated with the help of various statistical methods and business rules.**ML Model retraining**− In case of AutoML pipeline, it is not necessary that the first model is best one. The first model is considered as a baseline model and we can train it repeatably to increase models accuracy.**Deployment**− At last, we need to deploy the model. This step involves applying and migrating the model to business operations for their use.
## Challenges Accompanying ML Pipelines

In order to create ML pipelines, data scientists face many challenges. These challenges fall into the following three categories −

### Quality of Data

The success of any ML model depends heavily on the quality of data. If the data we are providing to ML model is not accurate, reliable and robust, then we are going to end with wrong or misleading output.

### Data Reliability

Another challenge associated with ML pipelines is the reliability of data we are providing to the ML model. As we know, there can be various sources from which data scientist can acquire data but to get the best results, it must be assured that the data sources are reliable and trusted.

### Data Accessibility

To get the best results out of ML pipelines, the data itself must be accessible which requires consolidation, cleansing and curation of data. As a result of data accessibility property, metadata will be updated with new tags.

## Modelling ML Pipeline and Data Preparation

Data leakage, happening from training dataset to testing dataset, is an important issue for data scientist to deal with while preparing data for ML model. Generally, at the time of data preparation, data scientist uses techniques like standardization or normalization on entire dataset before learning. But these techniques cannot help us from the leakage of data because the training dataset would have been influenced by the scale of the data in the testing dataset.

By using ML pipelines, we can prevent this data leakage because pipelines ensure that data preparation like standardization is constrained to each fold of our cross-validation procedure.

### Example

The following is an example in Python that demonstrate data preparation and model evaluation workflow. For this purpose, we are using Pima Indian Diabetes dataset from Sklearn. First, we will be creating pipeline that standardized the data. Then a Linear Discriminative analysis model will be created and at last the pipeline will be evaluated using 10-fold cross validation.

First, import the required packages as follows −

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
```

Next, we will create a pipeline with the help of the following code −

```
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
```

At last, we are going to evaluate this pipeline and output its accuracy as follows −

```
kfold = KFold(n_splits=20, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

#### Output

```
0.7790148448043184
```

The above output is the summary of accuracy of the setup on the dataset.

## Modelling ML Pipeline and Feature Extraction

Data leakage can also happen at feature extraction step of ML model. That is why feature extraction procedures should also be restricted to stop data leakage in our training dataset. As in the case of data preparation, by using ML pipelines, we can prevent this data leakage also. FeatureUnion, a tool provided by ML pipelines can be used for this purpose.

### Example

The following is an example in Python that demonstrates feature extraction and model evaluation workflow. For this purpose, we are using Pima Indian Diabetes dataset from Sklearn.

First, 3 features will be extracted with PCA (Principal Component Analysis). Then, 6 features will be extracted with Statistical Analysis. After feature extraction, result of multiple feature selection and extraction procedures will be combined by using

FeatureUnion tool. At last, a Logistic Regression model will be created, and the pipeline will be evaluated using 10-fold cross validation.

First, import the required packages as follows −

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
```

Next, feature union will be created as follows −

```
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
```

Next, pipeline will be creating with the help of following script lines −

```
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)
```

At last, we are going to evaluate this pipeline and output its accuracy as follows −

```
kfold = KFold(n_splits=20, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

#### Output

```
0.7789811066126855
```

The above output is the summary of accuracy of the setup on the dataset.

---

## 24. Machine Learning - Boost Model Performance

*Source: [https://www.tutorialspoint.com/machine_learning/machine_learning_boost_model_performance.htm](https://www.tutorialspoint.com/machine_learning/machine_learning_boost_model_performance.htm)*

---

---
[Previous](/machine_learning/machine_learning_automatic_workflows.htm)[Quiz](/machine_learning/quiz_on_machine_learning_boost_model_performance.htm)[Next](/machine_learning/machine_learning_gradient_boosting.htm)
Boosting is a popular ensemble learning technique that combines several weak learners to create a strong learner. It works by iteratively training weak learners on subsets of the data and assigning higher weights to the misclassified samples to increase their importance in the subsequent iterations. This process is repeated until the desired level of performance is achieved.

Here are some techniques to boost model performance in machine learning −

- **Feature Engineering**− Feature engineering involves creating new features from the existing features or transforming the existing features to make them more informative for the model. This can include techniques such as one-hot encoding, scaling, normalization, and feature selection.
- **Hyperparameter Tuning**− Hyperparameters are parameters that are not learned during training but are set by the data scientist. They control the behavior of the model, and tuning them can significantly impact model performance. Grid search and randomized search are common techniques for hyperparameter tuning.
- **Ensemble Learning**− Ensemble learning involves combining multiple models to improve performance. Techniques such as bagging, boosting, and stacking can be used to create ensembles. Random forests are an example of a bagging ensemble, while gradient boosting machines (GBMs) are an example of a boosting ensemble.
- **Regularization**− Regularization is a technique that prevents overfitting by adding a penalty term to the loss function. L1 regularization (Lasso) and L2 regularization (Ridge) are common techniques used in linear models, while dropout is a technique used in neural networks.
- **Data Augmentation**− Data augmentation involves generating new data from the existing data by applying transformations such as rotation, scaling, and flipping. This can help to reduce overfitting and improve model performance.
- **Model Architecture**− The architecture of the model can significantly impact its performance. Techniques such as deep learning and convolutional neural networks (CNNs) can be used to create more complex models that are better able to learn complex patterns in the data.
- **Early Stopping**− Early stopping is a technique used to prevent overfitting by stopping the training process once the model performance stops improving on a validation set. This prevents the model from continuing to learn the noise in the data and can help to improve generalization.
- **Cross-Validation**− Cross-validation is a technique used to evaluate the performance of a model on multiple subsets of the data. This can help to identify overfitting and can be used to select the best hyperparameters for the model.
These techniques can be implemented in Python using various machine learning libraries such as scikit-learn, TensorFlow, and Keras. By using these techniques, data scientists can improve the performance of their models and create more accurate predictions.

The following example below in which implement cross-validation using Scikit-learn −

## Example

```
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier()

# Perform 5-fold cross-validation on the classifier
scores = cross_val_score(gb_clf, X, y, cv=5)

# Print the average accuracy and standard deviation of the cross-validation scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

### Output

When you execute this code, it will produce the following output −

```
Accuracy: 0.96 (+/- 0.07)
```

## Performance Improvement with Ensembles

Ensembles can give us boost in the machine learning result by combining several models. Basically, ensemble models consist of several individually trained supervised learning models and their results are merged in various ways to achieve better predictive performance compared to a single model. Ensemble methods can be divided into following two groups −

### Sequential ensemble methods

As the name implies, in these kind of ensemble methods, the base learners are generated sequentially. The motivation of such methods is to exploit the dependency among base learners.

### Parallel ensemble methods

As the name implies, in these kind of ensemble methods, the base learners are generated in parallel. The motivation of such methods is to exploit the independence among base learners.

## Ensemble Learning Methods

The following are the most popular ensemble learning methods i.e. the methods for combining the predictions from different models −

### Bagging

The term bagging is also known as bootstrap aggregation. In bagging methods, ensemble model tries to improve prediction accuracy and decrease model variance by combining predictions of individual models trained over randomly generated training samples. The final prediction of ensemble model will be given by calculating the average of all predictions from the individual estimators. One of the best examples of bagging methods are random forests.

### Boosting

In boosting method, the main principle of building ensemble model is to build it incrementally by training each base model estimator sequentially. As the name suggests, it basically combine several week base learners, trained sequentially over multiple iterations of training data, to build powerful ensemble. During the training of week base learners, higher weights are assigned to those learners which were misclassified earlier. The example of boosting method is AdaBoost.

### Voting

In this ensemble learning model, multiple models of different types are built and some simple statistics, like calculating mean or median etc., are used to combine the predictions. This prediction will serve as the additional input for training to make the final prediction.

## Bagging Ensemble Algorithms

The following are three bagging ensemble algorithms −

### Bagged Decision Tree

As we know that bagging ensemble methods work well with the algorithms that have high variance and, in this concern, the best one is decision tree algorithm. In the following Python recipe, we are going to build bagged decision tree ensemble model by using BaggingClassifier function of sklearn with DecisionTreeClasifier (a classification & regression trees algorithm) on Pima Indians diabetes dataset.

First, import the required packages as follows −

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
```

Now, we need to load the Pima diabetes dataset as we did in the previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
```

Next, give the input for 10-fold cross validation as follows −

```
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
```

We need to provide the number of trees we are going to build. Here we are building 150 trees −

```
num_trees = 150
```

Next, build the model with the help of following script −

```
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
```

Calculate and print the result as follows −

```
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

### Output

```
0.7733766233766234
```

The output above shows that we got around 77% accuracy of our bagged decision tree classifier model.

### Random Forest

It is an extension of bagged decision trees. For individual classifiers, the samples of training dataset are taken with replacement, but the trees are constructed in such a way that reduces the correlation between them. Also, a random subset of features is considered to choose each split point rather than greedily choosing the best split point in construction of each tree.

In the following Python recipe, we are going to build bagged random forest ensemble model by using RandomForestClassifier class of sklearn on Pima Indians diabetes dataset.

First, import the required packages as follows −

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
```

Next, give the input for 10-fold cross validation as follows −

```
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
```

We need to provide the number of trees we are going to build. Here we are building 150 trees with split points chosen from 5 features −

```
num_trees = 150
max_features = 5
```

Next, build the model with the help of following script −

```
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
```

Calculate and print the result as follows −

```
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

### Output

```
0.7629357484620642
```

The output above shows that we got around 76% accuracy of our bagged random forest classifier model.

### Extra Trees

It is another extension of bagged decision tree ensemble method. In this method, the random trees are constructed from the samples of the training dataset.

In the following Python recipe, we are going to build extra tree ensemble model by using ExtraTreesClassifier class of sklearn on Pima Indians diabetes dataset.

First, import the required packages as follows −

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
```

Next, give the input for 10-fold cross validation as follows −

```
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
```

We need to provide the number of trees we are going to build. Here we are building 150 trees with split points chosen from 5 features −

```
num_trees = 150
max_features = 5
```

Next, build the model with the help of following script −

```
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
```

Calculate and print the result as follows −

```
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

### Output

```
0.7551435406698566
```

The output above shows that we got around 75.5% accuracy of our bagged extra trees classifier model.

## Boosting Ensemble Algorithms

The followings are the two most common boosting ensemble algorithms −

### AdaBoost

It is one the most successful boosting ensemble algorithm. The main key of this algorithm is in the way they give weights to the instances in dataset. Due to this the algorithm needs to pay less attention to the instances while constructing subsequent models.

In the following Python recipe, we are going to build Ada Boost ensemble model for classification by using AdaBoostClassifier class of sklearn on Pima Indians diabetes dataset.

First, import the required packages as follows −

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
```

Next, give the input for 10-fold cross validation as follows −

```
seed = 5
kfold = KFold(n_splits=10, random_state=seed)
```

We need to provide the number of trees we are going to build. Here we are building 150 trees with split points chosen from 5 features −

```
num_trees = 50
```

Next, build the model with the help of following script −

```
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
```

Calculate and print the result as follows −

```
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

### Output

```
0.7539473684210527
```

The output above shows that we got around 75% accuracy of our AdaBoost classifier ensemble model.

### Stochastic Gradient Boosting

It is also called Gradient Boosting Machines. In the following Python recipe, we are going to build Stochastic Gradient Boostingensemble model for classification by using GradientBoostingClassifier class of sklearn on Pima Indians diabetes dataset.

First, import the required packages as follows −

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
```

Next, give the input for 10-fold cross validation as follows −

```
seed = 5
kfold = KFold(n_splits=10, random_state=seed)
```

We need to provide the number of trees we are going to build. Here we are building 150 trees with split points chosen from 5 features −

```
num_trees = 50
```

Next, build the model with the help of following script −

```
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
```

Calculate and print the result as follows −

```
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

### Output

```
0.7746582365003418
```

The output above shows that we got around 77.5% accuracy of our Gradient Boosting classifier ensemble model.

## Voting Ensemble Algorithms

As discussed, voting first creates two or more standalone models from training dataset and then a voting classifier will wrap the model along with taking the average of the predictions of sub-model whenever needed new data.

In the following Python recipe, we are going to build Voting ensemble model for classification by using VotingClassifier class of sklearn on Pima Indians diabetes dataset. We are combining the predictions of logistic regression, Decision Tree classifier and SVM together for a classification problem as follows −

First, import the required packages as follows −

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
```

Next, give the input for 10-fold cross validation as follows −

```
kfold = KFold(n_splits=10, random_state=7)
```

Next, we need to create sub-models as follows −

```
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
```

Now, create the voting ensemble model by combining the predictions of above created sub models.

```
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
```

### Output

```
0.7382262474367738
```

The output above shows that we got around 74% accuracy of our voting classifier ensemble model.

---

## 25. Improving Performance of ML Model (Contd)

*Source: [https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_improving_performance_of_ml_model.htm](https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_improving_performance_of_ml_model.htm)*

---

---
[Previous](/machine_learning_with_python/machine_learning_improving_performance_of_ml_models.htm)[Quiz](/machine_learning_with_python/quiz_on_machine_learning_improving_performance_of_ml_model.htm)[Next](/machine_learning_with_python/machine_learning_with_python_quick_guide.htm)
## Performance Improvement with Algorithm Tuning

As we know that ML models are parameterized in such a way that their behavior can be adjusted for a specific problem. Algorithm tuning means finding the best combination of these parameters so that the performance of ML model can be improved. This process sometimes called hyperparameter optimization and the parameters of algorithm itself are called hyperparameters and coefficients found by ML algorithm are called parameters.

Here, we are going to discuss about some methods for algorithm parameter tuning provided by Python Scikit-learn.

### Grid Search Parameter Tuning

It is a parameter tuning approach. The key point of working of this method is that it builds and evaluate the model methodically for every possible combination of algorithm parameter specified in a grid. Hence, we can say that this algorithm is having search nature.
**Example**
In the following Python recipe, we are going to perform grid search by using GridSearchCV class of sklearn for evaluating various alpha values for the Ridge Regression algorithm on Pima Indians diabetes dataset.

First, import the required packages as follows −

```
import numpy
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
```

Next, evaluate the various alpha values as follows −

```
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
```

Now, we need to apply grid search on our model −

```
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
```

Print the result with following script line −

```
print(grid.best_score_)
print(grid.best_estimator_.alpha)
```
**Output**
```
0.2796175593129722
1.0
```

The above output gives us the optimal score and the set of parameters in the grid that achieved that score. The alpha value in this case is 1.0.

### Random Search Parameter Tuning

It is a parameter tuning approach. The key point of working of this method is that it samples the algorithm parameters from a random distribution for a fixed number of iterations.
**Example**
In the following Python recipe, we are going to perform random search by using RandomizedSearchCV class of sklearn for evaluating different alpha values between 0 and 1 for the Ridge Regression algorithm on Pima Indians diabetes dataset.

First, import the required packages as follows −

```
import numpy
from pandas import read_csv
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
```

Now, we need to load the Pima diabetes dataset as did in previous examples −

```
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
```

Next, evaluate the various alpha values on Ridge regression algorithm as follows −

```
param_grid = {'alpha': uniform()}
model = Ridge()
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50,
random_state=7)
random_search.fit(X, Y)
```

Print the result with following script line −

```
print(random_search.best_score_)
print(random_search.best_estimator_.alpha)
```
**Output**
```
0.27961712703051084
0.9779895119966027
```

The above output gives us the optimal score just similar to the grid search.

---


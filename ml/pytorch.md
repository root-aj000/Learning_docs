# Pytorch

## Table of Contents

1. [PyTorch - Introduction](#pytorch---introduction)
2. [PyTorch - Installation](#pytorch---installation)
3. [Mathematical Building Blocks of Neural Networks](#mathematical-building-blocks-of-neural-networks)
4. [PyTorch - Neural Network Basics](#pytorch---neural-network-basics)
5. [Universal Workflow of Machine Learning](#universal-workflow-of-machine-learning)
6. [PyTorch - Machine Learning vs. Deep Learning](#pytorch---machine-learning-vs-deep-learning)
7. [PyTorch - Implementing First Neural Network](#pytorch---implementing-first-neural-network)
8. [PyTorch - Neural Networks to Functional Blocks](#pytorch---neural-networks-to-functional-blocks)
9. [PyTorch - Terminologies](#pytorch---terminologies)
10. [PyTorch - Loading Data](#pytorch---loading-data)
11. [PyTorch - Linear Regression](#pytorch---linear-regression)
12. [PyTorch - Convolutional Neural Network](#pytorch---convolutional-neural-network)
13. [PyTorch - Recurrent Neural Network](#pytorch---recurrent-neural-network)
14. [PyTorch - Datasets](#pytorch---datasets)
15. [PyTorch - Introduction to Convents](#pytorch---introduction-to-convents)
16. [PyTorch - Training a Convent from Scratch](#pytorch---training-a-convent-from-scratch)
17. [PyTorch - Feature Extraction in Convents](#pytorch---feature-extraction-in-convents)
18. [PyTorch - Visualization of Convents](#pytorch---visualization-of-convents)
19. [PyTorch - Sequence Processing with Convents](#pytorch---sequence-processing-with-convents)
20. [PyTorch - Word Embedding](#pytorch---word-embedding)
21. [PyTorch - Recursive Neural Networks](#pytorch---recursive-neural-networks)

---

## 1. PyTorch - Introduction

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_introduction.htm](https://www.tutorialspoint.com/pytorch/pytorch_introduction.htm)*

---

---

## 2. PyTorch - Installation

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_installation.htm](https://www.tutorialspoint.com/pytorch/pytorch_installation.htm)*

---

---
[Previous](/pytorch/pytorch_introduction.htm)[Quiz](/pytorch/quiz_on_pytorch_installation.htm)[Next](/pytorch/pytorch_mathematical_building_blocks_of_neural_networks.htm)
PyTorch is a popular deep learning framework. In this tutorial, we consider Windows 10 as our operating system. The steps for a successful environmental setup are as follows −

## Step 1

The following link includes a list of packages which includes suitable packages for PyTorch.
[https://drive.google.com/drive/folders/0B-X0-FlSGfCYdTNldW02UGl4MXM](https://drive.google.com/drive/folders/0B-X0-FlSGfCYdTNldW02UGl4MXM)
All you need to do is download the respective packages and install it as shown in the following screenshots −
![pytorch](/pytorch/images/pytorch_ex.jpg)![packages](/pytorch/images/packages.jpg)
## Step 2

It involves verifying the installation of PyTorch framework using Anaconda Framework.

Following command is used to verify the same −

```
conda list
```
![PyTorch framework](/pytorch/images/pytorch_framework.jpg)
Conda list shows the list of frameworks which is installed.
![Conda list](/pytorch/images/conda_list.jpg)
The highlighted part shows that PyTorch has been successfully installed in our system.

---

## 3. Mathematical Building Blocks of Neural Networks

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_mathematical_building_blocks_of_neural_networks.htm](https://www.tutorialspoint.com/pytorch/pytorch_mathematical_building_blocks_of_neural_networks.htm)*

---

---
[Previous](/pytorch/pytorch_installation.htm)[Quiz](/pytorch/quiz_on_pytorch_mathematical_building_blocks_of_neural_networks.htm)[Next](/pytorch/pytorch_neural_network_basics.htm)
Mathematics is vital in any machine learning algorithm and includes various core concepts of mathematics to get the right algorithm designed in a specific way.

The importance of mathematics topics for machine learning and data science is mentioned below −
![machine learning](/pytorch/images/machine_learning.jpg)
Now, let us focus on the major mathematical concepts of machine learning which is important from Natural Language Processing point of view −

## Vectors

Vector is considered to be array of numbers which is either continuous or discrete and the space which consists of vectors is called as vector space. The space dimensions of vectors can be either finite or infinite but it has been observed that machine learning and data science problems deal with fixed length vectors.

The vector representation is displayed as mentioned below −

```
temp = torch.FloatTensor([23,24,24.5,26,27.2,23.0])
temp.size()
Output - torch.Size([6])
```

In machine learning, we deal with multidimensional data. So vectors become very crucial and are considered as input features for any prediction problem statement.

## Scalars

Scalars are termed to have zero dimensions containing only one value. When it comes to PyTorch, it does not include a special tensor with zero dimensions; hence the declaration will be made as follows −

```
x = torch.rand(10)
x.size()
Output - torch.Size([10])
```

## Matrices

Most of the structured data is usually represented in the form of tables or a specific matrix. We will use a dataset called Boston House Prices, which is readily available in the Python scikit-learn machine learning library.

```
boston_tensor = torch.from_numpy(boston.data)
boston_tensor.size()
Output: torch.Size([506, 13])
boston_tensor[:2]
Output:
Columns 0 to 7
0.0063 18.0000 2.3100 0.0000 0.5380 6.5750 65.2000 4.0900
0.0273 0.0000 7.0700 0.0000 0.4690 6.4210 78.9000 4.9671
Columns 8 to 12
1.0000 296.0000 15.3000 396.9000 4.9800
2.0000 242.0000 17.8000 396.9000 9.1400
```

---

## 4. PyTorch - Neural Network Basics

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_neural_network_basics.htm](https://www.tutorialspoint.com/pytorch/pytorch_neural_network_basics.htm)*

---

---
[Previous](/pytorch/pytorch_mathematical_building_blocks_of_neural_networks.htm)[Quiz](/pytorch/quiz_on_pytorch_neural_network_basics.htm)[Next](/pytorch/pytorch_universal_workflow_of_machine_learning.htm)
The main principle of neural network includes a collection of basic elements, i.e., artificial neuron or perceptron. It includes several basic inputs such as x1, x2.. xn which produces a binary output if the sum is greater than the activation potential.

The schematic representation of sample neuron is mentioned below −
![sample neuron](/pytorch/images/sample_neuron.jpg)
The output generated can be considered as the weighted sum with activation potential or bias.

$$Output=\sum_jw_jx_j+Bias$$

The typical neural network architecture is described below −
![neural network architecture](/pytorch/images/neural_network_architecture.jpg)
The layers between input and output are referred to as hidden layers, and the density and type of connections between layers is the configuration. For example, a fully connected configuration has all the neurons of layer L connected to those of L+1. For a more pronounced localization, we can connect only a local neighbourhood, say nine neurons, to the next layer. Figure 1-9 illustrates two hidden layers with dense connections.

The various types of neural networks are as follows −

## Feedforward Neural Networks

Feedforward neural networks include basic units of neural network family. The movement of data in this type of neural network is from the input layer to output layer, via present hidden layers. The output of one layer serves as the input layer with restrictions on any kind of loops in the network architecture.
![Feedforward Neural Networks](/pytorch/images/feedforward_neural_networks.jpg)
## Recurrent Neural Networks

Recurrent Neural Networks are when the data pattern changes consequently over a period. In RNN, same layer is applied to accept the input parameters and display output parameters in specified neural network.
![Recurrent Neural Networks](/pytorch/images/recurrent_neural_networks.jpg)
Neural networks can be constructed using the torch.nn package.
![Construct Neural Networks](/pytorch/images/construct_neural_networks.jpg)
It is a simple feed-forward network. It takes the input, feeds it through several layers one after the other, and then finally gives the output.

With the help of PyTorch, we can use the following steps for typical training procedure for a neural network −

- Define the neural network that has some learnable parameters (or weights).
- Iterate over a dataset of inputs.
- Process input through the network.
- Compute the loss (how far is the output from being correct).
- Propagate gradients back into the networks parameters.
- Update the weights of the network, typically using a simple update as given below
```
rule: weight = weight -learning_rate * gradient
```

---

## 5. Universal Workflow of Machine Learning

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_universal_workflow_of_machine_learning.htm](https://www.tutorialspoint.com/pytorch/pytorch_universal_workflow_of_machine_learning.htm)*

---

---
[Previous](/pytorch/pytorch_neural_network_basics.htm)[Quiz](/pytorch/quiz_on_pytorch_universal_workflow_of_machine_learning.htm)[Next](/pytorch/pytorch_machine_learning_vs_deep_learning.htm)
Artificial Intelligence is trending nowadays to a greater extent. Machine learning and deep learning constitutes artificial intelligence. The Venn diagram mentioned below explains the relationship of machine learning and deep learning.
![relationship machine learning](/pytorch/images/relationship_machine_learning.jpg)
## Machine Learning

Machine learning is the art of science which allows computers to act as per the designed and programmed algorithms. Many researchers think machine learning is the best way to make progress towards human-level AI. It includes various types of patterns like −

- Supervised Learning Pattern
- Unsupervised Learning Pattern
## Deep Learning

Deep learning is a subfield of machine learning where concerned algorithms are inspired by the structure and function of the brain called Artificial Neural Networks.

Deep learning has gained much importance through supervised learning or learning from labelled data and algorithms. Each algorithm in deep learning goes through same process. It includes hierarchy of nonlinear transformation of input and uses to create a statistical model as output.

Machine learning process is defined using following steps −

- Identifies relevant data sets and prepares them for analysis.
- Chooses the type of algorithm to use.
- Builds an analytical model based on the algorithm used.
- Trains the model on test data sets, revising it as needed.
- Runs the model to generate test scores.

---

## 6. PyTorch - Machine Learning vs. Deep Learning

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_machine_learning_vs_deep_learning.htm](https://www.tutorialspoint.com/pytorch/pytorch_machine_learning_vs_deep_learning.htm)*

---

---
[Previous](/pytorch/pytorch_universal_workflow_of_machine_learning.htm)[Quiz](/pytorch/quiz_on_pytorch_machine_learning_vs_deep_learning.htm)[Next](/pytorch/pytorch_implementing_first_neural_network.htm)
In this chapter, we will discuss the major difference between Machine and Deep learning concepts.

## Amount of Data

Machine learning works with different amounts of data and is mainly used for small amounts of data. Deep learning on the other hand works efficiently if the amount of data increases rapidly. The following diagram depicts the working of machine learning and deep learning with respect to amount of data −
![Amount Data](/pytorch/images/amount_data.jpg)
## Hardware Dependencies

Deep learning algorithms are designed to heavily depend on high end machines on a contrary to traditional machine learning algorithms. Deep learning algorithms perform a large amount of matrix multiplication operations which requires a huge hardware support.

## Feature Engineering

Feature engineering is the process of putting domain knowledge into specified features to reduce the complexity of data and make patterns which are visible to learning algorithms.

For instance, traditional machine learning patterns focusses on pixels and other attributes needed for feature engineering process. Deep learning algorithms focusses on high level features from data. It reduces the task of developing new feature extractor for every new problem.

---

## 7. PyTorch - Implementing First Neural Network

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_implementing_first_neural_network.htm](https://www.tutorialspoint.com/pytorch/pytorch_implementing_first_neural_network.htm)*

---

---
[Previous](/pytorch/pytorch_machine_learning_vs_deep_learning.htm)[Quiz](/pytorch/quiz_on_pytorch_implementing_first_neural_network.htm)[Next](/pytorch/pytorch_neural_networks_to_functional_blocks.htm)
PyTorch includes a special feature of creating and implementing neural networks. In this chapter, we will create a simple neural network with one hidden layer developing a single output unit.

We shall use following steps to implement the first neural network using PyTorch −

## Step 1

First, we need to import the PyTorch library using the below command −

```
import torch 
import torch.nn as nn
```

## Step 2

Define all the layers and the batch size to start executing the neural network as shown below −

```
# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 10, 5, 1, 10
```

## Step 3

As neural network includes a combination of input data to get the respective output data, we will be following the same procedure as given below −

```
# Create dummy input and target tensors (data)
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], 
[1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
```

## Step 4

Create a sequential model with the help of in-built functions. Using the below lines of code, create a sequential model −

```
# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h),
   nn.ReLU(),
   nn.Linear(n_h, n_out),
   nn.Sigmoid())
```

## Step 5

Construct the loss function with the help of Gradient Descent optimizer as shown below −

```
Construct the loss function
criterion = torch.nn.MSELoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
```

## Step 6

Implement the gradient descent model with the iterating loop with the given lines of code −

```
# Gradient Descent
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x)

   # Compute and print loss
   loss = criterion(y_pred, y)
   print('epoch: ', epoch,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()
```

## Step 7

The output generated is as follows −

```
epoch: 0 loss: 0.2545787990093231
epoch: 1 loss: 0.2545052170753479
epoch: 2 loss: 0.254431813955307
epoch: 3 loss: 0.25435858964920044
epoch: 4 loss: 0.2542854845523834
epoch: 5 loss: 0.25421255826950073
epoch: 6 loss: 0.25413978099823
epoch: 7 loss: 0.25406715273857117
epoch: 8 loss: 0.2539947032928467
epoch: 9 loss: 0.25392240285873413
epoch: 10 loss: 0.25385022163391113
epoch: 11 loss: 0.25377824902534485
epoch: 12 loss: 0.2537063956260681
epoch: 13 loss: 0.2536346912384033
epoch: 14 loss: 0.25356316566467285
epoch: 15 loss: 0.25349172949790955
epoch: 16 loss: 0.25342053174972534
epoch: 17 loss: 0.2533493936061859
epoch: 18 loss: 0.2532784342765808
epoch: 19 loss: 0.25320762395858765
epoch: 20 loss: 0.2531369626522064
epoch: 21 loss: 0.25306645035743713
epoch: 22 loss: 0.252996027469635
epoch: 23 loss: 0.2529257833957672
epoch: 24 loss: 0.25285571813583374
epoch: 25 loss: 0.25278574228286743
epoch: 26 loss: 0.25271597504615784
epoch: 27 loss: 0.25264623761177063
epoch: 28 loss: 0.25257670879364014
epoch: 29 loss: 0.2525072991847992
epoch: 30 loss: 0.2524380087852478
epoch: 31 loss: 0.2523689270019531
epoch: 32 loss: 0.25229987502098083
epoch: 33 loss: 0.25223103165626526
epoch: 34 loss: 0.25216227769851685
epoch: 35 loss: 0.252093642950058
epoch: 36 loss: 0.25202515721321106
epoch: 37 loss: 0.2519568204879761
epoch: 38 loss: 0.251888632774353
epoch: 39 loss: 0.25182053446769714
epoch: 40 loss: 0.2517525553703308
epoch: 41 loss: 0.2516847252845764
epoch: 42 loss: 0.2516169846057892
epoch: 43 loss: 0.2515493929386139
epoch: 44 loss: 0.25148195028305054
epoch: 45 loss: 0.25141456723213196
epoch: 46 loss: 0.2513473629951477
epoch: 47 loss: 0.2512802183628082
epoch: 48 loss: 0.2512132525444031
epoch: 49 loss: 0.2511464059352875
```

---

## 8. PyTorch - Neural Networks to Functional Blocks

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_neural_networks_to_functional_blocks.htm](https://www.tutorialspoint.com/pytorch/pytorch_neural_networks_to_functional_blocks.htm)*

---

---
[Previous](/pytorch/pytorch_implementing_first_neural_network.htm)[Quiz](/pytorch/quiz_on_pytorch_neural_networks_to_functional_blocks.htm)[Next](/pytorch/pytorch_terminologies.htm)
Training a deep learning algorithm involves the following steps −

- Building a data pipeline
- Building a network architecture
- Evaluating the architecture using a loss function
- Optimizing the network architecture weights using an optimization algorithm
Training a specific deep learning algorithm is the exact requirement of converting a neural network to functional blocks as shown below −
![Functional Blocks](/pytorch/images/functional_blocks.jpg)
With respect to the above diagram, any deep learning algorithm involves getting the input data, building the respective architecture which includes a bunch of layers embedded in them.

If you observe the above diagram, the accuracy is evaluated using a loss function with respect to optimization of the weights of neural network.

---

## 9. PyTorch - Terminologies

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_terminologies.htm](https://www.tutorialspoint.com/pytorch/pytorch_terminologies.htm)*

---

---
[Previous](/pytorch/pytorch_neural_networks_to_functional_blocks.htm)[Quiz](/pytorch/quiz_on_pytorch_terminologies.htm)[Next](/pytorch/pytorch_loading_data.htm)
In this chapter, we will discuss some of the most commonly used terms in PyTorch.

## PyTorch NumPy

A PyTorch tensor is identical to a NumPy array. A tensor is an n-dimensional array and with respect to PyTorch, it provides many functions to operate on these tensors.

PyTorch tensors usually utilize GPUs to accelerate their numeric computations. These tensors which are created in PyTorch can be used to fit a two-layer network to random data. The user can manually implement the forward and backward passes through the network.

## Variables and Autograd

When using autograd, the forward pass of your network will define a
**computational graph**− nodes in the graph will be Tensors, and edges will be functions that produce output Tensors from input Tensors.
PyTorch Tensors can be created as variable objects where a variable represents a node in computational graph.

## Dynamic Graphs

Static graphs are nice because user can optimize the graph up front. If programmers are re-using same graph over and over, then this potentially costly up-front optimization can be maintained as the same graph is rerun over and over.

The major difference between them is that Tensor Flows computational graphs are static and PyTorch uses dynamic computational graphs.

## Optim Package

The optim package in PyTorch abstracts the idea of an optimization algorithm which is implemented in many ways and provides illustrations of commonly used optimization algorithms. This can be called within the import statement.

## Multiprocessing

Multiprocessing supports the same operations, so that all tensors work on multiple processors. The queue will have their data moved into shared memory and will only send a handle to another process.

---

## 10. PyTorch - Loading Data

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_loading_data.htm](https://www.tutorialspoint.com/pytorch/pytorch_loading_data.htm)*

---

---
[Previous](/pytorch/pytorch_terminologies.htm)[Quiz](/pytorch/quiz_on_pytorch_loading_data.htm)[Next](/pytorch/pytorch_linear_regression.htm)
PyTorch includes a package called torchvision which is used to load and prepare the dataset. It includes two basic functions namely Dataset and DataLoader which helps in transformation and loading of dataset.

## Dataset

Dataset is used to read and transform a datapoint from the given dataset. The basic syntax to implement is mentioned below −

```
trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
   download = True, transform = transform)
```

DataLoader is used to shuffle and batch data. It can be used to load the data in parallel with multiprocessing workers.

```
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
   shuffle = True, num_workers = 2)
```

### Example: Loading CSV File

We use the Python package Panda to load the csv file. The original file has the following format: (image name, 68 landmarks - each landmark has a x, y coordinates).

```
landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)
```

---

## 11. PyTorch - Linear Regression

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_linear_regression.htm](https://www.tutorialspoint.com/pytorch/pytorch_linear_regression.htm)*

---

---

## 12. PyTorch - Convolutional Neural Network

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_convolutional_neural_network.htm](https://www.tutorialspoint.com/pytorch/pytorch_convolutional_neural_network.htm)*

---

---
[Previous](/pytorch/pytorch_linear_regression.htm)[Quiz](/pytorch/quiz_on_pytorch_convolutional_neural_network.htm)[Next](/pytorch/pytorch_recurrent_neural_network.htm)
Deep learning is a division of machine learning and is considered as a crucial step taken by researchers in recent decades. The examples of deep learning implementation include applications like image recognition and speech recognition.

The two important types of deep neural networks are given below −

- Convolutional Neural Networks
- Recurrent Neural Networks.
In this chapter, we will be focusing on the first type, i.e., Convolutional Neural Networks (CNN).

## Convolutional Neural Networks

Convolutional Neural networks are designed to process data through multiple layers of arrays. This type of neural networks are used in applications like image recognition or face recognition.

The primary difference between CNN and any other ordinary neural network is that CNN takes input as a two dimensional array and operates directly on the images rather than focusing on feature extraction which other neural networks focus on.

The dominant approach of CNN includes solution for problems of recognition. Top companies like Google and Facebook have invested in research and development projects of recognition projects to get activities done with greater speed.

Every convolutional neural network includes three basic ideas −

- Local respective fields
- Convolution
- Pooling
Let us understand each of these terminologies in detail.

## Local Respective Fields

CNN utilize spatial correlations that exists within the input data. Each in the concurrent layers of neural networks connects of some input neurons. This specific region is called Local Receptive Field. It only focusses on hidden neurons. The hidden neuron will process the input data inside the mentioned field not realizing the changes outside the specific boundary.

The diagram representation of generating local respective fields is mentioned below −
![Local Respective Fields](/pytorch/images/local_respective_fields.jpg)
## Convolution

In the above figure, we observe that each connection learns a weight of hidden neuron with an associated connection with movement from one layer to another. Here, individual neurons perform a shift from time to time. This process is called convolution.

The mapping of connections from the input layer to the hidden feature map is defined as shared weights and bias included is called shared bias.

## Pooling

Convolutional neural networks use pooling layers which are positioned immediately after CNN declaration. It takes the input from the user as a feature map which comes out convolutional networks and prepares a condensed feature map. Pooling layers help in creating layers with neurons of previous layers.

## Implementation of PyTorch

Following steps are used to create a Convolutional Neural Network using PyTorch.

### Step 1

Import the necessary packages for creating a simple neural network.

```
from torch.autograd import Variable
import torch.nn.functional as F
```

### Step 2

Create a class with batch representation of convolutional neural network. Our batch shape for input x is with dimension of (3, 32, 32).

```
class SimpleCNN(torch.nn.Module):
   def __init__(self):
      super(SimpleCNN, self).__init__()
      #Input channels = 3, output channels = 18
      self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
      self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
      #4608 input features, 64 output features (see sizing flow below)
      self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
      #64 input features, 10 output features for our 10 defined classes
      self.fc2 = torch.nn.Linear(64, 10)
```

### Step 3

Compute the activation of the first convolution size changes from (3, 32, 32) to (18, 32, 32).

Size of the dimension changes from (18, 32, 32) to (18, 16, 16). Reshape data dimension of the input layer of the neural net due to which size changes from (18, 16, 16) to (1, 4608).

Recall that -1 infers this dimension from the other given dimension.

```
def forward(self, x):
   x = F.relu(self.conv1(x))
   x = self.pool(x)
   x = x.view(-1, 18 * 16 *16)
   x = F.relu(self.fc1(x))
   #Computes the second fully connected layer (activation applied later)
   #Size changes from (1, 64) to (1, 10)
   x = self.fc2(x)
   return(x)
```

---

## 13. PyTorch - Recurrent Neural Network

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_recurrent_neural_network.htm](https://www.tutorialspoint.com/pytorch/pytorch_recurrent_neural_network.htm)*

---

---

## 14. PyTorch - Datasets

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_datasets.htm](https://www.tutorialspoint.com/pytorch/pytorch_datasets.htm)*

---

---
[Previous](/pytorch/pytorch_recurrent_neural_network.htm)[Quiz](/pytorch/quiz_on_pytorch_datasets.htm)[Next](/pytorch/pytorch_introduction_to_convents.htm)
In this chapter, we will focus more on
**torchvision.datasets**and its various types. PyTorch includes following dataset loaders −
- MNIST
- COCO (Captioning and Detection)
Dataset includes majority of two types of functions given below −

- **Transform**− a function that takes in an image and returns a modified version of standard stuff. These can be composed together with transforms.
- **Target_transform**− a function that takes the target and transforms it. For example, takes in the caption string and returns a tensor of world indices.
## MNIST

The following is the sample code for MNIST dataset −

```
dset.MNIST(root, train = TRUE, transform = NONE, 
target_transform = None, download = FALSE)
```

The parameters are as follows −

- **root**− root directory of the dataset where processed data exist.
- **train**− True = Training set, False = Test set
- **download**− True = downloads the dataset from the internet and puts it in the root.
## COCO

This requires the COCO API to be installed. The following example is used to demonstrate the COCO implementation of dataset using PyTorch −

```
import torchvision.dataset as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root =  dir where images are, 
annFile = json annotation file,
transform = transforms.ToTensor())
print(Number of samples: , len(cap))
print(target)
```

The output achieved is as follows −

```
Number of samples: 82783
Image Size: (3L, 427L, 640L)
```

---

## 15. PyTorch - Introduction to Convents

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_introduction_to_convents.htm](https://www.tutorialspoint.com/pytorch/pytorch_introduction_to_convents.htm)*

---

---

## 16. PyTorch - Training a Convent from Scratch

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_training_a_convent_from_scratch.htm](https://www.tutorialspoint.com/pytorch/pytorch_training_a_convent_from_scratch.htm)*

---

---
[Previous](/pytorch/pytorch_introduction_to_convents.htm)[Quiz](/pytorch/quiz_on_pytorch_training_a_convent_from_scratch.htm)[Next](/pytorch/pytorch_feature_extraction_in_convents.htm)
In this chapter, we will focus on creating a convent from scratch. This infers in creating the respective convent or sample neural network with torch.

## Step 1

Create a necessary class with respective parameters. The parameters include weights with random value.

```
class Neural_Network(nn.Module):
   def __init__(self, ):
      super(Neural_Network, self).__init__()
      self.inputSize = 2
      self.outputSize = 1
      self.hiddenSize = 3
      # weights
      self.W1 = torch.randn(self.inputSize, 
      self.hiddenSize) # 3 X 2 tensor
      self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
```

## Step 2

Create a feed forward pattern of function with sigmoid functions.

```
def forward(self, X):
   self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" 
   does not broadcast in PyTorch
   self.z2 = self.sigmoid(self.z) # activation function
   self.z3 = torch.matmul(self.z2, self.W2)
   o = self.sigmoid(self.z3) # final activation 
   function
   return o
   def sigmoid(self, s):
      return 1 / (1 + torch.exp(-s))
   def sigmoidPrime(self, s):
      # derivative of sigmoid
      return s * (1 - s)
   def backward(self, X, y, o):
      self.o_error = y - o # error in output
      self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
      self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
      self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
      self.W1 + = torch.matmul(torch.t(X), self.z2_delta)
      self.W2 + = torch.matmul(torch.t(self.z2), self.o_delta)
```

## Step 3

Create a training and prediction model as mentioned below −

```
def train(self, X, y):
   # forward + backward pass for training
   o = self.forward(X)
   self.backward(X, y, o)
def saveWeights(self, model):
   # Implement PyTorch internal storage functions
   torch.save(model, "NN")
   # you can reload model with all the weights and so forth with:
   # torch.load("NN")
def predict(self):
   print ("Predicted data based on trained weights: ")
   print ("Input (scaled): \n" + str(xPredicted))
   print ("Output: \n" + str(self.forward(xPredicted)))
```

---

## 17. PyTorch - Feature Extraction in Convents

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_feature_extraction_in_convents.htm](https://www.tutorialspoint.com/pytorch/pytorch_feature_extraction_in_convents.htm)*

---

---

## 18. PyTorch - Visualization of Convents

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_visualization_of_convents.htm](https://www.tutorialspoint.com/pytorch/pytorch_visualization_of_convents.htm)*

---

---

## 19. PyTorch - Sequence Processing with Convents

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_sequence_processing_with_convents.htm](https://www.tutorialspoint.com/pytorch/pytorch_sequence_processing_with_convents.htm)*

---

---
[Previous](/pytorch/pytorch_visualization_of_convents.htm)[Quiz](/pytorch/quiz_on_pytorch_sequence_processing_with_convents.htm)[Next](/pytorch/pytorch_word_embedding.htm)
In this chapter, we propose an alternative approach which instead relies on a single 2D convolutional neural network across both sequences. Each layer of our network re-codes source tokens on the basis of the output sequence produced so far. Attention-like properties are therefore pervasive throughout the network.

Here, we will focus on
**creating the sequential network with specific pooling from the values included in dataset**. This process is also best applied in Image Recognition Module.![Sequential Network](/pytorch/images/sequential_network.jpg)
Following steps are used to create a sequence processing model with convents using PyTorch −

## Step 1

Import the necessary modules for performance of sequence processing using convents.

```
import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
import numpy as np
```

## Step 2

Perform the necessary operations to create a pattern in respective sequence using the below code −

```
batch_size = 128 
num_classes = 10 
epochs = 12
# input image dimensions 
img_rows, img_cols = 28, 28
# the data, split between train and test sets 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1) 
x_test = x_test.reshape(10000,28,28,1)
print('x_train shape:', x_train.shape) 
print(x_train.shape[0], 'train samples') 
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes) 
y_test = keras.utils.to_categorical(y_test, num_classes)
```

## Step 3

Compile the model and fit the pattern in the mentioned conventional neural network model as shown below −

```
model.compile(loss = 
keras.losses.categorical_crossentropy, 
optimizer = keras.optimizers.Adadelta(), metrics = 
['accuracy'])
model.fit(x_train, y_train, 
batch_size = batch_size, epochs = epochs, 
verbose = 1, validation_data = (x_test, y_test)) 
score = model.evaluate(x_test, y_test, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])
```

The output generated is as follows −
![Neural network model](/pytorch/images/neural_network_model.jpg)

---

## 20. PyTorch - Word Embedding

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_word_embedding.htm](https://www.tutorialspoint.com/pytorch/pytorch_word_embedding.htm)*

---

---
[Previous](/pytorch/pytorch_sequence_processing_with_convents.htm)[Quiz](/pytorch/quiz_on_pytorch_word_embedding.htm)[Next](/pytorch/pytorch_recursive_neural_networks.htm)
In this chapter, we will understand the famous word embedding model − word2vec. Word2vec model is used to produce word embedding with the help of group of related models. Word2vec model is implemented with pure C-code and the gradient are computed manually.

The implementation of word2vec model in PyTorch is explained in the below steps −

## Step 1

Implement the libraries in word embedding as mentioned below −

```
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
```

## Step 2

Implement the Skip Gram Model of word embedding with the class called word2vec. It includes
**emb_size, emb_dimension, u_embedding, v_embedding**type of attributes.
```
class SkipGramModel(nn.Module):
   def __init__(self, emb_size, emb_dimension):
      super(SkipGramModel, self).__init__()
      self.emb_size = emb_size
      self.emb_dimension = emb_dimension
      self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
      self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse = True)
      self.init_emb()
   def init_emb(self):
      initrange = 0.5 / self.emb_dimension
      self.u_embeddings.weight.data.uniform_(-initrange, initrange)
      self.v_embeddings.weight.data.uniform_(-0, 0)
   def forward(self, pos_u, pos_v, neg_v):
      emb_u = self.u_embeddings(pos_u)
      emb_v = self.v_embeddings(pos_v)
      score = torch.mul(emb_u, emb_v).squeeze()
      score = torch.sum(score, dim = 1)
      score = F.logsigmoid(score)
      neg_emb_v = self.v_embeddings(neg_v)
      neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
      neg_score = F.logsigmoid(-1 * neg_score)
      return -1 * (torch.sum(score)+torch.sum(neg_score))
   def save_embedding(self, id2word, file_name, use_cuda):
      if use_cuda:
         embedding = self.u_embeddings.weight.cpu().data.numpy()
      else:
         embedding = self.u_embeddings.weight.data.numpy()
      fout = open(file_name, 'w')
      fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
      for wid, w in id2word.items():
         e = embedding[wid]
         e = ' '.join(map(lambda x: str(x), e))
         fout.write('%s %s\n' % (w, e))
def test():
   model = SkipGramModel(100, 100)
   id2word = dict()
   for i in range(100):
      id2word[i] = str(i)
   model.save_embedding(id2word)
```

## Step 3

Implement the main method to get the word embedding model displayed in proper way.

```
if __name__  ==  '__main__':
   test()
```

---

## 21. PyTorch - Recursive Neural Networks

*Source: [https://www.tutorialspoint.com/pytorch/pytorch_recursive_neural_networks.htm](https://www.tutorialspoint.com/pytorch/pytorch_recursive_neural_networks.htm)*

---

---


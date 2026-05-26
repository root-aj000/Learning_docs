# MACHINE LEARNING & DEEP LEARNING
### Complete Detailed Index

---

## PHASE 3 — MACHINE LEARNING FUNDAMENTALS

---

### 3.1 Introduction to ML

- 3.1.1 What is Machine Learning
- 3.1.2 How Machine Learns from Data
- 3.1.3 Difference Between AI, ML, DL
- 3.1.4 Types of ML
  - Supervised Learning
  - Unsupervised Learning
  - Semi Supervised Learning
  - Self Supervised Learning
  - Reinforcement Learning
- 3.1.5 When to Use ML
- 3.1.6 ML Workflow Overview
  - Data Collection
  - Data Preprocessing
  - Model Training
  - Model Evaluation
  - Model Deployment

---

### 3.2 Supervised Learning

---

#### 3.2.1 Regression

- What is Regression
- Simple Linear Regression
  - Equation of a Line
  - Slope and Intercept
  - How Model Learns
- Multiple Linear Regression
- Polynomial Regression
- Ridge Regression
  - L2 Regularization
- Lasso Regression
  - L1 Regularization
- ElasticNet Regression
- Regression Concepts
  - Bias Variance Tradeoff
  - Overfitting
  - Underfitting
  - Regularization
- Loss Functions
  - Mean Squared Error
  - Mean Absolute Error
  - Root Mean Squared Error
- Evaluation Metrics
  - MSE
  - RMSE
  - MAE
  - R Squared
  - Adjusted R Squared

---

#### 3.2.2 Classification

- What is Classification
- Binary Classification
- Multiclass Classification
- Algorithms
  - Logistic Regression
    - Sigmoid Function
    - Decision Boundary
    - Log Loss
  - K Nearest Neighbors
    - How KNN Works
    - Choosing K Value
    - Distance Metrics
      - Euclidean
      - Manhattan
      - Minkowski
  - Naive Bayes
    - Bayes Theorem
    - Gaussian Naive Bayes
    - Multinomial Naive Bayes
    - Bernoulli Naive Bayes
  - Decision Trees
    - How Trees Split Data
    - Entropy
    - Information Gain
    - Gini Impurity
    - Tree Pruning
    - Overfitting in Trees
  - Random Forest
    - What is Bagging
    - How Random Forest Works
    - Feature Importance
  - Support Vector Machine
    - Hyperplane
    - Support Vectors
    - Margin
    - Kernel Trick
      - Linear Kernel
      - RBF Kernel
      - Polynomial Kernel
  - Gradient Boosting
    - How Boosting Works
    - Weak Learners
    - Residual Learning
  - XGBoost
    - How XGBoost Improves Boosting
    - Regularization in XGBoost
    - Hyperparameters
  - LightGBM
    - Leaf Wise Growth
    - Speed Advantages
  - CatBoost
    - Handling Categorical Features
- Classification Concepts
  - Confusion Matrix
    - True Positive
    - True Negative
    - False Positive
    - False Negative
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC Curve
  - AUC Score
  - Imbalanced Datasets
    - Oversampling
    - Undersampling
    - SMOTE

---

### 3.3 Unsupervised Learning

---

#### 3.3.1 Clustering

- What is Clustering
- K Means Clustering
  - How K Means Works
  - Choosing K
  - Elbow Method
  - Silhouette Score
- Hierarchical Clustering
  - Agglomerative
  - Divisive
  - Dendrogram
- DBSCAN
  - Core Points
  - Border Points
  - Noise Points
  - Epsilon and MinPoints
- Gaussian Mixture Models
  - Expectation Maximization
  - Soft Clustering

---

#### 3.3.2 Dimensionality Reduction

- Why Reduce Dimensions
- Curse of Dimensionality
- PCA
  - Variance Explained
  - Principal Components
  - Covariance Matrix
  - Eigenvectors in PCA
- t SNE
  - How t SNE Works
  - When to Use t SNE
- UMAP
  - Difference from t SNE
  - When to Use UMAP

---

#### 3.3.3 Association Rule Learning

- What is Association Rule Learning
- Support
- Confidence
- Lift
- Apriori Algorithm
- FP Growth Algorithm

---

#### 3.3.4 Anomaly Detection

- What is Anomaly Detection
- Statistical Methods
- Isolation Forest
- One Class SVM
- Autoencoders for Anomaly Detection

---

### 3.4 Model Evaluation & Selection

---

- 3.4.1 Train Test Split
- 3.4.2 Validation Set
- 3.4.3 Cross Validation
  - K Fold Cross Validation
  - Stratified K Fold
  - Leave One Out
- 3.4.4 Hyperparameter Tuning
  - What is a Hyperparameter
  - Grid Search
  - Random Search
  - Bayesian Optimization
- 3.4.5 Bias Variance Tradeoff Deep Dive
- 3.4.6 Learning Curves
- 3.4.7 Validation Curves

---

### 3.5 Ensemble Learning

---

- 3.5.1 What is Ensemble Learning
- 3.5.2 Bagging
  - How Bagging Works
  - Bootstrap Sampling
  - Random Forest as Bagging
- 3.5.3 Boosting
  - AdaBoost
  - Gradient Boosting
  - XGBoost
  - LightGBM
- 3.5.4 Stacking
  - Base Models
  - Meta Model
  - How Stacking Works
- 3.5.5 Voting Classifiers
  - Hard Voting
  - Soft Voting
- 3.5.6 Blending

---

### 3.6 Time Series ML

---

- 3.6.1 What is Time Series Data
- 3.6.2 Time Series Components
  - Trend
  - Seasonality
  - Noise
  - Cyclicity
- 3.6.3 Stationarity
  - What is Stationarity
  - ADF Test
  - Making Series Stationary
- 3.6.4 Autocorrelation
  - ACF Plot
  - PACF Plot
- 3.6.5 Traditional Models
  - ARIMA
    - AR Component
    - I Component
    - MA Component
  - SARIMA
  - Exponential Smoothing
- 3.6.6 Modern Approaches
  - Prophet
  - ML for Time Series
  - LSTM for Time Series
- 3.6.7 Evaluation Metrics for Time Series
  - MAE
  - MAPE
  - RMSE

---
---

## PHASE 4 — DEEP LEARNING

---

### 4.1 Neural Network Fundamentals

---

- 4.1.1 What is a Neural Network
- 4.1.2 Biological Neuron vs Artificial Neuron
- 4.1.3 Perceptron
  - Single Layer Perceptron
  - How Perceptron Learns
  - Perceptron Limitations
- 4.1.4 Multilayer Perceptron
  - Input Layer
  - Hidden Layers
  - Output Layer
- 4.1.5 Activation Functions
  - Why Activation Functions
  - Sigmoid
  - Tanh
  - ReLU
  - Leaky ReLU
  - ELU
  - Softmax
  - Linear
  - When to Use Which
- 4.1.6 Forward Propagation
  - How Data Flows Forward
  - Matrix Multiplications
  - Layer by Layer Calculation
- 4.1.7 Loss Functions
  - Mean Squared Error
  - Binary Cross Entropy
  - Categorical Cross Entropy
  - Huber Loss
  - When to Use Which
- 4.1.8 Backpropagation
  - What is Backpropagation
  - Chain Rule in Backprop
  - Computing Gradients
  - Updating Weights
- 4.1.9 Gradient Descent
  - Batch Gradient Descent
  - Stochastic Gradient Descent
  - Mini Batch Gradient Descent
- 4.1.10 Optimizers
  - SGD with Momentum
  - RMSProp
  - Adam
  - AdaGrad
  - AdamW
  - Comparison of Optimizers
- 4.1.11 Weight Initialization
  - Zero Initialization Problem
  - Random Initialization
  - Xavier Initialization
  - He Initialization
- 4.1.12 Batch Normalization
  - Why Batch Normalization
  - How it Works
  - Where to Apply
- 4.1.13 Dropout
  - What is Dropout
  - How Dropout Prevents Overfitting
  - Dropout Rate
- 4.1.14 Regularization in Deep Learning
  - L1 Regularization
  - L2 Regularization
  - Early Stopping
- 4.1.15 Hyperparameters in Neural Networks
  - Learning Rate
  - Batch Size
  - Number of Layers
  - Number of Neurons
  - Epochs

---

### 4.2 Deep Learning Frameworks

---

#### 4.2.1 PyTorch

- What is PyTorch
- Tensors
  - Creating Tensors
  - Tensor Operations
  - Tensor on GPU
- Autograd
  - Automatic Differentiation
  - Computational Graph
- Building Models
  - nn.Module
  - Defining Layers
  - Forward Method
- Training Loop
  - DataLoader
  - Loss Calculation
  - Optimizer Step
  - Zero Grad
- Saving and Loading Models
- PyTorch Lightning Basics

---

#### 4.2.2 TensorFlow & Keras

- What is TensorFlow
- Keras as High Level API
- Sequential API
- Functional API
- Model Subclassing
- Model Compilation
  - Loss
  - Optimizer
  - Metrics
- Model Training
  - fit method
  - Callbacks
    - EarlyStopping
    - ModelCheckpoint
    - ReduceLROnPlateau
- Saving and Loading Models

---

### 4.3 Convolutional Neural Networks

---

- 4.3.1 Why CNNs for Images
- 4.3.2 Convolution Operation
  - Filter / Kernel
  - Stride
  - Padding
  - Feature Map
- 4.3.3 Pooling Layers
  - Max Pooling
  - Average Pooling
  - Global Average Pooling
- 4.3.4 Fully Connected Layers
- 4.3.5 CNN Architecture Design
- 4.3.6 Transfer Learning
  - What is Transfer Learning
  - Feature Extraction
  - Fine Tuning
  - When to Use Transfer Learning
- 4.3.7 Data Augmentation
  - Flipping
  - Rotation
  - Zoom
  - Color Jitter
- 4.3.8 Famous CNN Architectures
  - LeNet
  - AlexNet
  - VGG
  - ResNet
    - Residual Connections
    - Skip Connections
  - InceptionNet
  - EfficientNet
  - MobileNet
- 4.3.9 Object Detection
  - YOLO Family
  - SSD
  - Faster RCNN
- 4.3.10 Image Segmentation
  - Semantic Segmentation
  - Instance Segmentation
  - U Net Architecture

---

### 4.4 Recurrent Neural Networks & Sequence Models

---

- 4.4.1 What is Sequential Data
- 4.4.2 Why Normal Networks Fail on Sequences
- 4.4.3 Recurrent Neural Network
  - Hidden State
  - How RNN Processes Sequences
  - Unrolling RNN
- 4.4.4 Problems with RNN
  - Vanishing Gradient Problem
  - Exploding Gradient Problem
- 4.4.5 LSTM
  - Cell State
  - Forget Gate
  - Input Gate
  - Output Gate
  - How LSTM Solves Vanishing Gradient
- 4.4.6 GRU
  - Update Gate
  - Reset Gate
  - Difference from LSTM
- 4.4.7 Bidirectional RNN
- 4.4.8 Seq2Seq Models
  - Encoder
  - Decoder
  - Context Vector
  - Applications
    - Translation
    - Summarization
- 4.4.9 Attention Mechanism
  - Why Attention
  - How Attention Works
  - Bahdanau Attention
  - Self Attention

---

### 4.5 Transformers

---

- 4.5.1 Why Transformers Replaced RNNs
- 4.5.2 Self Attention
  - Query
  - Key
  - Value
  - Attention Score Calculation
  - Scaled Dot Product Attention
- 4.5.3 Multi Head Attention
  - Why Multiple Heads
  - How Heads Work Together
- 4.5.4 Positional Encoding
  - Why Position Matters
  - Sine Cosine Encoding
- 4.5.5 Feed Forward Network in Transformer
- 4.5.6 Layer Normalization
- 4.5.7 Encoder Architecture
  - Self Attention Layer
  - Feed Forward Layer
  - Residual Connections
- 4.5.8 Decoder Architecture
  - Masked Self Attention
  - Cross Attention
  - Feed Forward Layer
- 4.5.9 Encoder Only Models
  - BERT
    - Masked Language Modeling
    - Next Sentence Prediction
    - Fine Tuning BERT
- 4.5.10 Decoder Only Models
  - GPT Family
    - GPT 1
    - GPT 2
    - GPT 3
    - GPT 4
    - How GPT Generates Text
- 4.5.11 Encoder Decoder Models
  - T5
  - BART
- 4.5.12 Vision Transformer
  - Patch Embeddings
  - How ViT Works
- 4.5.13 Scaling Laws
  - Parameters vs Performance
  - Data vs Performance

---

### 4.6 Generative AI

---

- 4.6.1 What is Generative AI
- 4.6.2 Large Language Models
  - What is an LLM
  - How LLMs are Trained
  - Pretraining
  - Fine Tuning
  - RLHF
- 4.6.3 Prompt Engineering
  - Zero Shot Prompting
  - Few Shot Prompting
  - Chain of Thought
  - System Prompts
- 4.6.4 Fine Tuning LLMs
  - Full Fine Tuning
  - LoRA
  - QLoRA
  - PEFT
- 4.6.5 Embeddings
  - What are Embeddings
  - Word Embeddings
  - Sentence Embeddings
  - How to Use Embeddings
- 4.6.6 Vector Databases
  - What is a Vector Database
  - Similarity Search
  - FAISS
  - Pinecone
  - ChromaDB
- 4.6.7 RAG Systems
  - What is RAG
  - Retrieval Component
  - Generation Component
  - Building RAG Pipeline
- 4.6.8 Generative Adversarial Networks
  - Generator
  - Discriminator
  - Training GANs
  - Mode Collapse
  - Applications
- 4.6.9 Variational Autoencoders
  - Encoder
  - Latent Space
  - Decoder
  - Reparameterization Trick
- 4.6.10 Diffusion Models
  - Forward Diffusion Process
  - Reverse Diffusion Process
  - DDPM
  - Stable Diffusion
  - How Images are Generated

---

## CORRECT LEARNING SEQUENCE

```
3.1 Introduction to ML
        ↓
3.2 Supervised Learning
   Regression → Classification
        ↓
3.3 Unsupervised Learning
   Clustering → Dimensionality Reduction
        ↓
3.4 Model Evaluation
        ↓
3.5 Ensemble Learning
        ↓
3.6 Time Series
        ↓
4.1 Neural Network Fundamentals
        ↓
4.2 DL Frameworks
        ↓
4.3 CNNs
        ↓
4.4 RNNs & Sequence Models
        ↓
4.5 Transformers
        ↓
4.6 Generative AI
```
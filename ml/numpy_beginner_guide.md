# NumPy Beginner-Friendly Guide

*A simplified guide to NumPy for new programmers*

---

## Table of Contents

1. [What is NumPy?](#1-what-is-numpy)
2. [Getting Started](#2-getting-started)
3. [Creating Arrays](#3-creating-arrays)
4. [Understanding Array Properties](#4-understanding-array-properties)
5. [Indexing and Slicing](#5-indexing-and-slicing)
6. [Array Operations](#6-array-operations)
7. [Broadcasting](#7-broadcasting)
8. [Mathematical Functions](#8-mathematical-functions)
9. [Statistical Functions](#9-statistical-functions)
10. [Linear Algebra Basics](#10-linear-algebra-basics)
11. [Random Numbers](#11-random-numbers)
12. [Reshaping and Manipulation](#12-reshaping-and-manipulation)
13. [Working with Files](#13-working-with-files)
14. [Common Patterns and Tips](#14-common-patterns-and-tips)

---

## 1. What is NumPy?

### The Problem with Python Lists

Imagine you have 1 million numbers and you want to add 5 to each one. With Python lists:

```python
# Slow way with lists
numbers = list(range(1000000))
result = [x + 5 for x in numbers]  # Takes time!
```

### The NumPy Solution

NumPy (Numerical Python) is a library that provides:
- **Fast array operations** - NumPy arrays are stored more efficiently than Python lists
- **Mathematical functions** - Built-in functions for math operations
- **Easy data manipulation** - Reshape, slice, and combine arrays effortlessly

Think of NumPy arrays as **supercharged lists** that can handle numbers efficiently.

### Why is NumPy Faster?

| Aspect | Python List | NumPy Array |
|--------|-------------|-------------|
| Storage | Scattered in memory | Contiguous memory block |
| Operations | Loop through each element | Vectorized (whole array at once) |
| Type | Mixed types allowed | Same type (usually numbers) |

### What Can NumPy Do?

- Basic math: addition, subtraction, multiplication, division
- Statistical analysis: mean, median, standard deviation
- Linear algebra: matrix multiplication, inverses
- Random number generation
- Image processing (images are arrays of pixels)
- Machine learning (data is often numerical arrays)

---

## 2. Getting Started

### Installation

```bash
pip install numpy
```

### Importing NumPy

```python
import numpy as np
```

The `as np` convention lets you type `np.function()` instead of `numpy.function()`.

### Your First NumPy Array

```python
import numpy as np

# Create an array from a Python list
my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)

print(my_array)
# Output: [1 2 3 4 5]

print(type(my_array))
# Output: <class 'numpy.ndarray'>
```

---

## 3. Creating Arrays

### Method 1: From a Python List

```python
import numpy as np

# 1D array
arr1d = np.array([1, 2, 3])
print("1D array:", arr1d)

# 2D array (matrix)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D array:")
print(arr2d)
```

**Output:**
```
1D array: [1 2 3 4 5]
2D array:
[[1 2 3]
 [4 5 6]]
```

### Method 2: Arrays of Zeros or Ones

```python
import numpy as np

# Array of zeros
zeros = np.zeros(5)
print("Zeros:", zeros)
# Output: [0. 0. 0. 0. 0.]

# 2D array of ones
ones = np.ones((3, 4))
print("2D ones:")
print(ones)
```

**Output:**
```
Zeros: [0. 0. 0. 0. 0.]
2D ones:
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
```

### Method 3: Sequences with arange()

```python
import numpy as np

# From 0 to 9 (exclusive)
arr1 = np.arange(10)
print("0 to 9:", arr1)
# Output: [0 1 2 3 4 5 6 7 8 9]

# From 2 to 10
arr2 = np.arange(2, 11)
print("2 to 10:", arr2)
# Output: [ 2  3  4  5  6  7  8  9 10]

# With step
arr3 = np.arange(0, 20, 3)
print("Step of 3:", arr3)
# Output: [ 0  3  6  9 12 15 18]
```

### Method 4: Evenly Spaced Values with linspace()

```python
import numpy as np

# 5 values from 0 to 10 (inclusive)
arr = np.linspace(0, 10, 5)
print("Evenly spaced:", arr)
# Output: [ 0.   2.5  5.   7.5 10. ]
```

### Method 5: Random Arrays

```python
import numpy as np

# Random floats between 0 and 1
rand = np.random.rand(5)
print("Random 0-1:", rand)

# Random integers (1 to 100)
rand_int = np.random.randint(1, 101, 5)
print("Random integers:", rand_int)
```

---

## 4. Understanding Array Properties

### Key Properties

Every NumPy array has these important attributes:

| Property | Description | Example |
|----------|-------------|---------|
| `ndim` | Number of dimensions | 1 for 1D, 2 for 2D |
| `shape` | Size of each dimension | (3, 4) for 3 rows, 4 columns |
| `size` | Total number of elements | 12 for a 3×4 array |
| `dtype` | Data type of elements | int32, float64 |

### Examples

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Array:")
print(arr)
print()
print("Dimensions:", arr.ndim)      # 2
print("Shape:", arr.shape)          # (2, 3)
print("Size:", arr.size)            # 6
print("Data type:", arr.dtype)      # int64
```

**Output:**
```
Array:
[[1 2 3]
 [4 5 6]]

Dimensions: 2
Shape: (2, 3)
Size: 6
Data type: int64
```

### Data Types

```python
import numpy as np

# Integer arrays
int_arr = np.array([1, 2, 3], dtype='int32')
print("Integer:", int_arr.dtype)

# Float arrays
float_arr = np.array([1.5, 2.5, 3.5])
print("Float:", float_arr.dtype)

# Convert types
converted = int_arr.astype('float64')
print("Converted:", converted.dtype)
```

---

## 5. Indexing and Slicing

### Basic Indexing

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Get first element (index 0)
print("First:", arr[0])      # 10

# Get last element
print("Last:", arr[-1])      # 50

# Get third element
print("Third:", arr[2])      # 30
```

### 2D Array Indexing

```python
import numpy as np

arr = np.array([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])

# Get element at row 1, column 2
print("Row 1, Col 2:", arr[1, 2])   # 6

# Get entire first row
print("First row:", arr[0])          # [1 2 3]

# Get entire second column
print("Second column:", arr[:, 1])   # [2 5 8]
```

### Slicing Basics

Think of slicing as `[start:stop:step]`:

```python
import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Elements from index 2 to 5 (exclusive)
print("arr[2:5]:", arr[2:5])        # [2 3 4]

# First 5 elements
print("arr[:5]:", arr[:5])          # [0 1 2 3 4]

# Last 3 elements
print("arr[-3:]:", arr[-3:])        # [7 8 9]

# Every other element
print("arr[::2]:", arr[::2])        # [0 2 4 6 8]

# Reverse the array
print("Reversed:", arr[::-1])       # [9 8 7 6 5 4 3 2 1 0]
```

### 2D Slicing

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# First 2 rows, first 2 columns
print("Top-left 2x2:")
print(arr[:2, :2])
```

**Output:**
```
Top-left 2x2:
[[1 2]
 [5 6]]
```

### Boolean Indexing (Filtering)

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Find elements greater than 25
mask = arr > 25
print("Mask:", mask)                 # [False False  True  True  True]

# Get elements that are greater than 25
filtered = arr[mask]
print("Filtered:", filtered)         # [30 40 50]

# Shortcut - directly filter
print("Direct filter:", arr[arr > 25])  # [30 40 50]
```

---

## 6. Array Operations

### Basic Arithmetic

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Add 10 to each element
print("Add 10:", arr + 10)          # [11 12 13 14 15]

# Multiply by 2
print("Multiply by 2:", arr * 2)     # [ 2  4  6  8 10]

# Square each element
print("Square:", arr ** 2)           # [ 1  4  9 16 25]
```

### Array-to-Array Operations

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print("a + b:", a + b)              # [11 22 33]
print("b - a:", b - a)              # [ 9 18 27]
print("a * b:", a * b)              # [10 40 90]
print("b / a:", b / a)              # [10. 10. 10.]
```

### Aggregation Functions

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print("Sum:", np.sum(arr))          # 15
print("Mean:", np.mean(arr))        # 3.0
print("Min:", np.min(arr))          # 1
print("Max:", np.max(arr))          # 5
print("Std Dev:", np.std(arr))      # 1.414...
```

### Axis-Based Operations

When you have a 2D array, you can aggregate along rows or columns:

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Sum all elements
print("Total sum:", np.sum(arr))                    # 21

# Sum each column (axis=0)
print("Sum columns:", np.sum(arr, axis=0))          # [5 7 9]

# Sum each row (axis=1)
print("Sum rows:", np.sum(arr, axis=1))              # [ 6 15]
```

---

## 7. Broadcasting

### What is Broadcasting?

Broadcasting allows NumPy to perform operations on arrays of different shapes. The smaller array is "stretched" to match the larger one.

### Example 1: Adding a Scalar

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
result = arr + 5
print("Result:", result)            # [6 7 8 9]
```

The scalar `5` is broadcast to match the shape of `arr`.

### Example 2: Adding a 1D Array to 2D

```python
import numpy as np

arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

arr1d = np.array([10, 20, 30])

result = arr2d + arr1d
print(result)
```

**Output:**
```
[[11 22 33]
 [14 25 36]]
```

The 1D array `[10, 20, 30]` is added to each row.

### Example 3: Adding with a Column Vector

```python
import numpy as np

arr2d = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])

column = np.array([[10], [20], [30]])  # 3x1 column

result = arr2d + column
print(result)
```

**Output:**
```
[[11 12]
 [23 24]
 [35 36]]
```

### Broadcasting Rules

1. Compare dimensions from the right
2. Each dimension must be equal OR one must be 1
3. If neither, broadcasting fails

```python
# Valid: (3, 4) + (4,) → the (4,) becomes (1, 4) then broadcast
# Valid: (3, 4) + (3, 1) → the (3, 1) is broadcast to (3, 4)
# Invalid: (3, 4) + (2,) → can't broadcast!
```

---

## 8. Mathematical Functions

### Trigonometric Functions

```python
import numpy as np

angles = np.array([0, np.pi/2, np.pi])

print("Sine:", np.sin(angles))      # [0.000000e+00 1.000000e+00 1.224647e-16]
print("Cosine:", np.cos(angles))    # [ 1.000000e+00  6.123234e-17 -1.000000e+00]
print("Tangent:", np.tan(angles))   # [ 0.00000000e+00  1.63312394e+16 -1.22464680e-16]
```

### Inverse Trigonometric Functions

```python
import numpy as np

values = np.array([0, 0.5, 1])

print("Arcsin:", np.arcsin(values))   # [0.         0.52359878 1.57079633]
print("Arccos:", np.arccos(values))    # [1.57079633 1.04719755 0.        ]
print("Arctan:", np.arctan(values))     # [0.         0.46364761 0.78539816]
```

### Exponential and Logarithmic Functions

```python
import numpy as np

arr = np.array([0, 1, 2])

# Exponential: e^x
print("exp:", np.exp(arr))           # [1.         2.71828183 7.3890561 ]

# Natural log: ln(x)
print("log:", np.log(arr))            # [-inf  0.          0.69314718]

# Log base 10
print("log10:", np.log10(arr))       # [-inf  0.          0.30103   ]
```

### Rounding Functions

```python
import numpy as np

arr = np.array([1.234, 2.567, 3.891])

print("Round:", np.round(arr, 2))    # [1.23 2.57 3.89]
print("Floor:", np.floor(arr))       # [1. 2. 3.]
print("Ceil:", np.ceil(arr))         # [2. 3. 4.]
print("Truncate:", np.trunc(arr))    # [1. 2. 3.]
```

---

## 9. Statistical Functions

### Basic Statistics

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print("Sum:", np.sum(arr))                      # 55
print("Mean:", np.mean(arr))                    # 5.5
print("Median:", np.median(arr))               # 5.5
print("Std Dev:", np.std(arr))                  # 2.872...
print("Variance:", np.var(arr))                 # 8.25
print("Min:", np.min(arr))                      # 1
print("Max:", np.max(arr))                      # 10
print("Range:", np.ptp(arr))                    # 9 (max - min)
```

### Percentiles

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print("25th percentile:", np.percentile(arr, 25))   # 3.25
print("50th percentile:", np.percentile(arr, 50))   # 5.5
print("75th percentile:", np.percentile(arr, 75))   # 7.75
```

### Weighted Average

```python
import numpy as np

values = np.array([1, 2, 3, 4])
weights = np.array([4, 3, 2, 1])

# Simple average
print("Simple average:", np.mean(values))      # 2.5

# Weighted average
print("Weighted average:", np.average(values, weights=weights))  # 2.0
```

---

## 10. Linear Algebra Basics

### Creating Matrices

```python
import numpy as np

# 2x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print("Matrix:")
print(matrix)
```

### Matrix Transpose

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("Transposed:")
print(matrix.T)
```

**Output:**
```
Transposed:
[[1 4]
 [2 5]
 [3 6]]
```

### Matrix Multiplication

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Using @ operator
C = A @ B
print("A @ B:")
print(C)

# Using np.matmul
D = np.matmul(A, B)
print("np.matmul(A, B):")
print(D)
```

**Output:**
```
A @ B:
[[19 22]
 [43 50]]
```

### Dot Product

```python
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product: 1*4 + 2*5 + 3*6 = 32
result = np.dot(v1, v2)
print("Dot product:", result)         # 32
```

### Identity Matrix

```python
import numpy as np

# 3x3 identity matrix
I = np.eye(3)
print("Identity matrix:")
print(I)
```

**Output:**
```
Identity matrix:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

---

## 11. Random Numbers

### Random Integers

```python
import numpy as np

# Single random integer (1 to 100)
print("Random 1-100:", np.random.randint(1, 101))

# Array of 5 random integers
print("5 randoms:", np.random.randint(1, 101, 5))

# 2D array of random integers
print("2x3 array:")
print(np.random.randint(1, 101, (2, 3)))
```

### Random Floats

```python
import numpy as np

# Random float between 0 and 1
print("Random 0-1:", np.random.rand())

# Array of random floats
print("5 randoms:", np.random.rand(5))

# 2x3 array
print("2x3 array:")
print(np.random.rand(2, 3))
```

### Random with Distribution

```python
import numpy as np

# Normal distribution (bell curve)
normal = np.random.normal(loc=0, scale=1, size=5)
print("Normal distribution:", normal)

# Uniform distribution
uniform = np.random.uniform(low=0, high=10, size=5)
print("Uniform distribution:", uniform)
```

### Setting Seed (Reproducibility)

```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

print("First run:", np.random.rand(3))
print("Second run:", np.random.rand(3))

# Reset seed
np.random.seed(42)
print("Again (same as first):", np.random.rand(3))
```

---

## 12. Reshaping and Manipulation

### Reshape

```python
import numpy as np

arr = np.arange(12)  # [0, 1, 2, ..., 11]
print("Original:", arr)

# Reshape to 3x4
reshaped = arr.reshape(3, 4)
print("Reshaped to 3x4:")
print(reshaped)
```

**Output:**
```
Original: [ 0  1  2  3  4  5  6  7  8  9 10 11]
Reshaped to 3x4:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
```

### Flatten

```python
import numpy as np

arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# Flatten to 1D
flat = arr2d.flatten()
print("Flattened:", flat)            # [1 2 3 4 5 6]
```

### Stacking Arrays

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vertical stack (adds rows)
vstack = np.vstack([a, b])
print("Vertical stack:")
print(vstack)

# Horizontal stack (adds columns)
hstack = np.hstack([a, b])
print("Horizontal stack:", hstack)   # [1 2 3 4 5 6]
```

### Splitting Arrays

```python
import numpy as np

arr = np.arange(12)

# Split into 3 equal parts
split = np.split(arr, 3)
print("Split into 3:", split)
# [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([8, 9, 10, 11])]
```

---

## 13. Working with Files

### Saving Arrays

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Save to text file
np.savetxt('array.txt', arr)

# Save to binary file (.npy)
np.save('array.npy', arr)
```

### Loading Arrays

```python
import numpy as np

# Load from text file
loaded_txt = np.loadtxt('array.txt')
print("From text:", loaded_txt)

# Load from binary file
loaded_npy = np.load('array.npy')
print("From binary:", loaded_npy)
```

### Saving Multiple Arrays

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Save multiple arrays
np.savez('arrays.npz', first=arr1, second=arr2)

# Load them back
data = np.load('arrays.npz')
print("First:", data['first'])
print("Second:", data['second'])
```

---

## 14. Common Patterns and Tips

### Copy vs View

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# View - shares memory with original
view = arr[1:4]
view[0] = 100
print("Original changed:", arr)  # [1, 100, 3, 4, 5]

# Copy - independent
original = np.array([1, 2, 3, 4, 5])
copy = original.copy()
copy[0] = 100
print("Original unchanged:", original)  # [1, 2, 3, 4, 5]
```

### Finding Values

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Find index of value 30
index = np.where(arr == 30)[0][0]
print("Index of 30:", index)        # 2

# Find indices where condition is true
indices = np.where(arr > 25)
print("Indices where > 25:", indices)  # (array([2, 3, 4]),)
```

### Sorting

```python
import numpy as np

arr = np.array([5, 2, 8, 1, 9])

# Sort array
sorted_arr = np.sort(arr)
print("Sorted:", sorted_arr)         # [1 2 5 8 9]

# Get indices that would sort
indices = np.argsort(arr)
print("Sort indices:", indices)      # [3 1 0 2 4]
```

### Working with NaN

```python
import numpy as np

arr = np.array([1, 2, np.nan, 4, 5])

# Check for NaN
print("Has NaN:", np.isnan(arr))    # [False False  True False False]

# Get indices of NaN
nan_indices = np.where(np.isnan(arr))
print("NaN indices:", nan_indices)
```

### Conditional Replacement

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Replace values > 3 with 0
result = np.where(arr > 3, 0, arr)
print("After replacement:", result)  # [1 2 3 0 0]
```

---

## Quick Reference Card

### Creating Arrays
| Function | Description |
|----------|-------------|
| `np.array([1,2,3])` | From list |
| `np.zeros(5)` | Array of zeros |
| `np.ones(5)` | Array of ones |
| `np.arange(5)` | 0 to 4 |
| `np.linspace(0,1,5)` | 5 values from 0 to 1 |
| `np.random.rand(5)` | Random 0-1 |

### Array Properties
| Property | Description |
|----------|-------------|
| `arr.shape` | Dimensions |
| `arr.ndim` | Number of dimensions |
| `arr.size` | Total elements |
| `arr.dtype` | Data type |

### Basic Operations
| Operation | Description |
|-----------|-------------|
| `arr + 5` | Add scalar |
| `arr * 2` | Multiply |
| `np.sum(arr)` | Sum |
| `np.mean(arr)` | Mean |
| `arr.reshape(2,3)` | Reshape |

### Indexing
| Syntax | Description |
|--------|-------------|
| `arr[0]` | First element |
| `arr[-1]` | Last element |
| `arr[1:4]` | Slice |
| `arr[arr > 3]` | Boolean filter |

---

## Next Steps

Now that you've learned the fundamentals, you can explore:

1. **Advanced Linear Algebra** - Matrix inverses, determinants, eigenvalues
2. **Fourier Transforms** - Signal processing, frequency analysis
3. **Integration with Matplotlib** - Data visualization
4. **Machine Learning** - NumPy is the foundation of libraries like TensorFlow and PyTorch

### Practice Exercises

1. Create a 3x3 matrix and calculate its transpose
2. Generate 100 random numbers and find the mean and standard deviation
3. Filter an array to keep only values between 10 and 50
4. Multiply two 2x2 matrices
5. Save an array to a file and load it back

---

*Happy coding! NumPy is your foundation for scientific computing in Python.*
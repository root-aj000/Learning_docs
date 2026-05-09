# Numpy

## Table of Contents

1. [NumPy - Introduction](#numpy---introduction)
2. [NumPy - Environment](#numpy---environment)
3. [NumPy - Ndarray Object](#numpy---ndarray-object)
4. [NumPy - Data Types](#numpy---data-types)
5. [NumPy - Array Creation Routines](#numpy---array-creation-routines)
6. [NumPy - Array Manipulation](#numpy---array-manipulation)
7. [NumPy - Array From Existing Data](#numpy---array-from-existing-data)
8. [NumPy - Array From Numerical Ranges](#numpy---array-from-numerical-ranges)
9. [NumPy - Iterating Over Array](#numpy---iterating-over-array)
10. [NumPy - Reshaping Arrays](#numpy---reshaping-arrays)
11. [NumPy - Concatenating Arrays](#numpy---concatenating-arrays)
12. [NumPy - Stacking Arrays](#numpy---stacking-arrays)
13. [NumPy - Splitting Arrays](#numpy---splitting-arrays)
14. [NumPy - Flattening Arrays](#numpy---flattening-arrays)
15. [NumPy - Transposing Arrays](#numpy---transposing-arrays)
16. [NumPy - Indexing & Slicing](#numpy---indexing--slicing)
17. [NumPy - Indexing](#numpy---indexing)
18. [NumPy - Slicing](#numpy---slicing)
19. [NumPy - Advanced Indexing](#numpy---advanced-indexing)
20. [Numpy - Fancy Indexing](#numpy---fancy-indexing)
21. [NumPy - Field access](#numpy---field-access)
22. [NumPy - Slicing with Boolean Arrays](#numpy---slicing-with-boolean-arrays)
23. [NumPy - Array Attributes](#numpy---array-attributes)
24. [NumPy - Array Shape](#numpy---array-shape)
25. [NumPy - Array Size](#numpy---array-size)
26. [NumPy - Array Strides](#numpy---array-strides)
27. [NumPy - Array Itemsize](#numpy---array-itemsize)
28. [NumPy - Broadcasting](#numpy---broadcasting)
29. [NumPy - Arithmetic Operations](#numpy---arithmetic-operations)
30. [NumPy - Array Addition](#numpy---array-addition)
31. [NumPy - Array Subtraction](#numpy---array-subtraction)
32. [NumPy - Array Multiplication](#numpy---array-multiplication)
33. [NumPy - Array Division](#numpy---array-division)
34. [NumPy - Swapping Axes of Arrays](#numpy---swapping-axes-of-arrays)
35. [NumPy - Byte Swapping](#numpy---byte-swapping)
36. [NumPy - Copies & Views](#numpy---copies--views)
37. [NumPy - Element-wise Array Comparisons](#numpy---element-wise-array-comparisons)
38. [NumPy - Filtering Arrays](#numpy---filtering-arrays)
39. [NumPy - Joining Arrays](#numpy---joining-arrays)
40. [NumPy - Sort, Search & Counting Functions](#numpy---sort-search--counting-functions)
41. [NumPy - Searching Arrays](#numpy---searching-arrays)
42. [NumPy - Union of Arrays](#numpy---union-of-arrays)
43. [NumPy - Finding Unique Rows](#numpy---finding-unique-rows)
44. [NumPy - Creating Datetime Arrays](#numpy---creating-datetime-arrays)
45. [NumPy - Binary Operators](#numpy---binary-operators)
46. [NumPy - String Functions](#numpy---string-functions)
47. [NumPy - Matrix Library](#numpy---matrix-library)
48. [NumPy - Linear Algebra](#numpy---linear-algebra)
49. [NumPy - Matplotlib](#numpy---matplotlib)
50. [NumPy - Histogram Using Matplotlib](#numpy---histogram-using-matplotlib)
51. [NumPy - Sorting Arrays](#numpy---sorting-arrays)
52. [NumPy - Sorting Along an Axis](#numpy---sorting-along-an-axis)
53. [NumPy - Sorting with Fancy Indexing](#numpy---sorting-with-fancy-indexing)
54. [NumPy - Structured Arrays](#numpy---structured-arrays)
55. [NumPy - Creating Structured Arrays](#numpy---creating-structured-arrays)
56. [NumPy - Manipulating Structured Arrays](#numpy---manipulating-structured-arrays)
57. [NumPy - Record Arrays](#numpy---record-arrays)
58. [NumPy - Loading Arrays](#numpy---loading-arrays)
59. [NumPy - Saving Arrays](#numpy---saving-arrays)
60. [NumPy - Append Values to an Array](#numpy---append-values-to-an-array)
61. [NumPy - Swap Columns of Array](#numpy---swap-columns-of-array)
62. [NumPy - Insert Axes to an Array](#numpy---insert-axes-to-an-array)
63. [NumPy - Handling Missing Data](#numpy---handling-missing-data)
64. [NumPy - Identifying Missing Values](#numpy---identifying-missing-values)
65. [NumPy - Removing Missing Data](#numpy---removing-missing-data)
66. [NumPy - Imputing Missing Data](#numpy---imputing-missing-data)
67. [NumPy - Performance Optimization with Arrays](#numpy---performance-optimization-with-arrays)
68. [NumPy - Matrix Addition](#numpy---matrix-addition)
69. [NumPy - Matrix Subtraction](#numpy---matrix-subtraction)
70. [NumPy - Matrix Multiplication](#numpy---matrix-multiplication)
71. [NumPy - Element-wise Matrix Operations](#numpy---element-wise-matrix-operations)
72. [NumPy - Dot Product](#numpy---dot-product)
73. [NumPy - Matrix Inversion](#numpy---matrix-inversion)
74. [NumPy - Determinant Calculation](#numpy---determinant-calculation)
75. [NumPy - Eigenvalues](#numpy---eigenvalues)
76. [NumPy - Eigenvectors](#numpy---eigenvectors)
77. [NumPy - Singular Value Decomposition](#numpy---singular-value-decomposition)
78. [NumPy - Solving Linear Equations](#numpy---solving-linear-equations)
79. [NumPy - Matrix Norms](#numpy---matrix-norms)
80. [NumPy - Sum](#numpy---sum)
81. [NumPy - Mean](#numpy---mean)
82. [NumPy - Median](#numpy---median)
83. [NumPy - Min](#numpy---min)
84. [NumPy - Max](#numpy---max)
85. [NumPy - Unique Elements](#numpy---unique-elements)
86. [NumPy - Intersection](#numpy---intersection)
87. [NumPy - Union](#numpy---union)
88. [NumPy - Difference](#numpy---difference)
89. [NumPy - Random Generator](#numpy---random-generator)
90. [NumPy - Permutations and Shuffling](#numpy---permutations-and-shuffling)
91. [NumPy - Uniform Distribution](#numpy---uniform-distribution)
92. [NumPy - Normal Distribution](#numpy---normal-distribution)
93. [NumPy - Binomial Distribution](#numpy---binomial-distribution)
94. [NumPy - Poisson Distribution](#numpy---poisson-distribution)
95. [NumPy - Exponential Distribution](#numpy---exponential-distribution)
96. [NumPy - Rayleigh Distribution](#numpy---rayleigh-distribution)
97. [NumPy - Logistic Distribution](#numpy---logistic-distribution)
98. [NumPy - Pareto Distribution](#numpy---pareto-distribution)
99. [NumPy - Visualize Distributions With Seaborn](#numpy---visualize-distributions-with-seaborn)
100. [NumPy - Multinomial Distribution](#numpy---multinomial-distribution)
101. [NumPy - Chi Square Distribution](#numpy---chi-square-distribution)
102. [NumPy - Zipf Distribution](#numpy---zipf-distribution)
103. [I/O with NumPy](#i-o-with-numpy)
104. [NumPy - Reading Data from Files](#numpy---reading-data-from-files)
105. [NumPy - Writing Data to Files](#numpy---writing-data-to-files)
106. [NumPy - File Formats Supported](#numpy---file-formats-supported)
107. [NumPy - Mathematical Functions](#numpy---mathematical-functions)
108. [NumPy - Trigonometric Functions](#numpy---trigonometric-functions)
109. [NumPy - Exponential Functions](#numpy---exponential-functions)
110. [NumPy - Logarithmic Functions](#numpy---logarithmic-functions)
111. [NumPy - Hyperbolic Functions](#numpy---hyperbolic-functions)
112. [NumPy - Rounding Functions](#numpy---rounding-functions)
113. [NumPy - Discrete Fourier Transform](#numpy---discrete-fourier-transform)
114. [NumPy - Fast Fourier Transform](#numpy---fast-fourier-transform)
115. [NumPy - Inverse Fourier Transform](#numpy---inverse-fourier-transform)
116. [NumPy - Fourier Series and Transforms](#numpy---fourier-series-and-transforms)
117. [NumPy - Signal Processing Applications](#numpy---signal-processing-applications)
118. [NumPy - Convolution](#numpy---convolution)
119. [NumPy - Polynomial Representation](#numpy---polynomial-representation)
120. [NumPy - Polynomial Operations](#numpy---polynomial-operations)
121. [NumPy - Finding Roots of Polynomials](#numpy---finding-roots-of-polynomials)
122. [NumPy - Evaluating Polynomials](#numpy---evaluating-polynomials)
123. [NumPy - Statistical Functions](#numpy---statistical-functions)
124. [NumPy - Descriptive Statistics](#numpy---descriptive-statistics)
125. [NumPy - Basics of Dates and Times](#numpy---basics-of-dates-and-times)
126. [NumPy - Representing Dates and Times](#numpy---representing-dates-and-times)
127. [NumPy - Date and Time Arithmetic](#numpy---date-and-time-arithmetic)
128. [NumPy - Indexing with Datetimes](#numpy---indexing-with-datetimes)
129. [NumPy - Time Zone Handling](#numpy---time-zone-handling)
130. [NumPy - Time Series Analysis](#numpy---time-series-analysis)
131. [NumPy - Working with Time Deltas](#numpy---working-with-time-deltas)
132. [NumPy - Handling Leap Seconds](#numpy---handling-leap-seconds)
133. [NumPy - Vectorized Operations with Datetimes](#numpy---vectorized-operations-with-datetimes)
134. [NumPy - ufunc Introduction](#numpy---ufunc-introduction)
135. [NumPy - Creating Universal Functions (ufunc)](#numpy---creating-universal-functions-ufunc)
136. [NumPy - Arithmetic Universal Function (ufunc)](#numpy---arithmetic-universal-function-ufunc)
137. [NumPy - Rounding Decimal ufunc](#numpy---rounding-decimal-ufunc)
138. [NumPy - Logarithmic Universal Function (ufunc)](#numpy---logarithmic-universal-function-ufunc)
139. [NumPy - Summation Universal Function (ufunc)](#numpy---summation-universal-function-ufunc)
140. [NumPy - Product Universal Function (ufunc)](#numpy---product-universal-function-ufunc)
141. [NumPy - Difference Universal Function (ufunc)](#numpy---difference-universal-function-ufunc)
142. [NumPy - Finding LCM with ufunc](#numpy---finding-lcm-with-ufunc)
143. [NumPy - Finding GCD with ufunc](#numpy---finding-gcd-with-ufunc)
144. [NumPy - Trigonometric ufunc](#numpy---trigonometric-ufunc)
145. [NumPy - Hyperbolic ufunc](#numpy---hyperbolic-ufunc)
146. [NumPy - Set Operations ufunc](#numpy---set-operations-ufunc)

---

## 1. NumPy - Introduction

*Source: [https://www.tutorialspoint.com/numpy/numpy_introduction.htm](https://www.tutorialspoint.com/numpy/numpy_introduction.htm)*

---

---
[Previous](/numpy/index.htm)[Quiz](/numpy/quiz_on_numpy_introduction.htm)[Next](/numpy/numpy_environment.htm)
## Introduction to NumPy

NumPy is a Python package. It stands for 'Numerical Python'. It is a library consisting of multidimensional array objects and a collection of routines for processing of array.
**Numeric**, the ancestor of NumPy, was developed by Jim Hugunin. Another package Numarray was also developed, having some additional functionalities. In 2005, Travis Oliphant created NumPy package by incorporating the features of Numarray into Numeric package. There are many contributors to this open source project.
## Operations using NumPy

Using NumPy, a developer can perform the following operations −

- 
Mathematical and logical operations on arrays.

- 
Fourier transforms and routines for shape manipulation.

- 
Operations related to linear algebra. NumPy has in-built functions for linear algebra and random number generation.

## NumPy  A Replacement for MATLAB

NumPy is often used in conjunction with packages like
**SciPy**(Scientific Python) and**Matplotlib**(plotting library). This combination is widely used as a replacement for MATLAB, a popular platform for technical computing. However, Python alternative to MATLAB is now seen as a more modern and complete programming language.
One of the significant advantages of NumPy is that it is open-source, making it freely accessible to anyone.

## Why is NumPy Faster Than Lists?

NumPy arrays are significantly faster than Python lists for several reasons −
AspectNumPyListMemory StorageNumPy uses a contiguous block of memory, which improves cache efficiency and access speed.Python lists consist of pointers to objects, leading to more memory fragmentation and slower access.Data TypesNumPy supports homogeneous data types (all elements are of the same type), leading to more efficient memory use.Python lists can contain heterogeneous data types (elements can be of different types), resulting in higher memory overhead.OperationsNumPy uses vector operations that leverage SIMD (Single Instruction, Multiple Data) for parallel processing.Python lists rely on loop-based operations, which are slower due to the overhead of Python's interpreted nature.EfficiencyNumPy is written in C and optimized for performance, reducing the execution time of numerical operations.Python lists are executed as Python byte-code, which is generally slower compared to compiled C code.Memory UsageNumPy requires less memory due to fixed data types and contiguous storage.Python lists use more memory because each element is a separate Python object with additional overhead.BroadcastingNumPy supports broadcasting, allowing operations on arrays of different shapes without creating additional copies.Python lists do not support broadcasting, making element-wise operations less efficient.PerformanceBetter cache utilization due to contiguous memory storage, leading to faster access and processing.Poor cache utilization because of scattered memory allocation, slowing down access.FunctionalityNumPy provides a rich set of mathematical functions and tools optimized for array operations.Python lists are Limited to basic operations and lack advanced mathematical capabilities.
## Which Language is NumPy written in?

NumPy is primarily written in the following languages −

- **C:**The core functionality of NumPy, including the implementation of array objects and basic operations, is written in C. This provides the high performance and efficiency that NumPy is known for.
- **Python:**NumPys user interface and high-level functionalities are written in Python. This makes it easy to use and integrate with other Python libraries.
- **Fortran :**Some of the numerical routines in NumPy, especially those related to linear algebra (like LAPACK and BLAS), are written in Fortran. Fortran is known for its efficiency in numerical computing, which enhances NumPys performance for specific types of operations.

---

## 2. NumPy - Environment

*Source: [https://www.tutorialspoint.com/numpy/numpy_environment.htm](https://www.tutorialspoint.com/numpy/numpy_environment.htm)*

---

---

## 3. NumPy - Ndarray Object

*Source: [https://www.tutorialspoint.com/numpy/numpy_ndarray_object.htm](https://www.tutorialspoint.com/numpy/numpy_ndarray_object.htm)*

---

---

## 4. NumPy - Data Types

*Source: [https://www.tutorialspoint.com/numpy/numpy_data_types.htm](https://www.tutorialspoint.com/numpy/numpy_data_types.htm)*

---

---
[Previous](/numpy/numpy_ndarray_object.htm)[Quiz](/numpy/quiz_on_numpy_data_types.htm)[Next](/numpy/numpy_array_creation_routines.htm)
## NumPy Data Types

NumPy supports a much greater variety of numerical types than Python does. The following table shows different scalar data types defined in NumPy.
Sr.No.Data Types & Description1**bool_**
Boolean (True or False) stored as a byte
2**int_**
Default integer type (same as C long; normally either int64 or int32)
3**intc**
Identical to C int (normally int32 or int64)
4**intp**
Integer used for indexing (same as C ssize_t; normally either int32 or int64)
5**int8**
Byte (-128 to 127)
6**int16**
Integer (-32768 to 32767)
7**int32**
Integer (-2147483648 to 2147483647)
8**int64**
Integer (-9223372036854775808 to 9223372036854775807)
9**uint8**
Unsigned integer (0 to 255)
10**uint16**
Unsigned integer (0 to 65535)
11**uint32**
Unsigned integer (0 to 4294967295)
12**uint64**
Unsigned integer (0 to 18446744073709551615)
13**float_**
Shorthand for float64
14**float16**
Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
15**float32**
Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
16**float64**
Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
17**complex_**
Shorthand for complex128
18**complex64**
Complex number, represented by two 32-bit floats (real and imaginary components)
19**complex128**
Complex number, represented by two 64-bit floats (real and imaginary components)

NumPy numerical types are instances of dtype (data-type) objects, each having unique characteristics. The dtypes are available as np.bool_, np.float32, etc.

## Data Type Objects (dtype)

A data type object describes interpretation of fixed block of memory corresponding to an array, depending on the following aspects −

- 
Type of data (integer, float or Python object)

- 
Size of data

- 
Byte order (little-endian or big-endian)

- 
In case of structured type, the names of fields, data type of each field and part of the memory block taken by each field.

- 
If data type is a subarray, its shape and data type

> The byte order is decided by prefixing '<' or '>' to data type. '<' means that encoding is little-endian (least significant is stored in smallest address). '>' means that encoding is big-endian (most significant byte is stored in smallest address).

A dtype object is constructed using the following syntax −

```
numpy.dtype(object, align, copy)
```

The parameters are −

- **Object**− To be converted to data type object
- **Align**− If true, adds padding to the field to make it similar to C-struct
- **Copy**− Makes a new copy of dtype object. If false, the result is reference to builtin data type object
### Example: Using Array-scalar Type

```
import numpy as np
dt = np.dtype(np.int32)
print(dt)
```

Following is the output obtained −

```
int32
```

### Example: Using Equivalent String for Data Type

```
import numpy as np
dt = np.dtype('i4')
print(dt)
```

This will produce the following result −

```
int32
```

### Example: Using Endian Notation

```
import numpy as np
dt = np.dtype('>i4')
print(dt)
```

Following is the output of the above code −

```
>i4
```

### Example: Creating a Structured Data Type

The following examples show the use of structured data type. Here, the field name and the corresponding scalar data type is to be declared −

```
import numpy as np
dt = np.dtype([('age', np.int8)])
print(dt)
```

The output obtained is as shown below −

```
[('age', 'i1')]
```

### Example: Applying Structured Data Type to ndarray

```
import numpy as np
dt = np.dtype([('age', np.int8)])
a = np.array([(10,), (20,), (30,)], dtype=dt)
print(a)
```

After executing the above code, we get the following output −

```
[(10,) (20,) (30,)]
```

### Example: Accessing Field Content of Structured Data Type

```
import numpy as np
dt = np.dtype([('age', np.int8)])
a = np.array([(10,), (20,), (30,)], dtype=dt)
print(a['age'])
```

The result produced is as follows −

```
[10 20 30]
```

### Example: Defining a Complex Structured Data Type

The following examples define a structured data type called
**student**with a string field 'name', an**integer field**'age' and a**float field**'marks'. This dtype is applied to ndarray object −
```
import numpy as np
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(student)
```

We get the output as shown below −

```
[('name', 'S20'), ('age', 'i1'), ('marks', '<f4')])
```

### Example: Applying Complex Structured Data Type to ndarray

```
import numpy as np
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(a)
```

The output is as follows −

```
[('abc', 21, 50.0), ('xyz', 18, 75.0)]
```

Each built-in data type has a character code that uniquely identifies it.

- **'b'**− boolean
- **'i'**− (signed) integer
- **'u'**− unsigned integer
- **'f'**− floating-point
- **'c'**− complex-floating point
- **'m'**− timedelta
- **'M'**− datetime
- **'O'**− (Python) objects
- **'S', 'a'**− (byte-)string
- **'U'**− Unicode
- **'V'**− raw data (void)
## Checking the Data Type of an Array

You can check the data type of an array using the
**dtype**attribute. This attribute returns a dtype object, which describes the type of elements in the array as shown below −
```
import numpy as np
a = np.array([1, 2, 3])
print(a.dtype)
```

Following is the output obtained −

```
int64
```

## Create Arrays With Defined Data Type

In NumPy, you can explicitly specify the data type (dtype) of the elements in an array at the time of its creation.

We can use the
**dtype**parameter in array creation functions (such as np.array(), np.zeros(), np.ones(), etc.) to define the data type of the array elements. By default, NumPy refers the data type from the input data.
### Example: Creating an Integer Array

In this example, we create an array
**a**with elements of type**int32**, which means each element is a 32-bit integer −
```
import numpy as np

# Creating an array of integers with a specified dtype
a = np.array([1, 2, 3], dtype=np.int32)
print("Array:", a)
print("Data type:", a.dtype)
```

This will produce the following result −

```
Array: [1 2 3]
Data type: int32
```

### Example: Creating an Integer Array

Here, we create an array
**c**with elements of type**complex64**, indicating 64-bit complex numbers (32-bit real part and 32-bit imaginary part) −
```
import numpy as np

# Creating an array of complex numbers with a specified dtype
c = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
print("Array:", c)
print("Data type:", c.dtype)
```

Following is the output of the above code −

```
Array: [1.+2.j 3.+4.j 5.+6.j]Data type: complex64
```

## Convert Data Type of NumPy Arrays

NumPy provides several methods to convert the data type of arrays, allowing you to change how data is stored and processed without modifying the underlying values −

- **astype() Method −**It is the most commonly used method for type conversion.
- **numpy.cast() Functions −**A set of functions provided by NumPy for casting arrays to different types.
- **In-place Type Conversion −**It convert types directly while creating arrays.
### Example: Using the "astype" Method

The
**astype**method creates a copy of the array, cast to a specified type. This is the most commonly used method for changing the data type of an array.
Here, we are converting an array of integers to float data type using the astype() method in NumPy −

```
import numpy as np

# Creating an array of integers
a = np.array([1, 2, 3, 4, 5])
print("Original array:", a)
print("Original dtype:", a.dtype)

# Converting to float
a_float = a.astype(np.float32)
print("Converted array:", a_float)
print("Converted dtype:", a_float.dtype)
```

The output obtained is as shown below −

```
Original array: [1 2 3 4 5]
Original dtype: int64
Converted array: [1. 2. 3. 4. 5.]
Converted dtype: float32
```

### Example: Using "numpy.cast" Functions

NumPy also provides functions for casting arrays to specific types. These functions are less commonly used but can be handy in some cases.

In this example, we are creating an array of floats and converting it to integer using the numpy.int32() function −

```
import numpy as np

# Creating an array of floats
d = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
print("Original array:", d)
print("Original dtype:", d.dtype)

# Converting to integer using numpy.int32
d_int = np.int32(d)
print("Converted array:", d_int)
print("Converted dtype:", d_int.dtype)
```

After executing the above code, we get the following output −

```
Original array: [1.1 2.2 3.3 4.4 5.5]
Original dtype: float64
Converted array: [1 2 3 4 5]
Converted dtype: int32
```

### Example: In-place Type Conversion

You can also specify the data type during array creation to avoid the need to convert the type later.

Now, we are creating an array of integers by specifying the float data type using the numpy.float32() function −

```
import numpy as np

# Creating an array of integers with a specified dtype
e = np.array([1, 2, 3, 4, 5], dtype=np.float32)
print("Array:", e)
print("Data type:", e.dtype)
```

The result produced is as follows −

```
Array: [1. 2. 3. 4. 5.]
Data type: float32
```

## What if a Value Cannot Be Converted?

When converting data types in NumPy, you may encounter values that cannot be converted to the desired type. This situation typically raises an error or results in unexpected behavior.

Let us explore different scenarios where a value cannot be converted and how to handle them −

### Scenario 1: Converting Non-numeric Strings to Numbers

If you attempt to convert a non-numeric string to an integer or float, NumPy will raise a
**ValueError**as shown below −
```
import numpy as np

# Creating an array with non-numeric strings
a = np.array(['1', '2', 'three', '4', '5'])
print("Original array:", a)
print("Original dtype:", a.dtype)

try:
   # Attempting to convert to integer
   a_int = a.astype(np.int32)
   print("Converted array:", a_int)
   print("Converted dtype:", a_int.dtype)
except ValueError as e:
   print("Error:", e)
```

In this case, the string 'three' cannot be converted to an integer, resulting in a ValueError as shown in the output below −

```
Original array: ['1' '2' 'three' '4' '5']
Original dtype: <U5
Error: invalid literal for int() with base 10: 'three'
```

### Scenario 2: Converting Out-of-Range Numbers

If you attempt to convert numbers that are out of range for the target data type, NumPy will raise an
**OverflowError**−
```
import numpy as np

# Creating an array with large float values
b = np.array([1.1e10, 2.2e10, 3.3e10])
print("Original array:", b)
print("Original dtype:", b.dtype)

try:
   # Attempting to convert to integer
   b_int = b.astype(np.int32)
   print("Converted array:", b_int)
   print("Converted dtype:", b_int.dtype)
except OverflowError as e:
   print("Error:", e)
```

Here, the large float values cannot be converted to int32 without overflow −

```
Original array: [1.1e+10 2.2e+10 3.3e+10]
Original dtype: float64
Error: OverflowError: (34, 'Numerical result out of range')
```

### Scenario 3: Converting Complex Numbers to Real Numbers

When converting complex numbers to real numbers, NumPy discards the imaginary part and raises a
**ComplexWarning**−
```
import numpy as np

# Creating an array with complex numbers
c = np.array([1+2j, 3+4j, 5+6j])
print("Original array:", c)
print("Original dtype:", c.dtype)

# Converting to float, discarding imaginary part
c_float = c.astype(np.float32)
print("Converted array:", c_float)
print("Converted dtype:", c_float.dtype)
```

In this case, NumPy raises a ComplexWarning and discards the imaginary part during conversion −

```
Original array: [1.+2.j 3.+4.j 5.+6.j]
Original dtype: complex128
ComplexWarning: Casting complex values to real discards the imaginary partc_float = c.astype(np.float32)
Converted array: [1. 3. 5.]
Converted dtype: float32
```

### Scenario 4: Handling Conversion Errors

To handle conversion errors, you can use error handling techniques like
**try-except**blocks to catch and process exceptions.
```
import numpy as np

# Creating an array with mixed data
d = np.array(['1', '2', 'three', '4', '5'])
print("Original array:", d)
print("Original dtype:", d.dtype)

def safe_convert(arr, target_type):
   try:
      return arr.astype(target_type)
   except ValueError as e:
      print("Conversion error:", e)
      return None

# Attempting to convert to integer
d_int = safe_convert(d, np.int32)
if d_int is not None:
   print("Converted array:", d_int)
   print("Converted dtype:", d_int.dtype)
else:
   print("Conversion failed.")
```

In this example, the
**safe_convert()**function catches the "ValueError" and handles it by returning None and printing an error message as shown in the output below −
```
Original array: ['1' '2' 'three' '4' '5']
Original dtype: <U5
Conversion error: invalid literal for int() with base 10: 'three'
Conversion failed.
```

### Scenario 5: Using "np.nan" for Invalid Conversions

For numeric conversions, you can use np.nan (Not a Number) to handle invalid values. This approach is useful when dealing with missing or corrupt data.

```
import numpy as np

# Creating an array with strings, including an invalid entry
e = np.array(['1.1', '2.2', 'three', '4.4', '5.5'])
print("Original array:", e)
print("Original dtype:", e.dtype)

def convert_with_nan(arr):
   result = []
   for item in arr:
      try:
         result.append(float(item))
      except ValueError:
         result.append(np.nan)
   return np.array(result)

# Converting to float with np.nan for invalid entries
e_float = convert_with_nan(e)
print("Converted array:", e_float)
print("Converted dtype:", e_float.dtype)
```

Here, invalid entries are replaced with np.nan −

```
Original array: ['1.1' '2.2' 'three' '4.4' '5.5']
Original dtype: <U5
Converted array: [1.1 2.2 nan 4.4 5.5]
Converted dtype: float64
```

## Converting Data Type on Existing Arrays

You can also convert the data type of existing arrays using the
**view()**method to change the interpretation of the data without changing the underlying bytes.
### Example

Here, the data is reinterpreted as "float32", resulting in unexpected values because the underlying bytes remain unchanged −

```
import numpy as np

# Creating an array of integers
g = np.array([1, 2, 3, 4], dtype=np.int32)
print("Original array:", g)
print("Original dtype:", g.dtype)

# Viewing the array as float32
g_view = g.view(np.float32)
print("Viewed array:", g_view)
print("Viewed dtype:", g_view.dtype)
```

Following is the output of the above code −

```
Original array: [1 2 3 4]
Original dtype: int32
Viewed array: [1.4012985e-45 2.8025969e-45 4.2038954e-45 5.6051939e-45]
Viewed dtype: float32
```

---

## 5. NumPy - Array Creation Routines

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_creation_routines.htm](https://www.tutorialspoint.com/numpy/numpy_array_creation_routines.htm)*

---

---
[Previous](/numpy/numpy_data_types.htm)[Quiz](/numpy/quiz_on_numpy_array_creation_routines.htm)[Next](/numpy/numpy_array_manipulation.htm)
## Creating NumPy Array

We can create a NumPy array using various function provided by the Python NumPy library. This package provides a multidimensional array object and various other required objects, routines, for efficient functionality. Following are the functions using which we can create a NumPy array −

- Using numpy.array() Function
- Using numpy.zeros() Function
- Using numpy.ones() Function
- Using numpy.arange() Function
- Using numpy.linspace() Function
- Using numpy.random.rand() Function
- Using numpy.empty() Function
- Using numpy.full() Function
> Unlike Python lists, NumPy arrays support element-wise operations and are more memory-efficient, making them useful for mathematical computations.

## Using numpy.array() Function

We can use the
**numpy.array()**function to create an array by passing a Python list or tuple as an argument to the function.
This function converts input data (like lists, tuples, etc.) into an ndarray (NumPy array). Following is the syntax −

```
numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0, like=None)
```

### Example: Creating a 1D NumPy Array

In the following example, we are creating a 1-dimensional NumPy array from a list of integers using the numpy.array() function −

```
import numpy as np

# Creating a 1D array from a list
my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)

print("1D Array:", my_array)
```

Following is the output obtained −

```
1D Array: [1 2 3 4 5]
```

### Example: Creating a 2D NumPy Array

In here, we are creating a 2-dimensional NumPy array from a list of lists using the numpy.array() function −

```
import numpy as np

# Creating a 2D array from a list of lists
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("2D Array:\n", arr)
```

This will produce the following result −

```
2D Array:
 [[1 2 3]
 [4 5 6]]
```

## Using numpy.zeros() Function

We can also use the
**numpy.zeros()**function for creating an array by specifying the desired shape of the array as a tuple or an integer.
This function creates a NumPy array filled with zeros. It accepts the shape of the array as an argument and optionally the data type (dtype). By default, the data type is
**float64**. Following is the syntax −
```
numpy.zeros(shape, dtype=float, order='C')
```

### Example

In this example, we are creating a NumPy array with 5 elements, all initialized to zero using the numpy.zeros() function −

```
import numpy as np

# Creating an array of zeros 
arr = np.zeros(5)
print(arr)
```

Following is the output of the above code −

```
[0. 0. 0. 0. 0.]
```

## Using numpy.ones() Function

On the other hand, the numpy.ones() function creates an array where all elements are set to 1. It accepts three main parameters:
**shape**,**dtype**, and**order**.
- The**shape**parameter, which can be an integer or a tuple of integers, defines the dimensions of the array.
- The**dtype**parameter specifies the desired data type of the array elements, defaulting to "float64" if not provided.
- The**order**parameter determines the memory layout of the array, either row-major (C-style) or column-major (Fortran-style), with 'C' being the default.
Following is the syntax −

```
numpy.ones(shape, dtype=None, order='C')
```

### Example: Creating 1D array of ones

In the example below, we are creating a 1 dimensional NumPy array with 3 elements, all initialized to one using the numpy.ones() function −

```
import numpy as np

# Creating an array of ones 
arr = np.ones(3)
print(arr)
```

After executing the above code, we get the following output −

```
[1. 1. 1.]
```

### Example: Creating 2D array of ones

In here, we create a 2 dimensional NumPy array with 2 rows and 3 columns, filled with ones, using the np.ones() function −

```
import numpy as np

# Creating 2D array of ones 
array_2d = np.ones((4, 3))
print(array_2d)
```

The result produced is as follows −

```
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
```

### Example: Creating a Fortran-ordered array of ones

Now, we are creating a 2-dimensional NumPy array with 2 rows and 3 columns, filled with ones, using the np.ones() function with Fortran-style (column-major) order −

```
import numpy as np

# Creating Fortran-ordered array of ones 
array_F = np.ones((4, 3), order='F')
print(array_F)
```

We get the output as shown below −

```
[[1. 1. 1.][1. 1. 1.][1. 1. 1.][1. 1. 1.]]
```

## Using numpy.arange() Function

The
**numpy.arange()**function creates an array by generating a sequence of numbers based on specified start, stop, and step values. It is similar to Python's built-in range() function.
This function creates an array of evenly spaced values within a given interval. It allows specifying the start, stop, and step size, and returns a NumPy array.

- **start −**The starting value of the sequence. If not specified, it defaults to 0.
- **stop −**The end value of the sequence. This value is exclusive, meaning it is not included in the sequence.
- **step −**The step or interval between each pair of consecutive values in the sequence. If not specified, it defaults to 1.
Following is the syntax −

```
numpy.arange([start,] stop[, step,] dtype=None, *, like=None)
```

### Example

In the following example, we first create a NumPy array "array1" from 0 to 9. Then, we create another array "array2" with values starting from 1 up to (but not including) 10, with a step of 2 using the np.arange() function −

```
import numpy as np

# Providing just the stop value
array1 = np.arange(10)
print("array1:", array1)

# Providing start, stop and step value
array2 = np.arange(1, 10, 2)
print("array2:",array2)
```

Following is the output obtained −

```
array1: [0 1 2 3 4 5 6 7 8 9]
array2: [1 3 5 7 9]
```

## Using numpy.linspace() Function

We can even use the numpy.linspace() function to create an array by specifying the start, stop, and the number of elements we want.

The array created by this function consists of evenly spaced values over a specified interval. The function takes parameters for start, stop, and the number of elements, and generates values that are evenly distributed between the start and stop values, inclusive. Following is the syntax −

```
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

The numpy.linspace() function is particularly useful when you need a set number of points between two endpoints for plotting or numerical computations.

### Example

In the example below, we are using numpy.linspace() function to create three arrays (array1, array2, and array3) with specified ranges and configurations.

The "array1" is created with 10 evenly spaced values from 0 to 5, inclusive. The "array2 consists of 5 values ranging from 1 to just under 2, excluding the endpoint. The "array3" is created with 5 values from 0 to 10, and returns both the array and the step size between consecutive values −

```
import numpy as np

# Creating an array of 10 evenly spaced values from 0 to 5
array1 = np.linspace(0, 5, num=10)
print("array1:",array1)

# Creating an array with 5 values from 1 to 2, excluding the endpoint
array2 = np.linspace(1, 2, num=5, endpoint=False)
print("array2:",array2)

# Creating an array and returning the step value
array3, step = np.linspace(0, 10, num=5, retstep=True)
print("array3:",array3)
print("Step size:", step)
```

This will produce the following result −

```
array1: [0.         0.55555556 1.11111111 1.66666667 2.22222222 2.77777778
 3.33333333 3.88888889 4.44444444 5.        ]
array2: [1.  1.2 1.4 1.6 1.8]
array3: [ 0.   2.5  5.   7.5 10. ]
Step size: 2.5
```

## Using random.rand() Function

Alternatively, we can use the numpy.random.rand() function for creating an array by specifying the dimensions of the array as parameters.

This function is used to create an array of specified shape filled with random values sampled from a uniform distribution over [0, 1).

It accepts parameters for the dimensions of the array (like numpy.random.rand(rows, columns)), and generates an array of the specified shape with random values between 0 and 1. If no argument is provided, it returns a single random float value. Following is the syntax −

```
numpy.random.rand(d0, d1, ..., dn)
```

### Example

In the following example, we are using numpy.random.rand() function to generate arrays of random floats with different dimensions −

```
import numpy as np

# Generating a single random float
random_float = np.random.rand()
print("random_float:",random_float)

# Generating a 1D array of random floats
array_1d = np.random.rand(5)
print("array_1d:",array_1d)

# Generating a 2D array of random floats
array_2d = np.random.rand(2, 3)
print("array_2d:",array_2d)

# Generating a 3D array of random floats
array_3d = np.random.rand(2, 3, 4)
print("array_3d:",array_3d)
```

Following is the output of the above code −

```
random_float: 0.5030496450079744
array_1d: [0.19476581 0.54430648 0.64571106 0.27443774 0.71874319]
array_2d: [[0.91141582 0.58847504 0.37284854]
 [0.0715398  0.21305363 0.766954  ]]
array_3d: [[[0.7295106  0.1582053  0.91376381 0.14099229]
  [0.6876814  0.19351871 0.18056163 0.61370308]
  [0.42382443 0.6665121  0.42322218 0.11707395]]

 [[0.60883975 0.01724199 0.95753734 0.17805716]
  [0.47770594 0.55840874 0.7375783  0.50512301]
  [0.73730351 0.85900855 0.16472072 0.2338285 ]]]
```

## Using numpy.empty() Function

We can create a NumPy array using the numpy.empty() function by specifying the shape of the array as parameters.

This function initializes an array without initializing its elements; the content of the array is arbitrary and may vary. It is useful when you need an array of a specific size and data type, but you intend to fill it later with data. Following is the syntax −

```
numpy.empty(shape, dtype=float, order='C')
```

> Unlike numpy.zeros() function and numpy.ones() function, which initialize array elements to zero and one respectively, the numpy.empty() function does not initialize the elements. Instead, it allocates the memory required for the array without setting any values.

### Example

In this example, we are using numpy.empty() function to create a 2-dimensional array (empty_array) with 2 rows and 3 columns −

```
import numpy as np

empty_array = np.empty((2, 3))
print(empty_array)
```

The output obtained is as shown below −

```
[[1.13750619e-313 0.00000000e+000 0.00000000e+000]
[0.00000000e+000 0.00000000e+000 0.00000000e+000]]
```

Unlike numpy.zeros(), this function initializes the array with uninitialized values, which could be any random data left in memory, making it suitable for cases where immediate initialization is not required.

## Using numpy.full() Function

Using the numpy.full() function, we can create an array with a desired shape and set all the elements in  it to a specific value. Following is the syntax −

```
numpy.full(shape, fill_value, dtype=None, order='C')
```

### Example

In the following example, we are using the numpy.full() function to create a 2-dimensional array with dimensions 2x3, filled entirely with the value 5 −

```
import numpy as np

array1 = np.full((2, 3), 5)
print(array1)
```

After executing the above code, we get the following output −

```
[[5 5 5]
 [5 5 5]]
```

## Functions Used for Creation of NumPy Arrays

In the
**NumPy**module, there are various ways to create NumPy arrays that includes, basic creation methods, creation by reshaping and modifying data, creation using sequences, and creation using random functions. Following are the different functions used to create NumPy arrays −
### Basic Array Creation

Following are the basic functions for creation of array −
Sr.No.Functions & Description1[array()](/numpy/numpy_array_function.htm)
used to create a numpy array
2[asarray()](/numpy/numpy_asarray_function.htm)
Convert the input to an array
3[asanyarray()](/numpy/numpy_asanyarray_function.htm)
Convert the input to an ndarray, but pass ndarray subclasses through
4[copy()](/numpy/numpy_copy_function.htm)
Return an array copy of the given object

### Array Creation with Specific Shapes and Data

Following are the functions used to create an array with specified shapes and data −
Sr.No.Functions & Description1[zeros()](/numpy/numpy_zeros_function.htm)
Return a new array of given shape and type, filled with zeros
2[ones()](/numpy/numpy_ones_function.htm)
Return a new array of given shape and type, filled with ones
3[empty()](/numpy/numpy_empty_function.htm)
Return a new array of given shape and type, without initializing entries
4[full()](/numpy/numpy_full_function.htm)
Return a new array of given shape and type, filled with fill_value

### Array Creation with Sequences

Following are the functions used to create an array with sequences −
Sr.No.Functions & Description1[arange()](/numpy/numpy_arange_function.htm)
Return evenly spaced values within a given interval
2[linspace()](/numpy/numpy_linspace_function.htm)
Return evenly spaced numbers over a specified interval.
3[logspace()](/numpy/numpy_logspace_function.htm)
Return numbers spaced evenly on a log scale.

### Special Arrays

Following are the special functions to create an array −
Sr.No.Functions & Description1[eye()](/numpy/numpy_eye_function.htm)
Return a 2-D array with ones on the diagonal and zeros in all other positions
2[identity()](/numpy/numpy_identity_function.htm)
Return the identity array.
3[diag()](/numpy/numpy_diag_function.htm)
Extract a diagonal or construct a diagonal array
4[fromfunction()](/numpy/numpy_fromfunction_function.htm)
Construct an array by executing a function over each coordinate
5[fromfile()](/numpy/numpy_fromfile_function.htm)
Construct an array from data in a text or binary file

### Random Arrays

Following are the random functions to create an array −
Sr.No.Functions & Description1[random.rand()](/numpy/numpy_random_rand_function.htm)
Random values in a given shape
2[random.randn()](/numpy/numpy_random_randn_function.htm)
Return a sample from the standard normal distribution
3[random.randint()](/numpy/numpy_random_randint_function.htm)
Return random integers from low (inclusive) to high (exclusive)
4[random.random()](/numpy/numpy_random_random_function.htm)
Return random floats in the half-open interval [0.0, 1.0)
5[random.choice()](/numpy/numpy_random_choice_function.htm)
Generates a random sample from a given 1-D array

### Structured Array

Following are the structured functions used to create an array −
Sr.No.Functions & Description1[zeros_like()](/numpy/numpy_zeros_like_function.htm)
Return an array of zeros with the same shape and type as a given array
2[ones_like()](/numpy/numpy_ones_like_function.htm)
Return an array of ones with the same shape and type as a given array.
3[empty_like()](/numpy/numpy_empty_like_function.htm)
Return a new array with the same shape and type as a given array
4[full_like()](/numpy/numpy_full_like_function.htm)
Return a full array with the same shape and type as a given array

---

## 6. NumPy - Array Manipulation

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_manipulation.htm](https://www.tutorialspoint.com/numpy/numpy_array_manipulation.htm)*

---

---
[Previous](/numpy/numpy_array_creation_routines.htm)[Quiz](/numpy/quiz_on_numpy_array_manipulation.htm)[Next](/numpy/numpy_array_from_existing_data.htm)
Several routines are available in NumPy package for manipulation of elements in ndarray object. They can be classified into the following types −

## Changing Shape

In NumPy, to change shape is to alter the shape of arrays without changing their data −
Sr.No.Shape & Description1[reshape()](/numpy/numpy_reshape.htm)
Gives a new shape to an array without changing its data
2[flat()](/numpy/numpy_ndarray_flat.htm)
A 1-D iterator over the array
3[flatten()](/numpy/numpy_ndarray_flatten.htm)
Returns a copy of the array collapsed into one dimension
4[ravel()](/numpy/numpy_ndarray_ravel.htm)
Returns a contiguous flattened array
5[pad()](/numpy/numpy_pad_function.htm)
Returns a padded array with shape increased according to pad_width

## Transpose Operations

The NumPy transpose operations swap rows and columns in 2D arrays or rearrange axes in higher-dimensional arrays −
Sr.No.Operation & Description1[transpose](/numpy/numpy_transpose.htm)
Permutes the dimensions of an array
2[ndarray.T](/numpy/numpy_ndarray_t.htm)
Same as self.transpose()
3[rollaxis](/numpy/numpy_rollaxis.htm)
Rolls the specified axis backwards
4[swapaxes](/numpy/numpy_swapaxes.htm)
Interchanges the two axes of an array
5[moveaxis()](/numpy/numpy_moveaxis_function.htm)
Move axes of an array to new positions

## Changing Dimensions

Changing dimensions of arrays in NumPy involves reshaping or restructuring arrays to fit specific requirements without altering the data −
Sr.No.Dimension & Description1[broadcast](/numpy/numpy_broadcast.htm)
Produces an object that mimics broadcasting
2[broadcast_to](/numpy/numpy_broadcast_to.htm)
Broadcasts an array to a new shape
3[expand_dims](/numpy/numpy_expand_dims.htm)
Expands the shape of an array
4[squeeze](/numpy/numpy_squeeze.htm)
Removes single-dimensional entries from the shape of an array

## Joining Arrays

Joining arrays in NumPy concatenate multiple arrays along specified axes −
Sr.No.Array & Description1[concatenate](/numpy/numpy_concatenate.htm)
Joins a sequence of arrays along an existing axis
2[stack](/numpy/numpy_stack.htm)
Joins a sequence of arrays along a new axis
3[hstack](/numpy/numpy_hstack.htm)
Stacks arrays in sequence horizontally (column wise)
4[vstack](/numpy/numpy_vstack.htm)
Stacks arrays in sequence vertically (row wise)
5[dstack()](/numpy/numpy_dstack_function.htm)
Stack arrays in sequence depth wise (along third axis).
6[column_stack()](/numpy/numpy_column_stack_function.htm)
Stacks arrays in sequence vertically (row wise)
7[row_stack()](/numpy/numpy_row_stack_function.htm)
Stacks arrays in sequence vertically (row wise)

## Splitting Arrays

Splitting arrays in NumPy splits arrays into smaller arrays along specified axes −
Sr.No.Array & Description1[split](/numpy/numpy_split.htm)
Splits an array into multiple sub-arrays
2[hsplit](/numpy/numpy_hsplit.htm)
Splits an array into multiple sub-arrays horizontally (column-wise)
3[vsplit](/numpy/numpy_vsplit.htm)
Splits an array into multiple sub-arrays vertically (row-wise)
4[dsplit()](/numpy/python_numpy_dsplit_function.htm)
Split array into multiple sub-arrays along the 3rd axis (depth)
5[array_split](/numpy/numpy_array_split_function.htm)
Split an array into multiple sub-arrays

## Adding / Removing Elements

Adding or removing elements in NumPy append elements to arrays or remove elements −
Sr.No.Element & Description1[resize](/numpy/numpy_resize.htm)
Returns a new array with the specified shape
2[append](/numpy/numpy_append.htm)
Appends the values to the end of an array
3[insert](/numpy/numpy_insert.htm)
Inserts the values along the given axis before the given indices
4[delete](/numpy/numpy_delete.htm)
Returns a new array with sub-arrays along an axis deleted
5[unique](/numpy/numpy_unique.htm)
Finds the unique elements of an array

## Repeating and Tiling Arrays

In Numpy, Repeating and tiling arrays are techniques used to create larger arrays by duplicating the elements of an existing array in various patterns −
Sr.No.Array & Description1[repeat()](/numpy/numpy_repeat_function.htm)
Repeat each element of an array after themselves
2[tile()](/numpy/numpy_tile_function.htm)
Construct an array by repeating A the number of times given by reps

## Rearranging Elements

In NumPy, elements of an array can be rearranged using various methods to achieve the desired order or structure. Following are the common operations −
Sr.No.Array & Description1[flip()](/numpy/numpy_flip_function.htm)
Reverse the order of elements in an array along the given axis
2[fliplr()](/numpy/numpy_fliplr_function.htm)
Reverse the order of elements along axis 1 (left/right)
3[flipud()](/numpy/numpy_flipud_function.htm)
Reverse the order of elements along axis 0 (up/down)
4[roll()](/numpy/numpy_roll_function.htm)
Roll array elements along a given axis

## Sorting and Searching

NumPy offers powerful tools for sorting and searching within arrays, enabling efficient data manipulation and analysis −
Sr.No.Array & Description1[sort()](/numpy/numpy_sort_function.htm)
Return a sorted copy of an array
2[argsort()](/numpy/numpy_argsort_function.htm)
Returns the indices that would sort an array
3[lexsort()](/numpy/numpy_lexsort_function.htm)
Perform an indirect stable sort using a sequence of keys
4[searchsorted()](/numpy/numpy_searchsorted_function.htm)
Find indices where elements should be inserted to maintain order
5[argmax()](/numpy/numpy_argmax_function.htm)
Returns the indices of the maximum values along an axis
6[argmin()](/numpy/numpy_argmin_function.htm)
Returns the indices of the minimum values along an axis
7[nonzero()](/numpy/numpy_nonzero_function.htm)
Return the indices of the elements that are non-zero
8[where()](/numpy/numpy_where_function.htm)
Return elements chosen from x or y depending on condition

## Set Operations

Set operations in NumPy involve performing mathematical set operations on arrays, such as union, intersection, difference, and checking for unique elements. These operations are particularly useful for handling and analyzing distinct values within datasets −
Sr.No.Array & Description1[in1d()](/numpy/numpy_in1d_function.htm)
Test whether each element of a 1-D array is also present in a second array
2[intersect1d()](/numpy/numpy_intersect1d_function.htm)
Find the intersection of two arrays
3[setdiff1d()](/numpy/numpy_setdiff1d_function.htm)
Find the set difference of two arrays and returns the unique values in ar1 that are not in ar2
4[setxor1d()](/numpy/numpy_setxor1d_function.htm)
Find the set exclusive-or of two arrays and returns the sorted, unique values that are in only one (not both) of the input arrays
5[union1d()](/numpy/numpy_union1d_function.htm)
Find the union of two arrays and returns the unique, sorted array of values that are in either of the two input arrays.

## Other Arrays Operations

Following are the a=other arryas opertions in Numpy −
Sr.No.Array & Description1[clip()](/numpy/numpy_clip_function.htm)
Clip (limit) the values in an array.
2[round()](/numpy/numpy_round_function.htm)
Evenly round to the given number of decimals
3[diagonal()](/numpy/numpy_diagonal_function.htm)
Return specified diagonals
4[trace()](/numpy/numpy_trace_function.htm)
Return the sum along diagonals of the array
5[take()](/numpy/numpy_take_function.htm)
Take elements from an array along an axis
6[put()](/numpy/numpy_put_function.htm)
Replaces specified elements of an array with given values
7[choose()](/numpy/numpy_choose_function.htm)
Construct an array from an index array and a list of arrays to choose from

---

## 7. NumPy - Array From Existing Data

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_from_existing_data.htm](https://www.tutorialspoint.com/numpy/numpy_array_from_existing_data.htm)*

---

---
[Previous](/numpy/numpy_array_manipulation.htm)[Quiz](/numpy/quiz_on_numpy_array_from_existing_data.htm)[Next](/numpy/numpy_array_from_numerical_ranges.htm)
## Creating Array From Existing Data in NumPy

You can create arrays from existing data in NumPy by initializing NumPy arrays using data structures that already exist in Python, or can be converted to a format compatible with NumPy. Following are a few common ways to achieve this −

- Using numpy.asarray() Function
- Using numpy.frombuffer() Function
- Using numpy.fromiter() Function
- From Python Lists
- From Nested Lists
- From Python Tuples
- From Existing NumPy Arrays
- Using Range Objects
NumPy's ability to work fast and perform complex operations on arrays is really important in fields like data handling and performing scientific calculations.

## Using numpy.asarray() Function

The numpy.asarray() function is used to convert various Python objects into NumPy arrays. These objects includes Python lists, tuples, other arrays, and even scalar values.

This function ensures that the result is always a NumPy array, making it convenient for data manipulation and numerical computations. Following is the syntax −

```
numpy.asarray(a, dtype=None, order=None)
```

Where,

- **a −**It is the input data, which can be a list, tuple, array, or any object that can be converted to an array.
- **dtype (optional) −**Desired data type of the array. If not specified, NumPy determines the data type based on the input.
- **order (optional) −**Specifies whether to store the array in row-major (C) or column-major (F) order. Default is None, which means NumPy decides based on the input.
### Example: Convert Python List to NumPy Array

In the following example, we are using the numpy.asarray() function to convert a Python list into a NumPy array −

```
import numpy as np

# Convert list to array
my_list = [1, 2, 3, 4, 5]
arr_from_list = np.asarray(my_list)

print("Array from list:",arr_from_list)
```

Following is the output obtained −

```
Array from list: [1 2 3 4 5]
```

### Example: Preserve Data Type

In here, we are converting the Python list which contains elements of different data types (int, float, bool, str), into a NumPy array using the numpy.asarray() function −

```
import numpy as np

# Convert list with different data types to array
my_mixed_list = [1, 2.5, True, 'hello']
arr_from_mixed = np.asarray(my_mixed_list)

print("Array from mixed list:", arr_from_mixed)
```

This will produce the following result −

```
Array from mixed list: ['1' '2.5' 'True' 'hello']
```

## Using numpy.frombuffer()Function

The numpy.frombuffer() function creates an array from a buffer object, such as bytes objects or byte arrays. This is useful when working with raw binary data or memory buffers.

This function interprets the buffer object as one-dimensional array data. It allows you to specify the data type of the elements in the resulting array.Following is the syntax −

```
numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)
```

Where,

- **buffer −**It is the buffer object containing the data to be interpreted as an array.
- **dtype (optional) −**It is the desired data type of the elements in the resulting array. Default is float.
- **count (optional) −**It is the number of items to read from the buffer. Default is -1, which means all data is read.
- **offset (optional) −**It is the starting position within the buffer to begin reading data. Default is 0.
### Example

In this example, we are using the numpy.frombuffer() function to interpret the bytes object "my_bytes" as a one-dimensional array of bytes −

```
import numpy as np

# Create bytes object
my_bytes = b'hello world'

# Create array from bytes object
arr_from_bytes = np.frombuffer(my_bytes, dtype='S1')

print("Array from bytes object:",arr_from_bytes)
```

The resulting NumPy array contains each byte of the original bytes object 'hello world' −

```
Array from bytes object: [b'h' b'e' b'l' b'l' b'o' b' ' b'w' b'o' b'r' b'l' b'd']
```

## Using numpy.fromiter() Function

The numpy.fromiter() function creates a new one-dimensional array from an iterable object. It iterates over the iterable object, converting each element into an array element. Following is the syntax −

```
numpy.fromiter(iterable, dtype, count=-1)
```

Where,

- **iterable −**The iterable object that yields elements one by one.
- **dtype −**The data type of the elements in the resulting array.
- **count (optional) −**The number of items to read from the iterable. Default is -1, which means all items are read.
### Example

In the example below, we are using the numpy.fromiter() function to create a NumPy array "gen_array" from the generator "my_generator" that yields numbers from 0 to 4 −

```
import numpy as np

# Generator function that yields numbers
def my_generator(n):
   for i in range(n):
      yield i

# Create array from generator
gen_array = np.fromiter(my_generator(5), dtype=int)

print("Array from generator:",gen_array)
```

In the resulting array, each element corresponds to a value yielded by the generator function converted to integers −

```
Array from generator: [0 1 2 3 4]
```

> Generators in Python are functions that generate a sequence of values one at a time. NumPy provides np.fromiter() function to create arrays from generators.

## From Python Lists

One of the most common ways to create a NumPy array is by converting a Python list. This method provides the numpy.array() function or numpy.asarray() function to convert lists, which are commonly used data structures in Python, into NumPy arrays.

Following is the syntax −

```
numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
```

Where,

- **object −**The input data, which in this case is a Python list.
- **dtype (optional) −**Desired data type of the array. If not specified, NumPy interprets the data type from the input data.
- **copy (optional) −**If True, ensures that a copy of the input data is made. If False, avoids unnecessary copies when possible.
- **order (optional) −**Specifies the memory layout order of the array. 'C' for row-major (C-style), 'F' for column-major (Fortran-style), and 'K' for the layout of the input array (default).
- **subok (optional) −**If True, subclasses are passed through; otherwise, the returned array will be forced to be a base-class array.
- **ndmin (optional) −**Specifies the minimum number of dimensions the resulting array should have.
### Example

In the example below, we are converting the Python list "my_list" containing integers into a NumPy array using the numpy.array() function −

```
import numpy as np

# Convert list to array
my_list = [1, 2, 3, 4, 5]
arr_from_list = np.array(my_list)

print("Array from list:",arr_from_list)
```

After executing the above code, we get the following output −

```
Array from list: [1 2 3 4 5]
```

## From Nested Lists

Nested lists in Python are lists within lists, which can represent multi-dimensional data structures. NumPy provides the array() function to convert these nested lists into multi-dimensional arrays.

### Example: Convert a Nested List to a 2D NumPy Array

In this example, the nested list "nested_list" represents a 2D structure (list of lists). The array() function converts it into a 2D NumPy array "arr_from_nested_list" −

```
import numpy as np

# Convert nested list to array 
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr_from_nested_list = np.array(nested_list)

print("Array from nested list:")
print(arr_from_nested_list)
```

The resulting array retains the 2D structure and contains the same elements as the original nested list as shown in the output below −

```
Array from nested list:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

### Example: Convert a Nested List with Different Data Types

In here, the nested list contains elements of different data types (integers, floats, booleans, and strings). The array() function converts all elements to strings, resulting in a homogeneous 2D array with string data type −

```
import numpy as np

# Convert nested list with different data types to array
nested_mixed_list = [[1, 2.5], [True, 'hello']]
arr_from_nested_mixed_list = np.array(nested_mixed_list)

print("Array from nested mixed list:")
print(arr_from_nested_mixed_list)
```

The result produced is as follows −

```
Array from nested mixed list:
[['1' '2.5']
 ['True' 'hello']]
```

## From Python Tuples

Python tuples are another commonly used data structure that can be converted into NumPy arrays. Like lists, tuples can be used to store multiple items, but they are immutable, meaning their content cannot be changed after creation.

It can be used to represent both one-dimensional and multi-dimensional data using the numpy.array() function.

### Example

In the following example, we are converting the Python tuple containing integers into a NumPy array using the array() function −

```
import numpy as np

# Convert tuple to array
my_tuple = (1, 2, 3, 4, 5)
arr_from_tuple = np.array(my_tuple)

print("Array from tuple:",arr_from_tuple)
```

We get the output as shown below −

```
Array from tuple: [1 2 3 4 5]
```

## From Existing NumPy Arrays

NumPy provides several methods to create new arrays from existing NumPy arrays. They are −

- numpy.copy() Function
- numpy.asarray() Function
- numpy.view() Function
- numpy.reshape() Function
- Slicing
This can be useful for various tasks, such as copying data, changing data types, or creating new arrays with specific attributes derived from the original array.

### Example: Using numpy.copy() Function

The numpy.copy() function creates a new array that is a copy of the original array. This ensures that any modifications to the new array do not affect the original array −

```
import numpy as np

# Original array
original_array = np.array([1, 2, 3, 4, 5])

# Create a copy of the array
copied_array = np.copy(original_array)

print("Original array:",original_array)
print("Copied array:",copied_array)
```

Following is the output obtained −

```
Original array: [1 2 3 4 5]
Copied array: [1 2 3 4 5]
```

### Example: Using numpy.asarray() Function

The numpy.asarray() function converts the input to an array, but if the input is already an array, it does not create a copy unless necessary (e.g., if a different data type is specified) −

```
import numpy as np

# Original array
original_array = np.array([1, 2, 3, 4, 5])

# Create an array from the existing array
new_array = np.asarray(original_array, dtype=float)

print("Original array:",original_array)
print("New array:",new_array)
```

This will produce the following result −

```
Original array: [1 2 3 4 5]
New array: [1. 2. 3. 4. 5.]
```

### Example: Using numpy.view() Function

The numpy.view() function creates a new array object that looks at the same data as the original array. This can be useful for viewing the data with a different data type −

```
import numpy as np

# Original array
original_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)

# Create a view of the array with a different dtype
viewed_array = original_array.view(dtype=np.float32)

print("Original array:",original_array)
print("Viewed array with dtype float32:",viewed_array)
```

Following is the output of the above code −

```
Original array: [1 2 3 4 5]
Viewed array with dtype float32: [1.e-45 3.e-45 4.e-45 6.e-45 7.e-45]
```

### Example: Using numpy.reshape() Function

The numpy.reshape() function reshapes an existing array into a new shape without changing its data −

```
import numpy as np

# Original array
original_array = np.array([1, 2, 3, 4, 5, 6])

# Reshape the array to 2x3
reshaped_array = original_array.reshape((2, 3))

print("Original array:",original_array)
print("Reshaped array (2x3):",reshaped_array)
```

The output obtained is as shown below −

```
Original array: [1 2 3 4 5 6]
Reshaped array (2x3): [[1 2 3]
 [4 5 6]]
```

### Example: Using Slicing

Slicing an existing array creates a new array that is a subset of the original array −

```
import numpy as np

# Original array
original_array = np.array([1, 2, 3, 4, 5])

# Slice the array to get a subarray
sliced_array = original_array[1:4]

print("Original array:",original_array)
print("Sliced array (elements 1 to 3):",sliced_array)
```

The output obtained is as shown below −
Original array: [1 2 3 4 5]
Sliced array (elements 1 to 3): [2 3 4]
## Using Range Objects

Python
**range**object generates numbers within a specified range and can be converted into a NumPy array using the numpy.array() function or numpy.fromiter() function. This is useful when you need to create large sequences without explicitly storing all numbers in memory first.
### Example

In this example, the range() object generates numbers from 1 to 9. The numpy.array() function converts this range object into a NumPy array −

```
import numpy as np

# Create a range object
my_range = range(1, 10)
# Convert range object to array
arr_from_range = np.array(my_range)

print("Array from range object:",arr_from_range)
```

After executing the above code, we get the following output −

```
Array from range object: [1 2 3 4 5 6 7 8 9]
```

---

## 8. NumPy - Array From Numerical Ranges

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_from_numerical_ranges.htm](https://www.tutorialspoint.com/numpy/numpy_array_from_numerical_ranges.htm)*

---

---
[Previous](/numpy/numpy_array_from_existing_data.htm)[Quiz](/numpy/quiz_on_numpy_array_from_numerical_ranges.htm)[Next](/numpy/numpy_iterating_over_array.htm)
## Array From Numerical Ranges in NumPy

Creating arrays from numerical ranges in NumPy refers to generating arrays that contain sequences of numbers within a specified range. NumPy provides several functions to create such arrays, they are as follows −

- Using numpy.arange() Function
- Using numpy.linspace() Function
- Using numpy.logspace() Function
- Using numpy.meshgrid() Function
## Using numpy.arange() Function

The numpy.arange() function creates an array by generating a sequence of numbers based on specified start, stop, and step values. It is similar to Python's built-in range() function but returns a NumPy array. Following is the syntax −

```
numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)
```

Where,

- **start (optional) −**The starting value of the interval. Default is 0.
- **stop −**The end value of the interval (not included).
- **step (optional) −**The spacing between values. Default is 1.
- **dtype (optional) −**The desired data type for the array. If not given, NumPy interprets the data type from the input values.
### Example: Basic Usage

In the following example, we are using the numpy.arange() function to generate an array starting from 0 up to (but not including) 10 −

```
import numpy as np

# Create an array from 0 to 9
arr = np.arange(10)
print("Array using arange():", arr)
```

Following is the output obtained −

```
Array using arange(): [0 1 2 3 4 5 6 7 8 9]
```

### Example: Specifying Start, Stop, and Step

In here, we are generating an array starting from 1, up to (but not including) 10, with a step of 2 using the numpy.arange() function −

```
import numpy as np

# Create an array from 1 to 9 with a step of 2
arr = np.arange(1, 10, 2)
print("Array with start, stop, and step:", arr)
```

This will produce the following result −

```
Array with start, stop, and step: [1 3 5 7 9]
```

## Using numpy.linspace() Function

The numpy.linspace() function generates an array with evenly spaced values over a specified interval. It is useful when you need a specific number of points between two values.

This function is similar to the arange() function. In this function, instead of step size, the number of evenly spaced values between the interval is specified. Following is the syntax −

```
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

Where,

- **start −**The starting value of the interval.
- **stop −**The end value of the interval.
- **num (optional) −**The number of evenly spaced samples to generate. Default is 50.
- **endpoint (optional) −**If True, stop is the last sample. If False, it is not included. Default is True.
- **retstep (optional) −**If True, returns (samples, step), where step is the spacing between samples. Default is False.
- **dtype (optional) −**The desired data type for the array. If not given, NumPy interprets the data type from the input values.
- **axis (optional) −**The axis in the result along which the samples are stored. Default is 0.
### Example: Basic Usage

In this example, we are using the numpy.linspace() function to generate an array of 10 evenly spaced values from 0 to 1 (inclusive by default) −

```
import numpy as np

# Create an array of 10 evenly spaced values from 0 to 1
arr = np.linspace(0, 1, 10)
print("Array using linspace():", arr)
```

Following is the output of the above code −

```
Array using linspace(): [0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556
 0.66666667 0.77777778 0.88888889 1.        ]
```

### Example: Excluding Endpoint

In here, we are generating an array of 10 evenly spaced values from 0 to just below 1, excluding the endpoint using the numpy.linspace() function −

```
import numpy as np

# Create an array 
arr = np.linspace(0, 1, 10, endpoint=False)
print("Array with endpoint=False:", arr)
```

The output obtained is as shown below −

```
Array with endpoint=False: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
```

### Example: Returning Step Size

Now, we are generating an array of 10 evenly spaced values from 0 to 1 (inclusive by default) and also returns the step size −

```
import numpy as np

# Create an array 
arr, step = np.linspace(0, 1, 10, retstep=True)
print("Array with step size:", arr)
print("Step size:", step)
```

After executing the above code, we get the following output −

```
Array with step size: [0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556
 0.66666667 0.77777778 0.88888889 1.        ]
Step size: 0.1111111111111111
```

## Using numpy.logspace() Function

The numpy.logspace() function generates an array with values that are evenly spaced on a log scale. This is useful for generating values that span several orders of magnitude. Following is the syntax −

```
numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
```

Where,

- **start −**The starting value of the sequence (as a power of base).
- **stop −**The end value of the sequence (as a power of base).
- **num (optional) −**The number of samples to generate. Default is 50.
- **endpoint (optional) −**If True, stop is the last sample. If False, it is not included. Default is True.
- **base (optional) −**The base of the log space. Default is 10.0.
- **dtype (optional) −**The desired data type for the array. If not given, NumPy interprets the data type from the input values.
- **axis (optional) −**The axis in the result along which the samples are stored. Default is 0.
### Example

In the example below, we are using the numpy.logspace() function to generate an array of 10 values evenly spaced on a logarithmic scale from 2
to 2with base 2 −
```
import numpy as np

# Create an array 
arr = np.logspace(1, 10, 10, base=2)
print("Array with base 2:", arr)
```

We get the output as shown below −

```
Array with base 2: [   2.    4.    8.   16.   32.   64.  128.  256.  512. 1024.]
```

## Using numpy.meshgrid() Function

The numpy.meshgrid() function generates coordinate matrices from coordinate vectors. This is useful for creating grids of points for evaluating functions over a 2D or 3D space. Following is the syntax −

```
numpy.meshgrid(*xi, copy=True, sparse=False, indexing='xy')
```

Where,

- ***xi −**1-D arrays representing the coordinates of a grid.
- **copy (optional) −**If True, a copy of the input arrays is made. Default is True.
- **sparse (optional) −**If True, a sparse grid is returned to save memory. Default is False.
- **indexing (optional) −**Specifies the Cartesian ('xy', default) or matrix ('ij') indexing convention.
### Example: Creating a 2D Grid

In the example below, we are using the numpy.meshgrid() function to generate coordinate matrices "X" and "Y" from 1D arrays "x" and "y", where X represents the x-coordinates and Y represents the y-coordinates of a 2D grid −

```
import numpy as np

# Create 1D arrays for x and y coordinates
x = np.arange(1, 4)
y = np.arange(1, 3)

# Generate coordinate matrices
X, Y = np.meshgrid(x, y)

print("X grid:")
print(X)
print("Y grid:")
print(Y)
```

Following is the output obtained −

```
X grid:
[[1 2 3]
 [1 2 3]]
Y grid:
[[1 1 1]
 [2 2 2]]
```

### Example: Creating a 3D Grid

Now, we are generateing coordinate matrices X, Y, and Z from 1D arrays x, y, and z for a 3D grid using matrix indexing ('ij') −

```
import numpy as np

# Create 1D arrays for x, y, and z coordinates
x = np.arange(1, 4)
y = np.arange(1, 3)
z = np.arange(1, 3)

# Generate coordinate matrices
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

print("X grid:")
print(X)
print("Y grid:")
print(Y)
print("Z grid:")
print(Z)
```

After executing the above code, we get the following output −

```
X grid:
[[[1 1]
  [1 1]]

 [[2 2]
  [2 2]]

 [[3 3]
  [3 3]]]
Y grid:
[[[1 1]
  [2 2]]

 [[1 1]
  [2 2]]

 [[1 1]
  [2 2]]]
Z grid:
[[[1 2]
  [1 2]]

 [[1 2]
  [1 2]]

 [[1 2]
  [1 2]]]
```

---

## 9. NumPy - Iterating Over Array

*Source: [https://www.tutorialspoint.com/numpy/numpy_iterating_over_array.htm](https://www.tutorialspoint.com/numpy/numpy_iterating_over_array.htm)*

---

---
[Previous](/numpy/numpy_array_from_numerical_ranges.htm)[Quiz](/numpy/quiz_on_numpy_iterating_over_array.htm)[Next](/numpy/numpy_reshaping_arrays.htm)
## Iterating Over Array in NumPy

Iterating over an array in NumPy refers to the process of accessing each element in the array one by one in a systematic manner. This is typically done using loops. Iteration is used to perform operations on each element, such as calculations, modifications, or checks.

NumPy provides several ways to iterate over arrays −

- Using a for Loop
- Using 'nditer()' Iterator Object
- Flat Iteration
- Iteration Order
- Controlling Iteration Order
- Broadcasting Iteration
- Using Vectorized Operations
- External Loop
- Modifying Array Values
## Using a for Loop

In NumPy, you can use basic Python
**for loops**to iterate over arrays. A for loop is a control flow statement used for iterating over a sequence (such as a list, tuple, dictionary, set, or string). It allows you to execute a block of code repeatedly for each element in the sequence.
### Iterating Over One-dimensional Arrays

A 1-dimensional array is basically a list of elements. Iterating over it is simple and similar to iterating over a regular Python list.
**Example**
In the following example, we create a NumPy array from a list of elements and iterate over each element using a for loop −

```
import numpy as np

# Create a 1-dimensional NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Iterate over the array
for element in arr:
   print(element)
```

Following is the output obtained −

```
1
2
3
4
5
```

### Iterating Over Multi-Dimensional Arrays

NumPy arrays can have any number of dimensions, commonly referred to as axes. For instance −

- A 1-dimensional array is a list of elements.
- A 2-dimensional array is like a matrix with rows and columns.
- A 3-dimensional array can be visualized as a collection of 2D matrices.**Example: Iterating Over a 2D Array**
When iterating over a 2D array, each iteration accesses one entire row of the array as shown in the example below −

```
import numpy as np

# Create a 2D NumPy array (3x3 matrix)
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Iterate over the array
for row in arr_2d:
   print(row)
```

This will produce the following result −

```
[1 2 3]
[4 5 6]
[7 8 9]
```
**Example: Iterating Over a 3D Array**
When iterating over a 3D array, each iteration accesses one entire 2D sub-array (matrix) as shown in the following example −

```
import numpy as np

# Create a 3D NumPy array (2x2x3)
arr_3d = np.array([[[1, 2, 3],
                    [4, 5, 6]],
                   [[7, 8, 9],
                    [10, 11, 12]]])

# Iterate over the array
for matrix in arr_3d:
   print(matrix)
```

Following is the output of the above code −

```
[[1 2 3]
 [4 5 6]]
[[ 7  8  9]
 [10 11 12]]
```

### Iterating Over Elements in Multi-Dimensional Arrays

Iterating over elements in multi-dimensional arrays in NumPy is a way to access each individual element, regardless of the dimension of the array. This process requires nested loops to traverse through each dimension of the array structure.
**Example**
In this example, we are using nested loops to iterate over each row (i) and column (j) to access and print each element using indexing −

```
import numpy as np

# Create a 2-dimensional NumPy array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Iterating over elements in the 2D array
# Iterate over rows
for i in range(arr_2d.shape[0]):   
   # Iterate over columns
   for j in range(arr_2d.shape[1]):  
      print(arr_2d[i, j])
```

The output obtained is as shown below −

```
1
2
3
4
5
6
7
8
9
```

### Iterating with Indices

Iterating over NumPy arrays using indices is used to access array elements by their specific positions within each dimension of the array.

In NumPy, arrays are indexed starting from
**0**for each dimension. Iterating with indices involves using nested loops to traverse through each dimension of the array and access elements using their specific indices.
This approach allows for more precise control over element access and manipulation compared to simple element iteration.
**Example**
In the following example, we iterate over each element of the 2D array using nested loops. We access and print each element's value along with its indices −

```
import numpy as np

# Create a 2-dimensional NumPy array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Get the dimensions of the array
rows, cols = arr_2d.shape

# Iterate over the array using indices
for i in range(rows):
   for j in range(cols):
      print(f"Element at ({i}, {j}): {arr_2d[i, j]}")
```

After executing the above code, we get the following output −

```
Element at (0, 0): 1
Element at (0, 1): 2
Element at (0, 2): 3
Element at (1, 0): 4
Element at (1, 1): 5
Element at (1, 2): 6
Element at (2, 0): 7
Element at (2, 1): 8
Element at (2, 2): 9
```

## Using 'nditer' Iterator Object

The
**nditer()**function in NumPy provides an efficient multidimensional iterator object that can be used to iterate over elements of arrays. It uses Python's standard iterator interface to visit each element of an array.
### Example

In this example, the nditer() function is used to iterate over all elements of the array "arr" in a flattened order, printing each element sequentially −

```
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Iterate using nditer
for x in np.nditer(arr):
   print(x)
```

The result produced is as follows −

```
1234
```

## Flat Iteration

Flat iteration refers to iterating over all elements of a multi-dimensional array as if it were one-dimensional. This approach is useful when you need to process or manipulate every single element in the array without considering its original shape or dimensions explicitly.

### Example

In this example, we are creating a 2D NumPy array. We then iterate over this array in a flattened sequence using np.nditer with the 'buffered' flag, printing each element sequentially −

```
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Flat iteration using nditer with 'buffered' flag
print("Iterating over the array:")
for x in np.nditer(arr, flags=['buffered']):
   print(x, end=' ')
```

We get the output as shown below −

```
Iterating over the array:
1 2 3 4 5 6
```

## Iteration Order

Iteration order in NumPy refers to the sequence in which elements of an array are accessed during iteration. By default, NumPy arrays are iterated over in a
**row-major**order, also known as**C-style**order.
This means that for multi-dimensional arrays, iteration starts from the first dimension (rows), iterating through all elements along the last dimension (columns).

### Example

In this example, we iterate over an array using numpy.nditer() function, accessing each element in the default row-major order (C-style), and printing them sequentially −

```
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Iterate over the array
print("Default Iteration Order (C-style, row-major):")
for x in np.nditer(arr):
   print(x, end=' ')
```

Following is the output obtained −

```
Default Iteration Order (C-style, row-major):
1 2 3 4 5 6
```

## Controlling Iteration Order

Controlling iteration order in NumPy allows you to specify how elements of an array are accessed during iteration. NumPy provides options to control the iteration order based on memory layout and performance considerations −

- **Fortran-style Order −**Iterates over elements column-wise (column-major order).
```
for x in np.nditer(arr, order='F'):
    ...
```

- **External Loop −**Maintains the inner dimensions intact while iterating over the outer dimensions.
```
for row in np.nditer(arr, flags=['external_loop']):
    ...
```

### Example

In this example, we iterate over an array using numpy.nditer() function, accessing each element in the  Fortran-style order (F-style) −

```
import numpy as np

# array
arr = np.array([[1, 2], [3, 4]])

# Iterate in specified order (F-style)
for x in np.nditer(arr, order='F'):
   print(x)
```

This will produce the following result −

```
1
3
2
4
```

## Broadcasting Iteration

Broadcasting iteration in NumPy refers to iterating over multiple arrays simultaneously, where the arrays are broadcasted to have compatible shapes.

This allows element-wise operations to be applied efficiently across arrays without explicitly aligning their dimensions.

### Example

In the following example, we are demonstrating broadcasting iteration in NumPy by iterating over two arrays, arr1 and arr2, simultaneously.

Each pair of corresponding elements from "arr1" and "arr2" is summed using nditer() function, showing element-wise operations without explicit alignment of array dimensions −

```
import numpy as np

# arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([10, 20, 30])

# Broadcasting addition operation
print("Broadcasting Iteration:")
for x, y in np.nditer([arr1, arr2]):
   print(x + y, end=' ')
```

Following is the output of the above code −

```
Broadcasting Iteration:
11 22 33
```

## Using Vectorized Operations

Vectorized operations in NumPy refer to performing operations on entire arrays at once, rather than iterating over individual elements.

### Example

In the example below, we are demonstrating vectorized operations in NumPy by performing element-wise addition on two arrays, arr1 and arr2. This is achieved simply by using the
**+**operator between the arrays −
```
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

# Vectorized addition operation
result = arr1 + arr2

print("Vectorized Operations:")
print("Result of addition:", result)
```

The output obtained is as shown below −

```
Vectorized Operations:Result of addition: [11 22 33 44]
```

## External Loop

In NumPy, the concept of an
**external loop**refers to iterating over arrays while maintaining certain dimensions intact. This allows you to iterate over arrays with nested structures, without collapsing the inner dimensions into a single sequence.
The
**numpy.nditer()**function, when used with the**external_loop**flag, allows iterating through array elements while preserving the array's row structure. This ensures that each row is processed individually, demonstrating how the integrity of dimensions is maintained throughout the iteration process.
### Example

In the following example, we are illustrating external loop iteration in NumPy by iterating over a 2D array −

```
import numpy as np

# array with multiple dimensions
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# External loop iteration
print("External Loop:")
for row in np.nditer(arr, flags=['external_loop']):
   print(row)
```

After executing the above code, we get the following output −

```
External Loop:[1 2 3 4 5 6]
```

## Modifying Array Values

Modifying array values in NumPy is used to directly assign new values to specific elements or slices within an array. This helps in updating data in-place without needing to recreate the entire array.

The
**nditer**object has another optional parameter called**op_flags**. Its default value is read-only, but can be set to read-write or write-only mode. This will enable modifying array elements using this iterator.
### Example

In the example below, we are modifying the array values in NumPy using the nditer() function. By setting the "op_flags" parameter to 'readwrite', the iterator multiplies each element of the array arr by 2 −

```
import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Modify array elements
with np.nditer(arr, flags=['buffered'], op_flags=['readwrite']) as it:
   for x in it:
      # Multiply each element by 2
      x[...] = x * 2  

print("Modified Array Example:",arr)
```

The result produced is as follows −

```
Modified Array Example: [ 2  4  6  8 10]
```

---

## 10. NumPy - Reshaping Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_reshaping_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_reshaping_arrays.htm)*

---

---
[Previous](/numpy/numpy_iterating_over_array.htm)[Quiz](/numpy/quiz_on_numpy_reshaping_arrays.htm)[Next](/numpy/numpy_concatenating_arrays.htm)
## Reshaping NumPy Array

By reshaping a NumPy array, we mean to change its shape, i.e., modifying the number of elements along each dimension while keeping the total number of elements the same. In other words, the product of the dimensions in the new shape must equal the product of the dimensions in the original shape.

For instance, an array of shape (6,) can be reshaped to (2, 3) or (3, 2), but not to (2, 2) since 6 elements cannot fit into a 2x2 array.

## Reshaping 1D Array to 2D Array

We can reshape a 1-D array to a 2-D array in NumPy using the reshape() function. This is used to organize linear data into a matrix form.

The reshape() function changes the shape of an existing array without changing its data. Following is the syntax −

```
numpy.reshape(array, newshape)
```

Where,

- **array −**The array you want to reshape.
- **newshape −**The shape you want to give the array. It can be an integer or a tuple of integers. One dimension can be -1, which means it will be presumed from the length of the array and the remaining dimensions.
### Example: Basic Reshaping

In the following example, we are reshaping a 1D array "arr" with "6" elements into a 2D array using the reshape() function −

```
import numpy as np

# Original 1-D array
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape to 2-D array 
reshaped_arr = arr.reshape((2, 3))
print("1-D to 2-D Array (2x3):")
print(reshaped_arr)
```

Following is the output obtained −

```
1-D to 2-D Array (2x3):
[[1 2 3]
 [4 5 6]]
```

### Example: Practical Use Case of Reshaping

Imagine you have a list of test scores for students in a class, and you want to organize them into a table where each row represents a student, and each column represents a different test. You can do this by reshaping the 1D array of scores into a 2D array −

```
import numpy as np
# Original 1-D array of test scores
scores = np.array([85, 90, 78, 92, 88, 76])

# Reshape into a 2-D array where each row is a student's scores
scores_matrix = scores.reshape((2, 3))
print("Scores Matrix (2 students, 3 tests each):")
print(scores_matrix)
```

This will produce the following result −

```
Scores Matrix (2 students, 3 tests each):
[[85 90 78]
 [92 88 76]]
```

## Reshaping 1D Array to 3D Array

We can also reshape a 1-D array to a 3-D array in NumPy using the reshape() function. This helps you to represent data with more complex structures such as multi-channel images (e.g., RGB images), time-series data across different channels, or volumetric data.

### Example

In this example, we are reshaping a 1-D array "arr" with "12" elements into a 3-D array using the reshape() function −

```
import numpy as np

# Original 1-D array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Reshape to 3-D array
reshaped_arr = arr.reshape((2, 2, 3))
print("1-D to 3-D Array (2x2x3):")
print(reshaped_arr)
```

Following is the output of the above code −

```
1-D to 3-D Array (2x2x3):
[[[ 1  2  3]
  [ 4  5  6]]

 [[ 7  8  9]
  [10 11 12]]]
```

## Reshaping ND Array to 1D Array

Reshaping an N-Dimensional (N-D) array to a 1-Dimensional (1-D) array in NumPy is a process of flattening or collapsing the multi-dimensional array into a single linear array. We can achieve this as well using the reshape() function.

Flattening complex multi-dimensional arrays into a 1-D format simplifies certain data processing tasks and make the data easier to handle and analyse.

### Example

In the example below, we are reshaping a 2-D array "arr" with shape (2, 3) into a 1-D array using the reshape() function with -1 as the argument −

```
import numpy as np

# Original 2-D array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Reshape to 1-D array
reshaped_arr = arr.reshape(-1)
print("Reshaped Array:", reshaped_arr)
```

The output obtained is as shown below −

```
Reshaped Array:[1 2 3 4 5 6]
```

## Reshaping Unknown Dimension Arrays

You can reshape an array with an unknown dimension using the reshape() function in NumPy. By passing
**-1**as an argument to reshape() function, NumPy automatically calculates the size of that dimension based on the total number of elements in the array and the other specified dimensions.
This helps you to reshape arrays without explicitly computing the exact size of every dimension.

### Example

In the following example, we are reshaping a 1-D array arr with "12" elements into a "3-D" array using the reshape() function, specifying one of the dimensions as "-1" −

```
import numpy as np

# Original 1-D array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Reshape to 3-D array with one unknown dimension
reshaped_arr = arr.reshape((2, 2, -1))
print("Reshaped Array with Unknown Dimension:")
print(reshaped_arr)
```

After executing the above code, we get the following output −

```
Reshaped Array with Unknown Dimension:
[[[ 1  2  3]
  [ 4  5  6]][[ 7  8  9]
  [10 11 12]]]
```

## Error Occurrence while Reshaping Array

Reshaping arrays in NumPy can sometimes lead to errors, especially when the total number of elements does not match the product of the specified dimensions. It is important to ensure that the new shape is compatible with the number of elements in the array.

### Common Errors while Reshaping Array

Following are the most common errors that occur while reshaping an array in NumPy −

- **ValueError: Total size of new array must be unchanged −**This error occurs when the number of elements in the original array does not match the product of the dimensions specified for reshaping.
For example, trying to reshape a 1-D array of 10 elements into a 3x3 matrix (reshape((3, 3))) will raise this error because 3  3 = 9 which is different from 10.

- **ValueError: cannot reshape array of size X into shape (Y, Z) −**This error indicates that the original array size (X) is not compatible with the specified shape (Y, Z).
For instance, trying to reshape a 1-D array of size 10 into a 2x5 matrix (reshape((2, 5))) will raise this error because 2  5 = 10, but the array needs to be 2-dimensional to fit the new shape.

- **TypeError: 'numpy.ndarray' object cannot be interpreted as an integer −**This error occurs when the dimensions provided for reshaping are not integers or are incorrectly specified. Ensure that all dimensions passed to reshape() function are valid integers and that they correctly represent the new shape.
### Handling Errors while Reshaping Array

Following are the ways to handle errors that occur while reshaping an array in NumPy −

- **Check Array Size**Before reshaping, verify the size of your original array using array.size() function and ensure it matches the product of the new shape dimensions.
- **Use -1 for Unknown Dimensions**When reshaping, if one dimension is unknown, use -1 to let NumPy calculate it automatically based on the total number of elements in the array.
- **Catch Exceptions**Wrap your reshaping code in a try-except block to catch potential errors and handle them gracefully. This can prevent your program from crashing and allow for appropriate error messaging or fallback actions.**Example**
In the following example, we attempt to reshape a 1D array "arr" with 5 elements into a "2x3" 2D array, which results in a ValueError because the total number of elements does not match the specified shape −

```
import numpy as np

# Original 1-D array
arr = np.array([1, 2, 3, 4, 5])

try:
   # Attempt to reshape to an incompatible shape
   reshaped_arr = arr.reshape((2, 3))
except ValueError as e:
   print("Error Occurred During Reshaping:")
   print(e)
```

The error obtained is as follows −

```
Error Occurred During Reshaping:
cannot reshape array of size 5 into shape (2,3)
```

---

## 11. NumPy - Concatenating Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_concatenating_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_concatenating_arrays.htm)*

---

---
[Previous](/numpy/numpy_reshaping_arrays.htm)[Quiz](/numpy/quiz_on_numpy_concatenating_arrays.htm)[Next](/numpy/numpy_stacking_arrays.htm)
## Concatenating NumPy Array

By concatenating a NumPy array, we mean to combine two or more arrays across different dimensions and axes to create a new array. This is helpful for combining arrays either vertically (along rows) or horizontally (along columns), depending on the need.

You can join arrays in Numpy using the concatenate() function available in the NumPy module.

### The concatenate() Function

The concatenate() function in NumPy is used to concatenate (join together) arrays along a specified axis. It allows you to combine arrays either along rows or columns, depending on the axis parameter provided. Following is the syntax −

```
numpy.concatenate((a1, a2, ...), axis=0, out=None)
```

Where,

- **a1, a2, ... −**These are the sequence of arrays to be joined. These arrays must have the same shape along all axes except the one specified by axis. Default is 0 (along rows). Use 1 for joining along columns.
- **axis −**Specifies the axis along which the arrays will be joined.
- **out (optional) −**This allows you to specify an output array where the result of concatenation will be stored.
## Concatenating Arrays Along Rows

Concatenating arrays along rows in NumPy means stacking arrays vertically, placing one array on top of another to create a larger array. This is useful for combining datasets or expanding data vertically.

In NumPy, you can achieve this using the numpy.concatenate() function with "axis" argument set to "0".

### Example

In the following example, we are concatenating two NumPy arrays "arr1" and "arr2" along rows using the numpy.concatenate() function −

```
import numpy as np

arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])

arr2 = np.array([[7, 8, 9],
                 [10, 11, 12]])

# Concatenate along rows
concatenated_arr = np.concatenate((arr1, arr2), axis=0)
print("Concatenated Array along rows:",concatenated_arr)
```

Following is the output obtained −

```
Concatenated Array along rows:
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
```

## Concatenating Arrays Along Columns

We can also concatenate arrays along columns in NumPy by stacking arrays horizontally, placing one array beside another to extend data horizontally. This is useful for combining datasets where each array represents columns of data that need to be joined.

In NumPy, you achieve this using the numpy.concatenate() function with the "axis" argument set to "1".

### Example

In the example below, we are concatenating two NumPy arrays "arr1" and "arr2" along columns using the numpy.concatenate() function −

```
import numpy as np

# Create two arrays
arr1 = np.array([[1, 2],
                 [3, 4]])

arr2 = np.array([[5, 6],
                 [7, 8]])

# Concatenate along columns 
concatenated_arr = np.concatenate((arr1, arr2), axis=1)
print("Concatenated Array along columns:")
print(concatenated_arr)
```

This will produce the following result −

```
Concatenated Array along columns:
[[1 2 5 6]
 [3 4 7 8]]
```

## Concatenating Arrays with Mixed Dimensions

Concatenating arrays with mixed dimensions in NumPy involves combining arrays that initially have different shapes.

To achieve this, we use broadcasting techniques to adjust the shapes of the arrays so they are compatible for concatenation. This involves expanding the dimensions of the smaller arrays to match the larger arrays along the concatenation axis.

In NumPy, you can adjust the dimensions of arrays using functions such as np.reshape(), np.expand_dims(), and slicing.

### Example: Concatenate 1D array with 2D array

Let us consider concatenating a 1D array with a 2D array. The 1D array will be expanded along the appropriate dimension to match the 2D array −

```
import numpy as np

# Create a 1D array
arr1 = np.array([1, 2, 3])

# Create a 2D array
arr2 = np.array([[4, 5, 6],
                 [7, 8, 9]])

# Expand dimensions of the 1D array to match the 2D array for concatenation along rows
expanded_arr1 = np.expand_dims(arr1, axis=0)

# Concatenate along rows (axis=0)
concatenated_arr = np.concatenate((expanded_arr1, arr2), axis=0)
print("Concatenated Array with Mixed Dimensions along rows:")
print(concatenated_arr)
```

Following is the output of the above code −

```
Concatenated Array with Mixed Dimensions along rows:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

### Example: Concatenate 2D array with 3D array

If you have arrays with more dimensions, you can similarly expand their dimensions to match each other. For example, concatenating a 2D array with a 3D array involves expanding the dimensions of the 2D array −

```
import numpy as np

# Create a 2D array
arr1 = np.array([[1, 2],
                 [3, 4]])

# Create a 3D array
arr2 = np.array([[[5, 6],
                  [7, 8]],
                 
                 [[9, 10],
                  [11, 12]]])

# Expand dimensions of the 2D array to match the 3D array for concatenation along the third dimension (axis=2)
expanded_arr1 = np.expand_dims(arr1, axis=2)

# Concatenate along the third dimension (axis=2)
concatenated_arr = np.concatenate((expanded_arr1, arr2), axis=2)
print("Concatenated Array with Mixed Dimensions along axis=2:")
print(concatenated_arr)
```

The output obtained is as shown below −

```
Concatenated Array with Mixed Dimensions along axis=2:
[[[ 1  5  6]
  [ 2  7  8]]

 [[ 3  9 10]
  [ 4 11 12]]]
```

## Concatenating Arrays Along Specific Axes

You can concatenate arrays along axes other than "0" and "1" using the
**axis**parameter of the concatenate() function. This parameter determines the dimension along which the arrays will be joined. By changing the value of axis, you can control whether the arrays are concatenated along rows, columns, or higher dimensions.
For arrays with more than two dimensions, you can specify higher axes for concatenation. For example, concatenating along the third axis (axis=2) involves combining arrays along their depth.

### Example

In the following example, we are concatenating two 3D arrays along the third dimension −

```
import numpy as np

# 3D arrays
arr1 = np.array([[[1, 2],
                  [3, 4]],
                 
                 [[5, 6],
                  [7, 8]]])

arr2 = np.array([[[9, 10],
                  [11, 12]],
                 
                 [[13, 14],
                  [15, 16]]])

# Concatenate along the third dimension (axis=2)
result = np.concatenate((arr1, arr2), axis=2)
print("Concatenated along third dimension:")
print(result)
```

After executing the above code, we get the following output −

```
Concatenated along third dimension:
[[[ 1  2  9 10]
  [ 3  4 11 12]]

 [[ 5  6 13 14]
  [ 7  8 15 16]]]
```

## Concatenating Arrays Using stack() Function

The NumPy stack() function can also be used to concatenate arrays along a new axis. Unlike numpy.concatenate(), which joins arrays along an existing axis, numpy.stack() adds an additional dimension, creating a new axis in the result. Following is the syntax −

```
numpy.stack(arrays, axis=0)
```

Where,

- **arrays −**A sequence of arrays to be stacked. All arrays must have the same shape.
- **axis −**The axis along which the arrays will be stacked. The default is 0.
### Example

In the example below, we are stacking two 2D arrays along a new third axis (axis=2) using the NumPy stack() function −

```
import numpy as np

# Example 2D arrays
arr1 = np.array([[1, 2],
                 [3, 4]])

arr2 = np.array([[5, 6],
                 [7, 8]])

# Stack along a new third axis 
result = np.stack((arr1, arr2), axis=2)
print("Stacked along a new axis:")
print(result)
```

The result produced is as follows −

```
Stacked along a new axis:
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
```

---

## 12. NumPy - Stacking Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_stacking_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_stacking_arrays.htm)*

---

---
[Previous](/numpy/numpy_concatenating_arrays.htm)[Quiz](/numpy/quiz_on_numpy_stacking_arrays.htm)[Next](/numpy/numpy_splitting_arrays.htm)
## Stacking NumPy Array

Stacking arrays in NumPy refers to combining multiple arrays along a new dimension, creating higher-dimensional arrays. This is different from concatenation, which combines arrays along an existing axis without adding new dimensions.

NumPy provides several functions to achieve stacking. They are as follows −

- Using numpy.stack() Functiom
- Using numpy.vstack() Function
- Using numpy.hstack() Function
- Using numpy.dstack() Function
- Using numpy.column_stack() Function
## Stacking Arrays Using stack() Function

We can use the stack() function in NumPy to stack a sequence of arrays along a new axis, creating a new dimension in the result.

> Unlike numpy.concatenate() function, which combines arrays along an existing axis, numpy.stack() function adds a new axis at the specified position to the arrays being stacked.

Following is the syntax of the stack() function in NumPy −

```
np.stack(arrays, axis=0)
```

Where,

- **arrays −**A sequence of arrays to be stacked.
- **axis −**The axis along which to stack the arrays. The default is 0, which adds a new first axis.
### Example: Stacking 1D Arrays

In the below example, we are stacking three 1D arrays along a new axis (axis 0) using the numpy.stack() function, resulting in a 2D array −

```
import numpy as np

# arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Stack arrays along a new axis
stacked_arr = np.stack((arr1, arr2, arr3), axis=0)
print("Stacked Array along a new axis (Axis 0):")
print(stacked_arr)
```

Following is the output obtained −

```
Stacked Array along a new axis (Axis 0):
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

### Example: Changing the Axis

The "axis" parameter in numpy.stack() function determines where the new axis is inserted. By changing the value of axis, you can control how the arrays are stacked −

```
import numpy as np

# arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Stack arrays along axis 1
stacked_arr = np.stack((arr1, arr2, arr3), axis=1)
print("Stacked Array along Axis 1:")
print(stacked_arr)
```

This will produce the following result −

```
Stacked Array along Axis 1:
[[1 4 7]
 [2 5 8]
 [3 6 9]]
```

### Example: Stacking Multi-dimensional Arrays

The numpy.stack() function can also be used to stack multi-dimensional arrays. The function adds a new axis to the higher-dimensional arrays and stacks them accordingly.

In here, we are stacking two 2D arrays −

```
import numpy as np

# 2D arrays
arr1 = np.array([[1, 2],
                 [3, 4]])

arr2 = np.array([[5, 6],
                 [7, 8]])

# Stack arrays along a new axis
stacked_arr = np.stack((arr1, arr2), axis=0)
print("Stacked 2D Arrays along a new axis (Axis 0):")
print(stacked_arr)
```

Following is the output of the above code −

```
Stacked 2D Arrays along a new axis (Axis 0):
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
```

## Stacking Arrays Using column_stack() Function

The numpy.column_stack() function in NumPy is used to stack 1D arrays as columns into a 2D array or to stack 2D arrays column-wise. This function provides a way to combine arrays along the second axis (axis=1), effectively increasing the number of columns in the resulting array.

Following is the syntax −

```
np.column_stack(tup)
```

Where,
**tup**is a tuple of arrays to be stacked. The arrays can be either 1D or 2D, but they must have the same number of rows.
### Example: Stacking 1D arrays as columns

In the example below, we are stacking two two 1D arrays as columns into a 2D array using the NumPy column_stack() function −

```
import numpy as np

# 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Column-stack 1D arrays
stacked_arr_1d = np.column_stack((arr1, arr2))

print("Stacked 1D arrays as 2D array:")
print(stacked_arr_1d)
```

We get the output as shown below −

```
Stacked 1D arrays as 2D array:
[[1 4]
 [2 5]
 [3 6]]
```

### Example: Stacking 2D arrays column-wise

In here, we are stacking two 2D arrays column-wise using the NumPy column_stack() function −

```
import numpy as np
# 2D arrays
arr3 = np.array([[1, 2],
                 [3, 4]])
arr4 = np.array([[5, 6],
                 [7, 8]])

# Column-stack 2D arrays
stacked_arr_2d = np.column_stack((arr3, arr4))

print("Stacked 2D arrays column-wise:")
print(stacked_arr_2d)
```

Following is the output obtained −

```
Stacked 2D arrays column-wise:
[[1 2 5 6]
 [3 4 7 8]]
```

## Vertical Stacking

We can also stack arrays vertically (row-wise) using the vstack() function in NumPy. It is equivalent to using numpy.concatenate() function with "axis=0", where arrays are concatenated along the first axis.

This results in an array with an increased number of rows, combining multiple arrays row-wise. Following is the syntax −

```
numpy.vstack(tup)
```

Where,
**tup**is a tuple of arrays to be stacked vertically. All arrays must have the same number of columns.
### Example

In the example below, we are stacking two arrays vertically using the NumPy vstack() function −

```
import numpy as np

# arrays
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])

arr2 = np.array([[7, 8, 9],
                 [10, 11, 12]])

# Stack arrays vertically
stacked_arr = np.vstack((arr1, arr2))

print("Vertically Stacked Array:")
print(stacked_arr)
```

The output obtained is as shown below −

```
Vertically Stacked Array:
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
```

## Horizontal Stacking

We can stack arrays horizontally (column-wise) using the hstack() function in NumPy. It is equivalent to using numpy.concatenate() function with "axis=1", where arrays are concatenated along the second axis for 2D arrays.

This results in an array with an increased number of columns, combining multiple arrays column-wise. Following is the syntax −

```
numpy.hstack(tup)
```

Where,
**tup**is a tuple of arrays to be stacked horizontally. All arrays must have the same number of rows.
### Example

In the example below, we are stacking two arrays horizontally using the NumPy hstack() function −

```
import numpy as np

# arrays
arr1 = np.array([[1, 2],
                 [3, 4]])

arr2 = np.array([[5, 6],
                 [7, 8]])

# Stack arrays horizontally
stacked_arr = np.hstack((arr1, arr2))

print("Horizontally Stacked Array:")
print(stacked_arr)
```

After executing the above code, we get the following output −

```
Horizontally Stacked Array:
[[1 2 5 6]
 [3 4 7 8]]
```

## Depth Stacking

The numpy.dstack() function is used to stack arrays along the third dimension, also known as the depth dimension. This combines arrays depth-wise, effectively creating a new dimension in the resulting array.

It is particularly useful when you want to combine multiple 2D arrays into a single 3D array. Following is the syntax −

```
np.dstack(tup)
```

Where,
**tup**is a tuple of arrays to be stacked along the third dimension. All arrays must have the same shape in the first two dimensions.
### Example

In this example, we are stacking two arrays along the third dimension using the NumPy dstack() function −

```
import numpy as np

# arrays
arr1 = np.array([[1, 2],
                 [3, 4]])

arr2 = np.array([[5, 6],
                 [7, 8]])

# Stack arrays along the third dimension
stacked_arr = np.dstack((arr1, arr2))

print("Depth-wise Stacked Array:")
print(stacked_arr)
```

The result produced is as follows −

```
Depth-wise Stacked Array:
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
```

---

## 13. NumPy - Splitting Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_splitting_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_splitting_arrays.htm)*

---

---
[Previous](/numpy/numpy_stacking_arrays.htm)[Quiz](/numpy/quiz_on_numpy_splitting_arrays.htm)[Next](/numpy/numpy_flattening_arrays.htm)
## Splitting NumPy Array

Splitting arrays in NumPy is a way to divide a single array into multiple sub-arrays. This can be done along any axis, depending on how you want to partition the data. NumPy provides several functions to split arrays in different ways. They are as follows −

- Using numpy.split() Function
- Using numpy.array_split() Function
- Using numpy.hsplit() Function
- Using numpy.vsplit() Function
- Using numpy.dsplit() Function
## Splitting Arrays Using split() Function

We can use the split() function in NumPy to split an array into multiple sub-arrays along a specified axis. The array is divided based on the provided indices. Following is the syntax −

```
numpy.split(array, indices_or_sections, axis=0)
```

Where,

- **array −**The input array to be split.
- **indices_or_sections −**This can be either an integer or a 1D array of sorted integers.
If an integer, it specifies the number of equal-sized sub-arrays to create. The array must be divisible evenly into this number of sections.

If a 1D array of sorted integers, it specifies the points at which to split the array.

- **axis −**The axis along which to split the array. The default is 0 (split along rows for 2D arrays).
### Example: Splitting into Equal-sized Sub-arrays

In the below example, we are splitting an array "arr" into 3 equal sub-arrays along the columns (axis=1) using the numpy.split() function−

```
import numpy as np

# array
arr = np.arange(9).reshape(3, 3)

# Split into 3 equal sub-arrays 
split_arr = np.split(arr, 3, axis=1)

print("Original Array:")
print(arr)
print("\nSplit into 3 equal sub-arrays along axis 1:")
for sub_arr in split_arr:
   print(sub_arr)
```

Following is the output obtained −

```
Original Array:
[[0 1 2]
 [3 4 5]
 [6 7 8]]

Split into 3 equal sub-arrays along axis 1:
[[0]
 [3]
 [6]]
[[1]
 [4]
 [7]]
[[2]
 [5]
 [8]]
```

### Example: Splitting at Specific Indices

Here, we are splitting an array at indices [1, 2] along the rows (axis=0) using the split() function in NumPy −

```
import numpy as np

# array
arr = np.arange(9).reshape(3, 3)

# Split array at specified indices
split_arr = np.split(arr, [1, 2], axis=0)

print("\nSplit at indices [1, 2] along axis 0:")
for sub_arr in split_arr:
   print(sub_arr)
```

This will produce the following result −

```
Split at indices [1, 2] along axis 0:
[[0 1 2]]
[[3 4 5]]
[[6 7 8]]
```

## Splitting Arrays Using array_split() Function

We can also use the array_split() function in NumPy to split an array into multiple sub-arrays along a specified axis. Unlike numpy.split() function, the array_split() function allows for unequal splits if the array cannot be evenly divided.

When the array does not evenly divide into the specified number of sections, the numpy.array_split() function ensures that the resulting sub-arrays are as equal in size as possible, distributing any extra elements to the earlier sub-arrays. Following is the syntax −

```
numpy.array_split(array, indices_or_sections, axis=0)
```

Where,

- **array −**The input array to be split.
- **indices_or_sections −**This can be either an integer or a 1D array of sorted integers.
If an integer, it specifies the number of equal-sized sub-arrays to create. The array must be divided as equally as possible.

If a 1D array of sorted integers, it specifies the points at which to split the array.

- **axis −**The axis along which to split the array. The default is 0 (split along rows for 2D arrays).
### Example

In the example below, we are splitting a 1D array with "10" elements into 3 unequal sub-arrays using numpy.array_split() function −

```
import numpy as np

# array
arr = np.arange(10)

# Split into 3 sub-arrays along axis 0
split_arr = np.array_split(arr, 3)

print("Original Array:")
print(arr)
print("\nSplit into 3 unequal sub-arrays:")
for sub_arr in split_arr:
   print(sub_arr)
```

Following is the output of the above code −

```
Original Array:
[0 1 2 3 4 5 6 7 8 9]

Split into 3 unequal sub-arrays:
[0 1 2 3]
[4 5 6]
[7 8 9]
```

## Horizontal Splitting

We can split an array along the horizontal axis, which is axis = 1, for 2D arrays using the hsplit() function in NumPy. This function divides the array into sub-arrays horizontally, effectively separating columns of data. Following is the syntax −

```
numpy.hsplit(array, indices_or_sections)
```

Where,

- **array −**The input array to be split.
- **indices_or_sections −**Either an integer or a 1D array of indices that indicate how to split the array.
### Example

In this example, we are splitting a 2D array "arr" along its columns into 2 equal parts using the numpy.hsplit() function −

```
import numpy as np

# 2D array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]])

# Split into 2 equal parts along axis 1
split_arr = np.hsplit(arr, 2)

print("Original Array:")
print(arr)
print("\nSplit into 2 equal parts along axis 1:")
for sub_arr in split_arr:
   print(sub_arr)
```

After executing the above code, we get the following output −

```
Original Array:[[1 2 3 4]
 [5 6 7 8]]

Split into 2 equal parts along axis 1:
[[1 2]
 [5 6]]
[[3 4]
 [7 8]]
```

## Vertical Splitting

We can also split an array along the vertical axis, which is axis = 0, for 2D arrays using the vsplit() function in NumPy. This function divides the array into sub-arrays vertically, effectively separating rows of data. Following is the syntax −

```
numpy.vsplit(array, indices_or_sections)
```

Where,

- **array −**The input array to be split.
- **indices_or_sections −**Either an integer or a 1D array of indices that indicate how to split the array.
### Example

In the example below, we are splitting a 2D array "arr" along its rows into 3 equal parts using the numpy.vsplit() function −

```
import numpy as np

# 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Split into 3 equal parts along axis 0
split_arr = np.vsplit(arr, 3)

print("Original Array:")
print(arr)
print("\nSplit into 3 equal parts along axis 0:")
for sub_arr in split_arr:
   print(sub_arr)
```

The output obtained is as shown below −

```
Original Array:
[[1 2 3]
 [4 5 6]
 [7 8 9]]

Split into 3 equal parts along axis 0:
[[1 2 3]]
[[4 5 6]]
[[7 8 9]]
```

## Depth Splitting

The numpy.dsplit() function is used to split a 3D array along its third dimension. This dimension is commonly referred to as the depth dimension, corresponding to axis=2. Following is the syntax −

```
numpy.dsplit(array, indices_or_sections)
```

### Example

In this example, using the numpy.dsplit() function to split a 3D array "arr" into four equal parts along its third dimension −

```
import numpy as np

# Example 3D array
arr = np.arange(24).reshape((2, 3, 4))

# Split into 4 equal parts along axis 2 (depth)
split_arr = np.dsplit(arr, 4)

print("Original Array:")
print(arr)
print("\nSplit into 4 equal parts along axis 2 (depth):")
for sub_arr in split_arr:
   print(sub_arr)
   print()
```

The result produced is as follows −

```
Original Array:
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]

Split into 4 equal parts along axis 2 (depth):
[[[ 0]
  [ 4]
  [ 8]]

 [[12]
  [16]
  [20]]]

[[[ 1]
  [ 5]
  [ 9]]

 [[13]
  [17]
  [21]]]

[[[ 2]
  [ 6]
  [10]]

 [[14]
  [18]
  [22]]]

[[[ 3]
  [ 7]
  [11]]

 [[15]
  [19]
  [23]]]
```

---

## 14. NumPy - Flattening Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_flattening_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_flattening_arrays.htm)*

---

---
[Previous](/numpy/numpy_splitting_arrays.htm)[Quiz](/numpy/quiz_on_numpy_flattening_arrays.htm)[Next](/numpy/numpy_transposing_arrays.htm)
## Flattening NumPy Array

Flattening arrays in NumPy refers to the process of converting a multi-dimensional arrays into a one-dimensional array, where all elements are placed sequentially. This means that regardless of the dimensions (whether it's a 2D, 3D, or higher-dimensional array), flattening reduces it to a single vector of elements.

NumPy provides two functions,
**ndarray.flatten()**and**ndarray.ravel()**, both of which is used to flatten arrays.
## Flattening Arrays Using flatten() Function

The flatten() function in NumPy is used to convert multi-dimensional arrays into a one-dimensional array, also known as flattening.

It returns a new array that contains all the elements of the original array in a single row-major order (C-style) sequence. Following is the syntax −

```
arr.flatten(order='C')
```

Where,
**order**is an optional parameter specifying the order of elements. Default is 'C' for row-major order.
### Example

In the below example, we are flattening an array "arr" in a single row-major order using the flatten() function in NumPy −

```
import numpy as np

# array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Flattening the array
flattened_arr = arr.flatten()

print("Original Array:")
print(arr)
print("\nFlattened Array:", flattened_arr)
```

Following is the output obtained −

```
Original Array:
[[1 2 3]
 [4 5 6]]

Flattened Array: [1 2 3 4 5 6]
```

## Flattening Arrays Using ravel() Function

The ravel() function in NumPy is used to create a flattened 1D array from a multi-dimensional array. Unlike flatten() function, the ravel() function returns a flattened view of the original array without making a copy whenever possible. Following is the syntax −

```
arr.ravel(order='C')
```

### Example

In the example below, we are using the ravel() function to flatten a 2D array into a 1D array −

```
import numpy as np

# array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Flattening the array
raveled_arr = arr.ravel()

print("Original Array:")
print(arr)
print("\nRaveled Array:", raveled_arr)
```

Following is the output of the above code −

```
Original Array:
[[1 2 3]
 [4 5 6]]

Raveled Array:[1 2 3 4 5 6]
```

## Flattening Array in Fortran Order

When you flatten a multi-dimensional array in Fortran order, you convert it into a one-dimensional array where the elements are arranged as if you were reading the array column by column.

For example, if you have a 2D array
**A**with dimensions (rows, columns), flattening it in Fortran order would arrange the elements such that you iterate over all elements in the first column, then move to the second column, and so on.
In NumPy, you can flatten an array in Fortran order by setting the
**order**parameter to**F**in the flatten() function.
### Example

In this example, we are flattening an array "arr" in Fortran order using the array.flatten() function in NumPy −

```
import numpy as np

# array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Flatten in Fortran order
flattened_arr_fortran = arr.flatten(order='F')

print("Original Array:")
print(arr)
print("\nFlattened Array (Fortran order):",flattened_arr_fortran)
```

After executing the above code, we get the following output −

```
Original Array:
[[1 2 3]
 [4 5 6]]

Flattened Array (Fortran order):[1 4 2 5 3 6]
```

## Concatenating Flattened Arrays

In NumPy, you can concatenate flattened arrays using the numpy.concatenate() function. Here is how you can do it step-by-step −

- **Flatten Arrays −**First, you need to flatten each array that you want to concatenate using the flatten() function. This converts each multi-dimensional array into a one-dimensional array.
- **Concatenate −**Then, use the numpy.concatenate() function to concatenate the flattened arrays into a single array.
> Concatenation is the process of combining multiple arrays into one larger array. When you concatenate flattened arrays, you are mainly appending the elements of each flattened array one after another to create a single, longer array.

### Example

In the example below, we are first flattening 2D arrays "arr1" and "arr2" using the array.flatten() function. Then, we are concatenating these flattened arrays using the concatenate() function in NumPy −

```
import numpy as np

# arrays
arr1 = np.array([[1, 2],
                 [3, 4]])
arr2 = np.array([[5, 6],
                 [7, 8]])

# Flatten arrays
flat_arr1 = arr1.flatten()
flat_arr2 = arr2.flatten()

# Concatenate flattened arrays
concatenated_arr = np.concatenate((flat_arr1, flat_arr2))

print("Flattened Array 1:")
print(flat_arr1)
print("\nFlattened Array 2:")
print(flat_arr2)
print("\nConcatenated Flattened Array:",concatenated_arr)
```

The output obtained is as shown below −

```
Flattened Array 1:
[1 2 3 4]

Flattened Array 2:
[5 6 7 8]

Concatenated Flattened Array:[1 2 3 4 5 6 7 8]
```

## Initializing a Flattened Array with Zeros

Initializing a flattened array with zeros is a way of creating a one-dimensional array where all elements are set to zero.

NumPy provides a function numpy.zeros_like() to create an array of zeros with the same shape and type as a given array. Following is the syntax −

```
numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
```

Where,

- **a −**It is the input array.
- **dtype (optional) −**It specifies the data type of the output array. If not provided, the data type of a is used.
- **order (optional) −**It specifies the memory layout order of the result ('C' for row-major, 'F' for column-major, 'A' for any, 'K' for keep, 'C' is default).
- **subok (optional) −**If True, then sub-classes will be passed-through, otherwise, the returned array will be forced to be a base-class array (default).
- **shape (optional) −**It is the shape of the output array. If not given, it defaults to a.shape.
### Example

In this example, we are creating a 2D NumPy array "arr" initialized with specific values. We then flatten "arr" into a 1D array and initialize "flattened_zeros" with zeros −

```
import numpy as np

# Initializing a 2D array
arr = np.array([[1, 2],
                [3, 4]])

# Flattening and initializing with zeros
flattened_zeros = np.zeros_like(arr.flatten())

print("Original Array:")
print(arr)
print("\nFlattened Array with Zeros:",flattened_zeros)
```

The result produced is as follows −

```
Original Array:
[[1 2]
 [3 4]]

Flattened Array with Zeros: [0 0 0 0]
```

## Finding Maximum Value in Flattened Array

To find the maximum value in a flattened array means to determine the largest element within a one-dimensional representation of a multi-dimensional array.

In NumPy, you can find the maximum value in an array using the numpy.max() function. When applied to a flattened array, this function returns the highest value present in that array. Following is the syntax −

```
numpy.max(a, axis=None, out=None, keepdims=False, initial=None, where=True)
```

Where,

- **a −**It is the input array for which you want to compute the maximum value.
- **axis (optional) −**It specifies the axis along which to operate. By default "None" is returned as the maximum value of the flattened array.
- **out (optional) −**It is the output array where the result is stored. If provided, it must have the same shape and buffer length as the expected output.
### Example

In the following example we are using the numpy.max() function to find the maximum value in a flattened array −

```
import numpy as np

# array
arr = np.array([[1, 2],
                [3, 4]])

# Flatten array
flattened_arr = arr.flatten()

# Find maximum value
max_value = np.max(flattened_arr)

print("Original Array:")
print(arr)
print("\nFlattened Array:")
print(flattened_arr)
print("\nMaximum Value in Flattened Array:",max_value)
```

We get the output as shown below −

```
Original Array:
[[1 2]
 [3 4]]

Flattened Array:
[1 2 3 4]

Maximum Value in Flattened Array:4
```

---

## 15. NumPy - Transposing Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_transposing_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_transposing_arrays.htm)*

---

---
[Previous](/numpy/numpy_flattening_arrays.htm)[Quiz](/numpy/quiz_on_numpy_transposing_arrays.htm)[Next](/numpy/numpy_indexing_and_slicing.htm)
## Transposing NumPy Array

By transposing an array in NumPy, we mean to rearrange the dimensions of an array to access its data along different axes.

For a
**2-dimensional array**(matrix), transposing means flipping the array along its diagonal. This swaps the rows and columns. If you have an array "A" with shape "(m, n)", the transpose "A.T" will have shape "(n, m)", where each element at position "(i, j)" in A will be at position "(j, i)" in A.T.
For arrays with more than two dimensions, transposing involves reordering the axes according to a specified order.

## Transposing Arrays Using transpose() Function

The transpose() function in NumPy is used to rearrange the dimensions of an array. It returns a view of the array with its axes rearranged in a specified order.

If the order is not specified, the shape of the returned array is the same as the original array's shape, but with the dimensions permuted in reverse order. Following is the syntax −

```
numpy.transpose(a, axes=None)
```

Where,

- **a −**It is the array-like object to be transposed.
- **axes (Optional) −**It specifies the new order of axes. If not provided, it defaults to reversing the dimensions of the array.
### Example: Transposing a 2D Array

In the following example, we are transposing a 2D array "arr" using the numpy.transpose() function with default parameters −

```
import numpy as np

# 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Transposing the array 
transposed_arr = np.transpose(arr)

print("Original Array:")
print(arr)
print("\nTransposed Array:")
print(transposed_arr)
```

This swaps the rows and columns of the array as shown in the output below −

```
Original Array:
[[1 2 3]
 [4 5 6]]

Transposed Array:
[[1 4]
 [2 5]
 [3 6]]
```

### Example: Transposing a 3D Array

In here, we are transposing a 3D array "arr_3d" using the numpy.transpose() function with default parameters −

```
import numpy as np
# 3D array
arr_3d = np.array([[[1, 2],
                    [3, 4]],
                   [[5, 6],
                    [7, 8]]])

# Transposing a 3D array
transposed_arr_3d = np.transpose(arr_3d)

print("Original 3D Array:")
print(arr_3d)
print("\nTransposed 3D Array:")
print(transposed_arr_3d)
```

This changes the order of dimensions, effectively rearranging the depth and height of the array as shown in the output below −

```
Original 3D Array:
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

Transposed 3D Array:
[[[1 5]
  [3 7]]

 [[2 6]
  [4 8]]]
```

### Example: Transposing with Specified Axes

In the below example, we are rearranging the axes of a 3D array such that the first dimension (axis 0) remains unchanged, while axes "1" and "2" are swapped using the numpy.transpose() function −

```
import numpy as np

# 3D array
arr = np.array([[[1, 2],
                 [3, 4]],
                
                [[5, 6],
                 [7, 8]]])

# Transposing
transposed_arr = np.transpose(arr, axes=(0, 2, 1))

print("Original 3D Array:")
print(arr)
print("\nTransposed 3D Array:",transposed_arr)
```

Following is the output obtained −

```
Original 3D Array:
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

Transposed 3D Array: 
[[[1 3]
  [2 4]]
[[5 7]
  [6 8]]]
```

## Transposing Arrays Using "ndarray.T" Object

NumPy arrays have a convenient attribute
**".T"**that provides a quick way to transpose arrays without needing to call the**transpose()**function explicitly. In other words, it reverse the axes of multi-dimensional arrays without any additional arguments.
### Example

In this example, we are using the
**.T**attribute in NumPy to transpose the array "arr" −
```
import numpy as np

# Creating a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Transpose the array
transposed_arr = arr.T

print("Original Array:")
print(arr)
print("\nTransposed Array using .T:")
print(transposed_arr)
```

The result produced is as follows −

```
Original Array:
[[1 2 3]
 [4 5 6]]

Transposed Array using .T:
[[1 4]
 [2 5]
 [3 6]]
```

---

## 16. NumPy - Indexing & Slicing

*Source: [https://www.tutorialspoint.com/numpy/numpy_indexing_and_slicing.htm](https://www.tutorialspoint.com/numpy/numpy_indexing_and_slicing.htm)*

---

---
[Previous](/numpy/numpy_transposing_arrays.htm)[Quiz](/numpy/quiz_on_numpy_indexing_and_slicing.htm)[Next](/numpy/numpy_indexing.htm)
Contents of ndarray object can be accessed and modified by indexing or slicing, just like Python's in-built container objects.

## NumPy Indexing
[NumPy Indexing](/https://www.tutorialspoint.com/numpy/numpy_indexing.htm)is used to access or modify elements in an array. Three types of indexing methods are available**field access, basic slicing and advanced indexing**.
### Example 1

In the below example, we have created an array using arange() function and let us see how to access a single element from the array i.e, 6.

```
import numpy as np  
a = np.arange(10) 
b = a[6] 
print(b)
```

Following is an output of the above code −

```
6
```

### Example 2

Let's say we have a list which contains 5 student marks in English and we need to access the score of third student, we use arr[2] as the index starts from '0'.

```
import numpy as np
scores = ['86', '98', '100', '65', '75']
arr = np.array(scores)
print("Third student score is:", arr[2])
```

Following is an output of the above code −

```
Third student score is: 100
```

## Slicing in NumPy
[NumPy Slicing](https://www.tutorialspoint.com/numpy/numpy_slicing.htm)is an extension of Python's basic concept of slicing to n dimensions. A Python slice object is constructed by giving start, stop, and step parameters to the built-in slice function. This slice object is passed to the array to extract a part of array.
### Example 1

In the below code we will see how to access last two elements from the array using arr[-2:] as we didn't specify the stop parameter it access the elements from the second last to the end of the array.

```
import numpy as np
arr = np.arange(6)
print(arr[-2:])
```

Following is an output of the above code −

```
[4 5]
```

### Example 2

Let's say we have an array containing numbers 1 to 12 and need to access only even numbers, we use slicing with step parameter 'arr[::2]' as it slices every second element in the array.

```
import numpy as np
arr = np.arange(12)
even_num = arr[::2]
print("Even Numbers:", even_num)
```

Following is an output of the above code −

```
Even Numbers: [ 0  2  4  6  8 10]
```

### Example 3

Let's create a 2D array and use slicing to access second column in the array. To access all rows (:) but only the second column (index 1) we use arr[:, 1]

```
import numpy as np
arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print(arr[:, 1])
```

Following is an output of the above code −

```
[20 50 80]
```

### Example 4

In the below code we have created a 2D array and let us see how to access all elements from row 2 (index 1) & minus; we use a[1:]. Where, a[1:] which selects all rows starting from the second row (index 1) to the last, including all columns.

```
import numpy as np 
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 
print (a)  
# slice items starting from index
print ('Now we will slice the array from the index a[1:]')
print (a[1:])
```

Following is an output of the above code −

```
[[1 2 3]
 [3 4 5]
 [4 5 6]]

Now we will slice the array from the index a[1:]
[[3 4 5]
 [4 5 6]]
```

### Example 5

Let us see how to slice an array between indexes  −

```
import numpy as np
a = np.arange(10)
print("Array from index 1 to 6:", a[1:7])
```

When we run above program, it produces following result −

```
Array from index 1 to 6: [1 2 3 4 5 6]
```

---

## 17. NumPy - Indexing

*Source: [https://www.tutorialspoint.com/numpy/numpy_indexing.htm](https://www.tutorialspoint.com/numpy/numpy_indexing.htm)*

---

---
[Previous](/numpy/numpy_indexing_and_slicing.htm)[Quiz](/numpy/quiz_on_numpy_indexing.htm)[Next](/numpy/numpy_slicing.htm)
Indexing refers to finding or accessing a particular item or position in an organized list or data structure such as trees, lists, strings, arrays, graphs, matrices, etc. This technique lets us choose one or a group of elements from a data set.

## Indexing in NumPy

NumPy indexing is a method to access  or change specific values in an array using their position. Each position has a number called an index. Positive numbers count from the start (0, 1, 2, ), and negative numbers count backward (-1 for the last, -2 for the second last, etc.).

In NumPy, indexing has an important role in working with large arrays. It simplifies data operations and speeds up analysis by directly referencing array positions. This makes data manipulation and analysis faster.

> Python uses indexing to get items from lists or tuples starting at index 0. In contrast, NumPy indexing works with multi-dimensional arrays and offers more advanced techniques. These include slicing, boolean indexing, and advanced indexing.
![NumPy Indexing](/numpy/images/numpy_indexing.jpg)
## Simple Indexing

Simple indexing in NumPy allows you to use an array's location to access particular items. For a 1D array,  use a single index like arr[2]. For 2D arrays, you have to give both row and column indices, such as arr[1, 2]. For 3D arrays, you need to provide depth, row, and column indices, like this: arr[2, 0, 1].

Let's us take a few examples to understand simple indexing −

### Accessing 1D array using Indexing

Let's say we have a grocery list with vegetables and fruits. Suppose we want to access banana in the grocery list we use
**arr[3]**where 3 is index for banana. Following is the code −
```
import numpy as np
grocery_list = ['carrot', 'beetroot', 'brinjal', 'banana', 'mango', 'potato', 'apple']
arr = np.array(grocery_list)
print(arr[3])
```

Following is an output of the above code −

```
banana
```

### Accessing 2D Array Using Indexing

Let us create a 2D array on the grade book where each row represents a student and each column represents the student's exam score in different subjects. We need to access student 2's score in the third subject.

Accesses the student 2's score(index 1) and 3rd column (subject 3) which is '78' (index 2) we use
**student_score[1,2]**.![2D Indexing in NumPy](/numpy/images/2d_indexing_in_numpy.jpg)
```
import numpy as np
student_score = np.array([['99', '87', '63'],
                     ['100', '98', '78'],
                     ['95', '100', '76']])
print("Student 2's score in 3rd subject :", student_score[1,2])
```

Output of the above code is as follows −

```
Student 2's score in 3rd subject : 78
```

### Accessing 3D Array Using Indexing

First, let us create a 1D array with a sequence of numbers ranging from 1 to 26 using arange() function then we will convert this 1D to 3D using reshape() function. Using the index we will finally access particular elements of the 3D array based on their positions.

To access the element at the third depth (index 2), 0th row, and 3rd column (index 2), we use
**arr_3d[2, 0, 2]**.
```
import numpy as np
arr = np.arange(27)
arr_3d = arr.reshape(3,3,3)
print("3D array is :\n", arr_3d)
print("Element:",arr_3d[2,0,2])
```

Output of the above code is as follows −

```
Element: 20
```

## Negative Indexing in NumPy

We use negative indexing to access elements from the end of an array. The index -1 refers to the last element in the array, -2 refers second last, and so on. It is mostly useful for accessing elements in reverse order in multi-dimensional arrays.

### Example

Following is an example of the negative indexing in NumPy −

```
import numpy as np
arr = np.array([10, 20, 30, 40, 50])
print(arr[-1])
print(arr[-3])
```

Following is an output of the above code −

```
50
30
```

## Types of Indexing in NumPy

NumPy has a number of ways to access and manipulate array items. From simple indexing to advanced indexing, they provide you with more flexibility and control over your data. The following are the types of indexing -

- **Basic Indexing**: Basic indexing involves using integers or slices to obtain specific elements or ranges. This method generates a view of the original array.
- : With this method, you can extract elements using arrays or lists of indices. This can generate copies of the original array and allow for more complex selections, such as non-contiguous elements.[Advanced Indexing](/https://www.tutorialspoint.com/numpy/advanced_indexing.htm)
- : Field access is used for structured arrays to access or manipulate specific fields of the array. It provides an efficient way to handle heterogeneous data similar to columns in a table.[Field access](/https://www.tutorialspoint.com/numpy/field_access.htm)
## Basic Indexing

Basic indexing uses integers or slice objects to access an individual or a group of elements in the array. Basic slicing applies when you use one of the following methods:

- **Slice Object**: Created using the format**start: stop: step**.
- **Single Integer**: Accesses an element at a specific index.
- **Combination of Integers and Slice Objects**: A mix of integers and slice objects (e.g., start: stop, or a tuple containing both).
All arrays generated by basic slicing are always views of the original array.

### Example: Slicing with Start, Stop, and Step parameter

In the below example, an ndarray object is prepared by the arange() function. Then a slice object is defined with start, stop, and step values 2, 7, and 2 respectively.

When this slice object is passed to the ndarray, a part of it starting with index 2 up to 7 with a step of 2 is sliced.

```
import numpy as np 
a = np.arange(12) 
print(a)
#using start:stop:step format
print(a[2:7:2])
```

Following is an output of the above code −

```
[ 0  1  2  3  4  5  6  7  8  9 10 11]
[2 4 6]
```

### Example: Accessing Specific Rows and Elements

Let us create a 2D array for a classroom representing a seating chart where each row represents rows of seats and each column represents individual seats. Now let us extract a specific row and specific seat using a single integer.

The
**arr_2d[2, 0]**accesses the element in the 3rd row and 1st column (indexing starts at 0). Following is the code −
```
import numpy as np
arr = np.arange(12)
arr_2d = arr.reshape(3, 4)
print("arr_2d:\n",arr_2d)
#Using single integer
print("Element at 8th position is:",arr_2d[2,0])
```

Following is an output of the above code −

```
arr_2d:
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
Element at 8th position is: 8
```

### Example: Reshaping and Element Selection

In the below code we have created a 3D array and accessed the elements in the 2nd row (index1) from column indices 2 up to (but not including) index 4 using
**arr_3d[1, 2:4]**.
```
import numpy as np
arr = np.arange(12)
arr_3d = arr.reshape(3, 4)
print(arr_3d[1, 2:4])
```

Following is an output of the above code −

```
[6 7]
```

---

## 18. NumPy - Slicing

*Source: [https://www.tutorialspoint.com/numpy/numpy_slicing.htm](https://www.tutorialspoint.com/numpy/numpy_slicing.htm)*

---

---
[Previous](/numpy/numpy_indexing.htm)[Quiz](/numpy/quiz_on_numpy_slicing.htm)[Next](/numpy/numpy_advanced_indexing.htm)
Slicing is the way to extract a subset of data from a NumPy array. It can be performed on one or more dimensions of a NumPy array. We can define which part of the array to be sliced by specifying the start and end index values using [start : end] along with the array name.

Slicing can also use Python's built-in function
**slice**object, which is constructed by the same start, stop, and step parameters to define the range. This slice object is passed to the array to extract a part of array.
The syntax for slicing an array is [start:stop:step] where −

- 
Start is the index where the slice object begins (index inclusive) if we don't pass start then by default, it is considered as '0'.

- 
Stop is the index where the slice ends (exclusive, meaning the element at this index is not included) if we don't pass stop by default it is considered as length of array in that dimension.

- 
Step determines the interval between indices (i.e., how many elements to skip) if we don't pass the step by default it is 1 and the step cannot be zero.

## Slicing in 1D NumPy Arrays

Slicing in 1D array is used to access specific elements using start:stop:step parameters. It enables efficient sub-setting, skipping elements, or reversing an array.

### Example: Using start:stop:step

Let us create a 1D array, where we have a row of books labeled 0 to 9 on a shelf and we need to pick every second book from the 1st book to the 8th book.

In this example we use slicing parameters separated by a colon : (start:stop:step) directly to the ndarray object. Here start is 1 (second book), stop is 8 ends before the 8th book and step is 2, picks every second book between the indexes.

```
import numpy as np
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[1:8:2])
```

Following is the output of the above code −

```
[1 3 5 7]
```

### Example: Using Slice Object

The same result that we got in example 1 can be achieved by using the built-in slice function. Following is the code −

```
import numpy as np
arr = np.arange(10)
s = slice(1,8,2) 
print(arr[s])
```

Output of the above code is as follows −

```
[1 3 5 7]
```

### Example: Slicing with Start Parameter

Slice items starting from the index. When we use only the start parameter it will start from that index and as we didn't specify stop by default it will consider the length of the array −

```
import numpy as np 
a = np.arange(10) 
print(a[2:])
```

Following is the output of the above code −

```
[2 3 4 5 6 7 8 9]
```

### Example: Slicing with Stop Parameter

Slice items only using the stop parameter. When we only use the stop parameter and omit the start then it access elements from the beginning of the array up to, but not including the, specified stop index. Following is the code −

```
import numpy as np 
a = np.arange(10) 
print(a[:7])
```

Following is the output of the above code −

```
[0 1 2 3 4 5 6]
```

### Example: Using Step Parameter

Slice items only using the step parameter. The below code extracts every second element from the array, starting from the beginning till the end, as we didn't specify start and stop.

```
import numpy as np 
a = np.arange(10) 
print(a[::2])
```

When we run above program, it produces following result −

```
[0 2 4 6 8]
```

## Slicing in 2D NumPy Arrays

A 2D NumPy array resembles a matrix where it has 2 indices row and a column. To slice a 2D array we use same syntax as the 1D array. The only thing that changes is that we have to define a slice for every dimension of the array.

### Example

Let us create a 2D array for employee data where 3 columns contain details of employee ID, age, and salary and we will use slice parameters to get the information of employee 2 and get ages of all employees from index 2 −

```
import numpy as np
employees = np.array([
    [1, 25, 50000],  
    [2, 30, 60000], 
    [3, 28, 55000], 
    [4, 35, 65000], 
    [5, 40, 70000]   
])
print("Information of Employee 2:", employees[1])
print("Ages of employees from index 2 onwards:", employees[2:, 1])
```

Following is the output of the above code −

```
Information of Employee 2: [ 2  30 60000]
Ages of employees from index 2 onwards: [28 35 40]
```

## Slicing in 3D NumPy Arrays

A 3D array is a collection of 2D arrays, with three indices: depth(or plane), row, and a column. 3D array also has a the same syntax as 1D and 2D but we need to define slices for all three dimensions.

### Example

Let us create a 3D array using the arange() function and reshape it with values containing 0 to 23 then reshape this 1D into 3D representing a (2*3*4) matrix and use slice object for slicing to get a subarray which selects the first layer (depth), all rows, and the first 2 columns −

```
import numpy as np
arr_3d = np.arange(24).reshape(2, 3, 4)
print("Original 3D array: \n" , arr_3d)
subarray = arr_3d[0, :, :2]
print("\nSliced subarray:", subarray)
```

When we run above program, it produces following result −

```
Original 3D array: 
 [[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]

Sliced subarray: [[0 1]
 [4 5]
 [8 9]]
```

## Negative Slicing

Negative indexing allows accessing elements from the end of an array. The index -1 refers to the last element, -2 to the second last and so on. It is basically used when you have to access a specific items from the end. It gets combined with the slicing syntax (start:stop:step) to extract elements in the reverse order.

### Example: Accessing Lowest 5 Marks

Let us create an array where we store student marks and need to quickly identify the lowest 5 marks we use negative slicing with the start parameter −

```
import numpy as np
marks = np.array([93, 87, 98, 89, 67, 65, 54, 32, 21])
print("Lowest 5 marks is:", marks[-5:])
```

When we run above program, it produces following result −

```
Lowest 5 marks is: [67 65 54 32 21]
```

### Example: Slicing Every Second Element (Reverse Order)

Let us create a 1D array where we need to slice every second element from the end of an array −

```
import numpy as np
data=np.array(['H','A','R','R','Y'])
print(data[-1::-2])
```

When we run above program, it produces following result −

```
['Y' 'R' 'H']
```

### Example: Reversing an Array

We can also use negative slicing for reversing an array. Following is the code −

```
import numpy as np
data=np.array([98,87,86,65,54,32,21])
print("Reversed data :", data[::-1])
```

Following is the output of the above code −

```
Reversed data : [21 32 54 65 86 87 98]
```

## Special cases

In NumPy, Slicing we have special cases that enhance the flexibility of data manipulation. These include ellipsis (...) for partial indexing, full slices (:) to access all elements across dimensions, and newaxis for reshaping arrays.

### Ellipsis

Slicing can also include the ellipsis '()' to make a selection tuple of the same length as the dimension of an array. If the ellipsis is used at the row position, it will return an ndarray comprising of items in rows.

> Ellipsis (...) makes slicing in multiple dimensions much easier. It can use as many colons (:) as it wants to fill in the un-specified dimensions.

For example, in a three-dimensional array, a[..., 1] will pick all depths and rows but only the second column.

#### Example: Accessing Items in Specific Dimensions

Let us create a 2D array where we slice all the items in the second column using Ellipsis −

```
import numpy as np 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print('The items in the second column are:', a[...,1] )
print('The items in the second row are:',a[1,...] )
print('The items column 1 onwards are:',a[...,1:])
```

Following is the output of the above code −

```
The items in the second column are: [2 4 5]
The items in the second row are: [3 4 5]
The items column 1 onwards are: [[2 3]
 [4 5]
 [5 6]]
```

### Full Slices

Full slices are used to access all depths, rows, and columns from an array using (:). Let us see examples of each use case

#### Example: Accessing Rows, Columns, and Entire Array

Let us create a Food rating system where rows are users and columns are ratings of 4 different restaurants.

```
import numpy as np
Food_ratings = np.array([
    [4, 5, 3, 4],  # User 1
    [3, 4, 2, 5],  # User 2
    [5, 5, 4, 4]   # User 3
])
# Get all ratings by User 1
user1 = Food_ratings[0, :]
print("User 1 Ratings:", user1)

# Get all ratings for restaurant 1
restuarent_ratings = Food_ratings[:, 0]
print("restaurant 1 Ratings:", restuarent_ratings)

# Get all ratings (entire table)
all_ratings = Food_ratings[:, :]
print("All Ratings:\n", all_ratings)
```

Following is the output of the above code −

```
User 1 Ratings: [4 5 3 4]
restaurant 1 Ratings: [4 3 5]
All Ratings:
 [[4 5 3 4]
 [3 4 2 5]
 [5 5 4 4]]
```

### NewAxis

Newaxis in NumPy is the function that adds a new axis to an array, increasing it's dimensions. It is helpful in reshaping arrays and doing matrix transformations. By using newaxis, one can easily change a 1D array into a 2D row or column vector and vice versa: a 2D array into a 3D array and more.

#### Example: Converting 1D Array to 2D Column Vector

Let us create a 1D array and convert that into a 2D column vector

```
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[:, np.newaxis])
```

Following is the output of the above code −

```
[[1]
 [2]
 [3]
 [4]]
```

#### Example: Combining 1D and 2D Arrays with hstack()

Let us consider two 1D arrays: one for rainfall monthly amounts in millimeter units and the other is names of months. Then we will convert this 1D into 2D arrays and then concats both with its horizontally axis using the hstack() function. This 2-dimensional array will then hold down both rainfall totals with respect to the month.

When we convert this into 2D, NumPy converts a mixed data type into a single type. In this case, the result will be an array of strings because the Months array consists of strings. So Numerical operations on rainfall array will no longer work.

```
import numpy as np
Rainfall = np.array([120, 85, 60, 90, 150])
Months = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May'])
Rainfall_2d = Rainfall[:, np.newaxis]
Months_2d = Months[:, np.newaxis]
Rainfall_Data = np.hstack((Months_2d, Rainfall_2d))
print("Monthly Rainfall Data:")
print(Rainfall_Data)
```

Following is the output of the above code −

```
Monthly Rainfall Data:
[['Jan' '120']
 ['Feb' '85']
 ['Mar' '60']
 ['Apr' '90']
 ['May' '150']]
```

---

## 19. NumPy - Advanced Indexing

*Source: [https://www.tutorialspoint.com/numpy/numpy_advanced_indexing.htm](https://www.tutorialspoint.com/numpy/numpy_advanced_indexing.htm)*

---

---
[Previous](/numpy/numpy_slicing.htm)[Quiz](/numpy/quiz_on_numpy_advanced_indexing.htm)[Next](/numpy/numpy_fancy_indexing.htm)
Advanced indexing offers a robust method to select specific elements from a NumPy array based on predetermined conditions or guidelines.

> While basic indexing, like array[1:4] gives you a "view" of the original array (where modifications to the slice affect the original data), advanced indexing always creates a copy of the selected data.

It allows you to select elements from an ndarray that is a non-tuple sequence, ndarray object of integer or Boolean data type, or a tuple with at least one item being a sequence object. This method is helpful in dynamic data selection as well as conditional data extraction.

There are two types of advanced indexing −

- Integer Indexing
- Boolean Array Indexing
## Integer Indexing

This allows you to select specific elements from an array using their exact positions (indices)  based on its N dimensional index. Each integer array represents the number of indexes into that dimension.

If the number of integer arrays corresponds to the dimensions of the target ndarray, selecting items becomes simple and straightforward. It's like selecting item at x, y, z position.

### Example: Selecting Elements by Indices

The following example selects one element from each row of the array using integer arrays for both rows and columns. The selection includes elements at (0,0), (1,1) and (2,0) from the first array. Following is the code −

```
import numpy as np 
x = np.array([[1, 2], [3, 4], [5, 6]]) 
y = x[[0,1,2], [0,1,0]] 
print(x)
print('The new array is :\n', y)
```

Following is the output of the above code −

```
[[1 2]
 [3 4]
 [5 6]]
The new array is :
 [1 4 5]
```

### Example: Selecting Corner Elements

In the following example, elements placed at corners of a 4X3 array are selected. The row indices of selection are [0, 0] and [3,3] whereas the column indices are [0,2] and [0,2].

```
import numpy as np
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
print('Our array is: \n', x)
rows = np.array([[0,0],[3,3]])
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols] 
print('The corner elements of this array are: \n', y)
```

Following is the output of the above code −

```
Our array is: 
 [[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
The corner elements of this array are: 
 [[ 0  2]
 [ 9 11]]
```

### Example: Accessing Specific Marks

In this example we have 2D array which contains marks of three different students representing in rows and marks in three columns of different subjects Hindi, Maths, English.

Now, we will see how to access a score of student 1 in Hindi and score of student 2 in Maths. Following is the code −

```
import numpy as np
marks = np.array([[85, 99, 88], [78, 93, 85], [86, 45, 90]])
specific_results = marks[[0, 1], [0, 1]]
print('Marks:\n', marks)
print('Selected marks: \n', specific_results)
```

Output of the above code is as follows −

```
Marks:
 [[85 99 88]
 [78 93 85]
 [86 45 90]]
Selected marks: 
 [85 93]
```

### Example: Index Out of Bounds Error

In NumPy an index error (index out of bounds) occurs when you try to access an element in the array using the index that is outside the valid range of array dimensions.

Let us understand this with example where an array contains 3 rows but the code attempts to access the 4th row (index 3), throws index error −

```
import numpy as np
x = np.array([[0, 1], [2, 3], [4, 65]])
print('The 2D arrays is : \n', x)
print(x[3,1])
```

Following is the output of the above code −

```
The 2D arrays is : 
 [[ 0  1]
 [ 2  3]
 [ 4 65]]
Traceback (most recent call last):
  File "/home/cg/root/22245/main.py", line 4, in 
    print(x[3,1])
IndexError: index 3 is out of bounds for axis 0 with size 3
```

## Boolean Array Indexing

NumPy's Boolean indexing lets you select array elements that meet a certain condition. It involves creating a Boolean array where each element matches up with a True or False condition. The elements from the original array is selected where the Boolean array shows True.

This approach works well to filter data using logical conditions, like comparison operations. It is a powerful tool to pull out specific values from arrays.

### Example: Using a Boolean Array to Select Specific Elements

This code illustrates how to use a Boolean array as a mask for selecting certain elements from a NumPy array. The Boolean array specifies which elements are to be included (True) or excluded (False) in the final array.

```
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
bool_array = np.array([True, False, True, False, True])
selected_elements = arr[bool_array]

print("Original array:", arr)
print("Boolean array:", bool_array)
print("Selected elements:", selected_elements)
```

Output of the above code is as follows −

```
Original array: [10 20 30 40 50]
Boolean array: [ True False  True False  True]
Selected elements: [10 30 50]
```

### Example: Filtering Items Greater Than 5

In this example, items greater than 5 are returned as a result of Boolean indexing.

```
import numpy as np 
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
print('Our array is: \n', x)  
print('The items greater than 5 are:' , x[x > 5])
```

When we run above program, it produces following result −

```
Our array is: 
 [[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
The items greater than 5 are: [ 6  7  8  9 10 11]
```

### Example: Removing NaN Values

In this example, NaN (Not a Number) elements are omitted by using ~ (complement operator).

```
import numpy as np 
a = np.array([np.nan, 1,2,np.nan,3,4,5]) 
print(a[~np.isnan(a)])
```

When we run above program, it produces following result −

```
[1. 2. 3. 4. 5.]
```

### Example: Filtering Complex Numbers

The following example shows how to filter out the complex numbers from an array.

```
import numpy as np 
a = np.array([1, 2+6j, 5, 3.5+5j]) 
print(a[np.iscomplex(a)])
```

Output of the above code is as follows −

```
[2. +6.j 3.5+5.j]
```

### Example: Extracting Even Numbers

Let us see how to extract even numbers from an array using boolean indexing −

```
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print('The even numbers are:', x[x % 2 == 0])
```

Following is the output of the above code −

```
The even numbers are: [2 4 6 8]
```

---

## 20. Numpy - Fancy Indexing

*Source: [https://www.tutorialspoint.com/numpy/numpy_fancy_indexing.htm](https://www.tutorialspoint.com/numpy/numpy_fancy_indexing.htm)*

---

---
[Previous](/numpy/numpy_advanced_indexing.htm)[Quiz](/numpy/quiz_on_numpy_fancy_indexing.htm)[Next](/numpy/numpy_field_access.htm)
## Fancy Indexing in NumPy

Fancy indexing in NumPy is a method to select multiple elements from an array using arrays or lists of specific index where, index is used to represent the position of element in the array. Instead of picking elements one by one, you can select multiple elements at once on your choice.

It's like giving the array a list of "indices" you want, and it gives you those values directly, making data handling faster and quicker.

> Fancy indexing is the advance form of simple indexing. In simple indexing we access single elements or slices using integers, while fancy indexing access multiple elements using arrays or lists of integers. It returns a new array that is independent of the original one.

Fancy indexing allows you to access multiple elements using the following −

- 
Another NumPy Array

- 
Python List

### Example: Simple Indexing

Let us create a 1D array to access a single element using it's position. In the below code arr[3] access the element at index 3 (fourth position) of array which is 60.

```
import numpy as np
x = np.array([50, 90, 70, 60, 40, 100])
print("By using simple indexing:" ,x[3])
```

Following is the output of the above code −

```
By using simple indexing: 60
```

## Fancy Indexing Using a NumPy Array

Let us create a 1D array using arange() function that includes numbers from 20 to 30, then we will create a second NumPy array for indexing multiple elements at once.

In the below code we have created a second array which contains indices to access multiple elements i.e. 3, 4, 6 and the elements in those positions are 23, 24, 26.

```
import numpy as np
x = np.arange(20, 31)
print(x)
arr = np.array([ 3, 4, 6])
print("The elements at 3, 4, 6 positions are :\n" , x[arr])
```

Following is the output of the above code −

```
[20 21 22 23 24 25 26 27 28 29 30]
The elements at 3, 4, 6 positions are :
 [23 24 26]
```

## Fancy Indexing Using a Python List

Let us create a 1D array using randint which generates random integers. Then by using python list we will store the positions we want to access. Here [1, 0, 2] is stored in indices, then we will use the indices list to access the array.

```
import numpy as np
array = np.random.randint(10, 59, size = 10)
print(array)
indices = [1, 0, 2]
print("Accessing multiple elements using python list: \n", array[indices])
```

Following is the output of the above code −

```
[32 57 48 26 47 32 38 35 30 36]
Accessing multiple elements using python list: 
 [57 32 48]
```

## Converting 1D to 2D Array Using Fancy Indexing

In this example, we used the arange() function to build a one-dimensional array containing numbers from 1 to 10. Here the elements we need to access is specified in 2D. In fancy indexing the shape of result is reflected by the shape of index array. Following is the code −

```
import numpy as np
x = np.arange(1, 10)
indices = np.array([[5, 3], [4, 5]])
new_2D_arr = x[indices]
print(new_2D_arr)
```

Following is the output of the above code −

```
[[6 4]
 [5 6]]
```

## Fancy Indexing in 2D NumPy Array

In the below example we have used fancy indexing to select multiple elements from the 2D array. The row and column indices are provided in lists and the code selects the specified elements from the 2D array.

```
import numpy as np
x = np.arange(12)
x_2D = x.reshape(3,4)
row_indices = [ 1, 2]
col_indices = [0, 2]
selected_indices = x_2D[row_indices, col_indices]
print("2D array is :\n", x_2D)
print("selected elements are : \n", selected_indices)
```

Following is the output of the above code −

```
2D array is :
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
selected elements are : 
 [ 4 10]
```

## Fancy Indexing in 3D NumPy Array

In the below example we have created 3D array and specified depth_indices, row_indices, col_indices to select particular multiple elements using fancy indexing.

```
import numpy as np
x = np.arange(27)
x_3D = x.reshape(3, 3, 3)
depth_indices = [0, 1]
row_indices = [ 1, 2]
col_indices = [0, 2]
selected_indices = x_3D[depth_indices, row_indices, col_indices]
print("3D array is :\n", x_3D)
print("selected elements in the 3D array : \n", selected_indices)
```

Following is the output of the above code −

```
3D array is :
 [[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[ 9 10 11]
  [12 13 14]
  [15 16 17]]

 [[18 19 20]
  [21 22 23]
  [24 25 26]]]
selected elements in the 3D array : 
 [ 3 17]
```

## Fancy Indexing with Negative Indices

With the help of fancy indexing we can use negative indices to access multiple elements from the end of the array. Following is the example for fancy indexing with negative indices.

```
import numpy as np
x = np.arange(10)
indices = np.array([-1, -2, -3])  
print("Selected elements:", x[indices])
```

Following is the output of the above code −

```
Selected elements: [9 8 7]
```

---

## 21. NumPy - Field access

*Source: [https://www.tutorialspoint.com/numpy/numpy_field_access.htm](https://www.tutorialspoint.com/numpy/numpy_field_access.htm)*

---

---
[Previous](/numpy/numpy_fancy_indexing.htm)[Quiz](/numpy/quiz_on_numpy_field_access.htm)[Next](/numpy/numpy_slicing_with_boolean_arrays.htm)
## NumPy Field Access

Field access in NumPy refers to retrieving or modifying specific elements within a structured array based on their field names. It allows you to work with individual attributes or properties of each record in the array.

> Structured arrays in NumPy enable you to define arrays with records that contain multiple fields, each with its own data type. Fields in a structured array can be accessed individually, allowing for manipulation of data.

## Accessing Individual Fields by Name

Structured arrays in NumPy allow you to assign names to different fields within each element. This naming convention makes it easy to access specific fields directly using those names.

> In NumPy, when working with structured arrays, accessing individual fields allows you to interact with specific components or attributes of each element in the array. This is important when dealing with arrays that contain multiple types of data.

### Example

In the following example, we are accessing the 'name' field of a structured array to extract and retrieve all the names from the array −

```
import numpy as np

# Define a structured array with fields 'name' and 'age'
dtype = [('name', 'U10'), ('age', 'i4')]
data = [('Alice', 25), ('Bob', 30)]
structured_array = np.array(data, dtype=dtype)

# Access the 'name' field
names = structured_array['name']
print("Names:", names)
```

Following is the output obtained −

```
Names: ['Alice' 'Bob']
```

## Field Access in Multi-dimensional Arrays

To access specific fields in a multi-dimensional structured array, you can use indexing techniques similar to those used in 1D and 2D arrays but applied across multiple dimensions.

> A multi-dimensional structured array is an array where each element is itself a structured array, and these elements are organized in multiple dimensions (e.g., 2D, 3D arrays). Each element in the array can have multiple fields, similar to a table where each row is a record with several attributes.

### Example

In the example below, we are accessing the 'name' field from the first layer of a 3D structured array −

```
import numpy as np

# Define a 3D structured array
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = [[[('Alice', 25, 5.5), ('Bob', 30, 6.0)],
         [('Charlie', 35, 5.8), ('David', 40, 6.2)]],
        [[('Eve', 28, 5.7), ('Frank', 33, 6.1)],
         [('Grace', 29, 5.6), ('Hank', 32, 6.3)]]]
structured_array_3d = np.array(data, dtype=dtype)

# Access the 'name' field from the first layer
names_layer_0 = structured_array_3d[0]['name']
print("Names in the first layer:\n", names_layer_0)
```

Following is the output of the above code −

```
Names in the first layer:
[['Alice' 'Bob']
 ['Charlie' 'David']]
```

## Accessing Fields in Specific Slices

Accessing fields in specific slices means retrieving values from particular subsets or ranges of data within a structured array.

When you slice a single dimension of a structured array, you can then access specific fields from the resulting slice. To access fields from slices involving multiple dimensions, you need to apply slicing across dimensions and then select fields from the resulting sub-array.

### Example: Slicing 1D and Accessing Fields

In the following example, we are slicing a structured array to obtain a subset of rows, specifically rows 1 and 2. After slicing, we access and retrieve the 'name' and 'age' fields from this subset −

```
import numpy as np

# Define a structured array with fields 'name' and 'age'
dtype = [('name', 'U10'), ('age', 'i4')]
data = [('Alice', 25), ('Bob', 30), ('Charlie', 35), ('David', 40)]
structured_array = np.array(data, dtype=dtype)

# Slice the array to get a subset of rows
sliced_array = structured_array[1:3]  # Gets rows 1 and 2

# Access the 'name' field from the sliced array
names = sliced_array['name']
# Access the 'age' field from the sliced array
ages = sliced_array['age']

print("Sliced names:", names)
print("Sliced ages:", ages)
```

The output obtained is as shown below −

```
Sliced names: ['Bob' 'Charlie']
Sliced ages: [30 35]
```

### Example: Slicing 2D and Accessing Fields

Here, we are slicing a 2D structured array to extract a subset of rows and columns. We then access and retrieve the 'name' and 'age' fields from this sliced portion of the array −

```
import numpy as np

# Define a 2D array with structured data
dtype = [('name', 'U10'), ('age', 'i4')]
data = [[('Alice', 25), ('Bob', 30)],
        [('Charlie', 35), ('David', 40)]]
structured_array = np.array(data, dtype=dtype).view(np.recarray)

# Slice the array to get a subset of rows and columns
sliced_array = structured_array[0:2, 0:2]  # Gets all rows and columns

# Access the 'name' field from the sliced array
names = sliced_array['name']
# Access the 'age' field from the sliced array
ages = sliced_array['age']

print("Sliced names:", names)
print("Sliced ages:", ages)
```

After executing the above code, we get the following output −

```
Sliced names: 
[['Alice' 'Bob']
 ['Charlie' 'David']]
Sliced ages: 
[[25 30]
 [35 40]]
```

## Accessing Multiple Fields Simultaneously

Accessing multiple fields simultaneously means retrieving data from more than one field in a structured array at the same time, allowing you to work with a subset of fields together.

To access multiple fields simultaneously in NumPy, you can use the following ways −

- **Accessing with a List of Field Names:**You can specify multiple field names in a list to get a structured array containing only those fields.
- **Using Field Indexing with Structured Arrays:**If you need to access fields by their indices, you can select them using their positions.
### Example

In the example below, we are accessing different fields of a structured array by specifying field names or indices, and printing the results. We retrieve specific fields such as 'name' and 'age', as well as all fields simultaneously −

```
import numpy as np

# Define a structured array with fields 'name', 'age', and 'height'
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = [('Alice', 25, 5.5), ('Bob', 30, 6.0), ('Charlie', 35, 5.8)]
structured_array = np.array(data, dtype=dtype)

# 1. Accessing multiple fields with a list of field names
selected_fields = structured_array[['name', 'age']]
print("Selected fields (name and age):")
print(selected_fields)

# 2. Accessing fields by index
names = structured_array['name']
ages = structured_array['age']
heights = structured_array['height']

print("\nNames:", names)
print("Ages:", ages)
print("Heights:", heights)

# Accessing all fields simultaneously
all_fields = structured_array[['name', 'age', 'height']]
print("\nAll fields:")
print(all_fields)
```

The result produced is as follows −

```
Selected fields (name and age):
[('Alice', 25) ('Bob', 30) ('Charlie', 35)]

Names: ['Alice' 'Bob' 'Charlie']
Ages: [25 30 35]
Heights: [5.5 6.  5.8]

All fields:
[('Alice', 25, 5.5) ('Bob', 30, 6. ) ('Charlie', 35, 5.8)]
```

## Combining Field Access with Boolean Indexing

Combining field access with Boolean indexing means retrieving specific fields from a structured array based on a condition or filter applied to the array.

Boolean indexing allows you to select elements of an array that satisfy a given condition. By applying a boolean mask (an array of boolean values) to a structured array, you can filter the array based on conditions applied to one or more fields.

### Example

In the following example, we are using a boolean mask to filter a structured array based on the 'age' field. We then select and print the 'name' and 'height' fields of the entries where 'age' is greater than 30  −

```
import numpy as np

# Define a structured array
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = [('Alice', 25, 5.5), ('Bob', 30, 6.0), ('Charlie', 35, 5.8), ('David', 40, 6.2)]
structured_array = np.array(data, dtype=dtype)

# Create a boolean mask for filtering based on 'age'
mask = structured_array['age'] > 30

# Apply boolean indexing and select 'name' and 'height' fields
filtered_fields = structured_array[mask][['name', 'height']]
print("Filtered Fields (name and height) where age > 30:\n", filtered_fields)
```

We get the output as shown below −

```
Filtered Fields (name and height) where age > 30:[('Charlie', 5.8) ('David', 6.2)]
```

---

## 22. NumPy - Slicing with Boolean Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_slicing_with_boolean_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_slicing_with_boolean_arrays.htm)*

---

---
[Previous](/numpy/numpy_field_access.htm)[Quiz](/numpy/quiz_on_numpy_slicing_with_boolean_arrays.htm)[Next](/numpy/numpy_array_attributes.htm)
## Slicing with Boolean Arrays in NumPy

Slicing with Boolean arrays in NumPy allows you to select elements from an array based on a criteria. Instead of using specific indices or multiple elements, we provide a Boolean array in which True indicates the elements to be selected and False indicates those should be ignored.

> This method is useful for filtering elements from the array based on conditions instead of explicit loops making it easier, cleaner and concise to apply the conditions directly to the array. It is also useful for removing invalid values and modifying elements based on a condition.

## Selecting Positive Numbers

In the below example we have created a condition array > 0 which creates a boolean array [False False  True  True False  True False]. Then this boolean array is used to select elements from the array where the condition is true and gives [1, 3, 4].

```
import numpy as np
array = np.array([-5, -2, 1, 3, -7, 4, -8])
positive_arr = array > 0
print("The boolean array is : ", positive_arr)
print("positive numbers :" , array[positive_arr])
```

Following is the output of the above code −

```
The boolean array is :  [False False  True  True False  True False]
positive numbers : [1 3 4]
```

## Masking Data Based on a Condition

Let us create a 1D array with the condition arr > 30 = 0. This condition affects all elements in the array when the condition is True. This condition replaces the elements that are greater than 30 into 0.

```
import numpy as np
arr_1D= np.array([100, 28, 10, 34, 20, 15, 25])
arr_1D[arr_1D > 30] = 0
print("modified data :", arr_1D)
```

Following is the output of the above code −

```
modified data : [ 0 28 10  0 20 15 25]
```

## Filtering Data Using Logical Operators

In the example below, we have an array that contains details about a company's sales in a month, and we need to find the biggest sales, which are between $2500 and $3000 or more than $5,000 using logical operators. Following is the code −

```
import numpy as np
data = np.array([1200, 3400, 3500, 5500, 3400, 2300, 2600, 2900, 4500])
highest_sales = data[(data >= 2500) & (data<=3000) | (data > 5000) ]
print("Highest sales in the month is :", highest_sales)
```

Following is the output of the above code −

```
Highest sales in the month is : [5500 2600 2900]
```

---

## 23. NumPy - Array Attributes

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_attributes.htm](https://www.tutorialspoint.com/numpy/numpy_array_attributes.htm)*

---

---
[Previous](/numpy/numpy_slicing_with_boolean_arrays.htm)[Quiz](/numpy/quiz_on_numpy_array_attributes.htm)[Next](/numpy/numpy_array_shape.htm)
## NumPy Array Attributes

In NumPy, attributes are properties of array objects that provide important information about the arrays and their data. These attributes are used to access various details regarding the structure and configuration of the arrays without modifying them.

In this chapter, we will discuss the various array attributes of NumPy.

## NumPy Shape Attribute

The NumPy
**shape**attribute provides the dimensions of the array. It returns a tuple representing the size of the array along each dimension. It can also be used to resize the array.
### Example 1

In the following example, we are retrieving the shape of a NumPy array using the shape attribute −

```
import numpy as np 
a = np.array([[1,2,3],[4,5,6]]) 
print (a.shape)
```

Following is the output obtained −

```
(2, 3)
```

### Example 2

Here, we are resizing an array using the shape attribute in NumPy −

```
import numpy as np 

a = np.array([[1,2,3],[4,5,6]]) 
a.shape = (3,2) 
print (a)
```

This will produce the following result −

```
[[1, 2] 
 [3, 4] 
 [5, 6]]
```

### Example 3

NumPy also provides a reshape() function to resize an array −

```
import numpy as np 
a = np.array([[1,2,3],[4,5,6]]) 
b = a.reshape(3,2) 
print (b)
```

Following is the output of the above code −

```
[[1, 2] 
 [3, 4] 
 [5, 6]]
```

## NumPy Dimensions Attribute

The
**ndim**attribute returns the number of dimensions (axes) of the array.
In NumPy, the dimension of an array is known as its
**rank**. Each axis in a NumPy array corresponds to a dimension. The number of axes (dimensions) is referred to as the array's rank.
Arrays can be of any dimension, from one-dimensional (1D) arrays (also known as vectors) to multi-dimensional arrays like 2D arrays (matrices) or even higher-dimensional arrays.

### Example 1

In this example, we are creating a NumPy array a with "24" evenly spaced integers from "0" to "23" using the arange() function −

```
import numpy as np 
a = np.arange(24) 
print (a)
```

The output obtained is as shown below −

```
[0 1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16 17 18 19 20 21 22 23]
```

### Example 2

Here, we are creating a one-dimensional NumPy array a with "24" elements using the arange() function and then reshaping it into a three-dimensional array "b" with the shape provided, resulting in a 3D array −

```
# This is one dimensional array 
import numpy as np 
a = np.arange(24) 
a.ndim  

# Now reshape it 
b = a.reshape(2,4,3) 
print (b) 
# b is having three dimensions
```

After executing the above code, we get the following output −

```
[[[ 0,  1,  2] 
  [ 3,  4,  5] 
  [ 6,  7,  8] 
  [ 9, 10, 11]]  
  [[12, 13, 14] 
   [15, 16, 17]
   [18, 19, 20] 
   [21, 22, 23]]]
```

## NumPy Size Attribute

The
**size**attribute returns the total number of elements in the array. In NumPy, the size of an array refers to the total number of elements contained within the array.
- For a one-dimensional array, the size is simply the number of elements.
- For a two-dimensional array, the size is the product of the number of rows and columns.
- For a three-dimensional array, the size is the product of the sizes of all three dimensions.
### Example

In the example below, we are using the "size" attribute in NumPy to retrieve the size of a 3D arrray −

```
import numpy as np
# Creating a 3D array
array_3d = np.array([[[1, 2, 3], [4, 5, 6]], 
                     [[7, 8, 9], [10, 11, 12]]])
print("3D Array:\n", array_3d)
print("Size of the array:", array_3d.size)
```

Following is the output obtained −

```
3D Array:
[[[ 1  2  3]
  [ 4  5  6]]

 [[ 7  8  9]
  [10 11 12]]]
Size of the array: 12
```

## NumPy Data Type Attribute

The
**dtype**attribute describes the data type of the elements in the array. In NumPy, the data type of an array refers to the type of the elements stored in the array.
> NumPy supports a wide range of data types, including integers, floats, complex numbers, booleans, and more. Each data type is represented by a dtype object. The "dtype" not only specifies the type of the data but also its size and byte order.
**dtype**object. The "dtype" not only specifies the type of the data but also its size and byte order.
### Example

In this example, we are specifying the data type of a NumPy array at the time of its creation using the "dtype" attribute −

```
import numpy as np

# Creating an array of integers
int_array = np.array([1, 2, 3], dtype=np.int32)
print("Integer Array:", int_array)
print("Data type of int_array:", int_array.dtype)

# Creating an array of floats
float_array = np.array([1.1, 2.2, 3.3], dtype=np.float64)
print("Float Array:", float_array)
print("Data type of float_array:", float_array.dtype)

# Creating an array of complex numbers
complex_array = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
print("Complex Array:", complex_array)
print("Data type of complex_array:", complex_array.dtype)
```

This will produce the following result −

```
Integer Array: [1 2 3]
Data type of int_array: int32
Float Array: [1.1 2.2 3.3]
Data type of float_array: float64
Complex Array: [1.+2.j 3.+4.j]
Data type of complex_array: complex128
```

## NumPy Itemsize Attribute

The
**itemsize**attribute returns the the length of each element of array in bytes.
> The item size is determined by the data type (dtype) of the array. Different data types require different amounts of memory. For example, an int32 type requires "4" bytes per element, while a float64 type requires "8" bytes per element.
**int32**type requires "4" bytes per element, while a**float64**type requires "8" bytes per element.
### Example 1

In the following example, we are checking the item size for an array of integer data type "int8" −

```
# dtype of array is int8 (1 byte) 
import numpy as np 
x = np.array([1,2,3,4,5], dtype = np.int8) 
print (x.itemsize)
```

We get the output as shown below −

```
1
```

### Example 2

Now, we are checking the item size for an array of float data type "float32" −

```
# dtype of array is now float32 (4 bytes) 
import numpy as np 
x = np.array([1,2,3,4,5], dtype = np.float32) 
print (x.itemsize)
```

The result produced is as follows −

```
4
```

## NumPy Buffer Information Attribute

The
**nbytes**attribute returns the total number of bytes consumed by the elements of the array.
In NumPy, the buffer information of an array provides details about the underlying memory structure that stores the array data. This includes information on the memory layout, the data type, and the byte offset within the buffer.

### Example

In this example, we are using the "nbytes" attribute to retrieve the total memory used by the arrays data buffer −

```
import numpy as np
# Creating an array
array = np.array([1, 2, 3, 4, 5], dtype=np.int32)

# Checking total memory size of the array
print("Total memory size of the array:", array.nbytes, "bytes")
```

The output obtained is as shown below −

```
Total memory size of the array: 20 bytes
```

## NumPy Strides Attribute

The
**strides**attribute provides the number of bytes to step in each dimension when traversing an array.
Strides specify the number of bytes that must be skipped in memory to move from one element to the next along each axis. They help in determining how the array is laid out in memory and how to access elements.

### Example

In the example below, we are accessing an element in a 2D array using "strides" attribute to calculate the memory address −

```
import numpy as np

# Creating a 2D array
array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Checking the strides
print("Array shape:", array.shape)
print("Array strides:", array.strides)
```

The stride for the first axis (rows) is 16 bytes, which means to move from one row to the next, NumPy skips 16 bytes in memory. The stride for the second axis (columns) is 4 bytes, indicating that to move from one column to the next within the same row, NumPy skips 4 bytes −

```
Array shape: (3, 4)
Array strides: (32, 8)
```

## NumPy Flags Attribute

The
**flags**attribute returns information about the memory layout of the array, such as whether it is contiguous in memory.
NumPy provides several flags that describe different aspects of the arrays memory layout and properties −
Sr.No.Attribute & Description1**C_CONTIGUOUS (C)**
The data is in a single, C-style contiguous segment
2**F_CONTIGUOUS (F)**
The data is in a single, Fortran-style contiguous segment
3**OWNDATA (O)**
The array owns the memory it uses or borrows it from another object
4**WRITEABLE (W)**
The data area can be written to. Setting this to False locks the data, making it read-only
5**ALIGNED (A)**
The data and all elements are aligned appropriately for the hardware
6**UPDATEIFCOPY (U)**
This array is a copy of some other array. When this array is deallocated, the base array will be updated with the contents of this array

### Example

The following example shows the current values of array's flags −

```
import numpy as np 
x = np.array([1,2,3,4,5]) 
print (x.flags)
```

Following is the output obtained −

```
C_CONTIGUOUS : True 
F_CONTIGUOUS : True 
OWNDATA : True 
WRITEABLE : True 
ALIGNED : True 
UPDATEIFCOPY : False
```

## NumPy Base Attribute

The
**base**attribute returns the base object if the array is a view on another array. If the array owns its data, base is "None". In NumPy, the concept of "array base" refers to the original array from which a new array is derived.
### Example

In this example, "view_array" is a view into "original_array", and the base attribute of "view_array" points to "original_array" −

```
import numpy as np

# Creating an array
original_array = np.array([[1, 2, 3], [4, 5, 6]])

# Creating a view (a slice) of the original array
view_array = original_array[0:1, :]

# Checking the base of the view
print("Base array of view_array:", view_array.base)
```

The result produced is as follows −

```
Base array of view_array: [[1 2 3]
 [4 5 6]]
```

## NumPy Real and Imaginary Parts Attribute

For arrays with complex numbers, the
**real**and**imag**attributes return the real and imaginary parts, respectively.
### Example

In this example, we are using the "real" attribute to return an array with the real parts and "imag" attribute to return an array with the imaginary parts −

```
import numpy as np

# Creating an array of complex numbers
complex_array = np.array([1+2j, 3+4j, 5+6j])

# Accessing the real part
real_part = complex_array.real
print("Real part:", real_part)

# Accessing the imaginary part
imaginary_part = complex_array.imag
print("Imaginary part:", imaginary_part)
```

We get the output as shown below −

```
Real part: [1. 3. 5.]
Imaginary part: [2. 4. 6.]
```

## Attributes List

Following is the list of various Array attributes in NumPy −
Sr.No.Operation & Description1[ndarray.ndim](/numpy/numpy_ndarray_ndim_attribute.htm)
This attribute returns the number of dimensions in the array.
2[ndarray.shape](/numpy/numpy_ndarray_shape_attribute.htm)
This attribute returns the size of the array in each dimension..
3[ndarray.size](/numpy/numpy_ndarray_size_attribute.htm)
This attribute returns number of elements in the array.
4[ndarray.dtype](/numpy/numpy_ndarray_dtype_attribute.htm)
This attribute returns the datatype of elements in the array.
5[ndarray.itemsize](/numpy/numpy_ndarray_itemsize_attribute.htm)
This returns the size of each element in the array.
6[ndarray.nbytes](/numpy/numpy_ndarray_nbytes.htm)
This returns the total bytes consumed by the array elements.
7[ndarray.T](/numpy/numpy_ndarray_t_attribute.htm)
Returns the view of the transposed array.
8[ndarray.real](/numpy/numpy_ndarray_real_attribute.htm)
Returns the real part of the array.
9[ndarray.imag](/numpy/numpy_ndarray_imag_attribute.htm)
Returns the imaginary part of the array.
10[ndarray.flat](/numpy/numpy_ndarray_flat_attribute.htm)
1D array iterator over the array.
11[ndarray.ctypes](/numpy/numpy_ndarray_ctypes_attribute.htm)
This returns an object to simplify the interaction of the array with the ctypes module.
12[ndarray.data](/numpy/numpy_ndarray_data_attribute.htm)
This returns the buffer containing the actual elements of the array in the memory.

---

## 24. NumPy - Array Shape

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_shape.htm](https://www.tutorialspoint.com/numpy/numpy_array_shape.htm)*

---

---
[Previous](/numpy/numpy_array_attributes.htm)[Quiz](/numpy/quiz_on_numpy_array_shape.htm)[Next](/numpy/numpy_array_size.htm)
## NumPy Array Shape

The shape of a NumPy array is a tuple of integers. Each integer in the tuple represents the size of the array along a particular dimension or axis. For example, an array with shape (3, 4) has 3 rows and 4 columns.

- For a 2D array, the shape is a tuple with two elements:**number of rows**,**number of columns**.
- For a 3D array, the shape is a tuple with three elements:**depth**,**number of rows**,**number of columns**.
- Higher-dimensional arrays follow the same pattern, with each dimension's size represented as an additional element in the tuple.
## Accessing Array Shape

You can access the shape of a NumPy array using the
**shape**attribute. This attribute returns a tuple of integers, each representing the size of the array along a particular dimension.
### Example

In the following example, we are creating a 2D array and retrieving its shape using the NumPy "shape" attribute −

```
import numpy as np

# Creating a 2D array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Accessing the shape
print("Shape of the array:", array.shape)
print("Number of dimensions:", array.ndim)
print("Total number of elements:", array.size)
```

The shape (2, 3) indicates that the array has 2 rows and 3 columns. It is a two-dimensional array −

```
Shape of the array: (2, 3)
Number of dimensions: 2
Total number of elements: 6
```

## Changing Array Shape

Changing the shape of a NumPy array refers to transforming the dimensions of an array without altering its data. For instance, a one-dimensional array can be reshaped into a two-dimensional array or vice versa, as long as the total number of elements remains constant.

To reshape an array in NumPy, we use the
**reshape()**function. This function returns a new view of the array with the specified shape if possible. If the reshape is not possible with a view, a copy of the array is created.
### Example

In this example, we are changing the shape of a 2D array to 1D array by passing "-1" as an argument to the Numpy reshape() function. This automatically infer the size of one dimension −

```
import numpy as np

# Creating a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("Original 2D array:\n", array_2d)

# Reshaping to a 1D array
array_flattened = array_2d.reshape(-1)
print("Flattened to 1D array:", array_flattened)
```

This will produce the following result −

```
Original 2D array:
[[1 2 3]
 [4 5 6]]
Flattened to 1D array: [1 2 3 4 5 6]
```

## Handling Reshape Errors

Sometimes reshaping arrays in NumPy can lead to errors if not used correctly. This error occurs when you attempt to reshape an array into a shape that is incompatible with the total number of elements in the array.

> The total number of elements must remain constant when reshaping. If the reshape operation is incompatible with the total number of elements, NumPy will raise a ValueError.
**ValueError**.
### Example: Incompatible Shape Error

The
**incompatible shape error**occurs when you attempt to reshape an array into a shape that is incompatible with the total number of elements in the array.
In the example below, the original array has "12" elements. Reshaping it into a shape "(3, 5)" requires 15 elements, which causes a ValueError −

```
import numpy as np

# Creating an array with 12 elements
array = np.arange(12)

# Attempting to reshape into a shape that requires more elements
try:
   reshaped_array = array.reshape((3, 5))
except ValueError as e:
   print("Error:", e)
```

Following is the output of the above code −

```
Error: cannot reshape array of size 12 into shape (3,5)
```

### Example: Negative Dimension Error

Using
**-1**in the reshape dimensions tells NumPy to automatically calculate the size of that dimension. However, if the remaining dimensions don't align with the total number of elements, it will raise an error −
```
import numpy as np

# Creating an array with 10 elements
array = np.arange(10)

# Attempting to reshape with an incompatible automatic dimension
try:
   reshaped_array = array.reshape((2, -1, 4))
except ValueError as e:
   print("Error:", e)
```

The output obtained is as shown below −

```
Error: cannot reshape array of size 10 into shape (2,newaxis,4)
```

### Example: Incorrect Dimension Specification

Specifying an incorrect or non-integer dimension value (e.g., a negative value other than -1, or a non-integer) can lead to errors −

```
import numpy as np

# Creating an array with 16 elements
array = np.arange(16)

# Attempting to reshape with an invalid dimension
try:
   reshaped_array = array.reshape((4, 4.5))
except ValueError as e:
   print("Error:", e)
```

After executing the above code, we get the following output −

```
Traceback (most recent call last):
  File "/home/cg/root/669f5fd83ed84/main.py", line 8, in <module>
reshaped_array = array.reshape((4, 4.5))
TypeError: 'float' object cannot be interpreted as an integer
```

---

## 25. NumPy - Array Size

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_size.htm](https://www.tutorialspoint.com/numpy/numpy_array_size.htm)*

---

---
[Previous](/numpy/numpy_array_shape.htm)[Quiz](/numpy/quiz_on_numpy_array_size.htm)[Next](/numpy/numpy_array_strides.htm)
## NumPy Array Size

The size of a NumPy array refers to the total number of elements contained within the array. Understanding the size of an array is important for performing various operations, such as reshaping, broadcasting, and iterating through elements.

In this tutorial, we will discuss how to determine and manipulate the size of NumPy arrays.

## Checking Array Size

NumPy provides the
**size**attribute to check the number of elements in an array. This attribute returns an integer representing the total number of elements, regardless of its shape. Whether the array is one-dimensional or multi-dimensional, size will always provide the count of all elements present.
Knowing the size of an array can be useful in various scenarios, such as iterating over elements, reshaping arrays, and optimizing memory usage.

### Example: Iterating Over Elements

When you need to iterate through each element in an array, knowing the size helps you define the range for your loop −

```
import numpy as np

# Creating a 1D array
array = np.array([1, 2, 3, 4, 5])

# Iterating using the size attribute
for i in range(array.size):
   print(array[i])
```

Following is the output obtained −

```
1
2
3
4
5
```

### Example: Memory Calculation

You can calculate the memory consumption of an array by multiplying its size by the size of each element using the "itemsize" attribute in NumPy −

```
import numpy as np

array = np.ones((1000, 1000), dtype=np.float64)

memory_usage = array.size * array.itemsize
print("Memory usage in bytes:", memory_usage)
```

This will produce the following result −

```
Memory usage in bytes: 8000000
```

## NumPy Array Size vs. Array Shape

While the
**size**attribute gives the total number of elements in an array, the**shape**attribute provides the dimensions of the array. The size of an array can be calculated from the shape by multiplying all the dimensions.
### Example

In this example, we are retrieving the shape and size of an array using the NumPy "shape" and "size" attribute respectively −

```
import numpy as np

# Creating a 3D array
array = np.zeros((2, 3, 4))

# Checking the shape and size
print("Array shape:", array.shape)
print("Array size:", array.size)
```

The shape of the array is (2, 3, 4), and the size is 2 * 3 * 4 = 24 −

```
Array shape: (2, 3, 4)
Array size: 24
```

## Resizing Arrays

The
**resize()**function in NumPy is used to resize an array. This function changes the shape and size of the array in-place, and the new shape must be compatible with the total number of elements.
If the new shape is larger than the original, the new array is filled with repeated copies of the original array. If the new shape is smaller, the original array is truncated.

> Unlike the reshape() function, which only changes the shape of an array without changing its data, the resize() function can change the size of the array as well, adding new elements or removing existing ones.

### Example: Resizing to a Larger Array

In this example, the original array is resized to a "2x5" array. The new elements are filled by repeating the original array −

```
import numpy as np

# Creating a 1D array
array_1d = np.array([1, 2, 3])

# Resizing the array to a larger size
resized_array = np.resize(array_1d, (2, 5))

print("Original array:\n", array_1d)
print("Resized array:\n", resized_array)
```

After executing the above code, we get the following output −

```
Original array:
 [1 2 3]
Resized array:
 [[1 2 3 1 2]
  [3 1 2 3 1]]
```

### Example: Resizing to a Smaller Array

Here, the original "2x3" array is resized to a "2x2" array. The excess elements are truncated −

```
import numpy as np

# Creating a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Resizing the array to a smaller size
resized_array = np.resize(array_2d, (2, 2))

print("Original array:\n", array_2d)
print("Resized array:\n", resized_array)
```

The result produced is as follows −

```
Original array:
 [[1 2 3]
  [4 5 6]]
Resized array:
 [[1 2]
  [3 4]]
```

### Example: In-Place Resizing

In this example, the original array is resized in place to a "2x5" array. The new elements are filled with zeros −

```
import numpy as np

# Creating a 1D array
array_1d = np.array([1, 2, 3])

# In-place resizing the array to a larger size
array_1d.resize((2, 5))

print("Resized array:\n", array_1d)
```

We get the output as shown below −

```
Resized array:
 [[1 2 3 0 0]
  [0 0 0 0 0]]
```

---

## 26. NumPy - Array Strides

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_strides.htm](https://www.tutorialspoint.com/numpy/numpy_array_strides.htm)*

---

---
[Previous](/numpy/numpy_array_size.htm)[Quiz](/numpy/quiz_on_numpy_array_strides.htm)[Next](/numpy/numpy_array_itemsize.htm)
## NumPy Array Strides

In NumPy,
**strides**are tuples of integers representing the number of bytes to step in each dimension when traversing an array. It provide the ability to access elements in the array without explicitly copying data.
Strides are calculated based on the shape and data type of the array −

- For a 1D array with a data type of 4 bytes (e.g., int32), the stride is simply the data type size.
- For multi-dimensional arrays, strides are calculated by multiplying the size of the inner dimension by the stride of the previous dimension.
## Accessing Strides in NumPy

You can access the strides of a NumPy array using the
**strides**attribute. This attribute returns a tuple where each value represents the number of bytes to move in memory to access the next element along each dimension.
### Example

In the following example, we are calculating the stride of an array using the NumPy "stride" attribute −

```
import numpy as np

# Creating a 2D array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Accessing the strides
print("Array strides:", array.strides)
```

The strides (24, 8) indicate that to move from one row to the next, 24 bytes are skipped, and to move from one column to the next, 8 bytes are skipped −

```
Array strides: (24, 8)
```

## How NumPy Strides Work

Strides are calculated based on the shape and data type of the array. For a given dimension, the stride is the product of the element size (in bytes) and the number of elements in the subsequent dimensions.

For a 2D array with shape
**(m, n)**and data type**dtype**−
- Stride for the first dimension:**stride[0] = n * size_of(dtype)**
- Stride for the second dimension:**stride[1] = size_of(dtype)**
### Example: Basic Strides

In the example below, we are accessing the strides of a basic 1D NumPy array −

```
import numpy as np

# Creating a 1D array
array_1d = np.array([1, 2, 3, 4, 5])

# Accessing strides
print("1D Array strides:", array_1d.strides)
```

The stride (8,) indicates that each element is 8 bytes apart in memory, which is typical for an array of integers −

```
1D Array strides: (8,)
```

### Example: Changing Strides

Transposing the array changes the strides, reflecting the new memory layout as shown in the example below −

```
import numpy as np

# Creating a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Transposing the array
array_2d_T = array_2d.T

# Accessing strides
print("Original array strides:", array_2d.strides)
print("Transposed array strides:", array_2d_T.strides)
```

The stride (8,) indicates that each element is 8 bytes apart in memory, which is typical for an array of integers −

```
Original array strides: (24, 8)
Transposed array strides: (8, 24)
```

### Example: Memory Optimization with Strides

Using strides can help optimize memory usage by allowing efficient access patterns −

```
import numpy as np

# Creating a large array
large_array = np.zeros((1000, 1000))

# Accessing every 10th row
strided_array = large_array[::10, :]

print("Strided array shape:", strided_array.shape)
print("Strided array strides:", strided_array.strides)
```

The strides indicate that we are skipping 80,000 bytes (10 rows) to access the next row, optimizing memory access −

```
Strided array shape: (100, 1000)
Strided array strides: (80000, 8)
```

## Strides in Multi-Dimensional Arrays

Strides in multi-dimensional arrays work similarly, with each stride value indicating the step size in bytes for the corresponding dimension.

For a multi-dimensional array, the stride for each dimension is the product of the size of elements and the cumulative product of the sizes of subsequent dimensions.

This means the stride for the last dimension is simply the size of the data type, the stride for the second-to-last dimension is the size of the last dimension multiplied by the size of the data type, and so on.

### Example

In the example below, we are calculating the strides of a 3 dimensional array −

```
import numpy as np

# Creating a 3D array
array_3d = np.zeros((2, 3, 4))

# Accessing strides
print("3D Array strides:", array_3d.strides)
```

The strides obtained shows the byte steps for each dimension −

```
3D Array strides: (96, 32, 8)
```

## Strides for Slicing Operations

Strides are useful when it comes to performing slicing operations in NumPy arrays. When you slice a NumPy array, you often create a view rather than a copy of the array. This view shares the same underlying data but may have a different shape or memory layout.

Strides determine how many bytes to step in memory to move from one element to the next along each dimension. By adjusting strides, you can access specific patterns of data efficiently.

> Slicing operations in NumPy allow you to extract subsets of an array, ranging from individual elements to specific sections, without copying the underlying data.

### Example

In this example, we are creating a large 2D array and access every 10th row without copying the data using the slicing operation with strides −

```
import numpy as np

# Creating a large 2D array
large_array = np.arange(10000).reshape((100, 100))

# Accessing every 10th row
strided_array = large_array[::10, :]

print("Original array shape:", large_array.shape)
print("Strided array shape:", strided_array.shape)
print("Strided array strides:", strided_array.strides)
```

Following is the output obtained −

```
Original array shape: (100, 100)
Strided array shape: (10, 100)
Strided array strides: (4000, 40)
```

---

## 27. NumPy - Array Itemsize

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_itemsize.htm](https://www.tutorialspoint.com/numpy/numpy_array_itemsize.htm)*

---

---
[Previous](/numpy/numpy_array_strides.htm)[Quiz](/numpy/quiz_on_numpy_array_itemsize.htm)[Next](/numpy/numpy_broadcasting.htm)
## NumPy Array Itemsize

The
**itemsize**attribute in a NumPy array indicates the size, in bytes, of each element in the array. This size is determined by the data type of the array elements (e.g., integer, float).
By knowing the itemsize, you can estimate the total memory consumption of the array.  This is important for understanding the memory layout and storage requirements of arrays, especially when dealing with large datasets.

## Accessing Array Itemsize

You can access the itemsize of a NumPy array using the
**itemsize**attribute. This attribute returns an integer representing the size (in bytes) of each element in the array.
### Example

In the following example, we are accessing the itemsize of an integer array and a float array −

```
import numpy as np

# Creating arrays with different data types
array_int32 = np.array([1, 2, 3], dtype=np.int32)
array_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

# Checking itemsize
print("Itemsize of int32 array:", array_int32.itemsize)
print("Itemsize of float64 array:", array_float64.itemsize)
```

Following is the output obtained −

```
Itemsize of int32 array: 4
Itemsize of float64 array: 8
```

## Calculating Itemsize Memory Usage

To calculate the total memory occupied by the array, you can multiply the "itemsize" by the total number of elements in the array.

For example, if an array has "1000" elements and an itemsize of "8" bytes, the total memory used by the array would be "1000 * 8" = "8000" bytes.

### Example

In this example, we are calculating the total memory usage of a 2D array −

```
import numpy as np

# Create a 2D array
array_2d = np.array([[1, 2], [3, 4]], dtype=np.float32)

# Calculate total memory usage
total_memory_usage = array_2d.size * array_2d.itemsize
print(f"Total memory usage: {total_memory_usage} bytes")
```

This will produce the following result −

```
Total memory usage: 16 bytes
```

## Itemsize with Different Data Types

In NumPy, different data types have different itemsize values. For example −

- **np.int8**has an itemsize of**1**byte.
- **np.int16**has an itemsize of**2**bytes.
- **np.float64**has an itemsize of**8**bytes.
- **np.complex128**has an itemsize of**16**bytes.
### Example

In the example below, we are creating arrays with different data types and then checking itemsize of each array −

```
import numpy as np

# Creating arrays with different data types
array_int8 = np.array([1, 2, 3], dtype=np.int8)
array_int16 = np.array([1, 2, 3], dtype=np.int16)
array_uint32 = np.array([1, 2, 3], dtype=np.uint32)
array_float16 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
array_complex128 = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)

# Checking itemsize
print("Itemsize of int8 array:", array_int8.itemsize)
print("Itemsize of int16 array:", array_int16.itemsize)
print("Itemsize of uint32 array:", array_uint32.itemsize)
print("Itemsize of float16 array:", array_float16.itemsize)
print("Itemsize of complex128 array:", array_complex128.itemsize)
```

The output obtained is as shown below −

```
Itemsize of int8 array: 1
Itemsize of int16 array: 2
Itemsize of uint32 array: 4
Itemsize of float16 array: 2
Itemsize of complex128 array: 16
```

## Changing Itemsize by Modifying Data Types

By modifying the data type of the array, you can change its itemsize, which affects how much memory each element occupies and, consequently, the total memory usage of the array.

You can change the data type of a NumPy array using the astype() function. This function accepts the target data type to which you want to convert the array.

### Example

In the following example, we are changing the data type of an array from "int32" to "int8" and checking the itemsize of that array −

```
import numpy as np

# Original array with int32
array_original = np.array([1, 2, 3], dtype=np.int32)
print(f"Original itemsize: {array_original.itemsize} bytes")

# Change data type to int8
array_new = array_original.astype(np.int8)
print(f"New itemsize: {array_new.itemsize} bytes")
```

After executing the above code, we get the following output −

```
Original itemsize: 4 bytes
New itemsize: 1 bytes
```

---

## 28. NumPy - Broadcasting

*Source: [https://www.tutorialspoint.com/numpy/numpy_broadcasting.htm](https://www.tutorialspoint.com/numpy/numpy_broadcasting.htm)*

---

---
[Previous](/numpy/numpy_array_itemsize.htm)[Quiz](/numpy/quiz_on_numpy_broadcasting.htm)[Next](/numpy/numpy_arithmetic_operations.htm)
## NumPy Broadcasting

Broadcasting in NumPy refers to the ability of performing operations on arrays with different shapes by automatically expanding the smaller array's shape to match the larger array's shape. This is useful when performing arithmetic operations or applying functions to arrays of different dimensions.

When performing arithmetic operations, NumPy operates on corresponding elements of the arrays. If the arrays have the same shape, operations are are smoothly performed. However, if the arrays have different shapes, NumPy uses broadcasting to align them, allowing element-wise operations to be conducted easily.

### Rules of Broadcasting

For broadcasting to work, the following rules must be satisfied −

- If arrays have a different number of dimensions, the shape of the smaller-dimensional array is padded with ones on the left side until both shapes have the same length.
- The size of each dimension must either be the same or one of them must be one.
- Broadcasting is applied from the last dimension to the first dimension.
For instance, consider two arrays with shapes
**(3, 4)**and**(4,)**. The broadcasting rules will align the shapes as follows −
- **Pad the smaller array's shape:**The smaller array shape (4,) is padded to (1, 4).
- **Align dimensions:**The shapes (3, 4) and (1, 4) are aligned as (3, 4) and (3, 4) respectively.
- **Perform element-wise operation:**The operation is applied to the aligned shapes.
## Adding a Scalar to an Array

When adding a scalar to an array, NumPy uses broadcasting to apply the scalar to each element of the array. Broadcasting expands the scalar to match the shape of the array, enabling element-wise operations.

### Example

In the following example, we are broadcasting the scalar "10" to each element of the array, resulting in each element being increased by 10 −

```
import numpy as np

# Creating an array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Adding a scalar
result = array + 10
print(result)
```

Following is the output obtained −

```
[[11 12 13][14 15 16]]
```

## Adding Arrays of Different Shapes

When adding arrays of different shapes, NumPy applies broadcasting rules to make their shapes compatible. Broadcasting works by stretching the smaller array across the larger one, so that both arrays have the same shape for element-wise addition.

This process eliminates the need for manually reshaping arrays before performing operations.

### Example 1

In this example, we broadcast the second array "array2" to match the shape of the first array "array1" −

```
import numpy as np

# Creating two arrays with different shapes
array1 = np.array([[1, 2, 3], [4, 5, 6]])     
array2 = np.array([10, 20, 30])              

# Adding arrays with broadcasting
result = array1 + array2
print(result)
```

This will produce the following result −

```
[[11 22 33]
 [14 25 36]]
```

### Example 2

Following is another example to broadcast two arrays with different shapes in NumPy −

```
import numpy as np 
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])  
   
print ('First array:') 
print (a) 
print ('\n') 
   
print ('Second array:')
print (b)
print ('\n')
   
print ('First Array + Second Array')
print (a + b)
```

The output of this program would be as follows −

```
First array:
[[ 0.  0.  0.]
 [10. 10. 10.]
 [20. 20. 20.]
 [30. 30. 30.]]

Second array:
[1. 2. 3.]

First Array + Second Array
[[ 1.  2.  3.]
 [11. 12. 13.]
 [21. 22. 23.]
 [31. 32. 33.]]
```

The following figure demonstrates how array
**b**is broadcast to become compatible with**a**.![array](/numpy/images/array.jpg)
## Broadcasting with Multi-Dimensional Arrays

When performing operations between multi-dimensional arrays with different shapes, broadcasting rules align their dimensions so that they can be operated on element-wise.

This process involves stretching the smaller array to match the shape of the larger one, enabling operations to be performed smoothly.

### Example

In the following example, we are creating two 3D arrays and then adding them with broadcasting −

```
import numpy as np

# Creating two 3D arrays
array1 = np.ones((2, 3, 4))                    
array2 = np.arange(4)                        

# Adding arrays with broadcasting
result = array1 + array2
print(result)
```

Following is the output of the above code −

```
[[[1. 2. 3. 4.]
  [1. 2. 3. 4.]
[1. 2. 3. 4.]]

 [[1. 2. 3. 4.]
  [1. 2. 3. 4.]
  [1. 2. 3. 4.]]]
```

## Applying Functions with Broadcasting

Broadcasting not only simplifies arithmetic operations between arrays of different shapes but also allows functions to be applied across arrays. These functions can include −

- **Mathematical Functions:**Functions that perform mathematical operations, such as addition, subtraction, multiplication, and division.
- **Statistical Functions:**Functions that compute statistical properties, like mean, median, variance, and standard deviation.
- **Reduction Functions:**Functions that reduce the dimensions of an array by performing operations such as sum, product, or maximum.
- **Logical Operations:**Functions that perform logical operations, such as comparisons and logical operations (e.g., AND, OR, NOT).
When applying functions to arrays with different shapes, broadcasting ensures that the function is applied element-wise.

### Example

In this example, we use the numpy.maximum() function to perform an element-wise comparison between two arrays. The function compares each element of "array1" with the corresponding element of "array2", and the result is an array where each element is the maximum of the corresponding elements from the input arrays −

```
import numpy as np

# Creating arrays
array1 = np.array([[1, 2, 3], [4, 5, 6]])   
array2 = np.array([[10], [20]])              

# Applying a function with broadcasting
result = np.maximum(array1, array2)
print(result)
```

After executing the above code, we get the following output −

```
[[10 10 10]
 [20 20 20]]
```

---

## 29. NumPy - Arithmetic Operations

*Source: [https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.htm](https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.htm)*

---

---
[Previous](/numpy/numpy_broadcasting.htm)[Quiz](/numpy/quiz_on_numpy_arithmetic_operations.htm)[Next](/numpy/numpy_array_addition.htm)
## NumPy Arithmetic Operations

NumPy makes performing arithmetic operations on arrays simple and easy. With NumPy, you can add, subtract, multiply, and divide entire arrays element-wise, meaning that each element in one array is operated on by the corresponding element in another array.

When performing arithmetic operations with arrays of different shapes, NumPy uses a feature called broadcasting. It automatically adjusts the shapes of the arrays so that the operation can be performed, extending the smaller array across the larger one as needed.

## Basic NumPy Arithmetic Operations

NumPy provides several arithmetic operations that are performed element-wise on arrays. These include addition, subtraction, multiplication, division, and power.

### NumPy Array Addition

Addition in NumPy is performed element-wise. When two arrays of the same shape are added, corresponding elements are summed together. Broadcasting rules apply when arrays have different shapes −

```
import numpy as np

# Creating two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Adding arrays
result = a + b
print(result)
```

Following is the output obtained −

```
[5 7 9]
```

### NumPy Array Subtraction

Subtraction in NumPy is also element-wise. Subtracting two arrays with the same shape returns an array where each element is the difference of the corresponding elements in the input arrays −

```
import numpy as np

# Creating two arrays
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

# Subtracting arrays
result = a - b
print(result)
```

This will produce the following result −

```
[ 9 18 27]
```

### NumPy Array Multiplication

Element-wise multiplication is performed using the
*****operator in NumPy. When multiplying arrays, each element of the first array is multiplied by the corresponding element in the second array −
```
import numpy as np

# Creating two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Multiplying arrays
result = a * b
print(result)
```

Following is the output of the above code −

```
[ 4 10 18]
```

### NumPy Array Division

Division is performed element-wise using the
**/**operator in NumPy. It results in an array where each element is the quotient of the corresponding elements in the input arrays −
```
import numpy as np

# Creating two arrays
a = np.array([10, 20, 30])
b = np.array([1, 2, 5])

# Dividing arrays
result = a / b
print(result)
```

The output obtained is as shown below −

```
[10. 10.  6.]
```

### NumPy Array Power Operation

The power operation is performed element-wise using the
******operator in NumPy. Each element of the base array is raised to the power of the corresponding element in the exponent array −
```
import numpy as np

# Creating two arrays
a = np.array([2, 3, 4])
b = np.array([1, 2, 3])

# Applying power operation
result = a ** b
print(result)
```

After executing the above code, we get the following output −

```
[ 2  9 64]
```

We can also use the
**numpy.power()**function to raise elements of an array to the specified power. This function treats elements in the first input array as base and returns it raised to the power of the corresponding element in the second input array −
```
import numpy as np 
a = np.array([10,100,1000]) 

print ('Our array is:')
print (a)
print ('\n')

print ('Applying power function:')
print (np.power(a,2))
print ('\n') 

print ('Second array:')
b = np.array([1,2,3]) 
print (b) 
print ('\n') 

print ('Applying power function again:')
print (np.power(a,b))
```

It will produce the following output −

```
Our array is:
[  10  100 1000]

Applying power function:
[    100   10000 1000000]

Second array:
[1 2 3]

Applying power function again:
[       10      10000 1000000000]
```

## Advanced NumPy Arithmetic Operations

Advanced arithmetic operations in NumPy include operations such as modulo, floor division, and power. These operations handle more complex mathematical tasks and are performed element-wise, similar to basic arithmetic operations, but with additional functionality for modular arithmetic and exponentiation.

### NumPy Modulo Operation

The modulo operation is performed using the
**%**operator in NumPy. When applied to arrays, it operates element-wise, meaning each element of the first array is divided by the corresponding element in the second array, and the remainder is calculated −
```
import numpy as np

# Creating two arrays
a = np.array([10, 20, 30])
b = np.array([3, 7, 8])

# Applying modulo operation
result = a % b
print(result)
```

The result produced is as follows −

```
[1 6 6]
```

We can also use the
**numpy.mod()**function to calculate the element-wise remainder of division (modulus operation) between the elements of two arrays or between an array and a scalar.
This function returns the remainder when one array is divided by another array or scalar, applying the modulus operation element by element.

```
import numpy as np 
a = np.array([10,20,30]) 
b = np.array([3,5,7]) 

print ('First array:')
print (a)
print ('\n')  

print ('Second array:') 
print (b) 
print ('\n') 

print ('Applying mod() function:')
print (np.mod(a,b))
print ('\n')

print ('Applying remainder() function:') 
print (np.remainder(a,b))
```

Following is the output of the above code −

```
First array:                                                                  
[10 20 30]

Second array:                                                                 
[3 5 7]

Applying mod() function:                                                      
[1 0 2]

Applying remainder() function:                                                
[1 0 2]
```

### NumPy Floor Division

The floor division in NumPy is performed element-wise using the
**//**operator. It returns the largest integer less than or equal to the division result −
```
import numpy as np

# Creating two arrays
a = np.array([10, 20, 30])
b = np.array([3, 7, 8])

# Applying floor division
result = a // b
print(result)
```

We get the output as shown below −

```
[3 2 3]
```

## NumPy Arithmetic Operations with Broadcasting

Broadcasting allows NumPy to perform arithmetic operations on arrays of different shapes by virtually expanding the smaller array to match the shape of the larger array.

### Scalar and Array Operations in NumPy

When a scalar is used with an array, broadcasting expands the scalar to match the shape of the array, allowing element-wise operations −

```
import numpy as np

# Creating an array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Scalar value
scalar = 10

# Adding scalar to array
result = a + scalar
print(result)
```

Following is the output obtained −

```
[[11 12 13]
 [14 15 16]]
```

### Array with Different Shapes in NumPy

When two arrays of different shapes are used in NumPy, broadcasting aligns their shapes according to broadcasting rules −

```
import numpy as np

# Creating arrays with different shapes
a = np.array([[1, 2, 3], [4, 5, 6]])  
b = np.array([10, 20, 30])  

# Adding arrays
result = a + b
print(result)
```

This will produce the following result −

```
[[11 22 33]
 [14 25 36]]
```

## NumPy Aggregation Functions

Aggregation functions in NumPy perform operations like sum, mean, min, and max across arrays, often using broadcasting to handle arrays of different shapes.

### NumPy Sum Operation

The NumPy sum operation calculates the sum of array elements over a specified axis or over the entire array if no axis is specified. −

```
import numpy as np

# Creating an array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Summing elements
result = np.sum(a)
print(result)

# Summing along axis 0 (columns)
result_axis0 = np.sum(a, axis=0)
print(result_axis0)

# Summing along axis 1 (rows)
result_axis1 = np.sum(a, axis=1)
print(result_axis1)
```

Following is the output of the above code −

```
21
[5 7 9]
[ 6 15]
```

### NumPy Mean Operation

The NumPy mean operation calculates the average (arithmetic mean) of array elements over a specified axis or over the entire array −

```
import numpy as np

# Creating an array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Mean of elements
result = np.mean(a)
print(result)

# Mean along axis 0 (columns)
result_axis0 = np.mean(a, axis=0)
print(result_axis0)

# Mean along axis 1 (rows)
result_axis1 = np.mean(a, axis=1)
print(result_axis1)
```

The output obtained is as shown below −

```
3.5
[2.5 3.5 4.5]
[2. 5.]
```

## NumPy Array Operations with Complex Numbers

The following functions are used to perform operations on array with complex numbers.

- **numpy.real():**Returns the real part of the complex data type argument.
- **numpy.imag():**Returns the imaginary part of the complex data type argument.
- **numpy.conj():**Returns the complex conjugate, which is obtained by changing the sign of the imaginary part.
- **numpy.angle():**Returns the angle of the complex argument. The function has degree parameter. If true, the angle in the degree is returned, otherwise the angle is in radians.
### Example

In the following example, we are using NumPy functons: real(), imag(), conj() and angle() to perform operations on array with complex numbers −

```
import numpy as np 
a = np.array([-5.6j, 0.2j, 11. , 1+1j]) 

print ('Our array is:')
print (a)
print ('\n') 

print ('Applying real() function:')
print (np.real(a))
print ('\n')

print ('Applying imag() function:') 
print (np.imag(a))
print ('\n')  

print ('Applying conj() function:') 
print (np.conj(a)) 
print ('\n')  

print ('Applying angle() function:') 
print (np.angle(a)) 
print ('\n') 

print ('Applying angle() function again (result in degrees)') 
print (np.angle(a, deg = True))
```

It will produce the following output −

```
Our array is:
[ 0.-5.6j 0.+0.2j 11.+0.j 1.+1.j ]

Applying real() function:
[ 0. 0. 11. 1.]

Applying imag() function:
[-5.6 0.2 0. 1. ]

Applying conj() function:
[ 0.+5.6j 0.-0.2j 11.-0.j 1.-1.j ]

Applying angle() function:
[-1.57079633 1.57079633 0. 0.78539816]

Applying angle() function again (result in degrees)
[-90. 90. 0. 45.]
```

## Basic Array Operations

Several routines are available in NumPy package for operations of elements in array. NumPy arrays support operations like additions, subtraction and indexing for efficient data manipulation −
Sr.No.Operation & Description1[numpy.add()](/numpy/numpy_add_function.htm)
Enables efficient element-wise addition for data manipulation.
2[numpy.subtract()](/numpy/numpy_subtract_function.htm)
Computes element-wise differences between two arrays efficiently.
3[numpy.multiply()](/numpy/numpy_multiply_function.htm)
Computes element-wise products between two arrays efficiently.
4[numpy.divide()](/numpy/numpy_divide_function.htm)
Performs element-wise division, requiring non-zero, same-shaped arrays.
5[numpy.power()](/numpy/numpy_power_function.htm)
Raises elements of one array to another's element-wise.
6[numpy.mod()](/numpy/numpy_mod_function.htm)
Computes element-wise remainders of division between two arrays.
7[numpy.remainder()](/numpy/numpy_remainder_function.htm)
Computes element-wise remainders of division between two arrays.
8[numpy.divmod()](/numpy/numpy_divmod_function.htm)
Returns a tuple with quotient and remainder of division.
7[numpy.abs()](/numpy/numpy_abs_function.htm)
Returns the positive absolute value of numbers.
8[numpy.absolute()](/numpy/numpy_absolute_function.htm)
Computes the absolute value of each array element efficiently.
9[numpy.fabs()](/numpy/numpy_fabs_function.htm)
Returns the positive magnitudes.
10[numpy.sign()](/numpy/numpy_sign_function.htm)
Returns an element-wise indication of the sign of a number.
11[numpy.conj()](/numpy/numpy_conj_function.htm)
Returns the complex conjugate, element-wise.
12[numpy.exp()](/numpy/numpy_exp_function.htm)
Calculates the exponential of all elements in the input array.
13[numpy.expm1()](/numpy/numpy_expm1_function.htm)
Calculates exp(x)-1 for all elements in the array.
14[numpy.log()](/numpy/numpy_log_function.htm)
Natural logarithm, element-wise.
15[numpy.log1p()](/numpy/numpy_log1p_function.htm)
Computes element-wise natural logarithm of (1 + input array).
16[numpy.log2()](/numpy/numpy_log2_function.htm)
Base-2 logarithm of x.
17[numpy.log10()](/numpy/numpy_log10_function.htm)
Returns the base 10 logarithm of the input array, element-wise.
18[numpy.sqrt()](/numpy/numpy_sqrt_function.htm)
Returns the non-negative square-root of an array, element-wise.
19[numpy.square()](/numpy/numpy_square_function.htm)
Returns the element-wise square of the input.
20[numpy.cbrt()](/numpy/numpy_cbrt_function.htm)
Returns the cube-root of an array, element-wise.
21[numpy.reciprocal()](/numpy/numpy_reciprocal_function.htm)
Calculates the reciprocal of each element in arrays.

---

## 30. NumPy - Array Addition

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_addition.htm](https://www.tutorialspoint.com/numpy/numpy_array_addition.htm)*

---

---
[Previous](/numpy/numpy_arithmetic_operations.htm)[Quiz](/numpy/quiz_on_numpy_array_addition.htm)[Next](/numpy/numpy_array_subtraction.htm)
## NumPy Array Addition

NumPy array addition allows you to perform element-wise addition between arrays. This operation adds corresponding elements from two arrays of the same shape, producing a new array of the same shape with the summed values.

If the arrays have different shapes, NumPy can broadcast the smaller array to match the shape of the larger array under certain conditions.

## Element-wise Addition in NumPy

Element-wise addition is the most basic form of array addition in NumPy, where corresponding elements of two arrays are added together to produce a new array.

This type of addition operates on arrays of the same shape, performing the addition operation individually for each pair of elements from the two arrays.

### Example

In the following example, we are adding each element of array
**a**is to the corresponding element of array**b**−
```
import numpy as np

# Creating two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Performing element-wise addition
result = a + b
print(result)
```

Following is the output obtained −

```
[5 7 9]
```

## Adding a Scalar to a NumPy Array

When a scalar (a single value) is added to an array, the scalar is broadcasted to match the shape of the array. This means that the scalar is effectively treated as if it were an array of the same shape as the original array, with all elements equal to the scalar value.

> Broadcasting describes how NumPy handles arrays with different shapes during arithmetic operations. When arrays of different shapes are involved in operations, NumPy automatically adjusts their shapes to match each other, following specific broadcasting rules.

### Example

In this example, we are adding the scalar "10" to each element of the array "a" −

```
import numpy as np

# Creating an array
a = np.array([1, 2, 3])

# Adding a scalar
result = a + 10
print(result)
```

This will produce the following result −

```
[11 12 13]
```

## Adding NumPy Arrays of Different Shapes

Broadcasting in NumPy allows for the addition of arrays with different shapes by adjusting their dimensions to match each other.

NumPy aligns the dimensions of arrays for broadcasting by comparing dimensions from the rightmost side and working backward. Two dimensions are considered compatible if they are equal or if one of them is 1, in which case it is broadcasted to match the other dimension.

When dimensions do not directly match, NumPy stretches the smaller array along the mismatched dimensions as necessary to match the shape of the larger array.

### Example

In the example below, array "b" is broadcasted to match the shape of array "a", and then element-wise addition is performed −

```
import numpy as np

# Creating arrays with different shapes
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])

# Adding arrays with broadcasting
result = a + b
print(result)
```

Following is the output of the above code −

```
[[11 22 33]
 [14 25 36]]
```

## Adding Multi-Dimensional Arrays with Broadcasting

In NumPy, broadcasting allows for arithmetic operations, such as addition, between multi-dimensional arrays of different shapes by automatically expanding the dimensions of the smaller array to match the larger array's shape.

This process involves aligning dimensions from the rightmost side and stretching the smaller array's dimensions as needed.

### Example

In the example below, we broadcast the one-dimensional array "a" to match the dimensions of the two-dimensional array "b" −

```
import numpy as np

# Creating multi-dimensional arrays
a = np.array([1, 2, 3])
b = np.array([[10], [20], [30]])

# Adding multi-dimensional arrays with broadcasting
result = a + b
print(result)
```

The output obtained is as shown below −

```
[[11 12 13]
 [21 22 23]
 [31 32 33]]
```

## Adding By Applying Functions with Broadcasting

Broadcasting in NumPy not only simplifies direct element-wise arithmetic operations but also allows for applying functions to arrays of different shapes. Using broadcasting, you can apply various mathematical functions across arrays with differing shapes.

### Example

In this example, we are adding the scalar "10" to each element of the array "a", and then apply the "sine" function element-wise −

```
import numpy as np

# Creating an array
a = np.array([1, 2, 3])

# Applying a function with broadcasting
result = np.sin(a + 10)
print(result)
```

After executing the above code, we get the following output −

```
[-0.99999021 -0.53657292  0.42016704]
```

## Adding Incompatible Arrays

If we attempt to add incompatible arrays in NumPy, the operation will fail and raise a
**ValueError**. NumPy uses broadcasting for operations between arrays of different shapes, but this is only possible if the shapes are compatible according to specific rules.
Broadcasting works by aligning the dimensions of the arrays starting from the rightmost dimension and working backward. For two dimensions to be compatible, they must either be equal or one of them must be 1 (in which case it is broadcasted to match the other dimension).

If the shapes of the arrays do not meet these criteria, broadcasting cannot occur, and the operation results in an error.

### Example

In this case, the shapes of arrays "a" and "b" are not compatible for broadcasting, resulting in an error −

```
import numpy as np

# Creating arrays with incompatible shapes
a = np.array([1, 2, 3])
b = np.array([[10, 20], [30, 40]])

# Attempting to add incompatible arrays
result = a + b
print(result)
```

The result produced is as follows −

```
Traceback (most recent call last):File "/home/cg/root/66a1de2fae52f/main.py", line 8, in <module>result = a + bValueError: operands could not be broadcast together with shapes (3,) (2,2)
```

---

## 31. NumPy - Array Subtraction

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_subtraction.htm](https://www.tutorialspoint.com/numpy/numpy_array_subtraction.htm)*

---

---
[Previous](/numpy/numpy_array_addition.htm)[Quiz](/numpy/quiz_on_numpy_array_subtraction.htm)[Next](/numpy/numpy_array_multiplication.htm)
## NumPy Array Subtraction

NumPy array subtraction allows you to perform element-wise subtraction between arrays. This operation subtracts corresponding elements of one array from another array of the same shape, producing a new array of the same shape with the subtracted values.

If the arrays have different shapes, NumPy can broadcast the smaller array to match the shape of the larger array under certain conditions.

## Element-wise Subtraction in NumPy

Element-wise subtraction is the most basic form of array subtraction in NumPy, where corresponding elements of two arrays are subtracted to produce a new array.

This type of subtraction operates on arrays of the same shape, performing the subtraction operation individually for each pair of elements from the two arrays.

### Example

In the following example, we are subtracting each element of array
**a**from the corresponding element of array**a**−
```
import numpy as np

# Creating two arrays
a = np.array([5, 6, 7])
b = np.array([1, 2, 3])

# Performing element-wise subtraction
result = a - b
print(result)
```

Following is the output obtained −

```
[4 4 4]
```

## Subtracting a Scalar to a NumPy Array

When a scalar (a single value) is subtracted from an array, the scalar is broadcasted to match the shape of the array. This means that the scalar is effectively treated as if it were an array of the same shape as the original array, with all elements equal to the scalar value.

> Broadcasting explains how NumPy manages arithmetic operations involving arrays of different shapes. When arrays with varying shapes are used in calculations, NumPy automatically adjusts their shapes to be compatible with each other according to the broadcasting rules.

### Example

In this example, we are subtracting the scalar "10" from each element of the array "a" −

```
import numpy as np

# Creating an array
a = np.array([5, 6, 7])

# Subtracting a scalar
result = a - 2
print(result)
```

This will produce the following result −

```
[3 4 5]
```

## Subtracting NumPy Arrays of Different Shapes

Broadcasting in NumPy allows for the subtraction of arrays with different shapes by adjusting their dimensions to match each other.

NumPy aligns dimensions for broadcasting by comparing from the rightmost side and moving leftward. Dimensions are compatible if they are equal or if one dimension is 1, which is then expanded to match the other dimension.

When dimensions do not align directly, NumPy extends the smaller array along the mismatched dimensions as needed to fit the shape of the larger array.

### Example

In the example below, array "b" is broadcasted to match the shape of array "a", and then element-wise subtraction is performed −

```
import numpy as np

# Creating arrays with different shapes
a = np.array([[5, 6, 7], [8, 9, 10]])
b = np.array([1, 2, 3])

# Subtracting arrays with broadcasting
result = a - b
print(result)
```

Following is the output of the above code −

```
[[4 4 4]
 [7 7 7]]
```

## Subtracting Multi-Dimensional Arrays with Broadcasting

In NumPy, broadcasting allows for arithmetic operations, such as subtraction, between multi-dimensional arrays of different shapes by automatically expanding the dimensions of the smaller array to match the shape of the larger array.

### Example

In the example below, we broadcast the one-dimensional array "b" to match the dimensions of the two-dimensional array "a" −

```
import numpy as np

# Creating multi-dimensional arrays
a = np.array([[10, 20, 30], [40, 50, 60]])
b = np.array([5, 15, 25])

# Subtracting multi-dimensional arrays with broadcasting
result = a - b[np.newaxis, :]
print(result)
```

The output obtained is as shown below −

```
[[ 5 15 25]
 [25 35 45]]
```

## Subtracting By Applying Functions with Broadcasting

Broadcasting in NumPy not only allows for direct element-wise arithmetic operations but also facilitates applying functions to arrays with different shapes. With broadcasting, you can use various mathematical functions on arrays of different shapes.

### Example

In this example, we are subtracting the scalar "5" from each element of the array "a", and then apply the "sine" function element-wise −

```
import numpy as np

# Creating an array
a = np.array([10, 20, 30])

# Applying a function with broadcasting
result = np.sin(a - 5)
print(result)
```

After executing the above code, we get the following output −

```
[-0.95892427 -0.7568025   0.14112001]
```

## Subtracting Incompatible Arrays

If we attempt to subtract incompatible arrays in NumPy, the operation will fail and raise a
**ValueError**. NumPy uses broadcasting for operations between arrays of different shapes, but this is only possible if the shapes are compatible according to specific rules.
### Example

In this case, the shapes of arrays "a" and "b" are not compatible for broadcasting, resulting in an error −

```
import numpy as np

# Creating arrays with incompatible shapes
a = np.array([10, 20, 30])
b = np.array([[1, 2], [3, 4]])

# Subtracting incompatible arrays
result = a - b
print(result)
```

The result produced is as follows −

```
Traceback (most recent call last):File "/home/cg/root/66a1de2fae52f/main.py", line 8, in <module>result = a - bValueError: operands could not be broadcast together with shapes (3,) (2,2)
```

---

## 32. NumPy - Array Multiplication

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_multiplication.htm](https://www.tutorialspoint.com/numpy/numpy_array_multiplication.htm)*

---

---
[Previous](/numpy/numpy_array_subtraction.htm)[Quiz](/numpy/quiz_on_numpy_array_multiplication.htm)[Next](/numpy/numpy_array_division.htm)
## NumPy Array Multiplication

NumPy array multiplication refers to the process of multiplying two arrays element-wise. In this context, element-wise multiplication means that each element in one array is multiplied by the corresponding element in the other array.

The result is a new array where each element represents the product of the corresponding elements from the input arrays.

## Element-wise Multiplication in NumPy

Element-wise multiplication, also known as Hadamard product, is an operation in NumPy where two arrays of the same shape are multiplied together, and the operation is applied to each corresponding pair of elements.

For element-wise multiplication to be performed, the two arrays must have the same shape. If the arrays are of different shapes, broadcasting rules are applied to make them compatible.

### Example

In the following example, we are multiplying each element of array
**a**by the corresponding element of array**b**−
```
import numpy as np

# Creating two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Performing element-wise multiplication
result = a * b
print(result)
```

Following is the output obtained −

```
[ 4 10 18]
```

## Multiplying by a Scalar in NumPy

Scalar multiplication refers to multiplying every element of an array by a single scalar value. Each element in the array is multiplied by the scalar value, resulting in a new array where all elements are scaled accordingly.

The shape of the resulting array is the same as the original array, but each element is scaled by the scalar.

### Example

In this example, we are multiplying the scalar "3" with each element of the array "a" −

```
import numpy as np

# Creating an array
a = np.array([1, 2, 3])

# Multiplying by a scalar
result = a * 3
print(result)
```

This will produce the following result −

```
[3 6 9]
```

## Multiplying NumPy Arrays of Different Shapes

When multiplying arrays of different shapes, NumPy uses broadcasting to make the shapes compatible for element-wise operations. Broadcasting aligns dimensions from the rightmost side, adjusts shapes according to specific rules, and performs the operation smoothly.

### Example

In the example below, array "b" is broadcasted to match the shape of array "a", and then element-wise multiplication is performed −

```
import numpy as np

# Creating arrays with different shapes
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])

# Multiplying arrays with broadcasting
result = a * b
print(result)
```

Following is the output of the above code −

```
[[10 40 90]
 [40 100 180]]
```

## Matrix Multiplication in NumPy

Matrix multiplication is an operation in linear algebra where two matrices are multiplied according to specific rules. In NumPy, this operation is performed using the
**np.dot()**function, the**@**operator, or the**np.matmul()**function.
For matrix multiplication to be valid, the number of columns in the first matrix must be equal to the number of rows in the second matrix. If matrix
**A**has shape**(m, n)**and matrix**B**has shape**(n, p)**, the resulting matrix will have shape**(m, p)**.
### Example: Using numpy.dot() function

The np.dot() function in NumPy is used for calculating dot products of two arrays. It handles both matrix multiplication for 2D arrays and dot products for 1D arrays −

```
import numpy as np

# Define two matrices
matrix1 = np.array([[1, 2],
                    [3, 4]])

matrix2 = np.array([[5, 6],
                    [7, 8]])

# Perform matrix multiplication using np.dot
result = np.dot(matrix1, matrix2)
print(result)
```

The output obtained is as shown below −

```
[[19 22]
 [43 50]]
```

### Example: Using the "@" Operator

The
**@**operator in NumPy provides a shorthand for the np.dot() function. It performs matrix multiplication for 2D arrays and is used to calculate the dot product for 1D arrays −
```
import numpy as np

# Define the same matrices
matrix1 = np.array([[1, 2],
                    [3, 4]])

matrix2 = np.array([[5, 6],
                    [7, 8]])

# Perform matrix multiplication using the @ operator
result = matrix1 @ matrix2
print(result)
```

After executing the above code, we get the following output −

```
[[19 22]
 [43 50]]
```

### Example: Using the np.matmul() Function

The
**np.matmul()**function can handle arrays with more than two dimensions by performing matrix multiplication on the last two dimensions of the input arrays, with any preceding dimensions being broadcasted as needed −
```
import numpy as np

# Define a 3D array and a 2D array
array_3d = np.array([[[1, 2],
                      [3, 4]],

                     [[5, 6],
                      [7, 8]]])

array_2d = np.array([[1, 0],
                     [0, 1]])

# Perform matrix multiplication using np.matmul
result = np.matmul(array_3d, array_2d)
print(result)
```

The result produced is as follows −

```
[[[1 2]
  [3 4]]

  [[5 6]
  [7 8]]]
```

## Element-wise vs. Matrix Multiplication

It is important to differentiate between element-wise multiplication (using *) and matrix multiplication (using @ or np.dot()) −

- **Element-wise Multiplication −**Each element in the first array is multiplied by the corresponding element in the second array.
- **Matrix Multiplication −**Performs a dot product of rows and columns, following linear algebra rules.
### Example

In this example, we are highlighting the difference between element-wise multiplication and matrix multiplication −

```
import numpy as np

# Creating two 2D arrays
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
element_wise_result = a * b
print("Element-wise multiplication:\n", element_wise_result)

# Matrix multiplication
matrix_result = np.dot(a, b)
print("Matrix multiplication:\n", matrix_result)
```

We get the output as shown below −

```
Element-wise multiplication:
 [[ 5 12]
 [21 32]]
Matrix multiplication:
 [[19 22]
 [43 50]]
```

## Multiplying Arrays with Different Data Types

When multiplying arrays with different data types in NumPy,
**type coercion**rules are applied to ensure that the operation can be performed. This process involves converting the arrays to a common data type before performing the multiplication.
NumPy uses a set of promotion rules to determine the data type of the result. The general rule is to convert the operands to the type that can hold the result without losing precision. Following i the promotion order −

```
Boolean → Integer → Floating-point → Complex.
```

For example, integer and floating-point numbers are promoted to floating-point types to accommodate decimal values.

### Example

In this case, NumPy promotes the integer array
**b**to a float type to match the data type of array**a**−
```
import numpy as np

# Creating arrays with different data types
a = np.array([1.5, 2.5, 3.5])
b = np.array([2, 3, 4], dtype=np.int32)

# Performing multiplication
result = a * b
print(result)
```

Following is the output obtained −

```
[3.  7.5 14.]
```

## Handling Dimension Mismatch in Multiplication

When performing multiplication operations between arrays in NumPy, dimension mismatch can occur if the arrays do not share compatible shapes. NumPy addresses this issue through broadcasting and raises a
**ValueError**in such case.
### Example

In the following example, the shapes of "a" and "b" are not compatible for broadcasting, leading to an error −

```
import numpy as np

# Creating arrays with incompatible shapes
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

# Attempting to multiply incompatible arrays
result = a * b
print(result)
```

The result produced is as follows −

```
Traceback (most recent call last):File "/home/cg/root/66a1de2fae52f/main.py", line 8, in <module>result = a * bValueError: operands could not be broadcast together with shapes (3,) (2,2)
```

---

## 33. NumPy - Array Division

*Source: [https://www.tutorialspoint.com/numpy/numpy_array_division.htm](https://www.tutorialspoint.com/numpy/numpy_array_division.htm)*

---

---
[Previous](/numpy/numpy_array_multiplication.htm)[Quiz](/numpy/quiz_on_numpy_array_division.htm)[Next](/numpy/numpy_swapping_axes_of_arrays.htm)
## NumPy Array Division

NumPy array division refers to the process of dividing two arrays element-wise. In this context, element-wise division means that each element in one array is divided by the corresponding element in another array.

This operation supports broadcasting, allowing for division between arrays of different shapes. Additionally, division can be performed between an array and a scalar, which scales each element of the array accordingly.

## Element-wise Division in NumPy

Element-wise division in NumPy refers to dividing each element of one array by the corresponding element in another array. This operation is performed element by element, meaning that the division occurs pairwise between the elements of the two arrays at the same positions.

For element-wise division to work, the two arrays must have the same shape, or they must be broadcastable to a common shape.

In NumPy, the division operator
**/**is used to perform element-wise division. If the arrays have different shapes but are compatible for broadcasting, NumPy will automatically apply broadcasting rules to perform the division.
### Example

In the following example, we are dividing each element of array
**a**by the corresponding element of array**b**−
```
import numpy as np

# Creating two arrays
a = np.array([10, 20, 30])
b = np.array([2, 5, 10])

# Performing element-wise division
result = a / b
print(result)
```

Following is the output obtained −

```
[5. 4. 3.]
```

## Division by a Scalar in NumPy

Division by a scalar in NumPy refers to dividing each element of an array by a single scalar value. This operation applies the division to every element of the array, resulting in a new array where each element has been divided by the scalar.

### Example

In this example, we are dividing each element of the array "a" with the scalar "10" −

```
import numpy as np

# Creating an array
a = np.array([10, 20, 30])

# Dividing by a scalar
result = a / 10
print(result)
```

This will produce the following result −

```
[1. 2. 3.]
```

## Broadcasting in Array Division

Broadcasting allows for operations between arrays of different shapes. When performing division between arrays of different shapes, NumPy automatically adjusts the shapes so that the operation can be performed.

When performing array division, broadcasting allows you to divide arrays of different sizes or shapes by automatically expanding the smaller array along the missing dimensions to match the shape of the larger array.

This expansion is done in a way that allows element-wise operations, like division, to be performed without explicitly replicating the data.

### Example

In the example below, array "b" is broadcasted to match the shape of array "a", and then element-wise division is performed −

```
import numpy as np

# Creating arrays with different shapes
a = np.array([[10, 20, 30], [40, 50, 60]])
b = np.array([10, 5, 2])

# Performing division with broadcasting
result = a / b
print(result)
```

Following is the output of the above code −

```
[[ 1.  4. 15.]
 [ 4. 10. 30.]]
```

## Handling Division by Zero

When performing division operations in NumPy, one common issue is the possibility of dividing by zero, which can lead to undefined or infinite results. NumPy provides ways to handle such scenarios, either by returning special values (like inf or nan) or by raising warnings or errors depending on the specific use case.

- **Positive/Negative Infinity (inf or -inf) −**It is returned when a positive or negative number is divided by zero.
- **Not a Number (nan) −**It is returned when zero is divided by zero.
### Example: Basic Division by Zero

Let us see how NumPy handles a simple division by zero −

```
import numpy as np

# Creating an array with a zero element
array = np.array([10, 0, -5])

# Dividing a constant by the array
result = 100 / array
print(result)
```

NumPy does not raise an error but instead returns "inf" for division by zero −

```
/home/cg/root/66c57b63ef7f5/main.py:7: RuntimeWarning: divide by zero encountered in divide
  result = 100 / array
[ 10.  inf -20.]
```

### Example: Handling Division by Zero Using errstate() Function

You can control how NumPy handles division by zero using np.errstate() function. This function allows you to specify whether to ignore, warn, or raise an error for floating-point errors like division by zero −

```
import numpy as np

# Creating an array with a zero element
array = np.array([10, 0, -5])

with np.errstate(divide='ignore', invalid='ignore'):
   result = 100 / array
   print(result)
```

Here, we set the "divide" parameter to "ignore" in the errstate() function to ignore the warning that generally occurs with division by zero. The result still contains "inf", but without any interruption or warning −

```
/home/cg/root/66c57b63ef7f5/main.py:7: RuntimeWarning: divide by zero encountered in divide
  result = 100 / array
[ 10.  inf -20.]
```

### Example: Handling nan Values

When zero is divided by zero, the result is nan (Not a Number). This special value is used to represent undefined or unrepresentable values in floating-point calculations −

```
import numpy as np

# Creating arrays with zero elements
numerator = np.array([0, 1, 2])
denominator = np.array([0, 0, 2])

# Performing division
result = numerator / denominator
print(result)
```

The result produced is as follows −

```
[ 10.  inf -20.]
```

## Division with Different Data Types

When performing division between arrays of different data types, NumPy automatically promotes the data types to a common type that can safely contain the result.

For example, if you divide an integer array by a float array, the result will be a float array. This type promotion prevents data loss and ensures the precision of the results.

### Example

In this example, NumPy promotes the integer array
**a**to a float type to match the data type of array**b**before performing the division −
```
import numpy as np

# Creating arrays with different data types
a = np.array([10, 20, 30], dtype=np.int32)
b = np.array([2.5, 5.0, 10.0], dtype=np.float64)

# Performing division
result = a / b
print(result)
```

We get the output as shown below −

```
[4. 4. 3.]
```

## Matrix Division in NumPy

Matrix division is not as straightforward as element-wise division or scalar division. However, you can solve for a matrix division-like operation by using matrix multiplication with the inverse of a matrix in NumPy.

### Example

In this example, we use the inverse of matrix
**B**to perform a division-like operation in NumPy −
```
import numpy as np

# Creating a matrix
A = np.array([[1, 2], [3, 4]])

# Creating another matrix
B = np.array([[2, 0], [1, 3]])

# Solving the matrix division A/B
# Equivalent to A * B^-1
result = np.dot(A, np.linalg.inv(B))
print(result)
```

Following is the output obtained −

```
[[0.16666667 0.66666667]
 [0.83333333 1.33333333]]
```

## Handling Dimension Mismatch in Division

When performing division operations between arrays in NumPy, dimension mismatch can occur if the arrays do not share compatible shapes. NumPy addresses this issue through broadcasting and raises a
**ValueError**in such case.
### Example

In the following example, the shapes of "a" and "b" are not compatible for broadcasting, resulting in an error −

```
import numpy as np

# Creating arrays with incompatible shapes
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

# Attempting to divide incompatible arrays
result = a / b
print(result)
```

The result produced is as follows −

```
Traceback (most recent call last):
File "/home/cg/root/66c57b63ef7f5/main.py", line 8, in <module>
   result = a / b
ValueError: operands could not be broadcast together with shapes (3,) (2,2)
```

---

## 34. NumPy - Swapping Axes of Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_swapping_axes_of_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_swapping_axes_of_arrays.htm)*

---

---

## 35. NumPy - Byte Swapping

*Source: [https://www.tutorialspoint.com/numpy/numpy_byte_swapping.htm](https://www.tutorialspoint.com/numpy/numpy_byte_swapping.htm)*

---

---
[Previous](/numpy/numpy_swapping_axes_of_arrays.htm)[Quiz](/numpy/quiz_on_numpy_byte_swapping.htm)[Next](/numpy/numpy_copies_and_views.htm)
## Swapping Axes of Arrays in NumPy

Byte swapping is a process used to convert data between different byte orders, also known as
**endianness**. In computing, different systems might use different byte orders to represent multi-byte data types (e.g., integers, floats). Byte swapping ensures that data is interpreted correctly when transferred between systems with different**endianness**.
NumPy provides the
**byteswap()**function to swap the bytes of an array. This is particularly useful when you need to convert data to the correct endianness for compatibility with other systems or formats.
## Understanding Byte Order

Byte Order (Endianness) is the sequence in which bytes are ordered within larger data types. There are two primary byte orders −

- **Little-Endian −**The least significant byte is stored at the smallest address. For example, in the number 0x1234, 0x34 would be stored first.
- **Big-Endian −**The most significant byte is stored at the smallest address. For the same number 0x1234, 0x12 would be stored first.
## The numpy.ndarray.byteswap() Function

The
**numpy.ndarray.byteswap()**function is used to swap the byte order of the elements in a NumPy array. This function toggles between the two representations: bigendian and little-endian.
The byteswap() function is used on arrays with specific data types and does not affect the shape or size of the array. Following is the syntax −

```
numpy.ndarray.byteswap(inplace=False)
```

Where, if
**inplace**is "True", the array is modified in place. If "False" (default), a new array with swapped bytes is returned.
### Example: Swapping Bytes in a Simple Array

In the following example, we are swapping bytes in an array using the byteswap() function in NumPy −

```
import numpy as np 
a = np.array([1, 256, 8755], dtype = np.int16) 

print ('Our array is:', a)

print ('Representation of data in memory in hexadecimal form:', map(hex,a))
# We can see the bytes being swapped
print ('Applying byteswap() function:', a.byteswap())
print ('In hexadecimal form:', map(hex,a))
```

Following is the output obtained −

```
Our array is: [   1  256 8755]
Representation of data in memory in hexadecimal form: <map object at 0x7fdfa46a3370>
Applying byteswap() function: [  256     1 13090]
In hexadecimal form: <map object at 0x7fdff5867190>
```

### Example: Byte Swapping In-Place

We can modify the array in place by setting the "inplace" parameter to "True" in the byteswap() function, swapping the bytes directly within the original array −

```
import numpy as np

# Creating a NumPy array with 32-bit integers
arr = np.array([1, 256, 65535], dtype=np.int32)

print("Original Array:")
print(arr)

# Perform in-place byte swapping
arr.byteswap()

print("\nArray After In-Place Byte Swapping:")
print(arr)
```

The result produced is as follows −

```
Original Array:
[    1   256 65535]

Array After In-Place Byte Swapping:
[    1   256 65535]
```

## When to Use Byte Swapping

We can use Byte swapping in the following scenarios −

- **Interoperability −**When data is exchanged between systems with different endianness, byte swapping ensures correct interpretation.
- **Data Reading/Writing −**When dealing with raw binary files or network protocols that use different byte orders, byte swapping is necessary to correctly read or write data.
- **Legacy Systems −**Working with legacy systems or file formats that use specific byte orders might require byte swapping for correct data handling.

---

## 36. NumPy - Copies & Views

*Source: [https://www.tutorialspoint.com/numpy/numpy_copies_and_views.htm](https://www.tutorialspoint.com/numpy/numpy_copies_and_views.htm)*

---

---
[Previous](/numpy/numpy_byte_swapping.htm)[Quiz](/numpy/quiz_on_numpy_copies_and_views.htm)[Next](/numpy/numpy_element_wise_array_comparisons.htm)
In NumPy, when you perform operations on arrays, the result might be a
**copy**of the original data or just a**view**of the original data. Understanding the difference between these two is important for efficient memory management and avoiding unintended side effects in your code.
## Creating Copies in NumPy

We can create a copy of an array explicitly in NumPy using the copy() function. This function generates a new array and copies the data from the original array into this new array.

When you create a copy of an array in NumPy, the data is fully duplicated. This means that changes made to the copy do not affect the original array, and vice versa. Copies are useful when you need to work with a modified version of an array without altering the original data.

### Example

In the following example, modifying
**copied_array**does not affect**original_array**, demonstrating the independence of the two arrays −
```
import numpy as np

# Original array
original_array = np.array([1, 2, 3])

# Creating a copy
copied_array = original_array.copy()

# Modifying the copy
copied_array[0] = 10

print("Original Array:", original_array)  
print("Copied Array:", copied_array)
```

Following is the output obtained −

```
Original Array: [1 2 3]
Copied Array: [10  2  3]
```

## Shallow Copy Vs. Deep Copy

In the context of NumPy arrays, the difference between shallow and deep copies is important for understanding how data is handled when copied.

### Shallow Copy

A shallow copy of an array creates a new array object, but it does not create copies of the elements contained within the original array if those elements themselves are arrays or other complex objects.

Instead, the new array still references the same elements as the original array. This means that changes to the contents of the elements will affect both the original and the copied array.

- **Array-Level Copy −**In the case of NumPy arrays, a shallow copy means that while the top-level array object is duplicated, the underlying data buffer is not copied. The new array is simply a new view of the same data.
- **Usage −**Shallow copies are useful when you need a new array object but want to avoid the overhead of duplicating large amounts of data.**Example**
In this example, modifying
**shallow_copy**also modifies**original_array**because they share the same underlying data −
```
import numpy as np

# Original array
original_array = np.array([[1, 2, 3], [4, 5, 6]])

# Shallow copy
shallow_copy = original_array.view()

# Modify an element in the shallow copy
shallow_copy[0, 0] = 100

print("Original Array:")
print(original_array)

print("\nShallow Copy:")
print(shallow_copy)
```

This will produce the following result −

```
Original Array:
[[100   2   3]
 [  4   5   6]]

Shallow Copy:
[[100   2   3]
 [  4   5   6]]
```

### Deep Copy

A deep copy, on the other hand, creates a new array object along with copies of all the data it contains. This means that any changes made to the new array will not affect the original array, and vice versa. The data in the new array is completely independent of the data in the original array.

- **Full Duplication −**In the context of NumPy, a deep copy involves duplicating the entire data buffer of the array, ensuring that the new array is entirely separate from the original.
- **Usage −**Deep copies are important when you need to work with data independently of the original array, especially when the data may be modified in a way that should not impact the original.**Example**
In this case, modifying
**deep_copy**does not affect**original_array**demonstrating the independence of the two arrays −
```
import numpy as np

# Original array
original_array = np.array([[1, 2, 3], [4, 5, 6]])

# Deep copy
deep_copy = original_array.copy()

# Modify an element in the deep copy
deep_copy[0, 0] = 100

print("Original Array:")
print(original_array)

print("\nDeep Copy:")
print(deep_copy)
```

Following is the output of the above code −

```
Original Array:
[[1 2 3]
[4 5 6]]

Deep Copy:
[[100   2   3]
 [  4   5   6]]
```

## Copying Subarrays

To avoid modifying the original array when working with a subarray, you should create a copy of the subarray. This is useful when you need to manipulate or analyze the subarray independently of the original data.

A subarray is simply a portion of an existing NumPy array. You can extract subarrays using slicing techniques.

For example, if you have a 2D array, you can extract a smaller 2D subarray by slicing along its rows and columns. However, by default, slicing creates a view of the original array, not a separate copy. This means that changes to the subarray will also affect the original array unless you explicitly create a copy.

### Example

In the example below,
**sub_array**is a completely independent array due to the use of copy() function  −
```
import numpy as np

# Original 2D array
original_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Creating a copy of the subarray
sub_array = original_array[0:2].copy()
sub_array[0] = 20

print("Original Array after subarray copy:", original_array)  
print("Subarray:", sub_array)
```

The output obtained is as shown below −

```
Original Array after subarray copy: 
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Subarray: 
[[20 20 20]
 [ 4  5  6]]
```

## Creating Views in NumPy

Views are created when you slice an array or perform certain operations like reshaping. The data is not copied; instead, the new array is just a different way of viewing the original data.

In other words, a
**view**is a new array object that looks at the same data as the original array. This means that if you modify the view, the changes will be reflected in the original array, and vice versa.
### Example

In this example, modifying
**view_array**directly affects**original_array**, showing that they share the same data −
```
import numpy as np

# Original array
original_array = np.array([1, 2, 3])

view_array = original_array[0:2]

# Modifying the view
view_array[0] = 30

print("Original Array after view modification:", original_array) 
print("View Array:", view_array)
```

After executing the above code, we get the following output −

```
Original Array after view modification: [30  2  3]
View Array: [30  2]
```

## When Views are Returned?

Not all slicing or operations result in a view. If the memory layout of the array changes, NumPy might return a copy instead of a view.

### Views from Slicing

The most common scenario where views are returned in NumPy is when you slice an array. Slicing is a way to extract a portion of an array by specifying a range of indices. Instead of creating a new array with its own data, NumPy returns a view, meaning the sliced array shares the same data as the original array.
**Example**
In this example,
**view_array**is a view of**original_array**. The data is not copied, and both arrays share the same underlying memory. This means that any changes made to "view_array" will also affect "original_array" −
```
import numpy as np

# Original array
original_array = np.array([1, 2, 3, 4, 5])

# Creating a view by slicing the original array
view_array = original_array[1:4]

print("Original Array:")
print(original_array)

print("\nView Array (Sliced):")
print(view_array)
```

The result produced is as follows −

```
Original Array:
[1 2 3 4 5]

View Array (Sliced):
[2 3 4]
```

### Views from Reshaping

Another common scenario where views are returned is when you reshape an array. Reshaping changes the shape of the array (i.e., the number of elements in each dimension) without altering the underlying data. When possible, NumPy returns a view of the original array in the new shape.
**Example**
Here,
**reshaped_array**is a view of**original_array**, simply presented in a "2x3" format. The data remains the same, and modifying the "reshaped_array" will also modify the "original_array" −
```
import numpy as np

# Original 1D array
original_array = np.array([1, 2, 3, 4, 5, 6])

# Reshaping the array into a 2x3 matrix
reshaped_array = original_array.reshape(2, 3)

print("Original Array:")
print(original_array)

print("\nReshaped Array (View):")
print(reshaped_array)
```

We get the output as shown below −

```
Original Array:[1 2 3 4 5 6]
Reshaped Array (View):
[[1 2 3]
 [4 5 6]]
```

### Views from Transposing

Transposing an array involves flipping it over its diagonal, converting rows to columns and vice versa. When you transpose an array using functions like
**np.transpose()**or the**.T**attribute, NumPy returns a view, not a copy, whenever possible.**Example**
In this case,
**transposed_array**is a view of**original_array**, but with the axes swapped. The underlying data remains the same, and changes to "transposed_array" will reflect in "original_array" −
```
import numpy as np

# Original 2D array
original_array = np.array([[1, 2, 3], [4, 5, 6]])

# Transposing the array
transposed_array = original_array.T

print("Original Array:")
print(original_array)

print("\nTransposed Array (View):")
print(transposed_array)
```

Following is the output obtained −

```
Original Array:
[[1 2 3]
 [4 5 6]]

Transposed Array (View):
[[1 4]
 [2 5]
 [3 6]]
```

## The Base Attribute

In NumPy, the
**base**attribute of an array examines whether the array is a view or a copy of another array. It is a reference to the original array from which the current array was derived.
If the current array is a view of another array, "base" will point to that original array. If the current array is not a view (i.e., it is either the original array or a deep copy), "base" will be
**None**.
### Example: Base Attribute of an Original Array

When you create an array, its base attribute will be None because it is the original array −

```
import numpy as np

# Creating an original array
original_array = np.array([10, 20, 30, 40, 50])

# Checking the base attribute
print("Base of original array:", original_array.base)
```

This will produce the following result −

```
Base of original array: None
```

### Example: Base Attribute of a View

When you create a view of an array (for example, by slicing), the base attribute of the view will point to the original array −

```
import numpy as np

# Creating an original array
original_array = np.array([10, 20, 30, 40, 50])

# Creating a view of the original array
view_array = original_array[1:4]

# Checking the base attribute
print("Base of view array:", view_array.base)
```

Following is the output of the above code −

```
Base of view array: [10 20 30 40 50]
```

### Example: Base Attribute of a Copy

If you create a copy of an array, the base attribute will be None, indicating that the copied array is independent of the original −

```
import numpy as np

# Creating an original array
original_array = np.array([10, 20, 30, 40, 50])

# Creating a copy of the original array
copy_array = original_array.copy()

# Checking the base attribute
print("Base of copy array:", copy_array.base)
```

Following is the output of the above code −

```
Base of copy array: None
```

---

## 37. NumPy - Element-wise Array Comparisons

*Source: [https://www.tutorialspoint.com/numpy/numpy_element_wise_array_comparisons.htm](https://www.tutorialspoint.com/numpy/numpy_element_wise_array_comparisons.htm)*

---

---
[Previous](/numpy/numpy_copies_and_views.htm)[Quiz](/numpy/quiz_on_numpy_element_wise_array_comparisons.htm)[Next](/numpy/numpy_filtering_arrays.htm)
## Element-wise Comparisons in NumPy

Element-wise comparisons in NumPy allow you to compare each element of one array with the corresponding element of another array or a scalar value.

The comparison is performed across the entire array, and the result is a new array of the same shape where each element is a Boolean (True or False) indicating the outcome of the comparison.

## Basic Element-wise Comparison Operations

NumPy supports several basic comparison operations that can be performed element-wise. These include −

- **Equality (==):**Checks if elements in the two arrays (or an array and a scalar) are equal.
- **Inequality (!=):**Checks if elements are not equal.
- **Greater than (>):**Checks if elements in the first array are greater than the corresponding elements in the second array or a scalar.
- **Less than (<):**Checks if elements in the first array are less than the corresponding elements in the second array or a scalar.
- **Greater than or equal to (>=):**Checks if elements are greater than or equal to the corresponding elements in the second array or a scalar.
- **Less than or equal to (<=):**Checks if elements are less than or equal to the corresponding elements in the second array or a scalar.
### Example

In the following example, each comparison operation is performed between corresponding elements of "array1" and "array2" −

```
import numpy as np

# Creating two arrays for comparison
array1 = np.array([10, 20, 30, 40, 50])
array2 = np.array([15, 20, 25, 40, 55])

# Performing element-wise comparisons
equality = array1 == array2
inequality = array1 != array2
greater_than = array1 > array2
less_than = array1 < array2
greater_equal = array1 >= array2
less_equal = array1 <= array2

# Displaying the results
print("Equality:", equality)
print("Inequality:", inequality)
print("Greater than:", greater_than)
print("Less than:", less_than)
print("Greater than or equal to:", greater_equal)
print("Less than or equal to:", less_equal)
```

The result is a Boolean array indicating the outcome of each comparison as shown below −

```
Equality: [False  True False  True False]
Inequality: [ True False  True False  True]
Greater than: [False False  True False False]
Less than: [ True False False False  True]
Greater than or equal to: [False  True  True  True False]
Less than or equal to: [ True  True False  True  True]
```

## Element-wise Comparisons with Scalars

You can also compare an entire array with a single scalar value. The scalar value is compared to each element of the array, and the result is a Boolean array of the same shape.

### Example

In this example, each element of "array1" is compared to "30", and the result indicates whether each element is greater than "30" −

```
import numpy as np

# Creating two arrays for comparison
array1 = np.array([10, 20, 30, 40, 50])

# Comparing array elements with a scalar value
scalar_value = 30
comparison_result = array1 > scalar_value

print("Array elements greater than 30:", comparison_result)
```

This will produce the following result −

```
Array elements greater than 30: [False False False  True  True]
```

## Chaining Multiple Comparisons

Chaining multiple comparisons in NumPy involves using logical operators to combine several comparison operations. For instance, you might want to check if the elements of an array fall within a specific range or if they satisfy multiple criteria.

The operations are evaluated in sequence, and the result is a Boolean array where each element indicates whether the combined conditions are met.

In NumPy, you can chain comparisons using logical operators like
**&**(and),**|**(or), and**~**(not). When chaining comparisons, ensure that each comparison operation is enclosed in parentheses to maintain the correct order of operations. Here is the general syntax for chaining comparisons −
```
(condition1) & (condition2) & ... & (conditionN)
```

### Example: Chaining Comparisons

In the example below, we are checking if the elements of an array are within a specific range and satisfy additional conditions or not −

```
import numpy as np

# Creating an array
array = np.array([5, 10, 15, 20, 25, 30])

# Chaining multiple comparisons
result = (array > 10) & (array < 25) & (array % 5 == 0)

# Displaying the results
print("Array:", array)
print("Result of Chained Comparisons:", result)
```

Following is the output of the above code −

```
Array: [ 5 10 15 20 25 30]
Result of Chained Comparisons: [False False  True  True False False]
```

### Example: Chaining with Scalar Values

Here, the comparison checks if each element of the array is between "3" and "7", inclusive −

```
import numpy as np

# Creating an array
array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Chaining comparisons with scalar values
result = (array >= 3) & (array <= 7)

# Displaying the results
print("Array:", array)
print("Result of Chained Comparisons with Scalar:", result)
```

The output obtained is as shown below −

```
Array: [1 2 3 4 5 6 7 8 9]
Result of Chained Comparisons with Scalar: [False False  True  True  True  True  True False False]
```

## Using where() Function for Conditional Selection

The np.where() function uses the results of element-wise comparisons to selectively choose elements from one of two arrays (or values). This is particularly useful for filtering or replacing elements based on a condition.

### Example

In this example, elements of "array1" greater than "25" are kept, while all others are replaced with "0" −

```
import numpy as np

# Creating an array
array1 = np.array([10, 20, 30, 40, 50])

# Using np.where to replace elements based on a condition
replaced_array = np.where(array1 > 25, array1, 0)

print("Replaced array:", replaced_array)
```

After executing the above code, we get the following output −

```
Replaced array: [ 0  0 30 40 50]
```

## Finding Max and Min Elements with Comparisons

Element-wise comparisons can be used in conjunction with functions like
**np.maximum()**and**np.minimum()**to find the maximum or minimum values between two arrays.
### Example

In this example, we use np.maximum() function and np.minimum() function to compare elements of "array1" and "array2", returning arrays of the maximum and minimum values respectively −

```
import numpy as np

# Creating an array
array1 = np.array([10, 20, 30, 40, 50])
array2 = np.array([15, 20, 25, 40, 55])

# Finding maximum and minimum values between two arrays
max_array = np.maximum(array1, array2)
min_array = np.minimum(array1, array2)

print("Maximum values:", max_array)
print("Minimum values:", min_array)
```

The result produced is as follows −

```
Maximum values: [15 20 30 40 55]
Minimum values: [10 20 25 40 50]
```

---

## 38. NumPy - Filtering Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_filtering_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_filtering_arrays.htm)*

---

---

## 39. NumPy - Joining Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_joining_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_joining_arrays.htm)*

---

---
[Previous](/numpy/numpy_filtering_arrays.htm)[Quiz](/numpy/quiz_on_numpy_joining_arrays.htm)[Next](/numpy/numpy_sort_search_counting_functions.htm)
## Joining Arrays in NumPy

Joining arrays in NumPy refers to the process of combining two or more arrays into a single array. The result may vary depending on the dimensions and axes along which the arrays are joined.

NumPy provides several functions for joining of arrays along different axes, they are −

- The np.concatenate() Function
- The np.stack() Function
- The np.hstack() Function
- The np.vstack() Function
## Using concatenate() Function

The NumPy concatenate() function joins a sequence of arrays along a specified axis. The arrays must have the same shape, except in the dimension corresponding to the axis along which they are being concatenated. Following is the basic syntax −

```
np.concatenate((array1, array2, ...), axis=0)
```

Where,

- **array1, array2, ... −**It is the sequence of arrays to be concatenated. These arrays should have the same shape along all axes except for the one specified by axis.
- **axis −**It is the axis along which the arrays will be joined. The default value is 0, meaning the arrays will be concatenated along rows (for 2D arrays).
### Example: Concatenating 1D Arrays

In the following example, we are concatenating two 1D arrays, "array1" and "array2" along the default axis (axis 0) −

```
import numpy as np

# Create two 1D arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# Concatenate along the default axis (axis=0)
result = np.concatenate((array1, array2))

print("Concatenated Array:", result)
```

The resulting array is a single 1D array containing all elements from both input arrays −

```
Concatenated Array: [1 2 3 4 5 6]
```

### Example: Concatenating 2D Arrays Along Different Axes

Here, we are concatenating two 2D arrays along different axes in NumPy using the concatenate() function −

```
import numpy as np

# Create two 2D arrays
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

# Concatenate along axis 0 (row-wise)
result_axis_0 = np.concatenate((array1, array2), axis=0)

print("Concatenated along Axis 0:\n", result_axis_0)

# Concatenate along axis 1 (column-wise)
result_axis_1 = np.concatenate((array1, array2), axis=1)

print("Concatenated along Axis 1:\n", result_axis_1)
```

Following is the output obtained −

```
Concatenated along Axis 0:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
Concatenated along Axis 1:
 [[1 2 5 6]
 [3 4 7 8]]
```

### Example: Concatenating Arrays of Different Dimensions

In here, we reshape the 1D array "array1" into a 2D array with the given shape, making it compatible for concatenation with the 2D array "array2" along axis "0" −

```
import numpy as np

# Create a 1D and a 2D array
array1 = np.array([1, 2, 3])
array2 = np.array([[4, 5, 6], [7, 8, 9]])

# Reshape array1 to make it 2D for concatenation along axis 0
array1_reshaped = array1.reshape(1, -1)

# Concatenate along axis 0
result = np.concatenate((array1_reshaped, array2), axis=0)

print("Concatenated Array:\n", result)
```

This will produce the following result −

```
Concatenated Array:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

## Stacking Arrays Using stack() Function

The NumPy stack() function joins a sequence of arrays along a new axis, which you specify. This function is useful when you need to preserve the original dimensions of the arrays while adding a new axis for stacking. Following is the syntax −

```
np.stack((array1, array2, ...), axis=0)
```

Where,

- **array1, array2, ... −**It is the sequence of arrays to be stacked. These arrays must have the same shape.
- **axis −**It is the axis along which the arrays will be stacked. The default value is 0, which means the arrays are stacked along a new first axis.
### Example: Stacking 2D Arrays

In the following example, we are stacking two 2D arrays along a new third axis. This creates a 3D array with a particular shape −

```
import numpy as np

# Create two 2D arrays
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

# Stack along a new third axis (axis 2)
stacked_array = np.stack((array1, array2), axis=2)

print("Stacked Array along Axis 2:\n", stacked_array)
print("Shape of Stacked Array:", stacked_array.shape)
```

The new dimension represents the depth of the array, where corresponding elements from the original arrays are stacked together −

```
Stacked Array along Axis 2:
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
Shape of Stacked Array: (2, 2, 2)
```

### Example: Stacking Multiple Arrays

In this example, three 1D arrays are stacked along a new first axis using the stack() function −

```
import numpy as np

# Create three 1D arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
array3 = np.array([7, 8, 9])

# Stack along the first axis (new axis 0)
stacked_array = np.stack((array1, array2, array3), axis=0)

print("Stacked Array:\n", stacked_array)
print("Shape of Stacked Array:", stacked_array.shape)
```

The result is a 2D array where each original array forms a row in the new array −

```
Stacked Array:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Shape of Stacked Array: (3, 3)
```

## Horizontal Stacking Using hstack() Function

The NumPy hstack() function stacks arrays in sequence horizontally (column-wise). This means that the arrays are concatenated along their second dimension (axis 1 for 2D arrays). For 1D arrays, they are simply concatenated along the single available axis.

Following is the syntax −

```
np.hstack((array1, array2, ...))
```

Where,
**array1, array2, ...**is a sequence of arrays that you want to stack horizontally. All arrays must have the same shape along all but the second axis.
### Example

In the example below, we are stacking two 2D arrays, "array1" and "array2" horizontally, resulting in a new array where the columns of "array2" are placed to the right of the columns of "array1" −

```
import numpy as np

# Create two 2D arrays
array1 = np.array([[1, 2, 3],
                   [4, 5, 6]])
array2 = np.array([[7, 8, 9],
                   [10, 11, 12]])

# Horizontally stack the arrays
hstacked_array = np.hstack((array1, array2))

print("Horizontally Stacked 2D Array:\n", hstacked_array)
```

Following is the output of the above code −

```
Horizontally Stacked 2D Array:
 [[ 1  2  3  7  8  9]
  [ 4  5  6 10 11 12]]
```

## Vertical Stacking Using vstack() Function

The NumPy vstack() function stacks arrays in sequence vertically (row-wise). This means that the arrays are concatenated along their first dimension (axis 0 for 2D arrays). For 1D arrays, they are treated as rows and stacked on top of each other, resulting in a 2D array.

Following is the syntax −

```
np.vstack((array1, array2, ...))
```

Where,
**array1, array2, ...**is a sequence of arrays that you want to stack vertically. All arrays must have the same shape along all but the first axis.
### Example

In the following example, we are stacking two 2D arrays, "array1" and "array2" vertically, resulting in a new array where the rows of "array2" are placed below the rows of "array1" −

```
import numpy as np

# Create two 2D arrays
array1 = np.array([[1, 2, 3],
                   [4, 5, 6]])
array2 = np.array([[7, 8, 9],
                   [10, 11, 12]])

# Vertically stack the arrays
vstacked_array = np.vstack((array1, array2))

print("Vertically Stacked 2D Array:\n", vstacked_array)
```

The output obtained is as shown below −

```
Vertically Stacked 2D Array:
 [[ 1  2  3]
  [ 4  5  6]
  [ 7  8  9]
  [10 11 12]]
```

## Splitting Arrays After Joining

After joining arrays, you might want to split them back into their original or differently shaped parts. NumPy provides several functions for splitting arrays −

- **np.split() function −**Splits an array into multiple sub-arrays as specified by the indices or sections.
- **np.array_split() function −**Similar to np.split() function, but allows splitting into unequal sub-arrays.
- **np.hsplit() function −**Splits an array horizontally (column-wise).
- **np.vsplit() function −**Splits an array vertically (row-wise).
- **np.dsplit() function −**Splits an array along the third axis (depth-wise), used for 3D arrays.
> Array splitting is the process of dividing an array into multiple sub-arrays. This operation is the inverse of array joining.

### Example

In the following example, we are splitting the vertically stacked array into two equal parts using the np.vsplit() function, effectively reversing the vertical stacking operation −

```
import numpy as np

# Vertically stack two arrays
array1 = np.array([[1, 2, 3],
                   [4, 5, 6]])
array2 = np.array([[7, 8, 9],
                   [10, 11, 12]])

vstacked_array = np.vstack((array1, array2))

# Split the array back into the original arrays
split_arrays = np.vsplit(vstacked_array, 2)

print("Split Arrays:")
for arr in split_arrays:
   print(arr)
```

After executing the above code, we get the following output −

```
Split Arrays:
[[1 2 3]
 [4 5 6]]
[[ 7  8  9]
 [10 11 12]]
```

---

## 40. NumPy - Sort, Search & Counting Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_sort_search_counting_functions.htm](https://www.tutorialspoint.com/numpy/numpy_sort_search_counting_functions.htm)*

---

---
[Previous](/numpy/numpy_joining_arrays.htm)[Quiz](/numpy/quiz_on_numpy_sort_search_counting_functions.htm)[Next](/numpy/numpy_searching_arrays.htm)
NumPy provides a variety of functions for sorting, searching, and counting elements in arrays. These functions can be extremely useful for data manipulation and analysis.

NumPy offers several sorting algorithms, each with its characteristics. Following is a comparison of three common sorting algorithms −
kindspeedworst casework spacestable'quicksort'1O(n^2)0no'mergesort'2O(n*log(n))~n/2yes'heapsort'3O(n*log(n))0no
## The numpy.sort() Function

The sort() function returns a sorted copy of the input array. It can sort arrays along any specified axis and supports different sorting algorithms. Following is the syntax −

```
numpy.sort(a, axis, kind, order)
```

Where,
Sr.No.Parameter & Description1**a**
Array to be sorted
2**axis**
The axis along which the array is to be sorted. If none, the array is flattened, sorting on the last axis
3**kind**
Default is quicksort
4**order**
If the array contains fields, the order of fields to be sorted

### Example

In the following example, we are sorting a 2D NumPy array both by default and along a specific axis. We also demonstrate sorting a structured array by a specific field, such as 'name' −

```
import numpy as np  

# Create a 2D array
a = np.array([[3, 7], [9, 1]])

print("Our array is:",a)

# Default sort
print("Applying sort() function:",np.sort(a))

# Sort along axis 0
print("Sort along axis 0:",np.sort(a, axis=0))

# Order parameter in sort function
dt = np.dtype([('name', 'S10'), ('age', int)])
a = np.array([("raju", 21), ("anil", 25), ("ravi", 17), ("amar", 27)], dtype=dt)

print("Our array is:",a)

print("Order by name:",np.sort(a, order='name'))
```

It will produce the following output −

```
Our array is:
[[3 7]
 [9 1]]

Applying sort() function:
[[3 7]
 [1 9]]

Sort along axis 0:
[[3 1]
 [9 7]]

Our array is:
[('raju', 21) ('anil', 25) ('ravi', 17) ('amar', 27)]

Order by name:
[('amar', 27) ('anil', 25) ('raju', 21) ('ravi', 17)]
```

## The numpy.argsort() Function

The
**numpy.argsort()**function performs an indirect sort on input array, along the given axis and using a specified kind of sort to return the array of indices of data. This indices array is used to construct the sorted array.
### Example

In this example, we are retrieving indices, which are the positions of the sorted elements in the original array using the argsort() function. Using these indices, you can reconstruct the sorted array −

```
import numpy as np 

# Create an array
x = np.array([3, 1, 2])

print("Our array is:",x)

# Get indices that would sort the array
y = np.argsort(x)

print("Applying argsort() to x:",y)

# Reconstruct the sorted array using the indices
print("Reconstruct original array in sorted order:",x[y])

# Reconstruct the original array using a loop
print("Reconstruct the original array using loop:")
for i in y:
   print(x[i], end=' ')
```

It will produce the following output −

```
Our array is:
[3 1 2]

Applying argsort() to x:
[1 2 0]

Reconstruct original array in sorted order:
[1 2 3]

Reconstruct the original array using loop:
1 2 3
```

## The numpy.lexsort() Function

The NumPy lexort() function performs an indirect sort using a sequence of keys. The keys can be seen as a column in a spreadsheet. The function returns an array of indices, using which the sorted data can be obtained. Note, that the last key happens to be the primary key of sort.

### Example

In this example, we are using np.lexsort() function to sort a dataset based on multiple keys, where the last key "nm" is the primary sorting criterion. The sorted indices are then used to display the sorted data by combining the names and corresponding fields −

```
import numpy as np 

# Define keys
nm = ('raju', 'anil', 'ravi', 'amar') 
dv = ('f.y.', 's.y.', 's.y.', 'f.y.') 

# Get indices for sorted order
ind = np.lexsort((dv, nm))

print("Applying lexsort() function:",ind)

# Use indices to get sorted data
print("Use this index to get sorted data:",[nm[i] + ", " + dv[i] for i in ind])
```

It will produce the following output −

```
Applying lexsort() function:
[3 1 0 2]

Use this index to get sorted data:
['amar, f.y.', 'anil, s.y.', 'raju, f.y.', 'ravi, s.y.']
```

> NumPy provides functions to find indices of maximum, minimum, and non-zero elements, as well as elements satisfying a condition.

## The numpy.argmax() and numpy.argmin() Functions

The NumPy argmax() and argmin() functions return the indices of maximum and minimum elements respectively along the given axis.

### Example

In this example, we are using np.argmax() and np.argmin() functions to find the indices of maximum and minimum values in a 2D array, both in the flattened array and along specific axes −

```
import numpy as np 

# Create a 2D array
a = np.array([[30, 40, 70], [80, 20, 10], [50, 90, 60]])

print("Our array is:",a)

# Apply argmax() function
print("Applying argmax() function:",np.argmax(a))

# Index of maximum number in flattened array
print("Index of maximum number in flattened array:",a.flatten())

# Array containing indices of maximum along axis 0
print("Array containing indices of maximum along axis 0:")
maxindex = np.argmax(a, axis=0)
print(maxindex)

# Array containing indices of maximum along axis 1
print("Array containing indices of maximum along axis 1:")
maxindex = np.argmax(a, axis=1)
print(maxindex)

# Apply argmin() function
print("Applying argmin() function:")
minindex = np.argmin(a)
print(minindex)

# Flattened array
print("Flattened array:",a.flatten()[minindex])

# Flattened array along axis 0
print("Flattened array along axis 0:")
minindex = np.argmin(a, axis=0)
print(minindex)

# Flattened array along axis 1
print("Flattened array along axis 1:")
minindex = np.argmin(a, axis=1)
print(minindex)
```

The output includes indices for these extrema, demonstrating how to access and interpret these positions within the array −

```
Our array is:
[[30 40 70]
 [80 20 10]
 [50 90 60]]

Applying argmax() function:
7

Index of maximum number in flattened array
[30 40 70 80 20 10 50 90 60]

Array containing indices of maximum along axis 0:
[1 2 0]

Array containing indices of maximum along axis 1:
[2 0 1]

Applying argmin() function:
5

Flattened array:
10

Flattened array along axis 0:
[0 1 1]

Flattened array along axis 1:
[0 2 0]
```

## The numpy.nonzero() Function

The
**numpy.nonzero()**function returns the indices of non-zero elements in the input array.
### Example

In the example below, we are retrieving the indices of non-zero elements in the array "a" using the nonzero() function −

```
import numpy as np 
a = np.array([[30,40,0],[0,20,10],[50,0,60]]) 

print ('Our array is:',a)
print ('Applying nonzero() function:',np.nonzero (a))
```

It will produce the following output −

```
Our array is:
[[30 40 0]
 [ 0 20 10]
 [50 0 60]]

Applying nonzero() function:
(array([0, 0, 1, 1, 2, 2]), array([0, 1, 1, 2, 0, 2]))
```

## The numpy.where() Function

The where() function returns the indices of elements in an input array where the given condition is satisfied as shown in the example below −

```
import numpy as np 
x = np.arange(9.).reshape(3, 3) 

print ('Our array is:',x)  

print ('Indices of elements > 3')
y = np.where(x > 3) 
print (y)  

print ('Use these indices to get elements satisfying the condition',x[y])
```

It will produce the following output −

```
Our array is:
[[ 0. 1. 2.]
 [ 3. 4. 5.]
 [ 6. 7. 8.]]

Indices of elements > 3
(array([1, 1, 2, 2, 2]), array([1, 2, 0, 1, 2]))

Use these indices to get elements satisfying the condition
[ 4. 5. 6. 7. 8.]
```

## The numpy.extract() Function

The
**extract()**function returns the elements satisfying any condition as shown in the example below −
```
import numpy as np 
x = np.arange(9.).reshape(3, 3) 

print ('Our array is:',x)  

# define a condition 
condition = np.mod(x,2) == 0 

print ('Element-wise value of condition',condition)

print ('Extract elements using condition',np.extract(condition, x))
```

It will produce the following output −

```
Our array is:
[[ 0. 1. 2.]
 [ 3. 4. 5.]
 [ 6. 7. 8.]]

Element-wise value of condition
[[ True False True]
 [False True False]
 [ True False True]]

Extract elements using condition
[ 0. 2. 4. 6. 8.]
```

---

## 41. NumPy - Searching Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_searching_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_searching_arrays.htm)*

---

---
[Previous](/numpy/numpy_sort_search_counting_functions.htm)[Quiz](/numpy/quiz_on_numpy_searching_arrays.htm)[Next](/numpy/numpy_union_of_arrays.htm)
## Searching Arrays in NumPy

Searching arrays in NumPy refers to the process of locating elements in an array that meet specific criteria or retrieving their indices.

NumPy provides various functions to perform searches, even in large multi-dimensional arrays, they are as follows −

- The where() Function
- The nonzero() Function
- The searchsorted() Function
- The argmax() Function
- The argmin() Function
- The extract() Function
- 
## Using the where() Function

The NumPy where() function is used to find the indices of elements in an array that satisfy a given condition. The function can also be used to replace elements based on a condition. Following is the syntax −

```
np.where(condition, [x, y])
```

Where,

- **condition −**It is the condition to be checked.
- **x (Optional) −**It is the values to use where the condition is true.
- **y (Optional) −**It is the values to use where the condition is false.
### Example

In the following example, we are using the where() function to retrieve the indices of elements that are greater than "25" in the array and also to replace elements in the array that are less than or equal to "25" with "0" −

```
import numpy as np

array = np.array([10, 20, 30, 40, 50])
indices = np.where(array > 25)
print("Indices where array elements are greater than 25:", indices)

# Replacing elements based on condition
modified_array = np.where(array > 25, array, 0)
print("Array after replacing elements <= 25 with 0:", modified_array)
```

Following is the output obtained −

```
Indices where array elements are greater than 25: (array([2, 3, 4]),)
Array after replacing elements <= 25 with 0: [ 0  0 30 40 50]
```

## Using the nonzero() Function

The NumPy nonzero() function is used to find the indices of all non-zero elements in an array. 
It returns a tuple of arrays, where each array contains the indices of non-zero elements along a specific dimension.

This function is useful when you want to filter out zero elements or identify the location of significant elements in sparse arrays. Following is the syntax −

```
numpy.nonzero(a)
```

Where,
**a**is the input array for which you want to find the indices of non-zero elements.
### Example

In the example below, we are retrieving the indices of non-zero elements in the 1D array using the nonzero() function −

```
import numpy as np

array = np.array([0, 1, 2, 0, 3, 0, 4])
nonzero_indices = np.nonzero(array)
print("Indices of non-zero elements:", nonzero_indices)
```

This will produce the following result −

```
Indices of non-zero elements: (array([1, 2, 4, 6]),)
```

## Using the searchsorted() Function

The NumPy searchsorted() function is used to find the indices where elements should be inserted to maintain order in a sorted array.

This function is useful in algorithms that require maintaining a sorted order while dynamically inserting elements. Following is the syntax −

```
np.searchsorted(sorted_array, values, side='left')
```

Where,

- **sorted_array −**It is the sorted array to search.
- **values −**It is the values to insert.
- **side −**If 'left', the index of the first suitable location is given. If 'right', the index of the last suitable location is given.
### Example

In this example, we retrieve the indices at which the values "2", "4", and "6" should be inserted in the sorted array to maintain order −

```
import numpy as np

sorted_array = np.array([1, 3, 5, 7, 9])
values = np.array([2, 4, 6])
indices = np.searchsorted(sorted_array, values)
print("Indices where values should be inserted:", indices)
```

Following is the output of the above code −

```
Indices where values should be inserted: [1 2 3]
```

## Using the argmax() Function

The argmax() function in NumPy is used to find the indices of the maximum value along a specified axis in an array. If no axis is specified, it returns the index of the maximum value in the flattened array. Following is the syntax −

```
numpy.argmax(a, axis=None, out=None)
```

Where,

- **a −**It is the input array.
- **axis (Optional) −**It is the axis along which to find the maximum. If not specified, the array is flattened before performing the operation.
- **out (Optional) −**It is a location into which the result is stored. If provided, it must have the same shape as the expected output.
### Example: Using argmax() Function in a 2D Array

In the following example, we are using the argmax() function to find the index of maximum value along a specified axis in a 2D array −

```
import numpy as np

array = np.array([[10, 15, 5], [7, 12, 20]])
index_of_max_along_axis = np.argmax(array, axis=1)
print("Indices of the maximum values along axis 1:", index_of_max_along_axis)
```

The output obtained is as shown below −

```
Indices of the maximum values along axis 1: [1 2]
```

### Example: Using argmax() Function in a Flattened Array

Here, we are finding the index of maximum value in a flattened array using the argmax() function −

```
import numpy as np

array = np.array([[10, 15, 5], [7, 12, 20]])
index_of_max_flattened = np.argmax(array)
print("Index of the maximum value in the flattened array:", index_of_max_flattened)
```

After executing the above code, we get the following output −

```
Index of the maximum value in the flattened array: 5
```

## Using the argmin() Function

The argmin() function in NumPy is used to find the indices of the minimum value along a specified axis in an array. If no axis is specified, it returns the index of the minimum value in the flattened array. Following is the syntax −

```
numpy.argmin(a, axis=None, out=None)
```

Where,

- **a −**It is the input array.
- **axis (Optional) −**It is the axis along which to find the minimum. If not specified, the array is flattened before performing the operation.
- **out (Optional) −**It is a location into which the result is stored. If provided, it must have the same shape as the expected output.
### Example

In the following example, we are using the argmin() function to find the index of minimum value along a specified axis in a 2D array −

```
import numpy as np

array = np.array([[10, 15, 5], [7, 12, 2]])
index_of_min_along_axis = np.argmin(array, axis=1)
print("Indices of the minimum values along axis 1:", index_of_min_along_axis)
```

The result produced is as follows −

```
Indices of the minimum values along axis 1: [2 2]
```

## Using the extract() Function

The extract() function in NumPy is used to extract elements from an array based on a boolean condition. It returns a 1D array containing only the elements of the input array that correspond to
**True**values in the boolean condition.
> Unlike np.where() function, which returns indices, np.extract() function directly returns the elements that meet the condition.

Following is the syntax −

```
numpy.extract(condition, arr)
```

Where,

- **condition −**It is a boolean array or condition that specifies which elements to extract. It must be of the same shape as arr.
- **arr −**It is the input array from which elements are to be extracted.
### Example

In the example below, we are using the np.extract() function to filter and return elements from the array that are greater than "25" −

```
import numpy as np

array = np.array([10, 20, 30, 40, 50])
condition = array > 25
extracted_elements = np.extract(condition, array)
print("Elements greater than 25:", extracted_elements)
```

We get the output as shown below −

```
Elements greater than 25: [30 40 50]
```

## Searching Using Boolean Indexing

Boolean indexing in NumPy is used for searching and filtering arrays based on specific conditions. It involves creating a boolean array (or mask) where each value is
**True**or**False**based on whether a condition is satisfied.
This boolean array is then used to index into the original array, extracting only those elements where the condition is
**True**.
### Example

Following is a simple example of using boolean indexing in NumPy to filter elements based on a condition −

```
import numpy as np

array = np.array([10, 20, 30, 40, 50])
boolean_mask = array > 25
filtered_array = array[boolean_mask]
print("Filtered array (elements > 25):", filtered_array)
```

We get the output as shown below −

```
Filtered array (elements > 25): [30 40 50]
```

---

## 42. NumPy - Union of Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_union_of_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_union_of_arrays.htm)*

---

---
[Previous](/numpy/numpy_searching_arrays.htm)[Quiz](/numpy/quiz_on_numpy_union_of_arrays.htm)[Next](/numpy/numpy_finding_unique_rows.htm)
## Union of Arrays in NumPy

The union of arrays in NumPy refers to combining multiple arrays into a single array while removing duplicate elements. This ensures that each element appears only once in the array. In NumPy, we can achieve this using the
**union1d()**function.
> The union of arrays in NumPy is similar to the union operation in set theory where all unique elements from multiple sets are combined into one set.

## Using union1d() Function

The np.union1d() function in NumPy is used to compute the union of two arrays. This function returns a sorted array of unique values that are present in either of the input arrays. Following is the syntax −

```
numpy.union1d(arr1, arr2)
```

Where,

- **arr1 −**It is the first input array. This array can be of any shape or data type, but typically is a 1D array for simplicity.
- **arr2 −**It is the second input array. It should have the same data type as "arr1".
### Example

In the following example, we are using the union1d() function to find the union of 2 arrays "arr1" and "arr2" −

```
import numpy as np

# Define two arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

# Compute the union of the two arrays
union_result = np.union1d(arr1, arr2)
print("Union of two arrays:", union_result)
```

Following is the output obtained −

```
Union of two arrays: [1 2 3 4 5 6]
```

## Union of Multiple Arrays

To find the union of more than two arrays, you can use union1d() function multiple times. This involves applying the function iteratively to pairs of arrays until all arrays have been included in the union.

Alternatively, you can use functions like np.concatenate() along with np.unique(). Following is the syntax −

```
numpy.concatenate((array1, array2, ...))
numpy.unique(array)
```

### Example: Sequential Union

In the example below, we are finding the union of 3 arrays iteratively using the union1d() function −

```
import numpy as np

# Define multiple 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([2, 3, 4])
arr3 = np.array([4, 5, 6])

# Compute the union of three arrays
# Union of first two arrays
union_temp = np.union1d(arr1, arr2)  
# Union with the third array
union_result = np.union1d(union_temp, arr3) 

print("Union of multiple arrays (sequential):", union_result)
```

This will produce the following result −

```
Union of multiple arrays (sequential): [1 2 3 4 5 6]
```

### Example: Using np.concatenate() and np.unique() Functions

In this example, we first concatenate all arrays into a single array. Then, we extract the unique elements from this concatenated array using the unique() function −

```
import numpy as np

# Define multiple 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([2, 3, 4])
arr3 = np.array([4, 5, 6])

# Concatenate all arrays into one
concatenated_array = np.concatenate((arr1, arr2, arr3))

# Find unique elements
union_result = np.unique(concatenated_array)

print("Union of multiple arrays (concatenate and unique):", union_result)
```

Following is the output of the above code −

```
Union of multiple arrays (concatenate and unique): [1 2 3 4 5 6]
```

## Handling Multi-dimensional Arrays

We can also apply union operations to multi-dimensional arrays. To perform union operations on multi-dimensional arrays, you need to flatten the arrays first.

Flattening transforms a multi-dimensional array into a 1D array, allowing you to perform union operations as if the arrays were one-dimensional. After performing the union operation, you can reshape the result back into the original dimensions if needed.

> In NumPy, multi-dimensional arrays are arrays with more than one dimension. They are commonly referred to as "ndarrays" and are used to represent complex data structures like matrices or higher-dimensional tensors. 
> Managing these arrays involves understanding their structure, performing operations, and efficiently manipulating data.

In NumPy, multi-dimensional arrays are arrays with more than one dimension. They are commonly referred to as "ndarrays" and are used to represent complex data structures like matrices or higher-dimensional tensors.

Managing these arrays involves understanding their structure, performing operations, and efficiently manipulating data.

### Example

In this example, we first define two 2D arrays "arr1" and "arr2". We then flatten them into 1d arrays using the flatten() function and then compute their union −

```
import numpy as np

# Define 2D arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[3, 4], [5, 6]])

# Flatten the arrays and compute the union
flattened_arr1 = arr1.flatten()
flattened_arr2 = arr2.flatten()
union_result = np.union1d(flattened_arr1, flattened_arr2)

print("Union of 2D arrays:", union_result)
```

The output obtained is as shown below −

```
Union of 2D arrays: [1 2 3 4 5 6]
```

## Union with Complex Data Types

Union operations with complex data types in NumPy involve working with arrays that have structured or object-oriented data. Unlike simple numeric arrays, complex data types can include fields such as integers, floats, strings, and even other arrays.

The np.union1d() function can handle arrays of any data type as long as they are comparable.

### Example

In the following example, we first create 2 structured arrays and then combine them using the union1d() function, removing duplicates and keeping the unique entries −

```
import numpy as np

# Define structured arrays
arr1 = np.array([(1, 'a'), (2, 'b')], dtype=[('num', 'i4'), ('letter', 'S1')])
arr2 = np.array([(2, 'b'), (3, 'c')], dtype=[('num', 'i4'), ('letter', 'S1')])

# Compute the union of structured arrays
union_result = np.union1d(arr1, arr2)
print("Union of structured arrays:", union_result)
```

After executing the above code, we get the following output −

```
Union of structured arrays: [(1, b'a') (2, b'b') (3, b'c')]
```

---

## 43. NumPy - Finding Unique Rows

*Source: [https://www.tutorialspoint.com/numpy/numpy_finding_unique_rows.htm](https://www.tutorialspoint.com/numpy/numpy_finding_unique_rows.htm)*

---

---
[Previous](/numpy/numpy_union_of_arrays.htm)[Quiz](/numpy/quiz_on_numpy_finding_unique_rows.htm)[Next](/numpy/numpy_creating_datetime_arrays.htm)
## Finding Unique Rows in NumPy Array

In NumPy, arrays can contain multiple rows of data, and sometimes you might want to identify rows that are unique, meaning they appear only once in the array.
**Finding unique rows**involves determining which rows are distinct from others based on their content.
In NumPy, we can achieve this using the
**unique()**function.
## Using union1d() Function

The np.unique() function is commonly used to find unique elements in an array. When applied with the
**axis**parameter, it can be used to find unique rows. Following is the syntax −
```
numpy.unique(a, axis=None, return_index=False, return_inverse=False, return_counts=False)
```

Where,

- **a −**It is the input array.
- **axis −**It is the axis along which to find unique values. Set to 0 for rows.
- **return_index −**It determines whether to return the indices of the first occurrences.
- **return_inverse −**It determines whether to return the indices that can reconstruct the array.
- **return_counts −**It determines whether to return the counts of unique values.
### Example: Finding Unique Elements in a 1D Array

The simplest use of np.unique() function is to find unique elements in a one-dimensional array −

```
import numpy as np

# Define a 1D array with duplicate values
array = np.array([1, 2, 2, 3, 4, 4, 5])

# Find unique elements
unique_elements = np.unique(array)

print("Unique Elements:\n", unique_elements)
```

Following is the output obtained −

```
Unique Elements:
[1 2 3 4 5]
```

### Example: Unique Rows in a 2D Array

In the following example, we are using the unique() function to retrieve the unique rows in a 2D array, removing any duplicate rows −

```
import numpy as np

# Define an array with duplicate rows
array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3],
    [7, 8, 9]
])

# Find unique rows
unique_rows = np.unique(array, axis=0)

print("Unique Rows:\n", unique_rows)
```

This will produce the following result −

```
Unique Rows:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

## Finding Unique Rows with Indexes

We can find the indices of the unique rows in the original array in NumPy by setting the
**return_index**parameter to**True**in the unique() function.
### Example

In this example, we are finding unique rows and their indices using the unique() function −

```
import numpy as np

# Define an array with duplicate rows
array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3],
    [7, 8, 9]
])

# Find unique rows and their indices
unique_rows, indices = np.unique(array, axis=0, return_index=True)

print("Unique Rows:\n", unique_rows)
print("Indices of Unique Rows:\n", indices)
```

Following is the output of the above code −

```
Unique Rows:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Indices of Unique Rows:
[0 1 3]
```

## Reconstructing the Original Array

If you need to reconstruct the original array from the unique rows, you can use the indices returned by np.unique() function with the
**return_inverse**parameter set to**True**. The inverse indices can be used to map back to the original data from the unique values.
### Example

In this example, we are identifying unique rows in a NumPy array and their original indices using the unique() function. We then reconstruct the array using these indices to verify that the unique rows match the original array without duplicates −

```
import numpy as np

# Define an array with duplicate rows
array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3],
    [7, 8, 9]
])

# Find unique rows and their indices
unique_rows, indices = np.unique(array, axis=0, return_index=True)

# Reconstruct the original array using the indices
reconstructed_array = array[np.sort(indices)]

print("Reconstructed Array:\n", reconstructed_array)
```

The output obtained is as shown below −

```
Reconstructed Array:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

## Counting Unique Rows

In addition to finding unique rows, you might want to count how many times each unique row appears in the array. In NumPy, you can achieve this by setting the
**return_counts**parameter to**True**in the unique() function.
This is particularly useful when working with multi-dimensional arrays where each row represents a record or observation.

### Example

In the following example, we retrieve the count of each unique row in the original array using the unique() function −

```
import numpy as np

# Define an array with duplicate rows
array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3],
    [7, 8, 9]
])

# Find unique rows and their counts
unique_rows, counts = np.unique(array, axis=0, return_counts=True)

print("Unique Rows:\n", unique_rows)
print("Counts of Each Row:\n", counts)
```

After executing the above code, we get the following output −

```
Unique Rows:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Counts of Each Row:
[2 1 1]
```

## Multi-dimensional Arrays

For multi-dimensional arrays, you can use np.unique() function to find unique rows by setting the
**axis**parameter to**0**. To handle unique values across all dimensions, you can use the default settings.
### Example

In the example below, we flatten the 3D array into 2D and then find unique rows using the unique() function −

```
import numpy as np

# Define a 3D array
array = np.array([
    [[1, 2], [3, 4]],
    [[1, 2], [5, 6]],
    [[1, 2], [3, 4]]
])

# Flatten the 3D array to 2D for uniqueness check
array_2d = array.reshape(-1, array.shape[-1])

# Find unique rows in the flattened array
unique_rows = np.unique(array_2d, axis=0)

print("Unique Rows in 3D Array:\n", unique_rows)
```

The result produced is as follows −

```
Unique Rows in 3D Array:
[[1 2]
 [3 4]
 [5 6]]
```

---

## 44. NumPy - Creating Datetime Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_creating_datetime_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_creating_datetime_arrays.htm)*

---

---
[Previous](/numpy/numpy_finding_unique_rows.htm)[Quiz](/numpy/quiz_on_numpy_creating_datetime_arrays.htm)[Next](/numpy/numpy_binary_operators.htm)
## Datetime Arrays in NumPy

Datetime arrays are arrays that hold date and time values. NumPy provides the
**datetime64**and**timedelta64**data types for handling dates and times with a wide range of precision.
The "datetime64" type represents dates and times, while "timedelta64" represents differences between dates or times.

## Creating Datetime Arrays

In NumPy, we can create datetime arrrays using the array() function and the datetime64() function −

### Using the np.array() Function

You can create a datetime array by specifying date strings or timestamps with the numpy.array() function. You need to specify the
**dtype**as**datetime64**to ensure that the array elements are treated as datetime objects.
Following is the syntax −

```
numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
```

Where,

- **object:**This is the input data (e.g., list, tuple, or other array-like objects) that you want to convert into a NumPy array.
- **dtype:**Specifies the desired data type of the array elements. If not provided, NumPy will infer the data type from the input data.
- **copy:**If True, the function will create a copy of the input data. If False, a copy is made only if necessary.
- **order:**Specifies the memory layout order. 'C' is for row-major (C-style) order, 'F' is for column-major (Fortran-style) order, and 'A' or 'K' can be used for automatic order selection.
- **subok:**If True, subclasses of ndarray are passed-through; if False, the returned array will be forced to be a base-class ndarray.
- **ndmin:**Specifies the minimum number of dimensions that the resulting array should have. If necessary, new axes are added to the left of the shape.
### Example

In the following example, we are converting a list of date strings into a NumPy array by passing the "dtype" parameter to the array() function −

```
import numpy as np

# Creating a datetime array using date strings
dates = np.array(['2024-08-01', '2024-08-15', '2024-09-01'], dtype='datetime64')

print("Datetime Array:\n", dates)
```

Following is the output obtained −

```
Datetime Array:
 ['2024-08-01' '2024-08-15' '2024-09-01']
```

### Using the np.datetime64() Function

The datetime64() function in NumPy is used to create arrays of dates and times. This function provides a way to work with date and time data, enabling operations on time series data. The "datetime64" data type allows for date and time precision down to the "nanosecond" level.

Following is the syntax −

```
numpy.datetime64(datetime_string, unit)
```

Where,

- **datetime_string:**It is a string representing the date and/or time. The format of this string must match the unit specified.
- **unit (optional):**It specifies the time unit (e.g., Y, M, D, h, m, s, ms, us, ns). The unit defines the precision of the date and time representation.
### Example

Here, we create individual datetime objects and retrieve an array of "datetime64" objects created from specified dates −

```
import numpy as np

# Creating individual datetime objects
date1 = np.datetime64('2024-08-01')
date2 = np.datetime64('2024-08-15')

# Creating an array of datetime objects
dates = np.array([date1, date2, np.datetime64('2024-09-01')])

print("Datetime Array:\n", dates)
```

This will produce the following result −

```
Datetime Array:
['2024-08-01' '2024-08-15' '2024-09-01']
```

## Creating Datetime Arrays with Specific Frequencies

Creating datetime arrays with specific frequencies in NumPy allows you to generate sequences of dates or times that follow a regular interval, such as daily, monthly, hourly, etc.

In NumPy, you can create datetime arrays with specific frequencies using the
**np.arange()**or**np.linspace()**functions in combination with**datetime64**data types. These functions allow you to generate evenly spaced datetime values between a start and end date or time.
### Using the np.arange() Function

The np.arange() function is used to create datetime arrays with a specified frequency. The key parameters include the start date or time, the end date or time, and the step (frequency) between consecutive datetime values. Following is the syntax −

```
numpy.arange(start, stop, step, dtype='datetime64')
```

Where,

- **start:**It is the starting date or time in datetime64 format.
- **stop:**It is the ending date or time (exclusive) in datetime64 format.
- **step:**It is the frequency or interval between consecutive dates/times, specified using timedelta64.
- **dtype:**It is the data type, which should be 'datetime64'.
### Example

In the example below, we are creating an array of dates from "August 1, 2024" to "August 10, 2024", with a daily frequency −

```
import numpy as np

# Creating a daily datetime array
dates = np.arange('2024-08-01', '2024-08-11', dtype='datetime64[D]')
print("Daily Datetime Array:", dates)
```

Following is the output of the above code −

```
Daily Datetime Array: ['2024-08-01' '2024-08-02' '2024-08-03' '2024-08-04' '2024-08-05'
 '2024-08-06' '2024-08-07' '2024-08-08' '2024-08-09' '2024-08-10']
```

### Using the np.linspace() Function

While np.arange() function is commonly used for creating datetime arrays with specific frequencies, np.linspace()function can also be used when you want to specify the number of points between two datetime values rather than the interval. Following is the syntax −

```
numpy.linspace(start, stop, num, dtype='datetime64')
```

Where,

- **start:**It is the starting datetime value "datetime64" format.
- **stop:**It is the ending datetime value "datetime64" format.
- **num:**It is the number of datetime values to generate between start and stop.
- **dtype:**It is the data type, which should be 'datetime64'.
### Example

In this example, we are creating a datetime array with 5 evenly spaced datetime values between the start and end dates −

```
import numpy as np

# Convert start and end dates to datetime64
start_date = np.datetime64('2024-08-01')
end_date = np.datetime64('2024-08-10')

# Calculate the difference in days between start and end
date_range = np.arange(start_date, end_date + np.timedelta64(1, 'D'))

# Use linspace on the integer values of dates
datetimes = np.linspace(0, len(date_range)-1, num=5, dtype=int)

# Map back to the original date range
datetime_array = start_date + datetimes.astype('timedelta64[D]')

print("Datetime Array with 5 Points:", datetime_array)
```

The output obtained is as shown below −

```
Datetime Array with 5 Points: ['2024-08-01' '2024-08-03' '2024-08-05' '2024-08-07' '2024-08-10']
```

## Creating Time Arrays

Creating time arrays in NumPy involves generating sequences of time-related values, such as hours, minutes, or seconds, similar to how you would create arrays of dates.

In NumPy, you can create time arrays using the
**datetime64**and**timedelta64**data types. While "datetime64" is used for absolute points in time (such as a specific date and time), "timedelta64" represents a duration (such as hours or minutes). By combining these types, you can create arrays of time values that represent specific moments or intervals.
### Example

The following example produces a time array starting at midnight and ending at noon, with each element representing a specific hour −

```
import numpy as np

# Define the start and end times
start_time = np.datetime64('2024-08-01T00:00')
end_time = np.datetime64('2024-08-01T12:00')

# Create an array of hourly intervals
time_array = np.arange(start_time, end_time, np.timedelta64(1, 'h'))

print("Time Array:", time_array)
```

After executing the above code, we get the following output −

```
Time Array: 
['2024-08-01T00:00' '2024-08-01T01:00' '2024-08-01T02:00'
 '2024-08-01T03:00' '2024-08-01T04:00' '2024-08-01T05:00'
 '2024-08-01T06:00' '2024-08-01T07:00' '2024-08-01T08:00'
 '2024-08-01T09:00' '2024-08-01T10:00' '2024-08-01T11:00']
```

## Combining Date and Time

Combining date and time involves creating a "datetime64" object that consists of both the date and the specific time of day.

You can create a datetime64 array by specifying the date and time in a single string, or by combining separate date and time arrays using NumPy's vectorized operations.

### Example

In this example, we create an array where each element represents a specific date and time −

```
import numpy as np

# Creating datetime arrays with date and time
datetimes = np.array([np.datetime64('2024-08-01T08:00:00'), 
                      np.datetime64('2024-08-02T12:30:00'), 
                      np.datetime64('2024-08-03T16:45:00')])

print("Datetime Array with Date and Time:\n", datetimes)
```

The result produced is as follows −

```
Datetime Array with Date and Time:
 ['2024-08-01T08:00:00' '2024-08-02T12:30:00' '2024-08-03T16:45:00']
```

---

## 45. NumPy - Binary Operators

*Source: [https://www.tutorialspoint.com/numpy/numpy_binary_operators.htm](https://www.tutorialspoint.com/numpy/numpy_binary_operators.htm)*

---

---
[Previous](/numpy/numpy_creating_datetime_arrays.htm)[Quiz](/numpy/quiz_on_numpy_binary_operators.htm)[Next](/numpy/numpy_string_functions.htm)
## Binary Operators in NumPy

Binary operators in NumPy are operations that take two operands (usually arrays) and perform element-wise operations between corresponding elements of the arrays. These operations include addition, subtraction, multiplication, division, logical operations, and more.

For example, if you have two arrays, you can add them together, subtract one from the other, multiply their elements, and so on. Each of these operations is performed element-wise, meaning that the operation is applied to each corresponding pair of elements in the two arrays.

Following are the functions for bitwise operations available in NumPy package.
Sr.No.Operation & Description1[bitwise_and](/numpy/numpy_bitwise_and.htm)**bitwise_and**
Computes bitwise AND operation of array elements
2[bitwise_or](/numpy/numpy_bitwise_or.htm)**bitwise_or**
Computes bitwise OR operation of array elements
3[bitwise_xor](/numpy/numpy_bitwise_xor.htm)**bitwise_xor**
Computes bitwise XOR of array elements. Each bit of the result is 1 if the corresponding bits in the input are different.
4[left_shift](/numpy/numpy_left_shift.htm)**left_shift**
Shifts bits of a binary representation to the left
5[right_shift](/numpy/numpy_right_shift.htm)**right_shift**
Shifts bits of binary representation to the right
6[bitwise_right_shift](/numpy/numpy_bitwise_right_shift.htm)**bitwise_right_shift**
Shifts the bits of an integer to the right.
7[invert](/numpy/numpy_invert.htm)**invert**
Computes bitwise NOT
8[bitwise_invert](/numpy/numpy_bitwise_invert.htm)**bitwise_invert**
Computes bitwise inversion of the elements.
9[packbits](/numpy/numpy_packbits.htm)**packbits**
Packs the elements of a binary array into packed bitfield representation.
10[unpackbits](/numpy/numpy_unpackbits.htm)**unpackbits**
Unpacks elements of a binary array into a list of bits.
11[binary_repr](/numpy/numpy_binary_repr.htm)**binary_repr**
Converts an integer to its binary representation as a string.

## The Bitwise AND (&) Operation

The bitwise AND operation compares each bit of two numbers. If both bits are "1", the result is "1"; otherwise, it is "0" −

```
# (0101 in binary)
a = 5
# (0011 in binary)
b = 3
# (0001 in binary, which is 1 in decimal)
result = a & b  
print("The result obtained is:",result)
```

Following is the output obtained −

```
The result obtained is: 1
```

## The Bitwise OR (|) Operation

The bitwise OR operation compares each bit of two numbers. If either bit is "1", the result is "1"; if both are "0", the result is "0". −

```
# (0101 in binary)
a = 5       
# (0011 in binary)
b = 3       
# (0111 in binary, which is 7 in decimal)
result = a | b  
print("The result obtained is:",result)
```

This will produce the following result −

```
The result obtained is: 7
```

## The Bitwise NOT (~) Operation

The bitwise NOT operation inverts each bit of the number, turning "0" into "1" and "1" into "0". This is also known as the bitwise complement −

```
# (0101 in binary)
a = 5      
# (1010 in binary, which is -6 in decimal with two's complement)
result = ~a  
print("The result obtained is:",result)
```

Following is the output of the above code −

```
The result obtained is: -6
```

## The Left Shift (<<) Operation

The left shift operation shifts the bits of the number to the left by a specified number of positions. Bits shifted out on the left are discarded, and 0s are shifted in on the right −

```
# (0101 in binary)
a = 5    
# (1010 in binary, which is 10 in decimal)
result = a << 1  
print("The result obtained is:",result)
```

The output obtained is as shown below −

```
The result obtained is: 10
```

## The Right Shift (>>) Opeartion

The right shift operation shifts the bits of the number to the right by a specified number of positions. Bits shifted out on the right are discarded, and the leftmost bits are filled based on the sign of the number (arithmetic shift for signed integers) −

```
# (0101 in binary)
a = 5    
# (0010 in binary, which is 2 in decimal)
result = a >> 1  
print("The result obtained is:",result)
```

After executing the above code, we get the following output −

```
The result obtained is: 2
```

---

## 46. NumPy - String Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_string_functions.htm](https://www.tutorialspoint.com/numpy/numpy_string_functions.htm)*

---

---
[Previous](/numpy/numpy_binary_operators.htm)[Quiz](/numpy/quiz_on_numpy_string_functions.htm)[Next](/numpy/numpy_matrix_library.htm)
String functions in NumPy are designed to operate on arrays of strings. They are part of the NumPy
**char**module, which provides a set of vectorized string operations that can be applied to each element of a string array.
## Key Features of NumPy String Functions

Following are the key features of NumPy String Functions −

- **Element-wise Operations:**The core advantage of NumPy string functions is their ability to perform operations on each element of an array independently. This allows for efficient manipulation of large datasets.
- **Vectorization:**By using NumPy library operations are vectorized i.e., which enhances performance compared to traditional Python string handling methods. Vectorization utilizes optimized C libraries to perform computations by reducing the execution time significantly.
- **Compatibility with Arrays:**NumPy string functions work directly with arrays of strings by making it easier to process large amounts of textual data without needing to convert them into lists or other formats.
The String operations are performed element-wise on arrays. They are particularly useful for low-level data manipulation and efficient computation.

## List of String Functions

The following functions are used to perform vectorized string operations for arrays of dtype numpy.string_ or numpy.unicode_. They are based on the standard string functions in Python's built-in library.
Sr.No.Operation & Description1[numpy.char.add()](/numpy/numpy_char_add.htm)
Concatenates two arrays of strings element-wise.
2[numpy.char.center()](/numpy/numpy_char_center.htm)
Centers each string in an array within a specified width, padded with a specified character.
3[numpy.char.capitalize()](/numpy/numpy_char_capitalize.htm)
Capitalizes the first character of each string in the array.
4[numpy.char.decode()](/numpy/numpy_char_decode.htm)
Decodes each string in an array using the specified encoding.
5[numpy.char.encode()](/numpy/numpy_char_encode.htm)
Encodes each string in an array using the specified encoding.
6**numpy.char.ljust**
Left-justifies each string in an array, padding with a specified character.
7[numpy.char.lower()](/numpy/numpy_char_lower.htm)
Converts all characters of each string in the array to lowercase.
8**numpy.char.lstrip()**
Strips leading characters from each string in an array.
9**numpy.char.mod()**
Formats strings using specified values for placeholders in the strings.
10[numpy.char.multiply()](/numpy/numpy_char_multiply.htm)
Repeats each string in the array a specified number of times.
11[numpy.char.replace()](/numpy/numpy_char_replace.htm)
Replaces occurrences of a substring with another substring in each string.
12**numpy.char.rjust()**
Right-justifies each string in an array, padding with a specified character.
13**numpy.char.rstrip()**
Strips trailing characters from each string in an array.
14[numpy.char.strip()](/numpy/numpy_char_strip.htm)
Strips leading and trailing characters from each string in an array.
15**numpy.char.swapcase()**
Swaps the case of each character in each string.
16[numpy.char.title()](/numpy/numpy_char_title.htm)
Converts each string in the array to title case.
17**numpy.char.translate()**
Translates characters in each string according to a translation table.
18[numpy.char.upper()](/numpy/numpy_char_upper.htm)
Converts all characters of each string in the array to uppercase.
19**numpy.char.zfill()**
Pads each string with zeros on the left to fill a specified width.
20**numpy.char.equal()**
Compares each string in an array for equality with another array.
21**numpy.char.not_equal()**
Compares each string in an array for inequality with another array.
22**numpy.char.greater_equal()**
Compares each string in an array to see if it is greater than or equal to another.
23**numpy.char.less_equal()**
Compares each string in an array to see if it is less than or equal to another.
24**numpy.char.greater()**
Compares each string in an array to see if it is greater than another.
25**numpy.char.less()**
Compares each string in an array to see if it is less than another.
26**numpy.char.count()**
Counts occurrences of a substring in each string in the array.
27**numpy.char.endswith()**
Checks if each string in the array ends with a specified suffix.
28**numpy.char.find()**
Finds the lowest index of a substring in each string.
29**numpy.char.index()**
Similar to find, but raises an error if the substring is not found.
30**numpy.char.isalnum()**
Checks if each string is alphanumeric.
31**numpy.char.isalpha()**
Checks if each string is alphabetic.
32**numpy.char.isdecimal()**
Checks if each string is a decimal string.
33**numpy.char.isdigit**
Checks if each string contains only digits.
34**numpy.char.islower()**
Checks if each string is in lowercase.
35**numpy.char.isnumeric()**
Checks if each string is numeric.
36**numpy.char.isspace()**
Checks if each string contains only whitespace.
37**numpy.char.istitle()**
Checks if each string is title-cased.
38**numpy.char.isupper()**
Checks if each string is in uppercase.
39**numpy.char.rfind()**
Finds the highest index of a substring in each string.
40**numpy.char.rindex()**
Similar to rfind, but raises an error if the substring is not found.
41**numpy.char.startswith()**
Checks if each string starts with a specified prefix.
42**numpy.char.str_len()**
Returns the length of each string in the array.
43[numpy.char.split()](/numpy/numpy_char_string_split.htm)
Returns the splitted array string.
44[numpy.char.splitlines()](/numpy/numpy_char_splitlines.htm)
Split each element of an array of strings into a list of lines.
45[numpy.char.join()](/numpy/numpy_char_join.htm)
Join the elements of an array of strings with a specified delimiter.

Let's look at the important functions quickly −

## The add() Function

The add() function in NumPy is used to concatenate strings using the
**+**operator as shown in the example below −
```
a = "Hello"
b = "World"
result = a + " " + b
print(result)
```

Following is the output obtained −

```
Hello World
```

## The multiply() Function

The multiply() function in NumPy is used to multiply(repeat) strings using the
*****operator as shown in the example below −
```
a = "Hello"
result = a * 3
print(result)
```

This will produce the following result −

```
HelloHelloHello
```

## The center() Function

The center() function centers a string in a field of a specified width, padding it with spaces or a specified character −

```
s = "hello"
result = s.center(10, '*')
print(result)
```

Following is the output of the above code −

```
**hello***
```

## The capitalize() Function

The capitalize() function capitalizes the first character of the string and makes all other characters lowercase −

```
s = "hello world"
result = s.capitalize()
print(result)
```

The output obtained is as shown below −

```
Hello world
```

## The title() Function

The title() function capitalizes the first letter of each word in the string −

```
s = "hello world"
result = s.title()
print(result)
```

After executing the above code, we get the following output −

```
Hello World
```

## The lower() and upper() Functions

The lower() function converts all characters in the string to lowercase. Whereas, the upper() function converts all characters in the string to uppercase −

```
s = "Hello World"
res1 = s.lower()
res2 = s.upper()
print("Lowercase:", res1)
print("Uppercase:",res2)
```

The result produced is as follows −

```
Lowercase: hello world
Uppercase: HELLO WORLD
```

## The decode() Function

In Python 3, the decode() function is typically used for byte objects, not strings. To decode bytes to a string, you use the decode() function −

```
# Bytes object
b = b"hello world"
result = b.decode('utf-8')
print(result)
```

We get the output as shown below −

```
hello world
```

---

## 47. NumPy - Matrix Library

*Source: [https://www.tutorialspoint.com/numpy/numpy_matrix_library.htm](https://www.tutorialspoint.com/numpy/numpy_matrix_library.htm)*

---

---
[Previous](/numpy/numpy_string_functions.htm)[Quiz](/numpy/quiz_on_numpy_matrix_library.htm)[Next](/numpy/numpy_linear_algebra.htm)
## The NumPy Matrix Library

The NumPy matrix library provides functions for creating and manipulating matrices. This library allows you to perform a wide range of matrix operations, including matrix multiplication, inversion, and decomposition.

In NumPy, matrices can be created using the numpy.matrix() function or by converting existing arrays to matrices. This tutorial will cover different methods to create matrices.

## Using numpy.matrix() Function

The numpy.matrix() function is used to create a matrix from a string representation or from existing data structures. This function is best suitable for creating small matrices quickly.

### Example

In the following example, we are creating a matrix from a string representation and from an existing array. The np.matrix() function interprets the string as a 2x2 matrix, and the array is directly converted to a matrix format −

```
import numpy as np

# Creating a matrix from a string
matrix_str = np.matrix('1 2; 3 4')
print("Matrix from string:\n", matrix_str)

# Creating a matrix from an array
array_data = np.array([[1, 2], [3, 4]])
matrix_from_array = np.matrix(array_data)
print("Matrix from array:\n", matrix_from_array)
```

Following is the output obtained −

```
Matrix from string:
[[1 2]
 [3 4]]
Matrix from array:
 [[1 2]
 [3 4]]
```

## Using numpy.array() Function

You can convert a NumPy array into a matrix using the numpy.asmatrix() function. This is useful when you have existing data in array form on which you want to perform matrix operations.

### Example

In the example below, we are creating an array and then converting it to a matrix using np.asmatrix() function −

```
import numpy as np

# Creating an array
array_data = np.array([[5, 6], [7, 8]])

# Converting array to matrix
matrix_data = np.asmatrix(array_data)
print("Converted Matrix:\n", matrix_data)
```

This will produce the following result −

```
Converted Matrix:
[[5 6]
 [7 8]]
```

## Matrix Operations in NumPy

Once you have created a matrix, you can perform a wide range of matrix operations, such as addition, multiplication, transpose, inversion, and more.

### Matrix Addition

Adding two matrices involves adding the corresponding elements. If two matrices have the same shape, you can add them together element-wise.

#### Example

In this example, "matrix_1" and "matrix_2" are added together element-wise, meaning each element of "matrix_1" is added to the corresponding element in "matrix_2" −

```
import numpy as np

# Add two matrices
matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

result = matrix_1 + matrix_2
print(result)
```

Following is the output of the above code −

```
[[ 6  8]
 [10 12]]
```

### Matrix Multiplication

We can perform matrix multiplication using the following ways −

- Using the*****operator
- Using the**@**operator (Python 3.5+)
- Using np.dot() function
- Using the numpy.matmul() function
Unlike element-wise multiplication, matrix multiplication follows the linear algebra rules.

#### Example

In this example, we are multiplying two matrices using all the above given ways −

```
import numpy as np

matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

# Matrix multiplication using *
matrix_product1 = matrix_1 * matrix_2
print("Matrix Multiplication (*):\n", matrix_product1)

# Matrix multiplication using @
matrix_product2 = matrix_1 @ matrix_2
print("Matrix Multiplication (@):\n", matrix_product2)

# Matrix multiplication using np.dot()
matrix_product3 = np.dot(matrix_1, matrix_2)
print("Matrix Multiplication (np.dot()):\n", matrix_product3)

# Matrix multiplication using np.matmul()
matrix_product4 = np.matmul(matrix_1, matrix_2)
print("Matrix Multiplication (np.matmul()):\n", matrix_product4)
```

The output obtained is as shown below −

```
Matrix Multiplication (*):
[[ 5 12]
 [21 32]]
Matrix Multiplication (@):
[[19 22]
 [43 50]]
Matrix Multiplication (np.dot()):
[[19 22]
 [43 50]]
Matrix Multiplication (np.matmul()):
 [[19 22]
 [43 50]]
```

### Matrix Inversion

Matrix inversion is an operation to find a matrix that, when multiplied by the original matrix, yields the identity matrix. The inverse of a matrix can be calculated using the np.linalg.inv() function.

However, not all matrices are invertible. A matrix must be square and have a non-zero determinant to be invertible.

#### Example

In the following example, we are inverting a 2x2 matrix using np.linalg.inv() function. The output is a new matrix that, when multiplied by the original, results in the identity matrix −

```
import numpy as np

matrix = np.array([[1, 2], [3, 4]])

inverse_matrix = np.linalg.inv(matrix)
print(inverse_matrix)
```

After executing the above code, we get the following output −

```
[[-2.   1. ]
 [ 1.5 -0.5]]
```

### Matrix Transpose

Transposing a matrix involves flipping it over its diagonal, swapping the row and column indices. We can transpose a matrix in NumPy using the
**.T**attribute.
#### Example

In the following example, we are transposing a 2x2 matrix using the ".T" attribute −

```
import numpy as np

# Transpose of a matrix
matrix = np.array([[1, 2], [3, 4]])

transposed = matrix.T
print(transposed)
```

The result produced is as follows −

```
[[1 3]
 [2 4]]
```

## Matrix Determinant

The determinant of a matrix is a scalar value that can be calculated using the np.linalg.det() function. It provides information about the matrix's properties, such as whether it is invertible.

A non-zero determinant indicates that the matrix is invertible, while a determinant of zero means the matrix is singular.

### Example

In this example, np.linalg.det() function computes the determinant of the given matrix −

```
import numpy as np

# Compute the determinant
matrix = np.array([[1, 2], [3, 4]])

det = np.linalg.det(matrix)
print("Determinant:", det)
```

We get the output as shown below −

```
Determinant: -2.0000000000000004
```

## Eigenvalues and Eigenvectors

The numpy.linalg.eig() function is used to compute the eigenvalues and right eigenvectors of a square matrix. The eigenvalues indicate the magnitude of the vectors, while the eigenvectors provide the directions.

> Eigenvalues and eigenvectors are fundamental concepts in linear algebra, and are important in many areas such as PCA (Principal Component Analysis) and solving differential equations.

### Example

In this example, the np.linalg.eig() function computes the eigenvalues and eigenvectors of the matrix. Eigenvalues indicate the magnitude of scaling along each eigenvector direction −

```
import numpy as np

# Compute eigenvalues and eigenvectors
matrix = np.array([[4, -2], [1,  1]])

eigvals, eigvecs = np.linalg.eig(matrix)
print("Eigenvalues:", eigvals)
print("Eigenvectors:", eigvecs)
```

Following is the output obtained −

```
Eigenvalues: [3. 2.]
Eigenvectors: 
[[0.89442719 0.70710678]
 [0.4472136  0.70710678]]
```

## Singular Value Decomposition (SVD)

SVD is a factorization method for matrices that generalizes the eigendecomposition of a square matrix to any
**m x n**matrix. We can achieve this in NumPy using the numpy.linalg.svd() function.
Eigendecomposition is the process of breaking a matrix down into its eigenvalues and eigenvectors. These eigenvalues represent the scaling factor, while the eigenvectors show the directions in which the matrix stretches or compresses.

### Example

In the following example, we are performing singular value decomposition on a "2x2" matrix using np.linalg.svd() function. The result includes the U matrix, singular values, and the V matrix, which together represent the original matrix −

```
import numpy as np

matrix_a = np.matrix('1 2; 3 4')

# Performing SVD
U, S, V = np.linalg.svd(matrix_a)
print("U Matrix:\n", U)
print("Singular Values:\n", S)
print("V Matrix:\n", V)
```

This will produce the following result −

```
U Matrix:
[[-0.40455358 -0.9145143 ]
 [-0.9145143   0.40455358]]
Singular Values:
[5.4649857  0.36596619]
V Matrix:
[[-0.57604844 -0.81741556]
 [ 0.81741556 -0.57604844]]
```

---

## 48. NumPy - Linear Algebra

*Source: [https://www.tutorialspoint.com/numpy/numpy_linear_algebra.htm](https://www.tutorialspoint.com/numpy/numpy_linear_algebra.htm)*

---

---
[Previous](/numpy/numpy_matrix_library.htm)[Quiz](/numpy/quiz_on_numpy_linear_algebra.htm)[Next](/numpy/numpy_matplotlib.htm)
## Linear Algebra in NumPy

Linear algebra is a branch of mathematics that deals with vectors, matrices, and linear transformations.

NumPy package contains
**numpy.linalg**module that provides all the functionality required for linear algebra. Some of the important functions in this module are described in the following table.Sr.No.Function & Description1[dot](/numpy/numpy_dot.htm)
Dot product of the two arrays
2[vdot](/numpy/numpy_vdot.htm)
Dot product of the two vectors
3[inner](/numpy/numpy_inner.htm)
Inner product of the two arrays
4[matmul](/numpy/numpy_matmul.htm)
Matrix product of the two arrays
5[determinant](/numpy/numpy_determinant.htm)
Computes the determinant of the array
6[solve](/numpy/numpy_solve.htm)
Solves the linear matrix equation
7[inv](/numpy/numpy_inv.htm)
Finds the multiplicative inverse of the matrix

## Creating Matrices

In NumPy, we can create matrices using arrays. Matrices are simply two-dimensional arrays, and they can be created using the np.array() function. You can specify the elements of the matrix as nested lists.

### Example

Following is the basic example where we create a matrix that consists two rows and three columns −

```
import numpy as np

# Creating a 2x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Matrix:\n", matrix)
```

Following is the output obtained −

```
Matrix:
[[1 2 3]
 [4 5 6]]
```

## Matrix Operations

Matrix operations are fundamental in linear algebra and involve performing arithmetic on matrices. In NumPy, you can easily perform addition, subtraction, multiplication, and transposition of matrices.

### Matrix Addition and Subtraction

Matrix addition and subtraction are performed element-wise. This means corresponding elements from each matrix are added or subtracted. Both matrices must have the same shape for these operations.
**Example**
In the following example, we are performing element-wise matrix addition and subtraction using two 2x2 NumPy arrays,
**A**and**B**−
```
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix Addition
C = A + B
print("Matrix Addition:\n", C)

# Matrix Subtraction
D = A - B
print("Matrix Subtraction:\n", D)
```

This will produce the following result −

```
Matrix Addition:
[[ 6  8]
 [10 12]]
Matrix Subtraction:
[[-4 -4]
 [-4 -4]]
```

### Matrix Multiplication

Matrix multiplication can be done using the
**@**operator or the**np.dot()**function. Unlike element-wise multiplication, matrix multiplication involves summing the products of rows and columns.**Example**
Here, we are performing matrix multiplication on two 2x2 NumPy arrays,
**A**and**B**, using the**@**operator and the**np.dot()**function −
```
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix Multiplication
C = A @ B
print("Matrix Multiplication with @:\n", C)

D = np.dot(A, B)
print("Matrix Multiplication with np.dot():\n", D)
```

Following is the output of the above code −

```
Matrix Multiplication with @:
[[19 22]
 [43 50]]
Matrix Multiplication with np.dot():
[[19 22]
 [43 50]]
```

### Matrix Transposition

The transpose of a matrix is obtained by flipping it over its diagonal, effectively swapping rows with columns. This can be achieved using the
**.T**attribute.**Example**
Over here, we are transposing a 2x2 NumPy array
**A**to obtain its transpose A−
```
import numpy as np
A = np.array([[1, 2], [3, 4]])

# Transposing the matrix
A_T = A.T
print("Transpose of A:\n", A_T)
```

The output obtained is as shown below −

```
Transpose of A:
[[1 3]
 [2 4]]
```

## Determinants & Inverses

The determinant indicates whether a matrix is invertible (non-singular) or not. If the determinant of a matrix is non-zero, the matrix is invertible. Conversely, if the determinant is zero, the matrix is singular and not invertible. It is also used to −

- Solve the linear equations.
- Change variables in integrals.
- Calculate area and volume.
- Define characteristic polynomial of a square matrix.
The
**inverse**of a matrix is a matrix that, when multiplied by the original, results in the identity matrix.
```
A X A-1 = A-1 = I
```
= A= I
NumPy Provides various functions to calculate the determinant and inverse of a matrix.

### Computing the Determinant

We can calculate the determinant of a matrix using the
**linalg.det()**function. It internally uses LAPACK routine to calculate the determinant (via LU factorization).**Example**
In the example below, we are calculating the determinant of a 2x2 NumPy array
**A**using the np.linalg.det() function −
```
import numpy as np
A = np.array([[1, 2], [3, 4]])

# Determinant of the matrix
det = np.linalg.det(A)
print("Determinant of A:", det)
```

After executing the above code, we get the following output −

```
Determinant of A: -2.0000000000000004
```

### Computing the Inverse

We can compute the inverse of a matrix using the
**linalg.inv()**function.**Example**
Here, we are calculating the inverse of a 2x2 NumPy array
**A**using the np.linalg.inv() function −
```
import numpy as np
A = np.array([[1, 2], [3, 4]])

# Inverse of the matrix
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)
```

The result produced is as follows −

```
Inverse of A:
 [[-2.   1. ]
 [ 1.5 -0.5]]
```

## Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are fundamental in understanding linear transformations. They can be computed using the
**np.linalg.eig()**function.
Eigenvalues indicate the magnitude of the transformation, while eigenvectors indicate the direction.

### Example

In the example shown below, we are computing the eigenvalues and eigenvectors of a 2x2 NumPy array
**A**using the np.linalg.eig() function −
```
import numpy as np
A = np.array([[1, 2], [3, 4]])

# Computing eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

We get the output as shown below −

```
Eigenvalues: [-0.37228132  5.37228132]
Eigenvectors:
[[-0.82456484 -0.41597356]
 [ 0.56576746 -0.90937671]]
```

## Solving Linear Systems

In Numpy, linear systems of equations can be solved using the
**np.linalg.solve()**function. This function finds the values of variables that satisfy the linear equations represented by the matrix equation −
```
Ax = b
```

Where,
**A**represents matrix and**b**is a vector.
### Example

In this example, we are solving a linear system of equations represented by the matrix equation
**Ax=b**, where**A**is a 2x2 matrix and**b**is a vector. We use the np.linalg.solve() function to compute the values of**x**that satisfy the equation −
```
import numpy as np
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# Solving the linear system Ax = b
x = np.linalg.solve(A, b)
print("Solution of the linear system:", x)
```

Following is the output obtained −

```
Solution of the linear system: [2. 3.]
```

## Singular Value Decomposition (SVD)

SVD is a factorization of a matrix into three matrices:
**U**(the left singular vectors),**S**(the singular values) and**V**(the right singular vectors). It is useful in various applications, including signal processing and statistics.
You can perform SVD using the
**np.linalg.svd()**function in NumPy.
### Example

In the following example, we are performing Singular Value Decomposition (SVD) on a 2x2 matrix
**A**, which decomposes it into three components: U, S, and V −
```
import numpy as np
A = np.array([[1, 2], [3, 4]])

# Performing SVD
U, S, V = np.linalg.svd(A)
print("U matrix:\n", U)
print("Sigma values:", S)
print("V matrix:\n", V)
```

This will produce the following result −

```
U matrix:
[[-0.40455358 -0.9145143 ]
 [-0.9145143   0.40455358]]
Sigma values: [5.4649857  0.36596619]
V matrix:
[[-0.57604844 -0.81741556]
 [ 0.81741556 -0.57604844]]
```

## Norms and Conditions
**Norms**measure the size or length of vectors and matrices, helping quantify their magnitude.**Condition**numbers indicate how sensitive a matrix's solution is to changes in its input, indicating how well it can be solved numerically.
### Computing Norms

Norms measure the size or length of vectors and matrices. We can use the NumPy
**linalg.norm()**function to compute different types of norms, such as the Frobenius norm and the Euclidean norm.**Example**
In the following example, we are calculating the Frobenius norm of a 2x2 matrix
**A**, which provides a measure of its overall magnitude, similar to the Euclidean norm for vectors.
We also compute the L2 norm (Euclidean norm) of a 3D vector, which quantifies its length in space −

```
import numpy as np
A = np.array([[1, 2], [3, 4]])

# Frobenius norm
norm = np.linalg.norm(A, 'fro')
print("Frobenius norm of A:", norm)

# L2 norm (Euclidean norm)
vector = np.array([1, 2, 3])
l2_norm = np.linalg.norm(vector)
print("L2 norm of vector:", l2_norm)
```

Following is the output of the above code −

```
Frobenius norm of A: 5.477225575051661
L2 norm of vector: 3.7416573867739413
```

### Computing Conditions

The condition number of a matrix measures how sensitive the solution of a linear system is to errors in the data.

It can be computed using the NumPy
**linalg.cond()**function. A high condition number indicates that the matrix is close to singular, making it more challenging to solve linear equations accurately.**Example**
Here, we are calculating the condition number of a 2x2 matrix
**A**−
```
import numpy as np
A = np.array([[1, 2], [3, 4]])

# Condition number
cond = np.linalg.cond(A)
print("Condition number of A:", cond)
```

The output obtained is as shown below −

```
Condition number of A: 14.933034373659268
```

---

## 49. NumPy - Matplotlib

*Source: [https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm](https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm)*

---

---
[Previous](/numpy/numpy_linear_algebra.htm)[Quiz](/numpy/quiz_on_numpy_matplotlib.htm)[Next](/numpy/numpy_histogram_using_matplotlib.htm)
## NumPy and Matplotlib

NumPy is a Python library for numerical computing, providing support for arrays, mathematical functions, and efficient operations on large datasets.

Matplotlib is a Python library for creating static, interactive, and animated visualizations like plots and charts.

They are often used together, as NumPy generates and processes data arrays, while Matplotlib visualizes them. For example, you can use NumPy to create data points and Matplotlib to plot them as graphs.

## What is Matplotlib?

Matplotlib is a Python library used to create high-quality plots and charts. It is highly customizable and can produce various types of plots, such as line plots, scatter plots, bar plots, and histograms.

Matplotlib works seamlessly with NumPy, making it easy to visualize numerical data arrays or perform operations before plotting the results.

## Setting Up Matplotlib

Before starting with Matplotlib, ensure you have the library installed. You can install it using pip as shown below −

```
# Install Matplotlib
!pip install matplotlib
```

Once installed, you can import it alongside NumPy to begin creating visualizations as shown below −

```
import numpy as np
import matplotlib.pyplot as plt
```

Now that the libraries are ready, let us dive into various types of visualizations you can create using Matplotlib and NumPy.

## Line Plot

A line plot is one of the simplest and most commonly used visualizations. It is used to show trends or relationships between data points.

### Example

In the following example, we are creating a line plot using Matplotlib and NumPy −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate data using NumPy
# 100 evenly spaced points between 0 and 10
x = np.linspace(0, 10, 100)  
# Compute sine values for x
y = np.sin(x)  

# Create a line plot
plt.plot(x, y, label='sin(x)', color='blue', linestyle='--')
plt.title('Line Plot of sin(x)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
```

We can see a line plot displaying a sine wave with labeled axes, a legend, and a grid for better visualization −
![Line Plot](/numpy/images/numpy_matplotlib_line_plot.jpg)
## Scatter Plot

Scatter plots are used to display relationships between two variables by showing individual data points. They are useful for identifying patterns, correlations, or outliers in data.

### Example

Following is an example of creating a scatter plot showing random points with a transparent green color −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
x = np.random.rand(50)
y = np.random.rand(50)

# Create a scatter plot
plt.scatter(x, y, color='green', alpha=0.7)
plt.title('Scatter Plot of Random Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

We get the output as shown below −
![Scatter Plot](/numpy/images/numpy_matplotlib_scatter_plot.jpg)
## Bar Plot

Bar plots are used to compare different categories or groups. They display data using rectangular bars, where the height represents the value.

### Example

Following is an example of creating a bar plot showing the values for each category with orange-colored bars −

```
import numpy as np
import matplotlib.pyplot as plt

# Data for bar plot
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]

# Create a bar plot
plt.bar(categories, values, color='orange')
plt.title('Bar Plot Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
```

The output obtained is as shown below −
![Bar Plot](/numpy/images/numpy_matplotlib_bar_plot.jpg)
## Histogram

Histograms are used to visualize the frequency distribution of a dataset. They divide data into intervals (bins) and show the count of data points in each interval.

### Example

Following is an example of creating a histogram displaying the frequency distribution of the randomly generated data −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
# 1000 samples from a normal distribution
data = np.random.randn(1000)  

# Create a histogram
plt.hist(data, bins=30, color='purple', alpha=0.8)
plt.title('Histogram of Random Data')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.show()
```

After executing the above code, we get the following output −
![Histogram](/numpy/images/numpy_matplotlib_histogram.jpg)
## Pie Chart

Pie charts are used to represent data as slices of a circle, showing proportions or percentages. While not always the best for precise comparisons, they can be visually appealing for specific use cases.

### Example

Following is an example of creating a pie chart showing the proportion of each programming language in percentages −

```
import numpy as np
import matplotlib.pyplot as plt

# Data for pie chart
labels = ['Python', 'Java', 'C++', 'Ruby']
sizes = [50, 30, 15, 5]

# Create a pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart Example')
plt.show()
```

The result produced is as follows −
![Pie Chart](/numpy/images/numpy_matplotlib_pie_chart.jpg)
## Customizing Matplotlib Visualizations

Matplotlib offers extensive options to customize your visualizations. You can adjust colors, line styles, markers, fonts, and more.

### Example

Following is an example of creating a customized plot with different line styles, colors, and markers for sine and cosine waves −

```
import numpy as np
import matplotlib.pyplot as plt

# Customized line plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin(x)', color='red', linewidth=2, marker='o')
plt.plot(x, y2, label='cos(x)', color='blue', linestyle='dotted')
plt.title('Customized Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True, linestyle='--', color='gray', alpha=0.6)
plt.show()
```

We get the output as shown below −
![Customized NumPy Matplotlib Visuals](/numpy/images/custom_numpy_matplotlib.jpg)
## Combining NumPy and Matplotlib

Using NumPy and Matplotlib together can enhance your analysis and visualization workflow. NumPy can be used to preprocess and manipulate data, while Matplotlib can be used to visualize the results.

For instance, you can generate data transformations or statistical analyses using NumPy and then visualize them with Matplotlib.

### Example

Following is an example of creating a plot showing a parabolic curve representing the quadratic function −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate and visualize a quadratic function
x = np.linspace(-10, 10, 200)
y = x**2 - 5*x + 6

plt.plot(x, y, label='y = x^2 - 5x + 6', color='magenta')
plt.title('Quadratic Function Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
```

The output obtained is as shown below −
![NumPy Matplotlib](/numpy/images/numpy_matplotlib.jpg)

---

## 50. NumPy - Histogram Using Matplotlib

*Source: [https://www.tutorialspoint.com/numpy/numpy_histogram_using_matplotlib.htm](https://www.tutorialspoint.com/numpy/numpy_histogram_using_matplotlib.htm)*

---

---
[Previous](/numpy/numpy_matplotlib.htm)[Quiz](/numpy/quiz_on_numpy_histogram_using_matplotlib.htm)[Next](/numpy/numpy_sorting_arrays.htm)
NumPy has a
**numpy.histogram()**function that is a graphical representation of the frequency distribution of data. Rectangles of equal horizontal size corresponding to class interval called**bin**and**variable height**corresponding to frequency.
## numpy.histogram()

The numpy.histogram() function takes the input array and bins as two parameters. The successive elements in bin array act as the boundary of each bin.

```
import numpy as np 
   
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
np.histogram(a,bins = [0,20,40,60,80,100]) 
hist,bins = np.histogram(a,bins = [0,20,40,60,80,100]) 
print hist 
print bins
```

It will produce the following output −

```
[3 4 5 2 1]
[0 20 40 60 80 100]
```

## plt()

Matplotlib can convert this numeric representation of histogram into a graph. The
**plt() function**of pyplot submodule takes the array containing the data and bin array as parameters and converts into a histogram.
```
from matplotlib import pyplot as plt 
import numpy as np  
   
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
plt.hist(a, bins = [0,20,40,60,80,100]) 
plt.title("histogram") 
plt.show()
```

It should produce the following output −
![Histogram Plot](/numpy/images/histogram_plot.jpg)

---

## 51. NumPy - Sorting Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_sorting_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_sorting_arrays.htm)*

---

---

## 52. NumPy - Sorting Along an Axis

*Source: [https://www.tutorialspoint.com/numpy/numpy_sorting_along_an_axis.htm](https://www.tutorialspoint.com/numpy/numpy_sorting_along_an_axis.htm)*

---

---
[Previous](/numpy/numpy_sorting_arrays.htm)[Quiz](/numpy/quiz_on_numpy_sorting_along_an_axis.htm)[Next](/numpy/numpy_sorting_with_fancy_indexing.htm)
## Sorting Along an Axis in NumPy

In NumPy, arrays can be multi-dimensional, and sorting can be performed along any of these dimensions (axes). Sorting along an axis means arranging the elements of the array in a specific order based on the values along that axis.

In NumPy, the different axes are:
**0th axis**(rows),**1st axis**(columns), and**higher axes**(depth or additional dimensions).
## The np.sort() Function

The np.sort() function sorts the elements of an array and returns a new array containing the sorted elements. Sorting can be done along a specified axis, or if no axis is specified, the function defaults to sorting along the last axis. Following is the syntax −

```
numpy.sort(a, axis=-1, kind=None, order=None)
```

Where,

- **a:**It is the array to be sorted.
- **axis:**It is the axis along which to sort. Default is -1, which means sorting along the last axis.
- **kind:**It is the sorting algorithm to use. Options include 'quicksort', 'mergesort', 'heapsort', and 'stable'.
- **order:**It is used when sorting a structured array to define which fields to compare.
## Sorting Along Axis 0 (Rows)

Sorting along axis 0 in NumPy refers to sorting the array elements along the vertical axis, which corresponds to rows in a 2D array.

This operation is useful when you want to order the rows of a matrix based on the values in each column.

> In a 2D array −
>  
> Axis 0 corresponds to the rows.
> Axis 1 corresponds to the columns.

In a 2D array −

- Axis 0 corresponds to the rows.
- Axis 1 corresponds to the columns.
When sorting along axis 0, NumPy sorts the elements column-wise. For each column, the values are arranged in ascending order, and the rows are reordered accordingly.

### Example

In the following example, we are using the sort() function to sort the elements in each column −

```
import numpy as np

# Create a 2D array
arr = np.array([[3, 6, 4], [5, 1, 2]])

# Sort along axis 0 (rows)
sorted_arr_axis0 = np.sort(arr, axis=0)

print("Original Array:\n", arr)
print("Sorted Along Axis 0:\n", sorted_arr_axis0)
```

The output will have the rows reordered such that the values in each column are sorted in ascending order −

```
Original Array:
[[3 6 4]
 [5 1 2]]
Sorted Along Axis 0:
 [[3 1 2]
 [5 6 4]]
```

## Sorting Along Axis 1 (Columns)

Sorting along axis 1 in NumPy refers to sorting the array elements along the horizontal axis, which corresponds to columns in a 2D array. This operation is useful when you want to order the elements in each row based on their values.

When sorting along axis 1, NumPy sorts the elements row-wise. For each row, the values are arranged in ascending order, and the order of columns is adjusted accordingly.

### Example

In this example, we are using the sort() function to sort the elements in each row −

```
import numpy as np

# Create a 2D array
arr = np.array([[3, 2, 1], [6, 5, 4]])

# Sort along axis 1 (columns)
sorted_arr_axis1 = np.sort(arr, axis=1)

print("Original Array:\n", arr)
print("Sorted Along Axis 1:\n", sorted_arr_axis1)
```

The output will have the columns reordered within each row such that the values are sorted in ascending order −

```
Original Array:
[[3 2 1]
 [6 5 4]]
Sorted Along Axis 1:
 [[1 2 3]
 [4 5 6]]
```

## Sorting Multi-dimensional Arrays

Sorting multi-dimensional arrays in NumPy involves organizing the elements of the array along one or more specific axes.

Sorting along specific axes in a multi-dimensional array is similar to sorting in 2D arrays but extends to higher dimensions. Here is how sorting works along different axes −

- **Axis 0 (First Dimension):**Sorting along axis 0 affects rows.
- **Axis 1 (Second Dimension):**Sorting along axis 1 affects columns.
- **Axis 2 (Third Dimension):**In 3D arrays, sorting along axis 2 affects the depth slices.
### Example

In the example below, we are sorting a 3D array along different axes: depth (axis 0), rows (axis 1), and columns (axis 2). Each sorting operation arranges the elements within the specified dimension, resulting in differently ordered arrays −

```
import numpy as np

# Create a 3D array
arr = np.array([[[3, 2, 1], [6, 5, 4]], [[9, 8, 7], [12, 11, 10]]])

# Sort along axis 0 (depth)
sorted_arr_axis0 = np.sort(arr, axis=0)

# Sort along axis 1 (rows)
sorted_arr_axis1 = np.sort(arr, axis=1)

# Sort along axis 2 (columns)
sorted_arr_axis2 = np.sort(arr, axis=2)

print("Original Array:\n", arr)
print("Sorted Along Axis 0:\n", sorted_arr_axis0)
print("Sorted Along Axis 1:\n", sorted_arr_axis1)
print("Sorted Along Axis 2:\n", sorted_arr_axis2)
```

Following is the output of the above code −

```
Original Array:
[[[ 3  2  1]
  [ 6  5  4]]

 [[ 9  8  7]
  [12 11 10]]]
Sorted Along Axis 0:
 [[[ 3  2  1]
  [ 6  5  4]]

 [[ 9  8  7]
  [12 11 10]]]
Sorted Along Axis 1:
 [[[ 3  2  1]
  [ 6  5  4]]

 [[ 9  8  7]
  [12 11 10]]]
Sorted Along Axis 2:
[[[ 1  2  3]
  [ 4  5  6]]

 [[ 7  8  9]
  [10 11 12]]]
```

## Sorting with Different Algorithms

The NumPy sort() function supports several sorting algorithms, each with its own strengths −

- Quicksort
- Mergesort
- Heapsort
- Stable Sort
These algorithms can be chosen by setting the
**kind**parameter in the np.sort() function. By default, NumPy uses the "quicksort" algorithm.
> Sorting algorithms are methods for arranging elements in a specific order. Different algorithms have different performance characteristics and are optimized for various types of data and array sizes.

### Example

In this example, the array is sorted using "quicksort" and "mergesort" algorithms along axis "1". Different sorting algorithms can affect performance and stability −

```
import numpy as np

# Create a 2D array
arr = np.array([[3, 2, 1], [6, 5, 4]])

# Sort using different algorithms
sorted_quicksort = np.sort(arr, axis=1, kind='quicksort')
sorted_mergesort = np.sort(arr, axis=1, kind='mergesort')

print("Sorted with Quicksort:\n", sorted_quicksort)
print("Sorted with Mergesort:\n", sorted_mergesort)
```

The output obtained is as shown below −

```
Sorted with Quicksort:
[[1 2 3]
 [4 5 6]]
Sorted with Mergesort:
[[1 2 3]
 [4 5 6]]
```

---

## 53. NumPy - Sorting with Fancy Indexing

*Source: [https://www.tutorialspoint.com/numpy/numpy_sorting_with_fancy_indexing.htm](https://www.tutorialspoint.com/numpy/numpy_sorting_with_fancy_indexing.htm)*

---

---
[Previous](/numpy/numpy_sorting_along_an_axis.htm)[Quiz](/numpy/quiz_on_numpy_sorting_with_fancy_indexing.htm)[Next](/numpy/numpy_structured_arrays.htm)
## Sorting with Fancy Indexing in NumPy

Sorting with fancy indexing in NumPy involves using an array of indices to rearrange elements in another array. Fancy indexing allows you to sort or reorder an array based on specific indices, giving you more control over how you want the data to be organized.

For example, you can first determine the order of elements with argsort() function and then use these indices to sort the original array. This is useful for complex data manipulation tasks where you need precise control over the order of elements.

## Basic Fancy Indexing

Basic fancy indexing involves using one or more arrays of indices to select elements from a NumPy array. These indices can be arrays of integers or boolean values, allowing for non-contiguous selection of elements. This is especially useful for reordering or selecting multiple elements simultaneously.

### Example

In the following example, we are using fancy indexing to reorder an array based on a specified set of indices. By indexing "arr" with the "indices" array, we obtain a new array where the elements are arranged according to the given order −

```
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
indices = np.array([4, 2, 3, 1, 0])
sorted_arr = arr[indices]

print("Original Array:", arr)
print("Sorted Array with Fancy Indexing:", sorted_arr)
```

Following is the output obtained −

```
Original Array: [10 20 30 40 50]
Sorted Array with Fancy Indexing: [50 30 40 20 10]
```

## Sorting Arrays Using Fancy Indexing

Sorting arrays with fancy indexing involves first creating an array of indices that represent the desired order of elements. These indices are then used to reorder the array elements according to specific criteria.

Fancy indexing provides a way to perform advanced sorting operations that go beyond simple in-place sorting.

### Example

In this example, we are using np.argsort() function to retrieve the indices that would sort the array "arr". These indices are then used to rearrange "arr" into a sorted array −

```
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9])
# Get indices that would sort the array
sort_order = np.argsort(arr)  
sorted_arr = arr[sort_order]

print("Original Array:", arr)
print("Sorted Array:", sorted_arr)
```

This will produce the following result −

```
Original Array: [3 1 4 1 5 9]
Sorted Array: [1 1 3 4 5 9]
```

## Fancy Indexing with Multi-dimensional Arrays

In multi-dimensional arrays, fancy indexing can be applied to reorder or select elements along specific axes.

By using arrays of indices, you can achieve various operations, such as sorting elements, rearranging rows or columns, and extracting specific data points.

### Example: Fancy Indexing in 2D Arrays

In the example below, we are reordering the rows and columns of a 2D array using fancy indexing. By specifying "row_indices" and "col_indices", we rearrange the arrays rows and columns to produce a new reordered array −

```
import numpy as np

arr = np.array([[10, 20, 30], [40, 50, 60]])
row_indices = np.array([1, 0])
col_indices = np.array([2, 1, 0])

# Reorder rows and columns
reordered_arr = arr[row_indices][:, col_indices]

print("Original Array:\n", arr)
print("Reordered Array:\n", reordered_arr)
```

Following is the output of the above code −

```
Original Array:
[[10 20 30]
 [40 50 60]]
Reordered Array:
 [[60 50 40]
 [30 20 10]]
```

### Example: Fancy Indexing in 3D Arrays

In a 3D array, fancy indexing can be used to reorder slices along the first axis. Here, by specifying "slice_indices", we rearrange the arrays slices to produce a new array with reordered slices −

```
import numpy as np

# Define a 3D NumPy array with shape (3, 2, 2)
arr = np.array([[[1, 2], [3, 4]], 
                [[5, 6], [7, 8]], 
                [[9, 10], [11, 12]]])

# Define an array of indices to reorder slices along the first axis
slice_indices = np.array([2, 0, 1])

# Reorder the slices using fancy indexing
reordered_slices = arr[slice_indices]

print("Original Array:\n", arr)
print("Reordered Slices:\n", reordered_slices)
```

The output obtained is as shown below −

```
Original Array:
[[[ 1  2]
  [ 3  4]]

 [[ 5  6]
  [ 7  8]]

 [[ 9 10]
  [11 12]]]
Reordered Slices:
[[[ 9 10]
  [11 12]]

 [[ 1  2]
  [ 3  4]]

 [[ 5  6]
  [ 7  8]]]
```

## Sorting with Fancy Indexing and Structured Arrays

Fancy indexing can also be applied to structured arrays, where you sort based on the values of specific fields.

Structured arrays in NumPy are arrays with compound data types, where each element can have multiple fields with different data types. These are useful for managing heterogeneous data in a single array.

### Example

In the following example, we are sorting a structured NumPy array based on the "age" field. By using np.argsort() function on the age field, we obtain the indices to reorder the array, resulting in a new array sorted by age −

```
import numpy as np

dtype = [('name', 'S10'), ('age', 'i4')]
values = [('Alice', 25), ('Bob', 30), ('Charlie', 20)]
arr = np.array(values, dtype=dtype)

# Sort by age
sorted_indices = np.argsort(arr['age'])
sorted_arr = arr[sorted_indices]

print("Original Structured Array:\n", arr)
print("Sorted by Age:\n", sorted_arr)
```

We get the output as shown below −

```
Original Structured Array:
[(b'Alice', 25) (b'Bob', 30) (b'Charlie', 20)]
Sorted by Age:
[(b'Charlie', 20) (b'Alice', 25) (b'Bob', 30)]
```

---

## 54. NumPy - Structured Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_structured_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_structured_arrays.htm)*

---

---
[Previous](/numpy/numpy_sorting_with_fancy_indexing.htm)[Quiz](/numpy/quiz_on_numpy_structured_arrays.htm)[Next](/numpy/numpy_creating_structured_arrays.htm)
## Structured Arrays in NumPy

A structured array in NumPy is an array where each element is a compound data type. This compound data type can consist of multiple fields, each with its own data type, similar to a table or a record.

For example, you can have an array where each element holds both a name (as a string) and an age (as an integer). This helps you to work with complex data more flexibly, as you can access and manipulate each field separately.

## Creating Structured Arrays

The first step in creating a structured array is defining the data type (dtype) that specifies the structure of each element. The
**dtype**is defined as a list of tuples or a dictionary, where each tuple or dictionary entry defines a field name and its data type.
Following are the data types available in structured arrays −

- **'U10':**Unicode string of length 10
- **'i4':**4-byte integer
- **'f8':**8-byte floating point number
- **'b':**Boolean value
### Using a List of Tuples

You can define the dtype and create the structured array using a list of tuples, where each tuple represents a field. Each tuple contains two elements: the first element is the name of the field, and the second element is the data type of that field.
**Example**
In the following example, we are defining a structured array with fields for "name", "age", and "height" using a specified dtype. We then create this array with corresponding data −

```
import numpy as np

# Define the dtype
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]

# Define the data
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]

# Create the structured array
structured_array = np.array(data, dtype=dtype)

print("Structured Array:\n", structured_array)
```

Following is the output obtained −

```
Structured Array:
[('Alice', 30, 5.6) ('Bob', 25, 5.8) ('Charlie', 35, 5.9)]
```

### Using a Dictionary

Alternatively, you can define the data and dtype using a dictionary to clearly specify the names and types of fields. Each key in the dictionary represents a field name, and the value associated with each key defines the data type of that field.
**Example**
In this example, we are defining the dtype for a structured array using a dictionary format to specify fields for "name", "age", and "height". We then create and display this structured array with the corresponding data, organizing it into a format that supports multiple data types within each record −

```
import numpy as np

# Define the dtype using a dictionary
dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('height', 'f4')])

# Define the data
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]

# Create the structured array
structured_array = np.array(data, dtype=dtype)

print("Structured Array from Dictionary:\n", structured_array)
```

This will produce the following result −

```
Structured Array from Dictionary:
[('Alice', 30, 5.6) ('Bob', 25, 5.8) ('Charlie', 35, 5.9)]
```

## Accessing Fields in Structured Arrays

You can access individual fields in a structured array using field names. This is done by indexing the array with the field name as a string.

### Example: Accessing Individual Fields

In the example below, we are defining a structured array with fields for 'name', 'age', and 'height', and then accessing each of these fields separately −

```
import numpy as np

# Define a dtype and data for a structured array
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]
structured_array = np.array(data, dtype=dtype)

# Access the 'name' field
names = structured_array['name']
print("Names:", names)

# Access the 'age' field
ages = structured_array['age']
print("Ages:", ages)

# Access the 'height' field
heights = structured_array['height']
print("Heights:", heights)
```

Following is the output of the above code −

```
Names: ['Alice' 'Bob' 'Charlie']
Ages: [30 25 35]
Heights: [5.6 5.8 5.9]
```

### Example: Accessing Rows

You can access specific rows of the structured array using indexing. This allows you to retrieve complete records. Here, we retrieve the first and second rows of the structured array −

```
import numpy as np

# Define a dtype and data for a structured array
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]
structured_array = np.array(data, dtype=dtype)

# Access the first row
first_row = structured_array[0]
print("First Row:", first_row)

# Access the second row
second_row = structured_array[1]
print("Second Row:", second_row)
```

Following is the output of the above code −

```
First Row: ('Alice', 30, 5.6)
Second Row: ('Bob', 25, 5.8)
```

## Modifying Fields of Structured Arrays

You can modify the values of individual fields in the structured array by indexing and assigning new values to them.

To add new fields to a structured array, you can use a combination of np.concatenate() function and creating a new dtype that includes the additional fields.

> NumPy does not support adding fields directly to an existing structured array.

### Example: Updating Fields

In the example below, we are updating the 'age' field of the first record in a structured array by directly assigning a new value −

```
import numpy as np

# Define a dtype and data for a structured array
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]
structured_array = np.array(data, dtype=dtype)

# Update the 'age' of the first record
structured_array[0]['age'] = 31
print("Updated Structured Array:\n", structured_array)
```

The output obtained is as shown below −

```
Updated Structured Array:
[('Alice', 31, 5.6) ('Bob', 25, 5.8) ('Charlie', 35, 5.9)]
```

### Example: Adding New Fields

Here, we are extending a structured array by adding a new field, 'weight', to its dtype and updating the data to include this field −

```
import numpy as np

# Define a dtype and data for the original structured array
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]
structured_array = np.array(data, dtype=dtype)

# Define a new dtype with an additional field 'weight'
new_dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4'), ('weight', 'f4')]

# Define new data including the additional field
new_data = [('Alice', 30, 5.6, 55.0), ('Bob', 25, 5.8, 70.0), ('Charlie', 35, 5.9, 80.0)]

# Create a new structured array with the additional field
new_structured_array = np.array(new_data, dtype=new_dtype)
print("New Structured Array with Additional Field:\n", new_structured_array)
```

After executing the above code, we get the following output −

```
New Structured Array with Additional Field:
 [('Alice', 30, 5.6, 55.) ('Bob', 25, 5.8, 70.) ('Charlie', 35, 5.9, 80.)]
```

## Sorting Structured Arrays

Sorting structured arrays in NumPy means arranging the elements of an array based on the values of one or more fields.

Since structured arrays have multiple fields, sorting can be based on the values in these fields. For example, you might sort an array of people by their age or height.

### Example

In the following example, we are sorting a structured array based on the 'age' field by first obtaining the indices that would arrange the ages in ascending order. We then use these indices to reorder the entire array −

```
import numpy as np

# Define a structured array
dtype = [('name', 'U10'), ('age', 'i4')]
data = [('Alice', 30), ('Bob', 25), ('Charlie', 35)]
structured_array = np.array(data, dtype=dtype)

# Sort the array by 'age'
sorted_indices = np.argsort(structured_array['age'])
sorted_array = structured_array[sorted_indices]
print("Sorted by Age:\n", sorted_array)
```

The result produced is as follows −

```
Sorted by Age:
[('Bob', 25) ('Alice', 30) ('Charlie', 35)]
```

## Filtering Structured Arrays

Filtering structured arrays involves applying conditions to one or more fields and retrieving elements that satisfy these conditions.

This is useful when you want to retrieve records that meet certain criteria, such as extracting all entries where a specific field exceeds a threshold or matches a certain value.

### Example

In this example, we are filtering a structured array to include only the records where the 'age' field is greater than 30 −

```
import numpy as np

# Define a structured array
dtype = [('name', 'U10'), ('age', 'i4')]
data = [('Alice', 30), ('Bob', 25), ('Charlie', 35)]
structured_array = np.array(data, dtype=dtype)

# Filter array for ages greater than 30
filtered_array = structured_array[structured_array['age'] > 30]
print("Filtered Array (Age > 30):\n", filtered_array)
```

We get the output as shown below −

```
Filtered Array (Age > 30):[('Charlie', 35)]
```

## Combining Structured Arrays

Combining structured arrays involves merging or concatenating arrays that have a defined dtype with named fields. In NumPy, this can be done using the np.concatenate() function.

### Example

In the example below, we are combining two structured arrays with the same dtype into a single array using np.concatenate() function −

```
import numpy as np

# Define two structured arrays
dtype = [('name', 'U10'), ('age', 'i4')]
data1 = [('Alice', 30), ('Bob', 25)]
data2 = [('Charlie', 35), ('Dave', 40)]
structured_array1 = np.array(data1, dtype=dtype)
structured_array2 = np.array(data2, dtype=dtype)

# Combine the arrays
combined_array = np.concatenate((structured_array1, structured_array2))
print("Combined Structured Array:\n", combined_array)
```

This results in a new structured array that includes all the records from both original arrays as shown below −

```
Combined Structured Array:
[('Alice', 30) ('Bob', 25) ('Charlie', 35) ('Dave', 40)]
```

---

## 55. NumPy - Creating Structured Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_creating_structured_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_creating_structured_arrays.htm)*

---

---
[Previous](/numpy/numpy_structured_arrays.htm)[Quiz](/numpy/quiz_on_numpy_creating_structured_arrays.htm)[Next](/numpy/numpy_manipulating_structured_arrays.htm)
## Introduction to Structured Arrays

Structured arrays in NumPy allow for the representation of data with different types and sizes in a single array. Each element in a structured array can be a record with multiple fields, each field having its own data type.

This is similar to having a table where each row represents a record with various attributes. The key points are as follows −

- Each element (record) in a structured array can have multiple fields.
- Fields can have different data types (e.g., integers, floats, strings).
- Structured arrays are useful for representing complex data structures.
## Defining a Structured Array

The first step for creating a structured array is to define the data type for the structured array. This is done using a NumPy
**dtype**object, which specifies the names and types of the fields in the array as shown in the following example −
```
import numpy as np

# Define the data type for the structured array
dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('height', 'f4')])
```

Here,
**dtype**is a structured data type with three fields −
- **name:**A string of up to 10 characters ('U10')
- **age:**An integer ('i4')
- **height:**A floating-point number ('f4')
## Creating the Structured Array

Once you have defined the "dtype", you can create the structured array by passing the dtype to the np.array() function. Following is the syntax −

```
numpy.array(
   object, 
   dtype=None, 
   copy=True, 
   order='K', 
   subok=False, 
   ndmin=0
)
```

Where,

- **object:**This is the input data that you want to convert into a NumPy array. It can be a list, tuple, or any other sequence-like structure.
- **dtype (Optional):**This specifies the desired data type for the array. If not provided, NumPy will infer the data type from the input data. For example, you can use 'int32', 'float64', 'str', etc.
- **copy (Optional):**If True (default), a new array is created. If False, a new array is created only if necessary (i.e., if the input object is not already an array). If False, np.array may return a view of the original array if possible.
- **order (Optional):**This specifies the memory layout order. It can be 'C' for row-major (C-style) order, 'F' for column-major (Fortran-style) order, or 'K' for the order as found in the input. Default is 'K'.
- **subok (Optional):**If True, a subclass of ndarray will be used if the input is a subclass. Default is False, meaning the returned array will always be an instance of ndarray.
- **ndmin (Optional):**This specifies the minimum number of dimensions that the resulting array should have. For example, setting ndmin=2 ensures that the result is at least a 2-dimensional array.
### Example

In the following example, we are defining a structured array with a specified dtype that includes fields for "name", "age", and "height". We then create the array with data matching this structure and print the resulting structured array −

```
import numpy as np

# Define the dtype with field names and data types
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]

# Create data consistent with the dtype
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]

# Create the structured array
structured_array = np.array(data, dtype=dtype)
print("Structured Array:\n", structured_array)
```

Following is the output obtained −

```
Structured Array:
[('Alice', 30, 5.6) ('Bob', 25, 5.8) ('Charlie', 35, 5.9)]
```

## Create Structured Arrays with Different Data Types

You can create structured arrays with various data types, including strings, integers, and floats, depending on the needs of your application.

In the context of structured arrays −

- **Strings:**You can store text data in structured arrays. For example, a field for names might use a string data type.
- **Integers:**Numerical data, such as ages or counts, can be stored as integers. This might include data like age or quantity, which are whole numbers.
- **Floats:**For decimal numbers or real numbers, you can use float data types. This is useful for measurements or any value requiring precision, such as height or weight.
### Example

In the following example, we are creating a structured array with a data type (dtype) that includes mixed data types: integers for IDs, strings for names, and floating-point numbers for scores −

```
import numpy as np

# Define a dtype with mixed data types
dtype = [('id', 'i4'), ('name', 'U15'), ('score', 'f8')]
data = [(1, 'Alice', 88.5), (2, 'Bob', 91.2), (3, 'Charlie', 85.4)]

# Create the structured array
structured_array = np.array(data, dtype=dtype)
print("Structured Array with Mixed Data Types:\n", structured_array)
```

This will produce the following result −

```
Structured Array with Mixed Data Types:[(1, 'Alice', 88.5) (2, 'Bob', 91.2) (3, 'Charlie', 85.4)]
```

## Create Structured Arrays Using List of Tuples

You can define the dtype and create the structured array using a list of tuples, where each tuple represents a field. Each tuple contains two elements: the first element is the name of the field, and the second element is the data type of that field.
**Example**
In the following example, we are defining a structured array with fields for "name", "age", and "height" using a specified dtype. We then create this array with corresponding data −

```
import numpy as np

# Define the dtype
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]

# Define the data
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]

# Create the structured array
structured_array = np.array(data, dtype=dtype)

print("Structured Array:\n", structured_array)
```

Following is the output obtained −

```
Structured Array:
[('Alice', 30, 5.6) ('Bob', 25, 5.8) ('Charlie', 35, 5.9)]
```

## Create Structured Arrays Using Dictionary

Alternatively, you can define the data and dtype using a dictionary to clearly specify the names and types of fields. Each key in the dictionary represents a field name, and the value associated with each key defines the data type of that field.
**Example**
In this example, we are defining the dtype for a structured array using a dictionary format to specify fields for "name", "age", and "height". We then create and display this structured array with the corresponding data, organizing it into a format that supports multiple data types within each record −

```
import numpy as np

# Define the dtype using a dictionary
dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('height', 'f4')])

# Define the data
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]

# Create the structured array
structured_array = np.array(data, dtype=dtype)

print("Structured Array from Dictionary:\n", structured_array)
```

This will produce the following result −

```
Structured Array from Dictionary:
[('Alice', 30, 5.6) ('Bob', 25, 5.8) ('Charlie', 35, 5.9)]
```

---

## 56. NumPy - Manipulating Structured Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_manipulating_structured_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_manipulating_structured_arrays.htm)*

---

---
[Previous](/numpy/numpy_creating_structured_arrays.htm)[Quiz](/numpy/quiz_on_numpy_manipulating_structured_arrays.htm)[Next](/numpy/numpy_record_arrays.htm)
## Manipulating Structured Arrays in NumPy

Manipulating structured arrays in NumPy means modifying, rearranging, or working with the data in these arrays as per your requirement.

Structured arrays are special arrays where each element can have multiple fields (like name, age, height), and each field can have a different data type (like strings, integers, or floats).

In NumPy, you can manipulate structured arrays in several ways −

- Accessing and Modifying Fields
- Adding New Fields
- Deleting Fields
- Sorting Arrays
- Filtering Arrays
- Combining Arrays
- Reshaping Arrays
- Splitting Arrays
## Accessing and Modifying Fields

You can
**access**a specific field in a structured array by using the field name as a**key**. This is similar to how you access values in a dictionary. For example, if you have a structured array with fields like name, age, and height, you can access the age field to retrieve all the ages stored in the array.
Once you have accessed a field, you can also
**modify**its values. For instance, if you want to update someone's age in the array, you can do so by directly assigning a new value to the corresponding element in the age field.
### Example

In the following example, we are accessing and modifying the 'age' field in a structured array. Specifically, we update the age of the first element (Alice) from 30 to 31 and then retrieve the updated ages −

```
import numpy as np

# Define the dtype with field names and data types
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]

# Create the structured array with some initial data
data = [('Alice', 30, 5.6), ('Bob', 25, 5.8), ('Charlie', 35, 5.9)]
structured_array = np.array(data, dtype=dtype)

# Accessing the 'age' field
ages = structured_array['age']
print("Ages before modification:", ages)

# Modifying the 'age' field - let's update Alice's age to 31
structured_array['age'][0] = 31

# Accessing the 'age' field again to see the changes
print("Ages after modification:", structured_array['age'])
```

Following is the output obtained −

```
Ages before modification: [30 25 35]
Ages after modification: [31 25 35]
```

## Adding New Fields to Structured Arrays

To add a new field to an existing structured array, you need to create a new array with the additional field and copy the existing data over.

This process might be necessary when your data structure evolves and requires additional information.

### Example

In this example, we are expanding an existing structured array by adding a new field called 'Grade'. We copy the existing data into a new array with the additional field and then populate the new 'Grade' field with corresponding values −

```
import numpy as np

# Existing structured array
students = np.array([(1, 'Alice', 25), (2, 'Bob', 23), (3, 'Charlie', 35)],
                    dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4')])

# Define a new dtype with an additional field 'Grade'
new_dtype = [('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4'), ('Grade', 'f4')]

# Create a new structured array with the new dtype
students_with_grade = np.zeros(students.shape, dtype=new_dtype)

# Copy the old data
for field in students.dtype.names:
    students_with_grade[field] = students[field]

# Add data to the new 'Grade' field
students_with_grade['Grade'] = [85.5, 90.0, 88.0]

print(students_with_grade)
```

This will produce the following result −

```
[(1, 'Alice', 25, 85.5) (2, 'Bob', 23, 90. ) (3, 'Charlie', 35, 88. )]
```

## Deleting Fields from a Structured Array

To remove a field, you must create a new structured array with a modified
**dtype**that excludes the unwanted field and then copy the data from the original array to the new one.
### Example

In the example below, we are removing the 'Age' field from an existing structured array by creating a new array with a reduced dtype. We then copy the relevant fields from the original array into the new one −

```
import numpy as np

# Original structured array
students = np.array([(1, 'Alice', 25), (2, 'Bob', 23), (3, 'Charlie', 35)],
                    dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4')])

# Define a new dtype without the 'Age' field
reduced_dtype = [('ID', 'i4'), ('Name', 'U10')]

# Create a new structured array with the reduced dtype
students_without_age = np.zeros(students.shape, dtype=reduced_dtype)

# Copy the relevant fields
for field in students_without_age.dtype.names:
    students_without_age[field] = students[field]

# Verify the result
print(students_without_age)
```

Following is the output of the above code −

```
[(1, 'Alice') (2, 'Bob') (3, 'Charlie')]
```

## Sorting Structured Arrays

Sorting structured arrays in NumPy involves ordering the elements (rows) of the array based on one or more fields (columns).

Structured arrays can have multiple fields of different data types (e.g., integers, floats, strings), and sorting allows you to organize your data in a meaningful way, such as arranging records by age, name, or any other attribute.

### Example

In the following example, we are sorting a structured array by the 'Age' field using the np.sort() function with the "order" parameter. This rearranges the records in ascending order based on the 'Age' values −

```
import numpy as np

# Original structured array
students = np.array([(1, 'Alice', 25), (2, 'Bob', 23), (3, 'Charlie', 35)],
                    dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4')])

# Sort by 'Age'
sorted_students = np.sort(students, order='Age')
print(sorted_students)
```

The output obtained is as shown below −

```
[(2, 'Bob', 23) (1, 'Alice', 25) (3, 'Charlie', 35)]
```

## Filtering Data in Structured Arrays

Filtering data in structured arrays with NumPy involves selecting subsets of data that meet specific criteria.

To filter a structured array, you use boolean indexing. This involves creating a boolean mask (an array of True and False values) based on a condition applied to one or more fields. You then use this mask to index into the original array and extract the desired subset of records.

### Example

In this example, we are using a boolean mask to filter a structured array by selecting only those records where the 'Age' field is greater than 25 −

```
import numpy as np

# Original structured array
students = np.array([(1, 'Alice', 25), (2, 'Bob', 23), (3, 'Charlie', 30)],
                    dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4')])

# Create a boolean mask where Age > 25
mask = students['Age'] > 25

# Apply the mask to filter the array
filtered_students = students[mask]
print(filtered_students)
```

After executing the above code, we get the following output −

```
[(3, 'Charlie', 30)]
```

## Combining Structured Arrays

Combining structured arrays in NumPy is used to combine arrays with the same dtype along a single axis (usually the rows).

In NumPy, the np.concatenate() function is used to join arrays along an existing axis. For structured arrays, this requires that all arrays share the same dtype.

### Example

In the example below, we are combining two structured arrays with identical data types into one array using np.concatenate() function −

```
import numpy as np

# Define two structured arrays with the same dtype
students1 = np.array([(1, 'Alice', 25), (2, 'Bob', 23)],
                     dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4')])
students2 = np.array([(3, 'Charlie', 30), (4, 'David', 28)],
                     dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4')])

# Concatenate the arrays
combined_students = np.concatenate((students1, students2))
print(combined_students)
```

The result produced is as follows −

```
[(1, 'Alice', 25) (2, 'Bob', 23) (3, 'Charlie', 30) (4, 'David', 28)]
```

## Reshaping Structured Arrays

Reshaping structured arrays in NumPy involves changing the shape of an array while preserving its data structure. This means that the total number of elements (rows) remains the same before and after reshaping.

In NumPy, the np.reshape() function is used to change the shape of the structured array.

### Example

In the following example, we are reshaping a 1-D structured array into a 2-D array using np.reshape() function −

```
import numpy as np

# Define a 1-D structured array
students = np.array([(1, 'Alice', 25), (2, 'Bob', 23), (3, 'Charlie', 30)],
                    dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4')])

# Reshape the array from 1-D to 2-D
reshaped_students = np.reshape(students, (3, 1))
print(reshaped_students)
```

This transforms the array from a single row of records into a column format, while preserving the structured data as shown in the output below −

```
[[(1, 'Alice', 25)]
 [(2, 'Bob', 23)][(3, 'Charlie', 30)]]
```

## Splitting Structured Arrays

Splitting structured arrays in NumPy involves dividing a single structured array into multiple arrays based on certain criteria or sizes.

In NumPy, the np.split() function is used to split an array into multiple sub-arrays along a specified axis. For structured arrays, this function requires that the array be split along the axis where the elements can be evenly distributed.

### Example

In this example, we are splitting a structured array into two equal parts using np.split() function −

```
import numpy as np

# Define a structured array
students = np.array([(1, 'Alice', 25), (2, 'Bob', 23), (3, 'Charlie', 30), (4, 'David', 28)],
                    dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'i4')])

# Split the array into 2 equal parts
split_students = np.split(students, 2)
print(split_students[0])
print(split_students[1])
```

We get the output as shown below −

```
[(1, 'Alice', 25) (2, 'Bob', 23)]
[(3, 'Charlie', 30) (4, 'David', 28)]
```

---

## 57. NumPy - Record Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_record_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_record_arrays.htm)*

---

---
[Previous](/numpy/numpy_manipulating_structured_arrays.htm)[Quiz](/numpy/quiz_on_numpy_record_arrays.htm)[Next](/numpy/numpy_loading_arrays.htm)
## Record Aarrays in NumPy

Record arrays are similar to structured arrays but come with an additional feature: they allow you to access fields as attributes using attribute-style (dot notation) and index-style (dictionary-style) access.

This can make code more readable and easier to write, especially when dealing with complex data structures.

## Creating Record Arrays

We can create record arrays in multiple ways, they are −

- By converting an existing structured array
- Directly defining a record array
- Using the np.recarray constructor
### Converting a Structured Array to a Record Array

One of the simplest ways to create a record array is by converting an existing structured array using the view method. This approach allows you to maintain the structure of your data while gaining the convenience of attribute access.
**Example**
In the following example, we are converting a structured array to a record array in NumPy −

```
import numpy as np

# Create a structured array
structured_array = np.array([('Alice', 25, 5.5), ('Bob', 30, 6.0)], dtype=[('name', 'U10'), ('age', 'i4'), ('height', 'f4')])

# Convert the structured array to a record array
record_array = structured_array.view(np.recarray)

# Access fields as attributes
print(record_array.name)  
print(record_array.age)
```

Following is the output obtained −

```
['Alice' 'Bob']
[25 30]
```

### Creating a Record Array Directly

Another way to create a record array is to define it directly using the
**np.rec.array()**function. This method allows you to create a record array from scratch, specifying both the data and the structure.**Example**
In the example below, we are we are directly creating a record array in NumPy, which allows us to access the fields as attributes using dot notation −

```
import numpy as np 

# Directly create a record array
record_array_direct = np.rec.array([('Charlie', 35, 5.8), ('David', 40, 6.2)], dtype=[('name', 'U10'), ('age', 'i4'), ('height', 'f4')])

# Access fields as attributes
print(record_array_direct.name)   
print(record_array_direct.height)
```

This will produce the following result −

```
['Charlie' 'David']
[5.8 6.2]
```

### Using np.recarray Constructor

You can also create a record array using the np.recarray constructor. This method is less common but provides additional flexibility for specific use cases.
**Example**
In this example, we are creating a record array using np.recarray constructor, assigning data to it, and then accessing the fields
**name**and**age**as attributes −
```
import numpy as np

# Define a record array using np.recarray
record_array_custom = np.recarray((2,), dtype=[('name', 'U10'), ('age', 'i4'), ('height', 'f4')])

# Assign data to the record array
record_array_custom[0] = ('Eve', 28, 5.7)
record_array_custom[1] = ('Frank', 33, 6.1)

# Access fields as attributes
print(record_array_custom.name)   
print(record_array_custom.age)
```

Following is the output of the above code −

```
['Eve' 'Frank']
[28 33]
```

## Accessing Fields of Record Arrays

Once you have created a record array, you can access its fields (i.e., the columns of the structured data) as attributes, which simplifies data manipulation and querying.

The most common way to access the fields of a record array is by using the
**dot (.)**notation. Each field of the record array becomes an attribute that you can access directly, similar to how you would access an attribute of an object in Python.
### Example: Accessing Fields Using Attribute Notation

In the following example, we are creating a record array with fields
**name**,**age**, and**height**, and then accessing each field as an attribute of the record array −
```
import numpy as np

# Create a record array
record_array = np.rec.array([('Alice', 25, 5.5), ('Bob', 30, 6.0)], dtype=[('name', 'U10'), ('age', 'i4'), ('height', 'f4')])

# Accessing the 'name' field
print(record_array.name)  

# Accessing the 'age' field
print(record_array.age)   

# Accessing the 'height' field
print(record_array.height)
```

The output obtained is as shown below −

```
['Alice' 'Bob']
[25 30]
[5.5 6. ]
```

### Example: Accessing Multiple Fields Simultaneously

In addition to accessing individual fields, you can access multiple fields at once. However, unlike with structured arrays, record arrays do not directly support simultaneous field access using the attribute notation.

Instead, you need to revert to the traditional indexing method if you want to retrieve multiple fields at once as shown in the example below −

```
import numpy as np

# Create a record array
record_array = np.rec.array([('Alice', 25, 5.5), ('Bob', 30, 6.0)], dtype=[('name', 'U10'), ('age', 'i4'), ('height', 'f4')])

# Access multiple fields using indexing
multiple_fields = record_array[['name', 'height']]
print(multiple_fields)
```

After executing the above code, we get the following output −

```
[('Alice', 5.5) ('Bob', 6. )]
```

## Modifying Fields of Record Arrays

Modifying fields in a NumPy record array allows you to update or change the data within specific columns (fields) of your structured dataset. Record arrays provide a way to access and modify fields using
**attribute-style**access.
For instance, you can modify the values of individual fields in the record array by accessing the field through the dot (.) notation and then applying the desired operation or assignment.

To add new fields to a record array, you can use a combination of np.concatenat() function (or similar) to create a new record array with the additional fields.

### Example: Updating Fields

In the example below, we are creating a record array with fields "name", "age", and "height". We then change the "age" and "height" for specific records −

```
import numpy as np

# Step 1: Define the initial record array with fields 'name', 'age', and 'height'
dtype = [('name', 'S20'), ('age', 'i1'), ('height', 'f4')]
data = [('Alice', 30, 5.5), ('Bob', 25, 6.0), ('Charlie', 35, 5.8)]
record_array = np.array(data, dtype=dtype)

print("Original Record Array:")
print(record_array)

# Step 2: Update the fields for specific records
# Update 'age' and 'height' for 'Bob'
record_array[record_array['name'] == b'Bob']['age'] = 26
record_array[record_array['name'] == b'Bob']['height'] = 6.1

# Update 'age' for 'Charlie'
record_array[record_array['name'] == b'Charlie']['age'] = 36

print("\nUpdated Record Array:")
print(record_array)
```

The result produced is as follows −

```
Original Record Array:
[(b'Alice', 30, 5.5) (b'Bob', 25, 6. ) (b'Charlie', 35, 5.8)]

Updated Record Array:
[(b'Alice', 30, 5.5) (b'Bob', 25, 6. ) (b'Charlie', 35, 5.8)]
```

### Example: Adding New Fields

In this example, we first define a record array that includes only "name" and "age" fields. We then create a new record array with an additional "height" field −

```
import numpy as np

# Step 1: Define the initial record array with fields 'name' and 'age'
dtype = [('name', 'S20'), ('age', 'i1')]
data = [('Alice', 30), ('Bob', 25), ('Charlie', 35)]
record_array = np.array(data, dtype=dtype)

print("Original Record Array:")
print(record_array)

# Step 2: Define a new dtype that includes the new field 'height'
new_dtype = [('name', 'S20'), ('age', 'i1'), ('height', 'f4')]

# Step 3: Create a new record array with the new dtype, initialized with zeros or default values
new_record_array = np.zeros(record_array.shape, dtype=new_dtype)

# Step 4: Copy existing data into the new record array
for field in ['name', 'age']:
    new_record_array[field] = record_array[field]

# Optionally, set default values for the new field 'height'
new_record_array['height'] = 0.0

print("\nRecord Array with New Field:")
print(new_record_array)
```

We get the output as shown below −

```
Original Record Array:
[(b'Alice', 30) (b'Bob', 25) (b'Charlie', 35)]

Record Array with New Field:
[(b'Alice', 30, 0.) (b'Bob', 25, 0.) (b'Charlie', 35, 0.)]
```

## Combining Record Arrays

Combining record arrays in NumPy refers to merging or concatenating multiple record arrays into a single record array. This process can be useful when you need to combine datasets or extend existing datasets with additional rows.

Following are the ways for combining record arrays −

- **Concatenation:**Combine record arrays along an existing axis (typically along rows).
- **Stacking:**Stack record arrays along a new axis, which can be useful for adding a new dimension.
### Example: Concatenating Record Arrays

In the following example, we create two record arrays and then concatenate them along the rows to form a single combined record array −

```
import numpy as np

# Define two record arrays with fields 'name', 'age', and 'height'
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data1 = [('Alice', 25, 5.5), ('Bob', 30, 6.0)]
data2 = [('Charlie', 35, 5.8), ('David', 40, 5.9)]

# Create record arrays
record_array1 = np.array(data1, dtype=dtype).view(np.recarray)
record_array2 = np.array(data2, dtype=dtype).view(np.recarray)

# Concatenate the record arrays along the rows (axis 0)
combined_record_array = np.concatenate((record_array1, record_array2))

print("Combined record array:")
print(combined_record_array)
```

Following is the output obtained −

```
Combined record array:
[('Alice', 25, 5.5) ('Bob', 30, 6. ) ('Charlie', 35, 5.8)
 ('David', 40, 5.9)]
```

### Example: Stacking Record Arrays

In the example below, we create two record arrays and then stack them along a new axis to form a 3D record array −

```
import numpy as np

# Define two record arrays with fields 'name', 'age', and 'height'
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data1 = [('Alice', 25, 5.5), ('Bob', 30, 6.0)]
data2 = [('Charlie', 35, 5.8), ('David', 40, 5.9)]

# Create record arrays
record_array1 = np.array(data1, dtype=dtype).view(np.recarray)
record_array2 = np.array(data2, dtype=dtype).view(np.recarray)

# Stack the record arrays along a new axis
stacked_record_array = np.stack((record_array1, record_array2), axis=0)

print("Stacked record array:")
print(stacked_record_array)
```

This will produce the following result −

```
Stacked record array:
[[('Alice', 25, 5.5) ('Bob', 30, 6. )]
 [('Charlie', 35, 5.8) ('David', 40, 5.9)]]
```

## Filtering Record Arrays

Filtering record arrays means selecting specific rows or elements from a record array based on a condition or set of conditions.

This process is used to extract subsets of data that meet certain criteria, which helps us to do more focused analysis and manipulation.

### Example

In this example, we create a Boolean mask to filter records where the "age" is greater than "30" from a record array −

```
import numpy as np

# Define a record array with fields 'name', 'age', and 'height'
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = [('Alice', 25, 5.5), ('Bob', 30, 6.0), ('Charlie', 35, 5.8), ('David', 40, 5.9)]
record_array = np.array(data, dtype=dtype).view(np.recarray)

# Create a Boolean mask for ages greater than 30
mask = record_array.age > 30

# Apply the mask to filter the record array
filtered_record_array = record_array[mask]

print("Filtered record array (ages > 30):")
print(filtered_record_array)
```

Following is the output of the above code −

```
Filtered record array (ages > 30):
[('Charlie', 35, 5.8) ('David', 40, 5.9)]
```

---

## 58. NumPy - Loading Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_loading_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_loading_arrays.htm)*

---

---
[Previous](/numpy/numpy_record_arrays.htm)[Quiz](/numpy/quiz_on_numpy_loading_arrays.htm)[Next](/numpy/numpy_saving_arrays.htm)
## Loading Arrays in NumPy

NumPy loading arrays refers to the process of reading and loading data from external files or sources into NumPy arrays.

This functionality allows you to work with data that is stored in files such as text files, binary files, or other formats, and brings that data into the NumPy environment for analysis or manipulation. Following are the common methods used for loading arrays in NumPy −

- **Loading from Text Files:**Use functions like np.loadtxt() or np.genfromtxt() to read data from text files.
- **Loading from Binary Files:**Use np.fromfile() function to read data from binary files.
- **Loading from .npy Files:**Use np.load() function to read data from files saved in NumPys native binary format (.npy files).
## Loading Arrays from Text Files

Loading arrays from text files in NumPy is a common operation for importing data stored in plain text files into NumPy arrays.

NumPy provides
**np.loadtxt()**function and**np.genfromtxt()**function to handle different text file formats and structures, making it easy to work with various types of text-based data, they are −
### Using np.loadtxt() Function

The np.loadtxt() function is used for reading data from a text file into a NumPy array.

This function is commonly used for loading structured data that is organized in a tabular format, such as CSV files or space-separated files. It is suitable for data files where each line contains a row of numbers, and all rows have the same number of columns. Following is the syntax −

```
numpy.loadtxt(fname, dtype=<type>, delimiter=<delimiter>, comments=<char>, skiprows=<num>, usecols=<cols>)
```

Where,

- **fname:**Filename or file object to read.
- **dtype:**Data type of the resulting array (default is float).
- **delimiter:**String or character separating values (e.g., comma, space).
- **comments:**String indicating the start of a comment (e.g., #).
- **skiprows:**Number of rows to skip at the beginning of the file.
- **usecols:**Indices of columns to read (e.g., [0, 2] to read the first and third columns).**Example**
Assume you have a text file "data.txt" with the following content −

```
1 2 3
4 5 6
7 8 9
```

You can load this data into a NumPy array using the loadtxt() function as shown below −

```
import numpy as np

# Load data from a text file
array_from_text = np.loadtxt('data.txt')

print("Array loaded from text file:")
print(array_from_text)
```

### Using np.genfromtxt() Function

The np.genfromtxt() function is used to read data from text files into NumPy arrays. It is useful for handling more complex text file formats, including files with missing values, mixed data types, and irregular structures. Following is the syntax −

```
numpy.genfromtxt(fname, dtype=<type>, delimiter=<delimiter>, comments=<char>, skip_header=<num>, usecols=<cols>, filling_values=<value>, missing_values=<value>, converters=<dict>, encoding=<str>, names=<bool>)
```

Where,

- **fname:**Filename or file object to read.
- **dtype:**Data type of the resulting array. If not specified, defaults to float.
- **delimiter:**String or character separating values (e.g., comma for CSV, space for space-separated).
- **comments:**String indicating the start of a comment (e.g., #). Lines starting with this character are ignored.
- **skip_header:**Number of lines to skip at the beginning of the file (useful for skipping headers).
- **usecols:**Indices of columns to read. For example, [0, 2] will read only the first and third columns.
- **filling_values:**Values to use for missing data. Can be a scalar or a dictionary mapping column indices to fill values.
- **missing_values:**Values representing missing data in the file. Can be a scalar or a list of values.
- **converters:**Dictionary of functions for converting columns to specific formats.
- **encoding:**Encoding to use for reading the file (default is None, which uses the system default).
- **names:**If True, the first line of the file is assumed to contain column names.**Example**
In this example, we are loading the "data.txt" file into a NumPy array using the genfromtxt() function −

```
import numpy as np

# Load data from a text file
array = np.genfromtxt('data.txt')

print("Array loaded from text file:")
print(array)
```

## Loading Arrays from Binary Files

Loading arrays from binary files in NumPy involves reading data that has been stored in a binary format, which is generally more efficient for storage and retrieval than text formats.

Binary files contain raw data, which must be interpreted correctly based on the expected format and data type. NumPy provides
**np.fromfile()**function and**np.load()**function to load arrays from binary files.
### Using np.fromfile() Function

The np.fromfile() function is used to load binary data from a file into a NumPy array. This function requires knowledge of the data type and format of the binary file. Following is the syntax −

```
numpy.fromfile(file, dtype=<type>, count=-1, offset=0)
```

Where,

- **file:**Filename or file object to read.
- **dtype:**Data type of the resulting array (e.g., np.float32, np.int32).
- **count:**Number of items to read. If -1, read all data.
- **offset:**Number of bytes to skip at the beginning of the file.**Example**
Assume you have a binary file "data.bin" that contains "32-bit" float data. The file can be created using the following code −

```
import numpy as np

# Create a binary file with float data
data = np.array([1.1, 2.2, 3.3], dtype=np.float32)
data.tofile('data.bin')
print ('File created!!')
```

Now, to read this binary file, use the following code −

```
import numpy as np

# Load data from a binary file
array = np.fromfile('data.bin', dtype=np.float32)

print("Array loaded from binary file:")
print(array)
```

Following is the output of the above code −

```
Array loaded from binary file:
[1.1 2.2 3.3]
```

### Using np.load() Function for .npy Files

The np.load() function in NumPy is used to load arrays or data from files in NumPys native binary format
**.npy**or**.npz**. This format preserves the array's metadata, such as its shape and data type. The ".npz" format is used for storing multiple arrays in a compressed format.
Following is the syntax −

```
numpy.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
```

Where,

- **file:**The filename or file object to read. This can be a .npy file (for single arrays) or a .npz file (for multiple arrays).
- **mmap_mode:**If not None, it is used to memory-map the file, which allows for large arrays to be read without loading the entire file into memory. Valid values are 'r', 'r+', 'w+', etc.
- **allow_pickle:**If True, allows loading objects saved with Pythons pickle format. Be cautious with this option as it can execute arbitrary code and pose a security risk.
- **fix_imports:**If True, tries to detect and fix Python 2 to Python 3 compatibility issues when loading pickled data.
- **encoding:**The encoding used to decode Python 2 string data when loading Python 3 files. Default is 'ASCII'.**Example: Loading .npy Files**
Here, we are first saving an array to the ".npy" file format −

```
import numpy as np

# Create a NumPy array
array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

# Save the array to a .npy file
np.save('data.npy', array)
print ("Saved!!")
```

Now, we are loading the saved arrays from ".npy" files using the load() function in NumPy −

```
import numpy as np

# Load the array from the .npy file
array = np.load('data.npy')

print("Array loaded from .npy file:")
print(array)
```

The output obtained is as shown below −

```
Array loaded from .npy file:
[[1 2 3]
 [4 5 6]]
```
**Example: Loading .npz Files**
The .npz format is used for saving multiple arrays into a single compressed file. It creates a zip archive where each file inside is an ".npy" file as shown in the following example −

```
import numpy as np

# Save multiple arrays to a .npz file
array1 = np.array([1, 2, 3])
array2 = np.array([[4, 5, 6], [7, 8, 9]])
np.savez('data.npz', array1=array1, array2=array2)

# Load the arrays from the .npz file
data = np.load('data.npz')

# Access individual arrays using their keys
array1_loaded = data['array1']
array2_loaded = data['array2']

print("Array 1 loaded from .npz file:")
print(array1_loaded)

print("Array 2 loaded from .npz file:")
print(array2_loaded)
```

After executing the above code, we get the following output −

```
Array 1 loaded from .npz file:
[1 2 3]
Array 2 loaded from .npz file:
[[4 5 6]
 [7 8 9]]
```

---

## 59. NumPy - Saving Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_saving_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_saving_arrays.htm)*

---

---
[Previous](/numpy/numpy_loading_arrays.htm)[Quiz](/numpy/quiz_on_numpy_saving_arrays.htm)[Next](/numpy/numpy_append_values_to_an_array.htm)
## Saving Arrays in NumPy

Saving arrays in NumPy refers to the process of writing NumPy arrays to files so they can be stored and later reloaded.

NumPy provides several methods for saving arrays in various formats, they are −

- **np.save() Function:**Saves a single NumPy array to a file in binary .npy format.
- **np.savez() Function:**Saves multiple NumPy arrays into a single file in compressed .npz format.
- **np.savez_compressed() Function:**Similar to np.savez, but compresses the data for reduced file size.
- **np.savetxt() Function:**Saves NumPy arrays to a text file with a specific format.
## Saving Arrays to Text Files

Saving arrays to text files in NumPy is useful for exporting data in a human-readable format or for compatibility with other programs that require text input.

NumPy uses the
**np.savetxt()**function to write arrays to text files in a specified format. It is designed to handle arrays with numerical data, but it can be adapted for various use cases through formatting options. Following is the syntax −
```
numpy.savetxt(fname, data, fmt=<format>, delimiter=<delimiter>, header=<header>, footer=<footer>, comments=<char>)
```

Where,

- **fname:**Filename or file object where the data will be saved.
- **data:**The array to be saved. It can be a one-dimensional or multi-dimensional array.
- **fmt:**Format string for output. Defines how the data should be formatted (e.g., floating-point precision).
- **delimiter:**String or character separating values in the file (e.g., comma for CSV, space for space-separated).
- **header:**String to write at the beginning of the file. Useful for adding metadata or column names.
- **footer:**String to write at the end of the file. Can be used for additional information.
- **comments:**String indicating the start of a comment. Default is #.
### Example

In the following example, we are saving a 2D NumPy array to a text file using np.savetxt() function −

```
import numpy as np

# Create an array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Save the array to a text file
np.savetxt('data.txt', array, fmt='%d', delimiter=',', header='Column1,Column2,Column3')
print ('File Saved succesfully!!')
```

Following is the output obtained −

```
File Saved succesfully!!
```

## Saving Arrays to Binary Files

Saving arrays to binary files in NumPy is a way to store numerical data in a compact format. Binary files are often used for saving large datasets or for performance reasons, as they are generally faster to read from and write to compared to text files.

NumPy provides
**np.save()**function,**np.savez()**function and**np.savez_compressed()**function specifically designed for saving arrays in binary formats.
### Using np.save() Function

The np.save() function saves a single NumPy array to a binary file in NumPys native
**.npy**format. This format includes metadata such as the arrays shape and dtype, which allows for loading and manipulation later. Following is the syntax −
```
numpy.save(file, arr, allow_pickle=False, fix_imports=True)
```

Where,

- **file:**Filename or file object where the array will be saved. The file extension should be .npy.
- **arr:**The NumPy array to be saved.
- **allow_pickle:**If True, allows saving objects that can be pickled. Default is False.
- **fix_imports:**If True, attempts to fix compatibility issues when loading pickled data (for Python 2 to 3).**Example**
In this example, we are saving a 2D NumPy array to a binary ".npy" file using np.save() function, which stores the array data efficiently −

```
import numpy as np

# Create an array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Save the array to a .npy file
np.save('array.npy', array)
print ("File saved!!")
```

This will produce the following result −

```
File saved!!
```

### Using np.savez() Function

The np.savez() function saves multiple arrays into a single file with the ".npz" extension. The ".npz" file is a zip archive containing one ".npy" file for each array, which can be accessed by name. Following is the syntax −

```
numpy.savez(file, *args, **kwargs)
```

Where,

- **file:**Filename or file object where the arrays will be saved. The file extension should be .npz.
- ***args:**Arrays to be saved.
- ****kwargs:**Keyword arguments specifying names for each array.**Example**
In the example below, we are saving multiple NumPy arrays to a compressed ".npz" file using np.savez() function, where "array1" and "array2" are stored with their respective names −

```
import numpy as np

# Create multiple arrays
array1 = np.array([1, 2, 3])
array2 = np.array([[4, 5, 6], [7, 8, 9]])

# Save the arrays to a .npz file
np.savez('arrays.npz', array1=array1, array2=array2)
print ("File saved!!")
```

Following is the output of the above code −

```
File saved!!
```

### Using np.savez_compressed() Function

The np.savez_compressed() function is similar to np.savez() function, but it compresses the arrays to reduce file size. This is useful for storing large datasets more efficiently. Following is the syntax −

```
numpy.savez_compressed(file, *args, **kwargs)
```

Where,

- **file:**Filename or file object where the arrays will be saved. The file extension should be .npz.
- ***args:**Arrays to be saved.
- ****kwargs:**Keyword arguments specifying names for each array.**Example**
Here, we save multiple NumPy arrays to a compressed ".npz" file using np.savez_compressed() function, which reduces the file size while storing "array1" and "array2" with their respective names −

```
import numpy as np

# Create multiple arrays
array1 = np.array([1, 2, 3])
array2 = np.array([[4, 5, 6], [7, 8, 9]])

# Save the arrays to a compressed .npz file
np.savez_compressed('arrays_compressed.npz', array1=array1, array2=array2)
print ("File saved!!")
```

The output obtained is as shown below −

```
File saved!!
```

---

## 60. NumPy - Append Values to an Array

*Source: [https://www.tutorialspoint.com/numpy/numpy_append_values_to_an_array.htm](https://www.tutorialspoint.com/numpy/numpy_append_values_to_an_array.htm)*

---

---
[Previous](/numpy/numpy_saving_arrays.htm)[Quiz](/numpy/quiz_on_numpy_append_values_to_an_array.htm)[Next](/numpy/numpy_swap_columns_of_array.htm)
## Append Values to an Arrays in NumPy

Appending values to an array in NumPy means adding new elements or arrays to an existing NumPy array. This operation involves creating a new array that includes the original elements along with the new ones, as NumPy arrays have fixed sizes and do not support in-place modifications like traditional lists.

## Appending Values to a 1D Array

Appending values to a 1D array in NumPy involves adding new elements to the end of an existing one-dimensional array. Since NumPy arrays have a fixed size, this operation creates a new array with the original elements plus the newly appended values. To achieve this, we can use the
**np.append()**function in NumPy. Following is the syntax −
```
numpy.append(arr, values, axis=None)
```

Where,

- **arr:**The original array to which values will be appended.
- **values:**The values to be appended. Can be a single value or an array.
- **axis:**The axis along which values are appended. For 1D arrays, this parameter is ignored and can be None.
### Example

In the following example, we are using the np.append() to add elements to a 1D array: first appending a single value, and then multiple values −

```
import numpy as np

# Create an initial 1D array
arr = np.array([1, 2, 3])

# Append a single value
arr_appended_single = np.append(arr, 4)

# Append multiple values
arr_appended_multiple = np.append(arr, [4, 5, 6])

print("Array after appending a single value:", arr_appended_single)
print("Array after appending multiple values:", arr_appended_multiple)
```

Following is the output obtained −

```
Array after appending a single value: [1 2 3 4]
Array after appending multiple values: [1 2 3 4 5 6]
```

## Appending Values to a 2D Array

Appending values to a 2D array in NumPy involves adding rows or columns to an existing two-dimensional array. Since NumPy arrays have a fixed size, appending values results in creating a new array that combines the original data with the new data. Let us explore various ways and methods used to append values to a 2D array −

### Appending 2D Rows

To append rows to a 2D array, we can use the
**np.vstack()**function. This function stacks arrays vertically or concatenate along axis 0. Following is the syntax −
```
numpy.vstack(tup)
```

Where,
**tup**is a sequence of arrays to be stacked vertically. All arrays must have the same number of columns.**Example**
In this example, we are using the np.vstack() function to append rows to a 2D array, adding new rows beneath the existing ones −

```
import numpy as np

# Create an initial 2D array
arr = np.array([[1, 2], [3, 4]])

# Create an array of rows to append
rows_to_append = np.array([[5, 6], [7, 8]])

# Append rows to the initial array
arr_appended_rows = np.vstack((arr, rows_to_append))

print("Array after appending rows:")
print(arr_appended_rows)
```

The result is a new array with the additional rows stacked at the bottom −

```
Array after appending rows:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
```

### Appending 2D Columns

To append columns to a 2D array, we can use the
**np.hstack()**function. This function stacks arrays horizontally or concatenate along axis 1. Following is the syntax −
```
numpy.hstack(tup)
```

Where,
**tup**is a sequence of arrays to be stacked horizontally. All arrays must have the same number of rows.**Example**
In the example below, we are using the np.hstack() function to append columns to a 2D array, adding new columns to the right of the existing ones −

```
import numpy as np

# Create an initial 2D array
arr = np.array([[1, 2], [3, 4]])

# Create an array of columns to append
columns_to_append = np.array([[5], [6]])

# Append columns to the initial array
arr_appended_columns = np.hstack((arr, columns_to_append))

print("Array after appending columns:")
print(arr_appended_columns)
```

The result is a new array with the additional columns placed alongside the original data −

```
Array after appending columns:
[[1 2 5]
 [3 4 6]]
```

## Appending to Multi-dimensional Arrays

Appending to multi-dimensional arrays in NumPy involves adding new elements along specified axes. Unlike 1D and 2D arrays, multi-dimensional arrays (e.g., 3D or higher) require careful alignment of the dimensions and axes along which you want to append data.

### Example

In the following example, we are using the np.append() function to add values to a 3D array along the first axis −

```
import numpy as np

# Original 3D array
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Values to append (must match dimensions)
values = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])

# Append values along axis 0
result = np.append(arr, values, axis=0)
print(result)
```

The resulting array has the new values appended as additional layers on top of the original array −

```
[[[ 1  2]
 [ 3  4]]

[[ 5  6]
 [ 7  8]]

[[ 9 10]
 [11 12]]

[[13 14]
 [15 16]]]
```

## Appending with Different Data Types

Appending arrays with different data types in NumPy requires careful handling because NumPy arrays are homogeneous; they must contain elements of the same data type.

When appending arrays with different data types, NumPy will generally perform type coercion or create a new array with a common data type that can contains all the appended data.

### Example

In this example, we are  using the np.append() to add floating-point values to an integer array −

```
import numpy as np

# Original array of integers
arr = np.array([1, 2, 3])

# Values to append (floating-point)
values = np.array([4.5, 5.5])

# Append values
result = np.append(arr, values)
print(result)
```

After executing the above code, we get the following output −

```
[1.  2.  3.  4.5 5.5]
```

## Append Using np.concatenate() Function

We can use the np.concatenate() function for combining arrays along a specified axis. It is particularly useful for appending data to existing arrays. Following is the syntax −

```
numpy.concatenate((a1, a2, ...), axis=0)
```

Where,

- **a1, a2, ...:**Arrays to be concatenated. They must have the same shape except in the dimension corresponding to the axis parameter.
- **axis:**The axis along which to concatenate the arrays. 0 is the default for 1D arrays, and other values for multi-dimensional arrays.
### Example

In the following example, we are concatenating two 2D arrays along axis "0" using the np.concatenate() function −

```
import numpy as np

# Arrays to concatenate
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Concatenate along axis 0
result = np.concatenate((arr1, arr2), axis=0)
print(result)
```

The result produced is as follows −

```
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
```

---

## 61. NumPy - Swap Columns of Array

*Source: [https://www.tutorialspoint.com/numpy/numpy_swap_columns_of_array.htm](https://www.tutorialspoint.com/numpy/numpy_swap_columns_of_array.htm)*

---

---
[Previous](/numpy/numpy_append_values_to_an_array.htm)[Quiz](/numpy/quiz_on_numpy_swap_columns_of_array.htm)[Next](/numpy/numpy_insert_axes_to_an_array.htm)
## Swapping Columns of Array in NumPy

Swapping columns in a NumPy array refers to exchanging the positions of two or more columns within the array. This operation can be performed on both 1-dimensional and multi-dimensional arrays.

The primary approach for swapping columns involves
**slicing**and**indexing**, which allows you to access and rearrange the columns as needed.
## Swapping Columns Using Indexing

Swapping columns in a NumPy array using indexing is a technique where you change the order of columns within a 2D array by selecting and reassigning specific columns based on their indices.

### Example

In the following example, we are swapping the "first" and "last" columns of a 2D NumPy array using indexing −

```
import numpy as np

# Creating a 2D NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Swapping the first and last columns
arr[:, [0, 2]] = arr[:, [2, 0]]

print("Array after swapping columns:")
print(arr)
```

Following is the output obtained −

```
Array after swapping columns:
[[3 2 1]
 [6 5 4]
 [9 8 7]]
```

## Swapping Multiple Columns

Swapping multiple columns in a NumPy array involves changing the order of more than two columns simultaneously. The process uses advanced indexing to specify which columns to swap and their new positions.

### Example

In this example, we are swapping three columns of a 2D array such that the first column moves to the third position, the second column moves to the first position, and the third column moves to the second position −

```
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Swap columns in the order: 1st to 3rd, 2nd to 1st, 3rd to 2nd
arr[:, [0, 1, 2]] = arr[:, [1, 2, 0]]

print("Array after swapping columns:")
print(arr)
```

This will produce the following result −

```
Array after swapping columns:
[[2 3 1]
 [5 6 4]
 [8 9 7]]
```

## Swapping Columns in a 3D Array

Swapping columns in a 3D array in NumPy involves rearranging the elements along one of the axes of the array, usually the
**second axis**(axis 1), which corresponds to the columns when dealing with a slice of the 3D array.
### Example

In the example below, we swap the first and last columns along the second axis (axis 2) of the 3D array. The array is sliced along the first two axes, and the third axis is indexed to perform the column swap −

```
import numpy as np

# Creating a 3D NumPy array
arr = np.array([[[1, 2, 3], [4, 5, 6]], 
                [[7, 8, 9], [10, 11, 12]]])

# Swapping the first and last columns along the second axis
arr[:, :, [0, 2]] = arr[:, :, [2, 0]]

print("3D array after swapping columns:")
print(arr)
```

Following is the output of the above code −

```
3D array after swapping columns:
[[[ 3  2  1]
 [ 6  5  4]]
[[ 9  8  7]
 [12 11 10]]]
```

## Swapping Non-Adjacent Columns

Swapping non-adjacent columns in a NumPy array refers to rearranging columns that are not directly next to each other.

This operation can be performed in both 2D and 3D arrays, and it uses advanced indexing techniques to specify which columns to swap.

### Example

In the following example, we are swapping non-adjacent columns i.e. the first and third columns in a 2D NumPy array using advanced indexing −

```
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Swap the first (index 0) and third (index 2) columns
arr[:, [0, 2]] = arr[:, [2, 0]]

print("Array after swapping non-adjacent columns:")
print(arr)
```

The output obtained is as shown below −

```
Array after swapping non-adjacent columns:
[[ 3  2  1  4]
 [ 7  6  5  8]
 [11 10  9 12]]
```

## Swapping Columns in Structured Arrays

To swap columns in structured arrays, we need to rearrange fields within the structured array while preserving the integrity of each record.

Structured arrays in NumPy allow you to define arrays with heterogeneous data types, and each element of the array can contain multiple fields.

### Example

In this example, we are swapping the columns 'A' and 'C' in a structured NumPy array using field-based indexing −

```
import numpy as np

# Create a structured array
dtype = [('A', 'i4'), ('B', 'i4'), ('C', 'i4')]
arr = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype=dtype)

# Swap columns 'A' and 'C'
arr[['A', 'C']] = arr[['C', 'A']]

print("Structured array after swapping columns 'A' and 'C':")
print(arr)
```

After executing the above code, we get the following output −

```
Structured array after swapping columns 'A' and 'C':
[(3, 2, 1) (6, 5, 4) (9, 8, 7)]
```

## Swap Columns Using swapaxes() Function

We can use the np.swapaxes() function for more complex operations involving multiple axes. This function allows you to swap two specified axes of an array, which can be particularly useful in higher-dimensional arrays.

Following is the syntax −

```
numpy.swapaxes(a, axis1, axis2)
```

Where,

- **a:**The input array whose axes are to be swapped.
- **axis1:**The first axis to be swapped.
- **axis2:**The second axis to be swapped.
### Example

In the following example, we are using the np.swapaxes() function to swap the second and third axes, effectively reordering the columns in the 3D array −

```
import numpy as np

# Creating a 3D NumPy array
arr = np.array([[[1, 2, 3], [4, 5, 6]], 
                [[7, 8, 9], [10, 11, 12]]])

# Swapping axes 1 and 2
arr_swapped = np.swapaxes(arr, 1, 2)

print("3D array after swapping axes:")
print(arr_swapped)
```

The result produced is as follows −

```
3D array after swapping axes:
[[[ 1  4]
  [ 2  5]
  [ 3  6]]

 [[ 7 10]
  [ 8 11]
  [ 9 12]]]
```

---

## 62. NumPy - Insert Axes to an Array

*Source: [https://www.tutorialspoint.com/numpy/numpy_insert_axes_to_an_array.htm](https://www.tutorialspoint.com/numpy/numpy_insert_axes_to_an_array.htm)*

---

---
[Previous](/numpy/numpy_swap_columns_of_array.htm)[Quiz](/numpy/quiz_on_numpy_insert_axes_to_an_array.htm)[Next](/numpy/numpy_handling_missing_data.htm)
## Insert Axes to an Array in NumPy

Inserting axes into a NumPy array refers to adding new dimensions to the existing array. This is particularly useful when you need to align data for broadcasting, reshape arrays for specific operations, or add new dimensions to facilitate operations such as stacking or concatenation.

The primary function used for this purpose is
**np.expand_dims()**, which adds a new axis at a specified position.
## Insert Axes Using expand_dims() Function

The np.expand_dims() function in NumPy is used to insert a new axis (dimension) into an existing array, thereby increasing its dimensionality. Following is the syntax −

```
numpy.expand_dims(a, axis)
```

Where,

- **a:**It is the input array.
- **axis:**It is the position in the dimensions where the new axis is to be inserted.
### Example

In the following example, we start with creating a 1D array. Using np.expand_dims() function, we add a new axis at position 1, transforming the array into a 2D column vector −

```
import numpy as np

# Creating a 1D NumPy array
arr = np.array([1, 2, 3, 4])

# Adding a new axis to create a 2D column vector
expanded_arr = np.expand_dims(arr, axis=1)

print("Original Array:\n", arr)
print("Array with New Axis:\n", expanded_arr)
```

Following is the output obtained −

```
Original Array:
[1 2 3 4]
Array with New Axis:
[[1]
 [2]
 [3]
 [4]]
```

## Insert Axes to Create Higher-dimensional Arrays

Inserting axes to create higher-dimensional arrays means adding new dimensions to an existing array, which changes its shape. This process allows you to expand the array from, for example, 1D to 2D or from 2D to 3D.

This is useful for operations like broadcasting or reshaping data to fit specific requirements.

### Example

In this example, we are expanding a 1D NumPy array to a 3D array by adding two new axes using the np.expand_dims() function −

```
import numpy as np

# Creating a 1D NumPy array
arr = np.array([1, 2, 3])

# Adding two new axes to create a 3D array
expanded_arr = np.expand_dims(arr, axis=(0, 2))

print("Original Array:\n", arr)
print("3D Array with New Axes:\n", expanded_arr)
print("Shape of 3D Array:", expanded_arr.shape)
```

This will produce the following result −

```
Original Array:
[1 2 3]
3D Array with New Axes:
[[[1]
  [2]
  [3]]]
Shape of 3D Array: (1, 3, 1)
```

## Insert Axes Using "None" or "np.newaxis" Indexing

An alternative way to insert axes is by using
**None**or**np.newaxis**indexing. This approach is generally used for its simplicity and readability in code.
By indexing with "None" or "np.newaxis", you can expand a 1D array to 2D or 3D, or adjust the shape as needed.

### Example

In the example below, we are transforming a 1D NumPy array into a 2D row vector by adding a new axis with np.newaxis indexing −

```
import numpy as np

# Creating a 1D NumPy array
arr = np.array([1, 2, 3, 4])

# Adding a new axis to create a 2D row vector
row_vector = arr[:, np.newaxis]

print("Original Array:\n", arr)
print("2D Row Vector:\n", row_vector)
```

Following is the output of the above code −

```
Original Array:
[1 2 3 4]
2D Row Vector:
 [[1]
 [2]
 [3]
[4]]
```

## Combining Multiple Arrays with Inserted Axes

Combining multiple arrays with inserted axes involves adding new dimensions to each array so they align properly for concatenation or stacking.

By inserting axes, you ensure that arrays of different shapes can be joined together. This technique allows for flexible data manipulation and performing various operations.

### Example

In the following example, we are adding a new axis to each 1D array to convert them into 2D row vectors. We then concatenate these row vectors along axis 0, resulting in a 2D array where each original array is now a separate row −

```
import numpy as np

# Creating two 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Adding new axes to create 2D row vectors
arr1_expanded = arr1[np.newaxis, :]
arr2_expanded = arr2[np.newaxis, :]

# Concatenating the row vectors along axis 0
combined_arr = np.concatenate([arr1_expanded, arr2_expanded], axis=0)

print("Array 1 with New Axis:\n", arr1_expanded)
print("Array 2 with New Axis:\n", arr2_expanded)
print("Combined Array:\n", combined_arr)
```

The output obtained is as shown below −

```
Array 1 with New Axis:
[[1 2 3]]
Array 2 with New Axis:
 [[4 5 6]]
Combined Array:
[[1 2 3]
 [4 5 6]]
```

---

## 63. NumPy - Handling Missing Data

*Source: [https://www.tutorialspoint.com/numpy/numpy_handling_missing_data.htm](https://www.tutorialspoint.com/numpy/numpy_handling_missing_data.htm)*

---

---
[Previous](/numpy/numpy_insert_axes_to_an_array.htm)[Quiz](/numpy/quiz_on_numpy_handling_missing_data.htm)[Next](/numpy/numpy_identifying_missing_values.htm)
## Handling Missing Data in Arrays

Handling missing data is a common challenge in data analysis and processing. Missing data in arrays can arise due to various reasons, such as incomplete data collection, errors during data entry, or intentional omission.

In NumPy and data analysis, dealing with missing values involves identifying, handling, and processing them effectively to ensure data integrity and accurate results.

## Identifying Missing Data

To handle missing data, the very first step is to identify it. In NumPy, missing values are often represented as np.nan in floating-point arrays. You can use specific functions  such as
**np.isnan()**to detect these missing values.
### Example

In the following example, we create an array with missing values represented by
**np.nan**. We then use the np.isnan() function to create a mask that identifies these missing values −
```
import numpy as np

# Creating an array with missing values
arr = np.array([1, 2, np.nan, 4, np.nan, 6])

# Checking for missing values
is_nan = np.isnan(arr)

print("Array with Missing Values:\n", arr)
print("Missing Value Mask:\n", is_nan)
```

Following is the output obtained −

```
Array with Missing Values:
[ 1.  2. nan  4. nan  6.]
Missing Value Mask:
[False False  True False  True False]
```

## Removing Missing Data

Removing missing data involves eliminating parts of your dataset where data is missing.

In NumPy, you can use
**boolean indexing**to exclude**NaN**values from arrays. For example, creating a mask that identifies missing values and then using it to filter out those values.
### Example

In this example, we start with an array that contains missing values represented by "np.nan". We then remove these missing values using boolean indexing using the np.isnan() function to filter out the np.nan entries −

```
import numpy as np

# Creating an array with missing values
arr = np.array([1, 2, np.nan, 4, np.nan, 6])

# Removing missing values
cleaned_arr = arr[~np.isnan(arr)]

print("Original Array:\n", arr)
print("Array with Missing Values Removed:\n", cleaned_arr)
```

This will produce the following result −

```
Original Array:
[ 1.  2. nan  4. nan  6.]
Array with Missing Values Removed:
[1. 2. 4. 6.]
```

## Replacing Missing Data

Replacing missing data means filling in the gaps where data is missing with a substitute value. In NumPy, you can use the
**np.nan_to_num()**function to replace**NaN**values with a specific number, such as zero or the mean of the other values. Following is the syntax −
```
numpy.nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None)
```

Where,

- **x:**The input array containing NaN values, infinities, or other numerical values.
- **copy:**A boolean indicating whether to make a copy of the array (True by default). If False, the operation may be performed in place.
- **nan:**The value to replace NaN values with. The default is 0.0.
- **posinf:**The value to replace positive infinity (inf) with. If not specified, it defaults to a very large number.
- **neginf:**The value to replace negative infinity (-inf) with. If not specified, it defaults to a very small (negative) number.
### Example

In the example below, we create an array that contains missing values represented by "np.nan". We then replace these missing values with zero using the np.nan_to_num() function, which fills np.nan entries with the specified value −

```
import numpy as np

# Creating an array with missing values
arr = np.array([1, 2, np.nan, 4, np.nan, 6])

# Replacing missing values with zero
filled_arr = np.nan_to_num(arr, nan=0)

print("Original Array:\n", arr)
print("Array with Missing Values Replaced:\n", filled_arr)
```

Following is the output of the above code −

```
Original Array:
[ 1.  2. nan  4. nan  6.]
Array with Missing Values Replaced:
[1. 2. 0. 4. 0. 6.]
```

## Interpolating Missing Data

Interpolating missing data involves estimating and filling in missing values within a dataset based on the surrounding data.

Instead of replacing missing values with a constant like the mean, interpolation predicts what the missing value should be by analyzing the trend or pattern in the data.

For example, if a value is missing between "4" and "8", interpolation might estimate it as "6".

### Example

In the following example, we handle an array with missing values (np.nan) by applying linear interpolation using "interp1d" from SciPy. This function estimates and fills the missing values based on the non-missing data, resulting in a complete array −

```
import numpy as np
from scipy.interpolate import interp1d

# Creating an array with missing values
arr = np.array([1, 2, np.nan, 4, np.nan, 6])

# Creating an index array
indices = np.arange(len(arr))

# Creating a mask for non-missing values
mask = ~np.isnan(arr)

# Performing linear interpolation
interp_func = interp1d(indices[mask], arr[mask], kind='linear', fill_value='extrapolate')
filled_arr = interp_func(indices)

print("Original Array:\n", arr)
print("Array with Interpolated Missing Values:\n", filled_arr)
```

The output obtained is as shown below −

```
Original Array:
 [ 1.  2. nan  4. nan  6.]
Array with Interpolated Missing Values:
 [1. 2. 3. 4. 5. 6.]
```

---

## 64. NumPy - Identifying Missing Values

*Source: [https://www.tutorialspoint.com/numpy/numpy_identifying_missing_values.htm](https://www.tutorialspoint.com/numpy/numpy_identifying_missing_values.htm)*

---

---
[Previous](/numpy/numpy_handling_missing_data.htm)[Quiz](/numpy/quiz_on_numpy_identifying_missing_values.htm)[Next](/numpy/numpy_removing_missing_data.htm)
## Identifying Missing Values in Arrays

Identifying missing values in arrays means finding where data is missing, often represented as NaN (Not a Number) in NumPy. You can identify missing values in arrays using the NumPy
**np.isnan()**function.
> NaN is a special floating-point value defined by the IEEE floating-point standard. It is used to represent undefined or unrepresentable values, such as the result of 0/0 or a mathematical operation involving NaN.

## Using the isnan() Function

The np.isnan() function in NumPy is used to identify NaN (Not a Number) values in an array.

This function checks each element in the array and returns a boolean array of the same shape, where each element is True if the corresponding element in the original array is NaN and False otherwise. Following is the syntax −

```
numpy.isnan(x)
```

Where,
**x**is the input array in which to check for NaN values.
### Example

In the following example, we use np.isnan() function to create a mask that identifies NaN values in the array −

```
import numpy as np

# Creating an array with NaN values
arr = np.array([1.0, 2.5, np.nan, 4.7, np.nan, 6.2])

# Identifying NaN values using np.isnan()
nan_mask = np.isnan(arr)

print("Original Array:\n", arr)
print("NaN Mask:\n", nan_mask)
```

Following is the output obtained −

```
Original Array:
[1.  2.5 nan 4.7 nan 6.2]
NaN Mask:
 [False False  True False  True False]
```

## Identifying Missing Values in Multi-dimensional Arrays

Identifying missing values in multi-dimensional arrays refers to detecting NaN values across various dimensions of the array, such as in 2D matrices or 3D tensors.

This process is similar to working with 1D arrays but requires handling multiple dimensions while maintaining clarity on where the missing values are located.

### Example

In this example, we use the np.isnan() function to create a mask that identifies NaN values in a 2D array  −

```
import numpy as np 

# Creating a 2D array with NaN values
arr_2d = np.array([[1.0, np.nan, 3.5],
                   [np.nan, 5.1, 6.3]])

# Identifying NaN values in the 2D array
nan_mask_2d = np.isnan(arr_2d)

print("Original 2D Array:\n", arr_2d)
print("NaN Mask 2D:\n", nan_mask_2d)
```

This will produce the following result −

```
Original 2D Array:
[[1.  nan 3.5]
 [nan 5.1 6.3]]
NaN Mask 2D:
[[False  True False]
[ True False False]]
```

## Identifying Missing Values in Structured Arrays

Identifying missing values in structured arrays involves detecting NaN or other placeholders within fields of the array, especially when the array contains mixed data types and multiple fields.

Structured arrays are complex because each field can have its own data type, so handling missing values requires attention to each field individually.

### Example

In the example below, we use the np.isnan() function to create a mask that identifies NaN values specifically in the 'age' field of a structured array −

```
import numpy as np

# Creating a structured array with NaN values
dtype = [('name', 'U10'), ('age', 'f8')]
structured_arr = np.array([('Alice', 25), ('Bob', np.nan), ('Cathy', 23)], dtype=dtype)

# Checking for NaN values in the 'age' field
nan_mask_structured = np.isnan(structured_arr['age'])

print("Structured Array:\n", structured_arr)
print("NaN Mask for 'age' field:\n", nan_mask_structured)
```

Following is the output of the above code −

```
Structured Array:
[('Alice', 25.) ('Bob', nan) ('Cathy', 23.)]
NaN Mask for 'age' field:
[False  True False]
```

## Counting Missing Values in an Array

To determine the number of missing values in an array, you can use the np.isnan() function, which returns a boolean array indicating where the NaN values are located.

Each element in this boolean array is "True" if the corresponding element in the original array is NaN, and "False" otherwise. By summing this boolean array, you effectively count the number of True values, which corresponds to the number of missing values.

### Example

In the following example, we generate a boolean mask using np.isnan() function to identify NaN values in the array. We then count the number of NaN values by summing the mask, which provides the total count of missing values −

```
import numpy as np

# Create an array with some NaN values
arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])

# Generate a boolean array indicating NaN values
nan_mask = np.isnan(arr)

# Count the number of NaN values
nan_count = np.sum(nan_mask)

print("Boolean mask of NaN values:")
print(nan_mask)
print("Number of NaN values:")
print(nan_count)
```

The output obtained is as shown below −

```
Boolean mask of NaN values:
[False False  True False  True]
Number of NaN values:
2
```

## Boolean Indexing with np.isnan() Function

Once you have identified the missing values using np.isnan() function, you can combine this with Boolean indexing to perform various operations on those values.

Boolean indexing allows you to create a mask based on the condition (e.g., whether an element is NaN) and then use this mask to filter, replace, or analyze the elements that meet this condition.

### Example: Filtering Out Missing Values

You can use Boolean indexing to filter out the missing values from your array, retaining only the non-missing values −

```
import numpy as np

# Create an array with some NaN values
arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])

# Generate a boolean array indicating NaN values
nan_mask = np.isnan(arr)

# Filter out NaN values
filtered_arr = arr[~nan_mask]

print("Original array:")
print(arr)
print("Filtered array (without NaN values):")
print(filtered_arr)
```

After executing the above code, we get the following output −

```
Original array:
[ 1.  2. nan  4. nan]
Filtered array (without NaN values):
[1. 2. 4.]
```

### Example: Replacing Missing Values

You can replace NaN values with a specific value, such as the mean or median of the non-missing values −

```
import numpy as np

# Create an array with some NaN values
arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])

# Calculate the mean of non-NaN values
mean_value = np.nanmean(arr)

# Replace NaN values with the mean value
arr_with_replacement = np.where(np.isnan(arr), mean_value, arr)

print("Original array:")
print(arr)
print("Array with NaN replaced by mean:")
print(arr_with_replacement)
```

The result produced is as follows −

```
Original array:
[ 1.  2. nan  4. nan]
Array with NaN replaced by mean:
[1.         2.         2.33333333 4.         2.33333333]
```

### Example: Analyzing Missing Values

You can use Boolean indexing to analyze the distribution or patterns of missing values, for instance, checking which rows or columns have missing data −

```
import numpy as np

# Create a 2D array with some NaN values
arr_2d = np.array([[1.0, np.nan, 3.0],
                   [4.0, np.nan, 6.0],
                   [np.nan, 8.0, 9.0]])

# Identify NaN values
nan_mask_2d = np.isnan(arr_2d)

# Count NaN values per row
nan_count_per_row = np.sum(nan_mask_2d, axis=1)

print("Original 2D array:")
print(arr_2d)
print("NaN count per row:")
print(nan_count_per_row)
```

We get the output as shown below −

```
Original 2D array:
[[ 1. nan  3.]
 [ 4. nan  6.]
 [nan  8.  9.]]
NaN count per row:
[1 1 1]
```

---

## 65. NumPy - Removing Missing Data

*Source: [https://www.tutorialspoint.com/numpy/numpy_removing_missing_data.htm](https://www.tutorialspoint.com/numpy/numpy_removing_missing_data.htm)*

---

---
[Previous](/numpy/numpy_identifying_missing_values.htm)[Quiz](/numpy/quiz_on_numpy_removing_missing_data.htm)[Next](/numpy/numpy_imputing_missing_data.htm)
## Removing Missing Data from Arrays

Removing missing data from arrays involves cleaning the dataset by eliminating entries that contain
**NaN**or other indicators of missing values.
> NaN is used to denote undefined or unrepresentable values. It is important to address NaN values before performing any calculations to avoid misleading results or errors.

## Removing Missing Data from 1D Arrays

Removing missing data from 1D arrays involves filtering out elements that are marked as missing, usually represented by NaN (Not a Number). In a 1D array, missing values are identified using the
**np.isnan()**function, which creates a boolean array where each "True" value corresponds to a "NaN" entry in the original array.
To remove these missing values, you apply this boolean mask to the array, inverting the mask to focus on non-NaN entries. Specifically,
**~np.isnan()**generates a boolean array where True indicates valid data.
By using this mask to index the original array, you filter out all NaN values, resulting in a cleaned array that contains only valid entries.

### Example

In the following example, we use Boolean indexing with np.isnan() function to create a mask that identifies NaN values. We then apply this mask to remove NaN values from the original array −

```
import numpy as np

# Creating a 1D array with NaN values
arr = np.array([1.0, 2.5, np.nan, 4.7, np.nan, 6.2])

# Removing NaN values using Boolean indexing
cleaned_arr = arr[~np.isnan(arr)]

print("Original Array:\n", arr)
print("Cleaned Array (without NaN):\n", cleaned_arr)
```

Following is the output obtained −

```
Original Array:
[1.  2.5 nan 4.7 nan 6.2]
Cleaned Array (without NaN):
[1.  2.5 4.7 6.2]
```

## Removing Missing Data from 2D Arrays

Removing missing data from 2D arrays involves eliminating rows or columns that contain NaN (Not a Number) values.

This process ensures that the dataset is cleaned and suitable for analysis or modeling. Depending on the specific requirements, you can choose to remove entire rows or columns where missing values are present.

### Example

In this example, we use np.isnan() function combined with any() function to create a mask that identifies rows containing NaN values. We then use this mask to filter out and remove those rows from the original 2D array −

```
import numpy as np 

# Creating a 2D array with NaN values
arr_2d = np.array([[1.0, np.nan, 3.5],
                   [np.nan, 5.1, 6.3],
                   [7.2, 8.1, 9.4]])

# Removing rows with NaN values
cleaned_arr_2d = arr_2d[~np.isnan(arr_2d).any(axis=1)]

print("Original 2D Array:\n", arr_2d)
print("Cleaned 2D Array (rows without NaN):\n", cleaned_arr_2d)
```

This will produce the following result −

```
Original 2D Array:
[[1.  nan 3.5]
 [nan 5.1 6.3]
 [7.2 8.1 9.4]]
Cleaned 2D Array (rows without NaN):
[[7.2 8.1 9.4]]
```

## Removing Columns with Missing Data

Removing columns with missing data involves eliminating entire columns from a 2D array or dataset where any element is marked as missing, generally represented by NaN (Not a Number).

This is a common data cleaning step used to ensure that the dataset only includes columns with complete data, which can improve the quality of subsequent analyses.

### Example

In the example below, we are creating a 2D array with some NaN values and removing columns that contain any NaN values using np.isnan() function combined with the any() function. This identifies columns with NaN values and then filters the array to exclude those columns −

```
import numpy as np

# Create a 2D array with some NaN values
arr_2d = np.array([[1.0, np.nan, 3.0],
                   [4.0, 5.0, 6.0],
                   [np.nan, 8.0, 9.0]])

# Remove columns with any NaN values
cleaned_arr_2d_cols = arr_2d[:, ~np.isnan(arr_2d).any(axis=0)]

print("Original 2D array:")
print(arr_2d)
print("2D array with columns containing NaN removed:")
print(cleaned_arr_2d_cols)
```

Following is the output of the above code −

```
Original 2D array:
[[ 1. nan  3.]
 [ 4.  5.  6.]
 [nan  8.  9.]]
2D array with columns containing NaN removed:
[[3.]
 [6.]
 [9.]]
```

## Removing Missing Data from Multi-dimensional Arrays

Removing missing data from multi-dimensional arrays involves a process similar to that used for 1D and 2D arrays but applied to higher dimensions.

Multi-dimensional arrays (e.g., 3D or 4D arrays) present additional complexity because missing values may occur across multiple dimensions. The goal is to filter out slices or specific parts of the array that contain missing data.

### Example

In the following example, we are creating a 3D array with some NaN values and removing slices (2D arrays) that contain any NaN values. We use the np.isnan() function combined with the any() function to identify slices with NaN values and then filter out those slices from the array −

```
import numpy as np 

# Creating a 3D array with NaN values
arr_3d = np.array([[[1.0, np.nan],
                    [3.5, 4.2]],
                   [[np.nan, 6.3],
                    [7.2, 8.1]]])

# Removing slices with NaN values
cleaned_arr_3d = arr_3d[~np.isnan(arr_3d).any(axis=(1, 2))]

print("Original 3D Array:\n", arr_3d)
print("Cleaned 3D Array (slices without NaN):\n", cleaned_arr_3d)
```

The output obtained is as shown below −

```
Original 3D Array:
[[[1.  nan]
  [3.5 4.2]]

 [[nan 6.3]
  [7.2 8.1]]]
Cleaned 3D Array (slices without NaN):
[]
```

## Removing Missing Values from Structured Arrays

Removing missing values from structured arrays in NumPy involves handling arrays with complex data types where each element is a record or a row with multiple fields.

Structured arrays can include missing values (NaN or other placeholders) in specific fields. The goal is to filter out records that contain missing values, ensuring that only complete data is retained.

### Example

In the following example, we define a structured array with fields 'name' and 'age', using 'f4' (float32) for the 'age' field to accommodate NaN values. We then create a boolean mask to identify and remove records with missing values in the 'age' field −

```
import numpy as np

# Define a structured array with fields 'name' and 'age'
# Use 'f4' (float32) for the 'age' field to handle NaN values
dtype = [('name', 'U10'), ('age', 'f4')]
data = [('Alice', 25.0), ('Bob', np.nan), ('Charlie', 30.0)]
structured_array = np.array(data, dtype=dtype)

# Identify missing values in the 'age' field
nan_mask = np.isnan(structured_array['age'])

# Remove records with missing values in the 'age' field
cleaned_structured_array = structured_array[~nan_mask]

print("Original structured array:")
print(structured_array)
print("Structured array with missing values removed:")
print(cleaned_structured_array)
```

After executing the above code, we get the following output −

```
Original structured array:
[('Alice', 25.) ('Bob', nan) ('Charlie', 30.)]
Structured array with missing values removed:
[('Alice', 25.) ('Charlie', 30.)]
```

---

## 66. NumPy - Imputing Missing Data

*Source: [https://www.tutorialspoint.com/numpy/numpy_imputing_missing_data.htm](https://www.tutorialspoint.com/numpy/numpy_imputing_missing_data.htm)*

---

---
[Previous](/numpy/numpy_removing_missing_data.htm)[Quiz](/numpy/quiz_on_numpy_imputing_missing_data.htm)[Next](/numpy/numpy_performance_optimization_with_arrays.htm)
## Imputing Missing Data in Arrays

Imputing missing data in arrays involves filling in the missing values with estimated or calculated values based on the available data. This process helps in the following ways −

- **Preserve Data:**Avoids loss of information that might be important for analysis.
- **Improve Analysis:**Ensures complete datasets, which can lead to more accurate analyses.
- **Handle Missing Data:**Addresses gaps in data that could distort results if left unhandled.
## Imputing Missing Data with Mean

> The mean value, often referred to as the average, is a measure of central tendency that summarizes a set of numbers by finding their central value. 
> It is calculated by adding together all the numbers in a dataset and then dividing the sum by the count of those numbers.

The mean value, often referred to as the average, is a measure of central tendency that summarizes a set of numbers by finding their central value.

It is calculated by adding together all the numbers in a dataset and then dividing the sum by the count of those numbers.

### Example

In the following example, we calculate the mean of non-NaN values in an array and then use this mean to replace NaN values −

```
import numpy as np

# Creating an array with NaN values
arr = np.array([1.0, 2.5, np.nan, 4.7, np.nan, 6.2])

# Calculating the mean of non-NaN values
mean_value = np.nanmean(arr)

# Imputing NaN values with the mean
imputed_arr = np.where(np.isnan(arr), mean_value, arr)

print("Original Array:\n", arr)
print("Mean Value:", mean_value)
print("Imputed Array:\n", imputed_arr)
```

Following is the output obtained −

```
Original Array:[1.  2.5 nan 4.7 nan 6.2]
Mean Value: 3.5999999999999996
Imputed Array:[1.  2.5 3.6 4.7 3.6 6.2]
```

## Imputing Missing Data with Median

Imputing Missing Data with Median is a technique used to fill in missing values in a dataset by replacing them with the median value of the available data.

The median is the middle value in a dataset when it is ordered, or the average of the two middle values if the dataset has an even number of observations.

### Example

In this example, we are calculating the median of non-NaN values in an array and then using this median to replace NaN values −

```
import numpy as np

# Creating an array with NaN values
arr = np.array([1.0, 2.5, np.nan, 4.7, np.nan, 6.2])

# Calculating the median of non-NaN values
median_value = np.nanmedian(arr)

# Imputing NaN values with the median
imputed_arr = np.where(np.isnan(arr), median_value, arr)

print("Original Array:\n", arr)
print("Median Value:", median_value)
print("Imputed Array:\n", imputed_arr)
```

This will produce the following result −

```
Original Array: [1.  2.5 nan 4.7 nan 6.2]
Median Value: 3.6
Imputed Array: [1.  2.5 3.6 4.7 3.6 6.2]
```

## Imputing Missing Data with a Constant

Imputing Missing Data with a Constant is a technique used to fill in missing values in a dataset by replacing them with a predefined constant value.

> A constant value refers to a fixed, unchanging number or value that remains the same throughout a particular context or operation.

### Example

In the example below, we define a constant value for imputation and replace NaN values in an array with this constant −

```
import numpy as np

# Creating an array with NaN values
arr = np.array([1.0, 2.5, np.nan, 4.7, np.nan, 6.2])

# Defining a constant value for imputation
constant_value = 0

# Imputing NaN values with the constant
imputed_arr = np.where(np.isnan(arr), constant_value, arr)

print("Original Array:\n", arr)
print("Constant Value:", constant_value)
print("Imputed Array:\n", imputed_arr)
```

Following is the output of the above code −

```
Original Array: [1.  2.5 nan 4.7 nan 6.2]
Constant Value: 0
Imputed Array:[1.  2.5 0.  4.7 0.  6.2]
```

## Imputing Missing Data in Multi-dimensional Arrays

Imputing Missing Data in Multi-dimensional Arrays involves filling in missing values within arrays that have more than one dimension, such as 2D matrices or higher-dimensional arrays.

### Example: Imputing Missing Data in a 2D Array

In the following example, we calculate the mean of each column in a 2D array while ignoring NaN values. We then replace the NaN values with the mean of their respective columns −

```
import numpy as np

# Creating a 2D array with NaN values
arr_2d = np.array([[1.0, np.nan, 3.5],
                   [np.nan, 5.1, 6.3],
                   [7.2, 8.1, np.nan]])

# Imputing NaN values with the mean of each column
column_means = np.nanmean(arr_2d, axis=0)
inds = np.where(np.isnan(arr_2d))

# Replace NaN values with the mean of the respective column
arr_2d[inds] = np.take(column_means, inds[1])

print("Original 2D Array:\n", arr_2d)
print("Column Means:", column_means)
print("Imputed 2D Array:\n", arr_2d)
```

The output obtained is as shown below −

```
Original 2D Array:
[[1.  6.6 3.5]
 [4.1 5.1 6.3]
 [7.2 8.1 4.9]]
 
Column Means: [4.1 6.6 4.9]

Imputed 2D Array:
[[1.  6.6 3.5]
 [4.1 5.1 6.3]
 [7.2 8.1 4.9]]
```

### Example: Imputing Missing Data in a 3D Array

Here, we are calculating the median value for each column across all slices of a 3D array while ignoring NaNs. We then replace NaN values with the corresponding median value for each column −

```
import numpy as np

# Create a 3D array with some NaN values
arr_3d = np.array([[[1.0, 2.0, np.nan],
                    [np.nan, 5.0, 6.0],
                    [7.0, np.nan, 9.0]],
                   
                   [[np.nan, 2.0, 3.0],
                    [4.0, np.nan, np.nan],
                    [7.0, 8.0, np.nan]]])

# Calculate the median of each slice along the last axis, ignoring NaN values
median_value = np.nanmedian(arr_3d, axis=(0, 1))

# Find indices where NaN values are present
nan_indices = np.isnan(arr_3d)

# Replace NaN values with the median value of the corresponding slice
for i in range(arr_3d.shape[2]):  # Iterate over the third dimension
    arr_3d[:, :, i][nan_indices[:, :, i]] = median_value[i]

print("3D array after median imputation:")
print(arr_3d)
```

After executing the above code, we get the following output −

```
3D array after median imputation:
[[[1.  2.  6. ]
  [5.5 5.  6. ]
  [7.  3.5 9. ]]

 [[5.5 2.  3. ]
  [4.  3.5 6. ]
  [7.  8.  6. ]]]
```

## Imputing with Linear Interpolation

Imputing missing data using linear interpolation involves estimating the missing values based on the values that surround them. This technique is useful for data that is sequential or spatial, where the missing values can be inferred by the values that precede and follow them.

- **Linear interpolation**is a method of estimating unknown values that fall between known values.
- In**one-dimensional data**, it involves drawing a straight line between two known points and using this line to estimate the value at a point in between.
- For**multi-dimensional data**, linear interpolation can extend this concept to higher dimensions.
### Example

In the example below, we use linear interpolation to fill missing values (NaNs) in a 1D array. We achieve this by estimating NaN values based on the surrounding non-NaN values −

```
import numpy as np
from scipy import interpolate

# Creating an array with NaN values
arr = np.array([1.0, np.nan, 3.5, np.nan, 5.0])

# Interpolating missing values
nans, x = np.isnan(arr), lambda z: z.nonzero()[0]
arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])

print("Original Array:\n", arr)
print("Array with Interpolated Values:\n", arr)
```

We get the output as shown below −

```
Original Array:
[1.   2.25 3.5  4.25 5.  ]
Array with Interpolated Values:
[1.   2.25 3.5  4.25 5.  ]
```

---

## 67. NumPy - Performance Optimization with Arrays

*Source: [https://www.tutorialspoint.com/numpy/numpy_performance_optimization_with_arrays.htm](https://www.tutorialspoint.com/numpy/numpy_performance_optimization_with_arrays.htm)*

---

---
[Previous](/numpy/numpy_imputing_missing_data.htm)[Quiz](/numpy/quiz_on_numpy_performance_optimization_with_arrays.htm)[Next](/numpy/numpy_identifying_missing_values.htm)
## Performance Optimization with Arrays

Performance optimization with arrays involves improving the efficiency of operations on arrays, such as reducing computation time and memory usage.

We should optimize performance for the following reasons −

- **Speed:**Faster computations lead to quicker results and more responsive applications.
- **Scalability:**Optimized code can handle larger datasets and more complex operations efficiently.
- **Resource Efficiency:**Reduces memory usage and computational overhead.
## Using Vectorized Operations

Vectorized operations refer to the ability to perform operations on entire arrays or matrices in a single step without using explicit loops.

This is achieved through broadcasting and internal optimization, making these operations faster and more efficient.

### Example

In the following example, we are performing vectorized addition of two large arrays, "a" and "b", using NumPy's array operations. This operation calculates the element-wise sum of the arrays and stores the result in a new array "c" −

```
import numpy as np

# Create two large arrays
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Vectorized addition
c = a + b
print (c)
```

Following is the output obtained −

```
[0.91662816 0.65486861 1.60409272 ... 0.95122935 1.12795861 0.15812103]
```

## Utilizing Efficient Data Types

Choosing the appropriate data type for your arrays is important for optimizing performance and memory usage in NumPy.

For example, using
**np.float32**instead of**np.float64**can significantly impact memory usage and performance, particularly when working with large datasets.
> In NumPy, a data type (or dtype) defines the kind of elements that an array holds and how much space is required to store each element.

### Example

In this example, we are demonstrating the usage of precision change by creating an array with double precision (64-bit) floating-point numbers and then converting it to single precision (32-bit) using the astype() method −

```
import numpy as np

# Create an array with double precision (64-bit)
arr_double = np.array([1.0, 2.0, 3.0], dtype=np.float64)

# Print the original double precision array
print("Original double precision array:")
print(arr_double)
print("Data type:", arr_double.dtype)

# Convert to single precision (32-bit)
arr_single = arr_double.astype(np.float32)

# Print the converted single precision array
print("\nConverted single precision array:")
print(arr_single)
print("Data type:", arr_single.dtype)
```

This will produce the following result −

```
Original double precision array:
[1. 2. 3.]
Data type: float64

Converted single precision array:
[1. 2. 3.]
Data type: float32
```

## Avoiding Loops with NumPy Functions

In NumPy, one of the primary advantages is the ability to avoid explicit loops by using built-in functions and array operations. This approach is often referred to as vectorization.

By using NumPy functions, you can perform operations on entire arrays at once, which is more concise compared to using loops.

### Example

In the example below, we calculate the mean of the array elements using the np.mean() function, without using any explicit loops −

```
import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4, 5])

# Calculate the mean of array elements
mean = np.mean(arr)
print("mean:",mean)
```

Following is the output of the above code −

```
mean: 3.0
```

## Using Broadcasting for Vectorization

Broadcasting refers to the ability to perform element-wise operations on arrays with different shapes. It follows a set of rules to determine how arrays with different shapes can be aligned for operations −

- **Same Dimensions:**If the arrays have different dimensions, the smaller array's shape is padded with ones on the left until both shapes have the same length.
- **Dimension Compatibility:**Two dimensions are compatible when they are equal or one of them is 1. For each dimension, if the sizes are different and, if neither of them is 1, then the broadcasting fails.
- **Stretching:**Arrays with a dimension of size 1 are stretched along that dimension to match the size of the other arrays dimension.
### Example

In the following example, we are broadcasting "array_1d" to match the shape of "array_2d", allowing element-wise addition −

```
import numpy as np

# Create a 2D array and a 1D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
array_1d = np.array([10, 20, 30])

# Add the 1D array to each row of the 2D array
result = array_2d + array_1d
print(result)
```

The output obtained is as shown below −

```
[[11 22 33]
 [14 25 36]]
```

## In-place Operations for Vectorization

In-place operations in NumPy refer to modifying the data of an array directly, without creating a new array to store the result, saving memory and improving performance.

This is achieved by using operators and functions that alter the content of the original array. These operations generally use operators with an in-place suffix (e.g., +=, -=, *=, /=) or functions that support in-place modification.

### Example: Using In-place Operators

In this example, we are applying arithmetic operation "+=" directly on an array without creating a new one −

```
import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4, 5])

# Add 10 to each element in-place
arr += 10
print(arr)
```

After executing the above code, we get the following output −

```
[11 12 13 14 15]
```

### Example: Using In-place Functions

Here, we are calculating the exponential value of each element in an array in-place using NumPy exp() function −

```
import numpy as np

# Create an array with a floating-point data type
arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)

# Compute the exponential of each element in-place
np.exp(arr, out=arr)
print(arr)
```

After executing the above code, we get the following output −

```
[  2.71828183   7.3890561   20.08553692  54.59815003 148.4131591 ]
```

## Using Memory Views for Vectorization

Memory views refer to different ways of accessing or viewing the same underlying data in an array without duplicating it. This concept allows you to create different "views" or "slices" of the array that can operate on the same data in various ways −

- **Slicing:**When you slice an array, NumPy creates a view of the original array, not a copy. This view shares the same data buffer, so changes to the view affect the original array and vice versa.
- **Reshaping:**Reshaping an array creates a new view of the same data with a different shape. This does not alter the underlying data but changes how it is interpreted.
### Example: Slicing

In the example below, we create a 2D NumPy array and a view (slice) of the original array. Modifying the view also affects the original array −

```
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Create a view (slice) of the original array
view = arr[:, 1:]

# Modify the view
view[0, 0] = 99

print(arr)
```

We get the output as shown below −

```
[[ 1 99  3]
 [ 4  5  6]]
```

### Example: Reshaping

Here, we create a 1D NumPy array using the arange() function and then reshape it into a 2D array with 3 rows and 4 columns, changing its structure while preserving the original data −

```
import numpy as np

# Create a 1D array
arr = np.arange(12)

# Reshape to a 2D array
reshaped = arr.reshape((3, 4))

print(reshaped)
```

We get the output as shown below −

```
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
```

## Using Strides for Vectorization

Strides are a tuple that indicates the number of bytes to step in each dimension when traversing an array. They determine how array elements are accessed in memory, providing insight into how data is laid out and accessed.

Strides give you the memory offset for each dimension. For instance, in a 2D array, the stride for the second dimension tells you how many bytes to move in memory to access the next element in that row.

### Example

In the following example, we create a 2D NumPy array and use the
**strides**attribute to retrieve the number of bytes to step in each dimension when traversing the array −
```
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Print the strides of the array
print(arr.strides)
```

We get the output as shown below −

```
(24, 8)
```

---

## 68. NumPy - Matrix Addition

*Source: [https://www.tutorialspoint.com/numpy/numpy_matrix_addition.htm](https://www.tutorialspoint.com/numpy/numpy_matrix_addition.htm)*

---

---
[Previous](/numpy/numpy_matrix_library.htm)[Quiz](/numpy/quiz_on_numpy_matrix_addition.htm)[Next](/numpy/numpy_matrix_subtration.htm)
## What is Matrix Addition?

Matrix addition is the operation where two matrices of the same size are added together. In matrix addition, each element in one matrix is added to the corresponding element in the other matrix.

For matrix addition to be possible, both matrices must have the same dimensions i.e., the same number of rows and columns.

If you have two matrices, say
**A**and**B**, of the same size, then their sum**C**is defined as:
```
C = A + B
```

Where,

```
Cij = Aij + Bij
```
= A+ B
In other words, the element in the
**i**row and**j**column of matrix**C**is the sum of the corresponding elements in matrices**A**and**B**.
### Example of Matrix Addition

Consider the following two matrices:

```
A = [[1, 2], 
     [3, 4]]

B = [[5, 6], 
     [7, 8]]
```

The sum
**C = A + B**will be calculated as:
```
C = [[1+5, 2+6], 
     [3+7, 4+8]] 
  = [[6, 8], 
     [10, 12]]
```

So, the result of adding matrices
**A**and**B**gives us matrix**C**:
```
C = [[6, 8], 
     [10, 12]]
```

## Matrix Addition in NumPy

In NumPy, matrix addition is done using the
**+**operator or using the**numpy.add()**function. NumPy arrays provide the ability to perform matrix operations element-wise, including addition, which is useful for performing fast mathematical computations.
Following are the key points to remember while performing matrix addition −

- **Matrix Dimensions:**For matrix addition to be valid, the matrices must have the same dimensions (same number of rows and columns).
- **Element-wise Operations:**NumPy automatically handles element-wise operations, making it very easy to add matrices using the**+**operator or the**numpy.add()**function.
- **Flexible Arrays:**NumPy arrays are flexible and can handle matrices of different sizes as long as they are compatible in dimensions.
## Creating Matrices in NumPy

Before performing matrix addition, let us first create matrices in NumPy. Matrices in NumPy are essentially 2D arrays, and we can create them using the
**np.array()**function as shown below −
```
import numpy as np

# Creating two 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Print the matrices
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
```

Following is the output obtained −

```
Matrix A:
[[1 2]
 [3 4]]

Matrix B:
[[5 6]
 [7 8]]
```

## Matrix Addition Using the
**+**Operator
The simplest way to add two matrices in NumPy is by using the
**+**operator. This operator will automatically perform element-wise addition of the two matrices.
### Example

In the following example, we are adding two matrices "A" and "B" using the "+" operator −

```
import numpy as np

# Creating two 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# Adding two matrices using the + operator
C = A + B

# Print the result
print("Matrix C (A + B):")
print(C)
```

The output obtained is as shown below −

```
Matrix C (A + B):
[[ 6  8]
 [10 12]]
```

## Using the
**numpy.add()**Function
Alternatively, you can perform matrix addition using the
**numpy.add()**function, which works the same way as the**+**operator. This function takes two matrices (or arrays) as inputs and returns their sum.
### Example

In this example, we are adding two matrices "A" and "B" using the "numpy.add()" function −

```
import numpy as np

# Creating two 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# Adding two matrices using numpy.add() function
C = np.add(A, B)

# Print the result
print("Matrix C (A + B using numpy.add()):")
print(C)
```

We get the output as shown below −

```
Matrix C (A + B using numpy.add()):
[[ 6  8]
 [10 12]]
```

## Broadcasting in Matrix Addition

While matrix addition requires matrices of the same shape, NumPy has a powerful feature called
**broadcasting**that allows for element-wise operations between arrays of different shapes.
Broadcasting automatically adjusts the shapes of arrays to allow operations between them. However, for matrix addition specifically, both matrices must have the same shape.

### Example

To give you a sense of how broadcasting works (though not directly applicable to matrix addition), here is an example of adding a scalar to a matrix −

```
import numpy as np

# Create a 2x2 matrix
A = np.array([[1, 2], [3, 4]])

# Add a scalar to the matrix using broadcasting
B = A + 10

# Print the result
print("Matrix A + 10:")
print(B)
```

The result produced is as follows −

```
Matrix A + 10:
[[11 12]
 [13 14]]
```

## Error Handling in Matrix Addition

If you try to add two matrices with different shapes (i.e., different dimensions), NumPy will raise an error. This is an important point to watch for when performing matrix addition.

### Example

Following is an example of mismatch dimension in NumPy while performing matrix addition −

```
import numpy as np

# Create two matrices with different shapes
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6, 7]])
C = A + B
print(C)
```

After executing the above code, we get the following output −

```
Traceback (most recent call last):
  File "/home/cg/root/6734345c5507a/main.py", line 6, in <module>
C = A + B
ValueError: operands could not be broadcast together with shapes (2,2) (1,3)
```

## Applications of Matrix Addition

- **Image Processing:**Matrices represent images as pixel values. Matrix addition is used to manipulate images by adding brightness or adjusting pixel values.
- **Data Analysis:**In data science, matrices represent datasets. Matrix operations like addition help combine datasets or modify them.
- **Linear Systems:**Matrix addition is used to solve systems of linear equations by adding matrices that represent different coefficients.
- **Computer Graphics:**Matrix operations are central to 3D transformations, where matrix addition helps with transformations like translation.

---

## 69. NumPy - Matrix Subtraction

*Source: [https://www.tutorialspoint.com/numpy/numpy_matrix_subtration.htm](https://www.tutorialspoint.com/numpy/numpy_matrix_subtration.htm)*

---

---

## 70. NumPy - Matrix Multiplication

*Source: [https://www.tutorialspoint.com/numpy/numpy_matrix_multiplication.htm](https://www.tutorialspoint.com/numpy/numpy_matrix_multiplication.htm)*

---

---

## 71. NumPy - Element-wise Matrix Operations

*Source: [https://www.tutorialspoint.com/numpy/numpy_element_wise_matrix_operations.htm](https://www.tutorialspoint.com/numpy/numpy_element_wise_matrix_operations.htm)*

---

---
[Previous](/numpy/numpy_matrix_multiplication.htm)[Quiz](/numpy/quiz_on_numpy_element_wise_matrix_operations.htm)[Next](/numpy/numpy_dot_product.htm)
## Element-wise Matrix Operations in NumPy

Element-wise matrix operations in NumPy refer to performing operations on corresponding elements of two matrices or arrays.

These operations are performed on an element-by-element basis, meaning each element of the first matrix is operated on with the corresponding element in the second matrix, or by a scalar value. These operations can include addition, subtraction, multiplication, division, and more.

> Element-wise operations are widely used in data manipulation, machine learning, and mathematical computations, where such operations are performed on large datasets for analysis or transformation.

## Features of Element-wise Operations

Following are some important points about element-wise matrix operations −

- **Same Shape Requirement:**For element-wise operations between two matrices, both matrices should have the same shape (dimensions). If they have different shapes, NumPy will raise an error, unless broadcasting is used (discussed later).
- **Scalar Operations:**Element-wise operations can also be performed between a matrix and a scalar value. In this case, the scalar is applied to each element of the matrix individually.
- **Efficiency:**Element-wise operations in NumPy are highly optimized and are usually much faster than using traditional Python loops to perform the same operations.
## Common Element-wise Operations

In NumPy, following are the common element-wise matrix operations −

- **Element-wise Addition:**Adding corresponding elements of two matrices.
- **Element-wise Subtraction:**Subtracting corresponding elements of two matrices.
- **Element-wise Multiplication:**Multiplying corresponding elements of two matrices.
- **Element-wise Division:**Dividing corresponding elements of two matrices.
- **Scalar Operations:**Applying arithmetic operations between a matrix and a scalar value.
## Element-wise Matrix Addition

In this operation, we add two matrices element by element. Each element from the first matrix is added to the corresponding element from the second matrix.

### Example

In this example, each element of the resulting matrix is the sum of the corresponding elements of the two input matrices −

```
import numpy as np

# Define two matrices
matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

# Element-wise addition
result = matrix_1 + matrix_2
print(result)
```

Following is the output obtained −

```
[[ 6  8]
 [10 12]]
```

## Element-wise Matrix Subtraction

Element-wise subtraction involves subtracting corresponding elements of two matrices. Each element of the first matrix is subtracted from the corresponding element of the second matrix.

### Example

In the following example, each element of the resultant matrix is the difference between corresponding elements of the two matrices −

```
import numpy as np

# Define two matrices
matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

# Element-wise subtraction
result = matrix_1 - matrix_2
print(result)
```

Following is the output obtained −

```
[[-4 -4]
 [-4 -4]]
```

## Element-wise Matrix Multiplication

Element-wise matrix multiplication, often referred to as the Hadamard product, involves multiplying corresponding elements of two matrices. This operation is different from the traditional matrix multiplication.

### Example

Here, each element of the resulting matrix is the product of corresponding elements from the two matrices −

```
import numpy as np

# Define two matrices
matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
result = matrix_1 * matrix_2
print(result)
```

Following is the output obtained −

```
[[ 5 12]
 [21 32]]
```

## Element-wise Matrix Division

Element-wise matrix division divides corresponding elements of the two matrices. Each element of the first matrix is divided by the corresponding element of the second matrix.

### Example

In this example, the result is calculated by dividing corresponding elements of the two matrices −

```
import numpy as np

# Define two matrices
matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

# Element-wise division
result = matrix_1 / matrix_2
print(result)
```

Following is the output obtained −

```
[[0.2        0.33333333]
 [0.42857143 0.5      ]]
```

## Element-wise Operations with Scalars

Another useful feature of element-wise operations is performing operations between a matrix and a scalar value.

NumPy allows you to add, subtract, multiply, and divide a scalar value to/from each element of a matrix.

### Example

In the addition operation, the scalar value 2 is added to each element of the matrix. Similarly, in the multiplication operation, the scalar value 2 is multiplied with each element of the matrix in the example below −

```
import numpy as np

# Define a matrix and a scalar
matrix_1 = np.array([[1, 2], [3, 4]])
scalar = 2

# Element-wise addition with scalar
result_add = matrix_1 + scalar
print(result_add)

# Element-wise multiplication with scalar
result_mul = matrix_1 * scalar
print(result_mul)
```

Following is the output obtained −

```
Element-wise addition with scalar:
[[3 4]
 [5 6]]

Element-wise multiplication with scalar:
[[2 4]
 [6 8]]
```

## Broadcasting in NumPy

Broadcasting in NumPy allows operations to be performed on matrices of different shapes, as long as the shapes are compatible. When performing an element-wise operation between matrices of different shapes, NumPy automatically "broadcasts" the smaller matrix across the larger one.

For example, a 1D array can be added to a 2D matrix, and NumPy will repeat the 1D array across all rows of the matrix to perform the operation. However, there are specific rules that determine whether broadcasting is possible.

### Example

Let us say we have a 2D matrix and a 1D array. NumPy will broadcast the 1D array across the 2D matrix for element-wise operations −

```
import numpy as np

# Define two matrices
matrix_1 = np.array([[1, 2], [3, 4]])
array_1 = np.array([5, 6])

# Broadcasting example
result = matrix_1 + array_1
print(result)
```

Following is the output obtained −

```
[[ 6  8]
 [ 8 10]]
```

---

## 72. NumPy - Dot Product

*Source: [https://www.tutorialspoint.com/numpy/numpy_dot_product.htm](https://www.tutorialspoint.com/numpy/numpy_dot_product.htm)*

---

---
[Previous](/numpy/numpy_element_wise_matrix_operations.htm)[Quiz](/numpy/quiz_on_numpy_dot_product.htm)[Next](/numpy/numpy_matrix_inversion.htm)
## What is the Dot Product?

The dot product, also known as the scalar product, is a mathematical operation that takes two equal-length sequences of numbers (usually vectors) and returns a single number.

In the context of matrices, the dot product is used to perform matrix multiplication, which is a fundamental operation in many areas of mathematics, physics, and engineering.

The dot product of two vectors
**a**and**b**is defined as −
```
a . b = a1b1 + a2b2 + ... + anbn
```
b+ ab+ ... + ab
Where,
**a**and**b**are the components of vectors**a**and**b**respectively, and**n**is the number of dimensions.
## Matrix Multiplication Using Dot Product

In matrix multiplication, the dot product is used to multiply the rows of the first matrix by the columns of the second matrix. This produces a new matrix where each element is the dot product of the corresponding row and column vectors.

Consider two matrices
**A**and**B**−
```
A = [[a11, a12],
         [a21, a22]]

B = [[b11, b12],
         [b21, b22]]
```
, a],
         [a, a]]

B = [[b, b],
         [b, b]]
The product
**C**=**A**.**B**is −
```
C = [[a11b11 + a12b21, a11b12 + a12b22],
         [a21b11 + a22b21, a21b12 + a22b22]]
```
b+ ab, ab+ ab],
         [ab+ ab, ab+ ab]]
## Using NumPy for Dot Product

NumPy provides a convenient way to perform dot products using the
**dot()**function. This function can be used for both vector dot products and matrix multiplication.
### Example

In the following example, the dot product is calculated as (1 * 4) + (2 * 5) + (3 * 6) = 32 −

```
import numpy as np

# Define two vectors
vector_1 = np.array([1, 2, 3])
vector_2 = np.array([4, 5, 6])

# Compute dot product
dot_product = np.dot(vector_1, vector_2)
print(dot_product)
```

Following is the output obtained −

```
32
```

## Matrix Dot Product

To compute the dot product of two matrices, we use the same
**dot()**function.
### Example

In this example, the dot product of the two matrices is computed as −

```
[[1*5 + 2*7, 1*6 + 2*8],
 [3*5 + 4*7, 3*6 + 4*8]]
```

```
import numpy as np

# Define two matrices
matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

# Compute dot product
matrix_product = np.dot(matrix_1, matrix_2)
print(matrix_product)
```

Following is the output obtained −

```
[[19 22]
 [43 50]]
```

## Dot Product with Higher Dimensional Arrays

NumPy's
**dot()**function can also handle higher-dimensional arrays. In this case, the function computes the dot product over the last axis of the first array and the second-to-last axis of the second array.
### Example

In this example, the dot product is computed for each pair of sub-arrays, resulting in a new 3-dimensional array −

```
import numpy as np

# Define two 3-dimensional arrays
array_1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
array_2 = np.array([[[1, 0], [0, 1]], [[1, 1], [1, 0]]])

# Compute dot product
array_product = np.dot(array_1, array_2)
print(array_product)
```

Following is the output obtained −

```
[[[[ 1  2]
   [ 3  1]]

  [[ 3  4]
   [ 7  3]]]

 [[[ 5  6]
   [11  5]]

  [[ 7  8]
   [15  7]]]]
```

## Using the @ Operator for Dot Product

In Python 3.5 and later, the
**@**operator can be used as an alternative to the**dot()**function for matrix multiplication. This makes the code more readable and concise.
### Example

The result of the following example is the same as using the
**dot()**function, but the syntax is more cleaner −
```
import numpy as np

# Define two matrices
matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

# Using @ operator for matrix multiplication
matrix_product = matrix_1 @ matrix_2
print(matrix_product)
```

Following is the output obtained −

```
[[19 22]
 [43 50]]
```

## Applications of Dot Product

The dot product is a fundamental operation with a lot of applications in various fields −

- **Machine Learning:**Dot products are used in calculating the similarity between vectors, which is crucial in algorithms like support vector machines and neural networks.
- **Physics:**Dot products are used to compute work done by a force and to project vectors in different directions.
- **Computer Graphics:**Dot products are used in shading calculations and to determine angles between surfaces and light sources.
- **Linear Algebra:**Dot products are foundational in solving systems of linear equations and in transformations.
### Example: Using Dot Product in Machine Learning

In machine learning, dot products are often used to compute the weights and biases in neural networks.

In this example, the dot product computes the weighted sum of the input features, which is an important step in the computation of neural network outputs −

```
import numpy as np

# Define input vector (features)
input_vector = np.array([0.5, 1.5, -1.0])

# Define weight vector (weights)
weights = np.array([2.0, -1.0, 0.5])

# Compute the weighted sum (dot product)
output = np.dot(input_vector, weights)
print(output)
```

Following is the output obtained −

```
-1.0
```

---

## 73. NumPy - Matrix Inversion

*Source: [https://www.tutorialspoint.com/numpy/numpy_matrix_inversion.htm](https://www.tutorialspoint.com/numpy/numpy_matrix_inversion.htm)*

---

---
[Previous](/numpy/numpy_dot_product.htm)[Quiz](/numpy/quiz_on_numpy_matrix_inversion.htm)[Next](/numpy/numpy_determinant_calculation.htm)
## What is Matrix Inversion?

Matrix inversion is a process of finding a matrix, called the inverse matrix, which, when multiplied with the original matrix, produces the identity matrix. The identity matrix is a square matrix with ones on the main diagonal and zeros elsewhere.

Not all matrices have inverses. A matrix must be square (having the same number of rows and columns) and its determinant must be non-zero to have an inverse.

If
**A**is a square matrix, its inverse is denoted by**A**and is defined by the following property −
```
A . A-1 = A-1 . A = I
```
= A. A = I
Where
**I**is the identity matrix of the same dimension as**A**. This property means that when a matrix is multiplied by its inverse, the result is the identity matrix.
## Matrix Inversion in NumPy

NumPy provides the
**numpy.linalg.inv()**function to compute the inverse of a matrix. Let us see how this function works.
### Example

In the following example, the inverse of the matrix
**A**is computed using the**numpy.linalg.inv()**function. The result is a new matrix that satisfies the property**A . A**−= I
```
import numpy as np

# Define a square matrix
A = np.array([[1, 2], [3, 4]])

# Compute the inverse of the matrix
A_inv = np.linalg.inv(A)
print(A_inv)
```

Following is the output obtained −

```
[[-2.   1. ]
 [ 1.5 -0.5]]
```

## Verifying the Inverse

We can verify that the computed matrix is indeed the inverse by multiplying it with the original matrix and checking if the result is the identity matrix.

```
import numpy as np
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)

# Verifying the inverse
identity_matrix = np.dot(A, A_inv)
print(identity_matrix)
```

The output is the identity matrix, confirming that
**A**is the correct inverse of**A**−
```
[[1.0000000e+00 0.0000000e+00]
 [0.0000000e+00 1.0000000e+00]]
```

## Properties of Matrix Inversion

Matrix inversion has several important properties. They are as follows −

- **Uniqueness:**If a matrix has an inverse, it is unique.
- **Product of Inverses:**The inverse of a product of two matrices is the product of their inverses in reverse order:**(AB)**.= BA
- **Inverse of Transpose:**The inverse of the transpose of a matrix is the transpose of the inverse:**(A**.)= (A)
## Conditions for Matrix Inversion

Not all matrices can be inverted. For a matrix to have an inverse, it must meet the following conditions −

- **Square Matrix:**The matrix must have the same number of rows and columns.
- **Non-zero Determinant:**The determinant of the matrix must be non-zero. A matrix with a zero determinant is called singular and does not have an inverse.
## Matrix Inversion in Linear Equations

Matrix inversion is often used to solve systems of linear equations. If we have a system of equations represented by
**AX = B**, where**A**is the coefficient matrix,**X**is the vector of unknowns, and**B**is the constant vector, we can solve for**X**by multiplying both sides of the equation by**A**−
```
X = A-1 . B
```
. B
Following is an example to implement the same −

```
import numpy as np

# Coefficient matrix
A = np.array([[1, 2], [3, 4]])

# Constant vector
B = np.array([[5], [6]])

# Compute the inverse of A
A_inv = np.linalg.inv(A)

# Solve for X
X = np.dot(A_inv, B)
print(X)
```

This will produce the following result −

```
[[-4. ]
 [ 4.5]]
```

## Handling Non-Invertible Matrices

Sometimes, we may encounter matrices that are not invertible. In such cases, attempting to compute the inverse will result in an error. Here is how we can handle such scenarios using NumPy −

```
import numpy as np

def invert_matrix(matrix):
   try:
      return np.linalg.inv(matrix)
   except np.linalg.LinAlgError:
      return "Matrix is not invertible."

# Non-invertible matrix
A = np.array([[1, 2], [2, 4]])

# Attempt to compute the inverse
result = invert_matrix(A)
print(result)
```

Following is the output obtained −

```
Matrix is not invertible.
```

## Practical Applications of Matrix Inversion

Matrix inversion has many practical applications, they are −

- **Solving Systems of Linear Equations:**As shown earlier, matrix inversion can be used to solve systems of linear equations.
- **Computer Graphics:**In computer graphics, transformations such as rotation, scaling, and translation are often represented by matrices. Inverting these matrices can help revert the transformations.
- **Control Theory:**In control theory, matrix inversion is used to solve state-space representations of dynamic systems.

---

## 74. NumPy - Determinant Calculation

*Source: [https://www.tutorialspoint.com/numpy/numpy_determinant_calculation.htm](https://www.tutorialspoint.com/numpy/numpy_determinant_calculation.htm)*

---

---
[Previous](/numpy/numpy_matrix_inversion.htm)[Quiz](/numpy/quiz_on_numpy_determinant_calculation.htm)[Next](/numpy/numpy_eigenvalues.htm)
## What is a Matrix Determinant?

A matrix determinant is a special number that can be calculated from a square matrix. It provides important information about the matrix, such as whether the matrix is invertible.

The determinant of a matrix is a single number, not a matrix, and it can be used to solve systems of linear equations, find eigenvalues, and more.

For example, the determinant of a 2x2 matrix
**A**is calculated as follows −
```
A = [[a, b],
         [c, d]]
Det(A) = ad - bc
```

For larger matrices, the determinant is calculated recursively using a process called
**cofactor expansion**.
## Determinant Calculation in NumPy

NumPy provides the
**numpy.linalg.det()**function to compute the determinant of a matrix. Let us see how this function works with an example of a 2x2 matrix.
### Example

In the following example, the determinant of the matrix
**A**is computed using the**numpy.linalg.det()**function −
```
import numpy as np

# Define a 2x2 matrix
A = np.array([[1, 2], [3, 4]])

# Compute the determinant of the matrix
det_A = np.linalg.det(A)
print(det_A)
```

Following is the output obtained −

```
-2.0000000000000004
```

## Determinant of a 3x3 Matrix

The determinant of a 3x3 matrix is calculated using a more complex formula that involves minors and cofactors. Following is an example of calculating the determinant of a 3x3 matrix using NumPy −

```
import numpy as np

# Define a 3x3 matrix
B = np.array([[6, 1, 1], 
              [4, -2, 5], 
              [2, 8, 7]])

# Compute the determinant of the matrix
det_B = np.linalg.det(B)
print(det_B)
```

This will produce the following result −

```
-306.0
```

## Properties of Determinants

Determinants have several important properties. They are as follows −

- **Multiplicative Property:**The determinant of the product of two matrices is equal to the product of their determinants:**det(AB) = det(A) . det(B)**.
- **Transpose Property:**The determinant of a matrix is equal to the determinant of its transpose:**det(A) = det(A**.)
- **Inverse Property:**The determinant of the inverse of a matrix is the reciprocal of the determinant:**det(A**.) = 1 / det(A)
- **Row Operations:**Swapping two rows of a matrix multiplies its determinant by -1, scaling a row by a constant multiplies the determinant by that constant, and adding a multiple of one row to another does not change the determinant.
## Conditions for a Non-Zero Determinant

For a matrix to have a non-zero determinant, it must meet the following conditions −

- **Square Matrix:**The matrix must have the same number of rows and columns.
- **Full Rank:**The matrix must have full rank, meaning all its rows (or columns) are linearly independent.
## Using Determinants in Linear Equations

Determinants are used to solve systems of linear equations using Cramer's Rule. If we have a system of equations represented by
**AX = B**, where**A**is the coefficient matrix,**X**is the vector of unknowns, and**B**is the constant vector, we can solve for**X**if**det(A)  0**−
```
import numpy as np

# Coefficient matrix
A = np.array([[2, 1], [5, 7]])

# Constant vector
B = np.array([11, 13])

# Solve for X using Cramer's Rule
det_A = np.linalg.det(A)
A1 = np.array([[11, 1], [13, 7]])
A2 = np.array([[2, 11], [5, 13]])

X1 = np.linalg.det(A1) / det_A
X2 = np.linalg.det(A2) / det_A

X = np.array([X1, X2])
print(X)
```

Following is the output of the above code −

```
[ 7.11111111 -3.22222222]
```

## Practical Applications of Determinants

Determinants have many practical applications, such as −

- **Solving Systems of Linear Equations:**Determinants are used in Cramer's Rule to solve systems of linear equations.
- **Finding Eigenvalues:**Determinants are used to find the eigenvalues of a matrix.
- **Area and Volume Calculation:**Determinants can be used to calculate the area of parallelograms and the volume of parallelepipeds in higher dimensions.
- **Change of Variables in Integrals:**In multivariable calculus, the determinant of the Jacobian matrix is used in the change of variables formula for integrals.
## Handling Non-Invertible Matrices

If a matrix has a determinant of zero, it is called singular and does not have an inverse. Attempting to compute the inverse of a singular matrix will result in an error. Here is how we can handle such scenarios using NumPy −

```
import numpy as np

def calculate_determinant(matrix):
   try:
      return np.linalg.det(matrix)
   except np.linalg.LinAlgError:
      return "Matrix is singular and has no determinant."

# Singular matrix
C = np.array([[1, 2], [2, 4]])

# Attempt to compute the determinant
result = calculate_determinant(C)
print(result)
```

Following is the output obtained −

```
0.0
```

---

## 75. NumPy - Eigenvalues

*Source: [https://www.tutorialspoint.com/numpy/numpy_eigenvalues.htm](https://www.tutorialspoint.com/numpy/numpy_eigenvalues.htm)*

---

---
[Previous](/numpy/numpy_determinant_calculation.htm)[Quiz](/numpy/quiz_on_numpy_eigenvalues.htm)[Next](/numpy/numpy_eigenvectors.htm)
## What are Eigenvalues?

Eigenvalues are special numbers associated with a matrix that provide important information about the matrix's properties.

In the context of linear algebra, if
**A**is a square matrix, an eigenvalue is a scalarsuch that there exists a non-zero vector**v**(called an eigenvector) satisfying the equation −
```
Av = v
```
**Av = v**
This means that when the matrix
**A**multiplies the vector**v**, the result is the same as multiplying the vector**v**by the scalar.
## Computing Eigenvalues in NumPy

NumPy provides the
**numpy.linalg.eig()**function to compute the eigenvalues and eigenvectors of a square matrix. Let us see how this function works with an example.
### Example

In this example, the eigenvalues of the matrix
**A**are 3 and 2. The corresponding eigenvectors are shown in the output −
```
import numpy as np

# Define a 2x2 matrix
A = np.array([[4, -2], 
              [1,  1]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

The output from
**numpy.linalg.eig()**function provides two arrays: one for eigenvalues and one for eigenvectors.
The eigenvalues array contains the eigenvalues of the matrix, and each column of the eigenvectors array represents an eigenvector corresponding to the respective eigenvalue −

```
Eigenvalues: [3. 2.]
Eigenvectors:
 [[ 0.89442719  0.70710678]
 [ 0.4472136  -0.70710678]]
```

## Properties of Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors have several important properties. They are −

- **Linearity:**Eigenvectors corresponding to different eigenvalues are linearly independent.
- **Determinant Relation:**The product of the eigenvalues of a matrix is equal to its determinant.
- **Trace Relation:**The sum of the eigenvalues of a matrix is equal to its trace (the sum of its diagonal elements).
- **Similarity Transformation:**If a matrix**A**is similar to a matrix**B**(i.e.,**B = P**for some invertible matrixAP**P**), then**A**and**B**have the same eigenvalues.
## Applications of Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors have numerous applications, such as −

- **Principal Component Analysis (PCA):**Used in data analysis and machine learning for dimensionality reduction.
- **Stability Analysis:**Used in control theory to analyze the stability of systems.
- **Quantum Mechanics:**Used to solve the Schrdinger equation and find the energy levels of a system.
- **Vibration Analysis:**Used in engineering to analyze the natural frequencies of structures.
- **Graph Theory:**Used to analyze the properties of graphs and networks.
### Example: Eigenvalues of a 3x3 Matrix

In the following example, we are computing the eigenvalues and eigenvectors of a 3x3 matrix using NumPy −

```
import numpy as np

# Define a 3x3 matrix
B = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(B)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

This will produce the following result −

```
Eigenvalues: [-5.2296696  -0.02635282  7.25602242]
Eigenvectors:
[[ 0.22578016 -0.75769839 -0.49927017]
 [ 0.52634845  0.63212771 -0.46674201]
 [-0.81974424 -0.16219652 -0.72998712]]
```

## Symmetric Matrices and Real Eigenvalues

A symmetric matrix is a matrix that is equal to its transpose (i.e.,
**A = A**). Symmetric matrices have some special properties regarding their eigenvalues −
- **Real Eigenvalues:**The eigenvalues of a symmetric matrix are always real numbers.
- **Orthogonal Eigenvectors:**The eigenvectors of a symmetric matrix corresponding to distinct eigenvalues are orthogonal.
### Example

Let us compute the eigenvalues of a symmetric matrix −

```
import numpy as np

# Define a symmetric matrix
C = np.array([[4, 1, 1],
              [1, 4, 1],
              [1, 1, 4]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(C)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

Following is the output of the above code −

```
Eigenvalues: [6. 3. 3.]
Eigenvectors:
[[-0.57735027 -0.81649658 -0.15430335]
 [-0.57735027  0.40824829 -0.6172134 ]
 [-0.57735027  0.40824829  0.77151675]]
```

## Eigenvalues and Diagonalization

A square matrix
**A**is said to be diagonalizable if it can be written as −
```
A = PDP-1
```

where,
**D**is a diagonal matrix containing the eigenvalues of**A**, and**P**is a matrix whose columns are the eigenvectors of**A**.
### Example

Let us see how to diagonalize a matrix using NumPy −

```
import numpy as np

# Define a matrix
D = np.array([[2, 0, 0],
              [1, 3, 0],
              [4, 5, 6]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(D)

# Diagonal matrix of eigenvalues
D_diag = np.diag(eigenvalues)

# Reconstruct the original matrix
reconstructed_D = eigenvectors @ D_diag @ np.linalg.inv(eigenvectors)

print("Original matrix:\n", D)
print("Reconstructed matrix:\n", reconstructed_D)
```

The original matrix is successfully reconstructed using its eigenvalues and eigenvectors, demonstrating the process of diagonalization −

```
Original matrix:
 [[2 0 0]
 [1 3 0]
 [4 5 6]]
Reconstructed matrix:
 [[2. 0. 0.]
 [1. 3. 0.]
 [4. 5. 6.]]
```

---

## 76. NumPy - Eigenvectors

*Source: [https://www.tutorialspoint.com/numpy/numpy_eigenvectors.htm](https://www.tutorialspoint.com/numpy/numpy_eigenvectors.htm)*

---

---
[Previous](/numpy/numpy_eigenvalues.htm)[Quiz](/numpy/quiz_on_numpy_eigenvectors.htm)[Next](/numpy/numpy_singular_value_decomposition.htm)
## What are Eigenvectors?

Eigenvectors are special vectors associated with a matrix that provide information about the matrix's properties.

In the context of linear algebra, if
**A**is a square matrix, an eigenvector**v**corresponding to an eigenvalueis a non-zero vector that satisfies the equation −
```
Av = v
```

This means that when the matrix
**A**multiplies the vector**v**, the result is the same as multiplying the vector**v**by the scalar.
## Computing Eigenvectors in NumPy

NumPy provides the
**numpy.linalg.eig()**function to compute the eigenvalues and eigenvectors of a square matrix. Let us see how this function works with an example.
### Example

In this example, the eigenvalues of the matrix
**A**are 3 and 2. The corresponding eigenvectors are shown in the output −
```
import numpy as np

# Define a 2x2 matrix
A = np.array([[4, -2], 
              [1,  1]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

The output from
**numpy.linalg.eig()**function contains two arrays: one for eigenvalues and one for eigenvectors.
The eigenvalues array contains the eigenvalues of the matrix, and each column of the eigenvectors array represents an eigenvector corresponding to the respective eigenvalue −

```
Eigenvalues: [3. 2.]
Eigenvectors:
 [[ 0.89442719  0.70710678]
 [ 0.4472136  -0.70710678]]
```

## Properties of Eigenvectors

Eigenvectors have several important properties, they are −

- **Linearity:**Eigenvectors corresponding to different eigenvalues are linearly independent.
- **Scalability:**Any scalar multiple of an eigenvector is also an eigenvector corresponding to the same eigenvalue.
- **Invariance:**Eigenvectors remain unchanged (up to a scalar multiple) under the linear transformation defined by the matrix.
- **Orthogonality:**In the case of symmetric matrices, eigenvectors corresponding to distinct eigenvalues are orthogonal.
## Applications of Eigenvectors

Eigenvectors have numerous applications, they are −

- **Principal Component Analysis (PCA):**Used in data analysis and machine learning for dimensionality reduction.
- **Stability Analysis:**Used in control theory to analyze the stability of systems.
- **Quantum Mechanics:**Used to solve the Schrdinger equation and find the energy levels of a system.
- **Vibration Analysis:**Used in engineering to analyze the natural frequencies of structures.
- **Graph Theory:**Used to analyze the properties of graphs and networks.
### Example: Eigenvectors of a 3x3 Matrix

In the following example, we are computig the eigenvalues and eigenvectors of a 3x3 matrix using NumPy −

```
import numpy as np

# Define a 3x3 matrix
B = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(B)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

This will produce the following result −

```
Eigenvalues: [-5.2296696  -0.02635282  7.25602242]
Eigenvectors:
[[ 0.22578016 -0.75769839 -0.49927017]
 [ 0.52634845  0.63212771 -0.46674201]
 [-0.81974424 -0.16219652 -0.72998712]]
```

## Symmetric Matrices and Real Eigenvectors

A symmetric matrix is a matrix that is equal to its transpose (i.e.,
**A = A**). Symmetric matrices have some special properties regarding their eigenvalues and eigenvectors −
- **Real Eigenvalues:**The eigenvalues of a symmetric matrix are always real numbers.
- **Orthogonal Eigenvectors:**The eigenvectors of a symmetric matrix corresponding to distinct eigenvalues are orthogonal.
### Example

Let us compute the eigenvalues and eigenvectors of a symmetric matrix −

```
import numpy as np

# Define a symmetric matrix
C = np.array([[4, 1, 1],
              [1, 4, 1],
              [1, 1, 4]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(C)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

Following is the output of the above code −

```
Eigenvalues: [6. 3. 3.]
Eigenvectors:
[[-0.57735027 -0.81649658 -0.15430335]
 [-0.57735027  0.40824829 -0.6172134 ]
 [-0.57735027  0.40824829  0.77151675]]
```

## Eigenvectors and Diagonalization

A square matrix
**A**is said to be diagonalizable if it can be written as −
```
A = PDP-1
```

where,
**D**is a diagonal matrix containing the eigenvalues of**A**, and**P**is a matrix whose columns are the eigenvectors of**A**.
### Example

Let us see how to diagonalize a matrix using NumPy −

```
import numpy as np

# Define a matrix
D = np.array([[2, 0, 0],
              [1, 3, 0],
              [4, 5, 6]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(D)

# Diagonal matrix of eigenvalues
D_diag = np.diag(eigenvalues)

# Reconstruct the original matrix
reconstructed_D = eigenvectors @ D_diag @ np.linalg.inv(eigenvectors)

print("Original matrix:\n", D)
print("Reconstructed matrix:\n", reconstructed_D)
```

The original matrix is successfully reconstructed using its eigenvalues and eigenvectors, demonstrating the process of diagonalization −

```
Original matrix:
 [[2 0 0]
 [1 3 0]
 [4 5 6]]
Reconstructed matrix:
 [[2. 0. 0.]
 [1. 3. 0.]
 [4. 5. 6.]]
```

---

## 77. NumPy - Singular Value Decomposition

*Source: [https://www.tutorialspoint.com/numpy/numpy_singular_value_decomposition.htm](https://www.tutorialspoint.com/numpy/numpy_singular_value_decomposition.htm)*

---

---
[Previous](/numpy/numpy_eigenvectors.htm)[Quiz](/numpy/quiz_on_numpy_singular_value_decomposition.htm)[Next](/numpy/numpy_solving_linear_equations.htm)
## What is Singular Value Decomposition (SVD)?

Singular Value Decomposition, commonly abbreviated as SVD, is a matrix factorization technique in linear algebra. SVD decomposes a matrix into three other matrices, capturing important properties of the original matrix.

For instance, if you have a matrix
**A**, the SVD is given by −
```
A = UVT
```

Here,
**U**and**V**are orthogonal matrices, andis a diagonal matrix.
The columns of
**U**are called the left singular vectors, the columns of**V**(or rows of**V**) are the right singular vectors, and the entries ofare the singular values.
## SVD in NumPy

NumPy provides the
**numpy.linalg.svd()**function to compute the Singular Value Decomposition of a matrix. Let us see how to use this function with an example.
### Example

In this example, the matrix
**A**is decomposed into three matrices:**U**,(represented as the array of singular values**S**), and**V**−
```
import numpy as np

# Define a 3x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Compute the Singular Value Decomposition
U, S, VT = np.linalg.svd(A)

print("Matrix U:\n", U)
print("Singular values:", S)
print("Matrix V^T:\n", VT)
```

Following is the output obtained −

```
Matrix U:
[[-0.21483724  0.88723069  0.40824829]
 [-0.52058739  0.24964395 -0.81649658]
 [-0.82633754 -0.38794278  0.40824829]]
Singular values: [1.68481034e+01 1.06836951e+00 4.41842475e-16]
Matrix V^T:
[[-0.47967118 -0.57236779 -0.66506441]
 [-0.77669099 -0.07568647  0.62531805]
 [-0.40824829  0.81649658 -0.40824829]]
```

## Understanding the Components

The SVD components have specific properties and roles as shown below −

- **Matrix U:**The columns of**U**are the left singular vectors of**A**. These vectors form an orthogonal basis for the column space of**A**.
- **Singular values:**The diagonal entries ofare the singular values of**A**. These values give the magnitude of the action of**A**along the corresponding singular vectors.
- **Matrix V**The rows of:**V**are the right singular vectors of**A**. These vectors form an orthogonal basis for the row space of**A**.
## Reconstructing the Original Matrix

You can reconstruct the original matrix
**A**from its SVD components. In NumPy, you can achieve this by using the**numpy.dot()**function to perform matrix multiplication.
### Example

In the following example, we are reconstructing the original matrix "A" −

```
import numpy as np

# Define a 3x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Compute the Singular Value Decomposition
U, S, VT = np.linalg.svd(A)

# Create the diagonal matrix  from the singular values
Sigma = np.zeros((3, 3))
np.fill_diagonal(Sigma, S)

# Reconstruct the original matrix
A_reconstructed = np.dot(U, np.dot(Sigma, VT))

print("Original matrix:\n", A)
print("Reconstructed matrix:\n", A_reconstructed)
```

The original matrix
**A**is successfully reconstructed using its SVD components, demonstrating the accuracy of the decomposition −/p>
```
Original matrix:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Reconstructed matrix:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

## Applications of SVD

SVD is a powerful tool with numerous applications, such as −

- **Dimensionality Reduction:**In data analysis and machine learning, SVD is used to reduce the number of dimensions while preserving important information.
- **Image Compression:**SVD is applied to compress images by reducing the amount of data required to store them.
- **Noise Reduction:**SVD can help remove noise from data by identifying and discarding small singular values.
- **Signal Processing:**In signal processing, SVD is used to analyze and filter signals.
- **Recommendation Systems:**SVD is employed in recommendation systems to predict user preferences.
### Example: Image Compression using SVD

Let us see an example of how SVD can be used for image compression. We will use a grayscale image and compress it by retaining only the most significant singular values −

```
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# Load a sample image and convert it to grayscale
image = color.rgb2gray(data.astronaut())  
# Compute the Singular Value Decomposition
U, S, VT = np.linalg.svd(image, full_matrices=False)

# Retain only the first k singular values
k = 50
U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

# Reconstruct the compressed image
image_compressed = np.dot(U_k, np.dot(S_k, VT_k))

# Plot the original and compressed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Compressed Image with k={k}")
plt.imshow(image_compressed, cmap='gray')
plt.axis('off')

plt.show()
```

The original image and the compressed image are displayed side by side, demonstrating how SVD can reduce the size of the image while preserving its essential features −
![SVD Compression](/numpy/images/image_compression_output.jpg)
## Advantages of SVD

SVD provides several advantages, such as −

- **Numerical Stability:**SVD is numerically stable and can handle ill-conditioned matrices.
- **Optimal Low-Rank Approximation:**SVD provides the best low-rank approximation of a matrix, making it ideal for dimensionality reduction.
- **Robustness:**SVD is robust to small perturbations in the data.
- **Versatility:**SVD can be applied to any matrix, regardless of its properties.

---

## 78. NumPy - Solving Linear Equations

*Source: [https://www.tutorialspoint.com/numpy/numpy_solving_linear_equations.htm](https://www.tutorialspoint.com/numpy/numpy_solving_linear_equations.htm)*

---

---
[Previous](/numpy/numpy_singular_value_decomposition.htm)[Quiz](/numpy/quiz_on_numpy_solving_linear_equations.htm)[Next](/numpy/numpy_matrix_norms.htm)
## What is Solving Linear Equations?

Linear equations are mathematical equations that involve linear terms, meaning the highest power of the variable is one.

A system of linear equations can be expressed as a set of equations with multiple variables. The goal is to find the values of these variables that satisfy all equations simultaneously.

In matrix form, a system of linear equations can be represented as −

```
A * x = b
```

Here,

- **A**: A matrix representing the coefficients of the linear equations.
- **x**: A column vector representing the unknown variables.
- **b**: A column vector representing the constants on the right-hand side of the equations.
The goal is to find the vector
**x**, which contains the values of the unknown variables. NumPy provides methods to solve such systems of equations using matrix operations.
## Solving Linear Equations in NumPy

NumPy provides several methods to solve linear equations. The most commonly used method is by using the
**numpy.linalg.solve()**function, which directly solves the system of linear equations.
### Example

In the following example, the system of equations −

```
3x + 2y = 5
x + 2y = 5
```

has been solved to give the solution
**x = 1**and**y = 2**.
```
import numpy as np

# Define the coefficient matrix A and constant vector b
A = np.array([[3, 2], [1, 2]])
b = np.array([5, 5])

# Solve the system of linear equations
x = np.linalg.solve(A, b)

print("Solution vector x:", x)
```

Following is the output obtained −

```
Solution vector x: [0.  2.5]
```

## The numpy.linalg.solve() Function

The
**numpy.linalg.solve(A, b)**function computes the solution to the linear system**A * x = b**. The function takes two arguments:
- **A**: The coefficient matrix (2D NumPy array).
- **b**: The constant vector (1D NumPy array).
The function returns the solution vector
**x**that satisfies the equation. The function uses efficient methods, such as Gaussian elimination or LU decomposition, to solve the system.
## Alternative Ways to Solve Linear Equations

In addition to
**numpy.linalg.solve()**function, NumPy provides other ways to solve linear equations, such as using matrix inversion or the**numpy.dot()**function.
These methods are useful when you need more control over the solving process or want to explore the mathematical background of solving linear systems.

### Using Matrix Inversion

One way to solve a system of linear equations is by finding the inverse of the coefficient matrix
**A**. If**A**is invertible, the solution can be found by multiplying the inverse of**A**with the vector**b**:
```
x = A-1 * b
```
* b
NumPy provides the
**numpy.linalg.inv()**function to compute the inverse of a matrix. Let us see how to use it −
```
import numpy as np

# Define the coefficient matrix A and constant vector b
A = np.array([[3, 2], [1, 2]])
b = np.array([5, 5])

# Compute the inverse of A
A_inv = np.linalg.inv(A)

# Solve the system of equations using matrix inversion
x = np.dot(A_inv, b)

print("Solution vector x:", x)
```

Following is the output obtained −

```
Solution vector x: [4.4408921e-16 2.5000000e+00]
```

> This method also produces the same result as the numpy.linalg.solve() function. However, using matrix inversion is more expensive and less stable for large matrices.
**numpy.linalg.solve()**function. However, using matrix inversion is more expensive and less stable for large matrices.
### Using numpy.linalg.lstsq() Function

If the system of equations is overdetermined (more equations than unknowns), you can use the least squares solution.

The
**numpy.linalg.lstsq()**function is used to find the least squares solution to such systems. It minimizes the error between the observed and predicted values.
Let us look at an example −

```
import numpy as np

# Define an overdetermined system (more equations than unknowns)
A = np.array([[1, 1], [2, 1], [3, 1]])
b = np.array([6, 8, 10])

# Solve the system using the least squares method
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print("Solution vector x:", x)
```

The least squares solution minimizes the error in the system, and in this case, it finds the best-fit values for the unknowns
**x**and**y**−
```
Solution vector x: [2. 4.]
```

## Applications of Solving Linear Equations

Solving linear equations has numerous applications in various fields, such as −

- **Physics:**Linear equations are used to model physical systems, such as circuits, motion, and energy transfer.
- **Economics:**Economists use linear systems to model relationships between variables like supply and demand, production, and consumption.
- **Computer Graphics:**In graphics programming, linear equations are used in transformations, rendering, and 3D modeling.
- **Machine Learning:**Solving linear equations is a crucial step in algorithms such as linear regression and optimization problems.
## Advantages of Solving Linear Equations

Using NumPy to solve linear equations has several advantages, they are −

- **Efficiency:**NumPy is optimized for performance, making it much faster than manually solving equations.
- **Ease of Use:**NumPy provides simple functions like**linalg.solve()**and**linalg.lstsq()**functions that handle complex calculations with minimal effort.
- **Robustness:**NumPy handles edge cases, such as singular or ill-conditioned matrices, efficiently.
- **Versatility:**NumPy can handle systems with any number of equations and unknowns, making it applicable to a wide range of problems.

---

## 79. NumPy - Matrix Norms

*Source: [https://www.tutorialspoint.com/numpy/numpy_matrix_norms.htm](https://www.tutorialspoint.com/numpy/numpy_matrix_norms.htm)*

---

---
[Previous](/numpy/numpy_solving_linear_equations.htm)[Quiz](/numpy/quiz_on_numpy_matrix_norms.htm)[Next](/numpy/numpy_sum.htm)
## What are Matrix Norms?

A matrix norm is a function that assigns a non-negative number to a matrix. It provides a measure of the size or magnitude of a matrix.

In general, matrix norms are used to quantify how large or small a matrix is, and they play an important role in problems involving matrix equations, such as in solving systems of linear equations or performing matrix factorizations.

## Common Types of Matrix Norms

There are several types of matrix norms, but the most commonly used ones are −

- Frobenius norm
- 1-norm
- Infinity norm
- 2-norm (spectral norm)
## Frobenius Norm

The Frobenius norm is one of the simplest and most commonly used matrix norms. It is defined as the square root of the sum of the absolute squares of the matrix elements. Mathematically, it is given by −

```
&Verbar;A&Verbar;F = √(i=1 j=1 |aij|2)
```
= √(|a|)
Where
**A**is the matrix, and**a**are the elements of the matrix. The Frobenius norm is equivalent to the**L2**norm of the matrix treated as a vector.
## 1-Norm

The 1-norm (also called the maximum column sum norm) of a matrix is defined as the maximum absolute column sum. Mathematically, it is given by −

```
&Verbar;A&Verbar;1 = maxj i=1 |aij|
```
= max|a|
In simple terms, the 1-norm is the maximum sum of the absolute values of the elements in any column of the matrix.

## Infinity Norm

The infinity norm (also called the maximum row sum norm) of a matrix is defined as the maximum absolute row sum. Mathematically, it is given by −

```
&Verbar;A&Verbar;∞ = maxi j=1 |aij|
```
= max|a|
The infinity norm gives the maximum sum of absolute values of elements in any row of the matrix.

## 2-Norm (Spectral Norm)

The 2-norm (also called the spectral norm) of a matrix is defined as the largest singular value of the matrix. It measures the largest stretch factor of the matrix when applied to a vector. The 2-norm is given by −

```
&Verbar;A&Verbar;2 = max(A)
```
=(A)
Where,
is the largest singular value of matrix(A)**A**. In this case, the 2-norm is related to the matrix's singular values, which can be computed using singular value decomposition (SVD).
## Matrix Norms in NumPy

NumPy provides functions for calculating various matrix norms. The
**numpy.linalg.norm()**function can be used to compute most of the common matrix norms. Let us explore how to use this function for different types of matrix norms.
## Frobenius Norm Using NumPy

To compute the Frobenius norm using NumPy, we use the
**numpy.linalg.norm()**function with the parameter**ord='fro'**.
### Example

In the following example, the Frobenius norm of the matrix
**A**is calculated by taking the square root of the sum of the squares of all elements in the matrix −
```
import numpy as np

# Define a matrix A
A = np.array([[1, 2], [3, 4]])

# Compute the Frobenius norm of the matrix
frobenius_norm = np.linalg.norm(A, 'fro')

print("Frobenius norm of A:", frobenius_norm)
```

Following is the output obtained −

```
Frobenius norm of A: 5.477225575051661
```

## 1-Norm Using NumPy

To compute the 1-norm, we use the
**numpy.linalg.norm()**function with the parameter**ord=1**. The 1-norm of the matrix is the maximum sum of the absolute values of the elements in any column of the matrix.
### Example

In this case, the column sums are 4 and 6, so the 1-norm is 6 −

```
import numpy as np

# Define a matrix A
A = np.array([[1, 2], [3, 4]])

# Compute the 1-norm of the matrix
one_norm = np.linalg.norm(A, 1)

print("1-norm of A:", one_norm)
```

Following is the output obtained −

```
1-norm of A: 6.0
```

## Infinity Norm Using NumPy

To compute the infinity norm, we use the
**numpy.linalg.norm()**function with the parameter**ord=np.inf**. The infinity norm of the matrix is the maximum sum of the absolute values of the elements in any row of the matrix.
### Example

In this case, the row sums are 3 and 7, so the infinity norm is 7 −

```
import numpy as np

# Define a matrix A
A = np.array([[1, 2], [3, 4]])

# Compute the infinity norm of the matrix
infinity_norm = np.linalg.norm(A, np.inf)

print("Infinity norm of A:", infinity_norm)
```

Following is the output obtained −

```
Infinity norm of A: 7.0
```

## 2-Norm Using NumPy

To compute the 2-norm (spectral norm), we use the
**numpy.linalg.norm()**function with the parameter**ord=2**. The 2-norm (spectral norm) of the matrix is the largest singular value of the matrix, which measures the largest stretch factor of the matrix when applied to a vector.
### Example

Following is an example to compute the 2-norm in NumPy −

```
import numpy as np

# Define a matrix A
A = np.array([[1, 2], [3, 4]])

# Compute the 2-norm of the matrix
two_norm = np.linalg.norm(A, 2)

print("2-norm (spectral norm) of A:", two_norm)
```

Following is the output obtained −

```
2-norm (spectral norm) of A: 5.464985704219043
```

## Applications of Matrix Norms

Matrix norms have many practical applications in numerical analysis, machine learning, optimization, and more −

- **Numerical Stability:**Matrix norms are used to analyze the stability of numerical algorithms, especially when solving linear systems or performing matrix factorizations.
- **Machine Learning:**In machine learning, matrix norms are often used to regularize models and prevent overfitting. For example,**L2 regularization**uses the Frobenius norm to penalize large weights in a model.
- **Optimization:**Matrix norms are used to measure the error or deviation from the desired solution in optimization problems.
- **Signal Processing:**In signal processing, matrix norms are used to measure the "energy" or magnitude of signals and filters.

---

## 80. NumPy - Sum

*Source: [https://www.tutorialspoint.com/numpy/numpy_sum.htm](https://www.tutorialspoint.com/numpy/numpy_sum.htm)*

---

---
[Previous](/numpy/numpy_matrix_norms.htm)[Quiz](/numpy/quiz_on_numpy_sum.htm)[Next](/numpy/numpy_mean.htm)
## What is Sum?

In mathematics, a sum is the result of adding two or more numbers together. For example, the sum of 2 and 3 is 5.

It is often represented using the plus symbol (+). Summation can also involve adding a sequence of numbers, often using the Greek letter sigma () to denote the operation.

## The NumPy sum() Function

The sum() function in NumPy calculates the sum of array elements along a specified axis, providing flexibility to sum across rows, columns, or the entire array.

Following is the basic syntax of the sum() function in NumPy −

```
numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False)
```

Where,

- **a:**The input array containing the elements to sum.
- **axis:**The axis along which to sum. If**None**, it sums all the elements of the array. For multi-dimensional arrays, you can specify an axis (0 for rows, 1 for columns, etc.).
- **dtype:**The data type to use for the sum. If not specified, it defaults to the data type of the array.
- **out:**A location where the result will be stored. If provided, it must be of the same shape and type as the input array.
- **keepdims:**If**True**, the reduced axes are kept in the result as dimensions with size one. This is useful for broadcasting.
## Summing All Elements of a 1D Array

If you have a one-dimensional array, you can use the
**numpy.sum()**function to calculate the sum of all its elements. Following is an example −
```
import numpy as np

# Define a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Calculate the sum of all elements
total_sum = np.sum(arr)

print("Total sum of the array:", total_sum)
```

Following is the output obtained −

```
Total sum of the array: 15
```

## Summing Along a Specific Axis in a 2D Array

In a two-dimensional array, you can compute the sum along a specific axis. For example, summing along the rows or columns −

```
import numpy as np

# Define a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Sum along rows (axis=1)
sum_rows = np.sum(arr_2d, axis=1)

# Sum along columns (axis=0)
sum_columns = np.sum(arr_2d, axis=0)

print("Sum along rows:", sum_rows)
print("Sum along columns:", sum_columns)
```

Following is the output obtained −

```
Sum along rows: [ 6 15 24]
Sum along columns: [12 15 18]
```

## Summing with a Specified Data Type

You can also specify the data type in which you want the sum to be computed. This is especially useful when dealing with large numbers or when you need the result in a specific precision (such as float64). Here is an example −

```
import numpy as np

# Define an array of integers
arr_int = np.array([10, 20, 30])

# Calculate the sum with a specified data type (float64)
sum_float = np.sum(arr_int, dtype=np.float64)

print("Sum with dtype float64:", sum_float)
```

Following is the output obtained −

```
Sum with dtype float64: 60.0
```

## Summing with "Keepdims" Parameter

The
**keepdims**parameter helps preserve the dimensionality of the original array after the sum operation. If set to**True**, the result will have the same number of dimensions as the input array, but the size of the summed axes will be reduced to one.
```
import numpy as np

# Define a 2D array 
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Sum along columns while keeping dimensions
sum_keepdims = np.sum(arr_2d, axis=0, keepdims=True)

print("Sum with keepdims=True:", sum_keepdims)
```

Following is the output obtained −

```
Sum with keepdims=True: [[12 15 18]]
```

## Applications of NumPy Sum

The
**numpy.sum()**function has a wide range of applications in scientific computing, data analysis, and machine learning. Some common use cases are −
- **Summing over rows or columns in matrices:**In data science, you often need to calculate sums along specific axes to summarize data in tables or matrices.
- **Computing total values in an array:**Summing elements in an array can help in financial analysis, statistics, and scientific computations, such as calculating the total of measurements or quantities.
- **Data aggregation:**When analyzing data, summing values can be part of aggregation operations, such as finding total sales or calculating the cumulative sum of some data points.
- **Feature scaling:**In machine learning, the sum of features is often used in data normalization or scaling to adjust the range of features.
## Optimizing the Sum Calculation

NumPy is optimized for fast array operations, and the
**numpy.sum()**function is highly efficient. However, there are a few ways to further optimize your sum calculations −
- **Using the**If you want to store the result of the sum in a pre-existing array, you can use the**out**parameter:**out**parameter, which avoids creating a new array and helps save memory.
- **Using**Specify the axis only when necessary. Summing over the whole array by default is the fastest operation, but summing along specific axes might be slower depending on the data.**axis**wisely:

---

## 81. NumPy - Mean

*Source: [https://www.tutorialspoint.com/numpy/numpy_mean.htm](https://www.tutorialspoint.com/numpy/numpy_mean.htm)*

---

---
[Previous](/numpy/numpy_sum.htm)[Quiz](/numpy/quiz_on_numpy_mean.htm)[Next](/numpy/numpy_median.htm)
## What is Mean?

In mathematics, the
**mean**is the average value of a set of numbers. The most common type is the arithmetic mean, which is the sum of the numbers divided by the count of the numbers.
Other types include the geometric mean (nth root of the product of the numbers) and the harmonic mean (number of values divided by the sum of reciprocals).

These different means are used based on the nature of the data and specific needs of the analysis.

## The NumPy mean() Function

The mean() function in NumPy calculates the arithmetic mean (average) of the elements in an array. By default, it computes the mean of all elements, but you can specify an axis to compute the mean along rows or columns.

It can also handle different data types and allow you to define the output type. For example, np.mean([1, 2, 3, 4]) returns 2.5.

Following is the basic syntax of the mean() function in NumPy −

```
numpy.mean(a, axis=None, dtype=None, out=None, keepdims=False)
```

Where,

- **a:**The input array containing the elements for which the mean is to be calculated.
- **axis:**The axis along which to compute the mean. If**None**, it computes the mean of all the elements in the array. For multi-dimensional arrays, you can specify an axis (0 for rows, 1 for columns, etc.).
- **dtype:**The data type to use in computing the mean. If not specified, it defaults to the data type of the input array.
- **out:**A location where the result will be stored. If provided, it must be of the same shape and type as the expected output.
- **keepdims:**If**True**, the reduced axes are kept in the result as dimensions with size one. This is useful for broadcasting.
## Calculating the Mean of a 1D Array

If you have a one-dimensional array, you can use the
**numpy.mean()**function to calculate the mean of its elements. Here is an example −
```
import numpy as np

# Define a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Calculate the mean of all elements
mean_value = np.mean(arr)

print("Mean of the array:", mean_value)
```

Following is the output obtained −

```
Mean of the array: 3.0
```

## Mean Along a Specific Axis in a 2D Array

In a two-dimensional array, you can compute the mean along a specific axis. For example, calculating the mean along rows or columns −

```
import numpy as np

# Define a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Mean along rows (axis=1)
mean_rows = np.mean(arr_2d, axis=1)

# Mean along columns (axis=0)
mean_columns = np.mean(arr_2d, axis=0)

print("Mean along rows:", mean_rows)
print("Mean along columns:", mean_columns)
```

Following is the output obtained −

```
Mean along rows: [2. 5. 8.]
Mean along columns: [4. 5. 6.]
```

## Calculating Mean with a Specified Data Type

You can also specify the data type in which you want the mean to be computed. This is especially useful when dealing with large numbers or when you need the result in a specific precision (such as float64). Here is an example −

```
import numpy as np

# Define an array of integers
arr_int = np.array([10, 20, 30])

# Calculate the mean with a specified data type (float64)
mean_float = np.mean(arr_int, dtype=np.float64)

print("Mean with dtype float64:", mean_float)
```

Following is the output obtained −

```
Mean with dtype float64: 20.0
```

## Calculating Mean with Keepdims Parameter

The
**keepdims**parameter helps preserve the dimensionality of the original array after the mean operation. If set to**True**, the result will have the same number of dimensions as the input array, but the size of the reduced axes will be one.
```
import numpy as np

# Define a 2D array 
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Mean along columns while keeping dimensions
mean_keepdims = np.mean(arr_2d, axis=0, keepdims=True)

print("Mean with keepdims=True:", mean_keepdims)
```

Following is the output obtained −

```
Mean with keepdims=True: [[4. 5. 6.]]
```

## Applications of NumPy Mean

The
**numpy.mean()**function has a wide range of applications in scientific computing, data analysis, and machine learning. Some common use cases are −
- **Calculating average values in datasets:**The mean provides a central value for datasets, which is crucial in statistics and data analysis to understand the data distribution.
- **Feature scaling:**In machine learning, computing the mean of features helps in normalization and standardization, ensuring that each feature contributes equally to the model.
- **Financial analysis:**Calculating the mean of financial data, such as stock prices or sales figures, helps identify trends and make informed decisions.
- **Scientific measurements:**The mean is used in scientific research to summarize experimental data, providing a measure of central tendency.
## Optimizing the Mean Calculation

NumPy is optimized for fast array operations, and the
**numpy.mean()**function is highly efficient. However, there are a few ways to further optimize your mean calculations −
- **Using the**If you want to store the result of the mean in a pre-existing array, you can use the**out**parameter:**out**parameter, which avoids creating a new array and helps save memory.
- **Using**Specify the axis only when necessary. Calculating the mean over the whole array by default is the fastest operation, but computing the mean along specific axes might be slower depending on the data.**axis**wisely:

---

## 82. NumPy - Median

*Source: [https://www.tutorialspoint.com/numpy/numpy_median.htm](https://www.tutorialspoint.com/numpy/numpy_median.htm)*

---

---
[Previous](/numpy/numpy_mean.htm)[Quiz](/numpy/quiz_on_numpy_median.htm)[Next](/numpy/numpy_min.htm)
## What is Median?

In mathematics, the median is the middle value of a set of numbers when they are arranged in order.

If the set has an odd number of values, the median is the middle one. If it has an even number of values, the median is the average of the two middle values.

The median is useful for finding the central tendency of data, especially when there are outliers.

## The NumPy median() Function

The median() function in NumPy calculates the median of an array's elements. It sorts the values and returns the middle value, or the average of the two middle values if the array has an even number of elements.

You can also specify an axis to calculate the median along rows or columns. For example, np.median([1, 3, 2, 4]) returns 2.5.

Following is the basic syntax of the median() function in NumPy −

```
numpy.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)
```

Where,

- **a:**The input array or dataset for which the median is calculated.
- **axis:**Specifies the axis along which the median is computed. If**None**(default), the median is computed over the entire array.
- **out:**This allows you to specify a location where the result will be stored. If**None**(default), the result is returned as a new array.
- **overwrite_input:**If**True**, the input array is modified in place to save memory. This is useful when you do not need the original data.
- **keepdims:**If**True**, the result will retain the reduced dimensions, allowing for easier broadcasting. If**False**(default), the result is squeezed.
## Understanding the Median Calculation

The calculation of the median in a dataset follows these steps −

- **Step 1:**Sort the array in ascending order.
- **Step 2:**Find the middle element. If the number of elements is odd, the middle element is the median.
- **Step 3:**If the number of elements is even, calculate the average of the two middle elements to get the median.
### Example

Let us understand this concept with an example. Here, in the first example, the array has an odd number of elements (5), so the middle element (5) is returned as the median.

In the second example, the array has an even number of elements (4), so the median is calculated by averaging the two middle elements (3 and 5), which gives 4.0 as the result −

```
import numpy as np

data_odd = np.array([1, 3, 5, 7, 9])
data_even = np.array([1, 3, 5, 7])

# Calculating the median for both datasets
median_odd = np.median(data_odd)
median_even = np.median(data_even)

print("Median of odd dataset:", median_odd)
print("Median of even dataset:", median_even)
```

Following is the output obtained −

```
Median of odd dataset: 5.0
Median of even dataset: 4.0
```

## Computing Median along Different Axes

In NumPy, the
**axis**parameter allows you to compute the median along specific axes of a multi-dimensional array. The axis refers to the direction in which the median should be calculated. For example, in a 2D array −
- **axis=0:**Calculate the median along the columns (vertical axis).
- **axis=1:**Calculate the median along the rows (horizontal axis).
### Example

In the following example, we are computing the median along both axes of a 2D array −

```
import numpy as np

# Create a 2D array
data_2d = np.array([[1, 3, 5], [2, 4, 6], [7, 8, 9]])

# Calculate the median along axis 0 (columns)
median_axis_0 = np.median(data_2d, axis=0)

# Calculate the median along axis 1 (rows)
median_axis_1 = np.median(data_2d, axis=1)

print("Median along axis 0:", median_axis_0)
print("Median along axis 1:", median_axis_1)
```

In the output below, the median along axis 0 is computed by taking the median of each column. The median along axis 1 is calculated by taking the median of each row −

```
Median along axis 0: [2. 4. 6.]
Median along axis 1: [3. 4. 8.]
```

## Median for Higher-Dimensional Arrays

The
**numpy.median()**function also works for arrays with more than two dimensions. You can specify the axis along which to calculate the median, and the function will return the median for that axis while retaining the other dimensions. If no axis is specified, the median is calculated over the entire array.
### Example

Following is an example to compute the median of a 3D array −

```
import numpy as np

# Create a 3D array
data_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Median along axis 0
median_3d_axis_0 = np.median(data_3d, axis=0)

# Median along axis 1
median_3d_axis_1 = np.median(data_3d, axis=1)

# Median along axis 2
median_3d_axis_2 = np.median(data_3d, axis=2)

print("Median along axis 0:", median_3d_axis_0)
print("Median along axis 1:", median_3d_axis_1)
print("Median along axis 2:", median_3d_axis_2)
```

In this case, the median is calculated along each of the axes (0, 1, and 2) for the 3D array. The function returns the median values for each of the specified axes while preserving the other dimensions −

```
Median along axis 0: [[3. 4.]
 [5. 6.]]
Median along axis 1: [[2. 3.]
 [6. 7.]]
Median along axis 2: [[1.5 3.5]
 [5.5 7.5]]
```

## Handling NaN (Not a Number) Values

Sometimes, arrays may contain NaN (Not a Number) values, which can interfere with the calculation of the median. To handle NaN values, NumPy provides an option to ignore them during median calculation. You can use the
**numpy.nanmedian()**function, which computes the median while ignoring NaN values.
### Example

Following is an example to handle NaN values while calculating median in NumPy −

```
import numpy as np

# Create an array with NaN values
data_with_nan = np.array([1, 3, np.nan, 5, 7])

# Calculate the median while ignoring NaN values
median_without_nan = np.nanmedian(data_with_nan)

print("Median without NaN:", median_without_nan)
```

In this example, the
**np.nanmedian()**function ignores the NaN value and computes the median of the remaining numbers, resulting in 4.0.
```
Median without NaN: 4.0
```

---

## 83. NumPy - Min

*Source: [https://www.tutorialspoint.com/numpy/numpy_min.htm](https://www.tutorialspoint.com/numpy/numpy_min.htm)*

---

---
[Previous](/numpy/numpy_median.htm)[Quiz](/numpy/quiz_on_numpy_min.htm)[Next](/numpy/numpy_max.htm)
## What is Min?

In mathematics, the "min" (minimum) refers to the smallest value in a set of numbers. It identifies the least element, providing a measure of the lowest point in a data set.

For example, in the set {3, 1, 4, 2}, the minimum is 1. The minimum is useful for understanding the lower bound of a data set.

## The NumPy min() Function

The min() function in NumPy returns the smallest value in an array. It can be applied to the entire array or along a specified axis to find the minimum value in each row or column.

You can also use the amin() function, which is an alias for min() function. Following is the basic syntax of the min() function in NumPy −

```
numpy.min(a, axis=None, out=None, keepdims=False)
```
**False**)
Where,

- **a:**The input array or dataset from which the minimum value is to be found.
- **axis:**Specifies the axis along which the minimum value is computed. If**None**(default), the minimum value is computed over the entire array.
- **out:**This allows you to specify a location where the result will be stored. If**None**(default), the result is returned as a new array.
- **keepdims:**If**True**, the reduced dimensions are retained in the result, making it easier for broadcasting. If**False**(default), the result is squeezed.
## Understanding the Min Calculation

The calculation of the minimum value in a dataset is very easy. The function scans through all the elements in the array and identifies the smallest value. This process can be applied to arrays of any shape or size.

### Example

Let us understand this concept with an example −

```
import numpy as np

data = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])

# Calculating the minimum value
min_value = np.min(data)

print("Minimum value:", min_value)
```

Following is the output obtained −

```
Minimum value: 1
```

## Computing Min along Different Axes

In NumPy, the
**axis**parameter allows you to compute the minimum value along specific axes of a multi-dimensional array. The axis parameter refers to the direction along which the minimum value should be calculated. For example, in a 2D array −
- **axis=0:**Calculate the minimum value along the columns (vertical axis).
- **axis=1:**Calculate the minimum value along the rows (horizontal axis).
### Example

In the following example, we are computing the minimum value along both axes of the 2D array −

```
import numpy as np

# Create a 2D array
data_2d = np.array([[1, 3, 5], [2, 4, 6], [7, 8, 9]])

# Calculate the minimum value along axis 0 (columns)
min_axis_0 = np.min(data_2d, axis=0)

# Calculate the minimum value along axis 1 (rows)
min_axis_1 = np.min(data_2d, axis=1)

print("Minimum value along axis 0:", min_axis_0)
print("Minimum value along axis 1:", min_axis_1)
```

In the output below, the minimum value along axis 0 is computed by finding the smallest element in each column. The minimum value along axis 1 is calculated by finding the smallest element in each row −

```
Minimum value along axis 0: [1 3 5]
Minimum value along axis 1: [1 2 7]
```

## Min for Higher-Dimensional Arrays

The
**numpy.min()**function also works for arrays with more than two dimensions. You can specify the axis along which to calculate the minimum value, and the function will return the minimum value for that axis while retaining the other dimensions. If no axis is specified, the minimum value is calculated over the entire array.
### Example

Following is an example to compute minimum value of a 3D array −

```
import numpy as np

# Create a 3D array
data_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Minimum value along axis 0
min_3d_axis_0 = np.min(data_3d, axis=0)

# Minimum value along axis 1
min_3d_axis_1 = np.min(data_3d, axis=1)

# Minimum value along axis 2
min_3d_axis_2 = np.min(data_3d, axis=2)

print("Minimum value along axis 0:", min_3d_axis_0)
print("Minimum value along axis 1:", min_3d_axis_1)
print("Minimum value along axis 2:", min_3d_axis_2)
```

In this case, the minimum value is calculated along each of the axes (0, 1, and 2) for the 3D array. The function returns the minimum values for each of the specified axes while preserving the other dimensions −

```
Minimum value along axis 0: [[1 2]
 [3 4]]
Minimum value along axis 1: [[1 2]
 [5 6]]
Minimum value along axis 2: [[1 3]
 [5 7]]
```

## Handling NaN (Not a Number) Values

Sometimes, arrays may contain NaN (Not a Number) values, which can interfere with the calculation of the minimum value. To handle NaN values, NumPy provides an option to ignore them during the min calculation.

You can use the
**numpy.nanmin()**function, which computes the minimum value while ignoring NaN values.
### Example

In this example, we are handling NaN values while computing minimum value in NumPy −

```
import numpy as np

# Create an array with NaN values
data_with_nan = np.array([1, 3, np.nan, 5, 7])

# Calculate the minimum value while ignoring NaN values
min_without_nan = np.nanmin(data_with_nan)

print("Minimum value without NaN:", min_without_nan)
```

Following is the output obtained −

```
Minimum value without NaN: 1.0
```

## Using the "Out" Parameter

The
**out**parameter in the**numpy.min()**function allows you to store the result of the minimum value computation in a pre-allocated array.
This can be useful for memory management and efficiency when working with large datasets. The result is stored in the array specified by the
**out**parameter, which must have the same shape as the expected output.
### Example

In this example, the minimum value of the array
**data**is calculated and stored in the pre-allocated array**out_array**, which is then printed to show the result −
```
import numpy as np

# Create an array
data = np.array([5, 2, 9, 1, 5, 6])

# Create an output array
out_array = np.empty((), dtype=np.int32)

# Calculate the minimum value and store it in out_array
np.min(data, out=out_array)

print("Output array:", out_array)
```

Following is the output obtained −

```
Output array: 1
```

---

## 84. NumPy - Max

*Source: [https://www.tutorialspoint.com/numpy/numpy_max.htm](https://www.tutorialspoint.com/numpy/numpy_max.htm)*

---

---
[Previous](/numpy/numpy_min.htm)[Quiz](/numpy/quiz_on_numpy_max.htm)[Next](/numpy/numpy_unique_elements.htm)
## What is the Max?

In mathematics, the "max" (maximum) refers to the largest value in a set of numbers. It identifies the greatest element, providing a measure of the highest point in a data set.

For example, in the set {3, 1, 4, 2}, the maximum is 4. The maximum is useful for understanding the upper bound of a data set.

## The NumPy max() Function

The max() function in NumPy returns the largest value in an array. It can be applied to the entire array or along a specified axis to find the maximum value in each row or column.

You can also use the amax() function, which is an alias for max() function. Following is the basic syntax of the max() function in NumPy −

```
numpy.max(a, axis=None, out=None, keepdims=False)
```
**False**)
Where,

- **a:**The input array or dataset from which the maximum value is to be found.
- **axis:**Specifies the axis along which the maximum value is computed. If**None**(default), the maximum value is computed over the entire array.
- **out:**This allows you to specify a location where the result will be stored. If**None**(default), the result is returned as a new array.
- **keepdims:**If**True**, the reduced dimensions are retained in the result, making it easier for broadcasting. If**False**(default), the result is squeezed.
## Understanding the Max Calculation

The calculation of the maximum value in a dataset is simple and easy. The function scans through all the elements in the array and identifies the largest value. This process can be applied to arrays of any shape or size.

### Example

Let us understand this concept with an example −

```
import numpy as np

data = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])

# Calculating the maximum value
max_value = np.max(data)

print("Maximum value:", max_value)
```

This will produce the following result −

```
Maximum value: 9
```

## Computing Max along Different Axes

In NumPy, the
**axis**parameter allows you to compute the maximum value along specific axes of a multi-dimensional array. The axis parameter refers to the direction along which the maximum value should be calculated. For example, in a 2D array −
- **axis=0:**Calculate the maximum value along the columns (vertical axis).
- **axis=1:**Calculate the maximum value along the rows (horizontal axis).
### Example

In the following example, we are computing the maximum value along both axes of the 2D array −

```
import numpy as np

# Create a 2D array
data_2d = np.array([[1, 3, 5], [2, 4, 6], [7, 8, 9]])

# Calculate the maximum value along axis 0 (columns)
max_axis_0 = np.max(data_2d, axis=0)

# Calculate the maximum value along axis 1 (rows)
max_axis_1 = np.max(data_2d, axis=1)

print("Maximum value along axis 0:", max_axis_0)
print("Maximum value along axis 1:", max_axis_1)
```

In the output below, the maximum value along axis 0 is computed by finding the largest element in each column. The maximum value along axis 1 is calculated by finding the largest element in each row −

```
Maximum value along axis 0: [7 8 9]
Maximum value along axis 1: [5 6 9]
```

## Max for Higher-Dimensional Arrays

The
**numpy.max()**function also works for arrays with more than two dimensions. You can specify the axis along which to calculate the maximum value, and the function will return the maximum value for that axis while retaining the other dimensions. If no axis is specified, the maximum value is calculated over the entire array.
### Example

Following is an example to compute maximum value of a 3D array −

```
import numpy as np

# Create a 3D array
data_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Maximum value along axis 0
max_3d_axis_0 = np.max(data_3d, axis=0)

# Maximum value along axis 1
max_3d_axis_1 = np.max(data_3d, axis=1)

# Maximum value along axis 2
max_3d_axis_2 = np.max(data_3d, axis=2)

print("Maximum value along axis 0:", max_3d_axis_0)
print("Maximum value along axis 1:", max_3d_axis_1)
print("Maximum value along axis 2:", max_3d_axis_2)
```

In this case, the maximum value is calculated along each of the axes (0, 1, and 2) for the 3D array. The function returns the maximum values for each of the specified axes while preserving the other dimensions −

```
Maximum value along axis 0: [[5 6]
 [7 8]]
Maximum value along axis 1: [[3 4]
 [7 8]]
Maximum value along axis 2: [[2 4]
 [6 8]]
```

## Handling NaN (Not a Number) Values

Sometimes, arrays may contain NaN (Not a Number) values, which can interfere with the calculation of the maximum value. To handle NaN values, NumPy provides an option to ignore them during the max calculation.

You can use the
**numpy.nanmax()**function, which computes the maximum value while ignoring NaN values.
### Example

In this example, we are handling NaN values while computing maximum value in NumPy −

```
import numpy as np

# Create an array with NaN values
data_with_nan = np.array([1, 3, np.nan, 5, 7])

# Calculate the maximum value while ignoring NaN values
max_without_nan = np.nanmax(data_with_nan)

print("Maximum value without NaN:", max_without_nan)
```

After executing the above code, we get the following output −

```
Maximum value without NaN: 7.0
```

## Using the Out Parameter

The
**out**parameter in the**numpy.max()**function allows you to store the result of the maximum value computation in a pre-allocated array.
This can be useful for memory management and efficiency when working with large datasets. The result is stored in the array specified by the
**out**parameter, which must have the same shape as the expected output.
### Example

In this example, the maximum value of the array
**data**is calculated and stored in the pre-allocated array**out_array**, which is then printed to show the result −
```
import numpy as np

# Create an array
data = np.array([5, 2, 9, 1, 5, 6])

# Create an output array
out_array = np.empty((), dtype=np.int32)

# Calculate the maximum value and store it in out_array
np.max(data, out=out_array)

print("Output array:", out_array)
```

The result produced is as follows −

```
Output array: 9
```

---

## 85. NumPy - Unique Elements

*Source: [https://www.tutorialspoint.com/numpy/numpy_unique_elements.htm](https://www.tutorialspoint.com/numpy/numpy_unique_elements.htm)*

---

---
[Previous](/numpy/numpy_max.htm)[Quiz](/numpy/quiz_on_numpy_unique_elements.htm)[Next](/numpy/numpy_intersection.htm)
## What is Unique Elements?

Unique elements refer to the distinct values in a set or collection, where each value appears only once.

In other words, if a value appears multiple times, it is counted only once as a unique element. For example, in the set {1, 2, 2, 3, 4}, the unique elements are {1, 2, 3, 4}.

## The NumPy unique() Function

The unique() function in NumPy returns the sorted unique elements of an array. It removes any duplicate values, keeping only distinct ones.

You can also get additional information, such as the indices of the unique values or their counts. Following is the basic syntax of the unique() function in NumPy −

```
numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)
```

- **ar:**The input array from which unique elements are to be found.
- **return_index:**If True, also return the indices of the first occurrences of the unique values in the original array.
- **return_inverse:**If True, also return the indices to reconstruct the original array from the unique array.
- **return_counts:**If True, also return the number of times each unique value comes up in the original array.
- **axis:**If specified, the axis along which to find the unique values. If None (default), the unique values are found in the flattened array.
## Finding Unique Elements in a 1D Array

To find the unique elements in a one-dimensional array, you can simply pass the array to the
**numpy.unique()**function.
### Example

In this example, the
**numpy.unique()**function finds and returns the unique elements in the array**data**−
```
import numpy as np

# Create a one-dimensional array
data = np.array([1, 2, 2, 3, 4, 4, 4, 5])

# Find unique elements
unique_elements = np.unique(data)

print("Unique elements:", unique_elements)
```

Following is the output obtained −

```
Unique elements: [1 2 3 4 5]
```

## Returning Additional Information

The
**numpy.unique()**function can also return additional information about the unique elements, such as the indices of their first occurrences, the indices to reconstruct the original array, and the counts of each unique value.
This is controlled by the
**return_index**,**return_inverse**, and**return_counts**parameters, respectively.
### Returning Indices of First Occurrences

To get the indices of the first occurrences of the unique values in the original array, set the
**return_index**parameter to**True**−
```
import numpy as np

# Create an array
data = np.array([1, 2, 2, 3, 4, 4, 4, 5])

# Find unique elements and their indices
unique_elements, indices = np.unique(data, return_index=True)

print("Unique elements:", unique_elements)
print("Indices of first occurrences:", indices)
```

This will produce the following result −

```
Unique elements: [1 2 3 4 5]
Indices of first occurrences: [0 1 3 4 7]
```

### Returning Indices to Reconstruct the Original Array

To get the indices that can be used to reconstruct the original array from the unique array, set the
**return_inverse**parameter to**True**−
```
import numpy as np

# Create an array
data = np.array([1, 2, 2, 3, 4, 4, 4, 5])

# Find unique elements and inverse indices
unique_elements, inverse_indices = np.unique(data, return_inverse=True)

print("Unique elements:", unique_elements)
print("Inverse indices:", inverse_indices)
```

Following is the output of the above code −

```
Unique elements: [1 2 3 4 5]
Inverse indices: [0 1 1 2 3 3 3 4]
```

### Returning Counts of Unique Elements

To get the counts of each unique value in the original array, set the
**return_counts**parameter to**True**−
```
import numpy as np

# Create an array
data = np.array([1, 2, 2, 3, 4, 4, 4, 5])

# Find unique elements and their counts
unique_elements, counts = np.unique(data, return_counts=True)

print("Unique elements:", unique_elements)
print("Counts of unique elements:", counts)
```

The output obtained is as shown below −

```
Unique elements: [1 2 3 4 5]
Counts of unique elements: [1 2 1 3 1]
```

## Unique Elements in a Multi-Dimensional Array

The
**numpy.unique()**function can also be used to find unique elements in multi-dimensional arrays. By default, the function flattens the array and then finds the unique elements.
However, you can specify the axis along which to find the unique values using the
**axis**parameter.
### Default Behavior (Flattened Array)

Let us see an example of finding unique elements in a 2D array without specifying an axis −

```
import numpy as np

# Create a 2D array
data_2d = np.array([[1, 2, 2], [3, 4, 4], [4, 5, 5]])

# Find unique elements
unique_elements = np.unique(data_2d)

print("Unique elements:", unique_elements)
```

Here, the function flattens the 2D array and then finds the unique elements as shown in the output below −

```
Unique elements: [1 2 3 4 5]
```

### Finding Unique Elements along a Specific Axis

You can also find unique elements along a specific axis. For example, to find unique elements along the rows (axis=1) or columns (axis=0) of a 2D array −

```
import numpy as np

# Create a 2D array
data_2d = np.array([[1, 2, 2], [3, 4, 4], [4, 5, 5]])

# Find unique elements along axis 0 (columns)
unique_elements_axis_0 = np.unique(data_2d, axis=0)

# Find unique elements along axis 1 (rows)
unique_elements_axis_1 = np.unique(data_2d, axis=1)

print("Unique elements along axis 0:\n", unique_elements_axis_0)
print("Unique elements along axis 1:\n", unique_elements_axis_1)
```

The result produced is as follows −

```
Unique elements along axis 0:
 [[1 2 2]
  [3 4 4]
  [4 5 5]]
Unique elements along axis 1:
 [[1 2]
  [3 4]
  [4 5]]
```

## Unique Elements in Structured Arrays

NumPy also supports structured arrays, where each element can be a combination of multiple fields. You can find unique elements in structured arrays by specifying the fields to consider for uniqueness.

### Example

In this example, the function finds unique elements in the structured array by considering all fields −

```
import numpy as np

# Create a structured array
data_structured = np.array([(1, 'a'), (2, 'b'), (2, 'b'), (3, 'c')],
                           dtype=[('num', 'i4'), ('char', 'U1')])

# Find unique elements considering all fields
unique_elements_structured = np.unique(data_structured)

print("Unique elements in structured array:", unique_elements_structured)
```

We get the output as shown below −

```
Unique elements in structured array: [(1, 'a') (2, 'b') (3, 'c')]
```

---

## 86. NumPy - Intersection

*Source: [https://www.tutorialspoint.com/numpy/numpy_intersection.htm](https://www.tutorialspoint.com/numpy/numpy_intersection.htm)*

---

---
[Previous](/numpy/numpy_unique_elements.htm)[Quiz](/numpy/quiz_on_numpy_intersection.htm)[Next](/numpy/numpy_union.htm)
## Intersection in NumPy

In NumPy, the term "intersection" refers to the elements that are common between two or more arrays.

NumPy provides a built-in function called
**numpy.intersect1d()**that helps in finding the intersection between two arrays.
## What is Array Intersection?

When you work with arrays, you might often need to find the elements that appear in both of them. This process is called finding the intersection.

For instance, if you have two sets of numbers and you need to determine which numbers appear in both, you can perform an intersection operation.

## The NumPy intersect1d() Function

In NumPy, the
**intersect1d()**function is used to find the intersection of two 1-dimensional arrays, or even more arrays if necessary.
Following is the basic syntax of the NumPy intersect1d() function. It works by comparing two input arrays and returning an array containing the common elements −

```
numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)
```

Where,

- **ar1, ar2:**These are the two input arrays in which we want to find the common elements.
- **assume_unique:**If set to**True**, it assumes that both input arrays contain only unique elements, speeding up the computation.
- **return_indices:**If set to**True**, the function returns not only the intersection elements but also their indices in the original arrays.
### Example

In the following example, we are finding the common elements between two arrays using the numpy.intersect1d() function −

```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([4, 5, 6, 7, 8])

# Find intersection of the two arrays
intersection = np.intersect1d(array1, array2)

print("Intersection of array1 and array2:", intersection)
```

Following is the output obtained −

```
Intersection of array1 and array2: [4 5]
```

## Assuming Unique Elements for Faster Computation

In cases where you are sure that the input arrays contain only unique elements (i.e., no duplicates), you can pass
**True**to the**assume_unique**parameter. This speeds up the computation by avoiding the need to check for duplicates:
### Example

As in the previous example, the intersection remains the same, but the function is more efficient due to the assumption of uniqueness −

```
import numpy as np

# Define two arrays with unique elements
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([4, 5, 6, 7, 8])

# Find intersection assuming unique elements
intersection = np.intersect1d(array1, array2, assume_unique=True)

print("Intersection assuming unique elements:", intersection)
```

The output obtained is as follows −

```
Intersection assuming unique elements: [4 5]
```

## Returning Indices of Intersection Elements

In addition to the intersection elements, the
**numpy.intersect1d()**function can also return the indices of these elements in the input arrays.
This is particularly useful when you want to know the exact positions of the common elements in the original arrays. To achieve this, set the
**return_indices**parameter to**True**.
### Example

In this example, the intersection elements 4 and 5 appear at indices 3 and 4 in
**array1**and at indices 0 and 1 in**array2**−
```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([4, 5, 6, 7, 8])

# Find intersection and return indices
intersection, indices1, indices2 = np.intersect1d(array1, array2, return_indices=True)

print("Intersection elements:", intersection)
print("Indices in array1:", indices1)
print("Indices in array2:", indices2)
```

After executing the above code, we get the following output −

```
Intersection elements: [4 5]
Indices in array1: [3 4]
Indices in array2: [0 1]
```

## Intersection of More Than Two Arrays

The
**numpy.intersect1d()**function can also be used to find the intersection of more than two arrays.
While the function itself is designed to work with two arrays at a time, you can easily extend it to multiple arrays by using loops or the
**reduce()**function from the**functools**module.
### Example

As shown in the example below, the common element among all three arrays is 5, which forms the intersection −

```
import numpy as np
from functools import reduce

# Define multiple arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([4, 5, 6, 7, 8])
array3 = np.array([5, 6, 7, 8, 9])

# Find intersection of all arrays
intersection = reduce(np.intersect1d, [array1, array2, array3])

print("Intersection of multiple arrays:", intersection)
```

The result produced is as follows −

```
Intersection of multiple arrays: [5]
```

## Working with Arrays of Different Data Types

NumPy's
**intersect1d()**function can also handle arrays of different data types, such as integers, floats, and strings.
However, the function compares the elements based on their data types, meaning it performs type-sensitive matching.

### Example

In this example, the intersection element 4 is returned as a float because the first array contains floating-point numbers −

```
import numpy as np

# Define arrays with different data types
array1 = np.array([1.0, 2.0, 3.0, 4.0])
array2 = np.array([4, 5, 6, 7])

# Find intersection elements
intersection = np.intersect1d(array1, array2)

print("Intersection elements:", intersection)
```

The output obtained is as shown below −

```
Intersection elements: [4.]
```

## Dealing with Floating-Point Precision Issues

When working with floating-point numbers, precision issues can arise, especially when the values are very close to each other but not exactly the same due to the way floating-point arithmetic works. To avoid this, you can round the arrays before performing the intersection.

### Example

By rounding the arrays to two decimal places, the intersection operation works more accurately despite the small floating-point differences as shown in the example below −

```
import numpy as np

# Define floating-point arrays
array1 = np.array([1.234, 2.345, 3.456, 4.567])
array2 = np.array([4.567, 5.678, 6.789])

# Round arrays and find intersection
array1_rounded = np.round(array1, 2)
array2_rounded = np.round(array2, 2)

intersection = np.intersect1d(array1_rounded, array2_rounded)

print("Intersection after rounding:", intersection)
```

The output produced is as follows −

```
Intersection after rounding: [4.57]
```

---

## 87. NumPy - Union

*Source: [https://www.tutorialspoint.com/numpy/numpy_union.htm](https://www.tutorialspoint.com/numpy/numpy_union.htm)*

---

---
[Previous](/numpy/numpy_intersection.htm)[Quiz](/numpy/quiz_on_numpy_union.htm)[Next](/numpy/numpy_difference.htm)
## Union in NumPy

In NumPy, the term "union" refers to the operation that combines the elements of two or more arrays, removing any duplicate values.

It is commonly used when you want to merge multiple datasets or arrays, ensuring that each element appears only once in the final result.

NumPy provides the
**numpy.union1d()**function to easily find the union of two 1-dimensional arrays.
## What is Union of Arrays?

The union of two or more arrays refers to the combined set of unique elements from all the input arrays.

This means that no duplicate elements are present in the result. The union operation is closely related to the concept of sets in mathematics.

For example, if you have two arrays containing some common and some unique elements, the union will contain all the unique elements from both arrays.

## The NumPy union1d() Function

In NumPy, the
**numpy.union1d()**function is used to compute the union of two 1-dimensional arrays. This function ensures that no elements are repeated in the result, even if the same element appears in both input arrays.
Following is the basic syntax of the NumPy union1d() function −

```
numpy.union1d(ar1, ar2)
```

Where,
**ar1**and**ar2**are the two input arrays whose union is to be found. The arrays can contain any data type, and they may or may not have overlapping elements.
### Example

In the following example, we are calculating the union of two arrays using the union1d() function in NumPy −

```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([4, 5, 6, 7, 8])

# Find union of the two arrays
union = np.union1d(array1, array2)

print("Union of array1 and array2:", union)
```

As seen in the output, the union of
**array1**and**array2**contains all unique elements from both arrays, without any repetition. The numbers 4 and 5, which appeared in both arrays, appear only once in the final result −
```
Union of array1 and array2: [1 2 3 4 5 6 7 8]
```

## Union of Arrays with Different Data Types

NumPy's
**union1d()**function can also handle arrays of different data types, such as integers, floats, and even strings. The function will convert all elements to a common type before computing the union.
### Example

As shown in the example below, NumPy has automatically converted all elements to floats because the first array contains a floating-point number, and the union contains no duplicates −

```
import numpy as np

# Define arrays with different data types
array1 = np.array([1, 2, 3, 4.5])
array2 = np.array([4.5, 5, 6, 7])

# Find union of the arrays
union = np.union1d(array1, array2)

print("Union of array1 and array2 with different types:", union)
```

The result produced is as follows −

```
Union of array1 and array2 with different types: [1. 2. 3. 4.5 5. 6. 7.]
```

## Handling Multiple Arrays

The
**numpy.union1d()**function works with two arrays at a time. However, if you need to find the union of more than two arrays, you can use loops or the**reduce()**function from the**functools**module.
### Example

Below is an example that demonstrates how to compute the union of three arrays −

```
import numpy as np
from functools import reduce

# Define multiple arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([4, 5, 6, 7, 8])
array3 = np.array([7, 8, 9, 10])

# Find the union of all arrays
union = reduce(np.union1d, [array1, array2, array3])

print("Union of multiple arrays:", union)
```

As shown in the output, the union operation combines all the unique elements from the three arrays. There are no duplicates, and the union contains all the unique values across the arrays −

```
Union of multiple arrays: [1 2 3 4 5 6 7 8 9 10]
```

## Union with Arrays Containing Duplicates

When the input arrays contain duplicate elements,
**numpy.union1d()**automatically removes them in the final result. This ensures that the returned union consists of only unique elements.
### Example

Following is an example where we find the union of arrays containing duplicates −

```
import numpy as np

# Define arrays with duplicate elements
array1 = np.array([1, 2, 2, 3, 4])
array2 = np.array([3, 4, 4, 5, 6])

# Find union of the arrays
union = np.union1d(array1, array2)

print("Union with duplicates removed:", union)
```

The output obtained is as shown below −

```
Union with duplicates removed: [1 2 3 4 5 6]
```

## Union of Arrays with Strings

In NumPy, you can also perform union operations on arrays containing strings. The function will combine all unique strings from both arrays.

### Example

Let us take a look at an example with string arrays −

```
import numpy as np

# Define arrays with strings
array1 = np.array(['apple', 'banana', 'cherry'])
array2 = np.array(['banana', 'cherry', 'date'])

# Find the union of the string arrays
union = np.union1d(array1, array2)

print("Union of string arrays:", union)
```

We get the following output −

```
Union of string arrays: ['apple' 'banana' 'cherry' 'date']
```

## Performance Considerations

The
**numpy.union1d()**function is efficient, but the performance can depend on the size of the input arrays. When you are working with very large arrays, it is a good idea to ensure that the arrays are as efficient as possible.
For example, if the arrays contain only unique elements, you can set the
**assume_unique**parameter to**True**to speed up the union operation:
### Example

By assuming that the arrays contain only unique elements, NumPy can perform the union operation more quickly as shown in the example below −

```
import numpy as np

# Define arrays with unique elements
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([6, 7, 8, 9, 10])

# Find union assuming unique elements
union = np.union1d(array1, array2)

print("Union of unique arrays:", union)
```

The result produced is as follows −

```
Union of unique arrays: [1 2 3 4 5 6 7 8 9 10]
```

---

## 88. NumPy - Difference

*Source: [https://www.tutorialspoint.com/numpy/numpy_difference.htm](https://www.tutorialspoint.com/numpy/numpy_difference.htm)*

---

---
[Previous](/numpy/numpy_union.htm)[Quiz](/numpy/quiz_on_numpy_difference.htm)[Next](/numpy/numpy_random_generator.htm)
## Difference in NumPy

In NumPy, the difference operation is used to find elements present in one array but not in another. It is commonly used to compare two arrays and identify the unique elements in one array that do not exist in the other.

In NumPy, the
**setdiff1d()**function is used to perform this operation.
## What is Set Difference?

The "set difference" operation refers to finding the elements that are present in one set but not in another. In NumPy, this operation is applied to arrays, and it returns the elements in the first array that are not found in the second array.

This concept is closely related to the mathematical set theory, where the difference of two sets
**A - B**contains elements in set**A**but not in set**B**.
For example, given two arrays −

```
array1 = [1, 2, 3, 4, 5]
array2 = [3, 4, 5, 6, 7]
```

The set difference will give us the elements in
**array1**that are not in**array2**, which are**[1, 2]**.
### Syntax

Following is the basic syntax for the setdiff1d() function in NumPy −

```
numpy.setdiff1d(ar1, ar2)
```

Where,

- **ar1:**The first input array. It is the array from which we want to subtract elements.
- **ar2:**The second input array. It contains elements that will be removed from the first array.
The result is a sorted array containing the unique values that are in
**ar1**but not in**ar2**.
### Example

In the following example, we are calculating the difference between two arrays using the setdiff1d() function in NumPy −

```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])

# Find the difference between the two arrays
difference = np.setdiff1d(array1, array2)

print("Difference between array1 and array2:", difference)
```

Following is the output obtained −

```
Difference between array1 and array2: [1 2]
```

## Handling Arrays with Duplicate Elements

If the input arrays contain duplicate elements, the
**numpy.setdiff1d()**function will remove the duplicates before performing the difference operation. This ensures that the result contains only unique values.
### Example

Here, we are removing the duplicates in
**array1**before computing the difference, resulting in the final output containing only the unique elements −
```
import numpy as np

# Define arrays with duplicate elements
array1 = np.array([1, 2, 2, 3, 4])
array2 = np.array([3, 4, 4, 5, 6])

# Find the difference between the arrays
difference = np.setdiff1d(array1, array2)

print("Difference with duplicates removed:", difference)
```

The result produced is as follows −

```
Difference with duplicates removed: [1 2]
```

## Handling Arrays with Different Data Types

NumPys
**setdiff1d()**function works with arrays of different data types, including integers, floats, and strings.
However, the function will automatically convert the data types to a common type before performing the difference operation.

### Example

Let us take a look at an example where we calculate the difference between an integer array and a float array −

```
import numpy as np

# Define arrays with different data types
array1 = np.array([1, 2, 3, 4.5])
array2 = np.array([4.5, 5, 6])

# Find the difference between the arrays
difference = np.setdiff1d(array1, array2)

print("Difference with different data types:", difference)
```

After executing the above code, we get the following output −

```
Difference with different data types: [1. 2. 3.]
```

## Difference with Multiple Arrays

In NumPy, you can only use
**setdiff1d()**function to compute the difference between two arrays at a time.
If you want to compute the difference with multiple arrays, you can use a combination of
**setdiff1d()**function and loops or**reduce()**function from the**functools**module.
### Example

Following is an example that demonstrates how to calculate the difference between multiple arrays −

```
import numpy as np
from functools import reduce

# Define multiple arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])
array3 = np.array([5, 6, 7, 8])

# Calculate the difference of all arrays
difference = reduce(lambda x, y: np.setdiff1d(x, y), [array1, array2, array3])

print("Difference of multiple arrays:", difference)
```

The output obtained is as shown below −

```
Difference of multiple arrays: [1 2]
```

## Performance Considerations

The
**numpy.setdiff1d()**function is quite efficient, but performance can be a consideration when dealing with large arrays.
If your arrays contain only unique elements, you can use the
**assume_unique**parameter to speed up the computation.
### Example

By setting the
**assume_unique**parameter to**True**, NumPy optimizes the operation when dealing with arrays that already contain unique elements, leading to faster performance as shown in the example below −
```
import numpy as np

# Define arrays with unique elements
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])

# Find the difference assuming unique elements
difference = np.setdiff1d(array1, array2, assume_unique=True)

print("Difference with unique elements:", difference)
```

The result produced is as follows −

```
Difference with unique elements: [1 2]
```

---

## 89. NumPy - Random Generator

*Source: [https://www.tutorialspoint.com/numpy/numpy_random_generator.htm](https://www.tutorialspoint.com/numpy/numpy_random_generator.htm)*

---

---
[Previous](/numpy/numpy_difference.htm)[Quiz](/numpy/quiz_on_numpy_random_generator.htm)[Next](/numpy/numpy_permutations_and_shuffling.htm)
## NumPy Random Generator

The random generator in NumPy is used to generate random numbers and perform random sampling.

The random module in NumPy offers a wide range of random number generation functions, from generating random integers and floating-point numbers to more complex distributions like normal, uniform, and binomial distributions.

In this tutorial, we will explore how to use the NumPy Random Generator to generate random data and discuss the important functions available in this module.

## The NumPy Random Module?

The NumPy random module is a submodule within the NumPy library that contains functions for generating random numbers, performing random sampling, and generating random distributions. It provides the
**numpy.random**package, which supports the creation of random numbers from various probability distributions like uniform, normal, and binomial.
By using NumPy's random generator, we can generate random values that can be used for simulations, randomized testing, or even cryptographic operations. The random numbers generated are pseudo-random, meaning they are generated using a deterministic process but appear random.

The sequence of random numbers can be controlled using a random seed, which ensures reproducibility in simulations and experiments.

## How Does NumPy's Random Generator Work?

The NumPy random number generator is built on top of a pseudorandom number generator (PRNG) algorithm called the Mersenne Twister. The key feature of a PRNG is that it generates a sequence of numbers that approximates true randomness, but it is determined by an initial value, called a seed.

By setting the same seed, you can ensure that the sequence of random numbers is the same every time you run your code, which is important for reproducibility in scientific experiments.

## Seeding the Random Generator

To control the random number generation process, NumPy provides the
**numpy.random.seed()**function, which sets the seed for the random number generator. This allows you to generate the same random numbers every time you run your program.
### Example

In the example below, by setting the same seed value (
**42**in this case), we get the same sequence of random numbers every time we run the code. This helps in ensuring consistency during experiments and debugging −
```
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate random numbers
random_numbers_1 = np.random.random(5)

# Generate the same random numbers with the same seed
np.random.seed(42)
random_numbers_2 = np.random.random(5)

# Display the results
print("First random numbers:", random_numbers_1)
print("Second random numbers:", random_numbers_2)
```

Following is the output obtained −

```
First random numbers: [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
Second random numbers: [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
```

## Generating Random Numbers

NumPy offers various functions for generating random numbers. Here, we will explore some of the most commonly used functions −

### Random Float Numbers

The
**numpy.random.random()**function generates random floating-point numbers between**0**and**1**. You can specify the shape of the output array by passing the desired dimensions as an argument. For example −
```
import numpy as np

# Generate 5 random float numbers between 0 and 1
random_floats = np.random.random(5)
print("Random float numbers:", random_floats)
```

The result produced is as follows −

```
Random float numbers: [0.96177309 0.75071326 0.44828032 0.53441928 0.56717514]
```

### Random Integers

The
**numpy.random.randint()**function generates random integers within a specified range. You can specify the low and high values, and it will return integers between the low (inclusive) and high (exclusive) values. Here is an example −
```
import numpy as np

# Generate 5 random integers between 10 (inclusive) and 100 (exclusive)
random_integers = np.random.randint(10, 100, size=5)
print("Random integers:", random_integers)
```

After executing the above code, we get the following output −

```
Random integers: [13 38 56 94 78]
```

### Random Numbers from Normal Distribution

The
**numpy.random.normal()**function generates random numbers from a normal (Gaussian) distribution with a specified mean and standard deviation. You can also specify the size of the output array. Here's how it works −
```
import numpy as np

# Generate 5 random numbers from a normal distribution with mean=0 and std=1
random_normal = np.random.normal(0, 1, 5)
print("Random numbers from normal distribution:", random_normal)
```

The output obtained is as shown below −

```
Random numbers from normal distribution: [ 0.52379705  0.3169246   0.76473415 -0.73006407 -0.50259886]
```

### Random Numbers from Uniform Distribution

The
**numpy.random.uniform()**function generates random numbers from a uniform distribution within a given range. Here's an example of generating 5 random numbers between**1.0**and**10.0**−
```
import numpy as np

# Generate 5 random numbers between 1.0 and 10.0
random_uniform = np.random.uniform(1.0, 10.0, 5)
print("Random numbers from uniform distribution:", random_uniform)
```

The result produced is as follows −

```
Random numbers from uniform distribution: [4.92412702 2.57524084 1.71870242 3.71017627 6.19920522]
```

## Random Sampling with Replacement

Sometimes, we need to randomly select elements from an array. NumPy provides the
**numpy.random.choice()**function, which allows you to perform random sampling with or without replacement.
### Example

In the example below, the function selects 3 random elements from the array with replacement, meaning elements can be selected multiple times −

```
import numpy as np

# Define an array of elements
array = np.array([1, 2, 3, 4, 5])

# Randomly select 3 elements from the array with replacement
sample_with_replacement = np.random.choice(array, 3, replace=True)
print("Random sample with replacement:", sample_with_replacement)
```

The result produced is as follows −

```
Random sample with replacement: [5 3 5]
```

## Shuffling an Array

Another useful operation is shuffling the elements of an array randomly. NumPy provides the
**numpy.random.shuffle()**function for this purpose. It randomly permutes the elements of an array in-place.
### Example

In the following example, we are shuffling an array in NumPy using the numpy.random.shuffle() function −

```
import numpy as np

# Define an array
array = np.array([1, 2, 3, 4, 5])

# Shuffle the array in place
np.random.shuffle(array)
print("Shuffled array:", array)
```

The output obtained is as shown below −

```
Shuffled array: [4 2 3 5 1]
```

---

## 90. NumPy - Permutations and Shuffling

*Source: [https://www.tutorialspoint.com/numpy/numpy_permutations_and_shuffling.htm](https://www.tutorialspoint.com/numpy/numpy_permutations_and_shuffling.htm)*

---

---

## 91. NumPy - Uniform Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_uniform_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_uniform_distribution.htm)*

---

---
[Previous](/numpy/numpy_permutations_and_shuffling.htm)[Quiz](/numpy/quiz_on_numpy_uniform_distribution.htm)[Next](/numpy/numpy_normal_distribution.htm)
## What is a Uniform Distribution?

A uniform distribution is a type of probability distribution where all outcomes are equally likely. This means that the probability of any given outcome is constant across the range of possible outcomes.

Uniform distributions can be continuous or discrete. In a continuous uniform distribution, the outcomes can take any value within a specified range. In contrast, a discrete uniform distribution has a finite set of possible outcomes.

## Uniform Distributions with NumPy

NumPy provides the
**numpy.random.uniform()**function to generate samples from a continuous uniform distribution. This function allows you to specify the range and size of the generated samples.
### Example

In this example, we generate 10 random samples from a uniform distribution between
**0**and**1**−
```
import numpy as np

# Generate 10 random samples from a uniform distribution between 0 and 1
samples = np.random.uniform(0, 1, 10)
print("Random samples from uniform distribution:", samples)
```

Following is the output obtained −

```
Random samples from uniform distribution: [0.70748409 0.45654756 0.73426382 0.15580835 0.70294526 0.12503631
 0.40303738  0.9862709  0.4923119  0.44059809]
```

## Visualizing Uniform Distributions

Visualizing uniform distributions helps to understand their properties better. We can use libraries such as Matplotlib to create histograms that display the distribution of generated samples.

### Example

In the following example, we are generating 1000 random samples from a uniform distribution between 0 and 1 and then create a histogram to visualize this distribution −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 random samples from a uniform distribution between 0 and 1
samples = np.random.uniform(0, 1, 1000)

# Create a histogram to visualize the distribution
plt.hist(samples, bins=30, edgecolor='black')
plt.title('Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

The histogram shows that the samples are uniformly distributed between
**0**and**1**, with an approximately equal frequency for each bin −![Uniform Distribution](/numpy/images/uniform_distribution.jpg)
## Applications of Uniform Distributions

Uniform distributions are used in various fields, including computer simulations, Monte Carlo methods, and random sampling. Here are a few practical applications −

- **Simulation:**Uniform distributions are used to simulate random events in models where each outcome is equally likely.
- **Random Sampling:**Uniform distributions are used to generate random samples from a population for statistical analysis.
- **Monte Carlo Methods:**Uniform distributions are used in Monte Carlo simulations to generate random numbers for estimating complex integrals and solving problems in physics and finance.
## Generating Discrete Uniform Distributions

NumPy also allows generating samples from a discrete uniform distribution using the
**numpy.random.randint()**function. This function generates random integers within a specified range.
### Example

In this example, we generate 10 random integers between
**1**and**10**, inclusive −
```
import numpy as np

# Generate 10 random integers between 1 and 10
samples = np.random.randint(1, 11, 10)
print("Random samples from discrete uniform distribution:", samples)
```

Following is the output obtained −

```
Random samples from discrete uniform distribution: [ 7  3  9 10  9  4  8  4  1  7]
```

## Properties of Uniform Distributions

Uniform distributions have several key properties, they are as follows −

- **Constant Probability Density:**In a continuous uniform distribution, the probability density function (PDF) is constant across the specified range.
- **Equal Likelihood:**All outcomes within the range are equally likely.
- **Mean and Variance:**For a continuous uniform distribution between**a**and**b**, the mean is**(a + b) / 2**and the variance is**((b - a)**.) / 12
## Calculating Mean and Variance

You can calculate the mean and variance of a uniform distribution using simple formulas. Let us see how to calculate the mean and variance for a uniform distribution between
**a**and**b**.
### Example

In this example, the mean and variance of the uniform distribution between
**0**and**1**are calculated −
```
import numpy as np

# Define the range of the uniform distribution
a = 0
b = 1

# Calculate the mean and variance
mean = (a + b) / 2
variance = ((b - a) ** 2) / 12

print("Mean:", mean)
print("Variance:", variance)
```

Following is the output obtained −

```
Mean: 0.5
Variance: 0.08333333333333333
```

## Uniform Distribution in Multidimensional Arrays

NumPy can generate uniform distributions for multidimensional arrays as well. Here is an example:

### Example

In this example, a 3x3 array of random samples from a uniform distribution between
**0**and**1**is generated −
```
import numpy as np

# Generate a 3x3 array of random samples from a uniform distribution between 0 and 1
samples = np.random.uniform(0, 1, (3, 3))
print("3x3 array of random samples from uniform distribution:", samples)
```

The output obtained is as shown below −

```
3x3 array of random samples from uniform distribution: 
[[0.18528116 0.65725829 0.06597822]
 [0.73183704 0.05931206 0.65555952]
 [0.92479579 0.89807463 0.02624335]]
```

## Seeding for Reproducibility

To ensure reproducibility, you can set a specific seed before generating uniform distributions. This ensures that the same sequence of random numbers is generated each time you run the code.

### Example

By setting the seed, you ensure that the random generation produces the same result every time the code is executed as shown in the following example −

```
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate 10 random samples from a uniform distribution between 0 and 1
samples = np.random.uniform(0, 1, 10)
print("Random samples with seed 42:", samples)
```

Following is the output obtained −

```
Random samples with seed 42: [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864 0.15599452
 0.05808361 0.86617615 0.60111501 0.70807258]
```

---

## 92. NumPy - Normal Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_normal_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_normal_distribution.htm)*

---

---
[Previous](/numpy/numpy_uniform_distribution.htm)[Quiz](/numpy/quiz_on_numpy_normal_distribution.htm)[Next](/numpy/numpy_binomial_distribution.htm)
## What is a Normal Distribution?

A normal distribution, also known as the Gaussian distribution, is a continuous probability distribution that is symmetric around its mean, indicating that data near the mean are more frequent in occurrence than data far from the mean.

The shape of the normal distribution is described by its mean () and standard deviation (). The mean determines the center of the distribution, while the standard deviation controls the spread of the data.

## Normal Distributions in NumPy

NumPy provides the
**numpy.random.normal()**function to generate samples from a normal distribution. This function allows you to specify the mean, standard deviation, and size of the generated samples.
### Example

In this example, we generate 10 random samples from a normal distribution with a mean of
**0**and a standard deviation of**1**−
```
import numpy as np

# Generate 10 random samples from a normal distribution with mean 0 and standard deviation 1
samples = np.random.normal(0, 1, 10)
print("Random samples from normal distribution:", samples)
```

Following is the output obtained −

```
Random samples from normal distribution: [ 1.45958315 -1.47376803  0.86885907  0.28076705 -2.16173553 -0.43457503
  0.47706858  0.65894456  0.56166159 -0.71025105]
```

## Visualizing Normal Distributions

Visualizing normal distributions helps to understand their properties better. We can use libraries such as Matplotlib to create histograms that display the distribution of generated samples.

### Example

In the following example, we are generating 1000 random samples from a normal distribution with mean 0 and standard deviation 1 and then create a histogram to visualize this distribution −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 random samples from a normal distribution with mean 0 and standard deviation 1
samples = np.random.normal(0, 1, 1000)

# Create a histogram to visualize the distribution
plt.hist(samples, bins=30, edgecolor='black', density=True)

# Plot the probability density function (PDF)
x = np.linspace(-4, 4, 1000)
pdf = 1/(np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
plt.plot(x, pdf, 'r', linewidth=2)
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

The histogram shows that the samples follow a bell-shaped curve, which is characteristic of a normal distribution. The red line represents the theoretical probability density function (PDF) of the normal distribution −
![Normal Distribution](/numpy/images/normal_distribution.jpg)
## Applications of Normal Distributions

Normal distributions are used in various fields, including statistics, finance, engineering, and the natural and social sciences. Here are a few practical applications:

- **Statistical Analysis:**Many statistical tests and methods assume that the data follow a normal distribution.
- **Quality Control:**In manufacturing, normal distributions are used to monitor and control processes.
- **Finance:**Asset returns are often modeled using normal distributions.
## Generating Multivariate Normal Distributions

NumPy also allows generating samples from a multivariate normal distribution using the
**numpy.random.multivariate_normal()**function. This function generates samples from a multivariate normal distribution with a specified mean vector and covariance matrix.
### Example

In this example, we generate 1000 random samples from a multivariate normal distribution with a specified mean vector and covariance matrix −

```
import numpy as np

# Define the mean vector and covariance matrix
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]

# Generate 1000 random samples from a multivariate normal distribution
samples = np.random.multivariate_normal(mean, cov, 1000)

print("Random samples from multivariate normal distribution:", samples[:5])
```

The output obtained is as shown below −

```
Random samples from multivariate normal distribution: 
[[-0.13543463  1.3100422 ]
 [-1.46447528 -0.42485422]
 [ 0.31941286 -0.33503219]
 [ 0.86726151  1.43161159]
 [ 0.12539345 -1.72856329]]
```

## Properties of Normal Distributions

Normal distributions have several key properties, they are −

- **Symmetry:**The normal distribution is symmetric around the mean.
- **Mean, Median, and Mode:**In a normal distribution, the mean, median, and mode are all equal.
- **Empirical Rule:**Approximately 68% of the data falls within one standard deviation of the mean, 95% within two standard deviations, and 99.7% within three standard deviations.
## Standard Normal Distribution

The standard normal distribution is a special case of the normal distribution with a mean of
**0**and a standard deviation of**1**. It is often used as a reference distribution. You can generate samples from a standard normal distribution using the**numpy.random.standard_normal()**function.
### Example

In this example, we generate 10 random samples from a standard normal distribution −

```
import numpy as np

# Generate 10 random samples from a standard normal distribution
samples = np.random.standard_normal(10)
print("Random samples from standard normal distribution:", samples)
```

The result produced is as follows −

```
Random samples from standard normal distribution: [ 0.41271088 -0.06102183 -0.48159376  0.63379932 -0.41831826 -0.67104197
  0.2019988   0.52954154 -0.39241029 -0.19626287]
```

## Seeding for Reproducibility

To ensure reproducibility, you can set a specific seed before generating normal distributions. This ensures that the same sequence of random numbers is generated each time you run the code.

### Example

By setting the seed, you ensure that the random generation produces the same result every time the code is executed as shown in the example below −

```
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate 10 random samples from a normal distribution with mean 0 and standard deviation 1
samples = np.random.normal(0, 1, 10)
print("Random samples with seed 42:", samples)
```

We get the output as shown below −

```
Random samples with seed 42: [ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337 -0.23413696
  1.57921282  0.76743473 -0.46947439  0.54256004]
```

---

## 93. NumPy - Binomial Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_binomial_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_binomial_distribution.htm)*

---

---
[Previous](/numpy/numpy_normal_distribution.htm)[Quiz](/numpy/quiz_on_numpy_binomial_distribution.htm)[Next](/numpy/numpy_poisson_distribution.htm)
## What is a Binomial Distribution?

The Binomial Distribution is a discrete probability distribution that describes the number of successes in a fixed number of independent trials, each with the same probability of success.

It is defined by two parameters: the number of trials (n) and the probability of success (p) in each trial. The probability mass function (PMF) of the binomial distribution gives the probability of getting exactly k successes in n trials. The formula for the PMF is −

```
P(X = k) = C(n, k) * pk * (1 - p)(n - k)
```
* (1 - p)
Where C(n, k) is the binomial coefficient, which can be calculated as −

```
C(n, k) = n! / (k! * (n - k)!)
```

## Binomial Distributions in NumPy

NumPy provides the
**numpy.random.binomial()**function to generate samples from a binomial distribution. This function allows you to specify the number of trials, the probability of success, and the size of the generated samples.
### Example

In this example, we generate 10 random samples from a binomial distribution with 10 trials and a success probability of 0.5 −

```
import numpy as np

# Generate 10 random samples from a binomial distribution with 10 trials and a success probability of 0.5
samples = np.random.binomial(n=10, p=0.5, size=10)
print("Random samples from binomial distribution:", samples)
```

Following is the output obtained −

```
Random samples from binomial distribution: [5 7 5 7 1 3 5 8 7 5]
```

## Visualizing Binomial Distributions

Visualizing binomial distributions helps to understand their properties better. We can use libraries such as Matplotlib to create histograms that display the distribution of generated samples.

### Example

In the following example, we are first generating 1000 random samples from a binomial distribution with 10 trials and a success probability of 0.5. We then visualize this distribution by creating a histogram −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 random samples from a binomial distribution with 10 trials and a success probability of 0.5
samples = np.random.binomial(n=10, p=0.5, size=1000)

# Create a histogram to visualize the distribution
plt.hist(samples, bins=np.arange(12) - 0.5, edgecolor='black', density=True)
plt.title('Binomial Distribution')
plt.xlabel('Number of successes')
plt.ylabel('Frequency')
plt.xticks(range(11))
plt.show()
```

The histogram shows the frequency of the number of successes in the binomial trials. The bars represent the probability of each possible outcome, which forms the characteristic shape of a binomial distribution −
![Binomial Distribution](/numpy/images/binomial_distribution.jpg)
## Applications of Binomial Distributions

Binomial distributions are used in various fields, including statistics, medicine, quality control, and social sciences. Here are a few practical applications −

- **Quality Control:**Used to model the number of defective items in a batch of products.
- **Medicine:**Used to model the number of patients who recover from a treatment out of a sample of patients.
- **Survey Analysis:**Used to model the number of people who respond positively to a survey question.
## Generating Cumulative Binomial Distributions

Sometimes, we are interested in the cumulative distribution function (CDF) of a binomial distribution, which gives the probability of getting up to and including k successes in n trials.

NumPy does not have a built-in function for the CDF of a binomial distribution, but we can calculate it using a loop and the
**scipy.stats.binom.cdf()**function from the SciPy library.
### Example

In this example, we are generating cumulative binomial distribution using NumPy library −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Define the number of trials and success probability
n = 10
p = 0.5

# Generate the cumulative distribution function (CDF) values
x = np.arange(0, n+1)
cdf = binom.cdf(x, n, p)

# Plot the CDF
plt.plot(x, cdf, marker='o', linestyle='-', color='b')
plt.title('Cumulative Binomial Distribution')
plt.xlabel('Number of successes')
plt.ylabel('Cumulative probability')
plt.grid(True)
plt.show()
```

The plot shows the cumulative probability of getting up to and including each number of successes in the binomial trials. The CDF is a step function that increases to 1 as the number of successes increases −
![Cumulative Probability](/numpy/images/cumulative_probability.jpg)
## Properties of Binomial Distributions

Binomial distributions have several key properties, they are −

- **Discrete Nature:**The binomial distribution is discrete, meaning it only takes on integer values.
- **Mean:**The mean of a binomial distribution is given by**n * p**.
- **Variance:**The variance of a binomial distribution is given by**n * p * (1 - p)**.
- **Symmetry:**When p = 0.5, the binomial distribution is symmetric.
## Binomial Distribution for Hypothesis Testing

Binomial distributions are often used in hypothesis testing, particularly in tests for proportions.

One common test is the binomial test, which is used to determine if the proportion of successes in a sample is significantly different from a specified proportion. Here is an example using the
**scipy.stats.binom_test()**function.
### Example

In this example, we perform a binomial test to determine if the proportion of successes (8 out of 10) is significantly different from 0.5. The p-value indicates the probability of obtaining a result at least as extreme as the observed result, assuming the null hypothesis is true −

```
from scipy.stats import binom_test

# Number of successes
successes = 8

# Number of trials
trials = 10

# Hypothesized probability of success
p = 0.5

# Perform the binomial test
p_value = binom_test(successes, trials, p)
print("P-value from binomial test:", p_value)
```

The output obtained is as shown below −

```
/home/cg/root/673c4ae169586/main.py:13: DeprecationWarning: 'binom_test' is deprecated in favour of 'binomtest' from version 1.7.0 and will be removed in Scipy 1.12.0.
  p_value = binom_test(successes, trials, p)
P-value from binomial test: 0.109375
```

## Seeding for Reproducibility

To ensure reproducibility, you can set a specific seed before generating binomial distributions. This ensures that the same sequence of random numbers is generated each time you run the code.

### Example

By setting the seed, you ensure that the random generation produces the same result every time the code is executed, as shown in the example below −

```
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate 10 random samples from a binomial distribution with 10 trials and a success probability of 0.5
samples = np.random.binomial(n=10, p=0.5, size=10)
print("Random samples with seed 42:", samples)
```

The result produced is as follows −

```
Random samples with seed 42: [4 8 6 5 3 3 3 7 5 6]
```

---

## 94. NumPy - Poisson Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_poisson_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_poisson_distribution.htm)*

---

---
[Previous](/numpy/numpy_binomial_distribution.htm)[Quiz](/numpy/quiz_on_numpy_poisson_distribution.htm)[Next](/numpy/numpy_exponential_distribution.htm)
## What is a Poisson Distribution?

The Poisson distribution is characterized by a single parameter,  (lambda), which is the average number of events in the given interval. The probability mass function (PMF) of the Poisson distribution gives the probability of observing k events in the interval and is defined as −

```
P(X = k) = (k * e(-)) / k!
```
* e) / k!
Where,

- **:**It is the average number of events in the interval.
- **k:**It represents the number of events.
- **e:**It is the Euler's number (approximately 2.71828).
## Poisson Distributions in NumPy

NumPy provides the
**numpy.random.poisson()**function to generate samples from a Poisson distribution. You can specify the mean rate () and the size of the generated samples.
### Example

In this example, we generate 10 random samples from a Poisson distribution with a mean rate of 3 events per interval −

```
import numpy as np

# Generate 10 random samples from a Poisson distribution with =3
samples = np.random.poisson(lam=3, size=10)
print("Random samples from Poisson distribution:", samples)
```

Following is the output obtained −

```
Random samples from Poisson distribution: [3 1 2 2 1 1 2 5 5 3]
```

## Visualizing Poisson Distributions

Visualizing Poisson distributions helps to understand their properties better. We can use libraries such as Matplotlib to create histograms that display the distribution of generated samples.

### Example

In the following example, we are first generating random samples from a Poisson distribution with =3. We then create a histogram to visualize this distribution −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 random samples from a Poisson distribution with =3
samples = np.random.poisson(lam=3, size=1000)

# Create a histogram to visualize the distribution
plt.hist(samples, bins=range(10), edgecolor='black', density=True)
plt.title('Poisson Distribution')
plt.xlabel('Number of events')
plt.ylabel('Frequency')
plt.xticks(range(10))
plt.show()
```

The histogram shows the frequency of the number of events in the Poisson trials. The bars represent the probability of each possible outcome, which forms the characteristic shape of a Poisson distribution −
![Poisson Distribution](/numpy/images/poisson_distribution.jpg)
## Applications of Poisson Distributions

Poisson distributions are used in various fields to model the occurrence of events over time or space. Here are a few practical applications −

- **Traffic Engineering:**Modeling the number of cars passing through a checkpoint.
- **Finance:**Modeling the number of trades executed on a stock exchange.
- **Queueing Theory:**Modeling the number of customers arriving at a service point.
## Generating Cumulative Poisson Distributions

Sometimes, we are interested in the cumulative distribution function (CDF) of a Poisson distribution, which gives the probability of getting up to and including k events in the interval.

NumPy does not have a built-in function for the CDF of a Poisson distribution, but we can calculate it using a loop and the
**scipy.stats.poisson.cdf()**function from the SciPy library.
### Example

In the example below, we are generating cumulative poisson distribution in NumPy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define the mean rate
lam = 3

# Generate the cumulative distribution function (CDF) values
x = np.arange(0, 10)
cdf = poisson.cdf(x, lam)

# Plot the CDF
plt.plot(x, cdf, marker='o', linestyle='-', color='b')
plt.title('Cumulative Poisson Distribution')
plt.xlabel('Number of events')
plt.ylabel('Cumulative probability')
plt.grid(True)
plt.show()
```

The plot shows the cumulative probability of getting up to and including each number of events in the Poisson trials. The CDF is a step function that increases to 1 as the number of events increases −
![Cumulative Poisson Distribution](/numpy/images/cumulative_poisson_distribution.jpg)
## Properties of Poisson Distributions

Poisson distributions have several key properties, they are −

- **Discrete Nature:**The Poisson distribution is discrete, meaning it only takes on integer values.
- **Mean and Variance:**The mean and variance of a Poisson distribution are both equal to .
- **Skewness:**The distribution is skewed to the right, especially for smaller values of .
## Poisson Distribution for Hypothesis Testing

Poisson distributions are often used in hypothesis testing, particularly in tests for counts of events.

One common test is the Poisson test, which is used to determine if the observed number of events is significantly different from the expected number. Here is an example using the
**scipy.stats.poisson()**function.
### Example

In this example, we perform a Poisson test to determine if the observed number of events (10) is significantly different from the expected rate (5). The p-value indicates the probability of obtaining a result at least as extreme as the observed result, assuming the null hypothesis is true −

```
from scipy.stats import poisson

# Observed number of events
observed_events = 10

# Expected number of events (mean rate )
expected_rate = 5

# Perform the Poisson test
p_value = poisson.sf(observed_events-1, expected_rate)
print("P-value from Poisson test:", p_value)
```

The output obtained is as shown below −

```
P-value from Poisson test: 0.03182805730620481
```

## Seeding for Reproducibility

To ensure reproducibility, you can set a specific seed before generating Poisson distributions. This ensures that the same sequence of random numbers is generated each time you run the code.

### Example

By setting the seed, you ensure that the random generation produces the same result every time the code is executed as shown in the example below −

```
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate 10 random samples from a Poisson distribution with =3
samples = np.random.poisson(lam=3, size=10)
print("Random samples with seed 42:", samples)
```

The result produced is as follows −

```
Random samples with seed 42: [4 1 3 3 2 3 2 3 0 2]
```

---

## 95. NumPy - Exponential Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_exponential_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_exponential_distribution.htm)*

---

---
[Previous](/numpy/numpy_poisson_distribution.htm)[Quiz](/numpy/quiz_on_numpy_exponential_distribution.htm)[Next](/numpy/numpy_rayleigh_distribution.htm)
## Exponential Distribution in NumPy

The Exponential Distribution is a continuous probability distribution used to model the time between independent events that occur at a constant average rate.

It is defined by a single parameter  (lambda), which is the rate parameter, where the mean time between events is 1/. This distribution is often used in scenarios such as the time between arrivals in a queue.

For example, if events occur on average every 5 minutes (=1/5), the exponential distribution models the time between these events.

The probability density function (PDF) of the exponential distribution is defined as −

```
f(x; ) =  * exp(-x) for x  0, 0 otherwise
```

Where,

- **:**Rate parameter (inverse of the mean).
- **x:**Time between events.
- **exp:**Exponential function.
## Exponential Distributions in NumPy

NumPy provides the
**numpy.random.exponential()**function to generate samples from an exponential distribution. You can specify the scale parameter (1/) and the size of the generated samples.
### Example

In this example, we generate 10 random samples from an exponential distribution with a rate parameter =2 (scale=0.5) −

```
import numpy as np

# Generate 10 random samples from an exponential distribution with =2 (scale=0.5)
samples = np.random.exponential(scale=0.5, size=10)
print("Random samples from exponential distribution:", samples)
```

Following is the output obtained −

```
Random samples from exponential distribution: [0.49251018 0.14301367 0.48682864 0.09396999 0.11923932 0.28674142
 0.14285472 0.11916798 1.59846326 0.27175065]
```

## Visualizing Exponential Distributions

Visualizing exponential distributions helps to understand their properties better. We can use libraries such as Matplotlib to create histograms that display the distribution of generated samples.

### Example

In the following example, we are first generating 1000 random samples from an exponential distribution with =2 (scale=0.5). We are then creating a histogram to visualize this distribution −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 random samples from an exponential distribution with =2 (scale=0.5)
samples = np.random.exponential(scale=0.5, size=1000)

# Create a histogram to visualize the distribution
plt.hist(samples, bins=30, edgecolor='black', density=True)
plt.title('Exponential Distribution')
plt.xlabel('Time between events')
plt.ylabel('Frequency')
plt.show()
```

The histogram shows the frequency of the time between events in the exponential trials. The bars represent the probability of each possible outcome, which forms the characteristic shape of an exponential distribution −
![Exponential Distribution](/numpy/images/exponential_distribution.jpg)
## Applications of Exponential Distributions

Exponential distributions are used in various fields to model the time between events in a Poisson process. Here are a few practical applications −

- **Reliability Analysis:**Modeling the time between failures of mechanical systems.
- **Queuing Theory:**Modeling the time between arrivals of customers in a queue.
- **Survival Analysis:**Modeling the time until an event, such as death or failure, occurs.
## Generating Cumulative Exponential Distributions

Sometimes, we are interested in the cumulative distribution function (CDF) of an exponential distribution, which gives the probability of getting up to and including x events in the interval.

NumPy does not have a built-in function for the CDF of an exponential distribution, but we can calculate it using a loop and the
**scipy.stats.expon.cdf()**function from the SciPy library.
### Example

Following is an example to generate cumulative exponential distribution in NumPy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Define the rate parameter
lam = 2

# Generate the cumulative distribution function (CDF) values
x = np.linspace(0, 3, 100)
cdf = expon.cdf(x, scale=1/lam)

# Plot the CDF
plt.plot(x, cdf, marker='o', linestyle='-', color='b')
plt.title('Cumulative Exponential Distribution')
plt.xlabel('Time between events')
plt.ylabel('Cumulative probability')
plt.grid(True)
plt.show()
```

The plot shows the cumulative probability of getting up to and including each time between events in the exponential trials. The CDF is a smooth curve that increases to 1 as the time between events increases −
![Cumulative Exponential Distribution](/numpy/images/cumulative_exponential_distribution.jpg)
## Properties of Exponential Distributions

Exponential distributions have several key properties, they are −

- **Memoryless Property:**The probability of an event occurring in the future is independent of the time that has already passed.
- **Mean and Variance:**The mean of an exponential distribution is 1/, and the variance is 1/.
- **Skewness:**The distribution is skewed to the right, with a long tail.
## Exponential Distribution for Hypothesis Testing

Exponential distributions are often used in hypothesis testing, particularly in tests for the time between events.

One common test is the exponential test, which is used to determine if the observed time between events is significantly different from the expected time. Here is an example using the
**scipy.stats.expon()**function:
### Example

In this example, we perform an exponential test to determine if the observed time between events (0.8) is significantly different from the expected rate (=2). The p-value indicates the probability of obtaining a result at least as extreme as the observed result, assuming the null hypothesis is true −

```
from scipy.stats import expon

# Observed time between events
observed_time = 0.8

# Expected rate parameter ()
expected_rate = 2

# Perform the exponential test
p_value = expon.sf(observed_time, scale=1/expected_rate)
print("P-value from exponential test:", p_value)
```

The output obtained is as shown below −

```
P-value from exponential test: 0.20189651799465538
```

## Seeding for Reproducibility

To ensure reproducibility, you can set a specific seed before generating exponential distributions. This ensures that the same sequence of random numbers is generated each time you run the code.

### Example

By setting the seed, you ensure that the random generation produces the same result every time the code is executed as shown in the example below −

```
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate 10 random samples from an exponential distribution with =2 (scale=0.5)
samples = np.random.exponential(scale=0.5, size=10)
print("Random samples with seed 42:", samples)
```

The result produced is as follows −

```
Random samples with seed 42: [0.23463404 1.50506072 0.65837285 0.45647128 0.08481244 0.08479815
 0.02991938 1.00561543 0.45954108 0.61562503]
```

---

## 96. NumPy - Rayleigh Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_rayleigh_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_rayleigh_distribution.htm)*

---

---
[Previous](/numpy/numpy_exponential_distribution.htm)[Quiz](/numpy/quiz_on_numpy_rayleigh_distribution.htm)[Next](/numpy/numpy_logistic_distribution.htm)
## What is a Rayleigh Distribution?

The Rayleigh Distribution is a continuous probability distribution used to model the magnitude of a vector in a two-dimensional plane, where its components are independent and normally distributed with equal variance.

It is defined by a scale parameter  (sigma). This distribution is often used in signal processing and communication theory to model scattered signals.

For example, the Rayleigh distribution can describe the distribution of wind speeds given that the wind velocity components in two orthogonal directions are independent and normally distributed.

The probability density function (PDF) of the Rayleigh distribution is defined as −

```
f(x; ) = (x / 2) * exp(-x2 / (22)) for x  0, 0 otherwise
```
) * exp(-x/ (2)) for x  0, 0 otherwise
Where,

- **:**Scale parameter (related to the standard deviation).
- **x:**Magnitude of the vector.
- **exp:**Exponential function.
## Generating Rayleigh Distributions with NumPy

NumPy provides the
**numpy.random.rayleigh()**function to generate samples from a Rayleigh distribution. You can specify the scale parameter  and the size of the generated samples.
### Example

In this example, we generate 10 random samples from a Rayleigh distribution with a scale parameter =1 −

```
import numpy as np

# Generate 10 random samples from a Rayleigh distribution with =1
samples = np.random.rayleigh(scale=1, size=10)
print("Random samples from Rayleigh distribution:", samples)
```

Following is the output obtained −

```
Random samples from Rayleigh distribution: [1.31998799 0.72631303 2.4544915  0.31195556 1.14244968 0.299947020.74889027 0.2239033  1.43290625 1.18894253]
```

## Visualizing Rayleigh Distributions

Visualizing Rayleigh distributions helps to understand their properties better. We can use libraries such as Matplotlib to create histograms that display the distribution of generated samples.

### Example

In the following example, we are first generating 1000 random samples from a rayleigh distribution with =1. We are then creating a histogram to visualize this distribution −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 random samples from a Rayleigh distribution with =1
samples = np.random.rayleigh(scale=1, size=1000)

# Create a histogram to visualize the distribution
plt.hist(samples, bins=30, edgecolor='black', density=True)
plt.title('Rayleigh Distribution')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.show()
```

The histogram shows the frequency of the magnitude of the vector in the Rayleigh trials. The bars represent the probability of each possible outcome, which forms the characteristic shape of a Rayleigh distribution −
![Rayleigh Distribution](/numpy/images/rayleigh_distribution.jpg)
## Applications of Rayleigh Distributions

Rayleigh distributions are used in various fields to model the magnitude of a vector whose components are Gaussian random variables. Here are a few practical applications −

- **Signal Processing:**Modeling the envelope of a signal with multiple scattered paths.
- **Radar Systems:**Modeling the received signal strength from targets with random scattering.
- **Communication Theory:**Modeling the amplitude of a signal affected by multipath fading.
## Generating Cumulative Rayleigh Distributions

Sometimes, we are interested in the cumulative distribution function (CDF) of a Rayleigh distribution, which gives the probability of getting up to and including x events in the interval.

NumPy does not have a built-in function for the CDF of a Rayleigh distribution, but we can calculate it using a loop and the
**scipy.stats.rayleigh.cdf()**function from the SciPy library.
### Example

Following is an example to generate cumulative Rayleigh distribution in NumPy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh

# Define the scale parameter
sigma = 1

# Generate the cumulative distribution function (CDF) values
x = np.linspace(0, 5, 100)
cdf = rayleigh.cdf(x, scale=sigma)

# Plot the CDF
plt.plot(x, cdf, marker='o', linestyle='-', color='b')
plt.title('Cumulative Rayleigh Distribution')
plt.xlabel('Magnitude')
plt.ylabel('Cumulative probability')
plt.grid(True)
plt.show()
```

The plot shows the cumulative probability of getting up to and including each magnitude in the Rayleigh trials. The CDF is a smooth curve that increases to 1 as the magnitude increases −
![Cumulative Rayleigh Distribution](/numpy/images/cumulative_rayleigh_distribution.jpg)
## Properties of Rayleigh Distributions

Rayleigh distributions have several key properties, such as −

- **Scale Parameter ():**The scale parameter is related to the standard deviation of the underlying Gaussian random variables.
- **Mean and Variance:**The mean of a Rayleigh distribution is (/2), and the variance is (2-/2).
- **Skewness:**The distribution is skewed to the right, with a long tail.
## Using Rayleigh Distribution for Hypothesis Testing

Rayleigh distributions are often used in hypothesis testing, particularly in tests for the magnitude of a vector.

One common test is the Rayleigh test, which is used to determine if the observed magnitude is significantly different from the expected magnitude. Here is an example using the
**scipy.stats.rayleigh()**function:
### Example

In this example, we perform a Rayleigh test to determine if the observed magnitude (1.5) is significantly different from the expected scale (=1). The p-value indicates the probability of obtaining a result at least as extreme as the observed result, assuming the null hypothesis is true −

```
from scipy.stats import rayleigh

# Observed magnitude
observed_magnitude = 1.5

# Expected scale parameter ()
expected_scale = 1

# Perform the Rayleigh test
p_value = rayleigh.sf(observed_magnitude, scale=expected_scale)
print("P-value from Rayleigh test:", p_value)
```

The output obtained is as shown below −

```
P-value from Rayleigh test: 0.32465246735834974
```

## Seeding for Reproducibility

To ensure reproducibility, you can set a specific seed before generating Rayleigh distributions. This ensures that the same sequence of random numbers is generated each time you run the code.

### Example

By setting the seed, you ensure that the random generation produces the same result every time the code is executed as shown in the example below −

```
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate 10 random samples from a Rayleigh distribution with =1
samples = np.random.rayleigh(scale=1, size=10)
print("Random samples with seed 42:", samples)
```

The result produced is as follows −

```
Random samples with seed 42: [0.96878077 2.45361832 1.62280356 1.35125316 0.58245149 0.58240242
 0.34594441 2.00560757 1.35578918 1.56923552]
```

---

## 97. NumPy - Logistic Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_logistic_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_logistic_distribution.htm)*

---

---
[Previous](/numpy/numpy_rayleigh_distribution.htm)[Quiz](/numpy/quiz_on_numpy_logistic_distribution.htm)[Next](/numpy/numpy_pareto_distribution.htm)
## What is a Logistic Distribution?

The Logistic Distribution is a continuous probability distribution used to model growth and logistic regression.

It is defined by two parameters: the location parameter  (mean) and the scale parameter s. The distribution is similar to the normal distribution but has heavier tails, meaning it has a higher probability of extreme values.

Example: The logistic distribution can describe population growth where the rate of increase is proportional to both the amount present and the amount of growth remaining.

The probability density function (PDF) of the logistic distribution is defined as −

```
f(x; , s) = (e-(x-)/s) / (s * (1 + e-(x-)/s)2)
```
) / (s * (1 + e))
Where,

- **:**Location parameter (mean).
- **s:**Scale parameter (related to the standard deviation).
- **x:**Value of the random variable.
- **e:**Euler's number (approximately 2.71828).
## Generating Logistic Distributions with NumPy

NumPy provides the
**numpy.random.logistic()**function to generate samples from a logistic distribution. You can specify the location parameter , the scale parameter s, and the size of the generated samples.
### Example

In this example, we generate 10 random samples from a logistic distribution with a location parameter =0 and a scale parameter s=1 −

```
import numpy as np

# Generate 10 random samples from a logistic distribution with =0 and s=1
samples = np.random.logistic(loc=0, scale=1, size=10)
print("Random samples from logistic distribution:", samples)
```

Following is the output obtained −

```
Random samples from logistic distribution: [-1.6473898   1.18698013 -0.24048488 -1.05235482  3.11858778 -1.40235809
  0.8399973  -1.46670621 -3.14359949 -0.80023521]
```

## Visualizing Logistic Distributions

Visualizing logistic distributions helps to understand their properties better. We can use libraries such as Matplotlib to create histograms that display the distribution of generated samples.

### Example

In the following example, we are first generating 1000 random samples logistic distribution with =0 and s=1. We are then creating a histogram to visualize this distribution −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 random samples from a logistic distribution with =0 and s=1
samples = np.random.logistic(loc=0, scale=1, size=1000)

# Create a histogram to visualize the distribution
plt.hist(samples, bins=30, edgecolor='black', density=True)
plt.title('Logistic Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

The histogram shows the frequency of the values in the logistic distribution. The bars represent the probability of each possible outcome, forming the characteristic S-shape of the logistic distribution −
![Logistic Distribution](/numpy/images/logistic_distribution.jpg)
## Applications of Logistic Distributions

Logistic distributions are used in various fields to model data with extreme values. Here are a few practical applications −

- **Machine Learning:**Modeling binary outcomes in logistic regression.
- **Economics:**Modeling growth and distribution of income and wealth.
- **Statistics:**Analyzing and predicting outcomes with a logistic model.
## Generating Cumulative Logistic Distributions

Sometimes, we are interested in the cumulative distribution function (CDF) of a logistic distribution, which gives the probability of getting up to and including x events in the interval.

NumPy does not have a built-in function for the CDF of a logistic distribution, but we can calculate it using a loop and the
**scipy.stats.logistic.cdf()**function from the SciPy library.
### Example

Following is an example to generate cumulative logistic distribution in NumPy −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic

# Define the location and scale parameters
loc = 0
scale = 1

# Generate the cumulative distribution function (CDF) values
x = np.linspace(-10, 10, 100)
cdf = logistic.cdf(x, loc=loc, scale=scale)

# Plot the CDF
plt.plot(x, cdf, marker='o', linestyle='-', color='b')
plt.title('Cumulative Logistic Distribution')
plt.xlabel('Value')
plt.ylabel('Cumulative probability')
plt.grid(True)
plt.show()
```

The plot shows the cumulative probability of getting up to and including each value in the logistic trials. The CDF is a smooth curve that increases to 1 as the value increases −
![Cumulative Logistic Distribution](/numpy/images/cumulative_logistic_distribution.jpg)
## Properties of Logistic Distributions

Logistic distributions have several key properties, such as −

- **Location Parameter ():**The location parameter is the mean of the distribution.
- **Scale Parameter (s):**The scale parameter is related to the standard deviation of the distribution.
- **Mean and Variance:**The mean of a logistic distribution is , and the variance is (s/3).
- **Skewness:**The distribution is symmetric around the mean, with heavier tails than a normal distribution.
## Using Logistic Distribution for Hypothesis Testing

Logistic distributions are often used in hypothesis testing, particularly in tests for binary outcomes.

One common test is logistic regression, which is used to model the probability of a binary outcome based on one or more predictor variables. Here is an example using the
**statsmodels**library:
### Example

In this example, we fit a logistic regression model to binary outcome data. The summary provides information about the coefficients, standard errors, and p-values of the model −

```
# Python version 3.11
import numpy as np
import statsmodels.api as sm

# Example data
X = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0, 0, 1, 1, 1])

# Add a constant to the predictor variable
X = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit(method='lbfgs', maxiter=100, disp=0)

# Print the model summary
print(result.summary())
```

The output obtained is as shown below −

```
Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                    6
Model:                          Logit   Df Residuals:                        4
Method:                           MLE   Df Model:                            1
Date:                Wed, 20 Nov 2024   Pseudo R-squ.:                   1.000
Time:                        12:29:27   Log-Likelihood:            -5.7054e-05
converged:                       True   LL-Null:                       -4.1589
Covariance Type:            nonrobust   LLR p-value:                  0.003926
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const        -52.2706    668.240     -0.078      0.938   -1361.997    1257.456
x1            20.9332    265.301      0.079      0.937    -499.046     540.913
==============================================================================

Complete Separation: The results show that there iscomplete separation or perfect prediction.
In this case the Maximum Likelihood Estimator does not exist and the parameters
are not identified.
```

## Seeding for Reproducibility

To ensure reproducibility, you can set a specific seed before generating logistic distributions. This ensures that the same sequence of random numbers is generated each time you run the code.

### Example

In this example, we set the seed to 42 before generating random samples from a logistic distribution. The seed ensures that the same sequence of samples is generated each time the code is run −

```
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate 10 random samples from a logistic distribution with =0 and s=1
samples = np.random.logistic(loc=0, scale=1, size=10)
print("Random samples with seed 42:", samples)
```

Following is the output of the above code −

```
Random samples with seed 42: [-0.51278827  2.95957976  1.00476265  0.39987857 -1.68815492 -1.68833811-2.78603295  1.86756387  0.41011316  0.88604138]
```

---

## 98. NumPy - Pareto Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_pareto_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_pareto_distribution.htm)*

---

---

## 99. NumPy - Visualize Distributions With Seaborn

*Source: [https://www.tutorialspoint.com/numpy/numpy_visualize_distributions_with_seaborn.htm](https://www.tutorialspoint.com/numpy/numpy_visualize_distributions_with_seaborn.htm)*

---

---
[Previous](/numpy/numpy_pareto_distribution.htm)[Quiz](/numpy/quiz_on_numpy_visualize_distributions_with_seaborn.htm)[Next](/numpy/numpy_matplotlib.htm)
## Visualizing Distributions with Seaborn

When working with data, visualizing distributions is an important step in understanding the characteristics of the data.

Seaborn, built on top of Matplotlib, is a powerful visualization library in Python that simplifies the process of creating informative and attractive statistical plots.

In this tutorial, we will explore how to use Seaborn to visualize different types of distributions, including normal, uniform, and other probability distributions. We will also demonstrate how to enhance the visualization with customization options and styling.

## What is Seaborn?

Seaborn is a Python visualization library that provides a high-level interface for creating attractive and informative statistical graphics. It integrates well with Pandas data structures and provides several functions to visualize distributions, relationships, and trends in data.

One of its key strengths is making it easy to visualize distributions, correlations, and data relationships with minimal code.

Seaborn builds on Matplotlib and provides more streamlined functions to create complex plots. It also automatically handles aesthetics, such as color schemes and labels, making your visualizations more attractive and easier to interpret.

## Setting Up Seaborn

Before we start visualizing distributions with Seaborn, we need to install the necessary libraries and set up the environment. You can install Seaborn using pip if it is not already installed as shown below −

```
# Install Seaborn using pip
!pip install seaborn
```

In addition to Seaborn, we will use NumPy to generate data for the distributions. Here is the typical setup for importing both libraries −

```
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```

Once the libraries are imported, we can start generating and visualizing different types of distributions.

## Visualizing a Normal Distribution

One of the most commonly used distributions in statistics is the normal distribution, also known as the Gaussian distribution. It is symmetric and bell-shaped, often used to model things like test scores, heights, and measurement errors.

We can generate random data from a normal distribution using NumPy's
**numpy.random.normal()**function and then use Seaborn's**seaborn.histplot()**function to visualize the distribution.
### Example

In the following example, the
**sns.histplot()**function automatically creates a histogram of the data, and by setting the**kde**parameter to**True**, it adds a smooth Kernel Density Estimate (KDE) curve over the histogram to visualize the probability density function (PDF) −
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data from a normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)

# Visualize the distribution using Seaborn
# kde=True adds a Kernel Density Estimate curve
sns.histplot(data, kde=True)  
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

The resulting plot will show a bell-shaped curve, which is characteristic of the normal distribution −
![Numpy Distribution with Seaborn](/numpy/images/numpy_distribution_with_seaborn.jpg)
## Visualizing a Uniform Distribution

A uniform distribution is a type of distribution in which all outcomes are equally likely. In a continuous uniform distribution, the data points are spread evenly across a given range.

We can generate data from a uniform distribution using NumPy's
**numpy.random.uniform()**function and visualize it using Seaborn.
### Example

Here, the
**numpy.random.uniform()**function generates random numbers between the specified low and high values (0 and 10 in this case). The histogram shows a flat distribution, indicating that all values are equally likely within the specified range −
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data from a uniform distribution
data_uniform = np.random.uniform(low=0, high=10, size=1000)

# Visualize the distribution using Seaborn
sns.histplot(data_uniform, kde=True)
plt.title('Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

The output produced will show a uniform distribution where the frequency of each value is approximately the same across the range −
![Uniform Numpy Seaborn Distribution](/numpy/images/uniform_numpy_seaborn_distribution.jpg)
## Visualizing Exponential Distribution

An exponential distribution is often used to model the time between events in a Poisson process. It is biased with a high frequency of small values and a long tail for larger values.

NumPy provides the
**numpy.random.exponential()**function to generate random data from an exponential distribution.
### Example

In the following example, we are creating a plot that will show a distribution with a peak near zero and a tail extending to the right. This is characteristic of exponential distributions, where the probability of a value occurring decreases exponentially as the value increases −

```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data from an exponential distribution
data_exponential = np.random.exponential(scale=1, size=1000)

# Visualize the distribution using Seaborn
sns.histplot(data_exponential, kde=True)
plt.title('Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

We get the output as shown below −
![NumPy Seaborn Exponential Distribution](/numpy/images/numpy_seaborn_exponential_distribution.jpg)
## Visualizing the Pareto Distribution

As we discussed earlier, the Pareto distribution follows a power-law and is often used in economics to model wealth distribution. You can generate data for a Pareto distribution using NumPy's
**numpy.random.pareto()**function.
### Example

Let us visualize the pareto distribution using Seaborn −

```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data from a Pareto distribution
# Adding 1 to shift the minimum value
data_pareto = np.random.pareto(a=2, size=1000) + 1  

# Visualize the distribution using Seaborn
sns.histplot(data_pareto, kde=True)
plt.title('Pareto Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

The Pareto distribution will show a highly skewed histogram with a long tail extending to the right, reflecting that a few large values dominate the dataset −
![NumPy Seaborn Pareto Distribution](/numpy/images/numpy_seaborn_pareto_distribution.jpg)
## Customizing Seaborn Plots

Seaborn allows you to customize the appearance of the plots easily. For instance, you can adjust the number of bins in the histogram, change the colors of the plot, or even modify the style of the plot. Here are a few ways to customize the appearance −

- **Change the number of bins:**You can control the number of bins in the histogram by specifying the**bins**parameter.
- **Change the color:**Use the**color**parameter to set a custom color for the plot.
- **Modify the style:**Seaborn provides several built-in styles (such as**'darkgrid'**,**'whitegrid'**, etc.) that can be applied to the plot using**sns.set_style()**.
### Example

In the following example, we are creating a plot with 30 bins, a blue color, and a white grid background −

```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data from a normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)

# Customize the plot style
sns.set_style('whitegrid')

# Plot with more bins and custom color
sns.histplot(data, bins=30, color='blue', kde=True)
plt.title('Customized Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

The result produced is as follows −
![NumPy Seaborn Customized Normal Distribution](/numpy/images/numpy_seaborn_customized_normal_distribution.jpg)

---

## 100. NumPy - Multinomial Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_multinomial_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_multinomial_distribution.htm)*

---

---

## 101. NumPy - Chi Square Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_chi_square_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_chi_square_distribution.htm)*

---

---
[Previous](/numpy/numpy_multinomial_distribution.htm)[Quiz](/numpy/quiz_on_numpy_chi_square_distribution.htm)[Next](/numpy/numpy_zipf_distribution.htm)
## What is the Chi-Square Distribution?

The Chi-Square Distribution is a continuous probability distribution used in statistics to test hypotheses about the variance of a population or the independence of two variables.

It is a special type of distribution derived from the sum of squares of independent standard normal random variables. Mathematically, if Z
, Z, ..., Zare independent standard normal variables, then −
```
X = Z12 + Z22 + ... + Zk2
```
+ Z+ ... + Z
It is defined by the degrees of freedom (df), which depend on the number of independent variables in the dataset. This distribution is skewed and becomes more symmetric as the degrees of freedom increase.

Hence, the resulting variable, X, follows a Chi-Square distribution with k degrees of freedom. The degrees of freedom, denoted as k, play an important role in determining the shape of the distribution. Higher degrees of freedom result in a more symmetrical distribution.

## Chi-Square Samples in NumPy

NumPy provides the
**numpy.random.chisquare()**function to generate random samples from a Chi-Square distribution. This function requires two main parameters −
- **df:**Degrees of freedom.
- **size (optional):**The number of samples to generate.
### Example: Generating Chi-Square Samples

The following example generates 10 random samples from a Chi-Square distribution with 5 degrees of freedom −

```
import numpy as np

# Generate Chi-Square samples
degrees_of_freedom = 5
samples = np.random.chisquare(degrees_of_freedom, size=10)
print("Generated Chi-Square samples:", samples)
```

Following is the output obtained −

```
Generated Chi-Square samples: [ 3.94124915  3.61732939  8.09217857  1.63322954  2.26579558  3.74957222
 10.88281092  1.98262239  3.816437   10.83575014]
```

## Properties of the Chi-Square Distribution

The Chi-Square distribution has several important properties that make it useful for statistical analysis, they are −

- **Asymmetry:**The distribution is skewed to the right, especially for lower degrees of freedom. The skewness decreases as the degrees of freedom increase.
- **Mean:**The mean of the Chi-Square distribution is equal to its degrees of freedom (df).
- **Variance:**The variance is twice the degrees of freedom, or 2 * df.
### Example

In the following example we are verifying mean and variance of the given degrees of freedom −

```
import numpy as np

# Verifying mean and variance
df = 5
samples = np.random.chisquare(df, size=1000)

mean = np.mean(samples)
variance = np.var(samples)

print("Mean of samples:", mean)
print("Variance of samples:", variance)
```

This will produce the following result −

```
Mean of samples: 5.04405316596172
Variance of samples: 10.565774002162097
```

## Applications of the Chi-Square Distribution

The Chi-Square distribution is primarily used in hypothesis testing and variance estimation. Common applications are −

- **Goodness-of-Fit Test:**Evaluating how well a set of observed data matches a theoretical distribution.
- **Test of Independence:**Analyzing the independence of two categorical variables using a contingency table.
- **Variance Analysis:**Assessing the variability of a population or comparing variances of two populations.
### Example: Goodness-of-Fit Test

Suppose we have observed frequencies of dice rolls and want to test whether the dice is fair using the Chi-Square distribution −

```
import numpy as np

# Observed and expected frequencies
observed = np.array([16, 18, 16, 14, 18, 18])
expected = np.array([15, 15, 15, 15, 15, 15])

# Chi-Square statistic
chi_square_stat = np.sum((observed - expected)**2 / expected)
print("Chi-Square statistic:", chi_square_stat)
```

This statistic can be compared to a critical value from the Chi-Square distribution table to determine the fairness of the dice −

```
Chi-Square statistic: 2.0
```

## Visualizing the Chi-Square Distribution

Visualization helps in understanding the shape and characteristics of the Chi-Square distribution. We can use Matplotlib to plot its probability density function (PDF).

### Example: Plotting the Chi-Square PDF

In the following example, we create a line plot showing the PDF of the Chi-Square distribution for varying degrees of freedom −

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Plotting PDF for different degrees of freedom
x = np.linspace(0, 20, 500)
dfs = [2, 4, 6, 8]

for df in dfs:
   plt.plot(x, chi2.pdf(x, df), label=f"df={df}")

plt.title("Chi-Square Distribution PDF")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
```

The curves demonstrate how the distribution becomes less skewed as the degrees of freedom increase −
![Chi-Square Distribution](/numpy/images/chi_square_distribution.jpg)
## Simulating Real-World Scenarios

The Chi-Square distribution is often used in practical scenarios such as quality control and risk analysis. Let us simulate a real-world example of quality control in a manufacturing process.

### Example: Quality Control in Manufacturing

Suppose a factory measures the variability of product dimensions. The Chi-Square distribution can test whether the variability is within acceptable limits. This statistic can be used to determine whether the observed variance exceeds the acceptable threshold −

```
import numpy as np

# Observed variance and acceptable threshold
observed_variance = 4.5
sample_size = 20
population_variance = 4.0

# Chi-Square statistic
chi_square_stat = (sample_size - 1) * observed_variance / population_variance
print("Chi-Square statistic:", chi_square_stat)
```

We get the output as shown below −

```
Chi-Square statistic: 21.375
```

---

## 102. NumPy - Zipf Distribution

*Source: [https://www.tutorialspoint.com/numpy/numpy_zipf_distribution.htm](https://www.tutorialspoint.com/numpy/numpy_zipf_distribution.htm)*

---

---
[Previous](/numpy/numpy_chi_square_distribution.htm)[Quiz](/numpy/quiz_on_numpy_zipf_distribution.htm)[Next](/numpy/numpy_with_io.htm)
## What is the Zipf Distribution?

The Zipf Distribution is a discrete probability distribution that describes the frequency of elements ranked in descending order.

It follows the principle that the frequency of an element is inversely proportional to its rank in the frequency table, often seen in natural languages where the most common word appears twice as often as the second most common word, three times as often as the third, and so on.

It is defined by a parameter "s", which determines the skewness of the distribution. Example: The distribution of word frequencies in a large text corpus often follows a Zipf distribution. Mathematically, the probability mass function (PMF) of the Zipf distribution is given by:

```
P(X = k) = (1 / ks) / H(N, s)
```
) / H(N, s)
Here, k is the rank, s is the exponent characterizing the distribution, and H(N, s) is the Nth generalized harmonic number, which normalizes the distribution.

## Generating Zipf Samples in NumPy

NumPy provides the
**numpy.random.zipf()**function to generate random samples from a Zipf distribution. This function requires two main parameters:
- **a:**The distribution parameter, also known as the exponent.
- **size:**The number of samples to generate (optional).
### Example: Generating Zipf Samples

The following example generates 10 random samples from a Zipf distribution with an exponent of 2 −

```
import numpy as np

# Generate Zipf samples
a = 2
samples = np.random.zipf(a, size=10)
print("Generated Zipf samples:", samples)
```

Following is the output obtained −

```
Generated Zipf samples: [1 3 1 2 1 1 1 1 2 1]
```

## Properties of the Zipf Distribution

The Zipf distribution has several unique properties that make it useful for modeling real-world phenomena, they are −

- **Heavy Tail:**The distribution has a heavy tail, meaning a small number of events are very common, while the majority are rare.
- **Power Law:**The probability of an event is inversely proportional to its rank raised to the power of the exponent.
- **Scale-Invariant:**The distribution remains the same when the scale of measurement changes.
### Example: Visualizing Zipf's Law

Let us visualize the Zipf distribution to understand its properties better. We can plot the frequency of occurrences of each rank using Matplotlib −

```
import numpy as np
import matplotlib.pyplot as plt

# Generate a large number of samples
a = 2
samples = np.random.zipf(a, size=1000)

# Count occurrences of each rank
unique, counts = np.unique(samples, return_counts=True)

# Plot the rank-frequency distribution
plt.figure(figsize=(10, 6))
plt.loglog(unique, counts, marker="o")
plt.title("Zipf Distribution")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.show()
```

We obtained a log-log plot showing the frequency of each rank, demonstrating the heavy tail and power-law nature of the Zipf distribution −
![Zipf Distribution](/numpy/images/zipf_distribution.jpg)
## Applications of the Zipf Distribution

The Zipf distribution is used in various fields to model phenomena where a few items are very common, and many items are rare. Common applications are −

- **Natural Language Processing (NLP):**Modeling word frequencies in a corpus.
- **Population Studies:**Analyzing city populations and sizes.
- **Internet Traffic:**Understanding the distribution of web page visits and hits.
### Example: Word Frequency in Text

Suppose we have a text document and want to model the frequency of words using the Zipf distribution −

```
import matplotlib.pyplot as plt
from collections import Counter

# Sample text
text = "the quick brown fox jumps over the lazy dog the quick brown fox"

# Split text into words
words = text.split()

# Count word frequencies
word_counts = Counter(words)

# Get ranks and frequencies
ranks = range(1, len(word_counts) + 1)
frequencies = [count for word, count in word_counts.most_common()]

# Plot the rank-frequency distribution
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker="o")
plt.title("Word Frequency Distribution")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.show()
```

We obtain a log-log plot showing the frequency of each word rank in the text, following the Zipf distribution −
![Word Frequency](/numpy/images/word_frequency.jpg)
## Estimating Zipf's Exponent

In real-world applications, we often need to estimate the exponent parameter of the Zipf distribution. This can be done using various statistical techniques. One simple method is to fit a line to the log-log plot of rank vs. frequency.

### Example: Estimating the Exponent

In the example below, the estimated exponent is close to the actual value (a = 2), demonstrating the accuracy of the estimation method −

```
import numpy as np
from scipy.stats import linregress

# Generate a large number of samples
a = 2
samples = np.random.zipf(a, size=1000)

# Count occurrences of each rank
unique, counts = np.unique(samples, return_counts=True)

# Perform linear regression on the log-log plot
log_ranks = np.log(unique)
log_counts = np.log(counts)
slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_counts)

print("Estimated exponent:", -slope)
```

The output obtained is as shown below −

```
Estimated exponent: 0.8852106553815038
```

## Simulating Real-World Scenarios

Let us simulate a real-world scenario using the Zipf distribution. Suppose we want to model the distribution of website visits on a popular news site.

### Example: Website Visits

We can generate a large number of samples from a Zipf distribution and analyze the distribution of visits to different pages −

```
import matplotlib.pyplot as plt
import numpy as np

# Generate Zipf samples
a = 1.5
samples = np.random.zipf(a, size=10000)

# Count occurrences of each page visit
unique, counts = np.unique(samples, return_counts=True)

# Plot the rank-frequency distribution
plt.figure(figsize=(10, 6))
plt.loglog(unique, counts, marker="o")
plt.title("Website Visits Distribution")
plt.xlabel("Page Rank")
plt.ylabel("Visit Frequency")
plt.show()
```

We obtain a log-log plot showing the distribution of website visits, demonstrating the heavy tail and power-law nature of the Zipf distribution −
![Website Visits](/numpy/images/website_visits.jpg)

---

## 103. I/O with NumPy

*Source: [https://www.tutorialspoint.com/numpy/numpy_with_io.htm](https://www.tutorialspoint.com/numpy/numpy_with_io.htm)*

---

---
[Previous](/numpy/numpy_zipf_distribution.htm)[Quiz](/numpy/quiz_on_numpy_with_io.htm)[Next](/numpy/numpy_reading_data_from_files.htm)
## I/O with NumPy

I/O in NumPy refers to input/output operations, allowing you to save and load arrays to and from files. Functions like np.save() and np.load() handle binary files, while np.savetxt() and np.loadtxt() work with text files.

## Reading and Writing Text Files

Text files are one of the most common ways to store data. NumPy provides functions to read from and write to text files. The primary functions for these operations are
**numpy.loadtxt()**and**numpy.savetxt()**.
### The numpy.loadtxt() Function

The
**numpy.loadtxt()**function loads data from a text file into a NumPy array. It can handle different data types and allows customization of delimiters and skipping of header lines.**Example: Reading a Text File**
In the following example, we are loading data from a text file in NumPy using the np.loadtxt() function −

```
import numpy as np

# Create a sample text file
with open('data.txt', 'w') as f:
   f.write("1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0")

# Load the data from the text file
data = np.loadtxt('data.txt')
print("Loaded data from text file:")
print(data)
```

Following is the output obtained −

```
Loaded data from text file:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

### The numpy.savetxt() Function

The
**numpy.savetxt()**function saves a NumPy array to a text file. It allows customization of the delimiter, format, and header information for the output file.**Example: Writing to a Text File**
In this example, we are saving a NumPy array to a text file in using the np.savetxt() function −

```
import numpy as np

# Create an array
data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Save the array to a text file
np.savetxt('output.txt', data, delimiter=' ', fmt='%1.1f')
print("Data saved to text file 'output.txt'")
```

The data is saved to the file 'output.txt' with space as the delimiter and one decimal point format −

```
Data saved to text file 'output.txt'
```

## Reading and Writing Binary Files

Binary files are more efficient for storing large datasets because they are compact and faster to read/write compared to text files. NumPy provides
**numpy.save()**function and**numpy.load()**function for binary I/O operations.
### The numpy.save() Function

The
**numpy.save()**function saves a NumPy array to a binary file in .npy format. This function preserves the array's data, shape, and dtype for efficient storage and retrieval.**Example: Writing to a Binary File**
In the example below, we are saving a NumPy array to a binary file using the np.save() function −

```
import numpy as np

# Create an array
data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Save the array to a binary file
np.save('data.npy', data)
print("Data saved to binary file 'data.npy'")
```

Following is the output of the above code −

```
Data saved to binary file 'data.npy'
```

### The numpy.load() Function

The
**numpy.load()**function loads arrays from binary files saved in .npy or .npz format. It efficiently restores the saved data, including its shape and dtype.**Example: Reading a Binary File**
In this example, we are loading an array from a binary file in NumPy using the np.load() function −

```
import numpy as np

# Load the array from the binary file
data = np.load('data.npy')
print("Loaded data from binary file:")
print(data)
```

The output obtained is as shown below −

```
Loaded data from binary file:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

## Handling CSV Files

Comma-Separated Values (CSV) files are widely used for data storage and exchange. NumPy can read and write CSV files using
**numpy.genfromtxt()**and**numpy.savetxt()**functions.
### The numpy.genfromtxt() Function

The
**numpy.genfromtxt()**function loads data from a text file, handling missing values and non-numeric data. It is more flexible than loadtxt() function, allowing for advanced customization like filling missing values and specifying data types.**Example: Reading a CSV File**
In the following example, we are creating a CSV file with sample data and saving it. Then, we use np.genfromtxt() to load the data from the CSV file into a NumPy array and print it −

```
import numpy as np

# Create a sample CSV file
with open('data.csv', 'w') as f:
   f.write("1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0")

# Load the data from the CSV file
data = np.genfromtxt('data.csv', delimiter=',')
print("Loaded data from CSV file:")
print(data)
```

After executing the above code, we get the following output −

```
Loaded data from CSV file:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```
**Example: Writing to a CSV File**
In this example, we are creating a NumPy array and saving it to a CSV file using np.savetxt(). The data is saved with a delimiter of commas and formatted to one decimal place −

```
import numpy as np

# Create an array
data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Save the array to a CSV file
np.savetxt('output.csv', data, delimiter=',', fmt='%1.1f')
print("Data saved to CSV file 'output.csv'")
```

We get the output as shown below −

```
Data saved to CSV file 'output.csv'
```

## Working with NumPy's NPZ Files

NumPy's NPZ format allows you to save multiple arrays in a single file, making it convenient to store related data together. You can
**numpy.savez()**and**numpy.load()**functions for these operations.
### The numpy.savez() Function

The
**numpy.savez()**function saves multiple NumPy arrays into a single compressed .npz file. It allows you to store multiple arrays with custom names and retrieve them later efficiently.**Example: Writing to an NPZ File**
In the example below, we are creating two NumPy arrays and saving them to a compressed .npz file using the np.savez() function. The arrays are stored with custom names, array1 and array2, in the file −

```
import numpy as np

# Create arrays
array1 = np.array([1, 2, 3])
array2 = np.array([[4, 5, 6], [7, 8, 9]])

# Save arrays to an NPZ file
np.savez('data.npz', array1=array1, array2=array2)
print("Arrays saved to NPZ file 'data.npz'")
```

Following is the output obtained −

```
Arrays saved to NPZ file 'data.npz'
```
**Example: Reading from an NPZ File**
In this example, we are loading the arrays stored in the .npz file using np.load() function. We access and print the arrays "array1" and "array2" by referencing their names from the loaded file −

```
import numpy as np

# Load the arrays from the NPZ file
data = np.load('data.npz')
print("Loaded array1 from NPZ file:")
print(data['array1'])
print("Loaded array2 from NPZ file:")
print(data['array2'])
```

The result produced is as follows −

```
Loaded array1 from NPZ file:
[1 2 3]
Loaded array2 from NPZ file:
[[4 5 6]
 [7 8 9]]
```

## Handling Large Datasets with Memory Mapping

Memory mapping allows you to work with large datasets that do not fit into memory by mapping a file to memory. NumPy provides the
**numpy.memmap()**function for this purpose.
### Example: Using Memory Mapping

In the following example, we are creating a large NumPy array and saving it to a binary file using np.save()function. We then use np.memmap() function to memory-map the file, allowing us to access large data without loading it entirely into memory, and print the first 10 elements −

```
import numpy as np

# Create a large array and save it to a binary file
data = np.arange(1e7)
np.save('large_data.npy', data)

# Memory map the binary file
mmapped_data = np.memmap('large_data.npy', dtype='float64', mode='r', shape=(int(1e7),))
print("Memory-mapped data:")
print(mmapped_data[:10])
```

The output of the above code is shown below −

```
Memory-mapped data:
[1.87585069e-309 1.17119999e+171 5.22741680e-037 8.44740097e+252
 2.65141232e+180 9.92152605e+247 2.16209968e+233 1.39837001e-076
 5.89250072e-096 6.01347002e-154]
```

---

## 104. NumPy - Reading Data from Files

*Source: [https://www.tutorialspoint.com/numpy/numpy_reading_data_from_files.htm](https://www.tutorialspoint.com/numpy/numpy_reading_data_from_files.htm)*

---

---
[Previous](/numpy/numpy_with_io.htm)[Quiz](/numpy/quiz_on_numpy_reading_data_from_files.htm)[Next](/numpy/numpy_writing_data_to_files.htm)
## File Reading in NumPy

Reading data from files involves opening a file and extracting its contents for further use. In Python, libraries like NumPy and Pandas provide functions to load data from various file formats, such as text, CSV, and binary. This allows easy access to stored information for analysis or processing.

In Python, files can be of various types, including text files, CSV files, and binary files. NumPy makes it easy to load data from these files into arrays, which can then be used for analysis or processing.

NumPy offers several functions to read data from files, enabling us to load the data into NumPy arrays for further processing and analysis. The primary functions we will cover are −

### NumPy Functions for Reading Data

Following are the functions used in NumPy to read data from a file −

- **numpy.loadtxt():**Reads data from text files where the values are separated by spaces, commas, or other delimiters.
- **numpy.genfromtxt():**Similar to**loadtxt()**function but more flexible, allowing you to handle missing values and different data types.
- **numpy.load():**Reads binary data from .npy or .npz files.
- **numpy.memmap():**Efficiently maps large binary files to memory without loading the entire file into memory.
## Reading Data from Text Files

Text files are simple and widely used for storing data. These files may contain numerical data separated by spaces, tabs, or commas. Let us explore how to read data from text files using NumPy.

### Reading Simple Text Files with loadtxt() Function

The
**numpy.loadtxt()**function is used to read simple, well-structured text files. By default, it assumes that the data in the file is numeric, and it can automatically split values by whitespace or a custom delimiter.**Example: Reading Data from a Text File**
Here, we have created a file with three rows of numbers. The
**numpy.loadtxt()**function reads the file and returns a 2D array, where each row corresponds to a line in the text file −
```
import numpy as np

# Create a sample text file
with open('data.txt', 'w') as f:
    f.write("1 2 3\n4 5 6\n7 8 9\n")

# Read the data from the text file
data = np.loadtxt('data.txt')

print("Loaded data from text file:")
print(data)
```

Following is the output obtained −

```
Loaded data from text file:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

### Custom Delimiters with loadtxt() Function

You can also specify a custom delimiter if your data is separated by commas, tabs, or other characters using the numpy.loadtxt() function.
**Example**
In this example, the file uses commas as separators, and we specify the
**','**delimiter in the**loadtxt()**function −
```
import numpy as np

# Create a CSV-like text file
with open('data.csv', 'w') as f:
   f.write("1,2,3\n4,5,6\n7,8,9\n")

# Load data with comma as delimiter
data = np.loadtxt('data.csv', delimiter=',')

print("Loaded data from CSV file:")
print(data)
```

This will produce the following result −

```
Loaded data from CSV file:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

### Handling Missing Data with genfromtxt() Function

Sometimes, datasets contain missing or incomplete values. The
**numpy.genfromtxt()**function is more flexible than**loadtxt()**function and can handle missing data or more complex file structures.**Example: Reading Data with Missing Values**
Here, the missing value in the second row is replaced with
**nan**(Not a Number). This is useful when working with real-world datasets where missing data is common −
```
import numpy as np

# Create a text file with missing values
with open('data_with_missing.csv', 'w') as f:
   f.write("1,2,3\n4,,6\n7,8,9\n")

# Load data, specifying the missing value
data = np.genfromtxt('data_with_missing.csv', delimiter=',', filling_values=np.nan)

print("Loaded data with missing values:")
print(data)
```

Following is the output of the above code −

```
Loaded data with missing values:
[[ 1.  2.  3.]
 [ 4. nan  6.]
 [ 7.  8.  9.]]
```

## Reading Data from Binary Files

Binary files are often used to store data because they are more efficient in terms of space and speed. NumPy supports reading and writing binary files using the
**numpy.load()**and**numpy.save()**functions. These functions are optimized for storing NumPy arrays in a binary format with .npy extension.**Example**
In this example, the
**numpy.save()**function writes the array to a binary .npy file, and the**numpy.load()**function loads it back. This format is compact and preserves the array's data types and structure −
```
import numpy as np

# Create a sample array
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Save the array to a binary file
np.save('data.npy', data)

# Load the data from the binary file
loaded_data = np.load('data.npy')

print("Loaded data from binary file:")
print(loaded_data)
```

The output obtained is as shown below −

```
Loaded data from binary file:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

## Memory-Mapped Files with memmap() Function

For working with large datasets that don't fit into memory, NumPy provides memory-mapped arrays using the
**numpy.memmap()**function. This function allows you to read and write large binary files without loading the entire file into memory.
### Example: Using Memory-Mapped Files

Memory mapping is ideal for large datasets as it allows you to access parts of the file directly without loading the entire file into memory −

```
import numpy as np

# Create a large binary file
data = np.arange(1e7)
np.save('large_data.npy', data)

# Memory-map the binary file
mmapped_data = np.memmap('large_data.npy', dtype='float64', mode='r', shape=(int(1e7),))

# Access a slice of the data
print("First 10 elements of the memory-mapped data:")
print(mmapped_data[:10])
```

After executing the above code, we get the following output −

```
First 10 elements of the memory-mapped data:
[1.87585069e-309 1.17119999e+171 5.22741680e-037 8.44740097e+252
 2.65141232e+180 9.92152605e+247 2.16209968e+233 1.39837001e-076
 5.89250072e-096 6.01347002e-154]
```

## Working with CSV Files

CSV (Comma-Separated Values) files are commonly used for storing tabular data. NumPy provides functions to read from and write to CSV files. The
**numpy.genfromtxt()**function can handle CSV files, and**numpy.savetxt()**function can be used to write data to CSV.
### Example: Writing Data to CSV

In the following example, we are creating a 2D NumPy array and writing it to a CSV file using np.savetxt() function. The data is saved with a comma delimiter and formatted as integers −

```
import numpy as np

# Create a 2D array
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Write data to a CSV file
np.savetxt('output.csv', data, delimiter=',', fmt='%d')

print("Data written to 'output.csv'.")
```

The result produced is as follows −

```
Data written to 'output.csv'.
```

---

## 105. NumPy - Writing Data to Files

*Source: [https://www.tutorialspoint.com/numpy/numpy_writing_data_to_files.htm](https://www.tutorialspoint.com/numpy/numpy_writing_data_to_files.htm)*

---

---

## 106. NumPy - File Formats Supported

*Source: [https://www.tutorialspoint.com/numpy/numpy_supported_file_formats.htm](https://www.tutorialspoint.com/numpy/numpy_supported_file_formats.htm)*

---

---
[Previous](/numpy/numpy_writing_data_to_files.htm)[Quiz](/numpy/quiz_on_numpy_supported_file_formats.htm)[Next](/numpy/numpy_mathematical_functions.htm)
## File Formats in NumPy

NumPy provides support for saving and loading arrays in various file formats extensively. These formats allow you to store data in a manner that is easy to share, read, and process.

When working with large datasets, it is important to choose the appropriate file format for storing your NumPy arrays. The most common formats are:

- **.npy:**Used for saving a single array with its metadata.
- **.npz:**A compressed archive format used for saving multiple arrays in a single file.
- **Text Files:**For human-readable data storage, including CSV and custom text formats.
## The .npy Format

The
**.npy**format is the native binary format used by NumPy to store arrays. This format preserves the array's data type and shape information.
### Using numpy.save() Function

The
**numpy.save()**function saves a single NumPy array in the**.npy**format. This format is highly efficient for storing large arrays as it keeps all metadata intact.**Example**
In this example, we will create a NumPy array and save it to a
**.npy**file −
```
import numpy as np

# Create a NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save the array to a .npy file
np.save('array_data.npy', arr)

# Load the saved array to verify
loaded_array = np.load('array_data.npy')
print("Loaded Array:\n", loaded_array)
```

After executing the above code, the output will be −

```
Loaded Array:
 [[1 2 3]
 [4 5 6]]
```

## The .npz Format

The
**.npz**format is a compressed archive used to store multiple arrays in one file. It is helpful when working with several arrays, as it allows you to save them together in a single, compressed file.
### Using numpy.savez() Function

The
**numpy.savez**function saves multiple arrays in a single**.npz**file. You can also use**numpy.savez_compressed()**function to compress the data within the archive.**Example**
In the following example, we will save two arrays into a single
**.npz**file and load them back −
```
import numpy as np

# Create two arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([[4, 5, 6], [7, 8, 9]])

# Save arrays into a .npz file
np.savez('arrays_data.npz', array1=arr1, array2=arr2)

# Load the arrays from the .npz file
loaded_data = np.load('arrays_data.npz')
print("Loaded array1:\n", loaded_data['array1'])
print("Loaded array2:\n", loaded_data['array2'])
```

The output will be −

```
Loaded array1:
 [1 2 3]
Loaded array2:
 [[4 5 6]
 [7 8 9]]
```

## Text File Formats

Text file formats such as CSV (Comma Separated Values) are popular for storing tabular data. NumPy provides methods for saving and loading arrays to and from text files.

### Using numpy.savetxt() Function

The
**numpy.savetxt()**function is used to save arrays to text files. It allows you to specify a delimiter, format, and header.**Example**
In this example, we will save a NumPy array to a text file using the
**numpy.savetxt()**function −
```
import numpy as np

# Create a NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save the array to a text file
np.savetxt('array_data.txt', arr)

# Load the array from the text file to verify
loaded_array = np.loadtxt('array_data.txt')
print("Loaded Array:\n", loaded_array)
```

After executing the code, the output will be −

```
Loaded Array:
 [[1. 2. 3.]
 [4. 5. 6.]]
```

### Using numpy.genfromtxt() Function

The
**numpy.genfromtxt()**function is used for reading CSV files and converting them into NumPy arrays.**Example**
In the example below, we will read a CSV file into a NumPy array using the
**numpy.genfromtxt()**function −
```
import numpy as np

# Read data from a CSV file
data_from_csv = np.genfromtxt('array_data.txt')

# Print the loaded data
print("Data loaded from CSV:\n", data_from_csv)
```

The output obtained is as follows −

```
Data loaded from CSV:
 [[1. 2. 3.]
 [4. 5. 6.]]
```

## Custom Formats

In some situations, you may need to write or read data in a custom binary format. NumPy provides the
**numpy.ndarray.tofile()**and**numpy.fromfile()**functions for handling custom binary data formats.
### Using numpy.ndarray.tofile() Function

The
**numpy.ndarray.tofile()**function is used to write a NumPy array to a binary file in a raw format. You can specify the file and format of the data.**Example**
In the following example, we will write a NumPy array to a custom binary file using the
**tofile()**function −
```
import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5], dtype='int32')

# Write the array to a binary file
arr.tofile('binary_data.dat')
print("Array written to binary file:", arr)
```

The result produced is as shown below −

```
Array written to binary file: [1 2 3 4 5]
```

### Using numpy.fromfile() Function

The
**numpy.fromfile()**function is used to read data from a custom binary file into a NumPy array.**Example**
In this example, we will read the custom binary file and load it into a NumPy array using the
**fromfile()**function −
```
import numpy as np

# Read the binary data from the file
data_from_binary = np.fromfile('binary_data.dat', dtype='int32')

# Print the data loaded from the binary file
print("Array read from binary file:", data_from_binary)
```

We get the output as shown below −

```
Array read from binary file: [1 2 3 4 5]
```

---

## 107. NumPy - Mathematical Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_mathematical_functions.htm](https://www.tutorialspoint.com/numpy/numpy_mathematical_functions.htm)*

---

---
[Previous](/numpy/numpy_supported_file_formats.htm)[Quiz](/numpy/quiz_on_numpy_mathematical_functions.htm)[Next](/numpy/numpy_trigonometric_functions.htm)
## NumPy Mathematical Functions

NumPy provides a wide range of mathematical functions that are essential for performing numerical operations on arrays. These functions include basic arithmetic, trigonometric, exponential, logarithmic, and statistical operations, among others.

In this tutorial, we will explore the most commonly used mathematical functions in NumPy, with examples to help you understand their application.

## Basic Arithmetic Operations

In NumPy, basic arithmetic operations include addition, subtraction, multiplication, and division on arrays. These operations are element-wise, meaning they are applied to each corresponding element in the arrays.

For example, adding two arrays results in a new array where each element is the sum of the corresponding elements from the input arrays.

NumPy also supports scalar operations, allowing you to apply a number to each element of an array directly.

### Example: Addition, Subtraction, Multiplication, and Division

In the following example, we perform basic arithmetic operations like addition, subtraction, multiplication, and division on two NumPy arrays −

```
import numpy as np

# Define two arrays
a = np.array([10, 20, 30])
b = np.array([5, 10, 15])

# Perform basic arithmetic operations
addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b

print("Addition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)
```

We get the output as shown below −

```
Addition: [15 30 45]
Subtraction: [ 5 10 15]
Multiplication: [ 50 200 450]
Division: [2. 2. 2.]
```

## Trigonometric Functions

NumPy also provides several functions to perform trigonometric operations on arrays. These include basic trigonometric functions like sine, cosine, and tangent, which operate element-wise on arrays.

### Example: Sine, Cosine, and Tangent

Let us explore how to calculate the sine, cosine, and tangent of an array in NumPy −

```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/4, np.pi/2, np.pi])

# Calculate sine, cosine, and tangent
sine_values = np.sin(angles)
cosine_values = np.cos(angles)
tangent_values = np.tan(angles)

print("Sine values:", sine_values)
print("Cosine values:", cosine_values)
print("Tangent values:", tangent_values)
```

Following is the output obtained −

```
Sine values: [0.00000000e+00 7.07106781e-01 1.00000000e+00 1.22464680e-16]
Cosine values: [ 1.00000000e+00  7.07106781e-01  6.12323400e-17 -1.00000000e+00]
Tangent values: [ 0.00000000e+00  1.00000000e+00  1.63312394e+16 -1.22464680e-16]
```

> Note that the tangent of pi/2 is infinity because the cosine of pi/2 is zero, and division by zero is undefined.
**pi/2**is infinity because the cosine of**pi/2**is zero, and division by zero is undefined.
## Exponential and Logarithmic Functions

NumPy also provides functions to calculate exponential values and logarithms, which are used in various scientific and engineering calculations. These functions allow you to compute powers, roots, and logarithms of arrays.

### Exponentiation

The
**numpy.exp()**function calculates the exponential of all elements in the input array. Following is an example −
```
import numpy as np

# Create an array
arr = np.array([1, 2, 3])

# Calculate the exponential of the array
exp_array = np.exp(arr)
print("Exponential values:", exp_array)
```

This will produce the following result −

```
Exponential values: [ 2.71828183  7.3890561  20.08553692]
```

### Logarithm

NumPy provides the
**numpy.log()**function for calculating the natural logarithm (base e), and**numpy.log10()**function for calculating the logarithm to base 10.Following is an example −
```
import numpy as np

# Create an array
arr = np.array([1, 2, 10])

# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(arr)
log10_array = np.log10(arr)

print("Natural logarithm values:", log_array)
print("Base-10 logarithm values:", log10_array)
```

Following is the output of the above code −

```
Natural logarithm values: [0.         0.69314718 2.30258509]
Base-10 logarithm values: [0.      0.30103 1.     ]
```

## Statistical Functions in NumPy

NumPy also provides a wide variety of statistical functions. These include calculating mean, median, variance, standard deviation, and more. These functions are useful when analyzing data and performing statistical analysis on arrays.

### Mean and Median

The
**numpy.mean()**function calculates the arithmetic mean, and the**numpy.median()**function calculates the median of an array as shown in the example below −
```
import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4, 5])

# Calculate the mean and median of the array
mean_value = np.mean(arr)
median_value = np.median(arr)

print("Mean value:", mean_value)
print("Median value:", median_value)
```

The output obtained is as shown below −

```
Mean value: 3.0
Median value: 3.0
```

### Standard Deviation and Variance

The
**numpy.std()**function calculates the standard deviation, and**numpy.var()**function  calculates the variance of the array −
```
import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4, 5])

# Calculate the standard deviation and variance
std_dev = np.std(arr)
variance = np.var(arr)

print("Standard Deviation:", std_dev)
print("Variance:", variance)
```

After executing the above code, we get the following output −

```
Standard Deviation: 1.4142135623730951
Variance: 2.0
```

## Linear Algebra Functions

NumPy also provides a set of linear algebra functions, such as matrix multiplication, dot products, and matrix determinants, which are important for operations on vectors and matrices.

### Dot Product

The
**numpy.dot()**function is used to calculate the dot product of two arrays. It is commonly used in machine learning, physics, and engineering −
```
import numpy as np

# Create two arrays
arr1 = np.array([1, 2])
arr2 = np.array([3, 4])

# Calculate the dot product
dot_product = np.dot(arr1, arr2)
print("Dot product:", dot_product)
```

The result produced is as follows −

```
Dot product: 11
```

### Matrix Multiplication

Matrix multiplication in NumPy is performed using the
**@**operator or the**numpy.matmul()**function. It calculates the dot product of two arrays, where the number of columns in the first matrix must equal the number of rows in the second matrix.
The result is a new matrix where each element is the sum of the products of corresponding row and column elements. Matrix multiplication is commonly used in linear algebra and machine learning tasks.
**Example**
In this example, matrix A is multiplied by matrix B to produce matrix C, where each element is calculated by the dot product of rows from A and columns from B −

```
import numpy as np

# Define two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Perform matrix multiplication
C = np.matmul(A, B)

# Print the result
print(C)
```

We get the output as shown below −

```
[[19 22]
 [43 50]]
```

---

## 108. NumPy - Trigonometric Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_trigonometric_functions.htm](https://www.tutorialspoint.com/numpy/numpy_trigonometric_functions.htm)*

---

---
[Previous](/numpy/numpy_mathematical_functions.htm)[Quiz](/numpy/quiz_on_numpy_trigonometric_functions.htm)[Next](/numpy/numpy_exponential_functions.htm)
## NumPy Trigonometric Functions

In mathematics, trigonometric functions are used to relate the angles of a triangle to the lengths of its sides. The most common trigonometric functions are sine (sin), cosine (cos), and tangent (tan), which are based on a right triangle.

These functions help in various fields such as geometry, physics, and engineering, especially for calculating angles and distances.

NumPy provides several functions like sin(), cos(), and tan() to compute these values for arrays of angles.

## Sine, Cosine, and Tangent Functions

The basic trigonometric functions in NumPy include sine, cosine, and tangent. These functions operate element-wise on arrays, meaning they are applied to each individual element of the array. Trigonometric functions are typically applied to angles, and the angle is usually provided in radians.

For example, the sine of an angle is the ratio of the opposite side to the hypotenuse in a right triangle. The cosine is the ratio of the adjacent side to the hypotenuse, and the tangent is the ratio of the opposite side to the adjacent side.

### Example: Sine, Cosine, and Tangent

In the following example, we calculate the sine, cosine, and tangent of an array of angles in radians using NumPy functions −

```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/4, np.pi/2, np.pi])

# Calculate sine, cosine, and tangent
sine_values = np.sin(angles)
cosine_values = np.cos(angles)
tangent_values = np.tan(angles)

print("Sine values:", sine_values)
print("Cosine values:", cosine_values)
print("Tangent values:", tangent_values)
```

Following is the output obtained −

```
Sine values: [0.00000000e+00 7.07106781e-01 1.00000000e+00 1.22464680e-16]
Cosine values: [ 1.00000000e+00  7.07106781e-01  6.12323400e-17 -1.00000000e+00]
Tangent values: [ 0.00000000e+00  1.00000000e+00  1.63312394e+16 -1.22464680e-16]
```

> Note that the tangent of pi/2 is infinity because the cosine of pi/2 is zero, and division by zero is undefined.
**pi/2**is infinity because the cosine of**pi/2**is zero, and division by zero is undefined.
## Inverse Trigonometric Functions

In addition to basic trigonometric functions, NumPy also provides inverse trigonometric functions that can be used to calculate the angles given the values of the trigonometric ratios. These include the inverse sine (arcsine), inverse cosine (arccosine), and inverse tangent (arctangent) functions.

Inverse trigonometric functions are useful when you have the value of a trigonometric ratio and need to find the corresponding angle.

### Example: Inverse Sine, Inverse Cosine, and Inverse Tangent

In the following example, we calculate the inverse sine, inverse cosine, and inverse tangent of a set of values using NumPy functions −

```
import numpy as np

# Define an array of values for which we want to calculate inverse trigonometric functions
values = np.array([0, 0.5, 1])

# Calculate inverse sine, inverse cosine, and inverse tangent
arcsine_values = np.arcsin(values)
arccosine_values = np.arccos(values)
arctangent_values = np.arctan(values)

print("Inverse Sine values:", arcsine_values)
print("Inverse Cosine values:", arccosine_values)
print("Inverse Tangent values:", arctangent_values)
```

We get the output as shown below −

```
Inverse Sine values: [0.         0.52359878 1.57079633]
Inverse Cosine values: [1.57079633 1.04719755 0.        ]
Inverse Tangent values: [0.         0.46364761 0.78539816]
```

## Hyperbolic Trigonometric Functions

NumPy also includes functions to calculate hyperbolic trigonometric functions, which are analogs of the standard trigonometric functions but for a hyperbola instead of a circle. These functions include hyperbolic sine, cosine, and tangent.

The hyperbolic sine and cosine functions are similar to the regular sine and cosine functions but use the exponential function for their calculation. The hyperbolic tangent is the ratio of the hyperbolic sine to the hyperbolic cosine.

### Example: Hyperbolic Sine, Cosine, and Tangent

In the following example, we calculate the hyperbolic sine, hyperbolic cosine, and hyperbolic tangent of an array of values using NumPy functions −

```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2])

# Calculate hyperbolic sine, cosine, and tangent
sinh_values = np.sinh(values)
cosh_values = np.cosh(values)
tanh_values = np.tanh(values)

print("Hyperbolic Sine values:", sinh_values)
print("Hyperbolic Cosine values:", cosh_values)
print("Hyperbolic Tangent values:", tanh_values)
```

After executing the above code, we get the following output −

```
Hyperbolic Sine values: [0.         1.17520119 3.62686041]
Hyperbolic Cosine values: [1.         1.54308063 3.76219569]
Hyperbolic Tangent values: [0.         0.76159416 0.96402758]
```

## Trigonometric Functions with Degrees

By default, the input to trigonometric functions in NumPy is in radians. However, you may sometimes want to work with degrees instead of radians. To convert between degrees and radians, you can use the
**numpy.deg2rad()**function and**numpy.rad2deg()**function.
The
**numpy.deg2rad()**function converts an angle from degrees to radians, while the**numpy.rad2deg()**function converts an angle from radians to degrees. You can then use these functions to perform trigonometric calculations with angles in degrees.
### Example: Using Trigonometric Functions with Degrees

In this example, we first convert degrees to radians and then calculate the sine and cosine of the resulting angles −

```
import numpy as np

# Define an array of angles in degrees
angles_deg = np.array([0, 45, 90, 180])

# Convert degrees to radians
angles_rad = np.deg2rad(angles_deg)

# Calculate sine and cosine
sine_values = np.sin(angles_rad)
cosine_values = np.cos(angles_rad)

print("Sine values:", sine_values)
print("Cosine values:", cosine_values)
```

We get the following output as shown below −

```
Sine values: [0.00000000e+00 7.07106781e-01 1.00000000e+00 1.22464680e-16]
Cosine values: [ 1.00000000e+00  7.07106781e-01  6.12323400e-17 -1.00000000e+00]
```

## Applications of Trigonometric Functions

Trigonometric functions are widely used in various fields, especially in science, engineering, and computer graphics. They help model periodic behavior, oscillations, waveforms, and rotations. Some examples of applications are −

- **Signal Processing:**Trigonometric functions are used to model and analyze sound waves, electromagnetic waves, and other periodic signals.
- **Physics:**In physics, trigonometric functions are used to describe the behavior of waves, circular motion, and harmonic oscillators.
- **Computer Graphics:**Trigonometric functions are used in 2D and 3D transformations, rotations, and rendering.
## Functions List

NumPy has a standard trigonometric functions that return trigonometric ratios for angles given in radians. Following is the list of them −
Sr.No.Operation & Description1[numpy.sin()](/numpy/numpy_sin_function.htm)
Trigonometric sine, element-wise.
2[numpy.cos()](/numpy/numpy_cos_function.htm)
Cosine element wise.
3[numpy.tan()](/numpy/numpy_tan_function.htm)
Compute tangent element-wise.
4[numpy.arcsin()](/numpy/numpy_arcsin_function.htm)
Inverse sine, element-wise.
5[numpy.arccos()](/numpy/numpy_arccos_function.htm)
Trigonometric inverse cosine, element-wise.
6[numpy.arctan()](/numpy/numpy_arctan_function.htm)
Trigonometric inverse tangent, element-wise.
7[numpy.arctan2()](/numpy/numpy_arctan2_function.htm)
Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
8[numpy.hypot()](/numpy/numpy_hypot_function.htm)
Equivalent to element-wise sqrt(x12 + x22), broadcasting scalars.
9[numpy.sinh()](/numpy/numpy_sinh_function.htm)
Hyperbolic sine, element-wise.
10[numpy.cosh()](/numpy/numpy_cosh_function.htm)
Hyperbolic cosine, element-wise.
11[numpy.tanh()](/numpy/numpy_tanh_function.htm)
Compute hyperbolic tangent element-wise.
12[numpy.arcsinh()](/numpy/numpy_arcsinh_function.htm)
Inverse hyperbolic sine element-wise.
13[numpy.arccosh()](/numpy/numpy_arccosh_function.htm)
Inverse hyperbolic cosine, element-wise.
14[numpy.arctanh()](/numpy/numpy_arctanh_function.htm)
Inverse hyperbolic tangent element-wise.
15[numpy.deg2rad()](/numpy/numpy_deg2rad_function.htm)
Convert angles from degrees to radians.
16[numpy.rad2deg()](/numpy/numpy_rad2deg_function.htm)
Convert angles from radians to degrees.

---

## 109. NumPy - Exponential Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_exponential_functions.htm](https://www.tutorialspoint.com/numpy/numpy_exponential_functions.htm)*

---

---
[Previous](/numpy/numpy_trigonometric_functions.htm)[Quiz](/numpy/quiz_on_numpy_exponential_functions.htm)[Next](/numpy/numpy_logarithmic_functions.htm)
## NumPy Exponential Functions

In NumPy, exponential functions are provided to calculate powers of Euler's number (e), and to perform operations involving exponential growth or decay. NumPy provides the
**numpy.exp()**function to calculate exponentials.
In this tutorial, we will explore how to use NumPy's exponential functions to calculate powers of e, and perform other related operations.

## The numpy.exp() Function

The
**numpy.exp()**function calculates the exponential of all elements in the input array. The function calculates the value of**e**, where**e**is Euler's number (approximately 2.71828), and**x**is the exponent.
It is commonly used in applications that involve continuous growth or decay, such as calculating compound interest or solving differential equations.

### Example: Exponential Function

In the following example, we calculate the exponential of an array of values using the
**exp()**function −
```
import numpy as np

# Define an array of exponents
exponents = np.array([0, 1, 2, 3])

# Calculate the exponential of each element
exp_values = np.exp(exponents)

print("Exponential values:", exp_values)
```

The output will be −

```
Exponential values: [ 1.          2.71828183  7.3890561  20.08553692]
```

> Note that the exponential of 0 is always 1, and as the exponent increases, the value grows exponentially.

## Natural Logarithm Using log() Function

The
**numpy.log()**function calculates the natural logarithm (base e) of all elements in the input array. It is the inverse of the**numpy.exp()**function. In mathematics, the natural logarithm of a number is the exponent to which e must be raised to obtain that number.
The
**numpy.log()**function is useful for solving equations that involve exponential growth or decay.
### Example: Natural Logarithm

In the following example, we calculate the natural logarithm of an array of values using NumPy's
**log()**function −
```
import numpy as np

# Define an array of values
values = np.array([1, np.e, np.e**2, np.e**3])

# Calculate the natural logarithm of each element
log_values = np.log(values)

print("Natural Logarithm values:", log_values)
```

As expected, the natural logarithm of
**e**is simply**x**. We get the following output −
```
Natural Logarithm values: [0. 1. 2. 3.]
```

## Base-10 Logarithm Using log10() Function

In addition to the natural logarithm, NumPy provides the
**numpy.log10()**function to compute the base-10 logarithm of each element in the input array.
This function is commonly used in scientific fields that use logarithms with a base of 10, such as in sound intensity or earthquake magnitude calculations.

### Example: Base-10 Logarithm

In the following example, we calculate the base-10 logarithm of an array of values using NumPy's
**log10()**function −
```
import numpy as np

# Define an array of values
values = np.array([1, 10, 100, 1000])

# Calculate the base-10 logarithm of each element
log10_values = np.log10(values)

print("Base-10 Logarithm values:", log10_values)
```

As expected, the base-10 logarithm of powers of 10 follows the rule
**log**. We get the following output −(10) = x
```
Base-10 Logarithm values: [0. 1. 2. 3.]
```

## Exponential Function with Base-2

In some cases, it is useful to calculate logarithms with base-2, especially in areas like computer science and information theory. NumPy provides the
**numpy.log2()**function for this purpose.
### Example: Base-2 Logarithm

In the following example, we calculate the base-2 logarithm of an array of values using NumPy's
**log2()**function −
```
import numpy as np

# Define an array of values
values = np.array([1, 2, 4, 8])

# Calculate the base-2 logarithm of each element
log2_values = np.log2(values)

print("Base-2 Logarithm values:", log2_values)
```

As expected, the base-2 logarithm of powers of 2 follows the rule
**log**. The output is −(2) = x
```
Base-2 Logarithm values: [0. 1. 2. 3.]
```

## Exponential Growth and Decay

Exponential functions can model both growth and decay. In growth, the quantity increases over time, such as the growth of a population or investment. In decay, the quantity decreases over time, such as the decay of a radioactive substance.

Exponential growth is typically modeled by the function
**y = y**, where* e**y**is the initial value,**k**is the growth rate, and**t**is time. In decay, the function is**y = y**.* e
### Example: Exponential Growth

In the following example, we simulate exponential growth by using NumPy's
**exp()**function. We assume an initial population size of 10, a growth rate of 0.1, and a time period of 10 units −
```
import numpy as np
import matplotlib.pyplot as plt

# Define initial parameters
# initial population size
y0 = 10  
# growth rate
k = 0.1  
# time array
t = np.linspace(0, 10, 100)  

# Calculate population size at each time point
population = y0 * np.exp(k * t)

# Plot the result
plt.plot(t, population)
plt.title("Exponential Growth")
plt.xlabel("Time")
plt.ylabel("Population Size")
plt.grid(True)
plt.show()
```

This code generates a plot showing the exponential growth of the population over time −
![Exponential Function](/numpy/images/exponential_function.jpg)

---

## 110. NumPy - Logarithmic Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_logarithmic_functions.htm](https://www.tutorialspoint.com/numpy/numpy_logarithmic_functions.htm)*

---

---
[Previous](/numpy/numpy_exponential_functions.htm)[Quiz](/numpy/quiz_on_numpy_logarithmic_functions.htm)[Next](/numpy/numpy_hyperbolic_functions.htm)
## NumPy Logarithmic Functions

logarithmic functions are the inverse of exponential functions. They are used to determine the power to which a number (called the base) must be raised to produce a given value. The most common logarithmic functions are the natural logarithm (ln), with base 10.

In NumPy, several logarithmic functions are available for various types of logarithms. These functions help calculate natural, base-10, and base-2 logarithms, which are useful in solving mathematical equations involving logarithmic relationships.

## Natural Logarithm Using log() Function

The
**numpy.log()**function calculates the natural logarithm (base e) of all elements in the input array. The natural logarithm is commonly used in calculus and other mathematical computations involving continuous growth or decay.
In mathematics, the natural logarithm of a number is the exponent to which the base
**e**(Euler's number, approximately 2.71828) must be raised to obtain that number.
### Example: Natural Logarithm

In the following example, we calculate the natural logarithm of an array of values using NumPy's
**log()**function −
```
import numpy as np

# Define an array of values
values = np.array([1, np.e, np.e**2, np.e**3])

# Calculate the natural logarithm of each element
log_values = np.log(values)

print("Natural Logarithm values:", log_values)
```

As expected, the natural logarithm of
**e**is simply**x**. The output will be −
```
Natural Logarithm values: [0. 1. 2. 3.]
```

## Base-10 Logarithm Using log10() Function

The
**numpy.log10()**function calculates the base-10 logarithm of all elements in the input array.
The base-10 logarithm, also known as the common logarithm, is the inverse of the exponential function with base 10. It is widely used in fields that deal with large or small numbers, like sound intensity or the Richter scale for earthquake magnitudes.

### Example: Base-10 Logarithm

In the following example, we calculate the base-10 logarithm of an array of values using NumPy's
**log10()**function −
```
import numpy as np

# Define an array of values
values = np.array([1, 10, 100, 1000])

# Calculate the base-10 logarithm of each element
log10_values = np.log10(values)

print("Base-10 Logarithm values:", log10_values)
```

As expected, the base-10 logarithm of powers of 10 follows the rule
**log**. The output is −(10) = x
```
Base-10 Logarithm values: [0. 1. 2. 3.]
```

## Base-2 Logarithm Using log2() Function

The
**numpy.log2()**function calculates the base-2 logarithm of each element in the input array. Base-2 logarithms are used in various fields, including computer science, information theory, and coding theory, as binary systems and algorithms often rely on base-2 operations.
In computer science, the base-2 logarithm of a number is the number of times you can divide the number by 2 until you get 1. This function is often used in applications that involve binary data or computational complexity.

### Example: Base-2 Logarithm

In the following example, we calculate the base-2 logarithm of an array of values using NumPy's
**log2()**function −
```
import numpy as np

# Define an array of values
values = np.array([1, 2, 4, 8])

# Calculate the base-2 logarithm of each element
log2_values = np.log2(values)

print("Base-2 Logarithm values:", log2_values)
```

As expected, the base-2 logarithm of powers of 2 follows the rule
**log**. The output is −(2) = x
```
Base-2 Logarithm values: [0. 1. 2. 3.]
```

## Logarithm with a Custom Base

Although NumPy doesn't provide a direct function for logarithms with arbitrary bases, you can compute the logarithm with a custom base by using the change of base formula −

```
logbase(x) = loge(x) / loge(base)
```
(x) = log(x) / log(base)
This allows you to compute logarithms with any base by dividing the natural logarithm of the number by the natural logarithm of the base.

### Example: Custom Base Logarithm

In the following example, we compute the logarithm of an array of values with a custom base (e.g., base 3) using the change of base formula −

```
import numpy as np

# Define an array of values and the custom base
values = np.array([1, 3, 9, 27])
base = 3

# Calculate the logarithm with the custom base using the change of base formula
log_base3_values = np.log(values) / np.log(base)

print("Logarithm with base 3 values:", log_base3_values)
```

As expected, the logarithm of powers of 3 with base 3 follows the rule
**log**−(3) = x
```
Logarithm with base 3 values: [0. 1. 2. 3.]
```

## Handling Logarithms of Zero or Negative Numbers

Logarithms of zero or negative numbers are undefined in real numbers. If you try to compute the logarithm of 0 or a negative number using NumPy, it will return
**nan**(Not a Number) or**-inf**(negative infinity) depending on the context.
To avoid errors, it is often helpful to use the
**numpy.errstate()**function to handle such cases gracefully, suppressing warnings or handling invalid operations explicitly.
### Example: Handling Invalid Logarithms

In the following example, we attempt to calculate the logarithms of 0 and negative values −

```
import numpy as np

# Define an array with a zero and a negative value
values = np.array([0, -1, 1, 10])

# Calculate the natural logarithm of each element
log_values = np.log(values)

print("Logarithm values:", log_values)
```

The output shows that the logarithm of 0 results in
**nan**, and the logarithm of a negative number results in**-inf**−
```
Logarithm values: [ nan -inf   0.   2.30258509]
```

---

## 111. NumPy - Hyperbolic Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_hyperbolic_functions.htm](https://www.tutorialspoint.com/numpy/numpy_hyperbolic_functions.htm)*

---

---
[Previous](/numpy/numpy_logarithmic_functions.htm)[Quiz](/numpy/quiz_on_numpy_hyperbolic_functions.htm)[Next](/numpy/numpy_rounding_functions.htm)
## NumPy Hyperbolic Functions

The hyperbolic functions are similar to trigonometric functions but are based on hyperbolas instead of circles. The most common hyperbolic functions are sinh (hyperbolic sine), cosh (hyperbolic cosine), and tanh (hyperbolic tangent).

In NumPy, there are several hyperbolic functions available to calculate these values for arrays of numbers. These functions are widely used in scenarios involving hyperbolic curves or exponential growth and decay processes.

## Hyperbolic Sine Using sinh() Function

The
**numpy.sinh()**function calculates the hyperbolic sine of each element in the input array. The hyperbolic sine is defined as −
```
sinh(x) = (ex - e(-x)) / 2
```
- e) / 2
### Example: Hyperbolic Sine

In the following example, we calculate the hyperbolic sine of an array of values using NumPy's
**sinh()**function −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2, 3])

# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(values)

print("Hyperbolic Sine values:", sinh_values)
```

```
Hyperbolic Sine values: [ 0.          1.17520119  3.62686041 10.01787493]
```

## Hyperbolic Cosine Using cosh() Function

The
**numpy.cosh()**function calculates the hyperbolic cosine of each element in the input array. The hyperbolic cosine is defined as −
```
cosh(x) = (ex + e(-x)) / 2
```
+ e) / 2
### Example: Hyperbolic Cosine

In the following example, we calculate the hyperbolic cosine of an array of values using NumPy's
**cosh()**function −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2, 3])

# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(values)

print("Hyperbolic Cosine values:", cosh_values)
```

This will produce the following result −

```
Hyperbolic Cosine values: [ 1.          1.54308063  3.76219569 10.067662  ]
```

## Hyperbolic Tangent Using tanh() Function

The
**numpy.tanh()**function calculates the hyperbolic tangent of each element in the input array. The hyperbolic tangent is defined as −
```
tanh(x) = sinh(x) / cosh(x)
```

### Example: Hyperbolic Tangent

In the following example, we calculate the hyperbolic tangent of an array of values using NumPy's
**tanh()**function −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2, 3])

# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(values)

print("Hyperbolic Tangent values:", tanh_values)
```

Following is the output of the above code −

```
Hyperbolic Tangent values: [0.         0.76159416 0.96402758 0.99505475]
```

## Inverse Hyperbolic Sine

The
**numpy.arcsinh()**function calculates the inverse hyperbolic sine of each element in the input array. The inverse hyperbolic sine is defined as −
```
arcsinh(x) = log(x + sqrt(x2 + 1))
```
+ 1))
### Example: Inverse Hyperbolic Sine

In the following example, we calculate the inverse hyperbolic sine of an array of values using NumPy's
**arcsinh()**function −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2, 3])

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(values)

print("Inverse Hyperbolic Sine values:", asinh_values)
```

The output obtained is as shown below −

```
Inverse Hyperbolic Sine values: [0.         0.88137359 1.44363548 1.81844646]
```

## Inverse Hyperbolic Cosine

The
**numpy.arccosh()**function calculates the inverse hyperbolic cosine of each element in the input array. The inverse hyperbolic cosine is defined as −
```
arccosh(x) = log(x + sqrt(x2 - 1))
```
- 1))
### Example: Inverse Hyperbolic Cosine

In the following example, we calculate the inverse hyperbolic cosine of an array of values using NumPy's
**arccosh()**function −
```
import numpy as np

# Define an array of values
values = np.array([1, 2, 3, 4])

# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(values)

print("Inverse Hyperbolic Cosine values:", acosh_values)
```

After executing the above code, we get the following output −

```
Inverse Hyperbolic Cosine values: [0.         1.3169579  1.76274717 2.06343707]
```

## Inverse Hyperbolic Tangent

The
**numpy.arctanh()**function calculates the inverse hyperbolic tangent of each element in the input array. The inverse hyperbolic tangent is defined as −
```
arctanh(x) = 0.5 * log((1 + x) / (1 - x))
```

### Example: Inverse Hyperbolic Tangent

In the following example, we calculate the inverse hyperbolic tangent of an array of values using NumPy's
**arctanh()**function −
```
import numpy as np

# Define an array of values
values = np.array([0, 0.5, 0.9])

# Calculate the inverse hyperbolic tangent of each element
atanh_values = np.arctanh(values)

print("Inverse Hyperbolic Tangent values:", atanh_values)
```

The result produced is as follows −

```
Inverse Hyperbolic Tangent values: [0.         0.54930614 1.47221949]
```

---

## 112. NumPy - Rounding Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_rounding_functions.htm](https://www.tutorialspoint.com/numpy/numpy_rounding_functions.htm)*

---

---
[Previous](/numpy/numpy_hyperbolic_functions.htm)[Quiz](/numpy/quiz_on_numpy_rounding_functions.htm)[Next](/numpy/numpy_discrete_fourier_transform.htm)
## NumPy Rounding Functions

Rounding functions in NumPy are used to round off the values in arrays to a specified number of decimal places. These functions are helpful in various scenarios, such as when you need to present values in a cleaner format or when performing numerical computations where precision control is necessary.

NumPy provides several rounding functions, including
**round()**,**floor()**,**ceil()**, and**trunc()**, each serving a different purpose when it comes to rounding values.
## Rounding Using round() Function

The
**numpy.round()**function rounds each element in the input array to the specified number of decimal places. If no number of decimals is provided, it rounds to the nearest integer.
```
round(x, decimals) = rounded value to 'decimals' number of places
```

### Example: Rounding Values

In the following example, we round an array of values to 2 decimal places using NumPy's
**round()**function −
```
import numpy as np

# Define an array of values
values = np.array([3.14159, 2.71828, 1.61803, 0.57721])

# Round each element to 2 decimal places
rounded_values = np.round(values, 2)

print("Rounded values:", rounded_values)
```

The result of the above code is −

```
Rounded values: [3.14 2.72 1.62 0.58]
```

## Flooring Using floor() Function

The
**numpy.floor()**function rounds each element in the input array down to the nearest integer less than or equal to the element. This function always rounds down, regardless of the decimal part.
```
floor(x) = largest integer less than or equal to x
```

### Example: Flooring Values

In the following example, we round down an array of values using NumPy's
**floor()**function −
```
import numpy as np

# Define an array of values
values = np.array([3.14159, 2.71828, 1.61803, 0.57721])

# Round down each element to the nearest integer
floored_values = np.floor(values)

print("Floored values:", floored_values)
```

Following is the output obtained −

```
Floored values: [3. 2. 1. 0.]
```

## Ceiling Using ceil() Function

The
**numpy.ceil()**function rounds each element in the input array up to the nearest integer greater than or equal to the element. This function always rounds up, regardless of the decimal part.
```
ceil(x) = smallest integer greater than or equal to x
```

### Example: Ceiling Values

In the following example, we round up an array of values using NumPy's
**ceil()**function −
```
import numpy as np

# Define an array of values
values = np.array([3.14159, 2.71828, 1.61803, 0.57721])

# Round up each element to the nearest integer
ceiled_values = np.ceil(values)

print("Ceiled values:", ceiled_values)
```

This will produce the following result −

```
Ceiled values: [4. 3. 2. 1.]
```

## Truncating Using trunc() Function

The
**numpy.trunc()**function truncates each element in the input array by removing the decimal part. It essentially rounds the value towards zero, keeping the integer part.
```
trunc(x) = integer part of x, removing the decimal part
```

### Example: Truncating Values

In the following example, we truncate an array of values using NumPy's
**trunc()**function −
```
import numpy as np

# Define an array of values
values = np.array([3.14159, 2.71828, 1.61803, 0.57721])

# Truncate each element
truncated_values = np.trunc(values)

print("Truncated values:", truncated_values)
```

Following is the output of the above code −

```
Truncated values: [3. 2. 1. 0.]
```

## Rounding to a Specific Multiple

The
**numpy.around()**function is used to round elements in an array to the nearest specified multiple. This is different from the standard rounding because it allows for a custom rounding base.
```
around(x, decimals, out) = rounds to the nearest multiple of decimals
```

### Example: Rounding to a Multiple

In the following example, we round an array of values to the nearest multiple of 0.5 using NumPy's
**around()**function −
```
import numpy as np

# Define an array of values
values = np.array([3.14159, 2.71828, 1.61803, 0.57721])

# Round each element to the nearest multiple of 0.5
rounded_multiple_values = np.around(values * 2) / 2

print("Rounded to nearest multiple of 0.5:", rounded_multiple_values)
```

The output obtained is as shown below −

```
Rounded to nearest multiple of 0.5: [3.  2.5 1.5 0.5]
```

---

## 113. NumPy - Discrete Fourier Transform

*Source: [https://www.tutorialspoint.com/numpy/numpy_discrete_fourier_transform.htm](https://www.tutorialspoint.com/numpy/numpy_discrete_fourier_transform.htm)*

---

---
[Previous](/numpy/numpy_rounding_functions.htm)[Quiz](/numpy/quiz_on_numpy_discrete_fourier_transform.htm)[Next](/numpy/numpy_fast_fourier_transform.htm)
## NumPy Discrete Fourier Transform

The Discrete Fourier Transform (DFT) is a mathematical technique used to convert a sequence of values into components of different frequencies. It is widely used in signal processing, image analysis, and audio processing.

In NumPy, the DFT can be computed using the
**fft**(Fast Fourier Transform) module, which provides implementations for computing the DFT and its inverse.
The DFT helps analyze the frequency content of a signal, making it useful in many applications, including filtering, signal compression, and spectral analysis.

## Computing DFT Using fft() Function

The
**numpy.fft.fft()**function computes the one-dimensional n-point Discrete Fourier Transform (DFT) of an array. The function returns the frequency components of the input signal, with the zero frequency component at the beginning of the output array.
```
fft(x) = Discrete Fourier Transform of the input array x
```

### Example: Computing the DFT

In the following example, we compute the Discrete Fourier Transform of a one-dimensional array using NumPy's
**fft()**function −
```
import numpy as np

# Define an array of sample data (signal)
signal = np.array([1, 2, 3, 4])

# Compute the Discrete Fourier Transform of the signal
dft_signal = np.fft.fft(signal)

print("DFT of the signal:", dft_signal)
```

The result of the above code is −

```
DFT of the signal: [10.+0.j -2.+2.j -2.+0.j -2.-2.j]
```

## Computing the Inverse DFt

The
**numpy.fft.ifft()**function computes the inverse of the Discrete Fourier Transform, converting frequency components back into the original time-domain signal. This is useful when you want to reconstruct the signal from its frequency representation.
```
ifft(x) = Inverse Discrete Fourier Transform of the input array x
```

### Example: Inverse DFT

In the following example, we compute the inverse Discrete Fourier Transform of the DFT computed earlier using NumPy's
**ifft()**function −
```
import numpy as np

# Define the DFT of a signal (computed previously)
dft_signal = np.array([10+0j, -2+2j, -2+0j, -2-2j])

# Compute the Inverse Discrete Fourier Transform of the DFT
reconstructed_signal = np.fft.ifft(dft_signal)

print("Reconstructed signal:", reconstructed_signal)
```

Following is the output obtained −

```
Reconstructed signal: [1.+0.j 2.+0.j 3.+0.j 4.+0.j]
```

## Computing the DFT for a Real Signal

The
**numpy.fft.rfft()**function is optimized for computing the DFT of real input signals. This function returns only the non-negative frequency terms, as the negative frequency terms are redundant for real-valued signals.
```
rfft(x) = DFT of a real-valued input array x
```

### Example: DFT of a Real Signal

In the following example, we compute the Discrete Fourier Transform of a real-valued signal using NumPy's
**rfft()**function −
```
import numpy as np

# Define a real-valued signal
signal = np.array([1, 2, 3, 4])

# Compute the Discrete Fourier Transform of the real-valued signal
dft_real_signal = np.fft.rfft(signal)

print("DFT of the real-valued signal:", dft_real_signal)
```

This will produce the following result −

```
DFT of the real-valued signal: [10.+0.j -2.+2.j -2.+0.j]
```

## Computing the Inverse DFT for Real Signal

The
**numpy.fft.irfft()**function computes the inverse of the DFT for real-valued signals, reconstructing the original time-domain signal from its non-negative frequency components.
```
irfft(x) = Inverse DFT of a real-valued input array x
```

### Example: Inverse DFT of a Real Signal

In the following example, we compute the inverse DFT of a real-valued signal −

```
import numpy as np

# Define the DFT of a real-valued signal (computed previously)
dft_real_signal = np.array([10+0j, -2+2j])

# Compute the Inverse Discrete Fourier Transform of the real-valued DFT
reconstructed_real_signal = np.fft.irfft(dft_real_signal)

print("Reconstructed real-valued signal:", reconstructed_real_signal)
```

Following is the output of the above code −

```
Reconstructed real-valued signal: [4. 6.]
```

## Frequency Binning Using fftfreq() Function

The
**numpy.fft.fftfreq()**function is used to generate an array of sample frequencies corresponding to the components of the DFT. This is useful for plotting the frequency components of a signal or analyzing its spectral content.
```
fftfreq(n, d=1) = frequencies corresponding to the DFT of an array with n points, spaced by d
```

### Example: Frequency Binning

In the following example, we generate the frequencies corresponding to the DFT of a signal −

```
import numpy as np

# Define the number of points in the DFT and the sample spacing
n_points = 4
spacing = 1

# Generate the frequencies corresponding to the DFT
frequencies = np.fft.fftfreq(n_points, spacing)

print("Frequencies corresponding to the DFT:", frequencies)
```

The output obtained is as shown below −

```
Frequencies corresponding to the DFT: [ 0.    0.25 -0.5  -0.25]
```

---

## 114. NumPy - Fast Fourier Transform

*Source: [https://www.tutorialspoint.com/numpy/numpy_fast_fourier_transform.htm](https://www.tutorialspoint.com/numpy/numpy_fast_fourier_transform.htm)*

---

---
[Previous](/numpy/numpy_discrete_fourier_transform.htm)[Quiz](/numpy/quiz_on_numpy_fast_fourier_transform.htm)[Next](/numpy/numpy_inverse_fourier_transform.htm)
## NumPy Fast Fourier Transform

The Fast Fourier Transform (FFT) is a quick way to compute the Discrete Fourier Transform (DFT) and its inverse. It speeds up the process by reducing the time it takes from  O(n
) to O(nlogn), making it much faster, especially when working with large datasets.
In NumPy, you can calculate the FFT using the
**fft**module, which has functions for both one-dimensional and multi-dimensional FFT calculations.
## Computing the FFT Using fft() Function

The
**numpy.fft.fft()**function computes the one-dimensional n-point FFT of an array. It transforms a signal from the time domain into the frequency domain.
```
fft(x) = Fast Fourier Transform of the input array x
```

### Example: Computing the FFT

In the following example, we compute the Fast Fourier Transform of a one-dimensional array (signal) using the
**fft()**function −
```
import numpy as np

# Define a simple signal
signal = np.array([1, 2, 3, 4])

# Compute the Fast Fourier Transform of the signal
fft_signal = np.fft.fft(signal)

print("FFT of the signal:", fft_signal)
```

We get the output as shown below −

```
FFT of the signal: [10.+0.j -2.+2.j -2.+0.j -2.-2.j]
```

## Computing the Inverse FFT

The
**numpy.fft.ifft()**function computes the inverse of the Fast Fourier Transform, converting frequency components back into the original time-domain signal. This is useful when you want to reconstruct the signal from its frequency representation.
```
ifft(x) = Inverse Fast Fourier Transform of the input array x
```

### Example: Inverse FFT

In the following example, we compute the inverse FFT of the FFT computed earlier using NumPy's
**ifft()**function −
```
import numpy as np

# Define the FFT of a signal (computed previously)
fft_signal = np.array([10+0j, -2+2j, -2+0j, -2-2j])

# Compute the Inverse Fast Fourier Transform
reconstructed_signal = np.fft.ifft(fft_signal)

print("Reconstructed signal:", reconstructed_signal)
```

The result produced is as follows −

```
Reconstructed signal: [1.+0.j 2.+0.j 3.+0.j 4.+0.j]
```

## Computing the FFT for a Real Signal

The
**numpy.fft.rfft()**function is optimized for computing the FFT of real input signals. This function returns only the non-negative frequency terms, as the negative frequency terms are redundant for real-valued signals.
```
rfft(x) = FFT of a real-valued input array x
```

### Example: FFT of a Real Signal

In the following example, we compute the Fast Fourier Transform of a real-valued signal using NumPy's
**rfft()**function −
```
import numpy as np

# Define a real-valued signal
signal = np.array([1, 2, 3, 4])

# Compute the Fast Fourier Transform of the real-valued signal
fft_real_signal = np.fft.rfft(signal)

print("FFT of the real-valued signal:", fft_real_signal)
```

After executing the above code, we get the following output −

```
FFT of the real-valued signal: [10.+0.j -2.+2.j -2.+0.j]
```

## Computing Inverse FFT for a Real Signal

The
**numpy.fft.irfft()**function computes the inverse of the FFT for real-valued signals, reconstructing the original time-domain signal from its non-negative frequency components.
```
irfft(x) = Inverse FFT of a real-valued input array x
```

### Example: Inverse FFT of a Real Signal

In the following example, we compute the inverse FFT of a real-valued signal −

```
import numpy as np

# Define the FFT of a real-valued signal (computed previously)
fft_real_signal = np.array([10+0j, -2+2j])

# Compute the Inverse FFT of the real-valued FFT
reconstructed_real_signal = np.fft.irfft(fft_real_signal)

print("Reconstructed real-valued signal:", reconstructed_real_signal)
```

The output obtained is as shown below −

```
Reconstructed real-valued signal: [4. 6.]
```

## Frequency Binning Using fftfreq() Function

The
**numpy.fft.fftfreq()**function is used to generate an array of sample frequencies corresponding to the components of the FFT. This is useful for plotting the frequency components of a signal or analyzing its spectral content.
```
fftfreq(n, d=1) = frequencies corresponding to the FFT of an array with n points, spaced by d
```

### Example: Frequency Binning

In the following example, we generate the frequencies corresponding to the FFT of a signal −

```
import numpy as np

# Define the number of points in the FFT and the sample spacing
n_points = 4
spacing = 1

# Generate the frequencies corresponding to the FFT
frequencies = np.fft.fftfreq(n_points, spacing)

print("Frequencies corresponding to the FFT:", frequencies)
```

Following is the output of the above code −

```
Frequencies corresponding to the FFT: [ 0.    0.25 -0.5  -0.25]
```

## Using FFT in Multi-dimensional Arrays

NumPy's
**fft**module also supports multi-dimensional arrays, allowing you to compute the FFT along specific axes of the array. The**numpy.fft.fftn()**function computes the n-dimensional FFT, while**numpy.fft.ifftn()**function computes the inverse of the n-dimensional FFT.
```
fftn(x) = N-dimensional Fast Fourier Transform of an array x
```

### Example: 2D FFT

In the following example, we compute the 2-dimensional FFT of a 2D array −

```
import numpy as np

# Define a 2D signal
signal_2d = np.array([[1, 2], [3, 4]])

# Compute the 2D Fast Fourier Transform
fft_2d_signal = np.fft.fftn(signal_2d)

print("2D FFT of the signal:", fft_2d_signal)
```

This will produce the following result −

```
2D FFT of the signal: 
[[10.+0.j -2.+0.j]
 [-4.+0.j  0.+0.j]]
```

---

## 115. NumPy - Inverse Fourier Transform

*Source: [https://www.tutorialspoint.com/numpy/numpy_inverse_fourier_transform.htm](https://www.tutorialspoint.com/numpy/numpy_inverse_fourier_transform.htm)*

---

---
[Previous](/numpy/numpy_fast_fourier_transform.htm)[Quiz](/numpy/quiz_on_numpy_inverse_fourier_transform.htm)[Next](/numpy/numpy_fourier_series_and_transforms.htm)
## NumPy Inverse Fourier Transform

The Inverse Fourier Transform is the process of converting a frequency-domain representation of a signal back into the time-domain.

In NumPy, the Inverse Fourier Transform can be computed using the
**numpy.fft.ifft()**function for one-dimensional arrays and**numpy.fft.ifftn()**function for multi-dimensional arrays.
The inverse transform is essential when you need to reconstruct a signal after manipulating its frequency components, such as in filtering, noise reduction, or spectral analysis.

## Inverse Fast Fourier Transform

The
**numpy.fft.ifft()**function computes the Inverse Fast Fourier Transform (IFFT) of a one-dimensional array. It transforms a signal from the frequency domain back to the time domain, essentially reconstructing the original signal.
```
ifft(x) = Inverse FFT of the input array x
```

### Example: Computing the Inverse FFT

In the following example, we compute the Inverse Fast Fourier Transform of a one-dimensional array (frequency-domain signal) using the
**ifft()**function −
```
import numpy as np

# Define a frequency-domain signal (computed via FFT previously)
fft_signal = np.array([10+0j, -2+2j, -2+0j, -2-2j])

# Compute the Inverse Fast Fourier Transform
time_signal = np.fft.ifft(fft_signal)

print("Reconstructed time-domain signal:", time_signal)
```

We get the output as shown below −

```
Reconstructed time-domain signal: [1.+0.j 2.+0.j 3.+0.j 4.+0.j]
```

## IFFT for Real Signals

The
**numpy.fft.irfft()**function is used to compute the inverse FFT for real-valued signals. This function is optimized for reconstructing the original time-domain signal from its non-negative frequency components, which is particularly useful for real-valued input data.
```
irfft(x) = Inverse FFT of a real-valued input array x
```

### Example: Inverse FFT for Real Signals

In the following example, we compute the inverse FFT for a real-valued frequency-domain signal −

```
import numpy as np

# Define a real-valued frequency-domain signal (computed via rfft previously)
fft_real_signal = np.array([10+0j, -2+2j])

# Compute the Inverse FFT of the real-valued signal
reconstructed_real_signal = np.fft.irfft(fft_real_signal)

print("Reconstructed real-valued time-domain signal:", reconstructed_real_signal)
```

The result produced is as follows −

```
Reconstructed real-valued time-domain signal: [4. 6.]
```

## IFT in Multi-dimensional Arrays

For multi-dimensional arrays, NumPy provides the
**numpy.fft.ifftn()**function to compute the n-dimensional inverse FFT. This function can be used to reconstruct signals in multiple dimensions (e.g., 2D or 3D signals).
```
ifftn(x) = N-dimensional Inverse FFT of the input array x
```

### Example: 2D Inverse FFT

In the following example, we compute the 2-dimensional inverse FFT of a 2D frequency-domain array −

```
import numpy as np

# Define a 2D frequency-domain signal (computed via fftn previously)
fft_2d_signal = np.array([[10+0j, -2+2j], [-2+0j, -2-2j]])

# Compute the 2D Inverse Fast Fourier Transform
reconstructed_2d_signal = np.fft.ifftn(fft_2d_signal)

print("Reconstructed 2D time-domain signal:", reconstructed_2d_signal)
```

After executing the above code, we get the following output −

```
Reconstructed 2D time-domain signal: 
[[1.+0.j 3.+0.j]
 [3.+1.j 3.-1.j]]
```

## Important Considerations for IFT

Following are the important considerations for inverse Fourier Transforms −

- When performing an inverse Fourier transform, the result may contain complex numbers, even if the input was real-valued. The real part of the result can be extracted using**np.real()**function, or the imaginary part can be discarded if it is negligible.
- The inverse FFT is highly sensitive to the quality of the frequency-domain data. If any data is lost or corrupted in the frequency domain, the reconstruction in the time domain may be inaccurate.
- The sampling rate and spacing between points in the original signal (before applying FFT) are important when interpreting the reconstructed signal, as they determine the frequency resolution and the temporal resolution of the transform.

---

## 116. NumPy - Fourier Series and Transforms

*Source: [https://www.tutorialspoint.com/numpy/numpy_fourier_series_and_transforms.htm](https://www.tutorialspoint.com/numpy/numpy_fourier_series_and_transforms.htm)*

---

---
[Previous](/numpy/numpy_inverse_fourier_transform.htm)[Quiz](/numpy/quiz_on_numpy_fourier_series_and_transforms.htm)[Next](/numpy/numpy_signal_processing_applications.htm)
## NumPy Fourier Series and Transforms

In mathematics, a Fourier series breaks down periodic functions into sums of simpler sine and cosine waves. It is used to analyze functions or signals that repeat over time, such as sound waves or electrical signals.

Fourier transforms, on the other hand, extend this concept to non-periodic functions, converting them from the time domain to the frequency domain. This helps in understanding the frequency components of a signal, which is useful in signal processing and data analysis.

NumPy provides a range of functions to handle different types of Fourier transforms, both for real and complex data.

## Fourier Series

The Fourier series represents a periodic function as a sum of sines and cosines. It is useful for analyzing periodic signals, where the function can be broken down into a series of sinusoidal components with different frequencies.

```
f(t) = a_0 + Σ (a_n cos(nωt) + b_n sin(nωt))
```

where
**ω**is the fundamental angular frequency, and**a_n**and**b_n**are the Fourier coefficients.
### Example: Computing Fourier Coefficients

Although NumPy does not provide a direct function to compute Fourier series coefficients, we can use the
**fft()**function to approximate them. In the following example, we compute the Fourier coefficients for a simple periodic signal −
```
import numpy as np

# Define a periodic signal
t = np.linspace(0, 1, 500, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# Compute the Fourier coefficients using FFT
fourier_coeffs = np.fft.fft(signal)

print("Fourier coefficients:", fourier_coeffs)
```

The result shows the Fourier coefficients of the signal −

```
Fourier coefficients: [-3.87326339e-14+0.00000000e+00j  4.12725409e-14+2.68673972e-14j...]
```

## Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT) is an algorithm to compute the Discrete Fourier Transform (DFT) and its inverse. The FFT decomposes a signal into its frequency components, providing a frequency-domain representation.

NumPy provides several functions for computing the FFT, including
**numpy.fft.fft()**for one-dimensional transforms and**numpy.fft.fftn()**for multi-dimensional transforms.
### Example: Computing the FFT

In the following example, we compute the FFT of a one-dimensional signal −

```
import numpy as np

# Define a signal
t = np.linspace(0, 1, 500, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t)

# Compute the FFT
fft_values = np.fft.fft(signal)

print("FFT values:", fft_values)
```

The result shows the FFT values of the signal −

```
FFT values: [-2.37154386e-14+0.00000000e+00j  6.32657050e-15+6.43929354e-15j-4.51716973e-14+3.82357443e-14j ...]
```

## Inverse Fast Fourier Transform (IFFT)

The Inverse Fast Fourier Transform (IFFT) converts a frequency-domain representation of a signal back to the time domain. NumPy provides the
**numpy.fft.ifft()**function for one-dimensional IFFT and**numpy.fft.ifftn()**for multi-dimensional IFFT.
### Example: Computing the IFFT

In the following example, we compute the IFFT of a frequency-domain signal to reconstruct the original time-domain signal −

```
import numpy as np

# Define a frequency-domain signal (computed via FFT previously)
fft_values = np.array([10+0j, -2+2j, -2+0j, -2-2j])

# Compute the IFFT
time_signal = np.fft.ifft(fft_values)

print("Reconstructed time-domain signal:", time_signal)
```

The result shows the reconstructed time-domain signal −

```
Reconstructed time-domain signal: [1.+0.j 2.+0.j 3.+0.j 4.+0.j]
```

## Real-valued FFT (rFFT)

The Real-valued FFT (rFFT) is optimized for real-valued input signals. NumPy provides the
**numpy.fft.rfft()**function for computing the FFT of real-valued signals and**numpy.fft.irfft()**function for computing the IFFT of real-valued signals.
### Example: Computing the rFFT and irFFT

In the following example, we compute the rFFT of a real-valued signal and then reconstruct the signal using irFFT −

```
import numpy as np

# Define a real-valued signal
t = np.linspace(0, 1, 500, endpoint=False)
real_signal = np.sin(2 * np.pi * 5 * t)

# Compute the rFFT
rfft_values = np.fft.rfft(real_signal)

# Compute the inverse rFFT
reconstructed_signal = np.fft.irfft(rfft_values)

print("Reconstructed real-valued time-domain signal:", reconstructed_signal)
```

The result shows the reconstructed real-valued time-domain signal −

```
Reconstructed real-valued time-domain signal: [ 5.04870979e-32  6.27905195e-02  1.25333234e-01  1.87381315e-01 ...]
```

## Applications of Fourier Transforms

Fourier transforms are widely used in various fields, such as −

- **Signal processing:**Analyzing and filtering signals in communications, audio processing, and image processing.
- **Physics and engineering:**Studying wave-forms, vibrations, and oscillations.
- **Data compression:**Reducing the size of data by transforming and truncating frequency components.
- **Financial analysis:**Analyzing time-series data in stock market and economic forecasting.

---

## 117. NumPy - Signal Processing Applications

*Source: [https://www.tutorialspoint.com/numpy/numpy_signal_processing_applications.htm](https://www.tutorialspoint.com/numpy/numpy_signal_processing_applications.htm)*

---

---
[Previous](/numpy/numpy_fourier_series_and_transforms.htm)[Quiz](/numpy/quiz_on_numpy_signal_processing_applications.htm)[Next](/numpy/numpy_convolution.htm)
## NumPy Signal Processing

Signal processing in NumPy involves manipulating and analyzing signals, which are functions that convey information. These signals can be anything from sound waves to digital data. The goal is to transform or extract useful information from these signals.

NumPy provides tools for signal processing, such as filtering, Fourier transforms, and convolution. For example, you can use Fourier transforms to convert signals from the time domain to the frequency domain, making it easier to analyze their frequency components.

You can also apply filters to remove noise or extract specific parts of a signal. These operations are useful in various fields, including audio processing, communications, and image analysis.

## Filtering Signals

Filtering is a fundamental signal processing technique used to remove unwanted components from a signal or to extract useful parts. Filters can be low-pass, high-pass, band-pass, or band-stop, depending on the frequencies they allow or block.

### Example: Low-Pass Filtering

In the following example, we apply a low-pass filter to a noisy signal to remove high-frequency noise −

```
import numpy as np
from scipy.signal import butter, lfilter

# Generate a sample signal
t = np.linspace(0, 1, 500, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + np.random.randn(500) * 0.5

# Design a low-pass filter
def butter_lowpass(cutoff, fs, order=5):
   nyquist = 0.5 * fs
   normal_cutoff = cutoff / nyquist
   b, a = butter(order, normal_cutoff, btype='low', analog=False)
   return b, a

def lowpass_filter(data, cutoff, fs, order=5):
   b, a = butter_lowpass(cutoff, fs, order=order)
   y = lfilter(b, a, data)
   return y

# Apply the low-pass filter
cutoff_frequency = 10
sampling_rate = 500
filtered_signal = lowpass_filter(signal, cutoff_frequency, sampling_rate)

print("Filtered signal:", filtered_signal)
```

The result shows the filtered signal with high-frequency noise removed −

```
Filtered signal: [ 3.92381804e-07  3.54383602e-06  1.63967023e-05  5.28930079e-05 ... -9.55624060e-01 -9.81063288e-01 -1.00076423e+00]
```

## Fourier Transform for Frequency Analysis

Fourier Transform is widely used in signal processing to analyze the frequency components of a signal. It converts a time-domain signal into its frequency-domain representation, revealing the frequencies present in the signal.

### Example: Frequency Analysis Using FFT

In the following example, we use the Fast Fourier Transform (FFT) to analyze the frequency components of a signal −

```
import numpy as np

# Generate a sample signal
t = np.linspace(0, 1, 500, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# Compute the FFT
fft_values = np.fft.fft(signal)

# Compute the frequency bins
freqs = np.fft.fftfreq(len(signal), d=t[1] - t[0])

print("Frequency components:", freqs)
print("FFT values:", fft_values)
```

The result shows the frequency components and their corresponding FFT values −

```
Frequency components: [   0.    1.    2.    3.    4.    5. ...]
FFT values: [-3.87326339e-14+0.00000000e+00j  4.12725409e-14+2.68673972e-14j ...]
```

## Convolution and Correlation

Convolution and correlation are mathematical operations used in signal processing to analyze and modify signals. Convolution is used to filter signals, while correlation measures the similarity between signals.

### Example: Convolution

In the following example, we use convolution to apply a smoothing filter to a signal −

```
import numpy as np

# Generate a sample signal
t = np.linspace(0, 1, 500, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + np.random.randn(500) * 0.5

# Define a smoothing filter
filter_kernel = np.ones(10) / 10

# Apply convolution
smoothed_signal = np.convolve(signal, filter_kernel, mode='same')

print("Smoothed signal:", smoothed_signal)
```

The result shows the smoothed signal after applying convolution −

```
Smoothed signal: [ 0.10531384  0.10089093  0.10978193 ... -0.19290414-0.26696232 -0.2166795 ]
```

### Example: Cross-Correlation

In the following example, we compute the cross-correlation between two signals −

```
import numpy as np

# Generate two sample signals
t = np.linspace(0, 1, 500, endpoint=False)
signal1 = np.sin(2 * np.pi * 5 * t)
signal2 = np.sin(2 * np.pi * 5 * t + np.pi / 4)

# Compute the cross-correlation
cross_corr = np.correlate(signal1, signal2, mode='full')

print("Cross-correlation:", cross_corr)
```

The result shows the cross-correlation between the two signals −

```
Cross-correlation: [ 0.00000000e+00  4.15241156e-02  1.21369107e-01 ... -2.76126688e-01 -1.35723843e-01 -4.43996022e-02]
```

## Signal Resampling

Resampling involves changing the sampling rate of a signal, either increasing (upsampling) or decreasing (downsampling) the number of samples. This is often required when working with signals at different sampling rates.

### Example: Signal Resampling

In the following example, we downsample a signal by a factor of 2 −

```
import numpy as np
from scipy.signal import resample

# Generate a sample signal
t = np.linspace(0, 1, 500, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t)

# Downsample the signal by a factor of 2
num_samples = len(signal) // 2
downsampled_signal = resample(signal, num_samples)

print("Downsampled signal:", downsampled_signal)
```

The result shows the downsampled signal −

```
Downsampled signal: [-3.51535300e-16  1.25333234e-01  2.48689887e-01 ... -2.48689887e-01 -1.25333234e-01]
```

## Wavelet Transform

Wavelet Transform is another technique for analyzing signals, especially for non-stationary signals where frequency components vary over time. Wavelet Transform provides a time-frequency representation of the signal.

### Example: Continuous Wavelet Transform

In the following example, we use the Continuous Wavelet Transform (CWT) to analyze a signal −

```
import numpy as np
import pywt

# Generate a sample signal
t = np.linspace(0, 1, 500, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# Compute the Continuous Wavelet Transform
coeffs, freqs = pywt.cwt(signal, scales=np.arange(1, 128), wavelet='gaus1')

print("CWT coefficients:", coeffs)
print("Frequencies:", freqs)
```

The result shows the CWT coefficients and corresponding frequencies −

```
CWT coefficients: [[ 5.11544621e-01  5.51952738e-01  5.90635494e-01 ...  6.21499677e-01  5.95101447e-01  5.67801389e-01]
 [ 1.08933167e+00  1.07658265e+00  1.05980188e+00 ...  9.62940518e-01  9.99652239e-01  1.03508272e+00]
 ...
 [-5.28687791e-04 -4.50587872e-04 -3.71247724e-04 ... -1.84740494e-04 -3.07220733e-04 -4.29412055e-04]]
Frequencies: [ 0.03125     0.0625      0.09375    ...  1.65625     1.6875      1.71875]
```

## Applications of Signal Processing

Signal processing has numerous applications in various fields, such as −

- **Audio processing:**Noise reduction, equalization, and compression in music and speech signals.
- **Image processing:**Enhancement, filtering, and compression of digital images.
- **Communications:**Modulation, demodulation, and error correction in transmission systems.
- **Control systems:**Signal conditioning and feedback control in engineering systems.
- **Biomedical engineering:**Analysis and filtering of physiological signals such as ECG and EEG.

---

## 118. NumPy - Convolution

*Source: [https://www.tutorialspoint.com/numpy/numpy_convolution.htm](https://www.tutorialspoint.com/numpy/numpy_convolution.htm)*

---

---
[Previous](/numpy/numpy_signal_processing_applications.htm)[Quiz](/numpy/quiz_on_numpy_convolution.htm)[Next](/numpy/numpy_polynomial_representation.htm)
## NumPy Convolution

Convolution in NumPy is a mathematical operation used to combine two arrays (such as signals or images) in a specific way to produce a third array. This operation helps in filtering, smoothing, and detecting features within the data.

When you perform convolution, you slide one array (called the kernel or filter) over another array (the input) and calculate the sum of element-wise multiplications at each position. This process enhances certain aspects of the input array, like edges in an image or specific frequencies in a signal.

In NumPy, you can use the numpy.convolve() function for one-dimensional arrays and scipy.ndimage.convolve() for multi-dimensional arrays to perform convolution, which is widely used in signal processing and image analysis.

## 1D Convolution

One-dimensional convolution is commonly used in signal processing. It involves sliding one signal (kernel) over another (input signal) and computing the dot product at each position.

### Example: 1D Convolution

In the following example, we perform a 1D convolution between an input signal and a kernel using NumPy −

```
import numpy as np

# Define the input signal
input_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Define the convolution kernel
kernel = np.array([0.2, 0.5, 0.2])

# Perform 1D convolution
convolved_signal = np.convolve(input_signal, kernel, mode='same')

print("Input signal:", input_signal)
print("Kernel:", kernel)
print("Convolved signal:", convolved_signal)
```

The result shows the input signal, kernel, and convolved signal −

```
Input signal: [1 2 3 4 5 6 7 8 9]
Kernel: [0.2 0.5 0.2]
Convolved signal: [0.9 1.8 2.7 3.6 4.5 5.4 6.3 7.2 6.1]
```

## 2D Convolution

Two-dimensional convolution is widely used in image processing for tasks such as blurring, sharpening, and edge detection. It involves sliding a 2D kernel over a 2D input array (image) and computing the dot product at each position.

### Example: 2D Convolution

In the following example, we perform a 2D convolution on a sample image using a kernel −

```
import numpy as np
from scipy.signal import convolve2d

# Define a sample image (2D array)
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Define a 2D convolution kernel
kernel = np.array([
    [0.1, 0.2, 0.1],
    [0.2, 0.4, 0.2],
    [0.1, 0.2, 0.1]
])

# Perform 2D convolution
convolved_image = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

print("Image:\n", image)
print("Kernel:\n", kernel)
print("Convolved image:\n", convolved_image)
```

The result shows the original image, kernel, and convolved image −

```
Image:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]
Kernel:
[[0.1 0.2 0.1]
 [0.2 0.4 0.2]
 [0.1 0.2 0.1]]
Convolved image:
[[ 2.4  4.   5.2  4.5]
 [ 6.4  9.6 11.2  9.2]
 [11.2 16.  17.6 14. ]
 [10.8 15.2 16.4 12.9]]
```

## 3D Convolution

Three-dimensional convolution is used in various applications such as 3D image processing, video processing, and volumetric data analysis. It involves sliding a 3D kernel over a 3D input array and computing the dot product at each position.

### Example: 3D Convolution

In the following example, we perform a 3D convolution on a sample 3D array −

```
import numpy as np
from scipy.ndimage import convolve

# Define a sample 3D array
array_3d = np.random.rand(4, 4, 4)

# Define a 3D convolution kernel
kernel_3d = np.ones((3, 3, 3)) / 27

# Perform 3D convolution
convolved_3d = convolve(array_3d, kernel_3d, mode='constant', cval=0.0)

print("3D array:\n", array_3d)
print("Kernel:\n", kernel_3d)
print("Convolved 3D array:\n", convolved_3d)
```

The result shows the original 3D array, kernel, and convolved 3D array −

```
3D array:
[[[0.46186776 0.09130699 0.36913034 0.51669149]
  [0.90316515 0.38362845 0.90886156 0.60454144]
  [0.80756784 0.28656032 0.73140925 0.75789388]
  [0.36958966 0.66157156 0.19902489 0.89519004]]

 [[0.04953332 0.98571523 0.80654445 0.47526839]
  [0.67375222 0.31837149 0.20836025 0.10996474]
  [0.48799518 0.34754979 0.85689208 0.21079349]
  [0.91936308 0.79818294 0.18737238 0.01728286]]

 [[0.50793178 0.93691426 0.00515023 0.29870646]
  [0.22871996 0.3098202  0.0396516  0.98755326]
  [0.3347781  0.56108282 0.89520242 0.77143481]
  [0.64504437 0.0133608  0.61686021 0.01443242]]

 [[0.96126985 0.11224998 0.79332687 0.0438432 ]
  [0.39348891 0.36066344 0.06157876 0.29697117]
  [0.40409768 0.88212056 0.22872878 0.7545221 ]
  [0.09578231 0.06486727 0.94749091 0.79605238]]]
Kernel:
[[[0.03703704 0.03703704 0.03703704]
  [0.03703704 0.03703704 0.03703704]
  [0.03703704 0.03703704 0.03703704]]

 [[0.03703704 0.03703704 0.03703704]
  [0.03703704 0.03703704 0.03703704]
  [0.03703704 0.03703704 0.03703704]]

 [[0.03703704 0.03703704 0.03703704]
  [0.03703704 0.03703704 0.03703704]
  [0.03703704 0.03703704 0.03703704]]]
Convolved 3D array:
[[[0.14323484 0.22815693 0.21401425 0.14812454]
  [0.21470421 0.35845228 0.3322031  0.24282783]
  [0.25767769 0.37219326 0.3142019  0.21065136]
  [0.17327335 0.24641033 0.22036013 0.14280959]]

 [[0.21669359 0.30327501 0.30948818 0.19742312]
  [0.32134299 0.49990604 0.51018517 0.35385371]
  [0.33518903 0.5071755  0.47010555 0.3338045 ]
  [0.23083876 0.35997806 0.32674433 0.2279181 ]]

 [[0.21623817 0.28714973 0.26483904 0.15284887]
  [0.32800203 0.47227742 0.46885114 0.29053678]
  [0.29033486 0.44004365 0.43174681 0.29633869]
  [0.20571203 0.34395451 0.33200848 0.23322462]]

 [[0.14115031 0.17447281 0.15727516 0.0935845 ]
  [0.22196806 0.29691764 0.30887114 0.19172851]
  [0.15903061 0.26234589 0.31860718 0.23742514]
  [0.11115311 0.21071912 0.2424502  0.18610089]]]
```

## Applications of Convolution

Convolution has a wide range of applications, such as −

- **Image processing:**Blurring, sharpening, edge detection, and feature extraction.
- **Signal processing:**Filtering, smoothing, and noise reduction in audio and communication signals.
- **Machine learning:**Convolutional neural networks (CNNs) use convolutional layers for feature extraction in tasks such as image recognition and natural language processing.
- **Time series analysis:**Convolution can be used to smooth or filter time series data.

---

## 119. NumPy - Polynomial Representation

*Source: [https://www.tutorialspoint.com/numpy/numpy_polynomial_representation.htm](https://www.tutorialspoint.com/numpy/numpy_polynomial_representation.htm)*

---

---
[Previous](/numpy/numpy_convolution.htm)[Quiz](/numpy/quiz_on_numpy_polynomial_representation.htm)[Next](/numpy/numpy_polynomial_operations.htm)
## Polynomial Representation in NumPy

Polynomial representation in NumPy refers to how polynomials are expressed and manipulated using arrays. A polynomial is a mathematical expression involving a sum of powers of a variable, such as ax
+ bx + c, where a, b, and c are constants and x is the variable.
In NumPy, polynomials are represented as arrays, where each element corresponds to a coefficient of the polynomial.

For example, the polynomial 3x
+ 2x + 1 would be represented as the array [3, 2, 1], where the first element represents the coefficient of x, the second represents the coefficient of x, and the third represents the constant term.
NumPy provides the
**numpy.polynomial**module to create and work with polynomial objects, which allow you to perform operations like evaluation, differentiation, and integration on polynomials.
## Creating Polynomials

Polynomials in NumPy can be created using the
**numpy.polynomial.Polynomial**class, which represents polynomials in terms of their coefficients.
### Example: Creating a Polynomial

In the following example, we create a polynomial 2x
+ 3x+ x + 5 using NumPy −
```
import numpy as np
from numpy.polynomial import Polynomial

# Define the coefficients of the polynomial
coefficients = [5, 1, 3, 2]

# Create the polynomial
p = Polynomial(coefficients)

print("Polynomial:", p)
```

The result shows the created polynomial −

```
Polynomial: 5.0 + 1.0x + 3.0x + 2.0x
```

## Evaluating Polynomials

Polynomials can be evaluated at specific points using the
**__call__**method of the Polynomial class or the**numpy.polyval()**function, which is another method for evaluating polynomials.
### Example: Evaluating a Polynomial

In the following example, we evaluate the polynomial 2x
+ 3x+ x + 5 at x = 2 using both the**__call__**method and the**polyval()**function −
```
import numpy as np
from numpy.polynomial import Polynomial

# Define the polynomial
# 2x^3 + 3x^2 + x + 5
p = Polynomial([5, 1, 3, 2])

# Evaluate the polynomial at x = 2 using the __call__ method
value_call = p(2)

# Evaluate the polynomial at x = 2 using numpy.polyval
value_evaluate = np.polyval(p.coef, 2)

print("Value of the polynomial at x = 2 using __call__:", value_call)
print("Value of the polynomial at x = 2 using numpy.polyval:", value_evaluate)
```

The result shows the value of the polynomial at x = 2 using both methods −

```
Value of the polynomial at x = 2 using __call__: 35.0
Value of the polynomial at x = 2 using numpy.polyval: 52.0
```

## Performing Operations on Polynomials

NumPy allows various operations on polynomials, such as addition, subtraction, multiplication, and division.

You can create polynomial objects by instantiating the
**Polynomial**class and passing the coefficients of the polynomial. NumPy automatically handles the necessary calculations on the polynomial coefficients.
### Example: Polynomial Addition

In the following example, we add two polynomials 2x
+ 3x + 1 and x+ 4x + 2 using the**+**operator −
```
import numpy as np
from numpy.polynomial import Polynomial

# Define the polynomials
# 2x^2 + 3x + 1
p1 = Polynomial([1, 3, 2]) 
# x^2 + 4x + 2 
p2 = Polynomial([2, 4, 1])  

# Add the polynomials
result_add = p1 + p2

print("Result of polynomial addition:", result_add)
```

The result shows the sum of the two polynomials −

```
Result of polynomial addition: 3.0 + 7.0x + 3.0x
```

### Example: Polynomial Multiplication

Here, we are multiplying two polynomials 2x + 1 and x + 2 using the * operator −

```
import numpy as np
from numpy.polynomial import Polynomial

# Define the polynomials
# 2x + 1
p1 = Polynomial([1, 2])  
# x + 2
p2 = Polynomial([2, 1])  

# Multiply the polynomials
result_multiply = p1 * p2

print("Result of polynomial multiplication:", result_multiply)
```

The result shows the product of the two polynomials −

```
Result of polynomial multiplication: 2.0 + 5.0x + 2.0x
```

### Example: Polynomial Differentiation

In this example, we find the derivative of the polynomial 2x
+ 3x+ x + 5 −
```
import numpy as np
from numpy.polynomial import Polynomial

# Define the polynomial
# 2x^3 + 3x^2 + x + 5
p = Polynomial([5, 1, 3, 2])  

# Differentiate the polynomial
result_diff = p.deriv()

print("Result of polynomial differentiation:", result_diff)
```

The result shows the first derivative of the original polynomial −

```
Result of polynomial differentiation: 1.0 + 6.0x + 6.0x
```

## Fitting Polynomials to Data

Fitting polynomials to data involves finding the polynomial that best matches a given set of data points. In NumPy, this is done using the numpy.fit() function, which fits a polynomial of a specified degree to the data by minimizing the error.

The result is a polynomial that approximates the data, useful for tasks like curve fitting or trend analysis.

### Example: Fitting a Polynomial to Data

In the following example, we fit a polynomial of degree 2 to a set of data points −

```
import numpy as np
from numpy.polynomial import Polynomial

# Define data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 2, 0, 2, 1])

# Fit a polynomial of degree 2 to the data
p_fit = Polynomial.fit(x, y, deg=2)

print("Fitted polynomial:", p_fit)
```

The result shows the fitted polynomial −

```
Fitted polynomial: 1.2 - (7.02166694e-17)x + (1.22883961e-16)x
```

---

## 120. NumPy - Polynomial Operations

*Source: [https://www.tutorialspoint.com/numpy/numpy_polynomial_operations.htm](https://www.tutorialspoint.com/numpy/numpy_polynomial_operations.htm)*

---

---
[Previous](/numpy/numpy_polynomial_representation.htm)[Quiz](/numpy/quiz_on_numpy_polynomial_operations.htm)[Next](/numpy/numpy_finding_polynomial_roots.htm)
## Polynomial Operations in NumPy

Polynomial operations in NumPy refer to various mathematical tasks you can perform on polynomials, such as addition, subtraction, multiplication, division, and evaluation.

NumPy makes it easy to work with polynomials by using arrays to represent their coefficients. You can use functions like numpy.polyadd(), numpy.polysub(), and numpy.polymul() to perform these operations, and methods like numpy.polyval() to evaluate a polynomial at specific values of x.

## Adding Polynomials

Polynomials can be added using the
**numpy.polyadd()**function in NumPy. The result is a new polynomial whose coefficients are the sum of the corresponding coefficients of the original polynomials.
### Example: Polynomial Addition

In this example, we add two polynomials using the polyadd() function in NumPy −

```
import numpy as np

# Define two polynomials using their coefficients
# 1 + 2x + 3x
p1 = np.array([1, 2, 3])  
# 0 + 1x + 4x
p2 = np.array([0, 1, 4]) 

# Add the polynomials using numpy.polyadd
result_add = np.polyadd(p1, p2)

print("Result of polynomial addition:", result_add)
```

The result of adding the two polynomials is −

```
Result of polynomial addition: [1 3 7]
```

## Subtracting Polynomials

Polynomials can be subtracted using the
**numpy.polysub()**function. The result is a new polynomial whose coefficients are the differences of the corresponding coefficients of the original polynomials.
### Example: Polynomial Subtraction

In this example, we subtract one polynomial from another using the polysub() function in NumPy −

```
import numpy as np

# Define two polynomials using their coefficients
# 1 + 2x + 3x
p1 = np.array([1, 2, 3])  
# 0 + 1x + 4x
p2 = np.array([0, 1, 4]) 

# Subtract the polynomials using numpy.polysub
result_sub = np.polysub(p1, p2)

print("Result of polynomial subtraction:", result_sub)
```

The result of subtracting the two polynomials is −

```
Result of polynomial subtraction: [ 1  1 -1]
```

## Multiplying Polynomials

Polynomials can be multiplied using the
**numpy.polymul()**function. The multiplication of two polynomials is carried out by multiplying each term in the first polynomial by each term in the second polynomial and summing the results.
### Example: Polynomial Multiplication

In this example, we multiply two polynomials using the polymul() function in NumPy −

```
import numpy as np

# Define two polynomials using their coefficients
# 1 + 2x + 3x
p1 = np.array([1, 2, 3])  
# 0 + 1x + 4x
p2 = np.array([0, 1, 4]) 

# Multiply the polynomials using numpy.polymul
result_mul = np.polymul(p1, p2)

print("Result of polynomial multiplication:", result_mul)
```

The result of multiplying the two polynomials is −

```
Result of polynomial multiplication: [ 1  6 11 12]
```

## Evaluating Polynomials

The
**numpy.polyval()**function is used to evaluate a polynomial at a specific value of**x**. This can be useful for finding the value of the polynomial for any given**x**.
### Example: Polynomial Evaluation

In this example, we evaluate the polynomial at
**x = 2**using the polyval() function −
```
import numpy as np

# Define polynomial using its coefficients
# 1 + 2x + 3x
p1 = np.array([1, 2, 3])  

# Evaluate the polynomial at x = 2 using numpy.polyval
x_value = 2
result_eval = np.polyval(p1, x_value)

print(f"Polynomial evaluated at x = {x_value}:", result_eval)
```

The result of evaluating the polynomial at
**x = 2**is −
```
Polynomial evaluated at x = 2: 11
```

## Polynomial Differentiation

Polynomials can be differentiated using the
**numpy.polyder()**function. The derivative of a polynomial is a new polynomial obtained by differentiating each term of the original polynomial with respect to the variable.
### Example: Polynomial Differentiation

In this example, we differentiate the polynomial using the polyder() function in NumPy −

```
import numpy as np

# Define polynomial using its coefficients
# 1 + 2x + 3x
p1 = np.array([1, 2, 3])  

# Differentiate the polynomial using numpy.polyder
derivative = np.polyder(p1)

print("Derivative of the polynomial:", derivative)
```

The result of differentiating the polynomial is −

```
Derivative of the polynomial: [2 2]
```

## Polynomial Integration

Polynomials can be integrated using the
**numpy.polyint()**function. The integral of a polynomial is a new polynomial obtained by integrating each term of the original polynomial with respect to the variable.
### Example: Polynomial Integration

In this example, we integrate the polynomial using the polyint() function in NumPy −

```
import numpy as np

# Define polynomial using its coefficients
# 1 + 2x + 3x
p1 = np.array([1, 2, 3])  

# Integrate the polynomial using numpy.polyint
integral = np.polyint(p1)

print("Integral of the polynomial:", integral)
```

The result of integrating the polynomial is −

```
Integral of the polynomial: [0.33333333 1.         3.         0.        ]
```

---

## 121. NumPy - Finding Roots of Polynomials

*Source: [https://www.tutorialspoint.com/numpy/numpy_finding_polynomial_roots.htm](https://www.tutorialspoint.com/numpy/numpy_finding_polynomial_roots.htm)*

---

---
[Previous](/numpy/numpy_polynomial_operations.htm)[Quiz](/numpy/quiz_on_numpy_finding_polynomial_roots.htm)[Next](/numpy/numpy_evaluating_polynomials.htm)
## Finding Roots of Polynomials in NumPy

Finding the roots of a polynomial means determining the values of x where the polynomial equals zero. In NumPy, you can find the roots of a polynomial using the
**numpy.roots()**function.
Roots are fundamental in various mathematical applications, such as solving equations, optimization problems, and analyzing the behavior of polynomials in signal processing or numerical analysis.

## The numpy.roots() Function

The
**numpy.roots()**function takes the coefficients of a polynomial as an input and returns the roots of the polynomial.
The polynomial is expressed in terms of its coefficients, starting with the highest power and ending with the constant term. The function finds all the roots of the polynomial, including real and complex roots.

### Example

In this example, we find the roots of a polynomial defined by the equation using the roots() function in NumPy −

```
import numpy as np

# Define the coefficients of the polynomial 1x - 6x + 11x - 6
coefficients = np.array([1, -6, 11, -6])

# Find the roots of the polynomial using numpy.roots
roots = np.roots(coefficients)

print("Roots of the polynomial:", roots)
```

The result of finding the roots for the given polynomial is as follows −

```
Roots of the polynomial: [3. 2. 1.]
```

## Real and Complex Roots

NumPy handles both real and complex roots. If the polynomial has complex roots,
**numpy.roots()**will return them as complex numbers. For example, the roots of a quadratic polynomial with no real roots will appear as complex numbers.
### Example: Finding Complex Roots

In this example, we find the roots of a quadratic polynomial using the numpy.roots() function −

```
import numpy as np

# Define the coefficients of the polynomial x + 1
coefficients_complex = np.array([1, 0, 1])

# Find the roots of the polynomial using numpy.roots
roots_complex = np.roots(coefficients_complex)

print("Roots of the complex polynomial:", roots_complex)
```

The result of finding the roots of the complex polynomial is as shown below −

```
Roots of the complex polynomial: [-0.+1.j  0.-1.j]
```

## Verifying the Roots

After finding the roots, it is often useful to verify that they satisfy the original polynomial equation. This can be done by substituting the roots back into the polynomial and checking if the result is close to zero.

### Example

In this example, we verify the roots of a polynomial by evaluating the polynomial at the roots −

```
import numpy as np

# Define the polynomial coefficients
coefficients = np.array([1, -6, 11, -6])

# Find the roots of the polynomial
roots = np.roots(coefficients)

# Verify the roots by evaluating the polynomial at the roots
verification = np.polyval(coefficients, roots)

print("Verification results:", verification)
```

The verification results (should be close to zero for all roots) is −

```
Verification results: [3.55271368e-15 2.66453526e-15 0.00000000e+00]
```

## Handling Higher-Degree Polynomials

The NumPy roots() function is capable of finding the roots of polynomials of any degree. As the degree of the polynomial increases, the number of roots increases as well. The function will return all roots, real and complex, of the given polynomial.

### Example

In this example, we find the roots of a polynomial of degree 5 using the roots() function −

```
import numpy as np

# Define the coefficients of the polynomial 1x - 3x + 5x - 7x + 11x - 13
coefficients_high_degree = np.array([1, -3, 5, -7, 11, -13])

# Find the roots of the polynomial using numpy.roots
roots_high_degree = np.roots(coefficients_high_degree)

print("Roots of the higher-degree polynomial:", roots_high_degree)
```

The result of finding the roots for the higher-degree polynomial is as follows −

```
Roots of the higher-degree polynomial: [-0.56466348+1.42799478j -0.56466348-1.42799478j  1.18589358+1.31548183j
1.18589358-1.31548183j  1.7575398 +0.j        ]
```

---

## 122. NumPy - Evaluating Polynomials

*Source: [https://www.tutorialspoint.com/numpy/numpy_evaluating_polynomials.htm](https://www.tutorialspoint.com/numpy/numpy_evaluating_polynomials.htm)*

---

---
[Previous](/numpy/numpy_finding_polynomial_roots.htm)[Quiz](/numpy/quiz_on_numpy_evaluating_polynomials.htm)[Next](/numpy/numpy_statistical_functions.htm)
## Evaluating Polynomials in NumPy

Evaluating polynomials in NumPy means calculating the value of the polynomial at a specific point. You can do this using the numpy.polyval() function in NumPy.

The polynomial is defined by its coefficients, starting with the highest degree term and ending with the constant term. The function evaluates the polynomial at the given value of
**x**, and can handle multiple values of**x**at once, returning the corresponding results.
## The numpy.polyval() Function

The
**numpy.polyval()**function evaluates a polynomial for a given value (or array of values) of**x**. It takes two arguments: the coefficients of the polynomial and the value(s) at which the polynomial should be evaluated.
The polynomial is expressed in terms of its coefficients, starting with the highest power term and ending with the constant term.

### Example: Evaluating Polynomial at a Single Point

Let us consider the polynomial
**f(x) = 2x - 3x + 4**. We can evaluate this polynomial at**x = 2**−
```
import numpy as np

# Define the coefficients of the polynomial 2x - 3x + 4
coefficients = np.array([2, -3, 4])

# Evaluate the polynomial at x = 2 using numpy.polyval
x_value = 2
result = np.polyval(coefficients, x_value)

print(f"Value of the polynomial at x = {x_value}: {result}")
```

The result of evaluating the polynomial at
**x = 2**is −
```
Value of the polynomial at x = 2: 6
```

## Evaluating Polynomials at Multiple Points

In addition to evaluating the polynomial at a single point, the
**numpy.polyval()**function can also evaluate the polynomial at multiple points at once. The function accepts an array of**x**values and returns an array of corresponding results.
### Example

Let us evaluate the same polynomial
**f(x) = 2x - 3x + 4**at multiple values of**x**−
```
import numpy as np

# Define the coefficients of the polynomial 2x - 3x + 4
coefficients = np.array([2, -3, 4])

# Define the x values for evaluation
x_values = np.array([-1, 0, 1, 2])

# Evaluate the polynomial at multiple points using numpy.polyval
results = np.polyval(coefficients, x_values)

print("Values of the polynomial at x = [-1, 0, 1, 2]:", results)
```

The result of evaluating the polynomial at the given points is as shown below −

```
Values of the polynomial at x = [-1, 0, 1, 2]: [9 4 3 6]
```

## Evaluating Polynomials with Complex Roots

If a polynomial has complex roots or coefficients, the
**numpy.polyval()**function can still evaluate the polynomial correctly. The function will return complex numbers if the result involves complex arithmetic.
### Example

Consider the polynomial
**f(x) = (2 + 3i)x + (1 - 2i)x + (4 + i)**. We can evaluate this polynomial at**x = 1 + i**as shown below −
```
import numpy as np

# Define the coefficients of the polynomial with complex numbers
coefficients_complex = np.array([2 + 3j, 1 - 2j, 4 + 1j])

# Define the x value for evaluation
x_value_complex = 1 + 1j

# Evaluate the polynomial at x = 1 + 1j using numpy.polyval
result_complex = np.polyval(coefficients_complex, x_value_complex)

print(f"Value of the complex polynomial at x = {x_value_complex}: {result_complex}")
```

The result of evaluating the polynomial with complex coefficients is −

```
Value of the complex polynomial at x = (1+1j): (1+4j)
```

## Polynomial Evaluation for Curve Plotting

Evaluating polynomials at a range of
**x**values is commonly used in curve plotting, such as when plotting the graph of a polynomial function. This can be done by passing an array of**x**values to the**numpy.polyval()**function and plotting the resulting values.
### Example: Plotting a Polynomial Curve

Let us plot the polynomial
**f(x) = 2x - 3x + 4**over the range of**x = -5**to**x = 5**−
```
import numpy as np
import matplotlib.pyplot as plt

# Define the coefficients of the polynomial 2x - 3x + 4
coefficients = np.array([2, -3, 4])

# Define the range of x values
x_values_plot = np.linspace(-5, 5, 100)

# Evaluate the polynomial at the x values using numpy.polyval
y_values_plot = np.polyval(coefficients, x_values_plot)

# Plot the polynomial curve
plt.plot(x_values_plot, y_values_plot, label='f(x) = 2x - 3x + 4')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Polynomial Curve Plot')
plt.legend()
plt.grid(True)
plt.show()
```

The output displayed is as shown below −
![Polynomial Evaluation](/numpy/images/polynomial_evaluation.jpg)

---

## 123. NumPy - Statistical Functions

*Source: [https://www.tutorialspoint.com/numpy/numpy_statistical_functions.htm](https://www.tutorialspoint.com/numpy/numpy_statistical_functions.htm)*

---

---
[Previous](/numpy/numpy_evaluating_polynomials.htm)[Quiz](/numpy/quiz_on_numpy_statistical_functions.htm)[Next](/numpy/numpy_descriptive_statistics.htm)
## Statistical Functions in NumPy

NumPy offers a wide range of statistical functions that allow you to perform various statistical calculations on arrays. These functions can calculate metrics such as mean, median, variance, standard deviation, minimum, maximum, and more.

## The NumPy amin() and amax() Functions

The numpy.amin() function returns the minimum from the elements in the given array along the specified axis. Whereas, the numpy.amax() function returns the maximum from the elements in the given array along the specified axis.

### Example

In the following example, we are demonstrating how to use the amin() and amax() functions on a NumPy array −

```
import numpy as np 
a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 

print('Our array is:') 
print(a)  
print('\n')  

print('Applying amin() function:') 
print(np.amin(a,1)) 
print('\n')  

print('Applying amin() function again:') 
print(np.amin(a,0)) 
print('\n')  

print('Applying amax() function:') 
print(np.amax(a)) 
print('\n')  

print('Applying amax() function again:') 
print(np.amax(a, axis=0))
```

It will produce the following output −

```
Our array is:
[[3 7 5]
 [8 4 3]
 [2 4 9]]
Applying amin() function:[3 3 2]
Applying amin() function again:[2 4 3]
Applying amax() function:9
Applying amax() function again:[8 7 9]
```

## The numpy.ptp() Function

The
**numpy.ptp()**function returns the range (maximum - minimum) of values along an axis.
### Example

In the following example, we are using the ptp() function to calculate the range of values in a NumPy array −

```
import numpy as np 
a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 

print('Our array is:') 
print(a) 
print('\n')  

print('Applying ptp() function:') 
print(np.ptp(a)) 
print('\n')  

print('Applying ptp() function along axis 1:') 
print(np.ptp(a, axis=1)) 
print('\n')   

print('Applying ptp() function along axis 0:')
print(np.ptp(a, axis=0))
```

Following is the output obtained −

```
Our array is:
[[3 7 5] 
 [8 4 3]
 [2 4 9]]
Applying ptp() function:7
Applying ptp() function along axis 1:[4 5 7]
Applying ptp() function along axis 0:[6 3 6]
```

## The numpy.percentile() Function

Percentile (or a centile) is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations fall.

The
**numpy.percentile()**function computes the q-th percentile of the data along the specified axis. It takes the following arguments −
```
numpy.percentile(a, q, axis)
```

Where,

- **a:**It is the input array.
- **q:**It is the percentile to compute and it must be between 0-100.
- **axis:**It is the axis along which the percentile is to be calculated.
### Example

In this example, we are calculating the 50th percentile (median) of a NumPy array using the percentile() function −

```
import numpy as np 
a = np.array([[30,40,70],[80,20,10],[50,90,60]]) 

print('Our array is:') 
print(a) 
print('\n')  

print('Applying percentile() function:') 
print(np.percentile(a,50)) 
print('\n')  

print('Applying percentile() function along axis 1:') 
print(np.percentile(a,50, axis=1)) 
print('\n')  

print('Applying percentile() function along axis 0:') 
print(np.percentile(a,50, axis=0))
```

This will produce the following result −

```
Our array is:
[[30 40 70]
 [80 20 10]
 [50 90 60]]
Applying percentile() function:50.0
Applying percentile() function along axis 1:[40. 20. 60.]
Applying percentile() function along axis 0:[50. 40. 60.]
```

## Sum and Product of Array Elements

The
**numpy.sum()**function calculates the sum of all elements in the array, while the**numpy.prod()**function calculates the product of all elements in the array.
### Example

In the following example, we are calculating the sum and product of the elements in a NumPy array using the sum() and prod() functions −

```
import numpy as np 

# Create a NumPy array
data = np.array([1, 2, 3, 4])

# Calculate the sum of the array
sum_value = np.sum(data)

# Calculate the product of the array
prod_value = np.prod(data)

print(f"Sum of the array: {sum_value}")
print(f"Product of the array: {prod_value}")
```

It will produce the following output −

```
Sum of the array: 10
Product of the array: 24
```

## The numpy.median() Function

The
**numpy.median()**function computes the median along the specified axis. If no axis is specified, it computes the median of the flattened array. Median is defined as the value separating the higher half of a data sample from the lower half
### Example

In the following example, we are using the median() function to find the median of elements in a NumPy array −

```
import numpy as np 
a = np.array([[30,65,70],[80,95,10],[50,90,60]]) 

print('Our array is:') 
print(a) 
print('\n')  

print('Applying median() function:') 
print(np.median(a)) 
print('\n')  

print('Applying median() function along axis 0:') 
print(np.median(a, axis=0)) 
print('\n')  
 
print('Applying median() function along axis 1:') 
print(np.median(a, axis=1))
```

Following is the output of the above code −

```
Our array is:
[[30 65 70]
 [80 95 10]
 [50 90 60]]
Applying median() function:65.0
Applying median() function along axis 0:[50. 90. 60.]
Applying median() function along axis 1:[65. 80. 60.]
```

## The numpy.mean() Function

The
**numpy.mean()**function returns the arithmetic mean of elements in the array. If no axis is specified, it computes the mean of the flattened array. Arithmetic mean is the sum of elements along an axis divided by the number of elements.
### Example

In the following example, we are calculating the mean of elements in a NumPy array using the mean() function −

```
import numpy as np 
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

print('Our array is:') 
print(a) 
print('\n')  

print('Applying mean() function:') 
print(np.mean(a)) 
print('\n')  

print('Applying mean() function along axis 0:') 
print(np.mean(a, axis=0)) 
print('\n')  

print('Applying mean() function along axis 1:') 
print(np.mean(a, axis=1))
```

The output obtained is as shown below −

```
Our array is:
[[1 2 3]
 [3 4 5]
 [4 5 6]]
Applying mean() function:3.6666666666666665
Applying mean() function along axis 0:[2.66666667 3.66666667 4.66666667]
Applying mean() function along axis 1:[2. 4. 5.]
```

## The numpy.average() Function

The
**numpy.average()**function computes the weighted average of elements in an array according to their respective weight. Weighted average is an average resulting from the multiplication of each component by a factor.
### Example

In the example below, we are calculating the average of elements in a NumPy array using the average() function with and without weights −

```
import numpy as np 
a = np.array([1,2,3,4]) 

print('Our array is:') 
print(a) 
print('\n')  

print('Applying average() function without weights:') 
print(np.average(a)) 
print('\n')  

wts = np.array([4,3,2,1]) 

print('Applying average() function with weights:') 
print(np.average(a,weights=wts)) 
print('\n')  

print('Sum of weights') 
print(np.average([1,2,3,4],weights=[4,3,2,1], returned=True))
```

It will produce the following output −

```
Our array is:[1 2 3 4]
Applying average() function without weights:2.5
Applying average() function with weights:2.0
Sum of weights(2.0, 10.0)
```

## The numpy.std() Function

The
**numpy.std()**function returns the standard deviation of elements in the array. The**standard deviation**is the square root of the average of squared deviations from the mean, i.e., std = sqrt(mean(abs(x - x.mean())**2)).
### Example

In the following example, we are using the std() function to calculate the standard deviation of a NumPy array −

```
import numpy as np 
print(np.std([1,2,3,4]))
```

After executing the above code, we get the following output −

```
1.118033988749895
```

## The numpy.var() Function

The
**numpy.var()**function returns the variance of elements in the array. The**variance**is the average of squared deviations, i.e., var = mean(abs(x - x.mean())**2).
### Example

In the following example, we are using the var() function to calculate the variance of a NumPy array −

```
import numpy as np 
print(np.var([1,2,3,4]))
```

We get the output as shown below −

```
1.25
```

## Correlation Coefficient

The
**numpy.corrcoef()**function returns the Pearson correlation coefficients of the input array. It is useful for determining the relationship between two variables.
### Example

In the following example, we are calculating the correlation coefficient matrix for two arrays using the corrcoef() function −

```
import numpy as np 

# Define two arrays
data1 = np.array([1, 2, 3, 4, 5])
data2 = np.array([5, 4, 3, 2, 1])

# Calculate the correlation coefficient
corr_coef = np.corrcoef(data1, data2)

print("Correlation Coefficient Matrix:")
print(corr_coef)
```

The result is as shown below −

```
Correlation Coefficient Matrix:
[[ 1. -1.]
 [-1.  1.]]
```

## Statistical Funtions

Following are the different Statistical function in Numpy −
Sr.No.Functions & Description1[amin()](/numpy/numpy_amin_function.htm)
Return the minimum of an array or minimum along an axis
2[amax()](/numpy/numpy_amax_function.htm)
Return the maximum of an array or maximum along an axis
3[nanmin()](/numpy/numpy_nanmin_function.htm)
Return minimum of an array or minimum along an axis, ignoring any NaNs
4[nanmax()](/numpy/numpy_nanmax_function.htm)
Return the maximum of an array or maximum along an axis, ignoring any NaNs
5[ptp()](/numpy/numpy_ptp_function.htm)
Range of values (maximum - minimum) along an axis
6[percentile()](/numpy/numpy_percentile_function.htm)
Compute the q-th percentile of the data along the specified axis
7[nanpercentile()](/numpy/numpy_nanpercentile_function.htm)
Compute the qth percentile of the data along the specified axis, while ignoring nan values
8[quantile()](/numpy/numpy_quantile_function.htm)
Compute the q-th quantile of the data along the specified axis
9[nanquantile()](/numpy/numpy_nanquantile_function.htm)
Compute the qth quantile of the data along the specified axis, while ignoring nan values. Returns the qth quantile(s) of the array elements
10[median()](/numpy/numpy_median_function.htm)
Compute the median along the specified axis
11[average()](/numpy/numpy_average_function.htm)
Compute the weighted average along the specified axis
12[mean()](/numpy/numpy_mean_function.htm)
Compute the arithmetic mean along the specified axis
13[std()](/numpy/numpy_std_function.htm)
Compute the standard deviation along the specified axis
14[var()](/numpy/numpy_var_function.htm)
Compute the variance along the specified axis
15[nanmean()](/numpy/numpy_nanmean_function.htm)
Compute the arithmetic mean along the specified axis, ignoring NaNs
16[nanstd()](/numpy/numpy_nanstd_function.htm)
Compute the standard deviation along the specified axis, while ignoring NaNs
17[nanvar()](/numpy/numpy_nanvar_function.htm)
Compute the variance along the specified axis, while ignoring NaNs

---

## 124. NumPy - Descriptive Statistics

*Source: [https://www.tutorialspoint.com/numpy/numpy_descriptive_statistics.htm](https://www.tutorialspoint.com/numpy/numpy_descriptive_statistics.htm)*

---

---
[Previous](/numpy/numpy_statistical_functions.htm)[Quiz](/numpy/quiz_on_numpy_descriptive_statistics.htm)[Next](/numpy/numpy_basics_of_date_and_time.htm)
## Descriptive Statistics in NumPy

Descriptive statistics in NumPy refers to summarizing and understanding the main features of a dataset through various statistical measures. It includes operations like calculating the mean (average), median, standard deviation, variance, and percentiles.

NumPy provides functions like numpy.mean(), numpy.median(), numpy.std(), and numpy.percentile() to quickly calculate these statistics, helping you understand the central tendency, spread, and distribution of the data.

## The NumPy mean() Function

The
**numpy.mean()**function calculates the arithmetic mean of the elements along the specified axis. If no axis is specified, it computes the mean of the flattened array.
The mean is a measure of central tendency, representing the average of all the values in the dataset.

### Example: Calculating the Mean

In the following example, we are calculating the mean of an array of numbers using the
**numpy.mean()**function −
```
import numpy as np

# Define an array
data = np.array([1, 2, 3, 4, 5])

# Calculate the mean of the array
mean_value = np.mean(data)

print(f"Mean of the array: {mean_value}")
```

Following is the output obtained −

```
Mean of the array: 3.0
```

## The NumPy median() Function

The
**numpy.median()**function computes the median of the elements along the specified axis. If no axis is specified, it computes the median of the flattened array.
The median is the middle value in a sorted dataset and is useful when dealing with skewed distributions.

### Example: Calculating the Median

In the following example, we are calculating the median of an array using the
**numpy.median()**function −
```
import numpy as np

# Define an array
data = np.array([1, 2, 3, 4, 5])

# Calculate the median of the array
median_value = np.median(data)

print(f"Median of the array: {median_value}")
```

This will produce the following result −

```
Median of the array: 3.0
```

## Finding the Mode of a Dataset

NumPy does not have a direct function to compute the mode. However, you can use the
**scipy.stats.mode()**function from the SciPy library to calculate the mode. The mode represents the most frequent value in a dataset.
### Example: Calculating the Mode

In this example, we are using the
**scipy.stats.mode()**function to find the mode of the array −
```
import numpy as np
from scipy import stats

data = np.array([1, 2, 3, 4, 5])
# Calculate the mode of the array
mode_value = stats.mode(data)

print(f"Mode of the array: {mode_value.mode[0]}")
```

Following is the output of the above code −

```
/home/cg/root/6745741fe1e0a/main.py:6: FutureWarning: Unlike other reduction functions (e.g. 'skew', 'kurtosis'), the default behavior of 'mode' typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of 'keepdims' will become False, the 'axis' over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set 'keepdims' to True or False to avoid this warning.
  mode_value = stats.mode(data)
Mode of the array: 1
```

## The NumPy var() Function

The
**numpy.var()**function calculates the variance of the elements along the specified axis. Variance measures the spread of the data points.
Variance indicates how far the data points are from the mean, providing a measure of the data's dispersion.

### Example: Calculating the Variance

In the example below, we are calculating the variance of an array using the
**numpy.var()**function −
```
import numpy as np

# Define an array
data = np.array([1, 2, 3, 4, 5])

# Calculate the variance of the array
variance_value = np.var(data)

print(f"Variance of the array: {variance_value}")
```

The output obtained is as shown below −

```
Variance of the array: 2.0
```

## The NumPy std() Function

The
**numpy.std()**function computes the standard deviation of the elements along the specified axis. Standard deviation is the square root of the variance and provides a measure of the dispersion of the data points.
### Example: Calculating the Standard Deviation

In this example, we are calculating the standard deviation of an array using the
**numpy.std()**function −
```
import numpy as np

# Define an array
data = np.array([1, 2, 3, 4, 5])

# Calculate the standard deviation of the array
std_value = np.std(data)

print(f"Standard Deviation of the array: {std_value}")
```

After executing the above code, we get the following output −

```
Standard Deviation of the array: 1.4142135623730951
```

## Finding the Minimum and Maximum Values

The
**numpy.min()**and**numpy.max()**functions return the minimum and maximum values in the array, respectively. The minimum value is the smallest data point, and the maximum value is the largest data point in the array.
### Example: Finding the Minimum and Maximum Values

In the following example, we are calculating the minimum and maximum values of an array using the
**numpy.min()**and**numpy.max()**functions −
```
import numpy as np

# Define an array
data = np.array([1, 2, 3, 4, 5])

# Calculate the minimum and maximum of the array
min_value = np.min(data)
max_value = np.max(data)

print(f"Minimum of the array: {min_value}")
print(f"Maximum of the array: {max_value}")
```

The result produced is as follows −

```
Minimum of the array: 1
Maximum of the array: 5
```

## Calculating the Range of the Dataset

The range of a dataset is the difference between the maximum and minimum values. You can calculate it using
**numpy.ptp()**function. The range gives an indication of how spread out the values are in the dataset.
### Example: Calculating the Range

In this example, we are calculating the range of the array using the
**numpy.ptp()**function −
```
import numpy as np

# Define an array
data = np.array([1, 2, 3, 4, 5])

# Calculate the range of the array
range_value = np.ptp(data)

print(f"Range of the array: {range_value}")
```

We get the output as shown below −

```
Range of the array: 4
```

## Calculating Percentiles

The
**numpy.percentile()**function computes the q-th percentile of the data along the specified axis. Percentiles divide the dataset into 100 equal parts, helping us understand the distribution of the data.
### Example

In the following example, we are calculating the 25th, 50th (median), and 75th percentiles of an array using the
**numpy.percentile()**function −
```
import numpy as np

# Define an array
data = np.array([1, 2, 3, 4, 5])

# Calculate the 25th, 50th, and 75th percentiles
percentile_25 = np.percentile(data, 25)
percentile_50 = np.percentile(data, 50)
percentile_75 = np.percentile(data, 75)

print(f"25th percentile: {percentile_25}")
print(f"50th percentile (median): {percentile_50}")
print(f"75th percentile: {percentile_75}")
```

The results are:

```
25th percentile: 2.0
50th percentile (median): 3.0
75th percentile: 4.0
```

## Interquartile Range (IQR) Calculation

The Interquartile Range (IQR) is the range between the 75th percentile and the 25th percentile. It measures the spread of the middle 50% of the data. The IQR is a useful measure to understand the variability within the central 50% of the data.

### Example: Calculating the Interquartile Range (IQR)

In the following example, we are calculating the Interquartile Range (IQR) of an array by subtracting the 25th percentile from the 75th percentile −

```
import numpy as np

# Define an array
data = np.array([1, 2, 3, 4, 5])

# Calculate the interquartile range
iqr_value = np.percentile(data, 75) - np.percentile(data, 25)

print(f"Interquartile Range (IQR): {iqr_value}")
```

Following is the output obtained −

```
Interquartile Range (IQR): 2.0
```

---

## 125. NumPy - Basics of Dates and Times

*Source: [https://www.tutorialspoint.com/numpy/numpy_basics_of_date_and_time.htm](https://www.tutorialspoint.com/numpy/numpy_basics_of_date_and_time.htm)*

---

---
[Previous](/numpy/numpy_descriptive_statistics.htm)[Quiz](/numpy/quiz_on_numpy_basics_of_date_and_time.htm)[Next](/numpy/numpy_representing_date_time.htm)
## Dates and Times in NumPy

Dates and times in NumPy refer to handling and manipulating date and time data within arrays. NumPy provides the
**datetime64**and**timedelta64**data types for working with dates and times.
These types allow you to perform operations like addition, subtraction, and comparison of dates and times, as well as converting between different time units (e.g., days, hours, minutes).

## The NumPy datetime64 Data Type

The
**numpy.datetime64**data type is used to represent dates and times. It provides various units of time such as years, months, days, hours, minutes, and seconds. This data type allows for precise representation and manipulation of date and time data.
The datetime64 data type allows for flexible representation of dates and times with varying levels of precision.

### Example: Creating datetime64 Objects

In the following example, we are creating datetime64 objects using different units of time −

```
import numpy as np

# Create datetime64 objects
date1 = np.datetime64('2023-01-01')
date2 = np.datetime64('2023-01-01 12:30')
date3 = np.datetime64('2023-01-01 12:30:45')

print(date1)
print(date2)
print(date3)
```

Following is the output obtained −

```
2023-01-01
2023-01-01T12:30
2023-01-01T12:30:45
```

## Creating Arrays of datetime64

You can create arrays of datetime64 objects using the
**numpy.array()**function. This allows for storage and manipulation of multiple date and time values.
Arrays of datetime64 objects are useful for performing vectorized operations on date and time data.

### Example: Creating Arrays of datetime64

In this example, we are creating an array of datetime64 objects in NumPy −

```
import numpy as np

# Create an array of datetime64 objects
dates = np.array(['2023-01-01', '2023-02-01', '2023-03-01'], dtype='datetime64')

print(dates)
```

This will produce the following result −

```
['2023-01-01' '2023-02-01' '2023-03-01']
```

## Date Arithmetic with datetime64

NumPy allows for easy arithmetic operations with datetime64 objects, including addition and subtraction of time units.

You can add or subtract time units such as days, months, or years to manipulate date and time values.

### Example: Adding and Subtracting Time Units

In the following example, we are performing arithmetic operations on datetime64 objects −

```
import numpy as np
import datetime

# Define the initial date
date = np.datetime64('2023-01-01')

# Add 10 days to the initial date
date_plus_10_days = date + np.timedelta64(10, 'D')

# Subtract 1 month from the initial date by converting to datetime and using a timedelta
date_as_datetime = date.astype(datetime.datetime)
# Approximate a month as 30 days
date_minus_1_month = date_as_datetime - datetime.timedelta(days=30)  

print(date_plus_10_days)
print(np.datetime64(date_minus_1_month))
```

Following is the output of the above code −

```
2023-01-11
2022-12-02
```

## Comparing datetime64 Objects

In NumPy, you can use comparison operators with datetime64 objects to easily compare dates and times. These operators allow you to check whether one date is earlier, later, or the same as another date.

Following are the comparison operators for datetime64 data type −

- **Equality (==):**Checks if two dates are exactly the same.
- **Inequality (!=):**Checks if two dates are different.
- **Less than (<):**Checks if the first date is earlier than the second date.
- **Less than or equal to (<=):**Checks if the first date is earlier than or exactly the same as the second date.
- **Greater than (>):**Checks if the first date is later than the second date.
- **Greater than or equal to (>=):**Checks if the first date is later than or exactly the same as the second date.
### Example: Comparing Dates

In this example, we are comparing datetime64 objects using the less than and greater than comparison operators −

```
import numpy as np

# Comparing datetime64 objects
date1 = np.datetime64('2023-01-01')
date2 = np.datetime64('2023-02-01')

is_earlier = date1 < date2
is_later = date1 > date2

print(is_earlier)
print(is_later)
```

The output obtained is as shown below −

```
True
False
```

## Converting between datetime64 and timedelta64

NumPy allows you to convert between 'datetime64' and 'timedelta64' objects. This makes it easy to calculate time intervals and durations. For example, you can add or subtract days, months, or years from a specific date or find out the difference between two dates.

### Example: Conversion between datetime64 and timedelta64

In this example, we are converting datetime64 objects to timedelta64 objects and vice versa −

```
import numpy as np

# Converting datetime64 to timedelta64
start_date = np.datetime64('2023-01-01')
end_date = np.datetime64('2023-02-01')
duration = end_date - start_date

print(duration)

# Converting timedelta64 to datetime64
new_date = start_date + duration

print(new_date)
```

After executing the above code, we get the following output −

```
31 days
2023-02-01
```

## Working with Time Units

NumPy supports various time units for datetime64 and timedelta64, including years, months, weeks, days, hours, minutes, and seconds. Using appropriate time units ensures accurate representation and manipulation of date and time data.

### Example: Using Different Time Units

In this example, we are demonstrating the use of different time units with datetime64 and timedelta64 objects −

```
import numpy as np

# Using different time units
date_year = np.datetime64('2023', 'Y')
date_month = np.datetime64('2023-01', 'M')
date_week = np.datetime64('2023-01-01', 'W')

print(date_year)
print(date_month)
print(date_week)
```

We get the output as shown below −

```
2023
2023-01
2022-12-29
```

---

## 126. NumPy - Representing Dates and Times

*Source: [https://www.tutorialspoint.com/numpy/numpy_representing_date_time.htm](https://www.tutorialspoint.com/numpy/numpy_representing_date_time.htm)*

---

---

## 127. NumPy - Date and Time Arithmetic

*Source: [https://www.tutorialspoint.com/numpy/numpy_date_time_arithmetic.htm](https://www.tutorialspoint.com/numpy/numpy_date_time_arithmetic.htm)*

---

---
[Previous](/numpy/numpy_representing_date_time.htm)[Quiz](/numpy/quiz_on_numpy_date_time_arithmetic.htm)[Next](/numpy/numpy_indexing_with_datetime.htm)
## Date and Time Arithmetic in NumPy

Date and time arithmetic in NumPy refers to performing operations like adding or subtracting time from dates, or calculating the difference between two dates.

NumPy provides the
**datetime64**and**timedelta64**data types to perform these operations. For example, you can add days to a specific date or find how many days are between two dates.
This makes it easy to manipulate time-related data for tasks such as scheduling, time series analysis, and event tracking.

## Adding and Subtracting Dates and Times

NumPy allows for the addition and subtraction of
**timedelta64**objects to and from**datetime64**objects. This makes it easy to calculate new dates and times based on specific intervals.
### Example: Adding Time Intervals

In the following example, we are adding various
**timedelta64**intervals to a**datetime64**object −
```
import numpy as np

# Define a base date
base_date = np.datetime64('2024-01-01')

# Add different time intervals
date_plus_10_days = base_date + np.timedelta64(10, 'D')

# Add months manually by changing to a monthly precision
date_plus_2_months = np.datetime64(base_date, 'M') + np.timedelta64(2, 'M')

# Add years manually by changing to a yearly precision
date_plus_5_years = np.datetime64(base_date, 'Y') + np.timedelta64(5, 'Y')

print(date_plus_10_days)
print(date_plus_2_months)
print(date_plus_5_years)
```

We get the output as shown below −

```
2024-01-11
2024-03
2029
```

### Example: Subtracting Time Intervals

In this example, we are subtracting various
**timedelta64**intervals from a**datetime64**object −
```
import numpy as np

# Define a base date
base_date = np.datetime64('2024-01-01')

# Subtract different time intervals
date_minus_10_days = base_date - np.timedelta64(10, 'D')

# Subtract months manually by changing to monthly precision
date_minus_2_months = np.datetime64(base_date, 'M') - np.timedelta64(2, 'M')

# Subtract years manually by changing to yearly precision
date_minus_5_years = np.datetime64(base_date, 'Y') - np.timedelta64(5, 'Y')

print(date_minus_10_days)
print(date_minus_2_months)
print(date_minus_5_years)
```

Following is the output obtained −

```
2023-12-22
2023-11
2019
```

## Calculating Differences Between Dates

NumPy allows you to calculate the differences between two
**datetime64**objects, resulting in**timedelta64**objects. This is useful for determining the duration between two dates or times.
### Example

In this example, we are calculating the difference between two
**datetime64**objects −
```
import numpy as np

# Define two dates
date1 = np.datetime64('2024-01-01')
date2 = np.datetime64('2024-12-31')

# Calculate the difference
difference = date2 - date1

print(difference)
```

This will produce the following result −

```
365 days
```

## Using Vectorized Operations

NumPy supports vectorized operations on arrays of
**datetime64**and**timedelta64**objects, which means you can perform calculations on entire arrays of dates and times at once, without needing to loop through them one by one.
Vectorized operations allow you to apply the same operation across all elements in the array simultaneously, which is much quicker than processing each element individually.

### Example: Vectorized Date Arithmetic

In this example, we are performing vectorized addition of
**timedelta64**intervals to an array of**datetime64**objects −
```
import numpy as np

# Define an array of dates
dates = np.array(['2024-01-01', '2024-06-01', '2024-12-01'], dtype='datetime64[D]')

# Add 10 days to each date
new_dates = dates + np.timedelta64(10, 'D')

print(new_dates)
```

Following is the output of the above code −

```
['2024-01-11' '2024-06-11' '2024-12-11']
```

## Combine datetime64 and timedelta64 in Operations

NumPy allows you to combine
**datetime64**and**timedelta64**objects in arithmetic operations, making it easy to perform complex date and time calculations.
You can add or subtract time durations to/from specific dates, or even calculate the difference between two dates. This capability allows for easy manipulation of dates and times, supporting various temporal operations like shifting dates by a specific interval or calculating time differences.

### Example

In this example, we are performing multiple arithmetic operations combining
**datetime64**and**timedelta64**objects −
```
import numpy as np

# Define a base date
base_date = np.datetime64('2024-01-01')

# Perform complex date arithmetic
# Add 5 years
new_date = np.datetime64(base_date, 'Y') + np.timedelta64(5, 'Y')  
# Subtract 3 months
new_date = np.datetime64(new_date, 'M') - np.timedelta64(3, 'M')  
# Add 15 days
new_date = new_date + np.timedelta64(15, 'D')  

print(new_date)
```

The output obtained is as shown below −

```
2028-10-16
```

---

## 128. NumPy - Indexing with Datetimes

*Source: [https://www.tutorialspoint.com/numpy/numpy_indexing_with_datetime.htm](https://www.tutorialspoint.com/numpy/numpy_indexing_with_datetime.htm)*

---

---
[Previous](/numpy/numpy_date_time_arithmetic.htm)[Quiz](/numpy/quiz_on_numpy_indexing_with_datetime.htm)[Next](/numpy/numpy_time_zone_handling.htm)
## Indexing with Datetimes in NumPy

Indexing with datetimes in NumPy allows you to easily select and manipulate specific time-based data. This is helpful when dealing with time series data, like stock prices or temperature readings.

Using the
**datetime64**type in NumPy, you can slice, filter, and index data just like arrays. This allows you to focus on specific time periods, such as a particular day or range of dates, and perform operations like comparing or filtering dates for analysis.
## Basic Indexing with Datetime Arrays

Indexing and slicing with datetime arrays in NumPy allow you to easily access specific dates or ranges of dates. You can index a single date from a datetime array by specifying its position, just like with regular arrays.

For slicing, you can select a continuous range of dates by providing a start and end index. Additionally, NumPy supports boolean indexing, which allows you to filter dates based on conditions (e.g., selecting all dates after a specific day).

### Example

In the following example, we are slicing a
**datetime64**array to select specific ranges of dates −
```
import numpy as np

# Define a datetime array
dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'], dtype='datetime64[D]')

# Slice the datetime array
selected_dates = dates[1:4]

print(selected_dates)
```

This produces the following output −

```
['2024-01-02' '2024-01-03' '2024-01-04']
```

## Filtering with Boolean Indexing

Boolean indexing in NumPy allows you to filter elements in an array based on conditions. When working with datetime arrays, this feature is useful for selecting data within specific time ranges or satisfying certain time-based criteria.

To perform boolean indexing, you create a condition (a boolean array) that matches the structure of the original datetime array. The condition can be any logical expression that compares dates (or other data), and it will return an array of True or False values. These True values are then used to filter out the corresponding elements from the original array.

### Example

In this example, we are filtering a datetime array to select only the dates after a specific date, using boolean indexing −

```
import numpy as np

# Define a datetime array
dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'], dtype='datetime64[D]')

# Define the filter condition
filtered_dates = dates[dates > np.datetime64('2024-01-02')]

print(filtered_dates)
```

The output for this operation will be −

```
['2024-01-03' '2024-01-04' '2024-01-05']
```

## Indexing with Date Ranges

Indexing with date ranges in NumPy allows you to select and work with subsets of datetime data that fall within specific time intervals.

To index with date ranges, you define a condition that specifies the start and end of the range you are interested in. This can be done using comparison operators to filter dates that fall within the desired range. You can combine conditions using logical operators to filter data more precisely.

### Example

In this example, we are selecting data within a specific date range −

```
import numpy as np

# Define a datetime array
dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'], dtype='datetime64[D]')

# Define the start and end dates
start_date = np.datetime64('2024-01-02')
end_date = np.datetime64('2024-01-04')

# Select dates within the range
range_dates = dates[(dates >= start_date) & (dates <= end_date)]

print(range_dates)
```

The result produced is as follows −

```
['2024-01-02' '2024-01-03' '2024-01-04']
```

## Working with Different Time Units

In NumPy, the datetime64 and timedelta64 objects allow you to work with various time units ranging from years down to attoseconds. This helps for the precise manipulation and analysis of time data in different scales, such as days, hours, minutes, and even smaller units like nanoseconds or femtoseconds.

The time units in NumPy are represented by strings, such as 'Y' for years, 'M' for months, 'D' for days, 'h' for hours, 'm' for minutes, 's' for seconds, 'ms' for milliseconds, 'us' for microseconds, and 'ns' for nanoseconds. You can use these units to create datetime and timedelta objects or perform arithmetic operations involving time intervals.

### Example

In this example, we are indexing the
**datetime64**array to select dates within a specific month −
```
import numpy as np

# Define a datetime array
dates = np.array(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01'], dtype='datetime64[M]')

# Filter dates by the month of January
january_dates = dates[dates == np.datetime64('2024-01', 'M')]

print(january_dates)
```

After executing the above code, we get the following output −

```
['2024-01']
```

## Advanced Indexing with Structured Arrays

Structured arrays in NumPy allow you to store and manipulate complex data, such as records with multiple fields, each of which can be of a different type.

Advanced indexing techniques helps you to access and modify specific fields or subsets of the data. Structured arrays are created using the
**np.array()**function with a**dtype**argument that specifies the names and types of the fields.
> A structured array in NumPy is similar to a regular array, but it allows each element to have multiple fields, each with its own data type. These fields can represent different types of data, such as integers, floats, or strings, all organized under a single array.

### Example

In this example, we create a structured array and index it by date, selecting specific records based on the datetime values −

```
import numpy as np

# Define a structured array with dates and associated data
data = np.array([('2024-01-01', 100), ('2024-01-02', 200), ('2024-01-03', 300)],
                dtype=[('date', 'datetime64[D]'), ('value', 'i4')])

# Filter data where the date is after '2024-01-01'
filtered_data = data[data['date'] > np.datetime64('2024-01-01')]

print(filtered_data)
```

The output will be −

```
[('2024-01-02', 200) ('2024-01-03', 300)]
```

---

## 129. NumPy - Time Zone Handling

*Source: [https://www.tutorialspoint.com/numpy/numpy_time_zone_handling.htm](https://www.tutorialspoint.com/numpy/numpy_time_zone_handling.htm)*

---

---
[Previous](/numpy/numpy_indexing_with_datetime.htm)[Quiz](/numpy/quiz_on_numpy_time_zone_handling.htm)[Next](/numpy/numpy_time_series_analysis.htm)
## Time Zone Handling in NumPy

Time zone handling in NumPy allows you to manage datetime data across different time zones. This is important when working with global time-based data, such as stock market timestamps, weather data, or international events, where time zone conversions are required for accurate analysis.

While NumPy doesn't have built-in support for time zones like some other libraries (e.g., 'pytz' or 'pandas'), it does allow you to work with datetime objects, and you can use external libraries like 'pytz' for time zone conversion.

NumPy's
**datetime64**objects are timezone-naive, meaning they don't store time zone information, but you can manually adjust them using compatible tools.
## Handling Time Zones with 'datetime64'

By default, the 'datetime64' objects in NumPy are time zone-naive, which means they don't store time zone information. If you need to handle time zones explicitly, you need to combine NumPy with libraries like 'pytz' or 'timezone' from the 'datetime' module in Python.

To convert time zones or handle time zone-aware datetime data, you can use external tools in combination with NumPy arrays to manage the conversion and alignment of datetimes across different time zones.

### Example

In the following example, we will combine NumPy with the 'pytz' library to handle time zone conversions. First, we will create a datetime array using NumPy and then adjust the datetime to a different time zone −

```
import numpy as np
import pytz
from datetime import datetime

# Define a datetime array
dates = np.array(['2024-01-01T12:00', '2024-01-02T12:00'], dtype='datetime64[m]')

# Convert to Python datetime objects
# Use astype('O') to convert datetime64 to datetime object
dt_objects = [d.item() for d in dates]

# Define the timezone using pytz
timezone = pytz.timezone('US/Eastern')

# Convert each datetime to the new time zone
localized_dates = [timezone.localize(dt) for dt in dt_objects]

# Print the localized times
for date in localized_dates:
   print(date.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
```

The output will display the datetime values converted to the Eastern Time Zone (ET) −

```
2024-01-01 12:00:00 EST-0500
2024-01-02 12:00:00 EST-0500
```

## Converting Between Time Zones

When working with datetime data, converting between time zones is often necessary. This can be done using external libraries like 'pytz'. After creating a time zone-aware datetime object, you can easily convert it to another time zone by using the
**astimezone()**method.
### Example

In the following example, we will first convert a datetime object to Eastern Standard Time (EST), and then convert it to Pacific Standard Time (PST) −

```
import pytz
from datetime import datetime

# Create a datetime object
dt = datetime(2024, 1, 1, 12, 0, 0)

# Define the timezone (Eastern Time Zone)
eastern = pytz.timezone('US/Eastern')

# Localize the datetime to Eastern Time
localized_dt = eastern.localize(dt)

# Convert to Pacific Time Zone
pacific = pytz.timezone('US/Pacific')
pacific_dt = localized_dt.astimezone(pacific)

print("Eastern Time: ", localized_dt.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
print("Pacific Time: ", pacific_dt.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
```

After executing the above code, you will get the following output −

```
Eastern Time:  2024-01-01 12:00:00 EST-0500
Pacific Time:  2024-01-01 09:00:00 PST-0800
```

## Aligning Time Zones in Data

When working with multiple time zones in data, it is important to align datetime data across different time zones. You may need to convert all datetimes to a common time zone (e.g., UTC) to ensure accurate comparisons and calculations.

You can align time zone-aware datetimes to a common time zone by converting each datetime object to that time zone. This can be done using the same 'pytz' library or the 'datetime' module for adjustments.

### Example

In this example, we will align two datetime objects from different time zones to UTC −

```
import pytz
from datetime import datetime

# Create two datetime objects in different time zones
dt1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
dt2 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=pytz.timezone('US/Pacific'))

# Convert both datetimes to UTC
dt1_utc = dt1.astimezone(pytz.utc)
dt2_utc = dt2.astimezone(pytz.utc)

# Print both datetime objects in UTC
print("Eastern to UTC: ", dt1_utc.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
print("Pacific to UTC: ", dt2_utc.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
```

The output will show both datetimes aligned to UTC −

```
Eastern to UTC:  2024-01-01 16:56:00 UTC+0000
Pacific to UTC:  2024-01-01 19:53:00 UTC+0000
```

---

## 130. NumPy - Time Series Analysis

*Source: [https://www.tutorialspoint.com/numpy/numpy_time_series_analysis.htm](https://www.tutorialspoint.com/numpy/numpy_time_series_analysis.htm)*

---

---

## 131. NumPy - Working with Time Deltas

*Source: [https://www.tutorialspoint.com/numpy/numpy_working_with_time_deltas.htm](https://www.tutorialspoint.com/numpy/numpy_working_with_time_deltas.htm)*

---

---
[Previous](/numpy/numpy_time_series_analysis.htm)[Quiz](/numpy/quiz_on_numpy_working_with_time_deltas.htm)[Next](/numpy/numpy_handling_leap_seconds.htm)
## Working with Time Deltas in NumPy

Time deltas represent the difference between two datetime objects and are useful when working with time-based data.

In NumPy, time deltas are handled using the
**timedelta64**data type, which allows you to perform various operations involving differences between dates and times.
This chapter will explore how to work with time deltas in NumPy, including creating time deltas, performing arithmetic operations, and using them to manipulate datetime objects.

## Creating Time Deltas with NumPy

In NumPy, time deltas are represented by the
**timedelta64**data type. You can create time deltas by specifying a duration and a unit (e.g., days, hours, minutes).
Time deltas can be used to calculate the difference between two datetime objects or to add/subtract from a datetime.

### Example

In this example, we will create a time delta of 5 days and 10 hours −

```
import numpy as np

# Create a time delta of 5 days and 10 hours
time_delta = np.timedelta64(5, 'D') + np.timedelta64(10, 'h')

print("Time Delta:", time_delta)
```

The output will show the time delta as a combination of days and hours −

```
Time Delta: 130 hours
```

## Adding Time Deltas to DateTimes

Time deltas can be added or subtracted from datetime objects to adjust the date or time. This is useful when you want to shift a timestamp by a certain duration, for example, adding days or subtracting hours from a specific date.

### Example

In this example, we will add a time delta of 5 days and 10 hours to a specific date −

```
import numpy as np

# Create a datetime object
start_date = np.datetime64('2024-01-01')

# Create a time delta of 5 days and 10 hours
time_delta = np.timedelta64(5, 'D') + np.timedelta64(10, 'h')

# Add the time delta to the start date
new_date = start_date + time_delta

print("New Date:", new_date)
```

The output will show the new date after adding the time delta −

```
New Date: 2024-01-06T10
```

## Subtracting Time Deltas from DateTimes

Just as you can add time deltas to datetime objects, you can also subtract them to shift a datetime backward. Subtracting time deltas is often used to calculate past dates or times.

### Example

In this example, we will subtract a time delta of 5 days and 10 hours from a specific date −

```
import numpy as np

# Create a datetime object
start_date = np.datetime64('2024-01-06T10:00')

# Create a time delta of 5 days and 10 hours
time_delta = np.timedelta64(5, 'D') + np.timedelta64(10, 'h')

# Subtract the time delta from the start date
new_date = start_date - time_delta

print("New Date after Subtraction:", new_date)
```

The output will show the new date after subtracting the time delta −

```
New Date after Subtraction: 2024-01-01T00:00
```

## Comparing DateTimes with Time Deltas

You can compare datetime objects that have been modified with time deltas. This allows you to check whether one datetime is before or after another, or if two datetime objects are the same, given a time delta.

### Example

In this example, we will compare two datetime objects with different time deltas to determine if one is earlier than the other −

```
import numpy as np

# Create two datetime objects
date1 = np.datetime64('2024-01-01')
date2 = np.datetime64('2024-01-05')

# Create a time delta of 5 days
time_delta = np.timedelta64(5, 'D')

# Check if date1 is earlier than date2 after adding the time delta
new_date1 = date1 + time_delta
print("Is date1 + time_delta earlier than date2?", new_date1 < date2)
```

The output will indicate whether the new date1 is earlier than date2 −

```
Is date1 + time_delta earlier than date2? False
```

## Time Delta Arithmetic

Time deltas support arithmetic operations such as addition, subtraction, multiplication, and division. You can scale time deltas by multiplying them by an integer, or divide them by a scalar to get smaller time units.

### Example

In this example, we will multiply a time delta by 2 and divide it by 2 to see the resulting time deltas −

```
import numpy as np

# Create a time delta of 5 days
time_delta = np.timedelta64(5, 'D')

# Multiply the time delta by 2
double_delta = time_delta * 2

# Divide the time delta by 2
half_delta = time_delta / 2

print("Original Time Delta:", time_delta)
print("Time Delta * 2:", double_delta)
print("Time Delta / 2:", half_delta)
```

The output will show the time delta after multiplication and division −

```
Original Time Delta: 5 days
Time Delta * 2: 10 days
Time Delta / 2: 2 days
```

---

## 132. NumPy - Handling Leap Seconds

*Source: [https://www.tutorialspoint.com/numpy/numpy_handling_leap_seconds.htm](https://www.tutorialspoint.com/numpy/numpy_handling_leap_seconds.htm)*

---

---
[Previous](/numpy/numpy_working_with_time_deltas.htm)[Quiz](/numpy/quiz_on_numpy_handling_leap_seconds.htm)[Next](/numpy/numpy_vectorized_operations_with_datetime.htm)
## Handling Leap Seconds in NumPy

Leap seconds are an important concept in timekeeping, used to account for irregularities in the Earth's rotation. A leap second is added or subtracted occasionally to ensure that the timekeeping system remains in sync with the Earth's rotation.

NumPy's datetime functionality is limited when it comes to directly handling leap seconds, as it is based on the
**datetime64**data type, which is not capable of representing leap seconds by default.
In this tutorial, we will explore how NumPy handles timekeeping and provide workarounds for managing leap seconds in time-based data.

## Leap Seconds and the datetime64 Data Type

In NumPy, the
**datetime64**data type is used to represent date and time information. However, this data type operates on a fixed number of seconds per minute and does not take leap seconds into account. Therefore, leap seconds do not directly affect the handling of datetime values in NumPy.
Since leap seconds are added irregularly and typically at the end of a year, NumPy does not natively support leap second insertion or manipulation.

If you need to work with leap seconds, you will likely need to implement a custom approach or rely on external libraries like
**AstroPy**or**pytz**, which handle leap seconds in time zone conversions.
## Workaround to Handle Leap Seconds in NumPy

Although NumPy cannot directly handle leap seconds, you can simulate the handling of leap seconds using workarounds. This can be done by adjusting the time values manually based on external leap second tables or by ignoring leap seconds altogether if they are not critical for your application.

### Example: Adjusting Time for Leap Seconds

In this example, we simulate the addition of a leap second by manually adjusting the time. Let us assume that a leap second was added at the end of a particular year (e.g., 2024). We can manually adjust the datetime value by adding one second to the last minute of the year −

```
import numpy as np

# Define a datetime object for the last second of 2024
last_second_of_2024 = np.datetime64('2024-12-31T23:59:59')

# Simulate adding a leap second
leap_second_added = last_second_of_2024 + np.timedelta64(1, 's')

print("New Date after Leap Second:", leap_second_added)
```

The output will show the new date after adding the leap second −

```
New Date after Leap Second: 2025-01-01T00:00:00
```

## Using External Libraries to Handle Leap Seconds

As mentioned earlier, NumPy's datetime functionality does not natively handle leap seconds. However, you can use external libraries like
**AstroPy**or**pytz**to account for leap seconds when performing time zone conversions or dealing with astronomical data.
These libraries provide more sophisticated handling of leap seconds and allow you to adjust for them in your calculations.
**AstroPy**provides support for leap seconds by working with the**Time**class, which allows you to handle leap seconds in timekeeping systems. Similarly,**pytz**can handle leap seconds by converting time between different time zones while accounting for any adjustments.
### Example: Handling Leap Seconds with AstroPy

In this example, we will use
**AstroPy**to handle a leap second by adjusting the time from UTC −
```
from astropy.time import Time

# Define a time object in UTC with a leap second adjustment
time = Time('2024-12-31 23:59:59', scale='utc')

# Add a leap second
time_plus_leap = time + 1

print("Time after Leap Second Adjustment:", time_plus_leap)
```

The output will show the new time after adding a leap second −

```
Time after Leap Second Adjustment: 2025-01-01 00:00:00.000
```

## Considerations and Best Practices

When working with leap seconds, it is important to note that not all applications need to consider them. For most cases, especially where high precision is not required, ignoring leap seconds is an acceptable approach.

However, for scientific applications or systems that need precise synchronization with UTC, it is crucial to account for leap seconds using the right tools or external libraries.

### Best Practices for Leap Seconds

Following are the best practices for leap seconds −

- For applications where leap seconds are not critical, you can safely ignore them and use standard datetime operations in NumPy.
- For time-sensitive applications, consider using libraries like**AstroPy**or**pytz**to handle leap seconds.
- When working with external time data sources, ensure that leap second adjustments are accounted for in the raw data before processing.

---

## 133. NumPy - Vectorized Operations with Datetimes

*Source: [https://www.tutorialspoint.com/numpy/numpy_vectorized_operations_with_datetime.htm](https://www.tutorialspoint.com/numpy/numpy_vectorized_operations_with_datetime.htm)*

---

---
[Previous](/numpy/numpy_handling_leap_seconds.htm)[Quiz](/numpy/quiz_on_numpy_vectorized_operations_with_datetime.htm)[Next](/numpy/numpy_unfunc_introduction.htm)
## NumPy Vectorized Operations with Datetimes

Vectorized operations in NumPy allow you to perform operations on entire arrays of data without the need for explicit loops.

When dealing with datetime data, NumPy's vectorized operations enable you to perform time-based calculations across entire arrays of datetime values at once, without the need for manually iterating over each element.

Using the
**datetime64**type, you can perform various arithmetic and comparison operations across datetime arrays, such as adding or subtracting time intervals, comparing dates, or performing conditional operations.
## Adding or Subtracting Time Intervals

One of the most common operations with datetime data is adding or subtracting time intervals. NumPy allows you to perform these operations in a vectorized manner, meaning you can add or subtract time deltas from an entire array of datetime values at once.

To add or subtract a time interval, you use the
**timedelta64**object, which represents a time difference. This object can be added to or subtracted from a**datetime64**object to shift the date or time by the specified interval.
### Example

In this example, we are adding 5 days to each date in a datetime array −

```
import numpy as np

# Define a datetime array
dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[D]')

# Define a time delta of 5 days
time_delta = np.timedelta64(5, 'D')

# Add the time delta to the datetime array
new_dates = dates + time_delta

print(new_dates)
```

Following is the output obtained −

```
['2024-01-06' '2024-01-07' '2024-01-08']
```

## Subtracting Dates and Calculating Differences

Another common operation is calculating the difference between two dates, which results in a
**timedelta64**object. This is useful when you need to find the time difference between two points in time, such as the number of days between two dates.
In NumPy, you can subtract one datetime array from another to get an array of timedeltas, representing the difference between corresponding dates in the arrays.

### Example

In this example, we calculate the difference between two dates in a datetime array −

```
import numpy as np

# Define two datetime arrays
dates1 = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[D]')
dates2 = np.array(['2024-01-04', '2024-01-05', '2024-01-06'], dtype='datetime64[D]')

# Subtract the arrays to get the difference
time_diff = dates2 - dates1

print(time_diff)
```

The output will show the differences in days −

```
[3 3 3]
```

## Comparing Dates in a Vectorized Manner

NumPy allows you to perform element-wise comparisons between datetime arrays, enabling you to filter or analyze data based on time conditions. Vectorized comparison operations can be used to compare datetime values to a fixed point in time or to each other.

You can compare datetime arrays using standard comparison operators, such as
**>**,**<**,**>=**,**<=**,**==**, and**!=**, which return a boolean array indicating whether the condition is met for each element.
### Example

In this example, we filter dates that are greater than a specific date using vectorized comparison −

```
import numpy as np

# Define a datetime array
dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[D]')

# Define the filter condition (dates greater than '2024-01-02')
filtered_dates = dates[dates > np.datetime64('2024-01-02')]

print(filtered_dates)
```

This will produce the following result −

```
['2024-01-03']
```

## Vectorized Operations with Timedelta Arrays

In addition to working with datetime arrays, you can also perform vectorized operations with
**timedelta64**arrays, which represent differences between datetime values. These operations are useful when working with durations or intervals of time.
You can perform arithmetic operations, such as addition or subtraction, on timedelta arrays to calculate the total duration between multiple time intervals, or you can compare them to other time intervals.

### Example

In this example, we add two timedelta arrays to get the total duration −

```
import numpy as np

# Define two timedelta arrays
delta1 = np.array([np.timedelta64(5, 'D'), np.timedelta64(10, 'D')], dtype='timedelta64[D]')
delta2 = np.array([np.timedelta64(2, 'D'), np.timedelta64(3, 'D')], dtype='timedelta64[D]')

# Add the timedelta arrays
total_delta = delta1 + delta2

print(total_delta)
```

Following is the output of the above code −

```
[ 7 13]
```

## Working with Different Time Units

NumPy supports a variety of time units, including years, months, days, hours, minutes, and seconds. You can perform vectorized operations with datetime arrays across different time units, depending on your needs.

This is particularly useful when dealing with data that spans multiple time scales or when you need to convert between different units.

### Example

In this example, we work with a datetime array and a timedelta array with different time units −

```
import numpy as np

# Define a datetime array
dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[D]')

# Define a timedelta array with hours
timedelta = np.array([np.timedelta64(10, 'h'), np.timedelta64(5, 'h'), np.timedelta64(20, 'h')])

# Add the timedelta array to the datetime array
new_dates = dates + timedelta

print(new_dates)
```

After executing the above code, we get the following output −

```
['2024-01-01T10' '2024-01-02T05' '2024-01-03T20']
```

---

## 134. NumPy - ufunc Introduction

*Source: [https://www.tutorialspoint.com/numpy/numpy_unfunc_introduction.htm](https://www.tutorialspoint.com/numpy/numpy_unfunc_introduction.htm)*

---

---
[Previous](/numpy/numpy_vectorized_operations_with_datetime.htm)[Quiz](/numpy/quiz_on_numpy_unfunc_introduction.htm)[Next](/numpy/numpy_creating_ufunc.htm)
## NumPy Universal Functions (ufuncs)

Ufuncs, or universal functions, are functions in NumPy that apply operations element-wise on ndarrays. They act as vectorized wrappers for simple functions, meaning they can apply the same operation to each element in an array simultaneously, which is much faster than using traditional Python loops.

Common examples of ufuncs include basic arithmetic operations like addition, subtraction, multiplication, and division, as well as more complex functions like trigonometric, logarithmic, and exponential functions.

## Creating and Using ufuncs

NumPy provides a wide range of built-in ufuncs for performing common operations. You can also create custom ufuncs using the
**numpy.frompyfunc()**function, which allows you to turn a Python function into a ufunc. Let us explore how to use both built-in and custom ufuncs.
### Example: Using Built-in ufuncs

In this example, we use the built-in ufunc
**numpy.add()**to perform element-wise addition of two arrays −
```
import numpy as np

# Define two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Use the built-in ufunc numpy.add to add the arrays
result = np.add(a, b)

print(result)
```

Following is the output obtained −

```
[5 7 9]
```

### Example: Creating a Custom ufunc

In this example, we create a custom ufunc using the
**numpy.frompyfunc()**function. This ufunc will perform element-wise multiplication of two arrays −
```
import numpy as np

# Define a Python function for multiplication
def multiply(x, y):
   return x * y

# Create a custom ufunc from the Python function
multiply_ufunc = np.frompyfunc(multiply, 2, 1)

# Define two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Use the custom ufunc to multiply the arrays
result = multiply_ufunc(a, b)

print(result)
```

This will produce the following result −

```
[4 10 18]
```

## Advantages of Using ufuncs

Using ufuncs in NumPy provides several advantages over traditional loop-based approaches, they are −

- **Performance:**Ufuncs are highly optimized and can perform operations much faster than equivalent Python loops.
- **Vectorization:**Ufuncs operate on entire arrays at once, eliminating the need for explicit loops and reducing the likelihood of errors.
- **Broadcasting:**Ufuncs support broadcasting, allowing you to perform operations on arrays of different shapes in a flexible manner.
- **Flexibility:**Ufuncs can handle a wide range of operations, from simple arithmetic to complex mathematical functions.
## Broadcasting with ufuncs

Broadcasting is a feature of NumPy that allows ufuncs to operate on arrays of different shapes. When performing operations on arrays of different sizes, NumPy automatically broadcasts the smaller array across the larger array so that they have compatible shapes.

For example, you can add a scalar to an array, or add two arrays of different shapes, and NumPy will handle the broadcasting for you.

### Example

In this example, we add a scalar to an array using broadcasting in NumPy −

```
import numpy as np

# Define an array
a = np.array([1, 2, 3])

# Add a scalar to the array using broadcasting
result = np.add(a, 5)

print(result)
```

Following is the output of the above code −

```
[6 7 8]
```

## Commonly Used ufuncs in NumPy

NumPy provides a complete set of built-in ufuncs for performing various operations. Some of the most commonly used ufuncs are −

- **Arithmetic ufuncs:**np.add(), np.subtract(), np.multiply(), np.divide()
- **Trigonometric ufuncs:**np.sin(), np.cos(), np.tan(), np.arcsin(), np.arccos(), np.arctan()
- **Exponential and logarithmic ufuncs:**np.exp(), np.log(), np.log10(), np.log2()
- **Comparison ufuncs:**np.greater(), np.less(), np.equal(), np.not_equal()
### Example: Using Trigonometric ufuncs

In this example, we use the trigonometric ufunc
**numpy.sin()**to calculate the sine of each element in an array −
```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/2, np.pi])

# Use the trigonometric ufunc numpy.sin to calculate the sine of each angle
sine_values = np.sin(angles)

print(sine_values)
```

The output obtained is as shown below −

```
[0.0000000e+00 1.0000000e+00 1.2246468e-16]
```

### Example: Using Exponential ufuncs

Here, we use the exponential ufunc
**numpy.exp()**to calculate the exponential of all elements in the array −
```
import numpy as np

# array
arr = np.array([1, 2, 3])

# Calculate the exponential of each element
exp_result = np.exp(arr)

print("Exponential of each element:", exp_result)
```

After executing the above code, we get the following output −

```
Exponential of each element: [ 2.71828183  7.3890561  20.08553692]
```

### Example: Using Logarithmic Ufunc

Now, we use the logarithmic ufunc
**numpy.log()**to calculate the natural logarithm (base e) of all elements in the array −
```
import numpy as np

# array
arr = np.array([1, np.e, np.e**2])

# Calculate the natural logarithm of each element
log_result = np.log(arr)

print("Natural logarithm of each element:", log_result)
```

We get the output as shown below −

```
Natural logarithm of each element: [0. 1. 2.]
```

### Example: Using Comparison Ufunc

In here, we use the comparison ufunc
**numpy.greater()**that compares two arrays element-wise and returns a boolean array indicating where the elements of the first array are greater than those of the second array −
```
import numpy as np

# arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([2, 2, 2])

# Compare each element of arr1 with arr2
comparison_result = np.greater(arr1, arr2)

print("Comparison result (arr1 > arr2):", comparison_result)
```

The result produced is as follows −

```
Comparison result (arr1 > arr2): [False False  True]
```

---

## 135. NumPy - Creating Universal Functions (ufunc)

*Source: [https://www.tutorialspoint.com/numpy/numpy_creating_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_creating_ufunc.htm)*

---

---
[Previous](/numpy/numpy_unfunc_introduction.htm)[Quiz](/numpy/quiz_on_numpy_creating_ufunc.htm)[Next](/numpy/numpy_arithmetic_ufunc.htm)
## Creating Universal Functions

You can create ufuncs using the
**numpy.frompyfunc()**function, which takes a Python function and converts it into a ufunc that can operate on NumPy arrays.
## Creating a Custom ufunc

A custom ufunc in NumPy is a user-defined universal function that you create to perform element-wise operations on arrays, just like built-in ufuncs (e.g., addition, multiplication). Custom ufuncs allow you to extend NumPy's functionality with your own specialized operations.

To create a custom ufunc, you need to define a standard Python function that performs the desired operation. Then, you can use the
**numpy.frompyfunc**function to convert it into a ufunc.
The
**numpy.frompyfunc()**function requires three arguments: the Python function, the number of input arguments, and the number of output arguments.
### Example: Creating a Simple Custom ufunc

In this example, we create a custom ufunc to perform element-wise multiplication of two arrays −

```
import numpy as np

# Define a Python function for multiplication
def multiply(x, y):
   return x * y

# Create a custom ufunc from the Python function
multiply_ufunc = np.frompyfunc(multiply, 2, 1)

# Define two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Use the custom ufunc to multiply the arrays
result = multiply_ufunc(a, b)

print(result)
```

Following is the output obtained −

```
[4 10 18]
```

### Example: Custom ufunc for String Concatenation

In this example, we create a custom ufunc to perform element-wise string concatenation in NumPy −

```
import numpy as np

# Define a Python function for string concatenation
def concatenate_strings(x, y):
   return x + y

# Create a custom ufunc from the Python function
concatenate_ufunc = np.frompyfunc(concatenate_strings, 2, 1)

# Define two arrays of strings
a = np.array(["Hello", "Good"])
b = np.array([" World", " Morning"])

# Use the custom ufunc to concatenate the strings
result = concatenate_ufunc(a, b)

print(result)
```

This will produce the following result −

```
['Hello World' 'Good Morning']
```

## Advantages of Custom ufuncs

Creating custom ufuncs provides various advantages, such as −

- **Performance:**Custom ufuncs are optimized for element-wise operations, providing performance benefits over standard Python loops.
- **Reusability:**Once created, custom ufuncs can be reused across different projects and applications.
- **Flexibility:**Custom ufuncs allow you to implement specialized operations that are not available in NumPy's built-in functions.
- **Integration:**Custom ufuncs can be easily integrated with existing NumPy arrays and operations.
## Handling Multiple Outputs in Custom ufuncs

Custom ufuncs can also handle multiple outputs. To create a ufunc with multiple outputs, you need to specify the number of output arguments when using the
**numpy.frompyfunc()**function.
### Example

In this example, we create a custom ufunc that returns the quotient and remainder of division −

```
import numpy as np

# Define a Python function for division that returns quotient and remainder
def divide_and_remainder(x, y):
   return x // y, x % y

# Create a custom ufunc from the Python function
divide_ufunc = np.frompyfunc(divide_and_remainder, 2, 2)

# Define two arrays
a = np.array([10, 20, 30])
b = np.array([3, 5, 7])

# Use the custom ufunc to get quotient and remainder
quotient, remainder = divide_ufunc(a, b)

print("Quotient:", quotient)
print("Remainder:", remainder)
```

Following is the output of the above code −

```
Quotient: [3 4 4]
Remainder: [1 0 2]
```

---

## 136. NumPy - Arithmetic Universal Function (ufunc)

*Source: [https://www.tutorialspoint.com/numpy/numpy_arithmetic_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_arithmetic_ufunc.htm)*

---

---
[Previous](/numpy/numpy_creating_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_arithmetic_ufunc.htm)[Next](/numpy/numpy_rounding_decimal_ufunc.htm)
## Arithmetic Universal Function (ufunc)

An arithmetic universal function (ufunc) in NumPy is a special type of function designed to perform basic arithmetic operations (like addition, subtraction, multiplication, and division) element-wise on arrays.

These functions are optimized for performance, allowing them to execute these operations much faster than regular Python loops.

For example, when you use
**numpy.add()**function to add two arrays together, it applies the addition operation to each corresponding pair of elements in the arrays.
## NumPy Arithmetic Addition

The
**numpy.add()**function is used to perform element-wise addition of two arrays. It adds corresponding elements of the input arrays and returns a new array with the results.
### Example

In the following example, we use
**numpy.add()**function to add two arrays element-wise −
```
import numpy as np

# Define two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Perform element-wise addition
result = np.add(a, b)

print(result)
```

Following is the output obtained −

```
[5 7 9]
```

## NumPy Arithmetic Subtraction

The
**numpy.subtract()**function is used to perform element-wise subtraction of two arrays. It subtracts the elements of the second array from the corresponding elements of the first array and returns a new array with the results.
### Example

In the following example, we use
**numpy.subtract()**function to subtract one array from another element-wise −
```
import numpy as np

# Define two arrays
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

# Perform element-wise subtraction
result = np.subtract(a, b)

print(result)
```

This will produce the following result −

```
[ 9 18 27]
```

## NumPy Arithmetic Multiplication

The
**numpy.multiply()**function is used to perform element-wise multiplication of two arrays. It multiplies corresponding elements of the input arrays and returns a new array with the results.
### Example

In the following example, we use
**numpy.multiply()**function to multiply two arrays element-wise −
```
import numpy as np

# Define two arrays
a = np.array([2, 3, 4])
b = np.array([5, 6, 7])

# Perform element-wise multiplication
result = np.multiply(a, b)

print(result)
```

Following is the output of the above code −

```
[10 18 28]
```

## NumPy Arithmetic Division

The
**numpy.divide()**function is used to perform element-wise division of two arrays. It divides the elements of the first array by the corresponding elements of the second array and returns a new array with the results.
### Example

In the following example, we use
**numpy.divide()**function to divide one array by another element-wise −
```
import numpy as np

# Define two arrays
a = np.array([10, 20, 30])
b = np.array([2, 4, 5])

# Perform element-wise division
result = np.divide(a, b)

print(result)
```

The output obtained is as shown below −

```
[5. 5. 6.]
```

## Additional Arithmetic ufuncs

Besides the basic arithmetic operations, NumPy also provides other useful ufuncs for more complex mathematical operations, such as power, modulus, and trigonometric functions.

These ufuncs follow the same element-wise operation pattern and provide efficient ways to perform various calculations on arrays.

### The numpy.power() Function

The
**numpy.power()**function is used to raise elements of an array to the power of corresponding elements of another array.**Example**
In the following example, we use
**numpy.power()**function to raise one array to the power of another element-wise −
```
import numpy as np

# Define two arrays
a = np.array([2, 3, 4])
b = np.array([3, 2, 1])

# Perform element-wise power operation
result = np.power(a, b)

print(result)
```

After executing the above code, we get the following output −

```
[8 9 4]
```

### The numpy.mod() Function

The
**numpy.mod()**function is used to perform element-wise modulus operation, returning the remainder of the division of elements of the first array by corresponding elements of the second array.**Example**
In the following example, we use
**numpy.mod()**function to calculate the modulus of two arrays element-wise −
```
import numpy as np

# Define two arrays
a = np.array([10, 20, 30])
b = np.array([3, 4, 5])

# Perform element-wise modulus operation
result = np.mod(a, b)

print(result)
```

The result produced is as follows −

```
[1 0 0]
```

---

## 137. NumPy - Rounding Decimal ufunc

*Source: [https://www.tutorialspoint.com/numpy/numpy_rounding_decimal_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_rounding_decimal_ufunc.htm)*

---

---
[Previous](/numpy/numpy_arithmetic_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_rounding_decimal_ufunc.htm)[Next](/numpy/numpy_logarithmic_universal_ufunc.htm)
## Rounding Decimal Universal Function (ufunc)

A rounding decimal universal function (ufunc) in NumPy is a function designed to round the elements of an array to a specified number of decimal places. These ufuncs perform the rounding operation element-wise, ensuring that each value in the array is rounded according to the specified precision.

NumPy provides several rounding ufuncs, such as
**numpy.around()**,**numpy.floor()**,**numpy.ceil()**, and**numpy.trunc()**, each with a slightly different way of handling the rounding.
## The numpy.around() Function

The
**numpy.around()**function is used to round elements of an array to the specified number of decimals. It is versatile and can handle both integer and floating-point numbers.
### Example

In the following example, we use the
**numpy.around()**function to round elements of an array to 1 decimal place −
```
import numpy as np

# Define an array
a = np.array([1.123, 2.456, 3.789])

# Round elements to 1 decimal place
result = np.around(a, decimals=1)

print(result)
```

Following is the output obtained −

```
[1.1 2.5 3.8]
```

## The numpy.round_() Function

The
**numpy.round_()**function is an alias for**numpy.around()**function. It behaves the same way and rounds elements of an array to the specified number of decimals.
### Example

In the following example, we use the
**numpy.round_()**function to round elements of an array to 2 decimal places −
```
import numpy as np

# Define an array
a = np.array([1.12345, 2.45678, 3.78901])

# Round elements to 2 decimal places
result = np.round_(a, decimals=2)

print(result)
```

This will produce the following result −

```
[1.12 2.46 3.79]
```

## The numpy.floor() Function

The
**numpy.floor()**function is used to round elements of an array down to the nearest integer. It returns the largest integer less than or equal to each element in the array.
### Example

In the following example, we use the
**numpy.floor()**function to round elements of an array down to the nearest integer −
```
import numpy as np

# Define an array
a = np.array([1.7, 2.3, 3.9])

# Apply floor function
result = np.floor(a)

print(result)
```

Following is the output of the above code −

```
[1. 2. 3.]
```

## The numpy.ceil() Function

The
**numpy.ceil()**function is used to round elements of an array up to the nearest integer. It returns the smallest integer greater than or equal to each element in the array.
### Example

In the following example, we use the
**numpy.ceil()**function to round elements of an array up to the nearest integer −
```
import numpy as np

# Define an array
a = np.array([1.2, 2.5, 3.1])

# Apply ceil function
result = np.ceil(a)

print(result)
```

This will produce the following result −

```
[2. 3. 4.]
```

## The numpy.trunc() Function

The
**numpy.trunc()**function is used to truncate elements of an array to their integer parts by removing the fractional parts.
### Example

In the following example, we use the
**numpy.trunc()**function to truncate elements of an array −
```
import numpy as np

# Define an array
a = np.array([1.9, 2.6, 3.4])

# Apply trunc function
result = np.trunc(a)

print(result)
```

Following is the output obtained −

```
[1. 2. 3.]
```

---

## 138. NumPy - Logarithmic Universal Function (ufunc)

*Source: [https://www.tutorialspoint.com/numpy/numpy_logarithmic_universal_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_logarithmic_universal_ufunc.htm)*

---

---
[Previous](/numpy/numpy_rounding_decimal_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_logarithmic_universal_ufunc.htm)[Next](/numpy/numpy_summation_ufunc.htm)
## Logarithmic Universal Function (ufunc)

A logarithmic universal function (ufunc) in NumPy is a function that applies the logarithm operation to each element in an array. This means it computes the logarithm of every individual value in the array, either using the natural logarithm (base e) or a different base such as base-2 logarithm or base-10 logarithm.

NumPy provides several logarithmic ufuncs, such as
**numpy.log()**,**numpy.log2()**,**numpy.log10()**.
## NumPy Natural Logarithm

The
**numpy.log()**function is used to compute the natural logarithm (base-e) of each element in an array. This function is commonly used in mathematical computations involving exponential growth or decay.
### Example

In the following example, we use the
**numpy.log()**function to calculate the natural logarithm of each element in an array −
```
import numpy as np

# Define an array
a = np.array([1, 2, 3, 4, 5])

# Compute natural logarithm
result = np.log(a)

print(result)
```

Following is the output obtained −

```
[0.         0.69314718 1.09861229 1.38629436 1.60943791]
```

## NumPy Base-10 Logarithm

The
**numpy.log10()**function is used to compute the base-10 logarithm of each element in an array. This function is useful in scientific fields such as chemistry and physics, where logarithmic scales are often used.
### Example

In the following example, we use the
**numpy.log10()**function to calculate the base-10 logarithm of each element in an array −
```
import numpy as np

# Define an array
a = np.array([1, 10, 100, 1000, 10000])

# Compute base-10 logarithm
result = np.log10(a)

print(result)
```

This will produce the following result −

```
[0. 1. 2. 3. 4.]
```

## NumPy Base-2 Logarithm

The
**numpy.log2()**function is used to compute the base-2 logarithm of each element in an array. This function is often used in computer science and information theory.
### Example

In the following example, we use the
**numpy.log2()**function to calculate the base-2 logarithm of each element in an array −
```
import numpy as np

# Define an array
a = np.array([1, 2, 4, 8, 16])

# Compute base-2 logarithm
result = np.log2(a)

print(result)
```

Following is the output of the above code −

```
[0. 1. 2. 3. 4.]
```

## NumPy Logarithm with Any Base

While NumPy provides specific functions for base-e, base-10, and base-2 logarithms, you can compute logarithms with any base by using the
**numpy.log()**function in combination with the change of base formula.
### Example

In the following example, we calculate the base-3 logarithm of each element in an array using the change of base formula −

```
import numpy as np

# Define an array
a = np.array([1, 3, 9, 27, 81])

# Compute base-3 logarithm
result = np.log(a) / np.log(3)

print(result)
```

The result produced is as follows −

```
[0. 1. 2. 3. 4.]
```

## NumPy Logarithm of 1 plus Input

The
**numpy.log1p()**function is used to compute the natural logarithm of 1 plus the input array elements. This function provides more accurate results for small input values compared to directly using**numpy.log(1 + x)**function.
### Example

In the following example, we use the
**numpy.log1p()**function to calculate the natural logarithm of 1 plus each element in an array −
```
import numpy as np

# Define an array
a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# Compute natural logarithm of 1 plus the input array elements
result = np.log1p(a)

print(result)
```

Following is the output obtained −

```
[0.09531018 0.18232156 0.26236426 0.33647224 0.40546511]
```

---

## 139. NumPy - Summation Universal Function (ufunc)

*Source: [https://www.tutorialspoint.com/numpy/numpy_summation_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_summation_ufunc.htm)*

---

---

## 140. NumPy - Product Universal Function (ufunc)

*Source: [https://www.tutorialspoint.com/numpy/numpy_product_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_product_ufunc.htm)*

---

---
[Previous](/numpy/numpy_summation_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_product_ufunc.htm)[Next](/numpy/numpy_difference_ufunc.htm)
## Product Universal Function (ufunc)

A product universal function (ufunc) in NumPy is a function used to compute the product of elements in an array.

This operation multiplies all the elements together, either for the entire array or along a specific axis (such as rows or columns). The primary product ufunc in NumPy is numpy.prod() function.

## NumPy Product

The
**numpy.prod()**function is used to compute the product of array elements over a specified axis. It can compute the product of all elements in an array or along a specific axis (e.g., row-wise or column-wise).
### Example

In the following example, we use the
**numpy.prod()**function to calculate the product of elements in an array −
```
import numpy as np

# Define an array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Compute the product of all elements
total_product = np.prod(a)

# Compute the product along the columns
column_product = np.prod(a, axis=0)

# Compute the product along the rows
row_product = np.prod(a, axis=1)

print("Total product:", total_product)
print("Column-wise product:", column_product)
print("Row-wise product:", row_product)
```

Following is the output obtained −

```
Total product: 720
Column-wise product: [ 4 10 18]
Row-wise product: [ 6 120]
```

## NumPy Cumulative Product

The
**numpy.cumprod()**function is used to compute the cumulative product of array elements along a specified axis. It returns an array where each element is the cumulative product of the previous elements.
### Example

In the following example, we use the
**numpy.cumprod()**function to calculate the cumulative product of elements in an array −
```
import numpy as np

# Define an array
a = np.array([1, 2, 3, 4, 5])

# Compute the cumulative product
cumulative_product = np.cumprod(a)

print("Cumulative product:", cumulative_product)
```

This will produce the following result −

```
Cumulative product: [  1   2   6  24 120]
```

## NumPy Product with Conditions

The
**numpy.prod()**function can also be used with conditional statements to compute the product of elements that meet a specific condition.
### Example

In the following example, we use the
**numpy.prod()**function to calculate the product of elements that are greater than a specified value −
```
import numpy as np

# Define an array
a = np.array([1, 2, 3, 4, 5])

# Compute the product of elements greater than 2
conditional_product = np.prod(a[a > 2])

print("Product of elements greater than 2:", conditional_product)
```

The result produced is as follows −

```
Product of elements greater than 2: 60
```

## Matrix Product with NumPy ufuncs

The matrix product in NumPy refers to multiplying two matrices together, following the rules of linear algebra. This operation is done using
**numpy.matmul()**function or the**@**operator, which computes the dot product of two arrays.
### Example

In this example, np.matmul() function performs the matrix multiplication of
**matrix1**and**matrix2**, resulting in a new matrix −
```
import numpy as np

# Define two 2D arrays (matrices)
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Perform matrix multiplication
result = np.matmul(matrix1, matrix2)

print(result)
```

We get the output as shown below −

```
[[19 22]
 [43 50]]
```

## NumPy Dot Product and Cross Product

The dot product calculates the sum of the products of corresponding elements in two arrays, while the cross product finds a vector perpendicular to two input vectors in 3D space.

NumPy provides
**numpy.dot()**function for the dot product and**numpy.cross()**function for the cross product.
### Example

In this example, np.dot() function calculates the dot product of the two vectors
**vector1**and**vector2**−
```
import numpy as np

# Define two 1D arrays (vectors)
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

# Compute the dot product
dot_result = np.dot(vector1, vector2)

print(dot_result)
```

Following is the output obtained −

```
32
```

## NumPy Element-wise Product Operations

Element-wise product operations in NumPy involve multiplying corresponding elements of two arrays. This is done using the
**numpy.multiply()**function or the*****operator, and it is useful for operations like scaling values in an array.
### Example

In the following example, the np.multiply() function multiplies corresponding elements of
**array1**and**array2**element-wise −
```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# Perform element-wise multiplication
product = np.multiply(array1, array2)

print(product)
```

This will produce the following result −

```
[ 4 10 18]
```

---

## 141. NumPy - Difference Universal Function (ufunc)

*Source: [https://www.tutorialspoint.com/numpy/numpy_difference_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_difference_ufunc.htm)*

---

---
[Previous](/numpy/numpy_product_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_difference_ufunc.htm)[Next](/numpy/numpy_finding_lcm_ufunc.htm)
## Difference Universal Function (ufunc)

A difference universal function (ufunc) in NumPy is a function used to calculate the difference between elements in an array.

This operation can be applied element-wise between two arrays, or to compute the discrete difference along a specific axis of a single array. The primary function for computing differences in NumPy is numpy.diff().

## The NumPy diff() Function

The
**numpy.diff()**function computes the difference between consecutive elements in an array, effectively calculating the first order discrete difference. It can also compute higher-order differences by specifying the**n**parameter.
### Example

In the following example, we use the
**numpy.diff()**function to calculate the difference between consecutive elements of a 1D array −
```
import numpy as np

# Define an array
a = np.array([1, 3, 6, 10, 15])

# Compute the first-order differences
diff = np.diff(a)

print("First-order differences:", diff)
```

Following is the output obtained −

```
First-order differences: [2 3 4 5]
```

## NumPy Higher-Order Differences

The
**numpy.diff()**function can also compute higher-order differences. By specifying the**n**parameter, we can calculate the difference between elements multiple times.
### Example

In the following example, we calculate the second-order difference of an array using
**numpy.diff()**function with**n=2**−
```
import numpy as np

# Define an array
a = np.array([1, 3, 6, 10, 15])

# Compute the second-order differences
second_diff = np.diff(a, n=2)

print("Second-order differences:", second_diff)
```

This will produce the following result −

```
Second-order differences: [1 1 1]
```

## NumPy diff Function for 2D Arrays

The
**numpy.diff()**function can also be used with 2D arrays. By specifying the**axis**parameter, we can compute differences along a specific axis, such as row-wise or column-wise.
### Example

In the following example, we use
**numpy.diff()**function on a 2D array to compute the differences along the rows and columns −
```
import numpy as np

# Define a 2D array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Compute the differences along the rows (axis=1)
row_diff = np.diff(a, axis=1)

# Compute the differences along the columns (axis=0)
col_diff = np.diff(a, axis=0)

print("Row-wise differences:", row_diff)
print("Column-wise differences:", col_diff)
```

Following is the output obtained −

```
Row-wise differences: 
[[1 1]
 [1 1]]
Column-wise differences: [[3 3 3]]
```

## NumPy diff with Periodic Boundary

By default,
**numpy.diff()**function treats the array as open-ended, meaning it calculates differences between adjacent elements. However, we can specify the**prepend**or**append**arguments to include boundary elements for periodic data.
### Example

In the following example, we use
**numpy.diff()**function with the**prepend**argument to treat the array as a periodic sequence −
```
import numpy as np

# Define an array
a = np.array([1, 3, 6, 10, 15])

# Compute the first-order differences with a periodic boundary
periodic_diff = np.diff(a, append=a[0])

print("Periodic differences:", periodic_diff)
```

The result will be as follows −

```
Periodic differences: [  2   3   4   5 -14]
```

## The numpy.gradient() Function

The numpy.gradient() function computes the gradient of an array. The gradient is a multi-dimensional generalization of the derivative.

For a 1D array, it calculates the differences between consecutive elements and takes into account edge effects. For multi-dimensional arrays, it computes the gradient along each axis.

### Example

In this example, the gradient() function calculates the gradient of an array −

```
import numpy as np

# Create a NumPy array
array = np.array([1, 2, 4, 7, 11])

# Compute the gradient
gradient = np.gradient(array)

print(gradient)
```

The output represents the rate of change between each element, with special handling at the boundaries −

```
[1.  1.5 2.5 3.5 4. ]
```

## The numpy.ediff1d() Function

The numpy.ediff1d() function computes the differences between consecutive elements of an array in a flattened form. It is a simpler and faster alternative to numpy.diff() function for 1D arrays.

### Example

In the following example, the ediff1d() function calculates the difference between each consecutive element of the array −

```
import numpy as np

# Create a NumPy array
array = np.array([1, 2, 4, 7, 11])

# Compute the element-wise differences
ediff1d_result = np.ediff1d(array)

print(ediff1d_result)
```

This will produce the following result −

```
[1 2 3 4]
```

---

## 142. NumPy - Finding LCM with ufunc

*Source: [https://www.tutorialspoint.com/numpy/numpy_finding_lcm_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_finding_lcm_ufunc.htm)*

---

---
[Previous](/numpy/numpy_difference_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_finding_lcm_ufunc.htm)[Next](/numpy/numpy_finding_gcd_ufunc.htm)
## Finding LCM with Universal Function

NumPy provides a universal function (ufunc) called
**numpy.lcm()**to compute the Least Common Multiple (LCM) of two arrays element-wise. The LCM of two integers is the smallest positive integer that is divisible by both numbers.
This function is particularly useful when working with arrays of integers where you need to find the LCM of corresponding elements.

## The NumPy lcm() Function

The
**numpy.lcm()**function is used to compute the element-wise Least Common Multiple of two arrays. It returns a new array containing the LCM of the corresponding elements from the input arrays.
### Example

In the following example, we use the
**numpy.lcm()**function to find the LCM of elements in two arrays −
```
import numpy as np

# Define two arrays
a = np.array([4, 6, 8])
b = np.array([6, 8, 10])

# Compute the element-wise LCM
lcm_result = np.lcm(a, b)

print("LCM of arrays:", lcm_result)
```

Following is the output obtained −

```
LCM of arrays: [12 24 40]
```

## NumPy lcm() Function with Scalars

The
**numpy.lcm()**function can also be used with scalar values to compute the LCM of two single integers. It works the same way as with arrays, returning the LCM of the given scalars.
### Example

In the following example, we use the
**numpy.lcm()**function to find the LCM of two scalar values −
```
import numpy as np

# Define two scalars
a = 15
b = 20

# Compute the LCM of the scalars
lcm_result = np.lcm(a, b)

print("LCM of scalars:", lcm_result)
```

This will produce the following result −

```
LCM of scalars: 60
```

## LCM of Multi-dimensional Arrays

The
**numpy.lcm()**function can also be applied to multi-dimensional arrays. It computes the LCM for each corresponding element in the arrays, handling arrays of any shape as long as they are broadcastable to a common shape.
### Example

In the following example, we use the
**numpy.lcm()**function to compute the LCM of two 2D arrays element-wise −
```
import numpy as np

# Define two 2D arrays
a = np.array([[3, 4], [5, 6]])
b = np.array([[6, 8], [10, 12]])

# Compute the element-wise LCM
lcm_result = np.lcm(a, b)

print("LCM of 2D arrays:\n", lcm_result)
```

The result will be as follows −

```
LCM of 2D arrays:
[[ 6  8]
 [10 12]]
```

## The NumPy lcm.reduce() Function

The
**numpy.lcm.reduce()**function computes the LCM of array elements along a specified axis. This is useful for finding the LCM of multiple elements within an array.
### Example

In the following example, we use the
**numpy.lcm.reduce()**function to find the LCM of all elements in an array −
```
import numpy as np

# Define an array
a = np.array([12, 15, 20])

# Compute the LCM of all elements
lcm_result = np.lcm.reduce(a)

print("LCM of all elements:", lcm_result)
```

This will produce the following result −

```
LCM of all elements: 60
```

---

## 143. NumPy - Finding GCD with ufunc

*Source: [https://www.tutorialspoint.com/numpy/numpy_finding_gcd_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_finding_gcd_ufunc.htm)*

---

---
[Previous](/numpy/numpy_finding_lcm_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_finding_gcd_ufunc.htm)[Next](/numpy/numpy_trigonometric_ufunc.htm)
## Finding GCD with Universal Function

NumPy provides a universal function (ufunc) called
**numpy.gcd()**to compute the Greatest Common Divisor (GCD) of two arrays element-wise. The GCD of two integers is the largest positive integer that divides both numbers without leaving a remainder.
This function is particularly useful when working with arrays of integers where you need to find the GCD of corresponding elements.

## The NumPy gcd() Function

The
**numpy.gcd()**function is used to compute the element-wise Greatest Common Divisor of two arrays. It returns a new array containing the GCD of the corresponding elements from the input arrays.
### Example

In the following example, we use the
**numpy.gcd()**function to find the GCD of elements in two arrays −
```
import numpy as np

# Define two arrays
a = np.array([12, 18, 24])
b = np.array([15, 27, 36])

# Compute the element-wise GCD
gcd_result = np.gcd(a, b)

print("GCD of arrays:", gcd_result)
```

Following is the output obtained −

```
GCD of arrays: [3 9 12]
```

## NumPy gcd() Function with Scalars

The
**numpy.gcd()**function can also be used with scalar values to compute the GCD of two single integers. It works the same way as with arrays, returning the GCD of the given scalars.
### Example

In the following example, we use the
**numpy.gcd()**function to find the GCD of two scalar values −
```
import numpy as np

# Define two scalars
a = 48
b = 60

# Compute the GCD of the scalars
gcd_result = np.gcd(a, b)

print("GCD of scalars:", gcd_result)
```

This will produce the following result −

```
GCD of scalars: 12
```

## GCD of Multi-dimensional Arrays

The
**numpy.gcd()**function can also be applied to multi-dimensional arrays. It computes the GCD for each corresponding element in the arrays, handling arrays of any shape as long as they are broadcastable to a common shape.
### Example

In the following example, we use the
**numpy.gcd()**function to compute the GCD of two 2D arrays element-wise −
```
import numpy as np

# Define two 2D arrays
a = np.array([[14, 21], [35, 49]])
b = np.array([[7, 14], [21, 28]])

# Compute the element-wise GCD
gcd_result = np.gcd(a, b)

print("GCD of 2D arrays:\n", gcd_result)
```

The result will be as follows −

```
GCD of 2D arrays:
[[ 7  7]
 [ 7  7]]
```

## The NumPy gcd.reduce() Function

The
**numpy.gcd.reduce()**function computes the GCD of array elements along a specified axis. This is useful for finding the GCD of multiple elements within an array.
### Example

In the following example, we use the
**numpy.gcd.reduce()**function to find the GCD of all elements in an array −
```
import numpy as np

# Define an array
a = np.array([60, 90, 150])

# Compute the GCD of all elements
gcd_result = np.gcd.reduce(a)

print("GCD of all elements:", gcd_result)
```

This will produce the following result −

```
GCD of all elements: 30
```

---

## 144. NumPy - Trigonometric ufunc

*Source: [https://www.tutorialspoint.com/numpy/numpy_trigonometric_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_trigonometric_ufunc.htm)*

---

---
[Previous](/numpy/numpy_finding_gcd_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_trigonometric_ufunc.htm)[Next](/numpy/numpy_hyperbolic_ufunc.htm)
## Trigonometric Universal Functions (ufunc)

Trigonometric universal functions (ufuncs) in NumPy are functions that perform trigonometric operations on each element of an array. These functions can calculate various trigonometric values such as sine, cosine, tangent, and their inverses for each element in the input array.

These functions operate element-wise on arrays and are optimized for performance, making them much faster than using Python loops.

## NumPy Sine Function

The
**numpy.sin()**function is used to calculate the sine of each element in an array. The input values are assumed to be in radians.
The sine of an angle in a right-angled triangle is the ratio of the length of the opposite side to the hypotenuse. It is denoted as sin().

### Example

In the following example, we use the
**numpy.sin()**function to calculate the sine of each element in an array −
```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

# Calculate the sine of each angle
sine_values = np.sin(angles)

print("Sine values:", sine_values)
```

Following is the output obtained −

```
Sine values: [ 0.0000000e+00  1.0000000e+00  1.2246468e-16 -1.0000000e+00]
```

## NumPy Cosine Function

The
**numpy.cos()**function is used to calculate the cosine of each element in an array.
The cosine of an angle in a right-angled triangle is the ratio of the length of the adjacent side to the hypotenuse. It is denoted as cos().

### Example

In the following example, we use the
**numpy.cos()**function to calculate the cosine of each element in an array −
```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

# Calculate the cosine of each angle
cosine_values = np.cos(angles)

print("Cosine values:", cosine_values)
```

This will produce the following result −

```
Cosine values: [ 1.0000000e+00  6.1232340e-17 -1.0000000e+00 -1.8369702e-16]
```

## NumPy Tangent Function

The
**numpy.tan()**function is used to calculate the tangent of each element in an array.
The tangent of an angle in a right-angled triangle is the ratio of the length of the opposite side to the adjacent side. It is denoted as tan().

### Example

In the following example, we use the
**numpy.tan()**function to calculate the tangent of each element in an array −
```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/4, np.pi/2, np.pi])

# Calculate the tangent of each angle
tangent_values = np.tan(angles)

print("Tangent values:", tangent_values)
```

The result produced is as follows −

```
Tangent values: [ 0.00000000e+00  1.00000000e+00  1.63312394e+16 -1.22464680e-16]
```

> NumPy also provides functions for calculating the inverse trigonometric functions (arcsine, arccosine, and arctangent) of array elements. These functions return the angle in radians for a given trigonometric value.

## NumPy Arcsine Function

The
**numpy.arcsin()**function is used to calculate the inverse sine of each element in an array, returning the angle in radians.
The arcsine is the inverse function of sine, which returns the angle whose sine is a given number. It is denoted as arcsin(x) or sin(x).

### Example

In this example, we use the
**numpy.arcsin()**function to calculate the inverse sine of each element in an array −
```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/4, np.pi/2, np.pi])

# Calculate the Inverse Sine of each angle
inverse_sine_values = np.arcsin(angles)

print("Inverse Sine values:", inverse_sine_values)
```

We get the output as shown below −

```
Inverse Sine values: [0.         0.90333911        nan        nan]
```

## NumPy Arccosine Function

The
**numpy.arccos()**function is used to calculate the inverse cosine of each element in an array, returning the angle in radians.
The arccosine is the inverse function of cosine, which returns the angle whose cosine is a given number. It is denoted as arccos(x) or cos(x).

### Example

In this example, we use the
**numpy.arccos()**function to calculate the inverse cosine of each element in an array −
```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/4, np.pi/2, np.pi])

# Calculate the Inverse Cosine of each angle
inverse_cosine_values = np.arccos(angles)

print("Inverse Cosine values:", inverse_cosine_values)
```

After executing the above code, we get the following output −

```
Inverse Cosine values: [1.57079633 0.66745722        nan        nan]
```

## NumPy Arctangent Function

The
**numpy.arctan()**function is used to calculate the inverse tangent of each element in an array, returning the angle in radians.
The arctangent is the inverse function of tangent, which returns the angle whose tangent is a given number. It is denoted as arctan(x) or tan(x).

### Example

In the example below, we use the
**numpy.arctan()**function to calculate the inverse tangent of each element in an array −
```
import numpy as np

# Define an array of angles in radians
angles = np.array([0, np.pi/4, np.pi/2, np.pi])

# Calculate the Inverse Tangent of each angle
inverse_tangent_values = np.arctan(angles)

print("Inverse Tangent values:", inverse_tangent_values)
```

After executing the above code, we get the following output −

```
Inverse Tangent values: [0.         0.66577375 1.00388482 1.26262726]
```

## NumPy Hyperbolic Functions

Hyperbolic functions are mathematical functions similar to trigonometric functions but based on hyperbolas instead of circles.

NumPy also provides functions for calculating the hyperbolic sine (sinh), hyperbolic cosine (cosh), and hyperbolic tangent (tanh), along with their inverses for array elements. These functions are analogous to the trigonometric functions but are applied to hyperbolic angles.

### Example

In the following example, we use the
**numpy.sinh()**,**numpy.cosh()**, and**numpy.tanh()**functions to calculate the hyperbolic values of elements in an array −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2])

# Calculate the hyperbolic values
sinh_values = np.sinh(values)
cosh_values = np.cosh(values)
tanh_values = np.tanh(values)

print("Hyperbolic sine values:", sinh_values)
print("Hyperbolic cosine values:", cosh_values)
print("Hyperbolic tangent values:", tanh_values)
```

This will produce the following result −

```
Hyperbolic sine values: [0.         1.17520119 3.62686041]
Hyperbolic cosine values: [1.         1.54308063 3.76219569]
Hyperbolic tangent values: [0.         0.76159416 0.96402758]
```

---

## 145. NumPy - Hyperbolic ufunc

*Source: [https://www.tutorialspoint.com/numpy/numpy_hyperbolic_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_hyperbolic_ufunc.htm)*

---

---
[Previous](/numpy/numpy_trigonometric_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_hyperbolic_ufunc.htm)[Next](/numpy/numpy_set_operations_ufunc.htm)
## Hyperbolic Universal Functions (ufunc)

Hyperbolic universal functions (ufuncs) in NumPy are functions that perform hyperbolic operations on each element of an array. These functions can calculate various hyperbolic values such as hyperbolic sine, cosine, and tangent, and their inverses for each element in the input array.

These functions operate element-wise on arrays and are optimized for performance, making them much faster than using Python loops.

## NumPy Hyperbolic Sine Function

The
**numpy.sinh()**function is used to calculate the hyperbolic sine of each element in an array.
The hyperbolic sine function is defined as sinh(x) = (e
- e) / 2.
### Example

In the following example, we use the
**numpy.sinh()**function to calculate the hyperbolic sine of each element in an array −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2])

# Calculate the hyperbolic sine of each value
sinh_values = np.sinh(values)

print("Hyperbolic sine values:", sinh_values)
```

The output obtained is as follows −

```
Hyperbolic sine values: [0.         1.17520119 3.62686041]
```

## NumPy Hyperbolic Cosine Function

The
**numpy.cosh()**function is used to calculate the hyperbolic cosine of each element in an array.
The hyperbolic cosine function is defined as cosh(x) = (e
+ e) / 2.
### Example

In the following example, we use the
**numpy.cosh()**function to calculate the hyperbolic cosine of each element in an array −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2])

# Calculate the hyperbolic cosine of each value
cosh_values = np.cosh(values)

print("Hyperbolic cosine values:", cosh_values)
```

This will produce the following result −

```
Hyperbolic cosine values: [1.         1.54308063 3.76219569]
```

## NumPy Hyperbolic Tangent Function

The
**numpy.tanh()**function is used to calculate the hyperbolic tangent of each element in an array.
The hyperbolic tangent function is defined as tanh(x) = sinh(x) / cosh(x).

### Example

In the following example, we use the
**numpy.tanh()**function to calculate the hyperbolic tangent of each element in an array −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2])

# Calculate the hyperbolic tangent of each value
tanh_values = np.tanh(values)

print("Hyperbolic tangent values:", tanh_values)
```

The result produced is as follows −

```
Hyperbolic tangent values: [0.         0.76159416 0.96402758]
```

> NumPy also provides functions for calculating the inverse hyperbolic functions (arcsinh, arccosh, and arctanh) of array elements. These functions return the value whose hyperbolic sine, cosine, or tangent is the given number.

## NumPy Inverse Hyperbolic Sine Function

The
**numpy.arcsinh()**function is used to calculate the inverse hyperbolic sine of each element in an array.
The inverse hyperbolic sine function is defined as arcsinh(x) = ln(x + sqrt(x
+ 1)).
### Example

In this example, we use the
**numpy.arcsinh()**function to calculate the inverse hyperbolic sine of each element in an array −
```
import numpy as np

# Define an array of values
values = np.array([0, 1, 2])

# Calculate the inverse hyperbolic sine of each value
arcsinh_values = np.arcsinh(values)

print("Inverse hyperbolic sine values:", arcsinh_values)
```

We get the output as shown below −

```
Inverse hyperbolic sine values: [0.         0.88137359 1.44363548]
```

## NumPy Inverse Hyperbolic Cosine Function

The
**numpy.arccosh()**function is used to calculate the inverse hyperbolic cosine of each element in an array.
The inverse hyperbolic cosine function is defined as arccosh(x) = ln(x + sqrt(x
- 1)).
### Example

In this example, we use the
**numpy.arccosh()**function to calculate the inverse hyperbolic cosine of each element in an array −
```
import numpy as np

# Define an array of values
values = np.array([1, 2, 3])

# Calculate the inverse hyperbolic cosine of each value
arccosh_values = np.arccosh(values)

print("Inverse hyperbolic cosine values:", arccosh_values)
```

The output obtained is as follows −

```
Inverse hyperbolic cosine values: [0.         1.3169579  1.76274717]
```

## NumPy Inverse Hyperbolic Tangent Function

The
**numpy.arctanh()**function is used to calculate the inverse hyperbolic tangent of each element in an array.
The inverse hyperbolic tangent function is defined as arctanh(x) = 0.5 * ln((1 + x) / (1 - x)).

### Example

In the example below, we use the
**numpy.arctanh()**function to calculate the inverse hyperbolic tangent of each element in an array −
```
import numpy as np

# Define an array of values
values = np.array([0, 0.5, 0.9])

# Calculate the inverse hyperbolic tangent of each value
arctanh_values = np.arctanh(values)

print("Inverse hyperbolic tangent values:", arctanh_values)
```

The output produced is as follows −

```
Inverse hyperbolic tangent values: [0.         0.54930614 1.47221949]
```

---

## 146. NumPy - Set Operations ufunc

*Source: [https://www.tutorialspoint.com/numpy/numpy_set_operations_ufunc.htm](https://www.tutorialspoint.com/numpy/numpy_set_operations_ufunc.htm)*

---

---
[Previous](/numpy/numpy_hyperbolic_ufunc.htm)[Quiz](/numpy/quiz_on_numpy_set_operations_ufunc.htm)[Next](/numpy/numpy_quick_guide.htm)
## Set Operations Universal Functions (ufunc)

NumPy provides several universal functions (ufuncs) that perform set operations on arrays. These operations are used to compare and manipulate sets of elements within arrays. The main set operations include union, intersection, difference, and exclusive-or.

## NumPy Union Function

The
**numpy.union1d()**function is used to find the union of two arrays. The union of two sets is a set containing all elements from both sets, without duplicates.
### Example

In the following example, we use the
**numpy.union1d()**function to find the union of two arrays −
```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([3, 4, 5])

# Find the union of the arrays
union_result = np.union1d(array1, array2)

print("Union of arrays:", union_result)
```

The output obtained is as follows −

```
Union of arrays: [1 2 3 4 5]
```

## NumPy Intersection Function

The
**numpy.intersect1d()**function is used to find the intersection of two arrays. The intersection of two sets is a set containing only the elements that are present in both sets.
### Example

In the following example, we use the
**numpy.intersect1d()**function to find the intersection of two arrays −
```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([3, 4, 5])

# Find the intersection of the arrays
intersection_result = np.intersect1d(array1, array2)

print("Intersection of arrays:", intersection_result)
```

The output produced is as follows −

```
Intersection of arrays: [3]
```

## NumPy Set Difference Function

The
**numpy.setdiff1d()**function is used to find the set difference of two arrays. The set difference of two sets is a set containing elements that are present in the first set but not in the second set.
### Example

In this example, we use the
**numpy.setdiff1d()**function to find the difference between two arrays −
```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([3, 4, 5])

# Find the set difference of the arrays
difference_result = np.setdiff1d(array1, array2)

print("Difference of arrays:", difference_result)
```

We get the following result −

```
Difference of arrays: [1 2]
```

## NumPy Set Exclusive-or Function

The
**numpy.setxor1d()**function is used to find the set exclusive-or (symmetric difference) of two arrays. The set exclusive-or of two sets is a set containing elements that are present in either of the sets but not in both.
### Example

In this example, we use the
**numpy.setxor1d()**function to find the exclusive-or of two arrays −
```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([3, 4, 5])

# Find the set exclusive-or of the arrays
xor_result = np.setxor1d(array1, array2)

print("Exclusive-or of arrays:", xor_result)
```

The output produced is as follows −

```
Exclusive-or of arrays: [1 2 4 5]
```

## The NumPy in1d() Function

The
**numpy.in1d()**function tests whether each element of an array is also present in a second array. It returns a boolean array of the same shape as the first array, indicating whether each element is present in the second array.
### Example

In the example below, we use the
**numpy.in1d()**function to test whether elements of one array are present in another array −
```
import numpy as np

# Define two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([3, 4, 5])

# Test whether elements of array1 are in array2
in1d_result = np.in1d(array1, array2)

print("Elements of array1 in array2:", in1d_result)
```

The output obtained is as follows −

```
Elements of array1 in array2: [False False  True]
```

## The NumPy isin() Function

The
**numpy.isin()**function tests whether each element of an array is present in a list of values or another array. It is similar to**numpy.in1d()**function but can be used with multi-dimensional arrays.
### Example

In this example, we use the
**numpy.isin()**function to test whether elements of an array are present in another array −
```
import numpy as np

# Define an array and a list of values
array = np.array([1, 2, 3, 4])
values = [2, 4, 6]

# Test whether elements of the array are in the list of values
isin_result = np.isin(array, values)

print("Elements of array in values:", isin_result)
```

The result produced is as follows −

```
Elements of array in values: [False  True False  True]
```

---


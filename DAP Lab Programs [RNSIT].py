#Lab 1: Write a Python program to perform linear search

def linearSearch(array, n, x):

    # Going through array sequencially
    for i in range(0, n):
        if (array[i] == x):
            return i
    return -1


array = [2, 4, 0, 1, 9]
x = eval(input("enter the element to be searched: "))
n = len(array)
result = linearSearch(array, n, x)
if(result == -1):
    print("Element not found")
else:
    print("Element found at index: ", result)
##########################################################################################################################

#Lab 2: Write a Python program to insert an element into a sorted list

import bisect 
  
"""def insert(list, n):
    bisect.insort(list, n) 
    return list"""
  
# Driver function

list = [1, 2, 4]
n = eval(input("enter the value to be inserted "))
bisect.insort(list, n)
print(list)
#print(insert(list, n))
##########################################################################################################################

#Lab 3: Write a python program using object oriented programming to demonstrate encapsulation, overloading and inheritance

class Base:
    def __init__(self):
        self.a = 10
        self._b = 20
        
    def display(self):
         print(" the values are :")
         print(f"a={self.a} b={self._b}")

class Derived(Base):                                   # Creating a derived class
    def __init__(self):         
        Base.__init__(self)                        # Calling constructor of Base class
        self.d = 30
        
    def display(self):
        Base.display(self)
        print(f"d={self.d}")

    def __add__(self, ob):
        return self.a + ob.a+self.d + ob.d
        #return self.a + ob.a+self.d + ob.d+self.b + ob.b


obj2 = Derived()
obj3 = Derived()


obj2.display()
obj3.display()

print("\n Sum of two objects :",obj2 + obj3)
##########################################################################################################################

#Lab 4: Implement a python program to demonstrate

import pandas as pd
cars_data=pd.read_csv("Toyota.csv")

cars_data.head()

cars2 =cars_data.copy()
cars3 =cars_data.copy()

# identifying missing values(NaN -> Not a Number)
cars2.isna().sum()

#subsetting the rows that have one or more missing values 
missing = cars2[cars2.isnull().any(axis=1)]
missing

cars2.describe()

#calculating the meanvalue of the 'Age' variable
cars2['Age'].mean()

#To fill NA/NAN values using the specified value
cars2['Age'].fillna(cars2['Age'].mean(), inplace = True)
#cars2['KM'].fillna(cars2['KM'].median(), inplace = True)
#cars2['HP'].fillna(cars2['HP'].mean(), inplace = True)
cars2.isna().sum()

#imputing missing values of categorical variables
cars2['FuelType'].value_counts()

# to get the mode value of FuelType
cars2['FuelType'].value_counts().index[0]
cars2['FuelType'].fillna(cars2['FuelType'].value_counts().index[0], inplace = True)

cars2['MetColor'].fillna(cars2['MetColor'].mode()[0], inplace = True)

cars2.isna().sum()

print(cars_data.shape)
print(cars_data.index)
cars_data.ndim
##########################################################################################################################

#Lab 5: Implement a python program to demonstrate the following using NumPy

import numpy as np
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
print("concatenating two arrays \n",np.concatenate([arr1, arr2], axis=1))     #columnwise concatenation
print("vertical stacking  \n",np.vstack((arr1, arr2)))                      #Stack arrays in sequence vertically (row wise).
print("horizontal stacking \n",np.hstack((arr1,arr2)))

arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)

arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 5)
print(x)

arr = np.array([1, 3, 5, 7])
x = np.searchsorted(arr, [2, 4, 6])
print(x)

a = np.array([[1,4],[3,1]])
print("sorted array : ",np.sort(a))                # sort along the last axis
print("\n sorted flattened array:", np.sort(a, axis=0))    # sort the flattened array

x = np.array([3, 1, 2])

print("\n indices that would sort an array",np.argsort(x))

print("\n sorting complex number :" ,np.sort_complex([5, 3, 6, 2, 1]))


import numpy as np

x = np.arange(9.0)
print(np.split(x, 3))                # with no of partitions N, 
print(np.split(x, [3, 5, 6, 10]))   # with indices

#the array will be divided into N equal arrays along axis. If such a split is not possible, an error is raised.

x = np.arange(9)
np.array_split(x, 4)               

#Split an array into multiple sub-arrays of equal or near-equal size. Does not raise an exception if an
equal division cannot be made.

a = np.array([[1, 3, 5, 7, 9, 11],
              [2, 4, 6, 8, 10, 12]])
  
# horizontal splitting
print("Splitting along horizontal axis into 2 parts:\n", np.hsplit(a, 2))
  
# vertical splitting
print("\nSplitting along vertical axis into 2 parts:\n", np.vsplit(a, 2))

import numpy as np
 
v = np.array([1, 2, 3]) 
w = np.array([4, 5])  
print("v = ",v)
print("w = ",w)
 
# To compute an outer product we first
# reshape v to a column vector of shape 3x1
# then broadcast it against w to yield an output
# of shape 3x2 which is the outer product of v and w

print("\n outer product of v and w is :\n")
print(np.reshape(v, (3, 1)) * w)

 
X = np.array([[1, 2, 3], [4, 5, 6]])
print("\n X = ",X) 
print("\n v = ",v)
# x has shape  2x3 and v has shape (3, )
# so they broadcast to 2x3,
print("\n X + v = ",X + v)
 
# Add a vector to each column of a matrix X has
# shape 2x3 and w has shape (2, ) If we transpose X
# then it has shape 3x2 and can be broadcast against w
# to yield a result of shape 3x2.
 
# Transposing this yields the final result
# of shape  2x3 which is the matrix.
print("\n Transposing this  final result :")
print((X.T + w).T)
 
# Another solution is to reshape w to be a column
# vector of shape 2X1 we can then broadcast it
# directly against X to produce the same output.
print("\n X+ np.reshape(w, (2, 1))")
print(X+ np.reshape(w, (2, 1)))
 
# Multiply a matrix by a constant, X has shape  2x3.
# Numpy treats scalars as arrays of shape();
# these can be broadcast together to shape 2x3.
print(X * 2)

import numpy as np
import matplotlib.pyplot as plt
 
# Computes x and y coordinates for
# points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
 
# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
 
plt.show()
##########################################################################################################################

#Lab 6:  Implement a python program to demonstrate Data visualization with various Types of Graphs using matplotlib

#### Data Visualization
#### Library - Matplotlib
import matplotlib.pyplot as plt

#### Line Plot 
x1 = [1,2,3,4,5]
y1 = [2,5,2,6,8]
x2 = [1,2,3,4,5]
y2 = [4,5,8,9,10]
plt.xlabel("X Axis",fontsize=12,fontstyle='italic')
plt.ylabel("Y Axis",fontsize=12)
plt.title("Line Plot",fontsize=15,fontname='DejaVu Sans')
plt.plot(x1,y1,color="red",label="First Graph")   ### line plot
plt.plot(x2,y2,color="blue",label="Second Graph")   ### line plot
plt.legend(loc=2)
plt.grid()
#plt.axis('off')
plt.show()

#### Bar Plot 
#x = [1,2,3,4,5]
x = ['A',"B","C","D","E"]
y = [20,50,20,60,80]
plt.xlabel("X Axis",fontsize=12)
plt.ylabel("Y Axis",fontsize=12)
plt.title("Bar Plot",fontsize=15)
plt.bar(x,y,color="red",width=0.5)   ### bar plot
plt.show()

#### Scatter Plot 
x1 = [1,2,3,4,5]
y1 = [2,5,2,6,8]
x2 = [1,2,3,4,5]
y2 = [4,5,8,9,10]
plt.xlabel("X Axis",fontsize=12,fontstyle='italic')
plt.ylabel("Y Axis",fontsize=12)
plt.title("Line Plot",fontsize=15,fontname='Courier')
plt.scatter(x1,y1,color="red",label="First Graph")   ### line plot
plt.scatter(x2,y2,color="blue",s=150,marker="*",label="Second Graph")   ### line plot
plt.plot(x2,y2,color="blue")
plt.legend(loc=2)
#plt.axis('off')
plt.show()

### Histogtram
import numpy as np
sample = np.random.randint(10,100,30)
plt.hist(sample,rwidth=0.7)
plt.show()

### Pie Chart
plt.figure(figsize=(7,7))
slices = [10,20,50,30,34]
act = ["A","B","C","D","E"]
cols = ["red","blue","green","pink","yellow"]
plt.pie(slices,labels=act,colors=cols,
        autopct="%1.2f%%",explode=(0,0.2,0,0.1,0))
plt.show()
##########################################################################################################################

#Lab 7.Write a Python program that creates a mxn integer arrayand Prints its attributes using Numpy

a = np.array([[12,4,5],[23,45,66],[45,34,23]]) 
print("Printing Array")

print()
print(a)

print()
print("Printing numpy array Attributes")
print("1>. Array Shape is: ", a.shape)
print("2>. Array dimensions are ", a.ndim)
print("3>. Datatype of array  is ", a.dtype)
print("4>. Length of each element of array in bytes is ", a.itemsize)
print("5>. Number of elements in array are ", a.size)
##########################################################################################################################

#Lab8 Linear Regression
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate some sample data
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
y = np.array([2, 4, 6, 8, 10])

# Create a linear regression model and fit the data
model = LinearRegression()
model.fit(X, y)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Use the model to predict new data
new_data = np.array([[6, 12], [7, 14]])
predictions = model.predict(new_data)
print("Predictions:", predictions)
##########################################################################################################################

#Lab9 Logistic Regression
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# Generate some sample data
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
y = np.array([0, 0, 0, 1, 1])

# Create a logistic regression model and fit the data
model = LogisticRegression()
model.fit(X, y)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Use the model to predict new data
new_data = np.array([[6, 12], [7, 14]])
predictions = model.predict(new_data)
print("Predictions:", predictions)

# Plot the data points and the linear regression line
plt.scatter(X[:, 0], y, color='blue')
plt.plot(X[:, 0], model.predict(X), color='red')
plt.title("Logistic Regression Model")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
##########################################################################################################################

#Lab 10: Write a Python program to demonstrate Timeseries analysis with Pandas.

import pandas as pd
df = pd.read_csv("aapl.csv",parse_dates=["Date"], index_col="Date")
df.head()

df.index

# (1) Partial Date Index: Select Specific Months Data
df.loc['2017-06-30']

df.loc["2017-01"]

df.loc['2017-06'].head() 

df.loc['2017-06'].Close.mean()

df.loc['2017'].head(2) 

df['2017-01-08':'2017-01-03']

df.loc['2017-01']

df['Close'].resample('M').mean().head()

df.loc['2016-07']

%matplotlib inline
df['Close'].plot()

df['Close'].resample('M').mean().plot(kind='bar')
##########################################################################################################################

#Lab 11: Write a Python program to demonstrate Data Visualization using Seaborn.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])
cars_data.dropna(axis=0, inplace= True)
sns.set(style="darkgrid")
sns.regplot(x=cars_data['Age'],y= cars_data['Price'])

# scatter plot of Price and age without the regeression fitline
sns.regplot(x=cars_data['Age'],y= cars_data['Price'],marker='*' ,fit_reg=False)

# scatter plot of price vs age by FuelType
sns.lmplot(x='Age',y='Price', data=cars_data, fit_reg=False, hue="FuelType", legend=True, palette="Set1")

#distribution of the variable 'Age'
sns.histplot(cars_data['Age'])

sns.histplot(cars_data['Age'],kde=False, bins = 8)

# frequency distribution of categorical variable 'fuelType'
sns.countplot(x="FuelType", data=cars_data)

#grouped bar plot of FuelType and Automatic
sns.countplot(x="FuelType", data=cars_data, hue="Automatic")

# frequency distribution of categorical variable 'fuelType'
sns.countplot(x="FuelType", data=cars_data)

#Box and Whiskers plot for nmerical vs categorial variables
sns.boxplot(x= cars_data["FuelType"], y=cars_data["Price"])

#grouped box and Whiskers plot
sns.boxplot(x= cars_data["FuelType"], y=cars_data["Price"], hue="Automatic", data= cars_data)
##########################################################################################################################
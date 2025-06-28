# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Loading dataset
data = pd.read_csv("/Users/shanks/Downloads/archive (3)/ifood_df.csv")

# Printing the dataset
print(data)

#Removing rows with missing values
data1 = data[~data.isnull()].copy()

#Print the data with no missing values
print(data1)

#It show how many missing values each column has
print(data.isnull().sum())

# Displaying the Income values as proportions
print(data.Income.value_counts(normalize=True))

# Plotting a histogram of Age distribution
(data.Age.value_counts(normalize=True).plot.hist())
plt.show()

# Displaying basic statistics for the Kidhome column
print(data.Kidhome.describe())

# Creating a scatter plot between Income and Age
plt.scatter(data.Income, data.Age)
plt.show()

# Create a pairplot for Income, Age, and Kidhome
sns.pairplot(data=data, vars=['Income', 'Age', 'Kidhome'])
plt.show()

# Calculating correlation matrix between Income, Age, and Kidhome
data2 = data[['Income', 'Age', 'Kidhome']].corr()

# Plot a heatmap of the correlation matrix with data2
sns.heatmap(data2, annot=True, cmap="Reds")
plt.show()

#Creating a pivot table
result = pd.pivot_table(data=data, index="Income", columns="Age", values="Kidhome")

#Printing the pivot table
print(result)

# Plotting pivot table using a heatmap
sns.heatmap(result, annot=True, cmap="Reds", center=0.117)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display the first few rows and summary statistics
print("Initial Data Overview:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop rows where 'Embarked' is missing
df.dropna(subset=['Embarked'], inplace=True)

# Convert categorical variables to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Exploratory Data Analysis (EDA)
# Survival Rate
survival_rate = df['Survived'].mean()
print(f"\nOverall Survival Rate: {survival_rate:.2f}")

# Survival Rate by Gender
survival_by_gender = df.groupby('Sex')['Survived'].mean()
print("\nSurvival Rate by Gender:")
print(survival_by_gender)

# Survival Rate by Passenger Class
survival_by_pclass = df.groupby('Pclass')['Survived'].mean()
print("\nSurvival Rate by Passenger Class:")
print(survival_by_pclass)

# Survival Rate by Age Group
bins = [0, 12, 18, 30, 40, 50, 60, 100]
labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Middle-Aged', 'Senior', 'Elder']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Plot survival rate by age group
plt.figure(figsize=(10, 6))
survival_by_age_group = df.groupby('AgeGroup')['Survived'].mean()
survival_by_age_group.plot(kind='bar', color='skyblue')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.show()

# Survival Rate by Embarked
survival_by_embarked = df.groupby('Embarked')['Survived'].mean()
print("\nSurvival Rate by Embarked Port:")
print(survival_by_embarked)

# Correlations
correlations = df.corr()
print("\nCorrelation Matrix:")
print(correlations)

# Plot heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

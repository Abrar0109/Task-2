import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Initial Data Overview:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe(include='all'))


print("\nMissing Values:")
print(df.isnull().sum())


df['Age'].fillna(df['Age'].median(), inplace=True)


df.dropna(subset=['Embarked'], inplace=True)


df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


survival_rate = df['Survived'].mean()
print(f"\nOverall Survival Rate: {survival_rate:.2f}")


survival_by_gender = df.groupby('Sex')['Survived'].mean()
print("\nSurvival Rate by Gender:")
print(survival_by_gender)


survival_by_pclass = df.groupby('Pclass')['Survived'].mean()
print("\nSurvival Rate by Passenger Class:")
print(survival_by_pclass)


bins = [0, 12, 18, 30, 40, 50, 60, 100]
labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Middle-Aged', 'Senior', 'Elder']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)


plt.figure(figsize=(10, 6))
survival_by_age_group = df.groupby('AgeGroup')['Survived'].mean()
survival_by_age_group.plot(kind='bar', color='skyblue')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.show()


survival_by_embarked = df.groupby('Embarked')['Survived'].mean()
print("\nSurvival Rate by Embarked Port:")
print(survival_by_embarked)


correlations = df.corr()
print("\nCorrelation Matrix:")
print(correlations)


plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

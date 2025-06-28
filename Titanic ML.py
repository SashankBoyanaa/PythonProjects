#Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#Loading data
train = pd.read_csv('/Users/shanks/Downloads/titanic/train.csv')
test = pd.read_csv('/Users/shanks/Downloads/titanic/test.csv')

#Exploring the dataset
train.head()
train.info()
train.describe()

#Separating num and categorical columns for analysis
df_num = train[['Age', 'SibSp', 'Parch', 'Fare']]
df_cat = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

#ploting histgrams for num values
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()

print(pd.pivot_table(train, index='Survived', values=['Age', 'SibSp', 'Parch', 'Fare']))

#Ploting bar graph for categorical values
for i in df_cat.columns:
    sns.barplot(x=df_cat[i].value_counts().index, y=df_cat[i].value_counts().values)
    plt.title(i)
    plt.show()

#Checking survival count across different categories
print(pd.pivot_table(train, index='Survived', columns='Pclass', values='Ticket', aggfunc='count'))
print(pd.pivot_table(train, index='Survived', columns='Sex', values='Ticket', aggfunc='count'))
print(pd.pivot_table(train, index='Survived', columns='Embarked', values='Ticket', aggfunc='count'))

#Checking how many cabins were assigned to each passenger
train['cabin_multiple'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
print(train['cabin_multiple'].value_counts())
print(pd.pivot_table(train, index='Survived', columns='cabin_multiple', values='Ticket', aggfunc='count'))

#checking First letter of cabin
train['cabin_adv'] = train.Cabin.apply(lambda x: str(x)[0])

#checking if ticket is purely numeric?
train['numeric_ticket'] = train.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

#Letters in ticket
train['ticket_letters'] = train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 0)
print(train['numeric_ticket'].value_counts())
pd.set_option("display.max_rows", None)
print(train['ticket_letters'].value_counts())
print(pd.pivot_table(train, index='Survived', columns='numeric_ticket', values='Ticket', aggfunc='count'))

#Extracting title from name
train['name_title'] = train.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
test['name_title'] = test.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#Filling missing values
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna('S')
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

#Converting categorical values to num vales
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#Group uncommon titles under 'Rare'
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
train['name_title'] = train['name_title'].apply(lambda x: x if x in common_titles else 'Rare')
test['name_title'] = test['name_title'].apply(lambda x: x if x in common_titles else 'Rare')

#Encoded titles
train = pd.get_dummies(train, columns=['name_title'], drop_first=True)
test = pd.get_dummies(test, columns=['name_title'], drop_first=True)

#Matching dummy columns in test set
missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    if 'name_title_' in col:
        test[col] = 0

#Creating new family features
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

# Bin Fare and Age into categories
train['FareBin'] = pd.qcut(train['Fare'], 4, labels=False)
test['FareBin'] = pd.qcut(test['Fare'], 4, labels=False)

train['AgeBin'] = pd.cut(train['Age'].astype(int), 5, labels=False)
test['AgeBin'] = pd.cut(test['Age'].astype(int), 5, labels=False)

#Dropping irrelevant features
columns_to_drop = ['cabin_multiple', 'cabin_adv', 'numeric_ticket', 'ticket_letters']
train = train.drop(columns=[col for col in columns_to_drop if col in train.columns])
test = test.drop(columns=[col for col in columns_to_drop if col in test.columns])

#drop original Age and Fare
train = train.drop(columns=['Age', 'Fare'])
test = test.drop(columns=['Age', 'Fare'])

#test columns match train
test = test[train.columns.drop(['Survived'])]

#select final features for the model
features = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'FareBin', 'AgeBin'] + \
           [col for col in train.columns if 'name_title_' in col]

X = train[features]
y = train['Survived']
X_test = test[features]

#Building a Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

#Validate with cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.5f}")

#Train model and predict
model.fit(X, y)
predictions = model.predict(X_test)

#creating submission file
output = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
output.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
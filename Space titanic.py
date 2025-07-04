import pandas as pd
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier

train = pd.read_csv('/Users/shanks/Downloads/titanic/train.csv')
test = pd.read_csv('/Users/shanks/Downloads/titanic/test.csv')

df_num = train[['Age', 'SibSp', 'Parch', 'Fare']]
df_cat = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

train['cabin_multiple'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
train['cabin_adv'] = train.Cabin.apply(lambda x: str(x)[0])
train['numeric_ticket'] = train.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
train['ticket_letters'] = train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 0)

train['name_title'] = train.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
test['name_title'] = test.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna('S')
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
train['name_title'] = train['name_title'].apply(lambda x: x if x in common_titles else 'Rare')
test['name_title'] = test['name_title'].apply(lambda x: x if x in common_titles else 'Rare')

train = pd.get_dummies(train, columns=['name_title'], drop_first=True)
test = pd.get_dummies(test, columns=['name_title'], drop_first=True)

missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    if 'name_title_' in col:
        test[col] = 0

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

train['FareBin'] = pd.qcut(train['Fare'], 4, labels=False)
test['FareBin'] = pd.qcut(test['Fare'], 4, labels=False)

train['AgeBin'] = pd.cut(train['Age'].astype(int), 5, labels=False)
test['AgeBin'] = pd.cut(test['Age'].astype(int), 5, labels=False)

columns_to_drop = ['cabin_multiple', 'cabin_adv', 'numeric_ticket', 'ticket_letters']
train = train.drop(columns=[col for col in columns_to_drop if col in train.columns])
test = test.drop(columns=[col for col in columns_to_drop if col in test.columns])

train = train.drop(columns=['Age', 'Fare'])
test = test.drop(columns=['Age', 'Fare'])

test = test[train.columns.drop(['Survived'])]

features = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'FareBin', 'AgeBin'] + \
           [col for col in train.columns if 'name_title_' in col]

X = train[features]
y = train['Survived']
X_test = test[features]


model = CatBoostClassifier(verbose=0, random_state=42)
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.5f}")

model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
output.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")

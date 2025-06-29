# Importing libraries
import pandas as pd
import numpy as np


#Preprocessing and importing models from scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

#Setting a random seed
SEED = 42

#Loading data
train = pd.read_csv("/Users/shanks/Downloads/spaceship-titanic/train.csv")
test = pd.read_csv("/Users/shanks/Downloads/spaceship-titanic/test.csv")

#Storing lengths of train and test datasets
train_len = len(train)
test_len = len(test)

#Combining train and test datasets for preprocessing
def combine_df(train, test):
    combined_df = pd.concat([train, test], ignore_index=True)
    return combined_df

#Separating the combined dataset back into train and test using the saved lengths
def sep_df(combined_df, train_len, test_len):
    train = combined_df.iloc[:train_len]
    test = combined_df.iloc[train_len:]
    return train, test

#Merging datasets
full = combine_df(train, test)

#Columns that should be 0
zero_col = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

#Columns that should be False
false_col = ["VIP"]

#Filling NaN values incolumns with 0
condition = (full['CryoSleep'] == True)
full.loc[~condition, zero_col] = full.loc[~condition, zero_col].fillna(0)

#Filling missing VIP values with False
for col in false_col:
    full.loc[~condition, col] = full.loc[~condition, col].fillna(False).astype(bool)


condition = (full[zero_col].eq(0).all(axis=1))
full.loc[condition, 'CryoSleep'] = full.loc[condition, 'CryoSleep'].fillna(True).astype(bool)

# Filling remaining missing values with False
full['CryoSleep'] = full['CryoSleep'].fillna(False).astype(bool)

#Splitting 'Cabin' column into 'Deck', 'CabinNum', and 'Side'
full[['Deck', 'CabinNum', 'Side']] = full['Cabin'].str.split('/', expand=True)

#Converting CabinNum to num, replace errors with NaN, then fill NaNs with 0
full['CabinNum'] = pd.to_numeric(full['CabinNum'], errors='coerce').fillna(0)

#Defining which values are num and which are categorical
numerical_features = ['CabinNum', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Feature engineering
#Fill missing values with mean
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


#Filling missing values with the most frequent value and applying one-hot encoding to convert to num format
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combining the above pipelines into a column transformer
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

#Separating the combined data back into train and test
train, test = sep_df(full, train_len, test_len)

#Extracting the target variable and convert to boolean type
y = train['Transported'].astype(bool)

#Dropping target column from train and test datasets
train_df = train.drop(columns=['Transported'])
test_df = test.drop(columns=['Transported'])

#Applying the preprocessor to transform the training and test data
transformed_data_train = preprocessor.fit_transform(train_df)
transformed_data_test = preprocessor.transform(test_df)

#Getting the one-hot encoded feature names
feature_names = preprocessor.named_transformers_['categorical']\
    .named_steps['encoder'].get_feature_names_out(input_features=categorical_features)

#Combining all values
all_feature_names = numerical_features + list(feature_names)

#Converting transformed values into DataFrames
transformed_train_df = pd.DataFrame(transformed_data_train, columns=all_feature_names)
transformed_test_df = pd.DataFrame(transformed_data_test, columns=all_feature_names)

#Spliting train data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(transformed_train_df, y, test_size=0.2, random_state=SEED)

#Initializing 3 Gradient Boost models with  different parameters
gbm_model_1 = GradientBoostingClassifier(
    n_estimators=70, learning_rate=0.1, max_features='sqrt',
    max_depth=5, random_state=SEED, min_samples_split=2,
    min_samples_leaf=3, subsample=0.5, loss='exponential'
)

gbm_model_2 = GradientBoostingClassifier(
    n_estimators=70, learning_rate=0.1, max_features='log2',
    max_depth=5, random_state=SEED, min_samples_split=2,
    min_samples_leaf=3, subsample=0.5, loss='log_loss'
)

gbm_model_3 = GradientBoostingClassifier(
    n_estimators=70, learning_rate=0.1, max_features='log2',
    max_depth=5, random_state=SEED, min_samples_split=2,
    min_samples_leaf=3, subsample=0.5, loss='exponential'
)

#Training all 3 models on train set
gbm_model_1.fit(X_train, y_train)
gbm_model_2.fit(X_train, y_train)
gbm_model_3.fit(X_train, y_train)

#Generating predictions from each model on the validation set
gbm_1_predictions = gbm_model_1.predict(X_test)
gbm_2_predictions = gbm_model_2.predict(X_test)
gbm_3_predictions = gbm_model_3.predict(X_test)

#Stacking predictions to create input features for the meta-model
stacked_features = np.column_stack((gbm_1_predictions, gbm_2_predictions, gbm_3_predictions))

#Using Logistic Regression as the meta-model in stacking
meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_test)

#Using base models to generate predictions on the actual test dataset
gbm_1_base_preds = gbm_model_1.predict(transformed_test_df)
gbm_2_base_preds = gbm_model_2.predict(transformed_test_df)
gbm_3_base_preds = gbm_model_3.predict(transformed_test_df)

#Stacking base model predictions for the test set
stacked_base_preds = np.column_stack((gbm_1_base_preds, gbm_2_base_preds, gbm_3_base_preds))

#Making final prediction using the meta-model on the stacked test predictions
ensemble_predictions = meta_model.predict(stacked_base_preds)

#Prepare submission file with PassengerId and predicted Transported values
output = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': ensemble_predictions
})
# Evaluating base models on validation set
print("GradientBoosting Model 1 Accuracy:", accuracy_score(y_test, gbm_1_predictions))
print("GradientBoosting Model 2 Accuracy:", accuracy_score(y_test, gbm_2_predictions))
print("GradientBoosting Model 3 Accuracy:", accuracy_score(y_test, gbm_3_predictions))

# Cross-validation on one base model
cv_scores = cross_val_score(gbm_model_1, X_train, y_train, cv=5)
print("Cross-validation scores (Model 1):", cv_scores)
print("Average CV score:", np.mean(cv_scores))

# Accuracy of meta model
print("Meta Model (LogReg) Accuracy:", accuracy_score(y_test, meta_model.predict(stacked_features)))
#Saving submission files
output.to_csv('submission.csv', index=False)

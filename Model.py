import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from imblearn.ensemble import BalancedBaggingClassifier


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import RandomOverSampler
from collections import Counter



st.subheader('Assignment 2 (Group - 10')

st.subheader('Arshdeep Pranav Mohit Naveen')
@st.cache(allow_output_mutation=True)
def load_data():
    df=pd.read_csv('crunchbase.csv')
    return df
st.title('Crunchbase Data')
df=load_data()
num_comp=df.count()[0]

if st.sidebar.checkbox("Preprocessed Data"):
    st.subheader("Data")
    st.write(df)
    st.write('Number of initial companies:' , num_comp)

# Removing the companies that are older than 7 years or younger than 3 years of age
df=df[(df.age <2556) & (df.age > 1095)] 
no_companies=df.count()[0]


#Creating the target variable
def success(row):
    if row['ipo'] == True or row['is_acquired']== True and row['is_closed']== False:
        return 1
    else:
        return 0
df['success']= df.apply(success, axis=1)
#Remove is_aquired, ipo, is_closed cloumn    
df = df.drop('is_acquired', axis=1)
df = df.drop('ipo', axis=1)
df = df.drop('is_closed', axis=1)

#Replacing missing values of all the degrees with 0 
df['mba_degree'] = df['mba_degree'].fillna(0)
df['ms_degree'] = df['ms_degree'].fillna(0)
df['phd_degree'] = df['phd_degree'].fillna(0)
df['other_degree'] = df['other_degree'].fillna(0)

#Creating  feature named total degrees
df['number_degrees'] = df['mba_degree'] + df['phd_degree'] + df['ms_degree'] + df['other_degree']

#Dropping all the other degree features using which total degree was created
df = df.drop(labels=["mba_degree", "ms_degree","phd_degree", "mba_degree","other_degree"], axis=1)

#Identifying numerical features in the dataset
#Note: mba_degree, ms_degree, phd_degree and other_degree were replaced by total_degrees
numerical_features = ['average_funded', 'total_rounds', 'average_participants', 'products_number',
'acquired_companies', 'offices', 'age', 'number_degrees']
numerical_features_and_target = numerical_features + ['success']

if st.sidebar.checkbox("Show correlation matrix"):
    st.subheader("Correlation matrix")
# Show the correlations of the numerical features with one
# another and with the target
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(df[numerical_features_and_target].corr(), annot=True,
    fmt=".2f", ax=ax)
    ax.set_title("Correlations of numerical features and target")
    st.pyplot(fig)
#identifying categorical features in the dataset
categorical_features = ['category_code', 'country_code', 'state_code']
#check missing values:
def missing_values_ratios(df):
# Calculate the ratio of missing values in each column
    return df.isna().sum() / len(df)

if st.sidebar.checkbox("Show missing values ratios"):
    st.subheader("Missing values ratios")
    st.write(missing_values_ratios(df))
    
#choose a reasonable default for missing values
#If no information is provided there must be atleast 1 product or atleast 1 office for wach company thus I think it's better to replace missing values or products_number and offices with 1
df['products_number'] = df['products_number'].fillna(1)
df['offices'] = df['offices'].fillna(1)
#For aquired companies We will assume the missing value to be zero because this information is worth sharing if someone has aquired a company they would love to share that. So since information is not available it's better to assume that the missing value will be 0.
df['acquired_companies'] = df['acquired_companies'].fillna(0)
#For no information on average funding I think so it's better to replace it with mean because I don't think so if there is no infomration comoany has no funding at all.
mean_value=df['average_funded'].mean()  
df['average_funded'].fillna(value=mean_value, inplace=True)
#For feature like category_code it's better to replace the missing value with "other" which is one of the values that this feature takes.
df['category_code'] = df['category_code'].fillna("other")
#Some features are not useful like company_id. So it's better to drop it.
df = df.drop('company_id', axis=1)
number_comp=df.count()[0]

if st.sidebar.checkbox("Processed Data"):
    st.subheader("Final Processed Data before Modeling")
    st.write('Number of companies after processing :' , number_comp)
    st.write(df)    
    
#Modelling

#creating a pipeline for pre-processing the features
def pre_processor(numerical_features, categorical_features):
#pipeline for pre-processing numerical features. This pipeline also deals with missing values
  numerical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())])
#pipeline for pre-processing categorical features
  categorical_transformer = Pipeline(steps = [('onehot', OneHotEncoder(handle_unknown='ignore'))])
#combining two pipelines into one
  preprocessor = ColumnTransformer(transformers = [('num', numerical_transformer, numerical_features),
  ('cat', categorical_transformer, categorical_features)])
  return preprocessor



# pipeline for training the model
preprocessor = pre_processor(numerical_features, categorical_features)
type_of_classifier = st.sidebar.radio("Select type of classifier", ("Decision Tree", "RandomForestClassifier"))
if type_of_classifier == "Decision Tree":
    classifier = DecisionTreeClassifier(random_state = 1, max_depth = 16, min_samples_split = 2, min_samples_leaf = 1, class_weight='balanced',)
elif type_of_classifier == "RandomForestClassifier":
    classifier = RandomForestClassifier(random_state = 1, max_depth = 19,  n_estimators = 500, min_samples_split = 2, min_samples_leaf = 1, class_weight='balanced', n_jobs=-1)
model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', classifier)])
# Get features X and target y (y will not be part of input)
X = df.drop('success', axis=1)
y = df['success']

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

if st.sidebar.checkbox("Samples after Random Over Sampler"):
    st.subheader("data after Random_Sampler")
    st.write(sorted(Counter(y_resampled).items()))  

# Choosing train and test datasets
# Split the data into 70% training and 30% testing, we use the stratified sampling because our sample is not orderd
X_resampled_train, X_resampled_test, y_resampled_train, y_resampled_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)

# Evaluate the model against the test dataset
model.fit(X_resampled_train, y_resampled_train)
y_pred = model.predict(X_resampled_test)

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_resampled_test, y_pred)
fig, ax = plt.subplots(figsize=(6.4, 4.8))
sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax,cmap=plt.cm.Greens)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion matrix')
ax.xaxis.set_ticklabels(['No', 'Yes'])
ax.yaxis.set_ticklabels(['No', 'Yes'])
st.pyplot(fig)

# Use k-fold cross-validation to evaluate the model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_resampled_train, y_resampled_train,
scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'],cv=cv, n_jobs=-1)
if st.sidebar.checkbox("Show model performance (cross validation)"):
    st.subheader('Model performance (cross validation)')
# Performance scores for each fold
    st.write("Scores for each fold (only positive class):")
    df_scores = pd.DataFrame(scores).transpose()
    df_scores['mean'] = df_scores.mean(axis=1)
    st.dataframe(df_scores)
#Checking for Imbalance

counts = df['success'].value_counts()
fig, ax = plt.subplots(figsize=(6.4, 2.4))
sns.barplot(x=counts.index, y=counts.values, ax=ax)
ax.set_xticklabels(['No', 'Yes'])
ax.set_xlabel('Should invest or not')
ax.set_ylabel('Number of companies')
ax.set_title('Class imbalance')

should_invest = counts[1]
should_not_invest = counts[0]

if st.sidebar.checkbox('Check imbalance'):
    st.subheader('Class imbalance')
    st.pyplot(fig)
    st.write('Degree of imbalance: %.1f to 1 non-investable companies to investable-companies' % (should_not_invest/should_invest))
    st.write('Number of investable companies :' ,should_invest)
    st.write('Number of non-investable companies:', should_not_invest)
    
# Get feature names
# Optional: set verbose_feature_names_out to False

model.named_steps['preprocessor'].verbose_feature_names_out=False
feature_names = \
              model.named_steps['preprocessor'].get_feature_names_out()

# Get feature importance from the model
feature_importance = \
                   model.named_steps['classifier'].feature_importances_
# Create a dataframe with the feature names and their importance
feature_importance = pd.DataFrame({'feature': feature_names,
                                   'importance': model['classifier'].feature_importances_})
feature_importance = feature_importance.sort_values('importance',ascending=False)
feature_importance = feature_importance.head(10)

# Plot the feature importance as horizontal bar chart
if st.sidebar.checkbox("Show feature importance"):
    st.subheader('Ranking of Features')
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.barplot(x='importance', y='feature', data=feature_importance,
    ax=ax, color='blue')
    ax.set_xlabel('Feature importance')
    ax.set_ylabel('Feature')
    ax.set_title("Feature importance")
    st.pyplot(fig)
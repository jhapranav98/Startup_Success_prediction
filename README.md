Crunchbase Startup Success Prediction

This Streamlit app analyzes and predicts the success of startups using Crunchbase data. It applies data preprocessing, feature engineering, and machine learning models (Decision Tree and Random Forest) to identify which factors contribute most to startup success.

🚀 Features

Interactive Streamlit UI

View raw and preprocessed data

Visualize missing values and correlations

Check class imbalance

Compare model performance via cross-validation

Display feature importance rankings

Data Preprocessing

Filters startups aged between 3 and 7 years

Creates a binary success target variable based on IPO/acquisition

Handles missing values using contextual imputation strategies

Performs one-hot encoding and scaling using pipelines

Machine Learning

Uses Decision Tree and Random Forest classifiers

Applies Random Over Sampler to handle class imbalance

Performs 5-fold Stratified Cross Validation

Displays confusion matrix and key metrics (accuracy, precision, recall, f1, ROC-AUC)

🧠 Dataset

The app expects a CSV file named crunchbase.csv containing the following key columns:

age

ipo

is_acquired

is_closed

average_funded

total_rounds

average_participants

products_number

acquired_companies

offices

category_code

country_code

state_code

Degree-related columns (mba_degree, ms_degree, phd_degree, other_degree)

Make sure the file is in the same directory as the script.

⚙️ How It Works

Load Data
The app loads crunchbase.csv using pandas and displays initial stats.

Data Cleaning

Filters out startups too old or too young

Fills missing values with mean, zeros, or category placeholders

Drops unnecessary columns

Feature Engineering

Creates number_degrees

Defines success (IPO or acquired and not closed)

Modeling

Choose classifier via sidebar (Decision Tree or Random Forest)

Train/test split (70/30) with stratified sampling

Oversample using RandomOverSampler

Evaluation

Shows confusion matrix and model metrics

Displays feature importance for top predictors

📊 Visualizations

Correlation heatmap

Class imbalance bar chart

Confusion matrix

Top 10 feature importance chart

🧩 Dependencies

Install all required packages:

pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

▶️ Run the App

In your terminal:

streamlit run app.py


(Rename your script to app.py or update the file path accordingly.)


📝 Notes

The app uses @st.cache(allow_output_mutation=True) for caching data load.

The model uses class_weight='balanced' to handle imbalance along with oversampling.

To explore preprocessing steps or adjust thresholds, use the sidebar controls in Streamlit.

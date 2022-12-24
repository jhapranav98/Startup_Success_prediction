import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('crunchbase.csv', encoding='utf-8', delimiter=',')
    return df

st.title('Companies that got acquired or issued Ipo')
df = load_data()
#df = pre_process(df)
if st.checkbox("Show the data"):
    # use streamlit to display the dataframe
    st.write(df)

st.write("""Is Acquired VS Not Aquired""")
# default figure size is 6.4 x 4.8
fig, ax = plt.subplots(figsize=(6.4, 2.4))
# bar plot in seaborn, use black color
sns.countplot(x ='is_acquired', data = df)
st.pyplot(fig)

st.write("""Issued IPO VS Didn't issued IPO""")
# use seaborn to plot the count plot
fig, ax = plt.subplots(figsize=(6.4, 2.4))
# bar plot in seaborn, use black color
sns.countplot(x ='ipo', data = df)
st.pyplot(fig)

st.write("""Company is Closed VS Company is not closed""")
# use seaborn to plot the count plot
fig, ax = plt.subplots(figsize=(6.4, 2.4))
# bar plot in seaborn, use black color
sns.countplot(x ='is_closed', data = df)
st.pyplot(fig)






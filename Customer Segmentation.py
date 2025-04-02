import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import numpy as np

st.title("Customer Segmentation Group 4 ExcelR")
st.write("Deployment Stage")


def show_duplicate_rows(data):
    # Check for duplicate rows
    duplicate_rows = data[data.duplicated(keep=False)]
    # Sort the duplicate rows by all columns to group duplicates together
    duplicate_rows = duplicate_rows.sort_values(by=list(data.columns))

    if len(duplicate_rows) > 0:
        print('All rows which occur more than once :')
        display(duplicate_rows)
    else:
        print('There are no duplicate rows in the dataset.')

def describe(data):
    # Describe the data, transpose, and fill NaN values
    description = data.describe(include='all').T.round(3)

    required_columns = ['top', 'freq', 'first', 'last']
    if all(col in description.columns for col in required_columns):
        description.drop(['top','freq','first','last'],axis=1,inplace=True)
    elif 'unique' not in description.columns:
        description.insert(1, 'unique', np.nan)

    description['unique'] = description['unique'].fillna(data.nunique())

    # Get data types from data.info()
    data_types = data.dtypes

    # Insert data types as a new column
    description.insert(1, 'dtype', data_types)

    # Calculate null value percentages
    null_percentages = data.isnull().sum() / len(data) * 100

    # Insert null value percentages as the 2nd column
    description.insert(1, 'na %', null_percentages.round(3))

    # Print the resulting DataFrame
    display(description)


data = pd.read_excel('marketing_campaign1.xlsx')

option = st.selectbox(
    'Show',
    ('head', 'tail', 'sample'))

number = st.slider('number of rows', 1, 100, 5)


if option == 'head':
    st.write(data.head(number))
elif option == 'tail':
    st.write(data.tail(number))
elif option == 'sample':
    st.write(data.sample(number))


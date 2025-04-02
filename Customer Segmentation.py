import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import numpy as np

st.title("Customer Segmentation Group 4 ExcelR")
st.write("Deployment Stage")


# Modify the show_duplicate_rows function to use Streamlit elements
def show_duplicate_rows(data):
    # Check for duplicate rows
    duplicate_rows = data[data.duplicated(keep=False)]

    if not duplicate_rows.empty:
        # Sort the duplicate rows by all columns to group duplicates together
        duplicate_rows = duplicate_rows.sort_values(by=list(data.columns))
        st.write("Duplicate rows found:")
        st.dataframe(duplicate_rows) # Use st.dataframe for tables
    else:
        st.write("There are no duplicate rows in the dataset.")

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
    return(description)


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

if st.button('Describe Data'):
    st.dataframe(describe(data))


# Add this section to your Streamlit app script:
st.markdown("---") # Optional: Adds a visual separator
st.subheader("Check for Duplicate Rows")
if st.button('Show Duplicate Rows'):
    show_duplicate_rows(data) # Call the modified function

# Add this section to your Streamlit app script

st.markdown("---") # Optional: Adds a visual separator
st.subheader("Handle Missing Values")

# Check if 'data' is loaded (assuming you load it into st.session_state as 'data')
# If not using session state, replace st.session_state.data with your DataFrame variable (e.g., 'data')
# but be aware changes might not persist across interactions without session state.
if 'data' in st.session_state and isinstance(st.session_state.data, pd.DataFrame):
    # Check if 'Income' column exists
    if 'Income' in st.session_state.data.columns:
        initial_na_count = st.session_state.data['Income'].isnull().sum()
        st.write(f"Current missing values in 'Income': {initial_na_count}")

        # Ask the question and provide the button only if there are NAs
        if initial_na_count > 0:
            st.write("Do you want to fill missing 'Income' values with the median?")
            if st.button("Yes, fill NA in 'Income'"):
                try:
                    median_income = st.session_state.data['Income'].median()
                    # Fill NA by assigning back to the column in session state
                    # Using inplace=True with session state can sometimes be tricky, direct assignment is safer
                    st.session_state.data['Income'] = st.session_state.data['Income'].fillna(median_income)

                    # Confirmation
                    final_na_count = st.session_state.data['Income'].isnull().sum()
                    st.success(f"Missing 'Income' values filled with median ({median_income:.3f}).")
                    st.write(f"Missing values in 'Income' after filling: {final_na_count}")
                    # Force rerun to ensure all parts of the app update
                    st.rerun()
                except Exception as e:
                     st.error(f"An error occurred while filling NA values: {e}")
        else:
            st.info("No missing values found in the 'Income' column to fill.")
    else:
        st.warning("Column 'Income' not found in the dataset.")
else:
    st.warning("Data not loaded or not available in session state.")

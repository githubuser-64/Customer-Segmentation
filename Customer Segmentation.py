import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import numpy as np

st.set_page_config(layout="wide") # Optional: Use wider layout

st.title("Customer Segmentation Group 4 ExcelR")
st.write("Deployment Stage")

# --- Functions (show_duplicate_rows, describe) ---
# (Keep your existing functions as they were in the previous correct version)
def show_duplicate_rows(data_df):
    duplicate_rows = data_df[data_df.duplicated(keep=False)]
    if not duplicate_rows.empty:
        duplicate_rows = duplicate_rows.sort_values(by=list(data_df.columns))
        return duplicate_rows # Return the dataframe
    else:
        return None # Return None if no duplicates

def describe(data_df):
    description = data_df.describe(include='all').T.round(3)
    required_columns = ['top', 'freq', 'first', 'last']
    cols_to_drop = [col for col in required_columns if col in description.columns]
    if cols_to_drop:
        description = description.drop(columns=cols_to_drop)
    if 'unique' not in description.columns:
        description.insert(1, 'unique', data_df.nunique())
    else:
        description['unique'] = description['unique'].fillna(data_df.nunique())
    data_types = data_df.dtypes
    description.insert(1, 'dtype', data_types)
    null_percentages = data_df.isnull().sum() * 100 / len(data_df)
    description.insert(1, 'na %', null_percentages.round(3))
    return description
# --- End Functions ---

# --- Load data ---
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the Excel file: {e}")
        return None

# --- Initialize Session State ---
# Raw data
if 'data' not in st.session_state:
    st.session_state.data = load_data('marketing_campaign1.xlsx')

# State for description output
if 'description_df' not in st.session_state:
    st.session_state.description_df = None
if 'show_description' not in st.session_state:
    st.session_state.show_description = False

# State for duplicate rows output
if 'duplicate_rows_df' not in st.session_state:
    st.session_state.duplicate_rows_df = None
if 'show_duplicates' not in st.session_state:
    st.session_state.show_duplicates = False
# --- End Initialization ---


# --- Main App Logic ---
if st.session_state.data is not None:

    # --- Part 1: Data Exploration View (Always updates on interaction) ---
    st.header("Explore Data")
    col1, col2 = st.columns([1, 3]) # Adjust column widths
    with col1:
        option = st.selectbox(
            'Show', ('head', 'tail', 'sample'), key='display_option'
        )
        max_rows = len(st.session_state.data)
        current_slider_value = st.session_state.get('num_rows', 5)
        number = st.slider(
            'Number of rows', 1, max_rows, min(current_slider_value, max_rows), key='num_rows'
        )

    with col2:
        st.subheader(f"Displaying {option} ({number} rows)")
        current_df = st.session_state.data
        if option == 'head':
            st.dataframe(current_df.head(number))
        elif option == 'tail':
            st.dataframe(current_df.tail(number))
        elif option == 'sample':
            sample_size = min(number, len(current_df))
            if sample_size > 0:
                 st.dataframe(current_df.sample(sample_size))
            else:
                 st.write("No data to sample.")

    st.markdown("---")

    # --- Part 2: Action Buttons ---
    st.header("Data Analysis Actions")
    b_col1, b_col2, b_col3 = st.columns(3)

    with b_col1:
        if st.button('Generate Data Description'):
            # Calculate, store result in session state, set flag to show
            st.session_state.description_df = describe(st.session_state.data)
            st.session_state.show_description = True
            # Clear other persistent outputs if desired when generating a new one
            # st.session_state.show_duplicates = False

    with b_col2:
        if st.button('Find Duplicate Rows'):
            # Calculate, store result, set flag
            st.session_state.duplicate_rows_df = show_duplicate_rows(st.session_state.data)
            st.session_state.show_duplicates = True
            # Clear other persistent outputs if desired
            # st.session_state.show_description = False

    with b_col3:
        st.subheader("Handle Missing Income") # Keep this simpler for now
        if 'Income' in st.session_state.data.columns:
            initial_na_count = st.session_state.data['Income'].isnull().sum()
            st.write(f"Missing 'Income': {initial_na_count}")
            if initial_na_count > 0:
                if st.button("Fill Income NA with Median", key='fill_income_na'):
                    median_income = st.session_state.data['Income'].median()
                    st.session_state.data['Income'] = st.session_state.data['Income'].fillna(median_income)
                    st.success(f"Filled NA with median ({median_income:.3f}).")
                    # Clear potentially outdated persistent results after modifying data
                    st.session_state.show_description = False
                    st.session_state.show_duplicates = False
                    st.rerun()
            else:
                st.info("No missing 'Income' values.")
        else:
            st.warning("Column 'Income' not found.")

    st.markdown("---")

    # --- Part 3: Persistent Display Areas ---
    st.header("Analysis Results")

    # Display Area for Description
    # Check the flag first
    if st.session_state.get('show_description', False):
        st.subheader("Data Description")
        # Check if the result actually exists
        if st.session_state.description_df is not None:
            st.dataframe(st.session_state.description_df)
            # Add button to hide this specific result
            if st.button("Hide Description", key="hide_desc_btn"):
                st.session_state.show_description = False
                st.rerun() # Rerun immediately to hide it
        else:
            # This case shouldn't happen with current logic, but good practice
            st.warning("Description requested but no data available.")
            st.session_state.show_description = False # Reset flag


    # Display Area for Duplicates
    # Check the flag
    if st.session_state.get('show_duplicates', False):
        st.subheader("Duplicate Rows Analysis")
        # Check if result exists (it could be None if no duplicates found)
        if st.session_state.duplicate_rows_df is not None:
            st.write("Duplicate rows found:")
            st.dataframe(st.session_state.duplicate_rows_df)
        else:
            st.info("No duplicate rows were found in the dataset.")
        # Add button to hide this result section
        if st.button("Hide Duplicates Info", key="hide_dupl_btn"):
            st.session_state.show_duplicates = False
            st.rerun() # Rerun immediately to hide it


else:
    st.error("Data could not be loaded. Cannot proceed.")

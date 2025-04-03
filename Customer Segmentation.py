import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

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


def plot_histogram(data, columns_to_exclude=None, columns_to_include=None, title=None):
    """
    Create histograms for columns in the data with statistical annotations.

    Parameters:
    -----------
    data : pandas.DataFrame
        The data to plot
    columns_to_exclude : list, optional
        Columns to exclude from plotting
    columns_to_include : list, optional
        Specific columns to include in plotting

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the histograms
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Determine which columns to plot
    if columns_to_exclude is not None:
        data_to_plot = data.drop(columns=columns_to_exclude)
    elif columns_to_include is not None:
        data_to_plot = data[columns_to_include]
    else:
        data_to_plot = data

    # Calculate number of rows and columns for the subplots
    n_cols = 3  # Keep 3 columns as in original
    n_rows = len(data_to_plot.columns) // n_cols + (len(data_to_plot.columns) % n_cols > 0)

    # Adjust figure size based on number of plots
    # Each subplot gets roughly 5x4 inches
    fig_width = n_cols * 5
    fig_height = n_rows * 4

    # Create subplots with the calculated dimensions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Convert axes to array for easier indexing (handle the case where there's only one row)
    if n_rows == 1:
        axes = np.array([axes])
    elif n_rows > 1 and n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    else:
        axes = np.array(axes).reshape(n_rows, n_cols)

    if not title:
        plt.suptitle('Histograms of Numerical Variables', fontsize=16)
    else:
        plt.suptitle(title, fontsize=16)


    # Plot each histogram
    for i, col in enumerate(data_to_plot.columns):
        row_idx = i // n_cols
        col_idx = i % n_cols

        data_to_plot[col].hist(bins=30, ax=axes[row_idx, col_idx], edgecolor='black')
        axes[row_idx, col_idx].set_title(col)

        # Calculate statistics
        mu = round(np.mean(data_to_plot[col]), 3)
        sigma = round(np.std(data_to_plot[col]), 3)
        min_val = round(np.min(data_to_plot[col]), 3)
        percentile_25 = round(np.percentile(data_to_plot[col], 25), 3)
        median = round(np.median(data_to_plot[col]), 3)
        percentile_75 = round(np.percentile(data_to_plot[col], 75), 3)
        max_val = round(np.max(data_to_plot[col]), 3)


        # Create a text box outside the plot in the top right
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Format the statistics exactly as requested, one below the other
        textstr = '\n'.join([
            r'$\mu=%.2f$' % mu,
            r'$\sigma=%.2f$' % sigma,
            r'$\mathrm{min}=%.2f$' % min_val,
            r'$\mathrm{25}=%.2f$' % percentile_25,
            r'$\mathrm{median}=%.2f$' % median,
            r'$\mathrm{75}=%.2f$' % percentile_75,
            r'$\mathrm{max}=%.2f$' % max_val
        ])

        # Position the text box outside the plot in the top right
        axes[row_idx, col_idx].text(1.05, 0.95, textstr, transform=axes[row_idx, col_idx].transAxes,
                                   fontsize=9, verticalalignment='top', bbox=props)

    # Hide empty subplots
    for i in range(len(data_to_plot.columns), n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        fig.delaxes(axes[row_idx, col_idx])

    # Add legend at the top of the figure, outside the subplots
    handles = [plt.Rectangle((0,0),1,1, color='skyblue') for _ in data_to_plot.columns]
    labels = []
    for col in data_to_plot.columns:
        mean_val = data_to_plot[col].mean()
        labels.append(f"{col}: μ={mean_val:.2f}")

    # Move the legend to the very top of the figure
    # fig.legend(handles, labels, loc='upper center',
    #            bbox_to_anchor=(0.5, 1.05),
    #            fontsize=10, ncol=min(3, len(data_to_plot.columns)))

    # First adjust the spacing between subplots for the statistics boxes
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    # Then add right margin for the statistics boxes and top margin for the legend
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])

    # plt.show()

    return fig


def plot_stacked_histogram(data_copy):

    # Calculate total spending for each sector
    primary_expenditure = data_copy['MntFishProducts'] + data_copy['MntMeatProducts'] + data_copy['MntFruits']
    secondary_expenditure = data_copy['MntSweetProducts'] + data_copy['MntWines']
    tertiary_expenditure = data_copy['MntGoldProds']

    # Determine common bin edges based on the full range of all expenditures
    all_expenditures = np.concatenate([primary_expenditure, secondary_expenditure, tertiary_expenditure])
    bins = np.histogram_bin_edges(all_expenditures, bins=15)  # Common bins

    # Plot stacked histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        [tertiary_expenditure, secondary_expenditure, primary_expenditure],
        bins=bins,
        stacked=True,
        color=['blue', 'yellow', 'orange'],
        label=['Tertiary (gold)', 'Secondary (wine,sweets)', 'Primary (fish,meat,fruit)']
    )

    plt.xlabel("Spending Amount")
    plt.ylabel("Frequency")
    plt.title("Histogram of Spending Across Sectors")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    return fig


def plot_bar_chart(data, columns_to_include=None, rotation=0):
    # If columns_to_include is provided, drop those columns
    if columns_to_include is not None:
        data_to_plot = data[columns_to_include]
    else:
        data_to_plot = data

    # Create a subplot for all specified columns
    num_columns = len(data_to_plot.columns)
    num_rows = (num_columns // 3) + (num_columns % 3 > 0)  # Calculate number of rows needed for 3 columns
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))  # Create subplots
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Create bar charts for each specified column
    try:
        for i, column in enumerate(data_to_plot.columns):
            counts = data[column].value_counts().sort_index()  # Sort by index
            counts.plot(kind='bar', color='skyblue', ax=axes[i])
            axes[i].set_title(f'{column}', fontsize=16)
            axes[i].set_xlabel(None, fontsize=14)
            axes[i].set_ylabel('Frequency', fontsize=14)
            axes[i].tick_params(axis='x', rotation=rotation)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)

            # Add value labels on top of the bars
            for j in range(len(counts)):
                axes[i].text(j, counts.iloc[j], counts.iloc[j], ha='center', va='bottom')

    except:
        print('columns_to_include has to be a list.')

    for j in range(num_columns, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust layout to prevent overlap
    # plt.show()
    return fig

# Define function for pie charts
def plot_pie_chart(data,title,variable_name,variable_name_X,variable_name_Y):
    series_data = data.sum()
    total_value = data.sum().sum()
    # labels =

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(series_data, labels=series_data.index, autopct='%1.1f%%', startangle=90, labeldistance = 1.03)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add text annotation in the top-left corner
    plt.text(
        variable_name_X,  # x-coordinate (adjust as needed for positioning)
        variable_name_Y,   # y-coordinate (adjust as needed for positioning)
        f"{variable_name}: {total_value:.2f}",  # Display total value, formatted to 2 decimal places
        fontsize=10,
        fontweight='bold',
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}  # Optional: Add a box around the text
    )

    # plt.show()
    return fig
    

# Define function for scatter plots for pairs of variables
def plot_scatterplot(data, column_for_color=None, columns_to_exclude=None, heading_of_plot=None):
    # If columns_to_exclude is provided, drop those columns
    if columns_to_exclude is not None:
        data_to_plot = data.drop(columns=columns_to_exclude)
    else:
        data_to_plot = data  # Use the entire DataFrame if no columns are excluded

    sns.pairplot(data_to_plot, diag_kind='kde', markers='o', hue=column_for_color, palette='viridis')
    plt.suptitle(f'{heading_of_plot}', y=1.02)
    # plt.show()
    return fig
    

def plot_customer_segmentation_3d_interactive_scatter_plot(data, x='Recency', y='Frequency', z='Monetary', hue='Segment', palette='Plotly'):
    # Create the 3D scatter plot
    fig = px.scatter_3d(
        data,
        x=x,
        y=y,
        z=z,
        color=hue,
        color_discrete_sequence=px.colors.qualitative.Plotly if palette == 'Plotly' else palette,
        title='Customer Segmentation 3D Scatter Plot',
        labels={x: x, y: y, z: z},
        hover_name=hue,
        hover_data=[x, y, z, hue]
    )

    # Update layout to move legend outside the plot
    fig.update_layout(
        legend=dict(
            x=1.05,  # Move legend to the right of the plot
            y=1.0,
            xanchor='left',  # Use 'left' for xanchor
            yanchor='top'    # Use 'top' for yanchor
        ),
        margin=dict(r=200)  # Add margin to accommodate the legend
    )

    # Show the plot
    # fig.show()
    return fig

def plot_customer_segmentation_3d_scatter_plot(data, x='Recency', y='Frequency', z='Monetary', hue='Segment', palette='Set1', s=100):
    plt.figure(figsize=(10, 8))
    ax = plt.figure().add_subplot(111, projection='3d')

    # Get unique segments and their corresponding colors
    segments = data[hue].unique()
    color_map = sns.color_palette(palette, n_colors=len(segments))
    color_dict = {segment: color for segment, color in zip(segments, color_map)}

    for segment in segments:
        segment_data = data[data[hue] == segment]
        ax.scatter(
            segment_data[x],
            segment_data[y],
            segment_data[z],
            label=segment,
            color=color_dict[segment],
            s=s
        )

    ax.set_title('Customer Segmentation 3D Scatter Plot')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.legend(title='Segment', loc='upper right')

    # plt.show()
    return fig

# Function to rank values into 5 groups
def rank_group(series):
    return pd.qcut(series, q=5, labels=[1, 2, 3, 4, 5])


# Function to rank group for campaigns
def rank_group_cmps(series):
    # Define the custom mapping
    rank_mapping = {
        0: 1,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 5
    }

    # Map the series to ranks
    return series.map(rank_mapping)


# Define 'Segment' based on specified criteria
def segment_customer(row):
    if row['Recency_Rank'] <= 2 and row['Monetary_Rank'] >= 4 and row['Frequency_Rank'] >= 4:
        return 'Champions'
    elif row['Monetary_Rank'] >= 4 and row['Frequency_Rank'] >= 3:
        return 'Potential Loyalist'
    elif row['Recency_Rank'] <=2 and row['Monetary_Rank'] >= 3:
        return 'New Customers'
    elif row['Recency_Rank'] <= 2 and row['Frequency_Rank'] <= 3 and row['Monetary_Rank'] <= 3:
        return 'At Risk Customers'
    else:
        return 'Potential Churn Customers'


# Define 'Segment' based on modified criteria
def segment_customer_modified(row):
    if row['Recency_Rank'] <= 2 and row['Monetary_Rank'] >= 4 and row['Frequency_Rank'] >= 4 and row['Income_Rank'] >= 4 and row['Avg_Purchase_Per_Month_To_Monthly_Income_Ratio_Rank'] >= 4 and row['NumAcceptedCmps_Rank'] >= 4:
        return 'High Value'
    elif row['Monetary_Rank'] >= 4 and row['Frequency_Rank'] >= 3 and row['Income_Rank'] >= 4 and row['NumAcceptedCmps_Rank'] >= 2 and row['Avg_Purchase_Per_Month_To_Monthly_Income_Ratio_Rank'] >= 4:
        return 'Potential High Value'
    elif row['Recency_Rank'] <=2 and row['Monetary_Rank'] >= 3 and row['Income_Rank'] < 4:
        return 'High Spenders'
    elif row['Recency_Rank'] <= 2 and row['Frequency_Rank'] <= 3 and row['Monetary_Rank'] <= 3 and row['Avg_Purchase_Per_Month_To_Monthly_Income_Ratio_Rank'] < 4:
        return 'Low Spenders'
    elif row['Recency_Rank'] > 2 and row['Frequency_Rank'] <= 3 and row['Monetary_Rank'] <= 3 and row['Avg_Purchase_Per_Month_To_Monthly_Income_Ratio_Rank'] < 4 and row['NumAcceptedCmps_Rank'] < 2:
        return 'Require Different Campaign Strategy'
    else:
        return 'Uncategorized Customers'


# Create a scatter plot with color-coding for segments
def plot_customer_segmentation_scatter_plot(data,x='Recency',y='Frequency',hue='Segment',palette='Set1',s=100,):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=palette,
        s=s,
    )
    plt.title('Customer Segmentation Scatter Plot')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    plt.legend(title='Segment', loc='upper right')
    # plt.show()
    return fig

def plot_correlation_heatmap(data, threshold_correlation_value):
    plt.figure(figsize=(30, 30))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix[(correlation_matrix >= threshold_correlation_value) | (correlation_matrix <= -threshold_correlation_value)], annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Heatmap of Correlation Matrix', fontsize=16)
    # plt.show()
    return fig

def top_n_correlations(df, n_value_for_top_n_correlations=5, correlation_threshold=None):
    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Convert to series and filter out self-correlations
    top_correlations = correlation_matrix.unstack()
    top_correlations = top_correlations[top_correlations.index.get_level_values(0) != top_correlations.index.get_level_values(1)]

    # Remove duplicates (since corr(A,B) = corr(B,A))
    top_correlations = top_correlations[~top_correlations.index.duplicated(keep='first')]

    # Further filter to keep only one pair of each correlation
    mask = top_correlations.index.map(lambda x: x[0] < x[1])
    top_correlations = top_correlations[mask]

    # If correlation_threshold is provided, calculate percentage below threshold
    if correlation_threshold is not None:
        # Get total number of correlations
        total_correlations = len(top_correlations)

        # Count correlations below threshold (using absolute values)
        below_threshold = sum(abs(top_correlations) < correlation_threshold)

        # Calculate and print percentage
        percentage_below = (below_threshold / total_correlations) * 100
        print(f"{percentage_below:.2f}% of correlations are below the threshold of {correlation_threshold}")

    # Get top N correlations by absolute value
    top_correlations_abs = abs(top_correlations).sort_values(ascending=False)
    top_n_abs = top_correlations_abs.head(n_value_for_top_n_correlations)

    # Return original values (not absolute) for top N correlations
    top_n = top_correlations.loc[top_n_abs.index].sort_values(ascending=False)

    return top_n


# Define VIF to calculate VIF
def calculate_VIF(data,columns_to_exclude=None):
    # Calculate VIF for each feature in the dataset
    if columns_to_exclude:
        X = data.drop(columns_to_exclude, axis=1)
    else:
        X = data
    print('We drop the response_variable or dependent_variable or target variable in VIF calculations.\nWe need to find collinearity among independent variables.')
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Print the VIF results
    display(vif_data)
    display('VIF values <= 10', vif_data[vif_data['VIF'] <= 10])
    display('VIF values == inf', vif_data[vif_data['VIF'] == np.inf])
    display('fraction of non-multi-collinear variables:', round(len(vif_data[vif_data['VIF'] <= 10])/len(vif_data),3))


def create_pps_correlation_graph(df, target_column, feature_columns=None):
    """
    Create a graph comparing correlation and PPS values for features against a target variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing both features and target variable
    target_column : str
        The name of the target variable column
    feature_columns : list, optional
        List of feature columns to analyze. If None, all columns except target will be used.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    # If feature columns not specified, use all columns except target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    # Calculate correlation values (using Pearson correlation)
    corr_vals = []
    for feat in feature_columns:
        if pd.api.types.is_numeric_dtype(df[feat]) and pd.api.types.is_numeric_dtype(df[target_column]):
            corr_vals.append(df[feat].corr(df[target_column]))
        else:
            # For non-numeric data, correlation is not applicable
            corr_vals.append(0)

    # Calculate PPS matrix for all features
    pps_matrix = pps.matrix(df)

    # Extract PPS values for the target column
    pps_vals = []
    for feat in feature_columns:
        # Filter the matrix for the current feature and target
        filtered_rows = pps_matrix[(pps_matrix['x'] == feat) & (pps_matrix['y'] == target_column)]

        # If we found a match, extract the ppscore
        if not filtered_rows.empty:
            pps_vals.append(filtered_rows['ppscore'].values[0])
        else:
            # If no match found, set to 0
            pps_vals.append(0)

    # Create feature names for the x-axis
    feature_names = [f"{feat} -> {target_column}" for feat in feature_columns]

    # Sort by PPS values if desired
    if len(feature_names) > 10:  # Only sort if there are many features
        # Convert values to float to ensure they can be compared
        pps_vals_float = [float(v) if v is not None else 0 for v in pps_vals]
        indices = np.argsort(pps_vals_float)[::-1]  # Sort in descending order
        feature_names = [feature_names[i] for i in indices]
        pps_vals = [pps_vals[i] for i in indices]
        corr_vals = [corr_vals[i] for i in indices]

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(feature_names, corr_vals, marker='o', label='Correlation')
    ax.plot(feature_names, pps_vals, marker='o', label='PPS')
    ax.set_xlabel(f'Feature predictive power score of {target_column}')
    ax.set_ylabel('Score')
    ax.set_title(f'Correlation & PPS Values for Features (Target: {target_column})')
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    plt.tight_layout()

    return fig

def generate_all_pps_correlation_graphs(df, target_columns, feature_columns=None):
    """
    Generate PPS and correlation graphs for multiple target variables.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing features and target variables
    target_columns : list
        List of target variable column names
    feature_columns : list, optional
        List of feature columns to analyze. If None, all columns except targets will be used.

    Returns:
    --------
    None (displays plots)
    """
    # If feature columns not specified, use all columns except target columns
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in target_columns]

    # Generate a plot for each target variable
    for target in target_columns:
        try:
            fig = create_pps_correlation_graph(df, target, feature_columns)
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"Error creating graph for target {target}: {str(e)}")


def plot_pca_scree_plot(data):

    # Perform PCA
    pca = PCA()
    pca.fit(data)

    # Get explained variance ratios
    explained_variance_ratios = pca.explained_variance_ratio_

    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratios)

    # Plot the scree plot
    plt.figure(figsize=(22, 8))


    ### Bar Part Start

    # Plot individual explained variance
    # bars = plt.bar(range(1, len(explained_variance_ratios) + 1), explained_variance_ratios, alpha=0.7, align='center', label='Individual Explained Variance')

    # # Annotate each bar with the percentage of explained variance
    # for i, bar in enumerate(bars):
    #     height = bar.get_height()
    #     plt.annotate(f'{height * 100:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
    #                  xytext=(0, 3),  # 3 points vertical offset
    #                  textcoords="offset points",
    #                  ha='center', va='bottom')

    ### Bar Part End


    # Plot individual explained variance as a line
    plt.plot(range(1, len(explained_variance_ratios) + 1), explained_variance_ratios, marker='o', linestyle='-', label='Individual Explained Variance')

    # Annotate each point with the percentage of explained variance
    for i, var in enumerate(explained_variance_ratios):
        plt.annotate(f'{var * 100:.2f}%', xy=(i + 1, var),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Plot cumulative explained variance
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='red', label='Cumulative Explained Variance')

    # Add labels and title
    plt.title('PCA Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance')
    plt.xticks(range(1, len(explained_variance_ratios) + 1))
    plt.grid(True)
    plt.legend(loc='best')

    # Show the plot
    plt.show()
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
# --- Add this within the # --- Initialize Session State --- section ---

# State for Products Analysis
if 'product_hist_fig' not in st.session_state: st.session_state.product_hist_fig = None
if 'show_product_hist' not in st.session_state: st.session_state.show_product_hist = False
if 'product_pie_fig' not in st.session_state: st.session_state.product_pie_fig = None
if 'show_product_pie' not in st.session_state: st.session_state.show_product_pie = False
if 'product_stacked_hist_fig' not in st.session_state: st.session_state.product_stacked_hist_fig = None
if 'show_product_stacked_hist' not in st.session_state: st.session_state.show_product_stacked_hist = False
if 'product_log_hist_fig' not in st.session_state: st.session_state.product_log_hist_fig = None
if 'show_product_log_hist' not in st.session_state: st.session_state.show_product_log_hist = False

# State for People Analysis
if 'people_hist_fig' not in st.session_state: st.session_state.people_hist_fig = None
if 'show_people_hist' not in st.session_state: st.session_state.show_people_hist = False
if 'people_income_log_hist_fig' not in st.session_state: st.session_state.people_income_log_hist_fig = None
if 'show_people_income_log_hist' not in st.session_state: st.session_state.show_people_income_log_hist = False
if 'people_sorted_df' not in st.session_state: st.session_state.people_sorted_df = None
if 'show_people_sorted' not in st.session_state: st.session_state.show_people_sorted = False

# State for Age Analysis
if 'age_dt_birth_df' not in st.session_state: st.session_state.age_dt_birth_df = None
if 'show_age_dt_birth' not in st.session_state: st.session_state.show_age_dt_birth = False
if 'age_hist_fig' not in st.session_state: st.session_state.age_hist_fig = None
if 'show_age_hist' not in st.session_state: st.session_state.show_age_hist = False
if 'enroll_year_hist_fig' not in st.session_state: st.session_state.enroll_year_hist_fig = None
if 'show_enroll_year_hist' not in st.session_state: st.session_state.show_enroll_year_hist = False

# State for Circumstances Analysis
if 'circumstances_bar_fig' not in st.session_state: st.session_state.circumstances_bar_fig = None
if 'show_circumstances_bar' not in st.session_state: st.session_state.show_circumstances_bar = False

# State for Promotion Analysis
if 'promotion_bar_fig' not in st.session_state: st.session_state.promotion_bar_fig = None
if 'show_promotion_bar' not in st.session_state: st.session_state.show_promotion_bar = False

# State for Place Analysis
if 'place_bar_fig' not in st.session_state: st.session_state.place_bar_fig = None
if 'show_place_bar' not in st.session_state: st.session_state.show_place_bar = False
if 'place_pie_fig' not in st.session_state: st.session_state.place_pie_fig = None
if 'show_place_pie' not in st.session_state: st.session_state.show_place_pie = False

# State for Feature Engineering and People's Actions
if 'data_copy_engineered' not in st.session_state: st.session_state.data_copy_engineered = None # Store the engineered df
if 'feature_engineering_done' not in st.session_state: st.session_state.feature_engineering_done = False
if 'actions_eng_df' not in st.session_state: st.session_state.actions_eng_df = None
if 'show_actions_eng' not in st.session_state: st.session_state.show_actions_eng = False
if 'actions_tenure_hist_fig' not in st.session_state: st.session_state.actions_tenure_hist_fig = None
if 'show_actions_tenure_hist' not in st.session_state: st.session_state.show_actions_tenure_hist = False
if 'actions_no_visits_df' not in st.session_state: st.session_state.actions_no_visits_df = None
if 'show_actions_no_visits' not in st.session_state: st.session_state.show_actions_no_visits = False
if 'actions_recency_hist_fig' not in st.session_state: st.session_state.actions_recency_hist_fig = None
if 'show_actions_recency_hist' not in st.session_state: st.session_state.show_actions_recency_hist = False
if 'actions_complain_visits_bar_fig' not in st.session_state: st.session_state.actions_complain_visits_bar_fig = None
if 'show_actions_complain_visits_bar' not in st.session_state: st.session_state.show_actions_complain_visits_bar = False

# State for Cost/Revenue
if 'cost_rev_hist_fig' not in st.session_state: st.session_state.cost_rev_hist_fig = None
if 'show_cost_rev_hist' not in st.session_state: st.session_state.show_cost_rev_hist = False

# State for RFM Analysis
if 'rfm_calculated' not in st.session_state: st.session_state.rfm_calculated = False # Flag if RFM is done on engineered data
if 'rfm_df' not in st.session_state: st.session_state.rfm_df = None
if 'show_rfm_df' not in st.session_state: st.session_state.show_rfm_df = False
if 'rfm_ranks_df' not in st.session_state: st.session_state.rfm_ranks_df = None
if 'show_rfm_ranks' not in st.session_state: st.session_state.show_rfm_ranks = False
if 'rfm_scatter_fig' not in st.session_state: st.session_state.rfm_scatter_fig = None
if 'show_rfm_scatter' not in st.session_state: st.session_state.show_rfm_scatter = False
if 'rfm_3d_scatter_fig' not in st.session_state: st.session_state.rfm_3d_scatter_fig = None
if 'show_rfm_3d_scatter' not in st.session_state: st.session_state.show_rfm_3d_scatter = False
if 'rfm_3d_interactive_fig' not in st.session_state: st.session_state.rfm_3d_interactive_fig = None
if 'show_rfm_3d_interactive' not in st.session_state: st.session_state.show_rfm_3d_interactive = False

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


    # --- Add these sections within the main `if st.session_state.data is not None:` block ---
    
    # =========================================
    # Section: Products Analysis
    # =========================================
    st.markdown("---")
    st.header("Products Analysis")
    with st.expander("Show Products Analysis", expanded=False):
        st.markdown("""
        ### Products
        | Attribute | Description |
        |---|---|
        | MntWines | Amount spent on wine in last 2 years |
        | MntFruits | Amount spent on fruits in last 2 years |
        | MntMeatProducts | Amount spent on meat in last 2 years |
        | MntFishProducts | Amount spent on fish in last 2 years |
        | MntSweetProducts | Amount spent on sweets in last 2 years |
        | MntGoldProds | Amount spent on gold in last 2 years |
        """)
    
        # --- Action Buttons for Products ---
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            if st.button("Product Spending Histogram", key="show_prod_hist_btn"):
                data_to_plot = st.session_state.data.iloc[:,9:15]
                st.session_state.product_hist_fig = plot_histogram(data_to_plot) # Assumes returns fig
                st.session_state.show_product_hist = True
        with col_p2:
             if st.button("Product Shares Pie Chart", key="show_prod_pie_btn"):
                st.session_state.product_pie_fig = plot_pie_chart(st.session_state.data.iloc[:,9:15],title='Shares of Different Product Categories',variable_name='Total Value of Products',variable_name_X=-1.4,variable_name_Y=1) # Assumes returns fig
                st.session_state.show_product_pie = True
        with col_p3:
            if st.button("Product Stacked Histogram", key="show_prod_stack_btn"):
                data_copy_prod = st.session_state.data.copy() # Use a local copy
                st.session_state.product_stacked_hist_fig = plot_stacked_histogram(data_copy_prod) # Assumes returns fig
                st.session_state.show_product_stacked_hist = True
        with col_p4:
            if st.button("Log-Transformed Product Hist", key="show_prod_log_btn"):
                 data_log_prod = st.session_state.data.copy()
                 for col in ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']:
                     data_log_prod[f'log_transformed_{col}'] = np.log1p(data_log_prod[col])
                 st.session_state.product_log_hist_fig = plot_histogram(data_log_prod.iloc[:,29:35]) # Assumes returns fig
                 st.session_state.show_product_log_hist = True
    
        # --- Persistent Display Area for Products ---
        if st.session_state.get('show_product_hist', False):
            st.subheader("Product Spending Distribution")
            if st.session_state.product_hist_fig:
                st.pyplot(st.session_state.product_hist_fig) # Use st.pyplot for matplotlib
                if st.button("Hide Product Histogram", key="hide_prod_hist_btn"):
                    st.session_state.show_product_hist = False
                    st.rerun()
            else: st.warning("Generate histogram first.")
    
        if st.session_state.get('show_product_pie', False):
            st.subheader("Product Shares Pie Chart")
            if st.session_state.product_pie_fig:
                st.plotly_chart(st.session_state.product_pie_fig) # Use st.plotly_chart for Plotly
                if st.button("Hide Product Pie Chart", key="hide_prod_pie_btn"):
                    st.session_state.show_product_pie = False
                    st.rerun()
            else: st.warning("Generate pie chart first.")
    
        if st.session_state.get('show_product_stacked_hist', False):
            st.subheader("Product Stacked Histogram")
            if st.session_state.product_stacked_hist_fig:
                st.pyplot(st.session_state.product_stacked_hist_fig)
                if st.button("Hide Product Stacked Hist", key="hide_prod_stack_btn"):
                    st.session_state.show_product_stacked_hist = False
                    st.rerun()
            else: st.warning("Generate stacked histogram first.")
    
        if st.session_state.get('show_product_log_hist', False):
            st.subheader("Log-Transformed Product Spending Distribution")
            if st.session_state.product_log_hist_fig:
                st.pyplot(st.session_state.product_log_hist_fig)
                if st.button("Hide Log Product Hist", key="hide_prod_log_btn"):
                    st.session_state.show_product_log_hist = False
                    st.rerun()
            else: st.warning("Generate log histogram first.")
    
    # =========================================
    # Section: People Analysis
    # =========================================
    st.markdown("---")
    st.header("People Analysis")
    with st.expander("Show People Analysis", expanded=False):
        st.markdown("""
        ### People
        | Attribute | Description |
        |---|---|
        | ID | Customer's unique identifier |
        | Year_Birth | Customer's birth year |
        | Education | Customer's education level |
        | Marital_Status | Customer's marital status |
        | Income | Customer's yearly household income |
        | Kidhome | Number of children in customer's household |
        | Teenhome | Number of teenagers in customer's household |
        | Dt_Customer | Date of customer's enrollment with the company |
        """)
    
        # --- Action Buttons for People ---
        col_pe1, col_pe2, col_pe3 = st.columns(3)
        with col_pe1:
            if st.button("Birth Year/Income Histogram", key="show_people_hist_btn"):
                data_to_plot = st.session_state.data[['Year_Birth', 'Income']].dropna() # Handle potential NA in Income for plot
                st.session_state.people_hist_fig = plot_histogram(data_to_plot)
                st.session_state.show_people_hist = True
        with col_pe2:
            if st.button("Log-Transformed Income Hist", key="show_people_income_log_btn"):
                data_log_income = st.session_state.data[['Income']].dropna().copy() # Handle NA
                data_log_income['log_transformed_income'] = np.log1p(data_log_income['Income'])
                st.session_state.people_income_log_hist_fig = plot_histogram(data_log_income[['log_transformed_income']])
                st.session_state.show_people_income_log_hist = True
        with col_pe3:
            if st.button("Show Data Sorted by Birth Year", key="show_people_sorted_btn"):
                data_sorted = st.session_state.data.sort_values(by='Year_Birth')
                st.session_state.people_sorted_df = data_sorted
                st.session_state.show_people_sorted = True
    
        # --- Persistent Display Area for People ---
        if st.session_state.get('show_people_hist', False):
            st.subheader("Birth Year and Income Distribution")
            if st.session_state.people_hist_fig:
                st.pyplot(st.session_state.people_hist_fig)
                if st.button("Hide Birth/Income Hist", key="hide_people_hist_btn"):
                    st.session_state.show_people_hist = False
                    st.rerun()
            else: st.warning("Generate histogram first.")
    
        if st.session_state.get('show_people_income_log_hist', False):
            st.subheader("Log-Transformed Income Distribution")
            if st.session_state.people_income_log_hist_fig:
                st.pyplot(st.session_state.people_income_log_hist_fig)
                if st.button("Hide Log Income Hist", key="hide_people_income_log_btn"):
                    st.session_state.show_people_income_log_hist = False
                    st.rerun()
            else: st.warning("Generate log income histogram first.")
    
        if st.session_state.get('show_people_sorted', False):
            st.subheader("Data Sorted by Birth Year")
            if st.session_state.people_sorted_df is not None:
                st.dataframe(st.session_state.people_sorted_df)
                if st.button("Hide Sorted Data", key="hide_people_sorted_btn"):
                    st.session_state.show_people_sorted = False
                    st.rerun()
            else: st.warning("Generate sorted data first.")
    
    # =========================================
    # Section: Age Analysis
    # =========================================
    st.markdown("---")
    st.header("Age and Enrollment Analysis")
    with st.expander("Show Age/Enrollment Analysis", expanded=False):
        st.markdown("""
        ### How old was the customer on the day of customer's enrollment (enrollment day) with the company?
        ### Is there a relationship between age on enrollment day and their actual age?
        We will use a substitute which is the age of all the customers on the last enrollment day in the data set to keep things within the context of the dataset.
        """)
    
        # --- Action Buttons for Age ---
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            if st.button("Show Enrollment Date & Birth Year", key="show_age_dt_birth_btn"):
                st.session_state.age_dt_birth_df = st.session_state.data[['Dt_Customer','Year_Birth']]
                st.session_state.show_age_dt_birth = True
        with col_a2:
            if st.button("Show Age Histograms", key="show_age_hist_btn"):
                 data_copy_age = st.session_state.data.copy()
                 data_copy_age['Dt_Customer'] = pd.to_datetime(data_copy_age['Dt_Customer'], errors='coerce') # Ensure datetime
                 data_copy_age.dropna(subset=['Dt_Customer', 'Year_Birth'], inplace=True) # Drop rows where calculation isn't possible
                 data_copy_age['Age_On_Enrollment_Day'] = data_copy_age['Dt_Customer'].dt.year - data_copy_age['Year_Birth']
                 latest_enrollment_date = data_copy_age['Dt_Customer'].max()
                 data_copy_age['Age_On_Last_Enrollment_Date'] = latest_enrollment_date.year - data_copy_age['Year_Birth']
                 st.session_state.age_hist_fig = plot_histogram(data_copy_age[['Age_On_Enrollment_Day','Age_On_Last_Enrollment_Date']])
                 st.session_state.show_age_hist = True
        with col_a3:
             if st.button("Show Enrollment Year Histogram", key="show_enroll_year_btn"):
                 data_copy_enroll = st.session_state.data.copy()
                 data_copy_enroll['Dt_Customer'] = pd.to_datetime(data_copy_enroll['Dt_Customer'], errors='coerce') # Ensure datetime
                 data_copy_enroll.dropna(subset=['Dt_Customer'], inplace=True)
                 data_copy_enroll['Year_Of_Enrollment'] = data_copy_enroll['Dt_Customer'].dt.year
                 st.session_state.enroll_year_hist_fig = plot_histogram(data_copy_enroll[['Year_Of_Enrollment']])
                 st.session_state.show_enroll_year_hist = True
    
        st.markdown("""
        *Roughly, the same number of people are in the histogram bins. There seems to be a strong similarity between Age Enrollment and Age Current.*
        """)
        st.markdown("""
        *Now we understand why the Histograms of `Age_On_Enrollment_Day` and `Age_On_Last_Enrollment_Date` are so similar. **All of them were registered in just three years, 2012 to 2014.** *
        """)
    
    
        # --- Persistent Display Area for Age ---
        if st.session_state.get('show_age_dt_birth', False):
            st.subheader("Enrollment Date & Birth Year")
            if st.session_state.age_dt_birth_df is not None:
                st.dataframe(st.session_state.age_dt_birth_df)
                if st.button("Hide Date/Birth Data", key="hide_age_dt_birth_btn"):
                    st.session_state.show_age_dt_birth = False
                    st.rerun()
            else: st.warning("Generate Date/Birth data first.")
    
        if st.session_state.get('show_age_hist', False):
            st.subheader("Age on Enrollment vs. Age on Last Enrollment Date")
            if st.session_state.age_hist_fig:
                st.pyplot(st.session_state.age_hist_fig)
                if st.button("Hide Age Histograms", key="hide_age_hist_btn"):
                    st.session_state.show_age_hist = False
                    st.rerun()
            else: st.warning("Generate age histograms first.")
    
        if st.session_state.get('show_enroll_year_hist', False):
            st.subheader("Year of Enrollment Distribution")
            if st.session_state.enroll_year_hist_fig:
                st.pyplot(st.session_state.enroll_year_hist_fig)
                if st.button("Hide Enrollment Year Hist", key="hide_enroll_year_btn"):
                    st.session_state.show_enroll_year_hist = False
                    st.rerun()
            else: st.warning("Generate enrollment year histogram first.")
    
    # =========================================
    # Section: Circumstances Analysis
    # =========================================
    st.markdown("---")
    st.header("People's Circumstances Analysis")
    with st.expander("Show Circumstances Analysis", expanded=False):
        st.markdown("### People's Circumstances or People's Situation")
    
        # --- Action Button ---
        if st.button("Show Circumstances Bar Chart", key="show_circ_bar_btn"):
            cols_to_plot = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'Complain','NumWebVisitsMonth']
            # Ensure columns exist and handle potential errors if plot_bar_chart needs specific types
            valid_cols = [col for col in cols_to_plot if col in st.session_state.data.columns]
            if valid_cols:
                 st.session_state.circumstances_bar_fig = plot_bar_chart(st.session_state.data[valid_cols],rotation=15)
                 st.session_state.show_circumstances_bar = True
            else:
                 st.warning(f"Not all required columns found in data: {cols_to_plot}")
    
    
        # --- Persistent Display ---
        if st.session_state.get('show_circumstances_bar', False):
            st.subheader("Distribution of Circumstances")
            if st.session_state.circumstances_bar_fig:
                st.pyplot(st.session_state.circumstances_bar_fig) # Or st.plotly_chart if it returns Plotly
                if st.button("Hide Circumstances Chart", key="hide_circ_bar_btn"):
                    st.session_state.show_circumstances_bar = False
                    st.rerun()
            else: st.warning("Generate circumstances chart first.")
    
    # =========================================
    # Section: Promotion Analysis
    # =========================================
    st.markdown("---")
    st.header("Promotion Acceptance Analysis")
    with st.expander("Show Promotion Analysis", expanded=False):
        st.markdown("""
        ### Promotion
        | Attribute | Description |
        |---|---|
        | NumDealsPurchases | Number of purchases made with a discount |
        | AcceptedCmp1 | 1 if customer accepted the offer in the 1st campaign, 0 otherwise |
        | AcceptedCmp2 | 1 if customer accepted the offer in the 2nd campaign, 0 otherwise |
        | AcceptedCmp3 | 1 if customer accepted the offer in the 3rd campaign, 0 otherwise |
        | AcceptedCmp4 | 1 if customer accepted the offer in the 4th campaign, 0 otherwise |
        | AcceptedCmp5 | 1 if customer accepted the offer in the 5th campaign, 0 otherwise |
        | Response | 1 if customer accepted the offer in the last campaign, 0 otherwise |
        """)
    
        # --- Action Button ---
        if st.button("Show Promotion Bar Chart", key="show_promo_bar_btn"):
            cols_to_plot = ['NumDealsPurchases','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
            valid_cols = [col for col in cols_to_plot if col in st.session_state.data.columns]
            if valid_cols:
                st.session_state.promotion_bar_fig = plot_bar_chart(st.session_state.data[valid_cols],rotation=0)
                st.session_state.show_promotion_bar = True
            else:
                 st.warning(f"Not all required columns found in data: {cols_to_plot}")
    
        # --- Persistent Display ---
        if st.session_state.get('show_promotion_bar', False):
            st.subheader("Distribution of Promotion Variables")
            if st.session_state.promotion_bar_fig:
                st.pyplot(st.session_state.promotion_bar_fig) # Or st.plotly_chart
                if st.button("Hide Promotion Chart", key="hide_promo_bar_btn"):
                    st.session_state.show_promotion_bar = False
                    st.rerun()
            else: st.warning("Generate promotion chart first.")
    
    # =========================================
    # Section: Place Analysis
    # =========================================
    st.markdown("---")
    st.header("Place of Purchase Analysis")
    with st.expander("Show Place Analysis", expanded=False):
        st.markdown("""
        ### Place
        | Attribute | Description |
        |---|---|
        | NumWebPurchases | Number of purchases made through the company’s website |
        | NumCatalogPurchases | Number of purchases made using a catalog |
        | NumStorePurchases | Number of purchases made directly in stores |
        | NumDealsPurchases | Number of purchases made with a discount |
        """)
    
        # --- Action Buttons ---
        col_pl1, col_pl2 = st.columns(2)
        with col_pl1:
            if st.button("Show Place Bar Chart", key="show_place_bar_btn"):
                cols_to_plot = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']
                valid_cols = [col for col in cols_to_plot if col in st.session_state.data.columns]
                if valid_cols:
                     st.session_state.place_bar_fig = plot_bar_chart(st.session_state.data[valid_cols],rotation=0)
                     st.session_state.show_place_bar = True
                else:
                     st.warning(f"Not all required columns found in data: {cols_to_plot}")
        with col_pl2:
             if st.button("Show Place Pie Chart", key="show_place_pie_btn"):
                cols_to_plot = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']
                valid_cols = [col for col in cols_to_plot if col in st.session_state.data.columns]
                if valid_cols:
                     st.session_state.place_pie_fig = plot_pie_chart(st.session_state.data[valid_cols],title='Shares of different Platforms used for purchase',variable_name='Total Number of Pruchases',variable_name_X=-1.4,variable_name_Y=1)
                     st.session_state.show_place_pie = True
                else:
                     st.warning(f"Not all required columns found in data: {cols_to_plot}")
    
        st.markdown("""
        **Catalog Purchase**
        - **Definition**: Buying from a catalog or other printed list. Catalog received as a physical copy or a digital copy.
        - **Customers**: Individuals, businesses.
        - **Items**: Specific by individuals, bulk orders by businesses.
        - **Ordering**: Online, in-store, mail.
        - **Frequency**: Regular purchases.
        - **Promotion**: Advertise products.
    
        [Source](https://oboloo.com/glossary/catalog-purchase/#:~:text=The%20official%20business%20definition%20of,and%20have%20shipped%20to%20them.)
    
        #### Most valuable place of purchase
        (See Pie Chart)
        """)
    
        # --- Persistent Display ---
        if st.session_state.get('show_place_bar', False):
            st.subheader("Distribution of Purchases by Place")
            if st.session_state.place_bar_fig:
                st.pyplot(st.session_state.place_bar_fig) # Or st.plotly_chart
                if st.button("Hide Place Bar Chart", key="hide_place_bar_btn"):
                    st.session_state.show_place_bar = False
                    st.rerun()
            else: st.warning("Generate place bar chart first.")
    
        if st.session_state.get('show_place_pie', False):
            st.subheader("Shares of Purchases by Place")
            if st.session_state.place_pie_fig:
                st.plotly_chart(st.session_state.place_pie_fig) # Or st.pyplot
                if st.button("Hide Place Pie Chart", key="hide_place_pie_btn"):
                    st.session_state.show_place_pie = False
                    st.rerun()
            else: st.warning("Generate place pie chart first.")
    
    
    # ==================================================
    # Section: Feature Engineering & People's Actions
    # ==================================================
    st.markdown("---")
    st.header("Feature Engineering & People's Actions Analysis")
    with st.expander("Show Actions Analysis", expanded=False):
        st.markdown("""
        ### People's Actions
        | Attribute | Description |
        |---|---|
        | Recency | Number of days since customer's last purchase |
        | Complain | 1 if the customer complained in the last 2 years, 0 otherwise |
        | NumWebVisitsMonth | Number of visits to company’s website in the last month |
    
        
        """)
    
        # --- Button to Perform Feature Engineering ---
        if st.button("Perform Feature Engineering", key="do_feature_eng"):
             if st.session_state.data is not None:
                 try:
                     data_copy = st.session_state.data.copy() # Start fresh
    
                     # Ensure correct dtypes before calculations
                     data_copy['Dt_Customer'] = pd.to_datetime(data_copy['Dt_Customer'], errors='coerce')
                     data_copy['Income'] = pd.to_numeric(data_copy['Income'], errors='coerce').fillna(0) # Fill NA income with 0 for ratio calculation safety
                     numeric_cols = ['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds',
                                     'NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases',
                                     'AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response',
                                     'Customer_Tenure_months', 'NumWebVisitsMonth', 'Recency']
                     for col in numeric_cols:
                         if col in data_copy.columns:
                             data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce').fillna(0)
    
                     data_copy.dropna(subset=['Dt_Customer'], inplace=True) # Need Dt_Customer for tenure
    
                     # Calculate Tenure
                     latest_date = data_copy['Dt_Customer'].max() + pd.DateOffset(days=1)
                     data_copy['Customer_Tenure_months'] = ((latest_date - data_copy['Dt_Customer']).dt.days / 30.0).astype(float) # Use 30.0 for float division
    
                     # Calculate Monetary
                     Products_columns = ['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds']
                     data_copy['Monetary'] = data_copy[Products_columns].sum(axis=1)
    
                     # Calculate Avg_Purchase_Per_Month - handle potential division by zero
                     data_copy['Avg_Purchase_Per_Month'] = np.where(
                         data_copy['Customer_Tenure_months'] > 0,
                         round(data_copy['Monetary'] / data_copy['Customer_Tenure_months'], 3),
                         0 # Set to 0 if tenure is 0 or less
                     )
    
                     # Calculate NumAcceptedCmps
                     campaign_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
                     data_copy['NumAcceptedCmps'] = data_copy[campaign_cols].sum(axis=1)
    
                     # Calculate Avg_Purchase_Per_Month_To_Monthly_Income_Ratio
                     # Income already handled (fillna(0))
                     # Check for tenure > 0 and income > 0
                     data_copy['Avg_Purchase_Per_Month_To_Monthly_Income_Ratio'] = np.where(
                         (data_copy['Customer_Tenure_months'] > 0) & (data_copy['Income'] > 0),
                         round(data_copy['Avg_Purchase_Per_Month'] / (data_copy['Income'] / 12.0), 3), # Monthly income estimate
                         0 # Set to 0 if tenure or income is zero or less
                     )
    
                     # Calculate Ratio_of_Deals_Purchases_to_Total_Purchases
                     purchase_cols = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']
                     total_purchases = data_copy[purchase_cols].sum(axis=1)
                     data_copy['Ratio_of_Deals_Purchases_to_Total_Purchases'] = np.where(
                         total_purchases > 0,
                         round(data_copy['NumDealsPurchases'] / total_purchases, 3),
                         0 # Set to 0 if total purchases are 0
                     )
    
                     st.session_state.data_copy_engineered = data_copy
                     st.session_state.feature_engineering_done = True
                     # Clear RFM state if features are re-engineered
                     st.session_state.rfm_calculated = False
                     st.success("Feature engineering complete. You can now view related analyses.")
                 except Exception as e:
                     st.error(f"Error during feature engineering: {e}")
                     st.session_state.feature_engineering_done = False
             else:
                 st.warning("Load data first.")
    
        # --- Action Buttons dependent on Feature Engineering ---
        if st.session_state.get('feature_engineering_done', False):
            st.markdown("---")
            st.write("**Analyses based on Engineered Features:**")
            col_fa1, col_fa2, col_fa3, col_fa4 = st.columns(4)
            engineered_df = st.session_state.data_copy_engineered # Use the stored engineered df
    
            with col_fa1:
                if st.button("Show Engineered Data Sample", key="show_actions_eng_btn"):
                    cols_to_show = ['NumWebPurchases','NumCatalogPurchases','NumDealsPurchases','NumStorePurchases','NumWebVisitsMonth','Recency','NumAcceptedCmps','Ratio_of_Deals_Purchases_to_Total_Purchases','Avg_Purchase_Per_Month','Avg_Purchase_Per_Month_To_Monthly_Income_Ratio']
                    st.session_state.actions_eng_df = engineered_df[cols_to_show].head() # Show head
                    st.session_state.show_actions_eng = True
            with col_fa2:
                if st.button("Show Tenure >= 6m Histograms", key="show_actions_tenure_btn"):
                     cols_to_plot = ['Avg_Purchase_Per_Month','Avg_Purchase_Per_Month_To_Monthly_Income_Ratio','NumAcceptedCmps','Ratio_of_Deals_Purchases_to_Total_Purchases']
                     df_filtered = engineered_df[engineered_df['Customer_Tenure_months']>=6]
                     st.session_state.actions_tenure_hist_fig = plot_histogram(df_filtered[cols_to_plot],title='Analysis of Customer Tenures of at least 6 months')
                     st.session_state.show_actions_tenure_hist = True
            with col_fa3:
                if st.button("Show Recent Customers with 0 Visits", key="show_actions_novisit_btn"):
                    cols_to_show = ['NumWebPurchases','NumCatalogPurchases','NumDealsPurchases','NumStorePurchases','NumWebVisitsMonth','Recency','Monetary','NumAcceptedCmps','Avg_Purchase_Per_Month','Avg_Purchase_Per_Month_To_Monthly_Income_Ratio']
                    df_filtered = engineered_df[((engineered_df['Recency'] > 0) & (engineered_df['NumWebVisitsMonth'] == 0))]
                    st.session_state.actions_no_visits_df = df_filtered[cols_to_show]
                    st.session_state.show_actions_no_visits = True
            with col_fa4:
                # Placeholder for other action buttons if needed
                pass
    
            col_fa5, col_fa6 = st.columns(2)
            with col_fa5:
                if st.button("Show Recency Histogram", key="show_actions_recency_btn"):
                     # Use original data or engineered data? User code used original 'data'
                     st.session_state.actions_recency_hist_fig = plot_histogram(st.session_state.data[['Recency']])
                     st.session_state.show_actions_recency_hist = True
            with col_fa6:
                if st.button("Show Complain/Visits Bar Chart", key="show_actions_cv_bar_btn"):
                    # Use original data or engineered data? User code used original 'data'
                    st.session_state.actions_complain_visits_bar_fig = plot_bar_chart(st.session_state.data[['Complain','NumWebVisitsMonth']],rotation=0)
                    st.session_state.show_actions_complain_visits_bar = True
    
        else:
            st.info("Perform Feature Engineering first to enable subsequent analyses.")
    
    
        # --- Persistent Display Area for Actions ---
        if st.session_state.get('show_actions_eng', False):
            st.subheader("Sample of Engineered Features")
            if st.session_state.actions_eng_df is not None:
                st.dataframe(st.session_state.actions_eng_df)
                if st.button("Hide Engineered Sample", key="hide_actions_eng_btn"):
                    st.session_state.show_actions_eng = False
                    st.rerun()
            else: st.warning("Generate engineered sample first.")
    
        if st.session_state.get('show_actions_tenure_hist', False):
            st.subheader("Analysis for Tenure >= 6 months")
            if st.session_state.actions_tenure_hist_fig:
                st.pyplot(st.session_state.actions_tenure_hist_fig)
                if st.button("Hide Tenure Histograms", key="hide_actions_tenure_btn"):
                    st.session_state.show_actions_tenure_hist = False
                    st.rerun()
            else: st.warning("Generate tenure histograms first.")
    
        if st.session_state.get('show_actions_no_visits', False):
            st.subheader("Recent Customers with 0 Web Visits")
            if st.session_state.actions_no_visits_df is not None:
                st.dataframe(st.session_state.actions_no_visits_df)
                if st.button("Hide No-Visits Data", key="hide_actions_novisit_btn"):
                    st.session_state.show_actions_no_visits = False
                    st.rerun()
            else: st.warning("Generate no-visits data first.")
    
        if st.session_state.get('show_actions_recency_hist', False):
            st.subheader("Recency Distribution")
            if st.session_state.actions_recency_hist_fig:
                st.pyplot(st.session_state.actions_recency_hist_fig)
                if st.button("Hide Recency Hist", key="hide_actions_recency_btn"):
                    st.session_state.show_actions_recency_hist = False
                    st.rerun()
            else: st.warning("Generate recency histogram first.")
    
        if st.session_state.get('show_actions_complain_visits_bar', False):
            st.subheader("Complaints and Web Visits Distribution")
            if st.session_state.actions_complain_visits_bar_fig:
                st.pyplot(st.session_state.actions_complain_visits_bar_fig) # Or plotly
                if st.button("Hide Complain/Visits Chart", key="hide_actions_cv_bar_btn"):
                    st.session_state.show_actions_complain_visits_bar = False
                    st.rerun()
            else: st.warning("Generate complain/visits chart first.")
    
    
    # =========================================
    # Section: Cost and Revenue
    # =========================================
    st.markdown("---")
    st.header("Cost and Revenue Analysis")
    with st.expander("Show Cost/Revenue Analysis", expanded=False):
        st.markdown("""
        ### Cost and Revenue
        | Attribute | Description |
        |---|---|
        | Z_CostContact | Cost |
        | Z_Revenue | Revenue |
        """)
    
        # --- Action Button ---
        if st.button("Show Cost/Revenue Histogram", key="show_costrev_hist_btn"):
            cols_to_plot = ['Z_CostContact', 'Z_Revenue']
            valid_cols = [col for col in cols_to_plot if col in st.session_state.data.columns]
            if valid_cols:
                 st.session_state.cost_rev_hist_fig = plot_histogram(st.session_state.data[valid_cols])
                 st.session_state.show_cost_rev_hist = True
            else:
                st.warning(f"Cost/Revenue columns not found: {cols_to_plot}")
    
        st.markdown("""
        *The 'Z_CostContact', 'Z_Revenue' variables seem to be average values over all the customers and therefore, <span style="color: red;">of little use</span>.*
        """, unsafe_allow_html=True) # Allow HTML for color
    
        # --- Persistent Display ---
        if st.session_state.get('show_cost_rev_hist', False):
            st.subheader("Cost and Revenue Distribution")
            if st.session_state.cost_rev_hist_fig:
                st.pyplot(st.session_state.cost_rev_hist_fig)
                if st.button("Hide Cost/Revenue Hist", key="hide_costrev_hist_btn"):
                    st.session_state.show_cost_rev_hist = False
                    st.rerun()
            else: st.warning("Generate cost/revenue histogram first.")
    
    
    # =========================================
    # Section: RFM Analysis
    # =========================================
    st.markdown("---")
    st.header("RFM (Recency, Frequency, Monetary) Analysis")
    with st.expander("Show RFM Analysis", expanded=False):
        st.markdown("""
        *(Requires Feature Engineering step to be completed first)*
        ### Products (Reference for Monetary Calculation)
        | Attribute | Description |
        |---|---|
        | MntWines | Amount spent on wine in last 2 years |
        | MntFruits | Amount spent on fruits in last 2 years |
        | MntMeatProducts | Amount spent on meat in last 2 years |
        | MntFishProducts | Amount spent on fish in last 2 years |
        | MntSweetProducts | Amount spent on sweets in last 2 years |
        | MntGoldProds | Amount spent on gold in last 2 years |
        """)
    
        # --- Button to Perform RFM Calculation ---
        # Requires engineered data first
        if st.session_state.get('feature_engineering_done', False):
            if st.button("Calculate RFM Scores and Segments", key="calc_rfm_btn"):
                try:
                    # Use the already engineered DataFrame
                    rfm_data = st.session_state.data_copy_engineered.copy()
    
                    # Ensure necessary columns exist from engineering step
                    required_rfm_cols = ['ID', 'Recency', 'Monetary']
                    if not all(col in rfm_data.columns for col in required_rfm_cols):
                         st.error("Required columns (ID, Recency, Monetary) not found in engineered data. Rerun Feature Engineering.")
                    else:
                        # Calculate Frequency (if not already done, though it should be in Monetary section)
                        Place_columns = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases']
                        rfm_data['Frequency'] = rfm_data[Place_columns].sum(axis=1)
    
                        # Store the raw RFM values DataFrame view
                        st.session_state.rfm_df = rfm_data[['ID','Recency','Frequency','Monetary']]
    
                        # --- Assumes rank_group and segment_customer functions are defined ---
                        rfm_data['Recency_Rank'] = rank_group(rfm_data['Recency'])
                        rfm_data['Frequency_Rank'] = rank_group(rfm_data['Frequency'])
                        rfm_data['Monetary_Rank'] = rank_group(rfm_data['Monetary'])
                        rfm_data['Average_Rank'] = (rfm_data['Recency_Rank'].astype(int) + rfm_data['Frequency_Rank'].astype(int) + rfm_data['Monetary_Rank'].astype(int)) / 3
                        rfm_data['Segment'] = rfm_data.apply(segment_customer, axis=1)
                        # --- End of assumed functions ---
    
                        # Store the ranked/segmented DataFrame view
                        selected_columns = ['ID', 'Recency', 'Frequency', 'Monetary', 'Recency_Rank', 'Frequency_Rank', 'Monetary_Rank', 'Segment']
                        st.session_state.rfm_ranks_df = rfm_data[selected_columns]
    
                        # Generate plots and store figures
                        st.session_state.rfm_scatter_fig = plot_customer_segmentation_scatter_plot(data=rfm_data,x='Recency',y='Frequency',hue='Segment',palette='Set1',s=100)
                        st.session_state.rfm_3d_scatter_fig = plot_customer_segmentation_3d_scatter_plot(data=rfm_data,x='Recency',y='Frequency',z='Monetary',hue='Segment',palette='Set1',s=100)
                        # Assuming the interactive plot function returns a Plotly figure
                        st.session_state.rfm_3d_interactive_fig = plot_customer_segmentation_3d_interactive_scatter_plot(rfm_data[['Recency', 'Frequency', 'Monetary', 'Segment']])
    
                        st.session_state.rfm_calculated = True
                        st.success("RFM analysis complete. You can now view RFM results.")
    
                except NameError as ne:
                     st.error(f"RFM Calculation Error: Function not defined? {ne}. Make sure 'rank_group' and 'segment_customer' are defined.")
                except Exception as e:
                     st.error(f"An error occurred during RFM calculation: {e}")
                     st.session_state.rfm_calculated = False
        else:
            st.info("Perform Feature Engineering first to enable RFM Analysis.")
    
    
        # --- Action Buttons to Show RFM Results ---
        if st.session_state.get('rfm_calculated', False):
            st.markdown("---")
            st.write("**RFM Results:**")
            col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
            with col_r1:
                if st.button("Show RFM Values", key="show_rfm_val_btn"): st.session_state.show_rfm_df = True
            with col_r2:
                if st.button("Show RFM Ranks/Segments", key="show_rfm_rank_btn"): st.session_state.show_rfm_ranks = True
            with col_r3:
                if st.button("Show RFM Scatter Plot", key="show_rfm_scatter_btn"): st.session_state.show_rfm_scatter = True
            with col_r4:
                if st.button("Show RFM 3D Scatter", key="show_rfm_3d_btn"): st.session_state.show_rfm_3d_scatter = True
            with col_r5:
                if st.button("Show RFM Interactive 3D", key="show_rfm_int_btn"): st.session_state.show_rfm_3d_interactive = True
        else:
             st.info("Calculate RFM Scores first to view results.")
    
    
        # --- Persistent Display Area for RFM ---
        if st.session_state.get('show_rfm_df', False):
            st.subheader("RFM Values")
            if st.session_state.rfm_df is not None:
                st.dataframe(st.session_state.rfm_df.head()) # Show head
                if st.button("Hide RFM Values", key="hide_rfm_val_btn"):
                    st.session_state.show_rfm_df = False
                    st.rerun()
            else: st.warning("Calculate RFM first.")
    
        if st.session_state.get('show_rfm_ranks', False):
            st.subheader("RFM Ranks and Segments")
            if st.session_state.rfm_ranks_df is not None:
                st.dataframe(st.session_state.rfm_ranks_df.head()) # Show head
                if st.button("Hide RFM Ranks", key="hide_rfm_rank_btn"):
                    st.session_state.show_rfm_ranks = False
                    st.rerun()
            else: st.warning("Calculate RFM first.")
    
        if st.session_state.get('show_rfm_scatter', False):
            st.subheader("RFM Scatter Plot (Recency vs Frequency)")
            if st.session_state.rfm_scatter_fig:
                st.pyplot(st.session_state.rfm_scatter_fig) # Assuming matplotlib
                if st.button("Hide RFM Scatter", key="hide_rfm_scatter_btn"):
                    st.session_state.show_rfm_scatter = False
                    st.rerun()
            else: st.warning("Calculate RFM first.")
    
        if st.session_state.get('show_rfm_3d_scatter', False):
            st.subheader("RFM 3D Scatter Plot")
            if st.session_state.rfm_3d_scatter_fig:
                # 3D plots with matplotlib might not render well directly in st.pyplot.
                # Consider saving as image or using Plotly if possible.
                # For now, assuming st.pyplot might work for basic 3D.
                st.pyplot(st.session_state.rfm_3d_scatter_fig)
                if st.button("Hide RFM 3D Scatter", key="hide_rfm_3d_btn"):
                    st.session_state.show_rfm_3d_scatter = False
                    st.rerun()
            else: st.warning("Calculate RFM first.")
    
        if st.session_state.get('show_rfm_3d_interactive', False):
            st.subheader("RFM Interactive 3D Scatter Plot")
            if st.session_state.rfm_3d_interactive_fig:
                st.plotly_chart(st.session_state.rfm_3d_interactive_fig) # Assuming Plotly
                if st.button("Hide RFM Interactive 3D", key="hide_rfm_int_btn"):
                    st.session_state.show_rfm_3d_interactive = False
                    st.rerun()
            else: st.warning("Calculate RFM first.")
    
    
    # --- End of the main `if st.session_state.data is not None:` block ---

else:
    st.error("Data could not be loaded. Cannot proceed.")

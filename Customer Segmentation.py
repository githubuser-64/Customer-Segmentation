import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.patches as mpatches # For K-Means legend
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster # Needed for hierarchical labels
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

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

def plot_count_histogram(series, title="Histogram of Counts"):
    """Generates a histogram for count data."""
    fig, ax = plt.subplots(figsize=(5, 3))
    counts, bins, patches = ax.hist(series, bins=max(1, series.max() + 1), align='left', rwidth=0.8, alpha=0.7) # Bins for each integer count
    ax.set_xlabel("Count Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.set_xticks(bins[:-1]) # Label ticks at the start of each bin (integer value)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig


def plot_stacked_histogram(data_copy):

    # Calculate total spending for each sector
    primary_expenditure = data_copy['MntFishProducts'] + data_copy['MntMeatProducts'] + data_copy['MntFruits']
    secondary_expenditure = data_copy['MntSweetProducts'] + data_copy['MntWines']
    tertiary_expenditure = data_copy['MntGoldProds']

    # Determine common bin edges based on the full range of all expenditures
    all_expenditures = np.concatenate([primary_expenditure, secondary_expenditure, tertiary_expenditure])
    bins = np.histogram_bin_edges(all_expenditures, bins=15)  # Common bins

    # Create figure object explicitly before plotting on it
    fig, ax = plt.subplots(figsize=(5, 3)) # Use subplots to get fig and ax

    # Plot stacked histogram using the axes object (ax)
    ax.hist(
        [tertiary_expenditure, secondary_expenditure, primary_expenditure],
        bins=bins,
        stacked=True,
        color=['blue', 'yellow', 'orange'],
        label=['Tertiary (gold)', 'Secondary (wine,sweets)', 'Primary (fish,meat,fruit)']
    )

    ax.set_xlabel("Spending Amount") # Use ax.set_xlabel
    ax.set_ylabel("Frequency")     # Use ax.set_ylabel
    ax.set_title("Histogram of Spending Across Sectors") # Use ax.set_title
    ax.legend()                    # Use ax.legend
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Use ax.grid

    plt.tight_layout() # Optional: improve layout
    # plt.show() # Commented out for Streamlit

    # Return the figure object
    return fig



def plot_bar_chart(data, columns_to_include=None, rotation=0, tick_label_fontsize=8, bar_label_fontsize=8): # Added fontsize args
    # If columns_to_include is provided, use only those columns
    if columns_to_include is not None:
        # Ensure columns_to_include is a list
        if not isinstance(columns_to_include, list):
             print('Warning: columns_to_include should be a list. Using all columns.')
             data_to_plot = data
        else:
             # Filter out columns not present in the dataframe to avoid errors
             valid_cols = [col for col in columns_to_include if col in data.columns]
             if len(valid_cols) != len(columns_to_include):
                 missing_cols = [col for col in columns_to_include if col not in data.columns]
                 print(f"Warning: Columns not found and ignored: {missing_cols}")
             if not valid_cols:
                 print("Warning: No valid columns specified in columns_to_include. Using all columns.")
                 data_to_plot = data
             else:
                 data_to_plot = data[valid_cols]
    else:
        data_to_plot = data

    # Create a subplot for all specified columns
    num_columns = len(data_to_plot.columns)
    if num_columns == 0:
        print("No columns to plot.")
        # Return an empty figure or handle appropriately
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center')
        return fig

    num_rows = (num_columns + 2) // 3 # Simplified calculation for rows needed (handles 1, 2, 3+ cols)
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, num_rows * 2), squeeze=False) # Adjusted figsize, ensure axes is 2D
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Create bar charts for each specified column
    try:
        for i, column in enumerate(data_to_plot.columns):
            # Ensure the column exists in the original data passed to the function
            if column not in data.columns:
                 print(f"Skipping column '{column}' as it's not in the provided data.")
                 continue

            counts = data[column].value_counts().sort_index()  # Sort by index

            # Check if counts is empty (e.g., column had only NaNs)
            if counts.empty:
                axes[i].text(0.5, 0.5, f"{column}\n(No data)", ha='center', va='center', fontsize=8)
                axes[i].set_title(f'{column}', fontsize=10) # Still add title
                axes[i].set_xticks([]) # Remove ticks if no data
                axes[i].set_yticks([])
                continue # Skip plotting for this column

            counts.plot(kind='bar', color='skyblue', ax=axes[i])
            axes[i].set_title(f'{column}', fontsize=10) # Slightly larger title
            axes[i].set_xlabel(None) # Remove x label default text
            axes[i].set_ylabel('Frequency', fontsize=9) # Slightly larger y label

            # --- Adjust tick label font sizes ---
            axes[i].tick_params(axis='x', rotation=rotation, labelsize=tick_label_fontsize)
            axes[i].tick_params(axis='y', labelsize=tick_label_fontsize)
            # --- End tick label adjustment ---

            axes[i].grid(axis='y', linestyle='--', alpha=0.7)

            # Add value labels on top of the bars
            for j, value in enumerate(counts):
                # Use axes[i].get_ylim()[1] to position text relative to axis height
                # Add a small offset based on the max height
                max_height = counts.max()
                offset = max_height * 0.02 # Adjust 2% offset as needed
                axes[i].text(j, value + offset, f'{value:.0f}', # Format as integer
                             ha='center', va='bottom',
                             fontsize=bar_label_fontsize) # <-- Set font size here

            # Adjust y-limit to make space for labels
            axes[i].set_ylim(top=counts.max() * 1.1) # Add 10% padding at the top

    except Exception as e: # Catch specific errors if possible, or general Exception
        print(f'An error occurred during plotting: {e}')


    # Hide unused subplots
    for j in range(num_columns, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Add padding for main title if needed
    # fig.suptitle("Bar Charts", fontsize=14) # Optional overall title
    # plt.show() # Commented out for Streamlit
    return fig

# Define function for pie charts
def plot_pie_chart(data,title,variable_name,variable_name_X,variable_name_Y):
    series_data = data.sum()
    total_value = data.sum().sum()

    # Create figure and axes objects explicitly
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot on the axes object
    ax.pie(series_data, labels=series_data.index, autopct='%1.1f%%', startangle=90, labeldistance = 1.03)
    ax.set_title(title) # Use ax.set_title
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add text annotation using the axes object
    ax.text(
        variable_name_X,  # x-coordinate (adjust as needed for positioning)
        variable_name_Y,   # y-coordinate (adjust as needed for positioning)
        f"{variable_name}: {total_value:.2f}",  # Display total value, formatted to 2 decimal places
        fontsize=10,
        fontweight='bold',
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5},  # Optional: Add a box around the text
        transform=ax.transAxes # Use axes coordinates for text positioning relative to the axes
                               # (0,0 is bottom-left, 1,1 is top-right) - adjust X/Y accordingly if needed
                               # If you want data coordinates, remove transform=ax.transAxes
    )

    # Return the figure object
    return fig
    

# Define function for scatter plots for pairs of variables
def plot_scatterplot(data, column_for_color=None, columns_to_exclude=None, heading_of_plot=None):
    """Generates a pairplot and returns the Figure object."""
    # If columns_to_exclude is provided, drop those columns
    if columns_to_exclude is not None:
        # Ensure columns exist before dropping
        cols_exist = [col for col in columns_to_exclude if col in data.columns]
        data_to_plot = data.drop(columns=cols_exist)
    else:
        data_to_plot = data  # Use the entire DataFrame if no columns are excluded

    # Create the pairplot (figure-level function)
    g = sns.pairplot(data_to_plot, diag_kind='kde', markers='o', hue=column_for_color, palette='viridis')

    # Add the super title to the figure object managed by PairGrid
    if heading_of_plot:
        g.fig.suptitle(f'{heading_of_plot}', y=1.02)

    # Return the figure object from the PairGrid
    return g.fig
    

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
        height = 1000,
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
    """Generates a 3D scatter plot and returns the Figure object."""
    # Create figure and 3D axes ONCE
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Get unique segments and their corresponding colors
    segments = data[hue].unique()
    color_map = sns.color_palette(palette, n_colors=len(segments))
    color_dict = {segment: color for segment, color in zip(segments, color_map)}

    # Plot data for each segment on the axes
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

    # Set titles and labels using the axes object
    ax.set_title('Customer Segmentation 3D Scatter Plot')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.legend(title='Segment', loc='upper right')

    # Return the figure object
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
def plot_customer_segmentation_scatter_plot(data,x='Recency',y='Frequency',hue='Segment',palette='Set1',s=100):
    """Generates a 2D scatter plot and returns the Figure object."""
    # Create figure and axes objects explicitly
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot on the axes object, passing ax=ax
    sns.scatterplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=palette,
        s=s,
        ax=ax  # Specify the axes to plot on
    )

    # Set titles and labels using the axes object
    ax.set_title('Customer Segmentation Scatter Plot')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.legend(title='Segment', loc='upper right')

    # Return the figure object
    return fig


def plot_correlation_heatmap(data, threshold_correlation_value):
    """Generates a correlation heatmap and returns the Figure object."""
    # Create figure and axes objects explicitly
    # Increase figsize if needed for many variables
    fig, ax = plt.subplots(figsize=(8, 8)) # Adjusted size

    correlation_matrix = data.corr()

    # Filter the matrix based on the threshold for display purposes
    filtered_matrix = correlation_matrix[(correlation_matrix >= threshold_correlation_value) | (correlation_matrix <= -threshold_correlation_value)]

    # Plot on the axes object, passing ax=ax
    sns.heatmap(filtered_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, cbar_kws={"shrink": .8}, ax=ax) # Use filtered_matrix for display

    # Set title using the axes object
    ax.set_title('Heatmap of Correlation Matrix (Filtered)', fontsize=16)

    # Improve layout to prevent labels overlapping
    plt.tight_layout()

    # Return the figure object
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
    fig, ax = plt.subplots(figsize=(8, 5))
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
    """Performs PCA, generates a scree plot, and returns the Figure object."""
    # Perform PCA
    pca = PCA()
    pca.fit(data)

    # Get explained variance ratios
    explained_variance_ratios = pca.explained_variance_ratio_

    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratios)

    # Create figure and axes objects explicitly
    fig, ax = plt.subplots(figsize=(8, 3)) # Adjusted size


    ### Bar Part (Optional - uncomment if needed) ###
    # # Plot individual explained variance as bars
    # bars = ax.bar(range(1, len(explained_variance_ratios) + 1), explained_variance_ratios, alpha=0.7, align='center', label='Individual Explained Variance')
    # # Annotate each bar
    # for i, bar in enumerate(bars):
    #     height = bar.get_height()
    #     ax.annotate(f'{height * 100:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
    #                  xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    ### Bar Part End ###


    # Plot individual explained variance as a line on the axes object
    ax.plot(range(1, len(explained_variance_ratios) + 1), explained_variance_ratios, marker='o', linestyle='-', label='Individual Explained Variance')

    # Annotate each point on the axes object
    for i, var in enumerate(explained_variance_ratios):
        ax.annotate(f'{var * 100:.2f}%', xy=(i + 1, var),
                     xytext=(0, 5),  # Adjusted offset slightly
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=8)

    # Plot cumulative explained variance on the axes object
    ax.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='red', label='Cumulative Explained Variance')

    # Add labels and title using the axes object
    ax.set_title('PCA Scree Plot', fontsize=10)
    ax.set_xlabel('Principal Component', fontsize=10)
    ax.set_ylabel('Explained Variance Ratio', fontsize=10)

    # Set x-axis ticks and adjust their font size
    x_range = range(1, len(explained_variance_ratios) + 1)
    ax.set_xticks(x_range)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.grid(True)
    ax.legend(loc='best', fontsize=8)

    # Improve layout
    plt.tight_layout()

    # Remove plt.show()
    # plt.show()

    # Return the figure object
    return fig


# --- New Plotting Functions ---
def plot_dendrogram(linkage_matrix, optimal_y_value):
    """Generates a dendrogram plot and returns the Figure object."""
    fig, ax = plt.subplots(figsize=(8, 4)) # Adjusted size
    dendrogram(linkage_matrix, ax=ax)
    ax.set_title('Dendrogram for Hierarchical Clustering')
    ax.set_xlabel('Data Points (or Clusters)')
    ax.set_ylabel('Distance (Ward Linkage)')
    ax.axhline(y=optimal_y_value, color='r', linestyle='--', label=f'Optimal Cut ({optimal_y_value:.2f})')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_tsne(tsne_data):
    """Generates a t-SNE scatter plot and returns the Figure object."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(tsne_data[:, 0], tsne_data[:, 1], s=10, alpha=0.7) # Smaller points, slight transparency
    ax.set_title('t-SNE Visualization of Data')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_hierarchical_on_tsne(tsne_data, labels, n_clusters):
    """Generates a plot of Hierarchical clusters on t-SNE data."""
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis', s=20, alpha=0.8)
    ax.set_title(f'Hierarchical Clustering (k={n_clusters}) on t-SNE')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Create legend (handle potentially non-contiguous labels from fcluster)
    unique_labels = np.unique(labels)
    colors = [scatter.cmap(scatter.norm(label)) for label in unique_labels]
    patches = [mpatches.Patch(color=colors[i], label=f'Cluster {unique_labels[i]}') for i in range(len(unique_labels))]
    ax.legend(handles=patches, title='Cluster Labels', loc='best')

    plt.tight_layout()
    return fig

def plot_kmeans_clusters(tsne_data, labels, n_clusters):
    """Generates a K-Means cluster plot on t-SNE data and returns the Figure object."""
    fig, ax = plt.subplots(figsize=(6, 4)) # Adjusted size
    scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis', s=20, alpha=0.8) # Adjusted size/alpha
    ax.set_title(f'K-means Clustering ({n_clusters} Clusters) on t-SNE')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Create legend
    unique_labels = np.unique(labels)
    # Use matplotlib patches for legend handles
    patches = [mpatches.Patch(color=scatter.cmap(scatter.norm(label)), label=f'Cluster {label}') for label in unique_labels]
    ax.legend(handles=patches, title='Cluster Labels', loc='best',
                  title_fontsize=6, # <-- Font size for legend title
                  fontsize=6)

    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_actual, y_predicted, title="Actual vs. Predicted Counts"):
    """Generates a scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(3, 3))
    max_val = max(y_actual.max(), y_predicted.max()) * 1.1
    min_val = min(y_actual.min(), y_predicted.min()) * 0.9
    ax.scatter(y_actual, y_predicted, alpha=0.5, label='Predictions')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit (y=x)')
    ax.set_xlabel("Actual Counts")
    ax.set_ylabel("Predicted Counts")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    plt.tight_layout()
    return fig
# --- End New Plotting Functions ---


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

# State for Modeling Preparation
if 'data_prepared_for_modeling' not in st.session_state: st.session_state.data_prepared_for_modeling = None
if 'modeling_data_prepared' not in st.session_state: st.session_state.modeling_data_prepared = False

# State for Outlier Detection
if 'data_no_outliers' not in st.session_state: st.session_state.data_no_outliers = None
if 'outliers_removed' not in st.session_state: st.session_state.outliers_removed = False
if 'num_outliers_removed' not in st.session_state: st.session_state.num_outliers_removed = 0

# State for Scaling and PCA
if 'scaled_data' not in st.session_state: st.session_state.scaled_data = None
if 'pca_data' not in st.session_state: st.session_state.pca_data = None
if 'pca_model' not in st.session_state: st.session_state.pca_model = None
if 'pca_done' not in st.session_state: st.session_state.pca_done = False
if 'pca_scree_fig' not in st.session_state: st.session_state.pca_scree_fig = None
if 'show_pca_scree' not in st.session_state: st.session_state.show_pca_scree = False

# State for Hierarchical Clustering
if 'linkage_matrix' not in st.session_state: st.session_state.linkage_matrix = None
if 'hierarchical_done' not in st.session_state: st.session_state.hierarchical_done = False
if 'dendrogram_fig' not in st.session_state: st.session_state.dendrogram_fig = None
if 'show_dendrogram' not in st.session_state: st.session_state.show_dendrogram = False

# State for t-SNE
if 'tsne_data' not in st.session_state: st.session_state.tsne_data = None
if 'tsne_done' not in st.session_state: st.session_state.tsne_done = False
if 'tsne_plot_fig' not in st.session_state: st.session_state.tsne_plot_fig = None
if 'show_tsne_plot' not in st.session_state: st.session_state.show_tsne_plot = False

# State for K-Means
if 'n_clusters' not in st.session_state: st.session_state.n_clusters = 5 # Default k
if 'kmeans_labels' not in st.session_state: st.session_state.kmeans_labels = None
if 'kmeans_done' not in st.session_state: st.session_state.kmeans_done = False
if 'kmeans_plot_fig' not in st.session_state: st.session_state.kmeans_plot_fig = None
if 'show_kmeans_plot' not in st.session_state: st.session_state.show_kmeans_plot = False

# State for Silhouette Scores
if 'hierarchical_silhouette' not in st.session_state: st.session_state.hierarchical_silhouette = None
if 'kmeans_silhouette' not in st.session_state: st.session_state.kmeans_silhouette = None
if 'hierarchical_n_clusters' not in st.session_state: st.session_state.hierarchical_n_clusters = None # Store k for hierarchical

# # State for Zero-Inflated Model
# if 'zi_model_type' not in st.session_state: st.session_state.zi_model_type = 'ZIP' # Default ZI Poisson
# if 'zi_model_fitted' not in st.session_state: st.session_state.zi_model_fitted = None
# if 'zi_predictions' not in st.session_state: st.session_state.zi_predictions = None
# if 'zi_y_test' not in st.session_state: st.session_state.zi_y_test = None # Store actual test values
# if 'zi_mae' not in st.session_state: st.session_state.zi_mae = None
# if 'zi_rmse' not in st.session_state: st.session_state.zi_rmse = None
# if 'zi_baseline_mae' not in st.session_state: st.session_state.zi_baseline_mae = None
# if 'zi_baseline_rmse' not in st.session_state: st.session_state.zi_baseline_rmse = None
# if 'zi_plot_fig' not in st.session_state: st.session_state.zi_plot_fig = None
# if 'zi_model_trained_success' not in st.session_state: st.session_state.zi_model_trained_success = False # Track if training completed

# # State for Target Variable Histogram
# if 'numacceptedcmps_hist_fig' not in st.session_state: st.session_state.numacceptedcmps_hist_fig = None
# if 'show_numacceptedcmps_hist' not in st.session_state: st.session_state.show_numacceptedcmps_hist = False

# # State for Overdispersion Check
# if 'overdispersion_alpha' not in st.session_state: st.session_state.overdispersion_alpha = None
# if 'overdispersion_checked' not in st.session_state: st.session_state.overdispersion_checked = False

# # State for Hierarchical Clusters on t-SNE plot
# if 'hierarchical_tsne_fig' not in st.session_state: st.session_state.hierarchical_tsne_fig = None
# if 'show_hierarchical_tsne_plot' not in st.session_state: st.session_state.show_hierarchical_tsne_plot = False
# if 'hierarchical_labels_for_tsne' not in st.session_state: st.session_state.hierarchical_labels_for_tsne = None # Store labels used for the plot

# State for XGBoost Model
if 'xgb_model_fitted' not in st.session_state: st.session_state.xgb_model_fitted = None
if 'xgb_predictions' not in st.session_state: st.session_state.xgb_predictions = None
if 'xgb_y_test' not in st.session_state: st.session_state.xgb_y_test = None # Store actual test values
if 'xgb_mae' not in st.session_state: st.session_state.xgb_mae = None
if 'xgb_rmse' not in st.session_state: st.session_state.xgb_rmse = None
if 'xgb_baseline_mae' not in st.session_state: st.session_state.xgb_baseline_mae = None
if 'xgb_baseline_rmse' not in st.session_state: st.session_state.xgb_baseline_rmse = None
if 'xgb_plot_fig' not in st.session_state: st.session_state.xgb_plot_fig = None
if 'xgb_model_trained_success' not in st.session_state: st.session_state.xgb_model_trained_success = False # Track if training completed
if 'xgb_feature_importances_fig' not in st.session_state: st.session_state.xgb_feature_importances_fig = None # For feature importance

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
                st.pyplot(st.session_state.product_hist_fig, use_container_width=False) # Use st.pyplot for matplotlib
                if st.button("Hide Product Histogram", key="hide_prod_hist_btn"):
                    st.session_state.show_product_hist = False
                    st.rerun()
            else: st.warning("Generate histogram first.")
    
        if st.session_state.get('show_product_pie', False):
            st.subheader("Product Shares Pie Chart")
            if st.session_state.product_pie_fig:
                st.pyplot(st.session_state.product_pie_fig, use_container_width=False) # Use st.plotly_chart for Plotly
                if st.button("Hide Product Pie Chart", key="hide_prod_pie_btn"):
                    st.session_state.show_product_pie = False
                    st.rerun()
            else: st.warning("Generate pie chart first.")
    
        if st.session_state.get('show_product_stacked_hist', False):
            st.subheader("Product Stacked Histogram")
            if st.session_state.product_stacked_hist_fig:
                st.pyplot(st.session_state.product_stacked_hist_fig, use_container_width=False)
                if st.button("Hide Product Stacked Hist", key="hide_prod_stack_btn"):
                    st.session_state.show_product_stacked_hist = False
                    st.rerun()
            else: st.warning("Generate stacked histogram first.")
    
        if st.session_state.get('show_product_log_hist', False):
            st.subheader("Log-Transformed Product Spending Distribution")
            if st.session_state.product_log_hist_fig:
                st.pyplot(st.session_state.product_log_hist_fig, use_container_width=False)
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
                st.pyplot(st.session_state.age_hist_fig, use_container_width=False)
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
                st.pyplot(st.session_state.circumstances_bar_fig, use_container_width=False) # Or st.plotly_chart if it returns Plotly
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
        | Frequency | Sum of 'NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases' |
        | Monetary | Sum of 'NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases' |
        | NumAcceptedCmps | Sum of 'AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response' |
        | Age_On_Last_Enrollment_Date | Age on the last enrollment date in the dataset for better time context |
        | Customer_Tenure_months | No. of months passed since the date of enrollment |
        | NumWebVisitsMonth | Number of visits to company’s website in the last month |
    
        
        """)
    
        # --- Button to Perform Feature Engineering ---
        if st.button("Perform Feature Engineering", key="do_feature_eng"):
            if st.session_state.data is not None:
                try:
                    data_copy = st.session_state.data.copy() # Start fresh

                    # Ensure correct dtypes before calculations
                    data_copy['Dt_Customer'] = pd.to_datetime(data_copy['Dt_Customer'], errors='coerce')
                    data_copy.dropna(subset=['Dt_Customer', 'Year_Birth'], inplace=True) # Drop rows where calculation isn't possible
                    latest_enrollment_date = data_copy['Dt_Customer'].max()
                    data_copy['Age_On_Last_Enrollment_Date'] = latest_enrollment_date.year - data_copy['Year_Birth']
                    

                    # Ensure correct dtypes before calculations
                    data_copy['Income'] = pd.to_numeric(data_copy['Income'], errors='coerce').fillna(0) # Fill NA income with 0 for ratio calculation safety
                    numeric_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds',
                                    'NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases',
                                    'AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response',
                                    'Customer_Tenure_months','Age_On_Last_Enrollment_Date','NumWebVisitsMonth','Recency']

                    for col in numeric_cols:
                        if col in data_copy.columns:
                            data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce').fillna(0)

                    data_copy.dropna(subset=['Dt_Customer'], inplace=True) # Need Dt_Customer for tenure

                    # Calculate Tenure
                    latest_date = data_copy['Dt_Customer'].max() + pd.DateOffset(days=1)
                    data_copy['Customer_Tenure_months'] = ((latest_date - data_copy['Dt_Customer']).dt.days / 30.0).astype(float) # Use 30.0 for float division

                    # Calculate Frequency
                    Place_columns = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']
                    data_copy['Frequency'] = data_copy[Place_columns].sum(axis=1)

                    # Calculate Monetary
                    Products_columns = ['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds']
                    data_copy['Monetary'] = data_copy[Products_columns].sum(axis=1)

                    # # Calculate Avg_Purchase_Per_Month - handle potential division by zero
                    # data_copy['Avg_Purchase_Per_Month'] = np.where(
                    #     data_copy['Customer_Tenure_months'] > 0,
                    #     round(data_copy['Monetary'] / data_copy['Customer_Tenure_months'], 3),
                    #     0 # Set to 0 if tenure is 0 or less
                    # )

                    # Calculate NumAcceptedCmps
                    campaign_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
                    data_copy['NumAcceptedCmps'] = data_copy[campaign_cols].sum(axis=1)

                    # Calculate Avg_Purchase_Per_Month_To_Monthly_Income_Ratio
                    # Income already handled (fillna(0))
                    
                    # # Check for tenure > 0 and income > 0
                    # data_copy['Avg_Purchase_Per_Month_To_Monthly_Income_Ratio'] = np.where(
                    #     (data_copy['Customer_Tenure_months'] > 0) & (data_copy['Income'] > 0),
                    #     round(data_copy['Avg_Purchase_Per_Month'] / (data_copy['Income'] / 12.0), 3), # Monthly income estimate
                    #     0 # Set to 0 if tenure or income is zero or less
                    # )

                    # Calculate Ratio_of_Deals_Purchases_to_Total_Purchases
                    purchase_cols = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']
                    total_purchases = data_copy[purchase_cols].sum(axis=1)
                    
                    # data_copy['Ratio_of_Deals_Purchases_to_Total_Purchases'] = np.where(
                    #     total_purchases > 0,
                    #     round(data_copy['NumDealsPurchases'] / total_purchases, 3),
                    #     0 # Set to 0 if total purchases are 0
                    # )

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
                    cols_to_show = ['Age_On_Last_Enrollment_Date','Customer_Tenure_months','NumAcceptedCmps','Recency','Frequency','Monetary']
                    st.session_state.actions_eng_df = engineered_df[cols_to_show].sample(5) # Show head
                    st.session_state.show_actions_eng = True
            with col_fa2:
                if st.button("Show Tenure >= 2m Histograms", key="show_actions_tenure_btn"):
                     cols_to_plot = ['NumAcceptedCmps']
                     df_filtered = engineered_df[engineered_df['Customer_Tenure_months']>=2]
                     st.session_state.actions_tenure_hist_fig = plot_histogram(df_filtered[cols_to_plot],title='Analysis of Customer Tenures of at least 2 months')
                     st.session_state.show_actions_tenure_hist = True
            with col_fa3:
                if st.button("Show Recent Customers with 0 Visits", key="show_actions_novisit_btn"):
                    cols_to_show = ['NumWebPurchases','NumCatalogPurchases','NumDealsPurchases','NumStorePurchases','NumWebVisitsMonth','Recency','Monetary','NumAcceptedCmps']
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
                        # # Calculate Frequency (if not already done, though it should be in Monetary section)
                        # Place_columns = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases']
                        # rfm_data['Frequency'] = rfm_data[Place_columns].sum(axis=1)
    
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
    

    # Replace the entire "Outlier Detection & Clustering" section
    # (from `# ==================================================` down to the end of its `with st.expander(...)` block)
    # with this modified version:

    # ==================================================
    # Section: Outlier Detection & Clustering
    # ==================================================
    st.markdown("---")
    st.header("Outlier Detection & Clustering")
    with st.expander("Show Modeling Steps", expanded=True): # Start expanded

        # --- Step 1: Prepare Data for Modeling ---
        st.subheader("Step 1: Prepare Data for Modeling")
        st.markdown("*(Requires Feature Engineering to be completed first)*")

        if st.session_state.get('feature_engineering_done', False):
            if st.button("Prepare Data (Encode & Drop Features)", key="prep_model_data_btn"):
                if st.session_state.data_copy_engineered is not None:
                    try:
                        data_prep = st.session_state.data_copy_engineered.copy()

                        # Apply One-Hot Encoding
                        categorical_cols = ['Education', 'Marital_Status']
                        for col in categorical_cols:
                            if col in data_prep.columns:
                                # Ensure NaNs are explicitly handled or ignored if needed before get_dummies
                                # For simplicity here, assuming NaNs in these cols are not intended or handled upstream
                                dummies = pd.get_dummies(data_prep[col], prefix=col, drop_first=True, dummy_na=False) # dummy_na=False prevents NA column
                                data_prep.drop(col, axis=1, inplace=True)
                                data_prep = pd.concat([data_prep, dummies], axis=1)
                                st.write(f"Applied One-Hot Encoding to '{col}'.")

                        # Drop specified columns
                        columns_to_drop = ['ID', 'Dt_Customer', 'Year_Birth', 'Age_On_Enrollment_Day', 'MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds',
                                    'NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases',
                                           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                                           'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
                                           'Z_CostContact', 'Z_Revenue', 'Response']
                        actual_cols_to_drop = [col for col in columns_to_drop if col in data_prep.columns]
                        data_prep.drop(columns=actual_cols_to_drop, axis=1, inplace=True)
                        st.write(f"Dropped columns: {', '.join(actual_cols_to_drop)}")

                        # # Handle potential infinite values or NaNs introduced (e.g., from ratios) before modeling
                        # data_prep.replace([np.inf, -np.inf], np.nan, inplace=True)
                        # cols_with_na = data_prep.isnull().sum()
                        # cols_to_fill = cols_with_na[cols_with_na > 0].index.tolist()
                        # if cols_to_fill:
                        #      # Simple fill with 0 - review if median/mean is better for specific cols
                        #     data_prep.fillna(0, inplace=True)
                        #     st.warning(f"Filled NAs with 0 in columns: {', '.join(cols_to_fill)} before modeling.")

                        st.session_state.data_prepared_for_modeling = data_prep
                        st.session_state.modeling_data_prepared = True
                        # Reset downstream steps if data is re-prepared
                        st.session_state.outliers_removed = False
                        st.session_state.pca_done = False
                        st.session_state.tsne_done = False
                        st.session_state.hierarchical_done = False
                        st.session_state.kmeans_done = False
                        # Hide plots from previous runs if data is re-prepared
                        st.session_state.show_pca_scree = False
                        st.session_state.show_dendrogram = False
                        st.session_state.show_tsne_plot = False
                        st.session_state.show_kmeans_plot = False

                        st.success("Data prepared for modeling.")
                        st.dataframe(st.session_state.data_prepared_for_modeling.head()) # Show sample

                    except Exception as e:
                        st.error(f"Error during data preparation: {e}")
                        st.session_state.modeling_data_prepared = False
                else:
                    st.warning("Engineered data not found. Run Feature Engineering first.")
        else:
            st.info("Run Feature Engineering first to enable data preparation.")
        st.markdown("---") # Separator after step

        # --- Step 2: Outlier Detection ---
        st.subheader("Step 2: Detect and Remove Outliers")
        if st.session_state.get('modeling_data_prepared', False):
            contamination_level = st.slider("Select Contamination Level (Outlier %)", 0.001, 0.1, 0.01, 0.001, format="%.3f", key="iso_contamination")

            if st.button("Run Isolation Forest", key="run_iso_forest_btn"):
                if st.session_state.data_prepared_for_modeling is not None:
                    try:
                        data_to_scan = st.session_state.data_prepared_for_modeling

                        iso_forest = IsolationForest(n_estimators=100,
                                                     contamination=contamination_level,
                                                     random_state=42,
                                                     n_jobs=-1)
                        st.write(f"Fitting Isolation Forest (contamination={contamination_level:.3f})...")
                        iso_forest.fit(data_to_scan)
                        st.write("Predicting outliers...")
                        outlier_predictions = iso_forest.predict(data_to_scan)

                        is_inlier = outlier_predictions == 1
                        n_outliers_found = (outlier_predictions == -1).sum()
                        st.session_state.num_outliers_removed = n_outliers_found

                        st.session_state.data_no_outliers = data_to_scan[is_inlier].copy()
                        st.session_state.outliers_removed = True
                        # Reset downstream steps
                        st.session_state.pca_done = False
                        st.session_state.tsne_done = False
                        st.session_state.hierarchical_done = False
                        st.session_state.kmeans_done = False
                        # Hide plots from previous runs
                        st.session_state.show_pca_scree = False
                        st.session_state.show_dendrogram = False
                        st.session_state.show_tsne_plot = False
                        st.session_state.show_kmeans_plot = False

                        st.success(f"Outlier detection complete. Removed {n_outliers_found} potential outliers ({n_outliers_found / len(data_to_scan):.2%}).")
                        st.write("Shape after outlier removal:", st.session_state.data_no_outliers.shape)
                        st.dataframe(st.session_state.data_no_outliers.head())

                    except Exception as e:
                        st.error(f"Error during Isolation Forest: {e}")
                        st.session_state.outliers_removed = False
                else:
                    st.warning("Prepared data not found. Run Step 1 first.")
        else:
            st.info("Prepare data (Step 1) first.")
        st.markdown("---") # Separator after step

        # --- Step 3: Scaling and PCA ---
        st.subheader("Step 3: Scale Data and Run PCA")
        if st.session_state.get('outliers_removed', False):
            if st.button("Run Scaling and PCA", key="run_pca_btn"):
                if st.session_state.data_no_outliers is not None:
                    try:
                        data_to_scale = st.session_state.data_no_outliers.copy()
                        numerical_cols = data_to_scale.select_dtypes(include=np.number).columns.tolist()
                        st.write(f"Scaling numerical columns...") # : {', '.join(numerical_cols)}

                        scaler = StandardScaler()
                        scaled_data_array = scaler.fit_transform(data_to_scale[numerical_cols])
                        st.session_state.scaled_data = pd.DataFrame(scaled_data_array, columns=numerical_cols, index=data_to_scale.index)

                        st.write("Running PCA on scaled data...")
                        pca = PCA(random_state=42)
                        pca.fit(st.session_state.scaled_data)
                        st.session_state.pca_model = pca
                        st.session_state.pca_data = pca.transform(st.session_state.scaled_data)

                        st.session_state.pca_scree_fig = plot_pca_scree_plot(st.session_state.scaled_data)
                        st.session_state.pca_done = True
                        st.session_state.show_pca_scree = True # Set flag to show immediately
                         # Reset downstream steps
                        st.session_state.tsne_done = False
                        st.session_state.hierarchical_done = False
                        st.session_state.kmeans_done = False
                        # Hide plots from previous downstream runs
                        st.session_state.show_dendrogram = False
                        st.session_state.show_tsne_plot = False
                        st.session_state.show_kmeans_plot = False

                        st.success("Scaling and PCA complete.")
                        # Optionally show PCA results head: st.dataframe(pd.DataFrame(st.session_state.pca_data).head())

                    except Exception as e:
                        st.error(f"Error during Scaling/PCA: {e}")
                        st.session_state.pca_done = False
                        st.session_state.show_pca_scree = False # Ensure flag is false on error
                else:
                    st.warning("Outlier-free data not found. Run Step 2 first.")

            # Display Area for PCA Scree Plot (within Step 3)
            if st.session_state.get('show_pca_scree', False):
                st.markdown("#### PCA Scree Plot")
                if st.session_state.pca_scree_fig:
                    st.pyplot(st.session_state.pca_scree_fig, use_container_width=False)
                    if st.button("Hide PCA Scree Plot", key="hide_pca_btn_inline"):
                        st.session_state.show_pca_scree = False
                        st.rerun()
                else:
                    # This case means the button was clicked, flag is true, but fig is None (error happened)
                    st.warning("Could not generate PCA Scree Plot.")
                    st.session_state.show_pca_scree = False # Reset flag

        else:
            st.info("Remove outliers (Step 2) first.")
        st.markdown("---") # Separator after step

        # --- Step 4: Hierarchical Clustering (Dendrogram & Cluster Selection) ---
        st.subheader("Step 4: Hierarchical Clustering (Dendrogram & Cluster Selection)")
        if st.session_state.get('pca_done', False):
             if st.session_state.pca_data is not None:
                 max_pca_components = st.session_state.pca_data.shape[1]
                 default_hier_comps = min(10, max_pca_components)
                 n_components_for_clustering = st.slider("Number of PCA Components for Hierarchical Clustering", 2, max_pca_components, default_hier_comps, 1, key="pca_comps_hierarchical_slider_v2") # Changed key

                 # --- Calculate Linkage ---
                 recalculate_linkage = st.button("Recalculate Linkage Matrix", key="recalc_linkage_btn_v3") # Changed key
                 if recalculate_linkage or 'linkage_matrix' not in st.session_state or st.session_state.linkage_matrix is None:
                     # ... (keep the linkage calculation logic exactly as it was in the previous version) ...
                     try:
                         st.write(f"Performing Ward linkage on first {n_components_for_clustering} PCA components...")
                         pca_data_subset_link = st.session_state.pca_data[:, :n_components_for_clustering]
                         st.session_state.linkage_matrix = linkage(pca_data_subset_link, method='ward')
                         distances = st.session_state.linkage_matrix[:, 2]
                         diffs = np.diff(distances)
                         idx_largest_diff = np.argmax(diffs) if len(diffs) > 0 else 0
                         st.session_state.optimal_y_value = distances[idx_largest_diff] if len(distances) > idx_largest_diff else (distances[-1] if len(distances) > 0 else 0)
                         st.write("Linkage matrix calculated.")
                         st.session_state.hierarchical_done = False
                         st.session_state.show_dendrogram = False
                         st.session_state.hierarchical_silhouette = None
                         st.session_state.hierarchical_n_clusters = None
                         st.session_state.show_hierarchical_tsne_plot = False
                         st.session_state.hierarchical_labels_for_tsne = None
                     except Exception as e:
                         st.error(f"Error calculating linkage matrix: {e}")
                         st.session_state.linkage_matrix = None
                         st.session_state.optimal_y_value = None

                 # --- Display Dendrogram & Select k ---
                 if st.session_state.get('linkage_matrix') is not None:
                     st.markdown("#### Dendrogram (with suggested distance cut)")
                     optimal_y = st.session_state.get('optimal_y_value', 0)
                     fig_dendro = plot_dendrogram(st.session_state.linkage_matrix, optimal_y)
                     st.pyplot(fig_dendro, use_container_width=False)
                     st.caption(f"Red line shows cut based on largest distance jump (Distance ≈ {optimal_y:.2f}). Use this or visual inspection to choose k below.")
                     st.session_state.show_dendrogram = True

                     default_k_hier = st.session_state.get('hierarchical_n_clusters', 3)
                     st.session_state.hierarchical_n_clusters = st.number_input(
                         "Select Desired Number of Clusters (k) for Hierarchical",
                         min_value=2, max_value=20, value=default_k_hier, step=1,
                         key="hierarchical_k_selector_input_v2" # Changed key
                     )

                     # --- Assign Clusters & Calculate Score ---
                     if st.button(f"Assign Clusters & Calculate Score (k={st.session_state.hierarchical_n_clusters})", key="run_hierarchical_fcluster_exec_btn_v2"): # Changed key
                         st.session_state.hierarchical_done = False
                         st.session_state.show_hierarchical_tsne_plot = False # Reset plot display status
                         st.session_state.hierarchical_labels_for_tsne = None # Reset labels
                         try:
                             current_k_hierarchical = st.session_state.hierarchical_n_clusters
                             st.write(f"Assigning {current_k_hierarchical} clusters using 'maxclust' criterion...")
                             cluster_labels_hierarchical = fcluster(st.session_state.linkage_matrix, t=current_k_hierarchical, criterion='maxclust')
                             st.session_state.hierarchical_labels_for_tsne = cluster_labels_hierarchical # Store labels

                             # Calculate Silhouette Score using the same PCA subset used for linkage
                             pca_data_subset_score = st.session_state.pca_data[:, :n_components_for_clustering]
                             if current_k_hierarchical > 1 and len(pca_data_subset_score) >= current_k_hierarchical :
                                 try:
                                    score = silhouette_score(pca_data_subset_score, cluster_labels_hierarchical, metric='euclidean')
                                    st.session_state.hierarchical_silhouette = score
                                    st.session_state.hierarchical_done = True # Mark as done only after successful scoring attempt
                                    st.success(f"Clusters assigned & Silhouette Score calculated for k={current_k_hierarchical}.")
                                 except ValueError as sil_err:
                                     st.warning(f"Could not calculate silhouette score: {sil_err}")
                                     st.session_state.hierarchical_silhouette = None
                                     st.session_state.hierarchical_done = False # Mark as not done if score fails
                             else:
                                 st.warning(f"Silhouette score requires k > 1 and Samples >= k.")
                                 st.session_state.hierarchical_silhouette = None
                                 st.session_state.hierarchical_done = False # Mark as not done if score not calculated

                             # DO NOT generate t-SNE plot here automatically anymore

                         except Exception as e:
                             st.error(f"Error assigning clusters or calculating score: {e}")
                             st.session_state.hierarchical_done = False
                             st.session_state.hierarchical_silhouette = None
                             st.session_state.hierarchical_labels_for_tsne = None

                     # --- Display Silhouette Score ---
                     if st.session_state.get('hierarchical_done'): # Only show if assignment/scoring succeeded for the current k
                         k_hier_for_score = st.session_state.hierarchical_n_clusters # Use k selected when score was calc'd
                         if st.session_state.get('hierarchical_silhouette') is not None:
                             st.metric(label=f"Silhouette Score (Hierarchical, k={k_hier_for_score})",
                                       value=f"{st.session_state.hierarchical_silhouette:.3f}")

                     st.markdown("---") # Separator

                     # --- Button to Show Hierarchical Clusters on t-SNE ---
                     st.markdown("#### Visualize Hierarchical Clusters on t-SNE")
                     # Enable button only if clusters are assigned AND t-SNE is done
                     can_show_hier_tsne = st.session_state.get('hierarchical_done', False) and \
                                          st.session_state.get('tsne_done', False) and \
                                          st.session_state.get('tsne_data') is not None and \
                                          st.session_state.get('hierarchical_labels_for_tsne') is not None

                     if not st.session_state.get('tsne_done', False):
                         st.info("Run t-SNE (Step 5) first to enable this visualization.")
                     elif not st.session_state.get('hierarchical_done', False):
                         st.info("Assign Hierarchical Clusters (using the button above) first to enable this visualization.")

                     if st.button("Show Plot", key="show_hier_tsne_plot_btn", disabled=not can_show_hier_tsne):
                         try:
                             st.write("Generating t-SNE plot colored by hierarchical clusters...")
                             current_k_hier = st.session_state.hierarchical_n_clusters # k used for labels
                             st.session_state.hierarchical_tsne_fig = plot_hierarchical_on_tsne(
                                 st.session_state.tsne_data,
                                 st.session_state.hierarchical_labels_for_tsne,
                                 current_k_hier
                             )
                             st.session_state.show_hierarchical_tsne_plot = True # Show plot now
                         except Exception as e:
                             st.error(f"Failed to generate hierarchical t-SNE plot: {e}")
                             st.session_state.show_hierarchical_tsne_plot = False

                     # --- Display Hierarchical Clusters on t-SNE Plot ---
                     if st.session_state.get('show_hierarchical_tsne_plot', False):
                          if st.session_state.hierarchical_tsne_fig:
                              st.pyplot(st.session_state.hierarchical_tsne_fig, use_container_width=False)
                              if st.button("Hide Hierarchical t-SNE Plot", key="hide_hier_tsne_btn_v2"): # Changed key
                                   st.session_state.show_hierarchical_tsne_plot = False
                                   st.rerun()
                          else:
                              st.warning("Could not generate hierarchical t-SNE plot.")
                              st.session_state.show_hierarchical_tsne_plot = False

                 else: # Linkage matrix not calculated
                     st.info("Calculate Linkage Matrix first using the button above.")

             else: # PCA data not available
                 st.warning("PCA results (Step 3) not found.")
        else: # PCA not done
             st.info("Run PCA (Step 3) first.")
        st.markdown("---") # Separator after step


        # --- Step 5: t-SNE Visualization ---
        st.subheader("Step 5: t-SNE Visualization")
        if st.session_state.get('pca_done', False):
             if st.session_state.pca_data is not None:
                 max_pca_components_tsne = st.session_state.pca_data.shape[1]
                 # Default to more components for t-SNE usually
                 default_tsne_comps = min(30, max_pca_components_tsne)
                 n_components_for_tsne = st.slider("Number of PCA Components for t-SNE", 2, max_pca_components_tsne, default_tsne_comps, 1, key="pca_comps_tsne")
                 pca_data_subset_tsne = st.session_state.pca_data[:, :n_components_for_tsne]

                 perplexity_value = st.slider("t-SNE Perplexity.\nHigher perplexity values emphasize broader structures, while lower values prioritize local relationships. Higher perplexity, higher computational time and memory usage.\nVisit: https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html", 5, 50, 30, 1, key="tsne_perplexity")
                 n_iter_value = st.select_slider("t-SNE Iterations", options=[250, 500, 1000, 2000], value=1000, key="tsne_iter")

                 if st.button("Run t-SNE", key="run_tsne_btn"):
                     try:
                         st.write(f"Running t-SNE on first {n_components_for_tsne} PCA components (Perplexity={perplexity_value}, Iterations={n_iter_value})... This may take a moment.")
                         tsne = TSNE(n_components=2,
                                     perplexity=perplexity_value,
                                     n_iter=n_iter_value,
                                     random_state=42,
                                     n_jobs=-1)
                         st.session_state.tsne_data = tsne.fit_transform(pca_data_subset_tsne)
                         st.session_state.tsne_plot_fig = plot_tsne(st.session_state.tsne_data)
                         st.session_state.tsne_done = True
                         st.session_state.show_tsne_plot = True # Show immediately
                          # Reset K-Means if t-SNE is re-run
                         st.session_state.kmeans_done = False
                         st.session_state.show_kmeans_plot = False
                         st.success("t-SNE calculation complete.")

                     except Exception as e:
                         st.error(f"Error during t-SNE: {e}")
                         st.session_state.tsne_done = False
                         st.session_state.show_tsne_plot = False # Ensure flag is false on error

                 # Display Area for t-SNE Plot (within Step 5)
                 if st.session_state.get('show_tsne_plot', False):
                     st.markdown("#### t-SNE Plot")
                     if st.session_state.tsne_plot_fig:
                         st.pyplot(st.session_state.tsne_plot_fig, use_container_width=False)
                         if st.button("Hide t-SNE Plot", key="hide_tsne_btn_inline"):
                             st.session_state.show_tsne_plot = False
                             st.rerun()
                     else:
                         st.warning("Could not generate t-SNE plot.")
                         st.session_state.show_tsne_plot = False # Reset flag
             else:
                 st.warning("PCA results not found.")
        else:
            st.info("Run PCA (Step 3) first.")
        st.markdown("---") # Separator after step


                # --- Step 6: K-Means Clustering (on t-SNE results) ---
        st.subheader("Step 6: K-Means Clustering (on t-SNE results)")
        if st.session_state.get('tsne_done', False):
            if st.session_state.tsne_data is not None:
                 st.session_state.n_clusters = st.number_input("Select Number of Clusters (k)", min_value=2, max_value=20, value=st.session_state.get('n_clusters', 5), step=1, key="kmeans_k_selector")

                 if st.button(f"Run K-Means (k={st.session_state.n_clusters})", key="run_kmeans_btn"):
                     try:
                         current_k = st.session_state.n_clusters
                         st.write(f"Running K-Means with k={current_k} on t-SNE results...")
                         kmeans = KMeans(n_clusters=current_k,
                                         random_state=42,
                                         n_init='auto')
                         labels = kmeans.fit_predict(st.session_state.tsne_data)
                         st.session_state.kmeans_labels = labels

                         # --- Calculate Silhouette Score ---
                         if current_k > 1 and len(st.session_state.tsne_data) > current_k: # Check if calculation is possible
                             try:
                                score = silhouette_score(st.session_state.tsne_data, labels, metric='euclidean')
                                st.session_state.kmeans_silhouette = score
                                st.write(f"Calculated Silhouette Score.")
                             except ValueError as sil_err:
                                 st.warning(f"Could not calculate silhouette score: {sil_err}")
                                 st.session_state.kmeans_silhouette = None
                         else:
                             st.warning(f"Silhouette score requires at least 2 clusters and more samples than clusters.")
                             st.session_state.kmeans_silhouette = None
                         # --- End Silhouette Calculation ---

                         st.session_state.kmeans_plot_fig = plot_kmeans_clusters(st.session_state.tsne_data, labels, current_k)
                         st.session_state.kmeans_done = True
                         st.session_state.show_kmeans_plot = True
                         st.success("K-Means clustering complete.")

                     except Exception as e:
                         st.error(f"Error during K-Means: {e}")
                         st.session_state.kmeans_done = False
                         st.session_state.show_kmeans_plot = False
                         st.session_state.kmeans_silhouette = None # Clear score on error

                 # Display Area for K-Means Plot and Score (within Step 6)
                 if st.session_state.get('show_kmeans_plot', False):
                     k_used_for_plot = len(np.unique(st.session_state.kmeans_labels)) if st.session_state.kmeans_labels is not None else st.session_state.n_clusters
                     st.markdown(f"#### K-Means Clustering Plot (k={k_used_for_plot})")
                     if st.session_state.kmeans_plot_fig:
                         st.pyplot(st.session_state.kmeans_plot_fig, use_container_width=False)
                         # Display silhouette score if available
                         if st.session_state.get('kmeans_silhouette') is not None:
                              st.metric(label=f"Silhouette Score (k={k_used_for_plot})",
                                        value=f"{st.session_state.kmeans_silhouette:.3f}")

                         if st.button("Hide K-Means Plot", key="hide_kmeans_btn_inline"):
                             st.session_state.show_kmeans_plot = False
                             st.session_state.kmeans_silhouette = None # Hide score when hiding plot
                             st.rerun()
                     else:
                         st.warning("Could not generate K-Means plot.")
                         st.session_state.show_kmeans_plot = False
                         st.session_state.kmeans_silhouette = None

            else:
                 st.warning("t-SNE results not found.")
        else:
             st.info("Run t-SNE (Step 5) first.")
        # No final separator needed here as it's the end of the expander
    


    
    # ==================================================
    # Section: Supervised Learning - Predicting Campaign Acceptance (XGBoost)
    # ==================================================
    st.markdown("---")
    st.header("Supervised Learning: Predict Campaign Acceptance (XGBoost)")
    with st.expander("Show XGBoost Regression Steps", expanded=True):

        st.markdown("""
        Predict `NumAcceptedCmps` (total accepted campaigns) using XGBoost Regression.
        *(Uses the data prepared for modeling, **before** outlier removal)*
        """)

        # Define campaign flags list here - BEFORE any steps use it
        original_campaign_flags = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
        target_col = 'NumAcceptedCmps' # Define target col name

        # --- Step 1: Verify Data and Target ---
        st.subheader("Step 1: Verify Data and Target Variable")

        can_proceed_xgb = False
        model_data_xgb = None

        # We specifically want the data BEFORE outlier removal for this model
        if st.session_state.get('data_prepared_for_modeling') is None:
            st.warning("Please run 'Prepare Data for Modeling' (Step 1 in the Clustering section) first.")
        else:
            model_data_xgb = st.session_state.data_prepared_for_modeling.copy()
            st.info(f"Using 'Prepared Data (with potential outliers)'. Shape: {model_data_xgb.shape}")

            # --- Check/Calculate Target Variable ---
            if target_col not in model_data_xgb.columns:
                st.warning(f"'{target_col}' not found, attempting recalculation...")
                if all(c in st.session_state.data.columns for c in original_campaign_flags):
                    original_data_indices = model_data_xgb.index
                    if st.session_state.data is not None and all(idx in st.session_state.data.index for idx in original_data_indices):
                        try:
                            model_data_xgb[target_col] = st.session_state.data.loc[original_data_indices, original_campaign_flags].sum(axis=1)
                            st.info(f"Successfully recalculated '{target_col}'.")
                        except Exception as e_calc:
                             st.error(f"Error during target recalculation: {e_calc}")
                             model_data_xgb = None # Prevent proceeding on error
                    else:
                        st.error(f"Cannot recalculate target: Original data missing or indices mismatch.")
                        model_data_xgb = None
                else:
                    st.error(f"Cannot recalculate target: Required campaign flags missing in original data.")
                    model_data_xgb = None
            else:
                st.success(f"Target column '{target_col}' found.")

            # --- Proceed only if model_data_xgb and target column exist ---
            if model_data_xgb is not None and target_col in model_data_xgb.columns:
                can_proceed_xgb = True
                y_xgb = model_data_xgb[target_col] # Define y_xgb here

                # --- Display Target Histogram ---
                st.markdown("#### Target Variable Distribution (`NumAcceptedCmps`)")
                if st.checkbox("Show/Hide Histogram", value=st.session_state.get('show_numacceptedcmps_hist_xgb', False), key="toggle_nac_hist_xgb"):
                    st.session_state.show_numacceptedcmps_hist_xgb = True
                    try:
                        if 'plot_count_histogram' in globals() and y_xgb is not None:
                            # st.session_state.numacceptedcmps_hist_fig = plot_histogram(model_data_xgb[[target_col]], title=f"Histogram of {target_col}")
                            st.session_state.numacceptedcmps_hist_fig = plot_bar_chart(model_data_xgb[[target_col]])
                            st.pyplot(st.session_state.numacceptedcmps_hist_fig, use_container_width=False)
                        else: st.warning("Plotting function or data unavailable.")
                    except Exception as e_hist:
                        st.warning(f"Could not generate histogram: {e_hist}")
                        st.session_state.show_numacceptedcmps_hist_xgb = False
                else:
                    st.session_state.show_numacceptedcmps_hist_xgb = False
            else:
                st.error("Cannot proceed without valid source data and target column.")
                can_proceed_xgb = False

        st.markdown("---") # Separator

        # --- Step 2: Train and Evaluate XGBoost ---
        st.subheader("Step 2: Train & Evaluate XGBoost Model")
        if can_proceed_xgb:
            test_size_xgb = st.slider("Select Test Set Size:", 0.1, 0.5, 0.20, 0.05, key="xgb_test_size_slider")
            n_estimators_xgb = st.slider("Number of Estimators (Trees):", 50, 500, 100, 50, key="xgb_n_estimators")
            max_depth_xgb = st.slider("Max Tree Depth:", 3, 10, 5, 1, key="xgb_max_depth")
            learning_rate_xgb = st.select_slider("Learning Rate:", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1, key="xgb_learning_rate")

            if st.button("Train XGBoost Model and Evaluate", key="train_xgb_model_exec_btn"):
                st.session_state.xgb_model_trained_success = False # Reset success flag
                try:
                    # Re-fetch data and target INSIDE button press logic for safety
                    model_data_train_xgb = st.session_state.data_prepared_for_modeling.copy()

                    # Ensure target exists again
                    if target_col not in model_data_train_xgb.columns:
                        original_data_indices_train = model_data_train_xgb.index
                        if all(c in st.session_state.data.columns for c in original_campaign_flags):
                             if st.session_state.data is not None and all(idx in st.session_state.data.index for idx in original_data_indices_train):
                                model_data_train_xgb[target_col] = st.session_state.data.loc[original_data_indices_train, original_campaign_flags].sum(axis=1)
                             else: raise ValueError("Original data missing or index mismatch for target recalc.")
                        else: raise ValueError(f"Target column '{target_col}' missing and cannot be recalculated.")

                    y_train_eval_xgb = model_data_train_xgb[target_col]
                    # Select only numeric features for XGBoost, exclude target
                    X_train_eval_xgb = model_data_train_xgb.select_dtypes(include=np.number).drop(columns=[target_col], errors='ignore')

                    # Drop original campaign flags if they accidentally remain as numeric features
                    cols_to_drop_train = [c for c in original_campaign_flags if c in X_train_eval_xgb.columns]
                    if cols_to_drop_train:
                        st.write(f"Removing original campaign flags from features: {', '.join(cols_to_drop_train)}")
                        X_train_eval_xgb = X_train_eval_xgb.drop(columns=cols_to_drop_train)

                    # No need to add constant for XGBoost

                    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                        X_train_eval_xgb, y_train_eval_xgb, test_size=test_size_xgb, random_state=42
                    )
                    st.session_state.xgb_y_test = y_test_s # Store actual test values

                    st.write(f"Training Shape: {X_train_s.shape}, Test Shape: {X_test_s.shape}")

                    # --- Instantiate and Fit XGBoost Model ---
                    st.write(f"Fitting XGBoost Regressor...")
                    xgb_model = xgb.XGBRegressor(
                        objective='count:poisson', # Suitable objective for count data
                        n_estimators=n_estimators_xgb,
                        learning_rate=learning_rate_xgb,
                        max_depth=max_depth_xgb,
                        random_state=42,
                        n_jobs=-1,
                        early_stopping_rounds=10, # Stop if validation metric doesn't improve
                        eval_metric='rmse'       # Metric to monitor for early stopping
                    )

                    eval_set = [(X_test_s, y_test_s)] # Use test set for early stopping validation

                    xgb_model.fit(X_train_s, y_train_s, eval_set=eval_set, verbose=False) # verbose=False prevents printing iterations

                    st.session_state.xgb_model_fitted = xgb_model # Store the fitted model object
                    st.success("Model fitting complete.")

                    # --- Predict and Evaluate ---
                    st.write("Predicting on test set...")
                    predictions_float = st.session_state.xgb_model_fitted.predict(X_test_s)
                    # Round predictions for count comparison metrics
                    st.session_state.xgb_predictions = np.round(predictions_float).astype(int)
                    # Ensure predictions are not negative (common in regression for counts)
                    st.session_state.xgb_predictions[st.session_state.xgb_predictions < 0] = 0


                    st.session_state.xgb_mae = mean_absolute_error(y_test_s, st.session_state.xgb_predictions)
                    mse = mean_squared_error(y_test_s, st.session_state.xgb_predictions)
                    st.session_state.xgb_rmse = np.sqrt(mse)

                    # Calculate baseline AFTER y_train_s is defined
                    baseline_prediction = np.full_like(y_test_s, fill_value=y_train_s.mean())
                    st.session_state.xgb_baseline_mae = mean_absolute_error(y_test_s, baseline_prediction)
                    baseline_mse = mean_squared_error(y_test_s, baseline_prediction)
                    st.session_state.xgb_baseline_rmse = np.sqrt(baseline_mse)

                    # Generate plot only if predictions and actuals are valid
                    if st.session_state.xgb_predictions is not None and y_test_s is not None:
                        st.session_state.xgb_plot_fig = plot_actual_vs_predicted(y_test_s, st.session_state.xgb_predictions, title="XGBoost - Actual vs. Predicted")
                    else:
                        st.session_state.xgb_plot_fig = None

                    # Generate Feature Importance Plot
                    try:
                        importances = st.session_state.xgb_model_fitted.feature_importances_
                        feature_names = X_train_s.columns
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

                        fig_imp, ax_imp = plt.subplots(figsize=(5, max(2, len(feature_names) * 0.1))) # Adjust height
                        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), ax=ax_imp) # Show top 20
                        ax_imp.set_title('XGBoost Feature Importance (Top 20)')
                        plt.tight_layout()
                        st.session_state.xgb_feature_importances_fig = fig_imp
                    except Exception as imp_e:
                        st.warning(f"Could not generate feature importance plot: {imp_e}")
                        st.session_state.xgb_feature_importances_fig = None


                    st.session_state.xgb_model_trained_success = True # Set success flag HERE
                    st.info("Evaluation complete. Results displayed below.")

                except ValueError as ve:
                     st.error(f"Data Error during split/train: {ve}")
                     st.session_state.xgb_model_trained_success = False
                except Exception as e:
                     st.error(f"An error occurred during training/evaluation: {e}")
                     st.session_state.xgb_model_trained_success = False

        else: # can_proceed_xgb is False
            st.info("Complete Step 1 above (verify data and target) to enable training.")

        st.markdown("---") # Separator

        # --- Step 3: Display Results ---
        st.subheader("Step 3: Results")
        if st.session_state.get('xgb_model_trained_success', False):
            st.markdown("#### Model Performance Metrics (on Test Set)")
            # ... (keep metric display logic as before) ...
            col_m1_xgb, col_m2_xgb = st.columns(2)
            with col_m1_xgb:
                if st.session_state.get('xgb_mae') is not None:
                    st.metric("XGBoost MAE", f"{st.session_state.xgb_mae:.3f}")
                if st.session_state.get('xgb_baseline_mae') is not None:
                    st.metric("Baseline (Mean) MAE", f"{st.session_state.xgb_baseline_mae:.3f}")
            with col_m2_xgb:
                if st.session_state.get('xgb_rmse') is not None:
                     st.metric("XGBoost RMSE", f"{st.session_state.xgb_rmse:.3f}")
                if st.session_state.get('xgb_baseline_rmse') is not None:
                    st.metric("Baseline (Mean) RMSE", f"{st.session_state.xgb_baseline_rmse:.3f}")

            if all(k in st.session_state and st.session_state[k] is not None for k in ['xgb_baseline_mae', 'xgb_mae', 'xgb_baseline_rmse', 'xgb_rmse']):
                 delta_mae_xgb = st.session_state.xgb_baseline_mae - st.session_state.xgb_mae
                 delta_rmse_xgb = st.session_state.xgb_baseline_rmse - st.session_state.xgb_rmse
                 st.write(f"**Improvement over Baseline:** MAE reduced by {delta_mae_xgb:.3f}, RMSE reduced by {delta_rmse_xgb:.3f}")
                 st.caption("Lower MAE/RMSE is better. Positive improvement indicates the XGBoost model performed better than the simple mean.")
            else:
                 st.caption("Baseline or model metrics missing, cannot calculate improvement.")


            # --- Display Plots Side-by-Side ---
            st.markdown("---") # Separator before plots
            st.markdown("#### Visualizations (Test Set)")
            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                st.markdown("###### Actual vs. Predicted")
                if st.session_state.get('xgb_plot_fig'):
                    # Pass use_container_width=False here
                    st.pyplot(st.session_state.xgb_plot_fig, use_container_width=False)
                else:
                    st.warning("Actual vs Predicted plot not available.")

            with plot_col2:
                st.markdown("###### Feature Importances")
                if st.session_state.get('xgb_feature_importances_fig'):
                    # Pass use_container_width=False here
                    st.pyplot(st.session_state.xgb_feature_importances_fig, use_container_width=False)
                else:
                    st.warning("Feature importance plot not available.")
            # --- End Side-by-Side Display ---


            if st.button("Hide Results", key="hide_xgb_results_btn"):
                st.session_state.xgb_model_trained_success = False # Hide results section
                st.session_state.xgb_feature_importances_fig = None # Clear figure
                st.session_state.xgb_plot_fig = None # Clear figure
                st.rerun()
        else:
            st.info("Train and evaluate the XGBoost model first to see results.")

    # ==================================================
    # Section: Business Insights from Clustering (Multi-Level)
    # ==================================================
    st.markdown("---")
    st.header("Business Insights from Clustering")
    with st.expander("Show Cluster Profiles and Campaign Strategies", expanded=True):

        st.markdown("""
        Explore cluster profiles at two levels:
        - **Broad Insights:** Focus on key aggregate metrics (Recency, Frequency, Monetary, Campaign Acceptance).
        - **Detailed Insights:** Dive into the components of spending, purchase channels, and campaign responses, along with core metrics. Select columns to display.
        *(Requires Clustering steps and XGBoost training to be completed)*
        """)

        # --- Prerequisite Checks ---
        base_data_exists = 'data_copy_engineered' in st.session_state and st.session_state.data_copy_engineered is not None
        hier_labels_exist = base_data_exists and st.session_state.get('hierarchical_done', False) and st.session_state.get('hierarchical_labels_for_tsne') is not None
        kmeans_labels_exist = base_data_exists and st.session_state.get('kmeans_done', False) and st.session_state.get('kmeans_labels') is not None
        xgb_trained = st.session_state.get('xgb_model_trained_success', False)
        feature_importance_df = None # Initialize

        if not base_data_exists:
            st.warning("Run 'Feature Engineering' first to provide base data.")
        if not hier_labels_exist and not kmeans_labels_exist:
            st.warning("Run 'Hierarchical Clustering' or 'K-Means Clustering' first.")
        if not xgb_trained:
            st.info("Train the 'XGBoost Model' to enable campaign suggestions based on feature importance.")

        # --- Define Column Groups ---
        # Attempt to get Age column
        age_col_insight = None
        if base_data_exists:
            if 'Age_On_Last_Enrollment_Date' not in st.session_state.data_copy_engineered.columns:
                # Try to calculate it silently here if needed, maybe add a note if successful/failed
                try:
                    data_eng_temp = st.session_state.data_copy_engineered.copy()
                    data_eng_temp['Dt_Customer'] = pd.to_datetime(data_eng_temp['Dt_Customer'], errors='coerce')
                    data_eng_temp['Year_Birth'] = pd.to_numeric(data_eng_temp['Year_Birth'], errors='coerce')
                    data_eng_temp.dropna(subset=['Dt_Customer', 'Year_Birth'], inplace=True)
                    latest_enrollment_date_insight = data_eng_temp['Dt_Customer'].max()
                    data_eng_temp['Age_On_Last_Enrollment_Date'] = latest_enrollment_date_insight.year - data_eng_temp['Year_Birth']
                    st.session_state.data_copy_engineered = data_eng_temp # Update state
                    age_col_insight = 'Age_On_Last_Enrollment_Date'
                    # st.caption("Successfully calculated 'Age_On_Last_Enrollment_Date' for insights.") # Optional user feedback
                except Exception:
                    st.caption("Note: Could not calculate 'Age_On_Last_Enrollment_Date', age insights unavailable.")
                    age_col_insight = None
            else:
                age_col_insight = 'Age_On_Last_Enrollment_Date'

        # Core metrics (always include if available)
        core_cols = [col for col in [age_col_insight, 'Income', 'Recency'] if col is not None]
        # Aggregate metrics for Broad view
        broad_agg_cols = ['Frequency', 'Monetary', 'NumAcceptedCmps']
        # Component metrics for Detailed view
        freq_components = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']
        monetary_components = ['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds']
        campaign_components = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']

        # All potential columns for detailed view selector
        all_detailed_cols = core_cols + broad_agg_cols + freq_components + monetary_components + campaign_components

        # --- Helper function for Aggregation ---
        def get_cluster_summary(data_with_labels, cluster_col_name, columns_to_agg):
            """Aggregates specified columns for cluster profiling."""
            agg_dict_named = {}
            valid_columns = [col for col in columns_to_agg if col in data_with_labels.columns] # Ensure columns exist

            for col in valid_columns:
                # Use median for potentially skewed core metrics, mean for others
                agg_func = 'median' if col in core_cols else 'mean'
                agg_dict_named[col] = pd.NamedAgg(column=col, aggfunc=agg_func)

            # Add Size
            agg_dict_named['Size'] = pd.NamedAgg(column=cluster_col_name, aggfunc='size')

            if not agg_dict_named: # Handle case where no valid columns are found
                return pd.DataFrame()

            try:
                summary_df = data_with_labels.groupby(cluster_col_name).agg(**agg_dict_named)
                return summary_df
            except Exception as agg_e:
                st.error(f"Error during aggregation for {cluster_col_name}: {agg_e}")
                return pd.DataFrame() # Return empty DF on error

        # --- Prepare Data & Fetch Importance (Run once if possible) ---
        data_hier_insights = None
        data_kmeans_insights = None
        index_source_df = None

        if base_data_exists:
            # Determine source index (post-outlier or pre-outlier)
            if st.session_state.get('outliers_removed', False) and 'data_no_outliers' in st.session_state and st.session_state.data_no_outliers is not None:
                index_source_df = st.session_state.data_no_outliers
                index_source_caption = "using index from data *after* outlier removal."
            elif 'data_prepared_for_modeling' in st.session_state and st.session_state.data_prepared_for_modeling is not None:
                index_source_df = st.session_state.data_prepared_for_modeling
                index_source_caption = "using index from data *before* outlier removal (but after preparation)."
            else:
                st.warning("Could not identify source data index for mapping cluster labels.")

            if index_source_df is not None:
                # Prepare Hierarchical data
                if hier_labels_exist and len(st.session_state.hierarchical_labels_for_tsne) == len(index_source_df):
                    data_hier_insights = st.session_state.data_copy_engineered.loc[index_source_df.index].copy()
                    data_hier_insights['Hierarchical_Cluster'] = st.session_state.hierarchical_labels_for_tsne
                elif hier_labels_exist:
                    st.warning("Mismatch between Hierarchical labels and source index length.")

                # Prepare K-Means data
                if kmeans_labels_exist and len(st.session_state.kmeans_labels) == len(index_source_df):
                    data_kmeans_insights = st.session_state.data_copy_engineered.loc[index_source_df.index].copy()
                    data_kmeans_insights['KMeans_Cluster'] = st.session_state.kmeans_labels
                elif kmeans_labels_exist:
                    st.warning("Mismatch between K-Means labels and source index length.")

            # Fetch Feature Importance
            if xgb_trained and st.session_state.get('xgb_model_fitted') is not None:
                try:
                    model_xgb = st.session_state.xgb_model_fitted
                    importances = model_xgb.feature_importances_
                    try: # Try getting names from booster
                        feature_names = model_xgb.get_booster().feature_names
                    except Exception: feature_names = None
                    if feature_names is None: # Fallback
                        data_ref = st.session_state.get('data_prepared_for_modeling')
                        if data_ref is not None:
                            X_ref = data_ref.select_dtypes(include=np.number).drop(columns=['NumAcceptedCmps'], errors='ignore')
                            flags_to_drop_ref = [c for c in campaign_components if c in X_ref.columns] # Use defined list
                            X_ref = X_ref.drop(columns=flags_to_drop_ref, errors='ignore')
                            feature_names = X_ref.columns.tolist()
                        else: feature_names = [f'f{i}' for i in range(len(importances))]
                    if len(feature_names) == len(importances):
                        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                    else: st.warning(f"Feature name/importance mismatch. Importances may be unreliable.")
                except Exception as imp_err: st.warning(f"Could not retrieve feature importances: {imp_err}")

        # --- Create Tabs for Broad and Detailed Insights ---
        tab1, tab2 = st.tabs(["Broad Insights", "Detailed Insights"])

        with tab1:
            st.subheader("Broad Cluster Profiles & Strategies")
            st.caption(f"Aggregated profiles based on key metrics ({index_source_caption if index_source_df is not None else ''})")

            # --- Broad Hierarchical ---
            st.markdown("---")
            st.markdown("#### Hierarchical Clustering (Broad)")
            if data_hier_insights is not None:
                cols_for_broad = core_cols + broad_agg_cols
                hier_broad_summary = get_cluster_summary(data_hier_insights, 'Hierarchical_Cluster', cols_for_broad)
                if not hier_broad_summary.empty:
                    st.dataframe(hier_broad_summary.style.format("{:,.1f}").background_gradient(cmap='viridis', axis=0))
                    # Campaign Suggestions (using broad summary)
                    # --- Re-define key column names here for safety ---
                    income_col_insight = 'Income'
                    recency_col_insight = 'Recency'
                    # ----------------------------------------------------
                    if xgb_trained and feature_importance_df is not None and hier_broad_summary is not None and not hier_broad_summary.empty: # Check summary exists
                        st.markdown("**Campaign Suggestions (Hierarchical - Broad):**")
                        important_features = feature_importance_df['Feature'].head(5).tolist()
                        st.write(f"*Top driving features (XGBoost):* `{', '.join(important_features)}`")

                        # Calculate overall medians/means safely
                        # Ensure the column name variable exists AND the column is in the DataFrame
                        overall_median_income = hier_broad_summary[income_col_insight].median() if income_col_insight in hier_broad_summary.columns else None
                        overall_median_recency = hier_broad_summary[recency_col_insight].median() if recency_col_insight in hier_broad_summary.columns else None
                    if xgb_trained and feature_importance_df is not None and hier_broad_summary is not None and not hier_broad_summary.empty: # Check summary exists
                        st.markdown("**Campaign Suggestions (Hierarchical - Broad):**")
                        important_features = feature_importance_df['Feature'].head(5).tolist()
                        st.write(f"*Top driving features (XGBoost):* `{', '.join(important_features)}`")

                        # Calculate overall medians/means safely
                        overall_median_income = hier_broad_summary[income_col_insight].median() if income_col_insight in hier_broad_summary.columns else None
                        overall_median_recency = hier_broad_summary[recency_col_insight].median() if recency_col_insight in hier_broad_summary.columns else None

                        for cluster_id, profile in hier_broad_summary.iterrows():
                            st.markdown(f"**Cluster {cluster_id} (Size: {profile['Size']:.0f}):**")
                            insights = []
                            # Check column exists in profile before using
                            if age_col_insight and age_col_insight in profile and profile[age_col_insight] > 55: insights.append("Older.")
                            elif age_col_insight and age_col_insight in profile and profile[age_col_insight] < 40: insights.append("Younger.")

                            # Check overall median was calculable AND column exists in profile
                            if overall_median_income is not None and income_col_insight in profile:
                                if profile[income_col_insight] > overall_median_income * 1.1: insights.append("Higher Income.")
                                elif profile[income_col_insight] < overall_median_income * 0.9: insights.append("Lower Income.")
                            elif income_col_insight in profile: # If median failed but col exists, note it
                                insights.append("Avg Income.") # Or some other default

                            if overall_median_recency is not None and recency_col_insight in profile:
                                if profile[recency_col_insight] < overall_median_recency * 0.8: insights.append("Very Recent.")
                                elif profile[recency_col_insight] > overall_median_recency * 1.2: insights.append("Less Recent.")
                            elif recency_col_insight in profile:
                                insights.append("Avg Recency.")

                            suggestions = []
                            # Check column exists in important_features list AND insights were generated
                            if income_col_insight in important_features and "Higher Income." in insights: suggestions.append("Premium offers.")
                            if recency_col_insight in important_features and "Less Recent." in insights: suggestions.append("Reactivation.")
                            if recency_col_insight in important_features and "Very Recent." in insights: suggestions.append("Loyalty/Welcome.")
                            # ... add more rules ...
                            if not insights: insights.append("Avg profile.")
                            if not suggestions: suggestions.append("Standard.")
                            st.write(f"- **Profile:** {' '.join(insights)} **Suggestions:** {' '.join(suggestions)}")
                    elif hier_broad_summary is None or hier_broad_summary.empty:
                        st.info("Broad summary table missing, cannot generate suggestions.")
                    elif xgb_trained and feature_importance_df is None:
                        st.warning("Could not retrieve feature importances for suggestions.")
                    elif not xgb_trained:
                        st.info("Train XGBoost model for campaign suggestions.")
                else:
                    st.warning("Could not generate broad summary for Hierarchical clusters.")
            else:
                st.info("Hierarchical clustering results not available or index mismatch.")

            # --- Broad K-Means ---
            st.markdown("---")
            st.markdown("#### K-Means Clustering (Broad)")
            if data_kmeans_insights is not None:
                cols_for_broad = core_cols + broad_agg_cols
                kmeans_broad_summary = get_cluster_summary(data_kmeans_insights, 'KMeans_Cluster', cols_for_broad)
                if not kmeans_broad_summary.empty:
                    st.dataframe(kmeans_broad_summary.style.format("{:,.1f}").background_gradient(cmap='viridis', axis=0))
                    # Campaign Suggestions (using broad summary)
                    # --- Re-define key column names here for safety ---
                    income_col_insight = 'Income'
                    recency_col_insight = 'Recency'
                    # ----------------------------------------------------
                    if xgb_trained and feature_importance_df is not None and kmeans_broad_summary is not None and not kmeans_broad_summary.empty: # Check summary exists
                        st.markdown("**Campaign Suggestions (K-Means - Broad):**")
                        important_features = feature_importance_df['Feature'].head(5).tolist()
                        st.write(f"*Top driving features (XGBoost):* `{', '.join(important_features)}`")

                        # Calculate overall medians/means safely
                        # Ensure the column name variable exists AND the column is in the DataFrame
                        overall_median_income_km = kmeans_broad_summary[income_col_insight].median() if income_col_insight in kmeans_broad_summary.columns else None
                        overall_median_recency_km = kmeans_broad_summary[recency_col_insight].median() if recency_col_insight in kmeans_broad_summary.columns else None
                    if xgb_trained and feature_importance_df is not None and kmeans_broad_summary is not None and not kmeans_broad_summary.empty: # Check summary exists
                        st.markdown("**Campaign Suggestions (K-Means - Broad):**")
                        important_features = feature_importance_df['Feature'].head(5).tolist()
                        st.write(f"*Top driving features (XGBoost):* `{', '.join(important_features)}`")

                        # Calculate overall medians/means safely
                        overall_median_income_km = kmeans_broad_summary[income_col_insight].median() if income_col_insight in kmeans_broad_summary.columns else None
                        overall_median_recency_km = kmeans_broad_summary[recency_col_insight].median() if recency_col_insight in kmeans_broad_summary.columns else None

                        for cluster_id, profile in kmeans_broad_summary.iterrows():
                            st.markdown(f"**Cluster {cluster_id} (Size: {profile['Size']:.0f}):**")
                            insights = []
                            # Check column exists in profile before using
                            if age_col_insight and age_col_insight in profile and profile[age_col_insight] > 55: insights.append("Older.")
                            elif age_col_insight and age_col_insight in profile and profile[age_col_insight] < 40: insights.append("Younger.")

                            # Check overall median was calculable AND column exists in profile
                            if overall_median_income_km is not None and income_col_insight in profile:
                                if profile[income_col_insight] > overall_median_income_km * 1.1: insights.append("Higher Income.")
                                elif profile[income_col_insight] < overall_median_income_km * 0.9: insights.append("Lower Income.")
                            elif income_col_insight in profile:
                                insights.append("Avg Income.")

                            if overall_median_recency_km is not None and recency_col_insight in profile:
                                if profile[recency_col_insight] < overall_median_recency_km * 0.8: insights.append("Very Recent.")
                                elif profile[recency_col_insight] > overall_median_recency_km * 1.2: insights.append("Less Recent.")
                            elif recency_col_insight in profile:
                                insights.append("Avg Recency.")

                            suggestions = []
                            # Check column exists in important_features list AND insights were generated
                            if income_col_insight in important_features and "Higher Income." in insights: suggestions.append("Premium offers.")
                            if recency_col_insight in important_features and "Less Recent." in insights: suggestions.append("Reactivation.")
                            if recency_col_insight in important_features and "Very Recent." in insights: suggestions.append("Loyalty/Welcome.")
                            # ... add more rules ...
                            if not insights: insights.append("Avg profile.")
                            if not suggestions: suggestions.append("Standard.")
                            st.write(f"- **Profile:** {' '.join(insights)} **Suggestions:** {' '.join(suggestions)}")
                    elif kmeans_broad_summary is None or kmeans_broad_summary.empty:
                        st.info("Broad summary table missing, cannot generate suggestions.")
                    elif xgb_trained and feature_importance_df is None:
                        st.warning("Could not retrieve feature importances for suggestions.")
                    elif not xgb_trained:
                        st.info("Train XGBoost model for campaign suggestions.")
                else:
                    st.warning("Could not generate broad summary for K-Means clusters.")
            else:
                st.info("K-Means clustering results not available or index mismatch.")


        with tab2:
            st.subheader("Detailed Cluster Profiles & Strategies")
            st.caption(f"In-depth profiles including component metrics ({index_source_caption if index_source_df is not None else ''}). Use the text box to filter columns.")

            # --- Column Selection Text Box ---
            st.markdown("##### Select Columns to Display in Profiles")
            # Provide default columns
            default_cols_display = core_cols + broad_agg_cols[:1] # e.g., Age, Income, Recency, Frequency
            cols_list_str = ", ".join(all_detailed_cols)
            st.text(f"Available columns: {cols_list_str}")
            user_cols_input = st.text_area(
                "Enter column names separated by commas or newlines (leave blank for defaults):",
                value=", ".join(default_cols_display), # Pre-fill with defaults
                height=100,
                key="detailed_cols_selector"
            )

            # Parse user input
            selected_cols_to_display = [] # Initialize
            if user_cols_input.strip():
                selected_cols = [col.strip() for col in user_cols_input.replace('\n', ',').split(',') if col.strip()]
                # Validate selected columns against all available detailed columns
                valid_selected_cols = [col for col in selected_cols if col in all_detailed_cols]
                invalid_selected_cols = [col for col in selected_cols if col not in all_detailed_cols]
                if invalid_selected_cols:
                    st.warning(f"Ignoring invalid column names: {', '.join(invalid_selected_cols)}")
                if not valid_selected_cols: # If all were invalid or selection is empty after strip
                    st.info("No valid columns selected, showing defaults.")
                    selected_cols_to_display = default_cols_display
                else:
                    selected_cols_to_display = valid_selected_cols
            else: # If input is empty, use defaults
                selected_cols_to_display = default_cols_display

            # Always include 'Size' in the display if available
            if 'Size' not in selected_cols_to_display:
                selected_cols_to_display.insert(0, 'Size') # Add Size at the beginning


            # --- Detailed Hierarchical ---
            st.markdown("---")
            st.markdown("#### Hierarchical Clustering (Detailed Profile)")
            hier_detailed_summary = None # Initialize
            if data_hier_insights is not None:
                hier_detailed_summary = get_cluster_summary(data_hier_insights, 'Hierarchical_Cluster', all_detailed_cols)
                if not hier_detailed_summary.empty:
                    # Filter the summary table based on user selection
                    display_cols_hier = [col for col in selected_cols_to_display if col in hier_detailed_summary.columns]
                    if not display_cols_hier or (len(display_cols_hier) == 1 and display_cols_hier[0] == 'Size'):
                        st.warning("No valid data columns selected for display in profile.")
                    else:
                        st.dataframe(hier_detailed_summary[display_cols_hier].style.format("{:,.1f}").background_gradient(cmap='viridis', axis=0))
                else:
                    st.warning("Could not generate detailed summary for Hierarchical clusters.")
            else:
                st.info("Hierarchical clustering results not available or index mismatch.")

            # --- Detailed Hierarchical Campaign Strategy ---
            st.markdown("##### Campaign Strategies (Hierarchical - Detailed)")
            if hier_detailed_summary is not None and not hier_detailed_summary.empty:
                st.write("Suggestions based on detailed component preferences (product, place, past campaigns):")

                # Calculate overall means for components from the detailed summary
                component_overall_means_hier = {}
                for col_group in [monetary_components, freq_components, campaign_components]:
                    for col in col_group:
                        if col in hier_detailed_summary.columns:
                            component_overall_means_hier[col] = hier_detailed_summary[col].mean()

                for cluster_id, profile in hier_detailed_summary.iterrows():
                    st.markdown(f"**Cluster {cluster_id} (Size: {profile['Size']:.0f}):**")
                    detailed_suggestions = []

                    # Product Preferences -> Suggestions
                    for col in monetary_components:
                        if col in profile and col in component_overall_means_hier and profile[col] > component_overall_means_hier[col] * 1.2: # Threshold: 20% above average
                            product_name = col.replace('Mnt', '').replace('Products','').replace('Prods','') # Simplify name
                            detailed_suggestions.append(f"Promote {product_name}.")

                    # Place Preferences -> Suggestions
                    for col in freq_components:
                        if col in profile and col in component_overall_means_hier and profile[col] > component_overall_means_hier[col] * 1.2:
                            place_name = col.replace('Num','').replace('Purchases','')
                            if place_name == 'Deals':
                                detailed_suggestions.append("Highlight Discounts/Deals.")
                            else:
                                detailed_suggestions.append(f"Target via {place_name} channel.")

                    # Past Campaign Response -> Suggestions
                    for col in campaign_components:
                        if col in profile and col in component_overall_means_hier and profile[col] > component_overall_means_hier[col] * 1.1: # Threshold: 10% above average acceptance rate
                            campaign_name = col.replace('Accepted','').replace('Cmp',' Campaign ')
                            if campaign_name == 'Response': campaign_name = 'Last Campaign'
                            detailed_suggestions.append(f"Responded well to {campaign_name} type.")

                    if not detailed_suggestions:
                        detailed_suggestions.append("No strong component preferences detected; use broad strategy or general offers.")

                    st.write(f"- **Detailed Suggestions:** {'; '.join(detailed_suggestions)}")
            else:
                st.info("Detailed Hierarchical profile needed for detailed campaign suggestions.")


            # --- Detailed K-Means ---
            st.markdown("---")
            st.markdown("#### K-Means Clustering (Detailed Profile)")
            kmeans_detailed_summary = None # Initialize
            if data_kmeans_insights is not None:
                kmeans_detailed_summary = get_cluster_summary(data_kmeans_insights, 'KMeans_Cluster', all_detailed_cols)
                if not kmeans_detailed_summary.empty:
                    # Filter the summary table based on user selection
                    display_cols_kmeans = [col for col in selected_cols_to_display if col in kmeans_detailed_summary.columns]
                    if not display_cols_kmeans or (len(display_cols_kmeans) == 1 and display_cols_kmeans[0] == 'Size'):
                        st.warning("No valid data columns selected for display in profile.")
                    else:
                        st.dataframe(kmeans_detailed_summary[display_cols_kmeans].style.format("{:,.1f}").background_gradient(cmap='viridis', axis=0))
                else:
                    st.warning("Could not generate detailed summary for K-Means clusters.")
            else:
                st.info("K-Means clustering results not available or index mismatch.")

            # --- Detailed K-Means Campaign Strategy ---
            st.markdown("##### Campaign Strategies (K-Means - Detailed)")
            if kmeans_detailed_summary is not None and not kmeans_detailed_summary.empty:
                st.write("Suggestions based on detailed component preferences (product, place, past campaigns):")

                # Calculate overall means for components from the detailed summary
                component_overall_means_kmeans = {}
                for col_group in [monetary_components, freq_components, campaign_components]:
                    for col in col_group:
                        if col in kmeans_detailed_summary.columns:
                            component_overall_means_kmeans[col] = kmeans_detailed_summary[col].mean()

                for cluster_id, profile in kmeans_detailed_summary.iterrows():
                    st.markdown(f"**Cluster {cluster_id} (Size: {profile['Size']:.0f}):**")
                    detailed_suggestions = []

                    # Product Preferences -> Suggestions
                    for col in monetary_components:
                        if col in profile and col in component_overall_means_kmeans and profile[col] > component_overall_means_kmeans[col] * 1.2:
                            product_name = col.replace('Mnt', '').replace('Products','').replace('Prods','')
                            detailed_suggestions.append(f"Promote {product_name}.")

                    # Place Preferences -> Suggestions
                    for col in freq_components:
                        if col in profile and col in component_overall_means_kmeans and profile[col] > component_overall_means_kmeans[col] * 1.2:
                            place_name = col.replace('Num','').replace('Purchases','')
                            if place_name == 'Deals':
                                detailed_suggestions.append("Highlight Discounts/Deals.")
                            else:
                                detailed_suggestions.append(f"Target via {place_name} channel.")

                    # Past Campaign Response -> Suggestions
                    for col in campaign_components:
                        if col in profile and col in component_overall_means_kmeans and profile[col] > component_overall_means_kmeans[col] * 1.1:
                            campaign_name = col.replace('Accepted','').replace('Cmp',' Campaign ')
                            if campaign_name == 'Response': campaign_name = 'Last Campaign'
                            detailed_suggestions.append(f"Responded well to {campaign_name} type.")

                    if not detailed_suggestions:
                        detailed_suggestions.append("No strong component preferences detected; use broad strategy or general offers.")

                    st.write(f"- **Detailed Suggestions:** {'; '.join(detailed_suggestions)}")
            else:
                st.info("Detailed K-Means profile needed for detailed campaign suggestions.")

    # ==================================================
    # End of Business Insights Section
    # ==================================================
    
    
    # --- End of the main `if st.session_state.data is not None:` block ---

else:
    st.error("Data could not be loaded. Cannot proceed.")

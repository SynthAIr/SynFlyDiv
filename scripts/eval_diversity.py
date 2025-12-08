# Diversity evaluation conducted on (features used in the generation + relational features). 
# This evaluation is not used in optuna optimization. 

# Import necessary libraries
import argparse
import sys
from utils import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def rename_model(name: str) -> str:
    if name.startswith("Default "):
        return "$" + name.replace("Default ", "") + "$"
    elif name.startswith("Optimal "):
        return "$" + name.replace("Optimal ", "") + "_{opt.}$"
    elif name == "GC":
        return r"$GC$"
    elif name == "real":
        return r"$real$"
    return name

def scale_real_synth(real, synthetic):
    # To apply scalling to the numerical columns only
    numeric_cols = real.select_dtypes(include=['number']).columns
    # print("Scaling numeric columns: ", numeric_cols.tolist())

    # Replace NaN with 0 to avoide problems with PCA & t-SNE
    real_numeric = real[numeric_cols].fillna(0, inplace=False)
    synthetic_numeric = synthetic[numeric_cols].fillna(0, inplace=False)

    scaler = MinMaxScaler()  # I am dealing with times and +ve values so [0,1] is fine
    # scaler = StandardScaler() # Bad as the mean will be around 0

    # Fit to real only
    scaled_real = pd.DataFrame(scaler.fit_transform(real_numeric),
                               columns=real_numeric.columns,
                               index=real_numeric.index)

    scaled_synthetic = pd.DataFrame(scaler.transform(synthetic_numeric),
                                    columns=synthetic_numeric.columns,
                                    index=synthetic_numeric.index)
    return scaled_real, scaled_synthetic

#===========================================================================
def check_pca_explained_variance(scaled_real):
    plt.rcdefaults()
    pca = PCA()
    real_flight_information_PCA = pca.fit_transform(scaled_real)
    real_flight_information_PCA = pd.DataFrame(real_flight_information_PCA,
                                               index=scaled_real.index)

    p_components = range(1, pca.n_components_ + 1)
    plt.figure(figsize=(15, 5))
    plt.rc('font', size=15)
    plt.bar(p_components,
            pca.explained_variance_ratio_ * 100,
            color="blue",
            label='Individual explained variance')
    plt.plot(p_components,
             np.cumsum(pca.explained_variance_ratio_ * 100),
             marker='o',
             color="red",
             label='Cumulative explained variance')
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance ratio (%)')
    plt.xticks(p_components)
    plt.legend(loc='best')
    plt.title("Importance of the Principal Components")
    plt.show()

#===========================================================================
def check_diversity_pca(scaled_real, scaled_synthetic, plot_real_on_top=False):
    warnings.simplefilter("ignore", UserWarning)
    plt.rcdefaults()
    pca = PCA(n_components=2, svd_solver='auto')
    
    # Fit to real only
    real_flight_information_PCA = pca.fit_transform(scaled_real)
    real_flight_information_PCA = pd.DataFrame(real_flight_information_PCA,
                                               index=scaled_real.index,
                                               columns=['PC1', 'PC2'])
    # For synthetic data
    synthetic_flight_information_PCA = pca.transform(scaled_synthetic)
    synthetic_flight_information_PCA = pd.DataFrame(synthetic_flight_information_PCA, 
                                                    index=scaled_synthetic.index,
                                                    columns=['PC1', 'PC2'])

    # Scatter plot of the first two principal components
    plt.figure(figsize=(5, 5))

    if plot_real_on_top:
        plt.scatter(synthetic_flight_information_PCA['PC1'],
                    synthetic_flight_information_PCA['PC2'],
                    s=10,
                    color='red',
                    alpha=0.6,
                    label='Syn. Flight Diversions')
        plt.scatter(real_flight_information_PCA['PC1'],
                    real_flight_information_PCA['PC2'],
                    s=10,
                    color='blue',
                    alpha=0.6,
                    label='Real Flight Diversions')
    else:
        plt.scatter(real_flight_information_PCA['PC1'],
                    real_flight_information_PCA['PC2'],
                    s=10,
                    color='blue',
                    alpha=0.6,
                    label='Real Flight Diversions')
        plt.scatter(synthetic_flight_information_PCA['PC1'],
                    synthetic_flight_information_PCA['PC2'],
                    s=10,
                    color='red',
                    alpha=0.6,
                    label='Syn. Flight Diversions')
    plt.xlabel('$1^{{st}}$ PC ({}% variance)'.format(
        round(pca.explained_variance_ratio_[0] * 100, 1)),
               fontsize=14)
    plt.ylabel('$2^{{nd}}$ PC ({}% variance)'.format(
        round(pca.explained_variance_ratio_[1] * 100, 1)),
               fontsize=14)
    plt.tick_params(axis='both', labelsize=14)  
    plt.title('PCA: Real vs Syn. Flight Diversions', fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    model = model_name.lower().replace(" ", "_")
    # plt.savefig(f'../data/outputs/results/pca_{model}_{synth_sample_size}.png', dpi=300)
    plt.show()
    
    return real_flight_information_PCA, synthetic_flight_information_PCA

#===========================================================================
#===========================================================================
def check_tsne_perplexity_effect(scaled_real, scaled_synthetic, perplexities=[5, 30, 50]):
    """
    Function to visualize the effect of different perplexity values on t-SNE
    """
    warnings.simplefilter("ignore", UserWarning)
    plt.rcdefaults()
    
    # Combine real and synthetic data for consistent t-SNE embedding
    combined_data = pd.concat([scaled_real, scaled_synthetic], axis=0)
    
    # Create labels to distinguish real vs synthetic
    labels = ['Real'] * len(scaled_real) + ['Synthetic'] * len(scaled_synthetic)
    
    fig, axes = plt.subplots(1, len(perplexities), figsize=(4 * len(perplexities), 5))
    if len(perplexities) == 1:
        axes = [axes]
    
    for idx, perplexity in enumerate(perplexities):
        # Apply t-SNE
        tsne = TSNE(n_components=2, 
                   perplexity=perplexity, 
                   random_state=42, 
                   n_iter=1000,
                   learning_rate='auto',
                   init='pca')
        
        tsne_results = tsne.fit_transform(combined_data)
        
        # Split results back into real and synthetic
        real_tsne = tsne_results[:len(scaled_real)]
        synthetic_tsne = tsne_results[len(scaled_real):]
        
        # Plot
        axes[idx].scatter(real_tsne[:, 0], real_tsne[:, 1], 
                         s=10, color='blue', alpha=0.6, label='Real Flight Diversions')
        axes[idx].scatter(synthetic_tsne[:, 0], synthetic_tsne[:, 1], 
                         s=10, color='red', alpha=0.6, label='Syn. Flight Diversions')
        
        axes[idx].set_xlabel('$1^{{st}}$ t-SNE component', fontsize=14)
        axes[idx].set_ylabel('$2^{{nd}}$ t-SNE component', fontsize=14)
        axes[idx].set_title(f't-SNE: Perplexity={perplexity}')
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.suptitle('t-SNE: Real vs Syn. Flight Diversions - Perplexity Comparison', 
                 y=1.02, fontsize=16)
    plt.show()

#===========================================================================
def check_diversity_tsne(scaled_real, scaled_synthetic, perplexity=30, random_state=45, plot_real_on_top=False):
    """
    Visualize diversity between real and synthetic datasets using t-SNE.
    Equivalent to check_diversity_pca.
    """
    warnings.simplefilter("ignore", UserWarning)
    plt.rcdefaults()
    
    # Combine real and synthetic data for consistent t-SNE embedding
    combined_data = pd.concat([scaled_real, scaled_synthetic], axis=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, 
               perplexity=perplexity, 
               random_state=random_state, 
               n_iter=1000,
               learning_rate='auto',
               init='pca')
    
    tsne_results = tsne.fit_transform(combined_data)
    
    # Split results back into real and synthetic
    real_tsne = tsne_results[:len(scaled_real)]
    synthetic_tsne = tsne_results[len(scaled_real):]
    
    # Convert to DataFrames
    real_flight_information_TSNE = pd.DataFrame(real_tsne, 
                                               index=scaled_real.index,
                                               columns=['TSNE1', 'TSNE2'])
    synthetic_flight_information_TSNE = pd.DataFrame(synthetic_tsne, 
                                                    index=scaled_synthetic.index,
                                                    columns=['TSNE1', 'TSNE2'])
    
    # Scatter plot of the first two t-SNE components
    plt.figure(figsize=(5, 5))
    
    if plot_real_on_top:
        plt.scatter(synthetic_flight_information_TSNE['TSNE1'],
                    synthetic_flight_information_TSNE['TSNE2'],
                    s=10,
                    color='red',
                    alpha=0.6,
                    label='Syn. Flight Diversions')
        plt.scatter(real_flight_information_TSNE['TSNE1'],
                    real_flight_information_TSNE['TSNE2'],
                    s=10,
                    color='blue',
                    alpha=0.6,
                    label='Real Flight Diversions')
    else:
        plt.scatter(real_flight_information_TSNE['TSNE1'],
                    real_flight_information_TSNE['TSNE2'],
                    s=10,
                    color='blue',
                    alpha=0.6,
                    label='Real Flight Diversions')
        plt.scatter(synthetic_flight_information_TSNE['TSNE1'],
                    synthetic_flight_information_TSNE['TSNE2'],
                    s=10,
                    color='red',
                    alpha=0.6,
                    label='Syn. Flight Diversions')
    
    plt.xlabel('$1^{{st}}$ t-SNE component', fontsize=14)
    plt.ylabel('$2^{{nd}}$ t-SNE component', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.title(f't-SNE: Real vs Syn. Flight Diversions', fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    model = model_name.lower().replace(" ", "_")
    # plt.savefig(f'../data/outputs/results/tsne_{model}_{synth_sample_size}.png', dpi=300)
    plt.show()
    
    return real_flight_information_TSNE, synthetic_flight_information_TSNE

#===========================================================================
def compare_pca_tsne(scaled_real, scaled_synthetic, perplexity=30):
    """
    Function to compare PCA and t-SNE side by side
    """
    warnings.simplefilter("ignore", UserWarning)
    plt.rcdefaults()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    
    # PCA
    pca = PCA(n_components=2, svd_solver='auto')
    real_pca = pca.fit_transform(scaled_real)
    synthetic_pca = pca.transform(scaled_synthetic)
    
    ax1.scatter(real_pca[:, 0], real_pca[:, 1], 
               s=10, color='blue', alpha=0.6, label='Real Flight Diversions')
    ax1.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], 
               s=10, color='red', alpha=0.6, label='Syn. Flight Diversions')
    ax1.set_xlabel('$1^{{st}}$ PC ({}% variance)'.format(
        round(pca.explained_variance_ratio_[0] * 100, 1)), fontsize=14)
    ax1.set_ylabel('$2^{{nd}}$ PC ({}% variance)'.format(
        round(pca.explained_variance_ratio_[1] * 100, 1)), fontsize=14)
    ax1.set_title('PCA: Real vs Syn. Flight Diversions')
    ax1.legend()
    ax1.grid(True)
    
    # t-SNE
    combined_data = pd.concat([scaled_real, scaled_synthetic], axis=0)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
               n_iter=1000, learning_rate='auto', init='pca')
    tsne_results = tsne.fit_transform(combined_data)
    
    real_tsne = tsne_results[:len(scaled_real)]
    synthetic_tsne = tsne_results[len(scaled_real):]
    
    ax2.scatter(real_tsne[:, 0], real_tsne[:, 1], 
               s=10, color='blue', alpha=0.6, label='Real Flight Diversions')
    ax2.scatter(synthetic_tsne[:, 0], synthetic_tsne[:, 1], 
               s=10, color='red', alpha=0.6, label='Syn. Flight Diversions')
    ax2.set_xlabel('$1^{{st}}$ t-SNE component', fontsize=14)
    ax2.set_ylabel('$2^{{nd}}$ t-SNE component', fontsize=14)
    ax2.set_title(f't-SNE: Real vs Syn. Flight Diversions')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.suptitle('PCA vs t-SNE', y=1.02, fontsize=16)
    plt.show()
    
#===========================================================================
#===========================================================================
def check_class_balance(flight_info):
    # Get the id of the object passed to compare
    object_id = id(flight_info)
    # Find the variable name dynamically using globals()
    var_name = [
        name for name, value in globals().items() if id(value) == object_id
    ][0]
    var_name = var_name.split("_")[0].capitalize()
    var_name = "Syn." if var_name == "Synthetic" else var_name

    plt.rcdefaults()
    print("=" * 75)

    no_flights = flight_info.shape[0]
    # This will work even if the category 0 or 1 is not present
    delayed_dep = (flight_info["Departure Delay Label"].astype(float) == 1.0).sum()
    on_time_dep = (flight_info["Departure Delay Label"].astype(float) == 0.0).sum()
    delayed_arr = (flight_info["Arrival Delay Label"].astype(float) == 1.0).sum()
    on_time_arr = (flight_info["Arrival Delay Label"].astype(float) == 0.0).sum()

    print("Total number of flights --> ", no_flights)
    print("Departures --> {} on time, {} delayed, {} unknown".format(
        on_time_dep, delayed_dep, no_flights - on_time_dep - delayed_dep))
    print("Arrivals --> {} on time, {} delayed, {} unknown".format(
        on_time_arr, delayed_arr, no_flights - on_time_arr - delayed_arr))
    print("=" * 75)

    # Determine features and subplot count
    if on_time_arr == 0: # as in the case of diversions
        features = ["Departure Delay Label"]
    else:    
        features = ["Departure Delay Label", "Arrival Delay Label"]
    # Create subplots dynamically
    fig, axs = plt.subplots(1, len(features), figsize=(2.5 * len(features), 3))
    axs = np.atleast_1d(axs) # Ensures axs is always iterable


    # Define color mapping
    colors = {
        '1.0': '#ff7f0e',  # Orange for 1
        '0.0': '#1f77b4'  # Blue for 0
    }

    for i, feature in enumerate(features):
        value_counts = flight_info[feature].value_counts(dropna=False)

        # Sort the values to ensure consistent ordering (0 first, then 1)
        value_counts = value_counts.sort_index()

        labels = value_counts.index.astype(str)
        sizes = value_counts.values

        # Map colors to the corresponding values
        color_list = [colors[label] for label in labels]

        # Define function for percentage formatting with larger font
        def make_autopct(values):
            def my_autopct(pct):
                return f'{pct:.1f}%'

            return my_autopct

        # Increased font sizes for labels and percentage values
        axs[i].pie(sizes,
                   labels=labels,
                   colors=color_list,
                   autopct=make_autopct(sizes),
                   startangle=140,
                   textprops={'fontsize': 14
                              })  # This applies to both labels and percentages
        axs[i].axis('equal')
        axs[i].set_title("{} {}".format(var_name, feature), fontsize=12)

        if var_name == "Syn.":
            model = model_name
            # plt.savefig(f'../data/outputs/results/class_balance_{model_name.lower().replace(" ", "_")}_{synth_sample_size}.png', dpi=300)
        else:
            model = "real"
            # plt.savefig(f'../data/outputs/results/class_balance_{model}.png', dpi=300)
    plt.tight_layout()
    plt.show()
    
    # Store results
    results.append({
        "Model": model,
        "delayed_dep": round((delayed_dep / no_flights) * 100, 1),
        "on_time_dep": round((on_time_dep / no_flights) * 100, 1),
        "delayed_arr": round((delayed_arr / no_flights) * 100, 1),
        "on_time_arr": round((on_time_arr / no_flights) * 100, 1)
    })

#===========================================================================
def plot_comparaison(df_results):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_results.copy()
    # Clean model names by removing "Optimal" and "Default"
    df_plot["Model"] = df_plot["Model"].apply(rename_model)

    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7.1, 5))
    colors = ['#FF3300', '#0099FF']  # Note: swapped colors for correct stacking order
    
    # Create horizontal stacked bar chart
    y_pos = np.arange(len(df_plot['Model']))
    height = 0.8
    
    # Plot horizontal bars (note: left parameter for stacking)
    p1 = ax.barh(y_pos, df_plot['on_time_dep'], height, label='On-Time Departures', 
                 color=colors[1])
    p2 = ax.barh(y_pos, df_plot['delayed_dep'], height, left=df_plot['on_time_dep'], 
                 label='Delayed Departures', color=colors[0])
    
    # Customize the plot
    ax.set_ylabel('Model', fontsize=14)
    ax.set_xlabel('Percentage (%)', fontsize=14)
    ax.set_title('Class balance: real vs. synthetic data', fontsize=14)
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Model'])
          
    # Add percentage labels on bars
    for i, (delayed, on_time) in enumerate(zip(df_plot['delayed_dep'], df_plot['on_time_dep'])):
        # On-time percentage (left part)
        ax.text(on_time/2, i, f'{on_time:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.1'))
        # Delayed percentage (right part)
        shift = 1.6  # small shift to move the label inside 
        ax.text(on_time + delayed/2 - shift, i, f'{delayed:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.1'))

    # Hatch the bars of optimised models (check original names)
    for p, model_name in zip(ax.patches, list(df_results['Model'].values) * 2):
        if 'Optimal' in model_name:
            p.set_hatch('//')
            p.set_edgecolor('black')
    
    # Create custom legend entry for hatched bars
    hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(hatch_patch)
    ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9)
    
    # Set x-axis limits and ticks
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    
    # Add grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Invert y-axis to have first model at top
#     ax.invert_yaxis()
    ax.set_ylim(-0.6, len(df_plot['Model']) + 0.3)  # add space above and below
    plt.tight_layout()
    # plt.savefig(f'../data/outputs/results/class_balance_comparison_{synth_sample_size}.png', dpi=300)
    plt.show()

#===========================================================================
def realism_evaluation():
    before_cleaning = synth_sample_size
    after_cleaning = synthetic_flight_information.shape[0]
    realism = round((after_cleaning / before_cleaning) * 100, 1)

    return before_cleaning, after_cleaning, realism

#===========================================================================
def main():
    # Scale the data
    scaled_real_flight_information, scaled_synthetic_flight_information = scale_real_synth(
        real_div_with_relational, synthetic_flight_information)
    
    # Plot the explained variance of PCA
    check_pca_explained_variance(scaled_real_flight_information)
    # Check diversity using PCA
    check_diversity_pca(scaled_real_flight_information, scaled_synthetic_flight_information)
    
    # Check diversity using t-SNE
    check_diversity_tsne(scaled_real_flight_information, scaled_synthetic_flight_information)
    # Check the effect of perplexity on t-SNE
    check_tsne_perplexity_effect(scaled_real_flight_information, scaled_synthetic_flight_information, 
                            perplexities=[5, 30, 50])
    # Compare PCA and t-SNE
    compare_pca_tsne(scaled_real_flight_information, scaled_synthetic_flight_information)

    # Check class balance
    check_class_balance(real_div_with_relational)
    check_class_balance(synthetic_flight_information)

#===========================================================================
if __name__ == "__main__":
    # List of available synthetic sample sizes
    synth_sizes = [300, 500, 1000, 2000, 5000, 10000, 20000]
    # List of available synthesizers
    synthesizers = ["gc", "ctgan", "tvae", "copgan"]
    # Only diverted flights (with relational features for statistical & diversity evaluation)
    real_div_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_with_relational.pkl')

    if len(sys.argv) > 1 and not any(arg.startswith('--') for arg in sys.argv[1:]):
        # Case 1: run via Jupyter (e.g. %run diversity_eval.py 300 gc tvae)
        synth_sample_size = int(sys.argv[1])
        synth_types = sys.argv[2:]
        # Validate the synth_sample_size
        if synth_sample_size not in synth_sizes:
            raise ValueError(f"Invalid synth_sample_size: {synth_sample_size}. Choose from {synth_sizes}")
        # Validate the synth_types
        for synth_type in synth_types:
            if synth_type not in synthesizers:
                raise ValueError(f"Invalid synth_type: {synth_type}. Choose from {synthesizers}")
    else:
        # Case 2: execute the script with command line arguments (e.g. python diversity_eval.py --synth_sample_size 300 --synth_types gc tvae)
        parser = argparse.ArgumentParser()
        parser.add_argument('--synth_sample_size', type=int, required=True, choices=synth_sizes)
        parser.add_argument('--synth_types', type=str, nargs='+', required=True, choices=synthesizers)
        args = parser.parse_args()
        synth_sample_size = args.synth_sample_size
        synth_types = args.synth_types

    results = []
    # Load synthetic (all features, do not exclude calculated features)
    for synth_type in synth_types:
        if synth_type == 'gc':
            # Default Gaussian Copula Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Diversity evaluation:{color.END} Default GC ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "GC"
            main()            

        
        elif synth_type == 'tvae':
            # Default TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Diversity evaluation:{color.END} Default TVAE ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default TVAE"
            main()

            # Optimal TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Diversity evaluation:{color.END} Optimal TVAE ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal TVAE"
            main()
        
        elif synth_type == 'ctgan':
            # Default CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Diversity evaluation:{color.END} Default CTGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default CTGAN"
            main()

            # Optimal CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Diversity evaluation:{color.END} Optimal CTGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal CTGAN"
            main()

        elif synth_type == 'copgan':
            # Default CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Diversity evaluation:{color.END} Default CopulaGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default CopulaGAN"
            main()

            # Optimal CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Diversity evaluation:{color.END} Optimal CopulaGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal CopulaGAN"
            main()

    df_class_balance = pd.DataFrame(results)
    # Drop duplicate real rows, keeping the first occurrence
    df_class_balance = df_class_balance.drop_duplicates().reset_index(drop=True)
    plot_comparaison(df_class_balance)

    # Save class balance results
    df_class_balance.to_pickle(f'../data/outputs/eval_results/df_class_balance_{synth_sample_size}.pkl')
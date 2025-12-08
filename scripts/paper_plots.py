# Import necessary libraries
import glob
import os
from utils import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import statsmodels.api as sm
import matplotlib.ticker as mticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Read evaluation results
df_realism_results = pd.read_pickle('../data/outputs/eval_results/df_realism_results_1000.pkl')
df_utility_results = pd.read_pickle('../data/outputs/eval_results/utility_augmentation_size/df_utility_results_1000.pkl')
df_similarity_results = pd.read_pickle('../data/outputs/eval_results/df_similarity_results_1000.pkl')
df_fidelity_results = pd.read_pickle('../data/outputs/eval_results/df_fidelity_results_1000.pkl')
df_class_balance = pd.read_pickle('../data/outputs/eval_results/df_class_balance_1000.pkl')

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

#===========================================================================
# Realism
#===========================================================================
def realism_comparaison(df_results):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_results.copy()
    
    # Clean model names by removing "Optimal" and "Default"
    df_plot["Model"] = df_plot["Model"].apply(rename_model)
    
    colors = [["#0099FF"]] 
    columns = [['Realism']]

    for i in range(len(columns)):
        ax = df_plot.plot(
            x="Model",
            y=columns[i],
            kind="bar",
            figsize=(5, 3.5),
            rot=45,
            color=colors[i] 
        )

    # Grid and limits
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Labels and title
    plt.ylabel("Realism score (%)", fontsize=14)
#     plt.xlabel("Model", fontsize=14)
    plt.xlabel("")
#     plt.title("Realism evaluation (higher is better)", fontsize=14)

    # Hatch the bars of optimised models (check original names)
    for p, original_name in zip(ax.patches, list(df_results['Model'].values) * len(columns[i])):
        if 'Optimal' in original_name:
            p.set_hatch('//')
            p.set_edgecolor('black')

    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{height:.1f}',
            xy=(p.get_x() + p.get_width() / 2., height),
            xytext=(0, 3),          
            textcoords='offset points',
            ha='center', 
            va='bottom',
            fontsize=10,
            rotation=90)

    # Create custom legend entry for hatched bars
    hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(hatch_patch)
    ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=11)

    plt.tight_layout()
    ax.set_ylim(0, 119)
    plt.savefig(f'../data/outputs/paper_plots/realism/realism_1000.png', dpi=300)
    plt.show()

#===========================================================================
# Diversity: PCA & t-SNE
#===========================================================================
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
def check_diversity_pca(scaled_real, scaled_synthetic, model_name, plot_real_on_top=False):
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
    plt.figure(figsize=(4, 3.5))

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
               fontsize=15)
    plt.ylabel('$2^{{nd}}$ PC ({}% variance)'.format(
        round(pca.explained_variance_ratio_[1] * 100, 1)),
               fontsize=14)
    plt.tick_params(axis='both', labelsize=12)  
#     plt.title('PCA: Real vs Syn. Flight Diversions', fontsize=16)
#     plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    model = model_name.lower().replace(" ", "_")
    plt.savefig(f'../data/outputs/paper_plots/diversity/pca_{model}_1000.png', dpi=300)
    plt.show()
#===========================================================================
def check_diversity_tsne(scaled_real, scaled_synthetic, model_name, plot_real_on_top=False):
    """
    Visualize diversity between real and synthetic datasets using t-SNE.
    Equivalent to check_diversity_pca.
    """
    warnings.simplefilter("ignore", UserWarning)
    plt.rcdefaults()
    
    perplexity=30; random_state=45;
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
    plt.figure(figsize=(4, 3.5))
    
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
    
    plt.xlabel('$1^{{st}}$ t-SNE component', fontsize=15)
    plt.ylabel('$2^{{nd}}$ t-SNE component', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
#     plt.title(f't-SNE: Real vs Syn. Flight Diversions', fontsize=16)
#     plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    model = model_name.lower().replace(" ", "_")
    plt.savefig(f'../data/outputs/paper_plots/diversity/tsne_{model}_1000.png', dpi=300)
    plt.show()
#===========================================================================
def check_diversity(real_div_with_relational, synthetic_flight_information, model_name,  plot_real_on_top):
    # Scale the data
    scaled_real_flight_information, scaled_synthetic_flight_information = scale_real_synth(
        real_div_with_relational, synthetic_flight_information)
    # Check diversity using PCA
    check_diversity_pca(scaled_real_flight_information, scaled_synthetic_flight_information, model_name,  plot_real_on_top)
    # Check diversity using t-SNE
    check_diversity_tsne(scaled_real_flight_information, scaled_synthetic_flight_information, model_name,  plot_real_on_top)
#===========================================================================
def diversity_correlations():    
    synth_types = ["gc", "tvae", "ctgan", "copgan"]
    real_div_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_with_relational.pkl')

    # Load synthetic (all features, do not exclude calculated features)
    for synth_type in synth_types:
        if synth_type == 'gc':
            # Default Gaussian Copula Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Default GC{color.END}")
            model_name = "GC"
            check_diversity(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)


        elif synth_type == 'tvae':
            # Default TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Default TVAE{color.END}")
            model_name = "Default TVAE"
            check_diversity(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

            # Optimal TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Optimal TVAE{color.END}")
            model_name = "Optimal TVAE"
            check_diversity(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

        elif synth_type == 'ctgan':
            # Default CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Default CTGAN{color.END}")
            model_name = "Default CTGAN"
            check_diversity(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

            # Optimal CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Optimal CTGAN{color.END}")
            model_name = "Optimal CTGAN"
            check_diversity(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

        elif synth_type == 'copgan':
            # Default CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Default CopulaGAN{color.END}")
            model_name = "Default CopulaGAN"
            check_diversity(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

            # Optimal CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Optimal CopulaGAN{color.END}")
            model_name = "Optimal CopulaGAN"
            check_diversity(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

#===========================================================================
# Diversity: class Balance
#===========================================================================
def class_balance_comparaison(df_results):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_results.copy()
    # Clean model names by removing "Optimal" and "Default"
    df_plot["Model"] = df_plot["Model"].apply(rename_model)
    df_plot['Model'] = df_plot['Model'].str.replace('real', 'Real', regex=False)
    
    fig, ax = plt.subplots(figsize=(6.1, 3.5))
    colors = ['#FF3300', '#0099FF']  
    
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
    ax.set_xlabel('Class balance (%)', fontsize=14)
#     ax.set_title('Class balance: real vs. synthetic data', fontsize=14)
    
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
        shift = 2.1  # small shift to move the label inside 
        ax.text(on_time + delayed/2 - shift, i, f'{delayed:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.1'))

    # Hatch the bars of optimised models (check original names)
    for p, model_name in zip(ax.patches, list(df_results['Model'].values) * 2):
        if 'Optimal' in model_name:
            p.set_hatch('//')
            p.set_edgecolor('black')
    
    legend_fontsize = 11
    frameon = False
    # Create custom legend entry for hatched bars
    hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
    handles, labels = ax.get_legend_handles_labels()
    # Create two separate legends
    # First legend for the main items (top row)
    legend1 = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                    ncol=2, frameon=frameon, fontsize=legend_fontsize)
    # Second legend for the hatched pattern (bottom row, centered)
    legend2 = ax.legend(handles=[hatch_patch], loc='upper center', bbox_to_anchor=(0.5, 0.96), 
                    ncol=1, frameon=frameon, fontsize=legend_fontsize)

    # Add the first legend back to the plot (matplotlib removes it when creating the second one)
    ax.add_artist(legend1)
    
    # Set x-axis limits and ticks
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Invert y-axis to have first model at top
#     ax.invert_yaxis()
    ax.set_ylim(-0.6, len(df_plot['Model']) + 0.85)  # add space above and below
    plt.tight_layout()
    plt.savefig(f'../data/outputs/paper_plots/diversity/class_balance_comparison_1000.png', dpi=300)
    plt.show()

#===========================================================================
# Operational
#===========================================================================
def plot_operational_correlation(real_data, synthetic_data, model_name, plot_real_on_top=False):
    """
    Plots the correlation between "Distance" and "Air Time" for both real vs synthetic data.
    Used as an indicator for the operational correctness of the synthetic data.
    """
    warnings.simplefilter("ignore", UserWarning)
    plt.rcdefaults()

    # Scatter plot for the correlation between "Distance" and "Air Time"
    plt.figure(figsize=(4, 3.5))

    if plot_real_on_top:
        plt.scatter(synthetic_data['Distance (miles)'],
                    synthetic_data['Air Time (min)'],
                    s=10,
                    color='red',
                    alpha=0.6,
                    label='Syn. Flight Diversions')
        plt.scatter(real_data['Distance (miles)'],
                    real_data['Air Time (min)'],
                    s=10,
                    color='blue',
                    alpha=0.6,
                    label='Real Flight Diversions')
    else:
        plt.scatter(real_data['Distance (miles)'],
                    real_data['Air Time (min)'],
                    s=10,
                    color='blue',
                    alpha=0.6,
                    label='Real Flight Diversions')
        plt.scatter(synthetic_data['Distance (miles)'],
                    synthetic_data['Air Time (min)'],
                    s=10,
                    color='red',
                    alpha=0.6,
                    label='Syn. Flight Diversions')
    plt.xlabel("Distance (miles)", fontsize=16)
    plt.ylabel("Air Time (min)", fontsize=16)
    plt.tick_params(axis='both', labelsize=11)  
#     plt.title('Operational correlation', fontsize=16)
#     plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    model = model_name.lower().replace(" ", "_")
    plt.savefig(f'../data/outputs/paper_plots/operational/operational_{model}_1000.png', dpi=300)
    plt.show()

def operational_correlations():    
    synth_types = ["gc", "tvae", "ctgan", "copgan"]
    real_div_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_with_relational.pkl')

    # Load synthetic (all features, do not exclude calculated features)
    for synth_type in synth_types:
        if synth_type == 'gc':
            # Default Gaussian Copula Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Default GC{color.END}")
            model_name = "GC"
            plot_operational_correlation(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)


        elif synth_type == 'tvae':
            # Default TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Default TVAE{color.END}")
            model_name = "Default TVAE"
            plot_operational_correlation(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

            # Optimal TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Optimal TVAE{color.END}")
            model_name = "Optimal TVAE"
            plot_operational_correlation(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

        elif synth_type == 'ctgan':
            # Default CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Default CTGAN{color.END}")
            model_name = "Default CTGAN"
            plot_operational_correlation(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

            # Optimal CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Optimal CTGAN{color.END}")
            model_name = "Optimal CTGAN"
            plot_operational_correlation(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

        elif synth_type == 'copgan':
            # Default CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Default CopulaGAN{color.END}")
            model_name = "Default CopulaGAN"
            plot_operational_correlation(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)

            # Optimal CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_1000.pkl')
            print(f"{color.BOLD}{color.GREEN}Optimal CopulaGAN{color.END}")
            model_name = "Optimal CopulaGAN"
            plot_operational_correlation(real_div_with_relational, synthetic_flight_information, model_name, plot_real_on_top=False)
#===========================================================================
# Statistical
#===========================================================================
def statistical_comparaison(df_results):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_results.copy()
    
    # Clean model names by removing "Optimal" and "Default"
    df_plot["Model"] = df_plot["Model"].apply(rename_model)
    
    colors = [["#0099FF", "#FF3300"], ["#339933"]] 
    columns = [['Marginal similarity', 'Bivariate similarity'], ['Overall similarity']]
    
    for i in range(len(columns)):
        ax = df_plot.plot(
            x="Model",
            y=columns[i],
            kind="bar",
            figsize=(5, 3.5),
            rot=45,
            color=colors[i] 
        )

        # Grid and limits
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Labels and title
        plt.ylabel("Statistical similarity (%)", fontsize=13)
#         plt.xlabel("Model", fontsize=14)
        plt.xlabel("")
#         plt.title("Statistical evaluation (higher is better)", fontsize=14)

        # Hatch the bars of optimised models (check original names)
        for p, model_name in zip(ax.patches, list(df_results['Model'].values) * len(columns[i])):
            if 'Optimal' in model_name:
                p.set_hatch('//')
                p.set_edgecolor('black')

        # Add values on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height:.1f}',
                xy=(p.get_x() + p.get_width() / 2., height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=90
            )

        legend_fontsize = 9
        frameon = False
        
        if i==0:
            # Create custom legend entry for hatched bars
            hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
            handles, labels = ax.get_legend_handles_labels()

            # Create two separate legends
            # First legend for the main items (top row)
            legend1 = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                            ncol=2, frameon=frameon, fontsize=legend_fontsize)

            # Second legend for the hatched pattern (bottom row, centered)
            legend2 = ax.legend(handles=[hatch_patch], loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                            ncol=1, frameon=frameon, fontsize=legend_fontsize)

            # Add the first legend back to the plot (matplotlib removes it when creating the second one)
            ax.add_artist(legend1)
        else:
            # Create custom legend entry for hatched bars
            hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(hatch_patch)
            ax.legend(handles=handles, loc="upper center", ncol=3, frameon=frameon, bbox_to_anchor=(0.5, 1.02), fontsize=legend_fontsize)

        plt.tight_layout()
        if i==0:
            ax.set_ylim(0, 135)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            plt.savefig(f'../data/outputs/paper_plots/statistical/statistical_1000.png', dpi=300)
        else:
            ax.set_ylim(0, 115)
            plt.savefig(f'../data/outputs/paper_plots/statistical/statistical_overall_1000.png', dpi=300)
        plt.show()
#===========================================================================
def plot_distributions():
    # Real 
    real_div_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_with_relational.pkl')

    # Default and optimal synthetic 
    synth_type = 'ctgan'; synth_sample_size = 1000;
    default_synthetic_div_with_relational = pd.read_pickle(f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
    optimal_synthetic_div_with_relational = pd.read_pickle(f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')

    states = ['default', 'optimal']
    dfs = [default_synthetic_div_with_relational, optimal_synthetic_div_with_relational]

    cont_numeric = ['Scheduled Elapsed Time (min)', 'Distance (miles)'] 

    for state, synth_df in zip(states, dfs):
        for feature in cont_numeric:
            plt.figure(figsize=(5, 3.5))

            sns.kdeplot(real_div_with_relational[feature], 
                        color='blue', label='Real', fill=True, alpha=0.5)
            sns.kdeplot(synth_df[feature], 
                        color='red', label='Synthetic', fill=True, alpha=0.5)

            plt.legend(fontsize=16, frameon=False)
            plt.xlabel(feature, fontsize=16)  
            plt.ylabel('Density', fontsize=16)  
            plt.tick_params(axis='both', labelsize=16)  

            plt.tight_layout()
            plt.savefig(f'../data/outputs/paper_plots/statistical/distribution_{synth_type}_{state}_{feature}_{synth_sample_size}.png', dpi=300)
            plt.show()

#===========================================================================
# Fidelity
#===========================================================================
def fidelity_comparaison(df_results):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_results.copy()
    
    # Clean model names by removing "Optimal" and "Default"
    df_plot["Model"] = df_plot["Model"].apply(rename_model)
    
    colors = [["#0099FF", "#FF3300"], ["#339933"]] 
    columns = [['F1', 'Balanced Accuracy'], ['Overall Fidelity']]
    
    for i in range(len(columns)):
        ax = df_plot.plot(
            x="Model",
            y=columns[i],
            kind="bar",
            figsize=(5, 3.5),
            rot=45,
            color=colors[i] 
        )

        # Grid and limits
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Labels and title
        plt.ylabel("Score", fontsize=14)
#         plt.xlabel("Model", fontsize=14)
        plt.xlabel("")
#         plt.title("Fidelity evaluation (lower is better)", fontsize=14)

        # Hatch the bars of optimised models (check original names)
        for p, model_name in zip(ax.patches, list(df_results['Model'].values) * len(columns[i])):
            if 'Optimal' in model_name:
                p.set_hatch('//')
                p.set_edgecolor('black')
                
        # Add values on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(p.get_x() + p.get_width() / 2., height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=90
            )

        legend_fontsize = 9
        frameon = False
        
        if i==0:
            # Create custom legend entry for hatched bars
            hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
            handles, labels = ax.get_legend_handles_labels()

            # Create two separate legends
            # First legend for the main items (top row)
            legend1 = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                            ncol=2, frameon=frameon, fontsize=legend_fontsize)

            # Second legend for the hatched pattern (bottom row, centered)
            legend2 = ax.legend(handles=[hatch_patch], loc='upper center', bbox_to_anchor=(0.5, 0.96), 
                            ncol=1, frameon=frameon, fontsize=legend_fontsize)

            # Add the first legend back to the plot (matplotlib removes it when creating the second one)
            ax.add_artist(legend1)
        else:
            # Create custom legend entry for hatched bars
            hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(hatch_patch)
            ax.legend(handles=handles, loc="upper center", ncol=3, frameon=frameon, bbox_to_anchor=(0.5, 1.02), fontsize=legend_fontsize)

        
        plt.tight_layout()
        if i==0:
            ax.set_ylim(0, 1.19)
            plt.savefig(f'../data/outputs/paper_plots/fidelity/fidelity_1000.png', dpi=300)
        else:
            ax.set_ylim(0, 1.15)
            plt.savefig(f'../data/outputs/paper_plots/fidelity/fidelity_overall_1000.png', dpi=300)
        plt.show()

#===========================================================================
# Utility
#===========================================================================
def utility_comparaison(df_results):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_results.copy()
    
    # Clean model names by removing "Optimal" and "Default"
    df_plot["Model"] = df_plot["Model"].apply(rename_model)

    colors = [["#0099FF", "#FF3300"], ["#339933"]] 

    # Rename columns to show TATR in legend
    df_plot = df_plot.rename(columns={
        "PR-AUC": "PR-AUC (TATR)",
        "Normalised MCC": "Normalised MCC (TATR)",
        "Overall Utility": "Overall Utility (TATR)"
    })
    columns = [['PR-AUC (TATR)', 'Normalised MCC (TATR)'], ['Overall Utility (TATR)']]
    
    for i in range(len(columns)):
        # Removing TRTR values from the bar plot
        ax = df_plot.iloc[1:].plot(
            x="Model",
            y=columns[i],
            kind="bar",
            figsize=(5, 3.5),  # -1 to remove TRTR
            rot=45,
            color=colors[i] 
        )
        # Grid and limits
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Adding horizontal lines for TRTR values
        if i==0:
            trtr_pr_auc = df_plot[df_plot["Model"]=="TRTR"]["PR-AUC (TATR)"].iloc[0]
            plt.axhline(y=trtr_pr_auc, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='PR-AUC (TRTR)')
            trtr_mcc = df_plot[df_plot["Model"]=="TRTR"]["Normalised MCC (TATR)"].iloc[0]
            plt.axhline(y=trtr_mcc, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Normalised MCC (TRTR)')
        else:
            trtr_overall_utility = df_plot[df_plot["Model"]=="TRTR"]["Overall Utility (TATR)"].iloc[0]
            plt.axhline(y=trtr_overall_utility, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Overall Utility (TRTR)')

        # Labels and title
#         ax.tick_params(axis='x', labelsize=14)  # X-axis tick labels
#         ax.tick_params(axis='y', labelsize=14)  # Y-axis tick labels
        plt.ylabel("Score", fontsize=14)
#         plt.xlabel("Model", fontsize=14)
        plt.xlabel("")
#         plt.title("Utility evaluation (higher is better)", fontsize=14)
        
        # Hatch the bars of optimised models (check original names)
        for p, model_name in zip(ax.patches, list(df_results.iloc[1:]['Model'].values) * len(columns[i])): # excluding TRTR
            if 'Optimal' in model_name:
                p.set_hatch('//')
                p.set_edgecolor('black')

        # Add values on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(p.get_x() + p.get_width() / 2., height),
                xytext=(0, 3),           
                textcoords='offset points',
                ha='center', 
                va='bottom',
                fontsize=10,
                rotation=90)

        # Set legend font size
        legend_fontsize = 9 
        frameon = False

        # Create custom legend with two rows
        hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
        handles, labels = ax.get_legend_handles_labels()

        # Create two separate legends
        # First legend for the main items (top row)
        legend1 = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                        ncol=2, frameon=frameon, fontsize=legend_fontsize)

        # Second legend for the hatched pattern (bottom row, centered)
        y = 0.84 if i==0 else 0.95
        legend2 = ax.legend(handles=[hatch_patch], loc='upper center', bbox_to_anchor=(0.5, y), 
                        ncol=1, frameon=frameon, fontsize=legend_fontsize)

        # Add the first legend back to the plot (matplotlib removes it when creating the second one)
        ax.add_artist(legend1)
        
        plt.tight_layout()
        if i==0:
            ax.set_ylim(0, 1.28)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
            plt.savefig(f'../data/outputs/paper_plots/utility/utility_1000.png', dpi=300)
        else:
            ax.set_ylim(0, 0.9)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6])
            plt.savefig(f'../data/outputs/paper_plots/utility/utility_overall_1000.png', dpi=300)
        plt.show()
#===========================================================================
def utility_augmentation_size():
    # Find all pkl files in the directory
    pkl_files = glob.glob(os.path.join('../data/outputs/eval_results/utility_augmentation_size/', '*.pkl'))

    # Read into a list of DataFrames
    list_dfs = [pd.read_pickle(f) for f in pkl_files]

    # Concatenate into one DataFrame
    df_aug_sizes = pd.concat(list_dfs, ignore_index=True)

    # Extract model order from first DataFrame
    model_order = list_dfs[0]["Model"].tolist()

    # Group by model and sort by model_order and "Size before cleaning"
    # Create a mapping dictionary for ordering
    order_mapping = {model: i for i, model in enumerate(model_order)}
    # Add a temporary sort column for model ordering
    df_aug_sizes['model_sort_order'] = df_aug_sizes['Model'].map(order_mapping)
    # Sort by model order first, then by "Size before cleaning"
    df_aug_sizes = df_aug_sizes.sort_values(by=["model_sort_order", "Size before cleaning"]).reset_index(drop=True)
    # Drop the temporary sort column
    df_aug_sizes = df_aug_sizes.drop('model_sort_order', axis=1)
    # Drop duplicate trtr rows, keeping the first occurrence
    df_aug_sizes = df_aug_sizes.drop_duplicates().reset_index(drop=True)

    # print(f"Loaded {len(pkl_files)} files")
    # print(f"Final dataframe shape: {df_aug_sizes.shape}")

    # ============================(For paper)===========================
    # List of models to remove
    remove_models = ["Default TVAE", "Default CTGAN", "Default CopulaGAN"]
    # Filter the DataFrame
    df_aug_sizes = df_aug_sizes[~df_aug_sizes["Model"].isin(remove_models)]
    df_aug_sizes["Model"] = df_aug_sizes["Model"].apply(rename_model)

    for i in ["before", "after"]:
        for metric in ["PR-AUC", "Normalised MCC"]:
            if metric == "Normalised MCC":
                plt.figure(figsize=(4.1,4.1))
            else:
                plt.figure(figsize=(4,4))

            # Add horizontal lines for TRTR values
            trtr_val  = df_aug_sizes[df_aug_sizes["Model"]=="TRTR"][metric].iloc[0]
            plt.axhline(y=trtr_val , color='black', linestyle='--', linewidth=1.5, alpha=1, label='$TRTR$')

            vertical_line = True
            if vertical_line:
                # Add vertical line when augmentation size equals the size of real data
                plt.axvline(x=60000 , color='blue', linestyle='--', linewidth=1.5, alpha=1, label='Point of class balance')

            # Get a consistent color palette for the models
            palette = sns.color_palette("tab10", n_colors=df_aug_sizes["Model"].nunique())
            # Map models (exclude TRTR) -> colors
            model_colors = dict(zip(df_aug_sizes.iloc[1:]["Model"].unique(), palette))

            # Plot PR-AUC & Normalised MCC vs Size before & after cleaning (light markers)
            sns.lineplot(
                data=df_aug_sizes.iloc[1:], # plot from second row and skip row with TRTR values
                x=f"Size {i} cleaning", 
                y=metric, 
                hue="Model", 
                marker="o", 
                linestyle="",   
                alpha=0.4,  
                legend=True,
                palette=model_colors
            )

            # Smoothed LOWESS trend per model
            for model in df_aug_sizes.iloc[1:]["Model"].unique():   # remove TRTR from the list of models
                sub = df_aug_sizes[df_aug_sizes["Model"] == model]
                x = sub[f"Size {i} cleaning"].values
                y = sub[metric].values
                # Apply LOWESS smoothing
                lowess = sm.nonparametric.lowess(y, x, frac=0.3) 
                plt.plot(lowess[:,0], lowess[:,1], color=model_colors[model], alpha=1, linewidth=2)

            # Grid and limits
            plt.grid(True, linestyle="--", alpha=0.7)
            y_max = max(df_aug_sizes[metric].max(), trtr_val)
            y_min = min(df_aug_sizes[metric].min(), trtr_val)
            margin = 0.05 * (y_max - y_min)                 # 5% of the range
            plt.ylim(y_min - margin, y_max + 7.15 * margin)  # Set y-limits slightly above max and below min
            # plt.xscale("log")

            # Labels and title
            plt.xlabel(f"Augmentation size {i} cleaning", fontsize=13)
            plt.ylabel(f"Utility ({metric})", fontsize=14)
            # plt.title(f"Utility ({metric}) vs augmentation size {i} cleaning", fontsize=15)


            if vertical_line:
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01),
                        ncol=2, frameon=True, edgecolor='white',framealpha=1, handlelength=1.5, columnspacing=0.8, fontsize= 9.1)
            else:
                # Get handles and labels from the current plot
                handles, labels = plt.gca().get_legend_handles_labels()

                # Create a custom arrangement: first 2 in column 1, remaining 3 in column 2
                # Pad the first column with empty entries to align properly
                custom_handles = []
                custom_labels = []

                # Add first 2 items
                custom_handles.extend(handles[:2])
                custom_labels.extend(labels[:2])

                # Add empty entry to pad first column (to have 3 items in second column)
                from matplotlib.patches import Rectangle
                empty_handle = Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False)
                custom_handles.append(empty_handle)
                custom_labels.append('')

                # Add remaining 3 items
                custom_handles.extend(handles[2:])
                custom_labels.extend(labels[2:])

                plt.legend(custom_handles, custom_labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                        ncol=2, frameon=True, edgecolor='white', framealpha=1, handlelength=1.3, 
                        columnspacing=4.6,borderpad=0.0, borderaxespad=0.5, fontsize=11.8)


            plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            plt.tick_params(axis='both', labelsize=12)  
            plt.tight_layout()
            plt.savefig(f'../data/outputs/paper_plots/utility/augmentation_{metric.lower().replace(" ", "_")}_{i}_cleaning.png', dpi=300)
            plt.show()
#===========================================================================
#===========================================================================

print(f"{color.BOLD}{color.RED}={color.END}"*75)
print(f"{color.BOLD}{color.GREEN}Realism{color.END}")
print(f"{color.BOLD}{color.RED}={color.END}"*75)
realism_comparaison(df_realism_results)

print(f"{color.BOLD}{color.RED}={color.END}"*75)
print(f"{color.BOLD}{color.GREEN}Diversity{color.END}")
print(f"{color.BOLD}{color.RED}={color.END}"*75)
diversity_correlations()
class_balance_comparaison(df_class_balance)

print(f"{color.BOLD}{color.RED}={color.END}"*75)
print(f"{color.BOLD}{color.GREEN}Operational{color.END}")
print(f"{color.BOLD}{color.RED}={color.END}"*75)
operational_correlations()

print(f"{color.BOLD}{color.RED}={color.END}"*75)
print(f"{color.BOLD}{color.GREEN}Statistical{color.END}")
print(f"{color.BOLD}{color.RED}={color.END}"*75)
statistical_comparaison(df_similarity_results)
plot_distributions()

print(f"{color.BOLD}{color.RED}={color.END}"*75)
print(f"{color.BOLD}{color.GREEN}Fidelity{color.END}")
print(f"{color.BOLD}{color.RED}={color.END}"*75)
fidelity_comparaison(df_fidelity_results)

print(f"{color.BOLD}{color.RED}={color.END}"*75)
print(f"{color.BOLD}{color.GREEN}Utility{color.END}")
print(f"{color.BOLD}{color.RED}={color.END}"*75)
utility_comparaison(df_utility_results)
utility_augmentation_size()
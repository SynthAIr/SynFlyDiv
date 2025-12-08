# Statistical evaluation conducted on (features used in the generation + relational features). 
# Same as the evaluation used in optuna optimization (here we add visualization of distributions).

# Import necessary libraries
import argparse
import sys
from utils import color
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality
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

def plot_discret_numeric_features(dis_numeric):
    # Distribution of discrete numeric features
    plt.rcdefaults()
    i = 0
    subplot_rows = 4
    subplots_col = 2

    fig, axs = plt.subplots(subplot_rows,
                            subplots_col,
                            figsize=(13, 12),
                            constrained_layout=True)
    axs = axs.flatten()

    for feature in dis_numeric:
        sns.histplot(real_div_with_relational[feature],
                     ax=axs[i],
                     bins=80,
                     color='blue',
                     stat="density",   # or 'probability' to normalize counts
                    #  discrete=True,
                     label="Real")
        sns.histplot(synthetic_flight_information[feature],
                     ax=axs[i],
                     bins=80,
                     color='red',
                     stat="density",   # or 'probability' to normalize counts
                    #  discrete=True,
                     label="Synthetic")
        axs[i].legend()
        i += 1

    plt.show()

#===========================================================================
def plot_continuous_numeric_features(cont_numeric):
    # Distribution of continuous numeric features with kdeplot
    plt.rcdefaults()
    i = 0
    subplot_rows = 4
    subplots_col = 2

    fig, axs = plt.subplots(subplot_rows,
                            subplots_col,
                            figsize=(13, 12),
                            constrained_layout=True)
    axs = axs.flatten()

    for feature in cont_numeric:
        sns.kdeplot(real_div_with_relational[feature],
                    ax=axs[i],
                    color='blue',
                    label='Real',
                    fill=True,
                    alpha=0.5,
                    bw_adjust=1)   # Smoothing for fair comparison between large and small datasets
        sns.kdeplot(synthetic_flight_information[feature],
                    ax=axs[i],
                    color='red',
                    label='Synthetic',
                    fill=True,
                    alpha=0.5,
                    bw_adjust=1)   # Smoothing for fair comparison between large and small datasets
        axs[i].legend()
        i += 1

    plt.show()

#===========================================================================
def evaluate_statistical_similarity(real_data, synthetic_data, metadata, verbose=False):
    # Columns defined in the metadata as city, state, or state_bbr are excluded
    quality_report = evaluate_quality(
        real_data, 
        synthetic_data, 
        metadata,
        verbose=verbose
    )
    similarity = quality_report.get_properties()['Score'].tolist()
    similarity.append(np.mean(similarity))

    if verbose:
        quality_report.get_details('Column Shapes')
        fig = quality_report.get_visualization(property_name='Column Shapes')
        fig.update_layout(width=900, height=1000)
        fig.show()
    
    return similarity  # [Marginal, Bivariate, Overall similarity]

#==========================================================================
# def plot_comparaison(df_results):
#     colors = [["#0099FF", "#FF3300"], ["#339933"]] 
#     columns = [['Marginal similarity', 'Bivariate similarity'], ['Overall similarity']]
    
#     for i in range(len(columns)):
#         ax = df_results.plot(
#             x="Model",
#             y=columns[i],
#             kind="bar",
#             figsize=(len(df_results)*len(columns[i])*.7, 6),
#             rot=45,
#             color=colors[i] 
#         )

#         # Grid and limits
#         ax.grid(True, axis='y', linestyle='--', alpha=0.7)
#         ax.set_ylim(0, 110)

#         # Labels and title
#         plt.ylabel("Statistical similarity (%)", fontsize=14)
#         plt.xlabel("Model", fontsize=14)
#         plt.title("Statistical evaluation (higher is better)", fontsize=14)

#         # Hatch the bars of optimised models
#         for p, model_name in zip(ax.patches, list(df_results['Model'].values) * len(columns[i])):
#             if 'Optimal' in model_name:
#                 p.set_hatch('//')
#                 p.set_edgecolor('black')

#         # Add values on top of bars
#         for p in ax.patches:
#             height = p.get_height()
#             ax.annotate(
#                 f'{height:.1f}',
#                 xy=(p.get_x() + p.get_width() / 2., height),
#                 xytext=(0, 3),
#                 textcoords='offset points',
#                 ha='center',
#                 va='bottom',
#                 fontsize=10,
#                 rotation=45
#             )

#         # Create custom legend entry for hatched bars
#         hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
#         handles, labels = ax.get_legend_handles_labels()
#         handles.append(hatch_patch)
#         ax.legend(handles=handles, loc="upper center", ncol=3)

#         plt.tight_layout()
#         plt.show()

def plot_comparaison(df_results):
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
            figsize=(len(df_plot)*len(columns[i])*.7, 6),
            rot=45,
            color=colors[i] 
        )

        # Grid and limits
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 110)

        # Labels and title
        plt.ylabel("Statistical similarity (%)", fontsize=14)
        plt.xlabel("Model", fontsize=14)
        plt.title("Statistical evaluation (higher is better)", fontsize=14)

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
                rotation=45
            )

        # Create custom legend entry for hatched bars
        hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(hatch_patch)
        ax.legend(handles=handles, loc="upper center", ncol=3)

        plt.tight_layout()
        plt.show()
#===========================================================================
def realism_evaluation():
    before_cleaning = synth_sample_size
    after_cleaning = synthetic_flight_information.shape[0]
    realism = round((after_cleaning / before_cleaning) * 100, 1)

    return before_cleaning, after_cleaning, realism

#===========================================================================
def main():
    
    dis_numeric = [
        'Quarter', 'Day of Week', 'Origin Airport ID', 'Destination Airport ID',
        'Departure Delay Label', 'Arrival Delay Label', 'Diversion Label'
    ]
    cont_numeric = [
        'Departure ΔT (min)', 'Arrival ΔT (min)', 'Taxi Out Time (min)',
        'Taxi In Time (min)', 'Scheduled Elapsed Time (min)',
        'Actual Elapsed Time (min)', 'Air Time (min)', 'Distance (miles)'
    ]
    # Visual statistical similarity (distribution plots)
    print(f"{color.BOLD}Visual statistical similarity:{color.END}")
    plot_discret_numeric_features(dis_numeric)
    plot_continuous_numeric_features(cont_numeric)

    # Numerical evaluation of statistical similarity
    print(f"{color.BOLD}Numerical statistical similarity:{color.END}")
    similarity = evaluate_statistical_similarity(real_div_with_relational, synthetic_flight_information, 
                                                 metadata_with_relational, verbose=False)
    print(f"   Marginal similarity: {similarity[0]*100:.2f}%")
    print(f"   Bivariate similarity: {similarity[1]*100:.2f}%")
    print(f"   {color.BOLD}{color.BLUE}Overall similarity: {similarity[2]*100:.2f}%{color.END}")

    # Store results
    results.append({
        "Model": model_name,
        "Marginal similarity": similarity[0]*100,
        "Bivariate similarity": similarity[1]*100,
        "Overall similarity": similarity[2]*100
    })
#===========================================================================
if __name__ == "__main__":
    # List of available synthetic sample sizes
    synth_sizes = [300, 500, 1000, 2000, 5000, 10000, 20000]
    # List of available synthesizers
    synthesizers = ["gc", "ctgan", "tvae", "copgan"]
    # Only diverted flights (with relational features for statistical & diversity evaluation)
    real_div_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_with_relational.pkl')
    # Metadata (with relational features for statistical, diversity & utility evaluations)
    metadata_with_relational = Metadata.load_from_json('../data/preprocessed_data/metadata_with_relational.json')

    if len(sys.argv) > 1 and not any(arg.startswith('--') for arg in sys.argv[1:]):
        # Case 1: run via Jupyter (e.g. %run statistical_eval.py 300 gc tvae)
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
        # Case 2: execute the script with command line arguments (e.g. python statistical_eval.py --synth_sample_size 300 --synth_types gc tvae)
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
            print(f"{color.BOLD}{color.GREEN}Statistical evaluation:{color.END} Default GC ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "GC"
            main()
        
        elif synth_type == 'tvae':
            # Default TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Statistical evaluation:{color.END} Default TVAE ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default TVAE"
            main()

            # Optimal TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Statistical evaluation:{color.END} Optimal TVAE ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal TVAE"
            main()
        
        elif synth_type == 'ctgan':
            # Default CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Statistical evaluation:{color.END} Default CTGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default CTGAN"
            main()

            # Optimal CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Statistical evaluation:{color.END} Optimal CTGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal CTGAN"
            main()

        elif synth_type == 'copgan':
            # Default CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Statistical evaluation:{color.END} Default CopulaGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default CopulaGAN"
            main()

            # Optimal CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Statistical evaluation:{color.END} Optimal CopulaGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal CopulaGAN"
            main()

    # Visual comparison of discriminative scores across all models
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    print(f"{color.BOLD}{color.GREEN}Visual comparison of realism scores across all models:{color.END}")
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    df_similarity_results = pd.DataFrame(results)
    plot_comparaison(df_similarity_results)

    # Save statistical evaluation results
    df_similarity_results.to_pickle(f'../data/outputs/eval_results/df_similarity_results_{synth_sample_size}.pkl')
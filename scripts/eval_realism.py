# Realism evaluation measurimg the percentage of valid flight routes that remain after cleaning the synthetic data. 
# Same as the evaluation used in optuna optimization.

# Import necessary libraries
import argparse
import sys
from utils import color
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

def realism_evaluation():
    before_cleaning = synth_sample_size
    after_cleaning = synthetic_flight_information.shape[0]
    realism = round((after_cleaning / before_cleaning) * 100, 1)

    print(f"Number of synthetic flights before cleaning = {before_cleaning}")
    print(f"Number of synthetic flights after cleaning = {after_cleaning}") 
    print(f"{color.BOLD}{color.BLUE}Realism score = {realism}%{color.END}")

    # Store results
    results.append({
        "Model": model_name,
        "Realism": realism
    })

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
# def plot_comparaison(df_results):
#     colors = [["#0099FF"]] 
#     columns = [['Realism']]

#     for i in range(len(columns)):
#         ax = df_results.plot(
#             x="Model",
#             y=columns[i],
#             kind="bar",
#             figsize=(len(df_results)*len(columns[i])*.7, 6),
#             rot=45,
#             color=colors[i] 
#         )

#     # Grid and limits
#     ax.grid(True, axis='y', linestyle='--', alpha=0.7)
#     ax.set_ylim(0, 102)

#     # Labels and title
#     plt.ylabel("Valid records (%)", fontsize=14)
#     plt.xlabel("Model", fontsize=14)
#     plt.title("Realism evaluation (higher is better)", fontsize=14)

#     # Hatch the bars of optimised models
#     for p, model_name in zip(ax.patches, list(df_results['Model'].values) * len(columns[i])):
#         if 'Optimal' in model_name:
#             p.set_hatch('//')
#             p.set_edgecolor('black')

#     # Add values on top of bars
#     for p in ax.patches:
#         height = p.get_height()
#         ax.annotate(
#             f'{height:.1f}',
#             xy=(p.get_x() + p.get_width() / 2., height),
#             xytext=(0, 3),          
#             textcoords='offset points',
#             ha='center', 
#             va='bottom',
#             fontsize=10,
#             rotation=45)

#         # Create custom legend entry for hatched bars
#         hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
#         handles, labels = ax.get_legend_handles_labels()
#         handles.append(hatch_patch)
#         ax.legend(handles=handles, loc="upper center", ncol=3)

#     plt.tight_layout()
#     plt.show()

def plot_comparaison(df_results):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_results.copy()
    # # Clean model names by removing "Optimal" and "Default"
    df_plot["Model"] = df_plot["Model"].apply(rename_model)

    
    colors = [["#0099FF"]] 
    columns = [['Realism']]

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
    ax.set_ylim(0, 102)

    # Labels and title
    plt.ylabel("Valid records (%)", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.title("Realism evaluation (higher is better)", fontsize=14)

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
            rotation=45)

    # Create custom legend entry for hatched bars
    hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(hatch_patch)
    ax.legend(handles=handles, loc="upper center", ncol=3)

    plt.tight_layout()
    plt.show()
    
#===========================================================================
def main():
    realism_evaluation()

#===========================================================================
if __name__ == "__main__":
    # List of available synthetic sample sizes
    synth_sizes = [300, 500, 1000, 2000, 5000, 10000, 20000]
    # List of available synthesizers
    synthesizers = ["gc", "ctgan", "tvae", "copgan"]

    if len(sys.argv) > 1 and not any(arg.startswith('--') for arg in sys.argv[1:]):
        # Case 1: run via Jupyter (e.g. %run realism_eval.py 300 gc tvae)
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
        # Case 2: execute the script with command line arguments (e.g. python realism_eval.py --synth_sample_size 300 --synth_types gc tvae)
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
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Realism evaluation:{color.END} Default GC")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "GC"
            main()
        
        elif synth_type == 'tvae':
            # Default TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Realism evaluation:{color.END} Default TVAE")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default TVAE"
            main()

            # Optimal TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Realism evaluation:{color.END} Optimal TVAE")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal TVAE"
            main()
        
        elif synth_type == 'ctgan':
            # Default CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Realism evaluation:{color.END} Default CTGAN")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default CTGAN"
            main()

            # Optimal CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Realism evaluation:{color.END} Optimal CTGAN")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal CTGAN"
            main()

        elif synth_type == 'copgan':
            # Default CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Realism evaluation:{color.END} Default CopulaGAN")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default CopulaGAN"
            main()

            # Optimal CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Realism evaluation:{color.END} Optimal CopulaGAN")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal CopulaGAN"
            main()

    # Visual comparison of discriminative scores across all models
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    print(f"{color.BOLD}{color.GREEN}Visual comparison of realism scores across all models:{color.END}")
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    df_realism_results = pd.DataFrame(results)
    plot_comparaison(df_realism_results)

    # Save realism evaluation results
    df_realism_results.to_pickle(f'../data/outputs/eval_results/df_realism_results_{synth_sample_size}.pkl')
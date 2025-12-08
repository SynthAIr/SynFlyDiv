# Operational validity evaluation looking at correlations between specific features.
# This evaluation is not used in optuna optimization. 

# Import necessary libraries
import argparse
import sys
from utils import color
import pandas as pd
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import evaluate_quality
import warnings
warnings.filterwarnings('ignore')

def plot_correlation(real_data, synthetic_data, plot_real_on_top=False):
    """
    Plots the correlation between "Distance" and "Air Time" for both real vs synthetic data.
    Used as an indicator for the operational correctness of the synthetic data.
    """
    warnings.simplefilter("ignore", UserWarning)
    plt.rcdefaults()

    # Scatter plot for the correlation between "Distance" and "Air Time"
    plt.figure(figsize=(5, 5))

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
    plt.xlabel("Distance (miles)", fontsize=14)
    plt.ylabel("Air Time (min)", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)  
    plt.title('Operational correlation', fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    # model = model_name.lower().replace(" ", "_")
    # plt.savefig(f'../data/outputs/results/operational_{model}_{synth_sample_size}.png', dpi=300)
    plt.show()

#===========================================================================
def realism_evaluation():
    before_cleaning = synth_sample_size
    after_cleaning = synthetic_flight_information.shape[0]
    realism = round((after_cleaning / before_cleaning) * 100, 1)

    return before_cleaning, after_cleaning, realism

#===========================================================================
def main():
    
    # Plot correlation between "Distance" and "Air Time"
    print(f"{color.BOLD}Correlation between 'Distance' and 'Air Time' as indicator of the operational correctness of the synthetic data:{color.END}")
    plot_correlation(real_div_with_relational, synthetic_flight_information, plot_real_on_top=False)

#===========================================================================
if __name__ == "__main__":
    # List of available synthetic sample sizes
    synth_sizes = [300, 500, 1000, 2000, 5000, 10000, 20000]
    # List of available synthesizers
    synthesizers = ["gc", "ctgan", "tvae", "copgan"]
    # Only diverted flights (with relational features for statistical & diversity evaluation)
    real_div_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_with_relational.pkl')

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
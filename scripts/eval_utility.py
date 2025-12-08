# Utility evaluation conducted on (features used in the generation + relational features).
# Same as the evaluation used in optuna optimization.

# Import necessary libraries
import argparse
import sys
from utils import color, prepare_for_prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score, matthews_corrcoef
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

def evaluate_ml_utility(real_data, synthetic_data, model_name, target_column):
    """
    Conducted on the features used in the generation + relational features.
    Evaluate how well synthetic data preserves ML utility.
    """
    # In case of no valid routes after cleaning
    if synthetic_data.shape[0] == 0:
        raise ValueError("No valid synthetic routes remaining after cleaning. Cannot evaluate ML utility.")

    # Prepare for prediction by:
    # - Encoding categorical and datetime features
    # - Remove features that are not needed when predicting the target_column
    # - Work on copies inside this function to avoid messing the global DataFrames
    real, synthetic_diversions = prepare_for_prediction(real_data, synthetic_data, target_column)

    # Features and labels
    real_X = real.drop(columns=[target_column])
    real_Y = real[target_column]
    synthetic_X = synthetic_diversions.drop(columns=[target_column])
    synthetic_Y = synthetic_diversions[target_column]

    # Split with stratification (only real data)
    real_X_train, real_X_test, real_Y_train, real_Y_test = train_test_split(
        real_X,
        real_Y,
        test_size=0.3,
        stratify=real_Y,
        random_state=seed_val
        )

    # Augmentation (adding synthetic diversions to the real training data)
    augmented_X_train = pd.concat([real_X_train, synthetic_X], ignore_index=True)
    augmented_Y_train = pd.concat([real_Y_train, synthetic_Y], ignore_index=True)
    if model_name == "trtr":
        # ========== TRTR (Train on Real, Test on Real) ==========
        # Create pipeline with SMOTE and classifier
        classifier = RandomForestClassifier(
            class_weight="balanced",       # Important for imbalance
            n_estimators=100,
            random_state=seed_val
            )
        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=seed_val)), 
            ('model', classifier)
            ])
        pipe.fit(real_X_train, real_Y_train)
        Y_pred = pipe.predict(real_X_test)
        Y_pred_prob = pipe.predict_proba(real_X_test)  # Get probabilities instead of predictions (for PR AUC)

        # Precision-Recall AUC (PR AUC) for probability of positive class (diversion)
        pr_auc_trtr = average_precision_score(real_Y_test, Y_pred_prob[:, 1]) 

        # Matthews Correlation Coefficient (MCC)
        mcc_trtr = matthews_corrcoef(real_Y_test, Y_pred)
        mcc_trtr = (mcc_trtr + 1) / 2 # Normalize from [-1,1] to [0,1] for optimization consistency

        overall_utility_trtr = np.mean([pr_auc_trtr, mcc_trtr])

        # Numerical evaluation of fidelity
        print(f"{color.BOLD}Utility scores (higher is better):{color.END}")
        print(f"   PR-AUC (TRTR): {pr_auc_trtr:.3f}")
        print(f"   Normalised MCC (TRTR): {mcc_trtr:.3f}")
        print(f"   {color.BOLD}{color.BLUE}Overall utility (TRTR): {overall_utility_trtr*100:.2f}%{color.END}")

        # Store results
        results.append({
            "Model": model_name.upper(),
            "Size before cleaning": None,
            "Size after cleaning": None,
            "Realism (%)": None,
            "PR-AUC": pr_auc_trtr,
            "Normalised MCC": mcc_trtr,
            "Overall Utility": overall_utility_trtr
        })
    else:
        # ======== TATR (Train on Augmented, Test on Real) =======
        # TATR (Train on Augmented, Test on Real)
        # Create pipeline with SMOTE and classifier
        classifier = RandomForestClassifier(
            class_weight="balanced",       # Important for imbalance
            n_estimators=100,
            random_state=seed_val
            )
        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=seed_val)), 
            ('model', classifier)
            ])
        pipe.fit(augmented_X_train, augmented_Y_train)
        Y_pred = pipe.predict(real_X_test)
        Y_pred_prob = pipe.predict_proba(real_X_test)  # Get probabilities instead of predictions (for PR AUC)

        # Precision-Recall AUC (PR AUC) for probability of positive class (diversion)
        pr_auc_tatr = average_precision_score(real_Y_test, Y_pred_prob[:, 1]) 
            
        # Matthews Correlation Coefficient (MCC)
        mcc_tatr = matthews_corrcoef(real_Y_test, Y_pred)
        mcc_tatr = (mcc_tatr + 1) / 2 # Normalize from [-1,1] to [0,1] for optimization consistency

        overall_utility_tatr = np.mean([pr_auc_tatr, mcc_tatr])

        # Numerical evaluation of fidelity
        print(f"{color.BOLD}Utility scores (higher is better):{color.END}")
        print(f"   PR-AUC (TATR): {pr_auc_tatr:.3f}")
        print(f"   Normalised MCC (TATR): {mcc_tatr:.3f}")
        print(f"   {color.BOLD}{color.BLUE}Overall utility (TATR): {overall_utility_tatr*100:.2f}%{color.END}")

        # Store results
        results.append({
            "Model": model_name,
            "Size before cleaning": before_cleaning,
            "Size after cleaning": after_cleaning,
            "Realism (%)": realism,
            "PR-AUC": pr_auc_tatr,
            "Normalised MCC": mcc_tatr,
            "Overall Utility": overall_utility_tatr
        })

#===========================================================================
# def plot_comparaison(df_results):
#     # Choose legend style: 1 or 2
#     style = 2
#     frameon = False
#     colors = [["#0099FF", "#FF3300"], ["#339933"]] 

#     # Rename columns to show TATR in legend
#     df_results = df_results.rename(columns={
#         "PR-AUC": "PR-AUC (TATR)",
#         "Normalised MCC": "Normalised MCC (TATR)",
#         "Overall Utility": "Overall Utility (TATR)"
#     })
#     columns = [['PR-AUC (TATR)', 'Normalised MCC (TATR)'], ['Overall Utility (TATR)']]
    
#     for i in range(len(columns)):
#         # Removing TRTR values from the bar plot
#         ax = df_results.iloc[1:].plot(
#             x="Model",
#             y=columns[i],
#             kind="bar",
#             figsize=((len(df_results)-1)*len(columns[i])*.7, 6),  # -1 to remove TRTR
#             rot=45,
#             color=colors[i] 
#         )
#         # Grid and limits
#         ax.grid(True, axis='y', linestyle='--', alpha=0.7)
#         ax.set_ylim(0, 1.0)

#         # Adding horizontal lines for TRTR values
#         if i==0:
#             trtr_pr_auc = df_results[df_results["Model"]=="TRTR"]["PR-AUC (TATR)"].iloc[0]
#             plt.axhline(y=trtr_pr_auc, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='PR-AUC (TRTR)')
#             trtr_mcc = df_results[df_results["Model"]=="TRTR"]["Normalised MCC (TATR)"].iloc[0]
#             plt.axhline(y=trtr_mcc, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Normalised MCC (TRTR)')
#         else:
#             trtr_overall_utility = df_results[df_results["Model"]=="TRTR"]["Overall Utility (TATR)"].iloc[0]
#             plt.axhline(y=trtr_overall_utility, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Overall Utility (TRTR)')

#         # Labels and title
#         plt.ylabel("Score", fontsize=14)
#         plt.xlabel("Model", fontsize=14)
#         plt.title("Utility evaluation (higher is better)", fontsize=14)
        
#         # Hatch the bars of optimised models
#         for p, model_name in zip(ax.patches, list(df_results.iloc[1:]['Model'].values) * len(columns[i])): # excluding TRTR
#             if 'Optimal' in model_name:
#                 p.set_hatch('//')
#                 p.set_edgecolor('black')

#         # Add values on top of bars
#         for p in ax.patches:
#             height = p.get_height()
#             ax.annotate(
#                 f'{height:.2f}',
#                 xy=(p.get_x() + p.get_width() / 2., height),
#                 xytext=(0, 3),           
#                 textcoords='offset points',
#                 ha='center', 
#                 va='bottom',
#                 fontsize=10,
#                 rotation=45)

#         # Set legend font size based on plot
#         legend_fontsize = 11 if i == 0 else 9  # smaller font for second plot

#         if style == 1:
#             # Create custom legend with two rows
#             hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
#             handles, labels = ax.get_legend_handles_labels()
            
#             # Separate handles into bar colors and dashed lines
#             bar_handles = []
#             line_handles = []
#             for handle, label in zip(handles, labels):
#                 if 'TRTR' in label:
#                     line_handles.append(handle)
#                 else:
#                     bar_handles.append(handle)
            
#             # Add the hatch patch to bar handles
#             bar_handles.append(hatch_patch)
            
#             # Create two separate legends
#             # First legend for the bar colors and hatched pattern (top row)
#             if i == 0:
#                 # For the first plot, we have bar colors
#                 legend1 = ax.legend(handles=bar_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
#                                 ncol=len(bar_handles), frameon=frameon, fontsize=legend_fontsize)
#             else:
#                 # For the second plot, we only have the bar and hatch
#                 legend1 = ax.legend(handles=bar_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
#                                 ncol=len(bar_handles), frameon=frameon, fontsize=legend_fontsize)
            
#             # Second legend for the dashed lines (bottom row, centered)
#             if line_handles:
#                 legend2 = ax.legend(handles=line_handles, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
#                                 ncol=len(line_handles), frameon=frameon, fontsize=legend_fontsize)
#                 # Add the first legend back to the plot
#                 ax.add_artist(legend1)
#         elif style == 2:
#             # Create custom legend with two rows
#             hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
#             handles, labels = ax.get_legend_handles_labels()
            
#             # Create two separate legends
#             # First legend for the main items (top row)
#             legend1 = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
#                             ncol=len(handles), frameon=frameon, fontsize=legend_fontsize)
            
#             # Second legend for the hatched pattern (bottom row, centered)
#             legend2 = ax.legend(handles=[hatch_patch], loc='upper center', bbox_to_anchor=(0.5, 0.94), 
#                             ncol=1, frameon=frameon, fontsize=legend_fontsize)
            
#             # Add the first legend back to the plot (matplotlib removes it when creating the second one)
#             ax.add_artist(legend1)
        
#         plt.tight_layout()
#         plt.show()

def plot_comparaison(df_results):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_results.copy()
    
    # Clean model names by removing "Optimal" and "Default"
    df_plot["Model"] = df_plot["Model"].apply(rename_model)


    # Choose legend style: 1 or 2
    style = 2
    frameon = False
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
            figsize=((len(df_plot)-1)*len(columns[i])*.7, 6),  # -1 to remove TRTR
            rot=45,
            color=colors[i] 
        )
        # Grid and limits
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.0)

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
        plt.ylabel("Score", fontsize=14)
        plt.xlabel("Model", fontsize=14)
        plt.title("Utility evaluation (higher is better)", fontsize=14)
        
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
                rotation=45)

        # Set legend font size based on plot
        legend_fontsize = 11 if i == 0 else 9  # smaller font for second plot

        if style == 1:
            # Create custom legend with two rows
            hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
            handles, labels = ax.get_legend_handles_labels()
            
            # Separate handles into bar colors and dashed lines
            bar_handles = []
            line_handles = []
            for handle, label in zip(handles, labels):
                if 'TRTR' in label:
                    line_handles.append(handle)
                else:
                    bar_handles.append(handle)
            
            # Add the hatch patch to bar handles
            bar_handles.append(hatch_patch)
            
            # Create two separate legends
            # First legend for the bar colors and hatched pattern (top row)
            if i == 0:
                # For the first plot, we have bar colors
                legend1 = ax.legend(handles=bar_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
                                ncol=len(bar_handles), frameon=frameon, fontsize=legend_fontsize)
            else:
                # For the second plot, we only have the bar and hatch
                legend1 = ax.legend(handles=bar_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
                                ncol=len(bar_handles), frameon=frameon, fontsize=legend_fontsize)
            
            # Second legend for the dashed lines (bottom row, centered)
            if line_handles:
                legend2 = ax.legend(handles=line_handles, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                                ncol=len(line_handles), frameon=frameon, fontsize=legend_fontsize)
                # Add the first legend back to the plot
                ax.add_artist(legend1)
        elif style == 2:
            # Create custom legend with two rows
            hatch_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Optimised Models')
            handles, labels = ax.get_legend_handles_labels()
            
            # Create two separate legends
            # First legend for the main items (top row)
            legend1 = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
                            ncol=len(handles), frameon=frameon, fontsize=legend_fontsize)
            
            # Second legend for the hatched pattern (bottom row, centered)
            legend2 = ax.legend(handles=[hatch_patch], loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                            ncol=1, frameon=frameon, fontsize=legend_fontsize)
            
            # Add the first legend back to the plot (matplotlib removes it when creating the second one)
            ax.add_artist(legend1)
        
        plt.tight_layout()
        plt.show()
#===========================================================================
def realism_evaluation():
    before_cleaning = synth_sample_size
    after_cleaning = synthetic_flight_information.shape[0]
    realism = round((after_cleaning / before_cleaning) * 100, 1)

    return before_cleaning, after_cleaning, realism

#===========================================================================
if __name__ == "__main__":
    # List of available synthetic sample sizes
    synth_sizes = [300, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 
                   30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 
                   90000, 100000, 110000, 120000, 130000, 140000, 160000, 180000, 200000]
    # List of available synthesizers
    synthesizers = ["gc", "ctgan", "tvae", "copgan"]
    # Fixing randomness for reproducibility
    seed_val = 45
    # Real diverted and not diverted flights (with relational features for utility evaluation)
    real_div_notdiv_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_notdiv_with_relational.pkl')
    target_column='Diversion Label'

    if len(sys.argv) > 1 and not any(arg.startswith('--') for arg in sys.argv[1:]):
        # Case 1: run via Jupyter (e.g. %run utility_eval.py 300 gc tvae)
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
        # Case 2: execute the script with command line arguments (e.g. python utility_eval.py --synth_sample_size 300 --synth_types gc tvae)
        parser = argparse.ArgumentParser()
        parser.add_argument('--synth_sample_size', type=int, required=True, choices=synth_sizes)
        parser.add_argument('--synth_types', type=str, nargs='+', required=True, choices=synthesizers)
        args = parser.parse_args()
        synth_sample_size = args.synth_sample_size
        synth_types = args.synth_types

    results = []

    # TRTR  
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    print(f"{color.BOLD}{color.GREEN}Utility evaluation: TRTR (Train on Real, Test on Real){color.END}")
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    model_name = "trtr"
    synthetic_flight_information = pd.read_pickle(
        f'../data/outputs/synthetic/synthetic_GC_diversions_default_{synth_sample_size}.pkl')
    evaluate_ml_utility(real_div_notdiv_with_relational, synthetic_flight_information, model_name, target_column)


    # TATR (Train on Augmented, Test on Real)
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    print(f"{color.BOLD}{color.GREEN}Utility evaluation: TATR (Train on Augmented, Test on Real){color.END}")
    # Load synthetic (all features, do not exclude calculated features)
    for synth_type in synth_types:
        if synth_type == 'gc':
            # Default Gaussian Copula Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Utility evaluation:{color.END} Default GC ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "GC"
            evaluate_ml_utility(real_div_notdiv_with_relational, synthetic_flight_information, model_name, target_column)
        
        elif synth_type == 'tvae':
            # Default TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Utility evaluation:{color.END} Default TVAE ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default TVAE"
            evaluate_ml_utility(real_div_notdiv_with_relational, synthetic_flight_information, model_name, target_column)

            # Optimal TVAE Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Utility evaluation:{color.END} Optimal TVAE ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal TVAE"
            evaluate_ml_utility(real_div_notdiv_with_relational, synthetic_flight_information, model_name, target_column)
       
        elif synth_type == 'ctgan':
            # Default CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Utility evaluation:{color.END} Default CTGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default CTGAN"
            evaluate_ml_utility(real_div_notdiv_with_relational, synthetic_flight_information, model_name, target_column)

            # Optimal CTGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Utility evaluation:{color.END} Optimal CTGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal CTGAN"
            evaluate_ml_utility(real_div_notdiv_with_relational, synthetic_flight_information, model_name, target_column)

        elif synth_type == 'copgan':
            # Default CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_default_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Utility evaluation:{color.END} Default CopulaGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Default CopulaGAN"
            evaluate_ml_utility(real_div_notdiv_with_relational, synthetic_flight_information, model_name, target_column)

            # Optimal CopulaGAN Synthesizer
            synthetic_flight_information = pd.read_pickle(
                f'../data/outputs/synthetic/synthetic_{synth_type.upper()}_diversions_optimal_{synth_sample_size}.pkl')
            before_cleaning, after_cleaning, realism = realism_evaluation()
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            print(f"{color.BOLD}{color.GREEN}Utility evaluation:{color.END} Optimal CopulaGAN ({before_cleaning} --> {after_cleaning} [realism {realism}%])")
            print(f"{color.BOLD}{color.RED}={color.END}"*75)
            model_name = "Optimal CopulaGAN"
            evaluate_ml_utility(real_div_notdiv_with_relational, synthetic_flight_information, model_name, target_column)

    # Visual comparison of utility scores across all models
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    print(f"{color.BOLD}{color.GREEN}Visual comparison of utility scores across all models:{color.END}")
    print(f"{color.BOLD}{color.RED}={color.END}"*75)
    df_utility_results = pd.DataFrame(results)
    plot_comparaison(df_utility_results)

    # Save utility evaluation results
    df_utility_results.to_pickle(f'../data/outputs/eval_results/utility_augmentation_size/df_utility_results_{synth_sample_size}.pkl')
# This script load Optuna study and print/plot optimization results.
# Import necessary libraries
import optuna
import argparse
import sys
import plotly.io as pio
pio.renderers.default = "notebook"
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def prints(study):
    print("-"*75, f"\nTotal number of trials: {len(study.trials)} (best traial: {study.best_trial.number})")
    print("-"*75, f"\nBest trial details:\n{study.best_trial}")

    print("-"*75, f"\nBest parameters:\n{study.best_params}")
    print(f"\nParameter importance:\n{optuna.importance.get_param_importances(study)}")

    print("-"*75, f"\nBest score: {study.best_value:.4f}")
    print(f"\nScore details:\n{study.best_trial.user_attrs}")

    print("-"*75, "\nAll Trials:")
    for t in study.trials:
        print(f"Trial {t.number}: Value = {t.value}, Params = {t.params}")
    print("-"*75)

#=======================================================================
def plots(study):
    # Plot optimization history
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.show()
    # fig1.write_html("optimization_history_plot.html")
    # fig1.write_image("optimization_history_plot.png")

    # Plot parameter importance
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.show()
    # fig2.write_html("param_importances_plot.html")

    # Plot parameter relationships
    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    fig3.show()
    # fig3.write_html("parallel_coordinate_plot.html")

    # How each hyperparameter relates to objective value
    fig4 = optuna.visualization.plot_slice(study)
    fig4.show()
    # fig4.write_html("slice_plot.html")

    #Contour plots of parameter spaces vs. objective
    fig5 = optuna.visualization.plot_contour(study)
    fig5.show()
    # fig5.write_html("contour_plot.html")

    # # Displays intermediate results (only if your objective reports intermediate scores via trial.report()).
    # fig6 = optuna.visualization.plot_intermediate_values(study)
    # fig6.show()
    # # fig6.write_html("intermediate_values_plot.html")

    # # Compares distributions of objective values between multiple studies.
    # fig7 = optuna.visualization.plot_edf(study)
    # fig7.show()
    # # fig7.write_html("edf_plot.html")
#=======================================================================

def main():
    # Load study
    study = optuna.load_study(
        study_name=f"{synth_type}_hyperopt",
        storage=f"sqlite:///../data/outputs/optimization_studies/optuna_study_{synth_type}.db"
    )

    # Print study details
    prints(study)
    # Plot study results
    plots(study)

# execute the script with command line arguments or Jupyter notebook
if __name__ == "__main__":
    # List of available synthesizers
    synthesizers = ["ctgan", "tvae", "copgan"]

    if len(sys.argv) > 1 and not any(arg.startswith('--') for arg in sys.argv[1:]):
        # Case 1: run via Jupyter (e.g. %run optimizer_results.py tvae)
        synth_type = sys.argv[1].lower()

    else:
        # Case 2: execute the script with command line arguments (e.g. python optimizer_results.py --synth_type tvae)
        parser = argparse.ArgumentParser()
        parser.add_argument('--synth_type', type=str, required=True, choices=synthesizers)
        args = parser.parse_args()
        synth_type = args.synth_type
        
    # Validate the synth_type
    if synth_type not in synthesizers:
        raise ValueError(f"Invalid synthesizer type: {synth_type}. Choose from {synthesizers}")
    
    main()


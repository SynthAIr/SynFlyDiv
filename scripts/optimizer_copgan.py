# Import necessary libraries
import optuna
import numpy as np
import pandas as pd
from utils import color, set_seed, add_relatioanal_features_clean_inv_routes, prepare_for_prediction, simplify
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score, matthews_corrcoef, make_scorer, balanced_accuracy_score

import warnings
warnings.filterwarnings('ignore')

import plotly.io as pio
pio.renderers.default = "notebook"


# Configure Optuna logging
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce noise

class COPGANBayesianOptimizer:
    """
    Advanced Bayesian Optimization for COPGAN hyperparameters using Optuna.
    Finds optimal hyperparameters, trains and saves the best synthesizer.
    """
    
    def __init__(self, real_data, metadata, target_column=None, seed_val=45):
        self.real_div_no_relational = real_data    # Only diverted flights (without relational features to train the synthesizer & evaluate fidelity)
        self.metadata_no_relational = metadata      # Metadata (without relational features to train the synthesizer & evaluate fidelity)
        self.target_column = target_column
        self.seed_val  = seed_val  
        self.best_synthesizer = None
        self.study = None

        # Fixing randomness globally (reproducibility of NN models)
        # This is not enough, reseeding is still needed before each fit/sample from the synthesizer 
        set_seed(self.seed_val)

        # Real diverted and not diverted flights (with relational features for utility evaluation)
        self.real_div_notdiv_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_notdiv_with_relational.pkl')

        # Only diverted flights (with relational features for statistical & diversity evaluation)
        self.real_div_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_with_relational.pkl')
        
        # Metadata (with relational features for statistical, diversity & utility evaluations)
        self.metadata_with_relational = Metadata.load_from_json('../data/preprocessed_data/metadata_with_relational.json')

        # DataFrame with unique pairs of airport IDs and corresponding distances (for realism evaluation)
        self.df_distance = pd.read_pickle('../data/preprocessed_data/df_distance.pkl')

        # Number of layers for generator and discriminator
        self.num_gen_layers = None
        self.num_dis_layers = None

    def comprehensive_evaluation(self, synthesizer):
        """
        Multi-metric evaluation of COPGAN performance
        Returns composite score (higher is better)
        """
        try:
            # Reseeding is needed before each fit/sample from the synthesizer
            set_seed(self.seed_val)

            # Generate synthetic data
            synthetic_data = synthesizer.sample(num_rows=1000)

            scores = {}
            
            # 1. Synthetic data realism
            realism_score = self._evaluate_synthetic_realism(synthetic_data)
            scores['realism'] = realism_score

            # Adding relational features and removing invalid routes from synthetic data 
            # before other evaluations (post-generation cleaning)
            # This can lead to empty DataFrame due to no valid routes (handled in the following evaluations)
            synthetic_data = add_relatioanal_features_clean_inv_routes(synthetic_data)

            # 2. Statistical Similarity (SDV Quality Score)
            stat_sim = self._evaluate_statistical_similarity(synthetic_data, verbose=False)
            scores['marginal_sim'] = stat_sim[0]
            scores['bivariate_sim'] = stat_sim[1]
            scores['overall_sim'] = stat_sim[2]
            

            # 3. Fidelity (Discriminability)
            fidelity_score = self._evaluate_fidelity(synthetic_data, metric_to_use="f1")
            scores['fidelity'] = fidelity_score

            # 4. ML Utility (if target column specified)
            # Best evaluation metrics for unbalanced data ("pr_auc" or "mcc")
            utility_score = self._evaluate_ml_utility(synthetic_data, metric_to_use="pr_auc")
            scores['utility'] = utility_score          

            # Composite score with weights
            weights = {
                'realism': 0.25,
                # 'marginal_sim': 0.25,
                # 'bivariate_sim': 0.25,
                'overall_sim': 0.25,
                'fidelity': 0.25,
                'utility': 0.25
            }

            composite_score = sum(scores[metric] * weight 
                                for metric, weight in weights.items())
            
            return composite_score, scores
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0, {}

    def _evaluate_synthetic_realism(self, synthetic_data):
        """
        Conducted on the features used in the generation + "Distance (miles)" feature.
        Evaluate the realism of synthetic data by checking the presentage of valid routes.
        Valed routes are those that have a non-null distance (routes existed in historical data). 
        """
        try:
            # Merge synthetic_data DataFrame with df_distance to add the "Distance (miles)" column
            synthetic_data = synthetic_data.merge(self.df_distance, on=["Origin Airport ID", "Destination Airport ID"], how="left")

            # Return normalized score of valid routes (closer to 1 means better realism)
            return synthetic_data["Distance (miles)"].notna().mean()
            
        except Exception as e:
            message = f"Synthetic realism evaluation failed due to: \n   {e}"
            logging.warning(message)
            return 0.0  # Fails only when 0 valid routes


    def _evaluate_statistical_similarity(self, synthetic_data, verbose=False):
        """
        Conducted on the features used in the generation + relational features.
        Evaluate statistical similarity using SDV's evaluate_quality.
        Returns a list of similarity scores:
        - Marginal similarity
        - Bivariate similarity
        - Overall similarity (average of the above two)
        """
        try:
            # In case of no valid routes after cleaning
            if synthetic_data.shape[0] == 0:
                raise ValueError("No valid synthetic routes remaining after cleaning. Cannot evaluate statistical similarity.")

            # Columns defined in the metadata as city, state, or state_bbr are excluded
            quality_report = evaluate_quality(
                self.real_div_with_relational,  # Only diverted flights (with relational features for statistical & diversity evaluation)
                synthetic_data,                 # Relational features added outside this function
                self.metadata_with_relational,  # Metadata (with relational features for statistical, diversity & utility evaluations)
                verbose=verbose
            )
            similarity = quality_report.get_properties()['Score'].tolist()
            similarity.append(np.mean(similarity))

            return similarity  # [Marginal, Bivariate, Overall similarity]
        
        except Exception as e:
            message = f"Statistical similarity evaluation failed due to: \n   {e}"
            logging.warning(message)
            return [0.0, 0.0, 0.0]  # Neutral scores for all metrics
        

    def _evaluate_fidelity(self, synthetic_data, metric_to_use="f1"):
        """
        Conducted on the features used in the generation.
        Evaluate the ability of classifiers to discriminate between real and synthetic data.
        The more classifiers are confused, the better the fidelity (i.e., lower discriminability).
        """
        try:       
            # In case of no valid routes after cleaning
            if synthetic_data.shape[0] == 0:
                raise ValueError("No valid synthetic routes remaining after cleaning. Cannot evaluate fidelity.")   

            # Only features used in the generation
            gen_cols = self.real_div_no_relational.columns.to_list()

            # Encode categorical and datetime features
            # - Work on copies inside this function to avoid messing the global DataFrames
            real, synthetic = prepare_for_prediction(self.real_div_no_relational, synthetic_data[gen_cols])

            # Add 'label' column and combine real and synthetic data
            real['label'] = 1       # Real data labeled as 1
            synthetic['label'] = 0  # Synthetic data labeled as 0
            combined_data = pd.concat([real, synthetic], ignore_index=True)

            # Features and labels
            X = combined_data.drop(columns=['label'])
            Y = combined_data['label']
            
            classifier = RandomForestClassifier(random_state=self.seed_val)

            # StratifiedKFold to preserve class distribution in  each fold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed_val)

            if metric_to_use == "f1":
                scores = cross_val_score(classifier, X, Y, cv=cv, scoring='f1')
            elif metric_to_use == "accuracy":
                # scores = cross_val_score(classifier, X, Y, cv=cv, scoring='accuracy')
                # real diversions (class 1) are few compared to synthetic ones (class 0), thus we use balanced accuracy
                scores = cross_val_score(classifier, X, Y, cv=cv, scoring=make_scorer(balanced_accuracy_score))
            else:
                raise ValueError(f"Invalid metric: '{metric_to_use}'. Choose 'accuracy' or 'f1'.")

            # Higher is better for fidelity (closer to 1 means calssifiers are confused)
            return 1 - scores.mean() # To maximize in the opjective function

        except Exception as e:
            message = f"Fidelity evaluation failed due to: \n   {e}"
            logging.warning(message)
            return 0.0


    def _evaluate_ml_utility(self, synthetic_data, metric_to_use="pr_auc"):
        """
        Conducted on the features used in the generation + relational features.
        Evaluate how well synthetic data preserves ML utility.
        """
        try:
            # In case of no valid routes after cleaning
            if synthetic_data.shape[0] == 0:
                raise ValueError("No valid synthetic routes remaining after cleaning. Cannot evaluate ML utility.")

            # Prepare for prediction by:
            # - Encoding categorical and datetime features
            # - Remove features that are not needed when predicting the target_column
            # - Work on copies inside this function to avoid messing the global DataFrames
            real, synthetic_diversions = prepare_for_prediction(self.real_div_notdiv_with_relational, synthetic_data, self.target_column)

            # Features and labels
            real_X = real.drop(columns=[self.target_column])
            real_Y = real[self.target_column]
            synthetic_X = synthetic_diversions.drop(columns=[self.target_column])
            synthetic_Y = synthetic_diversions[self.target_column]

            # Split with stratification (only real data)
            real_X_train, real_X_test, real_Y_train, real_Y_test = train_test_split(
                real_X,
                real_Y,
                test_size=0.3,
                stratify=real_Y,
                random_state=self.seed_val
                )

            # Augmentation (adding synthetic diversions to the real training data)
            augmented_X_train = pd.concat([real_X_train, synthetic_X], ignore_index=True)
            augmented_Y_train = pd.concat([real_Y_train, synthetic_Y], ignore_index=True)
            
            # # ========== TRTR (Train on Real, Test on Real) ==========
            # # Create pipeline with SMOTE and classifier
            # classifier = RandomForestClassifier(
            #     class_weight="balanced",       # Important for imbalance
            #     n_estimators=100,
            #     random_state=self.seed_val
            #     )
            # pipe = ImbPipeline([
            #     ('scaler', StandardScaler()),
            #     ('smote', SMOTE(random_state=self.seed_val)), 
            #     ('model', classifier)
            #     ])
            # pipe.fit(real_X_train, real_Y_train)
            # Y_pred = pipe.predict(real_X_test)
            # Y_pred_prob = pipe.predict_proba(real_X_test)  # Get probabilities instead of predictions (for PR AUC)

            # if metric_to_use == "pr_auc":               
            #     # Precision-Recall AUC (PR AUC) for probability of positive class (diversion)
            #     pr_auc_trtr = average_precision_score(real_Y_test, Y_pred_prob[:, 1]) 
            # elif metric_to_use == "mcc":
            #     # Matthews Correlation Coefficient (MCC)
            #     mcc_trtr = matthews_corrcoef(real_Y_test, Y_pred)
            # else:
            #     raise ValueError(f"Invalid metric: {metric_to_use}. Choose 'pr_auc' or 'mcc'.")

            # ======== TATR (Train on Augmented, Test on Real) =======
            # TATR (Train on Augmented, Test on Real)
            # Create pipeline with SMOTE and classifier
            classifier = RandomForestClassifier(
                class_weight="balanced",       # Important for imbalance
                n_estimators=100,
                random_state=self.seed_val
                )
            pipe = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=self.seed_val)), 
                ('model', classifier)
                ])
            pipe.fit(augmented_X_train, augmented_Y_train)
            Y_pred = pipe.predict(real_X_test)
            Y_pred_prob = pipe.predict_proba(real_X_test)  # Get probabilities instead of predictions (for PR AUC)

            if metric_to_use == "pr_auc":               
                # Precision-Recall AUC (PR AUC) for probability of positive class (diversion)
                pr_auc_tatr = average_precision_score(real_Y_test, Y_pred_prob[:, 1]) 
                score = pr_auc_tatr
            elif metric_to_use == "mcc":             
                # Matthews Correlation Coefficient (MCC)
                mcc_tatr = matthews_corrcoef(real_Y_test, Y_pred)
                score = (mcc_tatr + 1) / 2 # Normalize from [-1,1] to [0,1] for optimization consistency
            else:
                raise ValueError(f"Invalid metric: {metric_to_use}. Choose 'pr_auc' or 'mcc'.")
        
            return score
        
        except Exception as e:
            # It falis when no valid routes after ckeaning synthetic data
            message = f"ML utility evaluation failed due to: \n   {e}"
            logging.warning(message)
            return 0.0

    def objective(self, trial):
        """
        Optuna objective function for COPGAN hyperparameters
        """
        # Define hyperparameter search space
        # Phase 1: epochs, network dims, learning rates
        # epochs = trial.suggest_int('epochs', 1, 5, step=1)
        epochs = trial.suggest_int('epochs', 300, 6000, step=100)

        # Method 1: fixed dimension combinations (results in lists and causes problems in "plot_parallel_coordinate")
        # generator_dim = trial.suggest_categorical('generator_dim', [(128, 128), (256, 256), (512, 512), (256, 512), (512, 256)])
        # discriminator_dim = trial.suggest_categorical('discriminator_dim', [(128, 128), (256, 256), (512, 512), (256, 512), (512, 256)])
        
        # # Method 1 (updated): fixed dimension combinations (as strings to avoid problems in "plot_parallel_coordinate")
        # generator_dim_str = trial.suggest_categorical('generator_dim', ['128,128', '256,256', '512,512', '256,512', '512,256'])
        # discriminator_dim_str = trial.suggest_categorical('discriminator_dim', ['128,128', '256,256', '512,512', '256,512', '512,256'])
        # # Convert back to tuples
        # generator_dim = tuple(map(int, generator_dim_str.split(',')))
        # discriminator_dim = tuple(map(int, discriminator_dim_str.split(',')))
        
        # Method 2: variable dimension combinations
        self.num_gen_layers = 2
        self.num_dis_layers = 2
        generator_dim = [
            trial.suggest_categorical(f'generator_{i}',
                                    [128, 256, 512])
            for i in range(self.num_gen_layers)
        ]
        discriminator_dim = [
            trial.suggest_categorical(f'discriminator_{i}',
                                    [128, 256, 512])
            for i in range(self.num_dis_layers)
        ]

        generator_lr = trial.suggest_float('generator_lr', 1e-5, 1e-3, log=True)
        discriminator_lr = trial.suggest_float('discriminator_lr', 1e-5, 1e-3, log=True)

        # # For variable number of layers
        # generator_num_layers = trial.suggest_int("generator_num_layers", 1, 4)
        # discriminator_num_layers = trial.suggest_int("discriminator_num_layers", 1, 4)
        # generator_dim = [trial.suggest_categorical(f'generator_layer_{i}',[8, 16, 32, 64, 128, 200, 256, 300, 400, 500, 600])for i in range(generator_num_layers)]
        # discriminator_dim = [trial.suggest_categorical(f'discriminator_layer_{i}',[8, 16, 32, 64, 128, 200, 256, 300, 400, 500, 600])for i in range(discriminator_num_layers)]

        # # Phase 2: embedding_dim, pac, discriminator_steps
        # # Size of the random sample passed to the Generator
        # embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256, 512])
        # # Number of samples to group together when applying the discriminator
        # pac = trial.suggest_int('pac', 1, 20)  
        # # Number of discriminator updates to do for each generator update
        # discriminator_steps = trial.suggest_int('discriminator_steps', 1, 5)

        # # Phase 3: weight decay, log_frequency
        # # Generator weight decay for the Adam Optimizer
        # generator_decay = trial.suggest_float('generator_decay', 1e-8, 1e-4, log=True)
        # # Discriminator weight decay for the Adam Optimizer
        # discriminator_decay = trial.suggest_float('discriminator_decay', 1e-8, 1e-4, log=True)
        # # Whether to use log frequency of categorical levels in conditional sampling
        # log_frequency = trial.suggest_categorical('log_frequency', [True, False])
        
        try:
            # Reseeding is needed before each fit/sample from the synthesizer
            set_seed(self.seed_val)  

            # Create synthesizer with suggested parameters
            synthesizer = CopulaGANSynthesizer(
                self.metadata_no_relational,
                default_distribution='gaussian_kde',
                epochs=epochs,
                generator_dim=generator_dim,
                discriminator_dim=discriminator_dim,
                generator_lr=generator_lr,
                discriminator_lr=discriminator_lr,
                # embedding_dim=embedding_dim,
                # pac=pac,
                # discriminator_steps=discriminator_steps,
                # generator_decay=generator_decay,
                # discriminator_decay=discriminator_decay,
                # log_frequency=log_frequency,
                verbose=False
            )
            
            # Fit the synthesizer
            synthesizer.fit(self.real_div_no_relational)
            
            # Evaluate performance
            score, detailed_scores = self.comprehensive_evaluation(synthesizer)
            
            # Log detailed metrics for analysis
            for metric, value in detailed_scores.items():
                trial.set_user_attr(f"score_{metric}", value)
            
            # Store best synthesizer
            if score > getattr(self, 'best_score', 0):
                self.best_score = score
                self.best_synthesizer = synthesizer
            
            return score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            # Return very low score for failed trials
            return 0.0
    
    def optimize(self, n_trials=50, timeout=None, n_jobs=1):
        """
        Run Bayesian optimization
        
        Parameters:
        - n_trials: Maximum number of trials
        - timeout: Maximum time in seconds
        - n_jobs: Number of parallel jobs (1 for sequential)
        """
        
        # Create study with TPE sampler
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,  # Random trials before TPE kicks in
            n_ei_candidates=24,   # Number of candidates for acquisition function
            seed=self.seed_val               # Fix randomness in "n_startup_trials"
        )
        
        # Other samplers:
        # sampler = optuna.samplers.CmaEsSampler(seed=self.seed_val)  # CMA-ES
        # sampler = optuna.samplers.RandomSampler(seed=self.seed_val)  # Random search
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='copgan_hyperopt',
            storage='sqlite:///../data/outputs/optimization_studies/optuna_study_copgan.db',
            load_if_exists=True

        )
        
        print(f"Starting Bayesian optimization with {n_trials} trials...")
        print(f"Data shape: {self.real_div_no_relational.shape}")
        print("-" * 75)
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        print("\nOptimization completed!")
       
        return self.study.best_params, self.study.best_value
    
    def get_optimization_results(self):
        """
        Get detailed results from optimization
        """
        if not self.study:
            return None
        
        results = {
            'n_trials': len(self.study.trials),
            'best_trial_id': self.study.best_trial.number, 
            'best_trial': self.study.best_trial,
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
        }
        
        # Get parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            results['param_importance'] = importance
        except:
            results['param_importance'] = {}
        
        return results
    
    def print_all_traials(self):
        """
        Inspect the completed trials
        """
        if not self.study:
            return None
        
        print("-"*75, "\nAll Trials:")
        for t in self.study.trials:
            print(f"Trial {t.number}: Value = {t.value}, Params = {t.params}")
        print("-"*75)
        
    def plot_optimization_history(self):
        """
        Plot optimization history
        """
        if not self.study:
            print("No study available. Run optimization first.")
            return
        
        try:
            # Plot optimization history
            fig1 = optuna.visualization.plot_optimization_history(self.study)
            fig1.show()
            # fig1.write_html("optimization_history_plot.html")

            # Plot parameter importance
            fig2 = optuna.visualization.plot_param_importances(self.study)
            fig2.show()
            # fig2.write_html("param_importances_plot.html")
            
            # Plot parameter relationships
            fig3 = optuna.visualization.plot_parallel_coordinate(self.study)
            fig3.show()
            # fig3.write_html("parallel_coordinate_plot.html")

            # How each hyperparameter relates to objective value
            fig4 = optuna.visualization.plot_slice(self.study)
            fig4.show()
            # fig4.write_html("slice_plot.html")

            #Contour plots of parameter spaces vs. objective
            fig5 = optuna.visualization.plot_contour(self.study)
            fig5.show()
            # fig5.write_html("contour_plot.html")

            # # Displays intermediate results (only if your objective reports intermediate scores via trial.report()).
            # fig6 = optuna.visualization.plot_intermediate_values(self.study)
            # fig6.show()
            # # fig6.write_html("intermediate_values_plot.html")

            # # Compares distributions of objective values between multiple studies.
            # fig7 = optuna.visualization.plot_edf(self.study)
            # fig7.show()
            # # fig7.write_html("edf_plot.html")

        except Exception as e:
            print(f"Plotting error: {e}")
            print("Install plotly for optuna.visualizationualization: pip install plotly")
    
    def create_final_synthesizer(self):
        """
        Create, train and save the final synthesizer with best parameters.
        """
        if not self.study:
            print("No optimization results available.")
            return None

        best_params = self.study.best_params

        # # Method 1: Convert list back to tuples
        # generator_dim=tuple(best_params['generator_dim'])
        # discriminator_dim=tuple(best_params['discriminator_dim'])

        # # Method 1 (updated): Convert strings back to tuples
        # generator_dim = tuple(map(int, best_params['generator_dim'].split(',')))
        # discriminator_dim = tuple(map(int, best_params['discriminator_dim'].split(',')))
        
        # Method 2
        generator_dim = tuple(best_params[f'generator_{i}'] for i in range(self.num_gen_layers))
        discriminator_dim = tuple(best_params[f'discriminator_{i}'] for i in range(self.num_dis_layers))

        # Reseeding is needed before each fit/sample from the synthesizer
        set_seed(self.seed_val)

        # Create final synthesizer with best parameters
        synthesizer = CopulaGANSynthesizer(
            self.metadata_no_relational,
            default_distribution='gaussian_kde',
            epochs=best_params['epochs'],
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=best_params['generator_lr'],
            discriminator_lr=best_params['discriminator_lr'],
            # embedding_dim=best_params['embedding_dim'],
            # pac=best_params['pac'],
            # discriminator_steps=best_params['discriminator_steps'],
            # generator_decay=best_params['generator_decay'],
            # discriminator_decay=best_params['discriminator_decay'],
            # log_frequency=best_params['log_frequency'],
            verbose=False
        )
        
        # Train final synthesizer with best parameters
        print("Training final synthesizer with best parameters...")
        synthesizer.fit(self.real_div_no_relational)
        
        # Save the final synthesizer
        synthesizer_path = '../data/outputs/synthesizers/synthesizer_COPGAN_diversions_optimal.pkl'
        print(f"Saving final synthesizer to {synthesizer_path}...") 
        synthesizer.save(filepath=synthesizer_path)

        return synthesizer

# Usage Example
def main():
    # Fixing randomness for reproducibility
    seed_val = 45

    # Load data
    # Only diverted flights (without relational features to train the synthesizer & evaluate fidelity)
    real_div_no_relational = pd.read_pickle('../data/preprocessed_data/real_div_no_relational.pkl')
    # Metadata (without relational features to train the synthesizer & evaluate fidelity)
    metadata_no_relational = Metadata.load_from_json('../data/preprocessed_data/metadata_no_relational.json')

    # Instantiate the optimizer class
    optimizer = COPGANBayesianOptimizer(
        real_data=real_div_no_relational,  # Only diverted flights (without relational features to train the synthesizer & evaluate fidelity)
        metadata=metadata_no_relational,  # Metadata (without relational features to train the synthesizer & evaluate fidelity)
        target_column='Diversion Label',  # Optional
        seed_val=seed_val                 # Fix randomness in optimization (reproducibility of paraeter search)
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize(
        n_trials=100,  # Number of trials
        # timeout=3600,  # 1 hour timeout
        n_jobs=1
    )

   
    # Get detailed results
    results = optimizer.get_optimization_results()
    print("-"*75, f"\nTotal number of trials: {results['n_trials']} (best traial: {results['best_trial_id']})")
    print("-"*75, f"\nBest trial details:\n{results['best_trial']}")
    
    print("-"*75, f"\nBest parameters:\n{results['best_params']}")
    print(f"\nParameter importance:\n{results['param_importance']}")
    
    print("-"*75, f"\nBest score: {results['best_score']:.4f}")
    print(f"\nScore details:\n{results['best_trial'].user_attrs}")
    
    optimizer.print_all_traials()   # Inspect the completed trials


    # Plot results
    optimizer.plot_optimization_history()
    
    # Create, train and save final synthesizer
    final_synthesizer = optimizer.create_final_synthesizer()
    
    # Generate synthetic data
    # synthetic_data = final_synthesizer.sample(num_rows=1000)


if __name__ == "__main__":
    main()
import optuna
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='optuna')

# Configuration 
MODELS = {
    'TVAE': {
        'study_name': 'tvae_hyperopt',
        'storage':    'sqlite:///../data/outputs/optimization_studies/optuna_study_tvae.db',
    },
    'CTGAN': {
        'study_name': 'ctgan_hyperopt',
        'storage':    'sqlite:///../data/outputs/optimization_studies/optuna_study_ctgan.db',
    },
    'CopulaGAN': {
        'study_name': 'copgan_hyperopt',
        'storage':    'sqlite:///../data/outputs/optimization_studies/optuna_study_copgan.db',
    },
}

WEIGHT_SCHEMES = {
    'Equal (baseline)':  dict(realism=0.25, overall_sim=0.25, fidelity=0.25, utility=0.25),
    'Utility-heavy':     dict(realism=0.10, overall_sim=0.10, fidelity=0.10, utility=0.70),
    'Fidelity-heavy':    dict(realism=0.10, overall_sim=0.10, fidelity=0.70, utility=0.10),
    'Statistical-heavy': dict(realism=0.10, overall_sim=0.70, fidelity=0.10, utility=0.10),
    'Realism-heavy':     dict(realism=0.70, overall_sim=0.10, fidelity=0.10, utility=0.10),
}

METRIC_KEYS = ['realism', 'overall_sim', 'fidelity', 'utility']

# load one study into a DataFrame
def load_study_scores(study_name, storage):
    study = optuna.load_study(study_name=study_name, storage=storage)
    records = []
    for t in study.trials:
        if t.value is None:
            continue
        row = {'trial': t.number}
        for key in METRIC_KEYS:
            row[key] = t.user_attrs.get(f'score_{key}', 0.0)
        records.append(row)
    return pd.DataFrame(records)

# Compute composite scores and find best trial per scheme 
def sensitivity_analysis(df, model_name):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    summary_rows = []

    for scheme_name, w in WEIGHT_SCHEMES.items():
        df[scheme_name] = sum(df[k] * w[k] for k in METRIC_KEYS)
        best_idx  = df[scheme_name].idxmax()
        best_row  = df.loc[best_idx]

        summary_rows.append({
            'Weight scheme':     scheme_name,
            'Best trial':        int(best_row['trial']),
            'Composite score':   round(best_row[scheme_name], 4),
            'Realism':           round(best_row['realism'],     4),
            'Statistical sim.':  round(best_row['overall_sim'], 4),
            'Fidelity':          round(best_row['fidelity'],    4),
            'Utility (PR-AUC)':  round(best_row['utility'],     4),
        })

    summary_df = pd.DataFrame(summary_rows).set_index('Weight scheme')
    print(summary_df.to_string())
    return summary_df

# Main 
all_summaries = {}

for model_name, cfg in MODELS.items():
    try:
        df = load_study_scores(cfg['study_name'], cfg['storage'])
        summary = sensitivity_analysis(df, model_name)
        all_summaries[model_name] = summary
    except Exception as e:
        print(f"Could not load study for {model_name}: {e}")

# Export to Excel
with pd.ExcelWriter('../data/outputs/optimization_studies/sensitivity_analysis.xlsx') as writer:
    for model_name, summary in all_summaries.items():
        summary.to_excel(writer, sheet_name=model_name)

print("\nResults exported to ../data/outputs/optimization_studies/sensitivity_analysis.xlsx")
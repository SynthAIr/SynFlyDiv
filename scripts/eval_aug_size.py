# Evaluation of the impact of augmented data size on data utility

# Import necessary libraries
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

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
df_plot = df_aug_sizes.copy()
df_plot["Model"] = df_plot["Model"].apply(rename_model)

print(f"Loaded {len(pkl_files)} files")
print(f"Final dataframe shape: {df_aug_sizes.shape}")

#======================(For slides)========================
for i in ["before", "after"]:
    for metric in ["PR-AUC", "Normalised MCC"]:
        if metric == "Normalised MCC":
            plt.figure(figsize=(7.6,6))
        else:
            plt.figure(figsize=(7.53,6))

        # Add horizontal lines for TRTR values
        trtr_val  = df_plot[df_plot["Model"]=="TRTR"][metric].iloc[0]
        plt.axhline(y=trtr_val , color='black', linestyle='--', linewidth=1.5, alpha=1, label='$TRTR$')

        # Add vertical line when augmentation size equals the size of real data
        plt.axvline(x=60000 , color='blue', linestyle='--', linewidth=1.5, alpha=1)

        # Get a consistent color palette for the models
        palette = sns.color_palette("tab10", n_colors=df_plot["Model"].nunique())
        # Map models (exclude TRTR) -> colors
        model_colors = dict(zip(df_plot.iloc[1:]["Model"].unique(), palette))

        # Plot PR-AUC & Normalised MCC vs Size before & after cleaning (light markers)
        sns.lineplot(
            data=df_plot.iloc[1:], # plot from second row and skip row with TRTR values
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
        for model in df_plot.iloc[1:]["Model"].unique():   # remove TRTR from the list of models
            sub = df_plot[df_plot["Model"] == model]
            x = sub[f"Size {i} cleaning"].values
            y = sub[metric].values
            # Apply LOWESS smoothing
            lowess = sm.nonparametric.lowess(y, x, frac=0.3) 
            plt.plot(lowess[:,0], lowess[:,1], color=model_colors[model], alpha=1, linewidth=2)

        # Grid and limits
        plt.grid(True, linestyle="--", alpha=0.7)
        y_max = max(df_plot[metric].max(), trtr_val)
        y_min = min(df_plot[metric].min(), trtr_val)
        margin = 0.05 * (y_max - y_min)                 # 5% of the range
        plt.ylim(y_min - margin, y_max + 5.5 * margin)  # Set y-limits slightly above max and below min
        # plt.xscale("log")

        # Labels and title
        plt.xlabel(f"Augmentation size {i} cleaning", fontsize=16)
        plt.ylabel(f"Utility ({metric})", fontsize=16)
        plt.title(f"Utility ({metric}) vs augmentation size {i} cleaning", fontsize=15)

        # ---- Main legend (models and TRTR line) ----
        handles, labels = plt.gca().get_legend_handles_labels()


        legend1 = plt.legend(handles=handles, labels=labels, 
                            loc="upper center", bbox_to_anchor=(0.5, 1.01), 
                            ncol=4, frameon=True, edgecolor='white', framealpha=1, fontsize=12.3, handlelength=1.9)
        plt.gca().add_artist(legend1)
        
        # ---- Second legend (vertical line) ----
        # Add a fake transparent handle for spacing
        dummy = Line2D([], [], color="none", alpha=0, label="")

        handles.append(dummy)
        labels.append("")
        vertical_line = Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5,
                            label='     Point of class balance      ')

        plt.legend(handles=[dummy, dummy, vertical_line, dummy, dummy], 
                loc='upper center', bbox_to_anchor=(0.5, 0.89), ncol=7, 
                frameon=True, edgecolor='white',framealpha=1, fontsize=12.3, handlelength=1.9)

        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        plt.tick_params(axis='both', labelsize=14)  
        plt.tight_layout()
        plt.show()
# =====================================================================
# g = sns.FacetGrid(df_plot.iloc[1:], col="Model", col_wrap=3, sharey=True, sharex=True, height=4)
# g.map_dataframe(sns.lineplot, x="Size before cleaning", y="PR-AUC", marker="o")
# g.set_axis_labels("Size before cleaning", "PR-AUC")
# g.set_titles(col_template="{col_name}")
# g.fig.suptitle("PR-AUC vs Size before cleaning per Model", y=1.05, fontsize=14)
# plt.show()
# # =======================================================================
# g = sns.FacetGrid(df_plot.iloc[1:], col="Model", col_wrap=3, sharey=True, sharex=True, height=4)
# g.map_dataframe(sns.lineplot, x="Size before cleaning", y="Normalised MCC", marker="o")
# g.set_axis_labels("Size before cleaning", "Normalised MCC")
# g.set_titles(col_template="{col_name}")
# g.fig.suptitle("Normalised MCC vs Size before cleaning per Model", y=1.05, fontsize=14)
# plt.show()
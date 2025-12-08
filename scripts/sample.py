# Samples from saved synthesizers (with default or optimal hyperparameters).
# Import necessary libraries
import argparse
import sys
import ast
import numpy as np
import pandas as pd
from utils import color, set_seed
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import Metadata
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
    CopulaGANSynthesizer,
)
    
# Dictionary of available synthesizer classes
SYNTHESIZER_CLASSES = {
    'gc': GaussianCopulaSynthesizer,
    'ctgan': CTGANSynthesizer,
    'tvae': TVAESynthesizer,
    'copgan': CopulaGANSynthesizer,
}

# execute the script with command line arguments or Jupyter notebook
if __name__ == "__main__":
    if any(arg.startswith('--') for arg in sys.argv[1:]):
        # Case 1: run via terminal with arguments like --synth_type ...
        parser = argparse.ArgumentParser()
        parser.add_argument('--synth_name', type=str, required=True, help= "Name of the synthesizer to use (e.g., 'synthesizer_TVAE_diversions_optimal_V0'") 
        parser.add_argument('--num_samples', type=int, nargs='+', required=True)  # e.g., 300 500
        args = parser.parse_args()
        synth_name = args.synth_name
        num_samples = args.num_samples
    else:
        # Case 2: run via Jupyter with: %run sample.py synthesizer_GC_diversions_default [300, 500]
        if len(sys.argv) < 3:
            raise ValueError("Usage: %run sample.py <synth_name> <num_samples>")
        synth_name = sys.argv[1]
        try:
            num_samples = ast.literal_eval(sys.argv[2])  # expects a string like "[300, 500]"
        except (SyntaxError, ValueError):
            raise ValueError("num_samples must be a list, e.g., \"[300, 500]\"")

    synth_type = synth_name.split('_')[1].lower()  # Extract synthesizer type from the name
    synth_state = synth_name.split('_')[3].lower()  # Extract synthesizer state
    synthesizer_path = f'../data/outputs/synthesizers/{synth_name}.pkl'

    print(f"{color.BOLD}{color.GREEN}Sampling:{color.END}")
    print("   Synthesizer type:", synth_type)
    print("   Synthesizer state:", synth_state)
    print("   Synthesizer path:", synthesizer_path)
    print("   Number of samples:", num_samples)

    # Load the synthesizer and print its parameters
    SynthClass = SYNTHESIZER_CLASSES[synth_type]  
    synthesizer = SynthClass.load(filepath=synthesizer_path)
    print(f"\n   {synth_type.upper()} parameters:")
    print("\n".join(f"      {k}: {v}" for k, v in synthesizer.get_parameters().items()))

    print("\n"+ "-"*75)
    print("|"*75)
    print("V"*75, "\n")

#=======================================================================
# Import data
#=======================================================================

# Load full DataFrame (to extract airport information)
df_utc = pd.read_pickle('../data/preprocessed_data/df_utc_weather.pkl')
# Only diverted flights (with relational features for statistical & diversity evaluation)
real_div_with_relational = pd.read_pickle('../data/preprocessed_data/real_div_with_relational.pkl')

# Metadata (without relational features to train the synthesizer & evaluate fidelity)
metadata_no_relational = Metadata.load_from_json('../data/preprocessed_data/metadata_no_relational.json')
# Metadata (with relational features for statistical, diversity & utility evaluations)
metadata_with_relational = Metadata.load_from_json('../data/preprocessed_data/metadata_with_relational.json')

sorted_features = ["Unique Carrier Code", "Tail Number", 
                    "Origin Airport ID", "ICAO Origin Airport", "Origin City", "Origin State Code", "Origin State Name", 
                    "Destination Airport ID", "ICAO Destination Airport", "Destination City", "Destination State Code", "Destination State Name", 
                    "Quarter", "Day of Week", 
                    "Scheduled Departure Time UTC", "Actual Departure Time UTC", "Departure ΔT (min)", "Departure Delay Label", 
                    "Taxi Out Time (min)", "Wheels Off Time UTC", 
                    "Wheels On Time UTC", "Taxi In Time (min)", 
                    "Scheduled Arrival Time UTC", "Actual Arrival Time UTC", "Arrival ΔT (min)", "Arrival Delay Label", 
                    "Scheduled Elapsed Time (min)", "Actual Elapsed Time (min)", "Air Time (min)", "Distance (miles)", 
                    "Diversion Label",

                    "Status", "Cancellation Code", 
                    "Carrier Delay (min)", "Weather Delay (min)", "National Air System Delay (min)", 
                    "Security Delay (min)", "Late Aircraft Delay (min)",
                    
                    "Origin_temperature_C", "Origin_dew_point_C",
                    "Origin_wind_speed_KMH", "Origin_wind_direction_deg",
                    "Origin_visibility_M", "Origin_pressure_hPa",
                    "Origin_weather", "Origin_cloud_cover",
                    "Destination_temperature_C", "Destination_dew_point_C", 
                    "Destination_wind_speed_KMH", "Destination_wind_direction_deg",
                    "Destination_visibility_M", "Destination_pressure_hPa",
                    "Destination_weather", "Destination_cloud_cover"]

#===============================================================================
# Sample Synthetic Data
#===============================================================================

# Fixing randomness for reproducibility
set_seed(45)

# Sample multiple times
for size in num_samples:
    print("-"*75 + "\nSampling {} synthetic diverted flights...".format(size))
    # Load trained model (inside the loop to ensure getting the same samples every run)
    synthesizer = SynthClass.load(filepath=synthesizer_path)

    # Sample (number of samples before cleaning)
    synthetic_flight_information = synthesizer.sample(num_rows=size)
    before = synthetic_flight_information.shape

    #===========================================================================
    # Check data validity, structure & statistical similarity (only columns used in the generation)
    #===========================================================================

    # Columns used in the generation (for psot-generation checking))
    gen_cols = synthetic_flight_information.columns.to_list()
    
    # Removing relational features (columns used in the generation)
    print("-"*75 + "\nChecking data validity and structure...")
    diagnostic = run_diagnostic(
        real_div_with_relational[gen_cols],
        synthetic_flight_information,
        metadata_no_relational
    )

    # Check statistical similarity (columns used in the generation - before cleaning)
    print("-"*75 + f"\nChecking statistical similarity ({color.BOLD}{color.BLUE}columns used in generation - before cleaning{color.END})...")
    quality_report = evaluate_quality(
        real_div_with_relational[gen_cols],
        synthetic_flight_information,
        metadata_no_relational
    )
    # Use if Data Validity < 100% to explore problematic columns
    # diagnostic.get_details(property_name='Data Validity')

    #===========================================================================
    # Recalculate relational features (removed from df_utc_durations_2)
    #===========================================================================

    #========================("Quarter", "Day of Week")=========================
    # Create "Quarter" and "Day of Week" (Monday=1, Sunday=7)
    synthetic_flight_information["Quarter"] = synthetic_flight_information["Scheduled Departure Time UTC"].dt.quarter
    synthetic_flight_information["Day of Week"] = synthetic_flight_information["Scheduled Departure Time UTC"].dt.weekday + 1

    #===========================("Distance (miles)")============================
    # Check if the distance is unique for each unique pair of airport IDs
    is_unique_distance = (
        df_utc.groupby(["Origin Airport ID", "Destination Airport ID"])["Distance (miles)"]
        .nunique().eq(1).all())

    # print("="*75)
    # if is_unique_distance:
    #     print("The distance is unique for each unique pair of airport IDs.")
    # else:
    #     print("There are pairs of airport IDs with multiple distances.")

    # Create a DataFrame with unique pairs of airport IDs and corresponding distances
    df_distance = df_utc.drop_duplicates(subset=["Origin Airport ID", "Destination Airport ID", "Distance (miles)"])[
        ["Origin Airport ID", "Destination Airport ID", "Distance (miles)"]
    ].reset_index(drop=True)

    # Merge synthetic_flight_information DataFrame with df_distance to add the "Distance (miles)" column
    synthetic_flight_information = synthetic_flight_information.merge(df_distance, on=["Origin Airport ID", "Destination Airport ID"], how="left")

    #===================(ICAO, City, State Code, State Name)====================
    # Extract unique airport IDs with their information from the full DataFrame
    origin_info = df_utc[[
        "Origin Airport ID",
        "ICAO Origin Airport",
        "Origin City",
        "Origin State Code",
        "Origin State Name"
    ]].rename(columns={
        "Origin Airport ID": "Airport ID",
        "ICAO Origin Airport": "ICAO Airport",
        "Origin City": "City",
        "Origin State Code": "State Code",
        "Origin State Name": "State Name"
    })

    destination_info = df_utc[[
        "Destination Airport ID",
        "ICAO Destination Airport",
        "Destination City",
        "Destination State Code",
        "Destination State Name"
    ]].rename(columns={
        "Destination Airport ID": "Airport ID",
        "ICAO Destination Airport": "ICAO Airport",
        "Destination City": "City",
        "Destination State Code": "State Code",
        "Destination State Name": "State Name"
    })

    # Combine origin and destination information from the full DataFrame
    airports_info = pd.concat([origin_info, destination_info], axis=0).drop_duplicates(subset=["Airport ID"]).reset_index(drop=True)

    # # Check the consistancy with the number of airports
    # print("-"*75)
    # print("{} 'Origin Airport' exist in the full DataFrame.".format(df_utc["Origin Airport ID"].nunique()))
    # print("{} 'Destination Airport' exist in the full DataFrame.".format(df_utc["Destination Airport ID"].nunique()))
    # print("{} airport information collected in 'airports_info' DataFrame.".format(airports_info.shape[0]))

    # Merge origin information
    synthetic_flight_information = synthetic_flight_information.merge(
        airports_info.rename(columns={
            "Airport ID": "Origin Airport ID",
            "ICAO Airport": "ICAO Origin Airport",
            "City": "Origin City",
            "State Code": "Origin State Code",
            "State Name": "Origin State Name"
        }),
        on="Origin Airport ID",
        how="left"
    )

    # Merge destination information
    synthetic_flight_information = synthetic_flight_information.merge(
        airports_info.rename(columns={
            "Airport ID": "Destination Airport ID",
            "ICAO Airport": "ICAO Destination Airport",
            "City": "Destination City",
            "State Code": "Destination State Code",
            "State Name": "Destination State Name"
        }),
        on="Destination Airport ID",
        how="left"
    )

    #==============================(Time features)==============================
    # 1. Departure Delay Label (NaN when "Departure ΔT (min)" = NaN)
    synthetic_flight_information["Departure Delay Label"] = synthetic_flight_information["Departure ΔT (min)"].apply(lambda x: 1.0 if pd.notna(x) and x >= 15 else (0.0 if pd.notna(x) else np.nan))
    # 2. Wheels Off Time UTC
    synthetic_flight_information["Wheels Off Time UTC"] = synthetic_flight_information["Actual Departure Time UTC"] + pd.to_timedelta(synthetic_flight_information["Taxi Out Time (min)"], unit="m")
    # 3. Wheels On Time UTC
    synthetic_flight_information["Wheels On Time UTC"] = synthetic_flight_information["Wheels Off Time UTC"] + pd.to_timedelta(synthetic_flight_information["Air Time (min)"], unit="m")
    # 4. Scheduled Arrival Time UTC
    synthetic_flight_information["Scheduled Arrival Time UTC"] = synthetic_flight_information["Scheduled Departure Time UTC"] + pd.to_timedelta(synthetic_flight_information["Scheduled Elapsed Time (min)"], unit="m")
    # 5. Actual Arrival Time UTC
    synthetic_flight_information["Actual Arrival Time UTC"] = synthetic_flight_information["Actual Departure Time UTC"] + pd.to_timedelta(synthetic_flight_information["Actual Elapsed Time (min)"], unit="m")
    # 6. Arrival Delay Label (NaN when "Arrival ΔT (min)" = NaN)
    synthetic_flight_information["Arrival Delay Label"] = synthetic_flight_information["Arrival ΔT (min)"].apply(lambda x: 1.0 if pd.notna(x) and x >= 15 else (0.0 if pd.notna(x) else np.nan))

    #=============================(Sorting columns)=============================
    synthetic_flight_information = synthetic_flight_information[sorted_features[:-23]]

    #==============================(Print shapes)===============================
    after = synthetic_flight_information.shape
    print("-" * 75, f"\nCalculating the relational features for {color.BOLD}{color.GREEN}\"synthetic_flight_information\"{color.END}:")
    print("   Shape before --> ", before)
    print("   Shape after ---> ", after)

    #===========================================================================
    # Post-generation cleaning
    #===========================================================================

    print("   Number of real flights = {}".format(real_div_with_relational.shape[0]))
    tot_syn = synthetic_flight_information.shape[0]

    # Removing invalid routes
    synthetic_flight_information = synthetic_flight_information.dropna(subset=["Distance (miles)"]).reset_index(drop=True)

    valid_syn = synthetic_flight_information.shape[0]
    valid_ratio = round((valid_syn / tot_syn) * 100, 1)

    print("-"*75, "\nPost-generation cleaning (removing invalid routes from synthetic data):")
    print("   Number of synthetic flights before cleaning = {}".format(tot_syn))
    print(f"   Number of synthetic flights after cleaning = {valid_syn} {color.BOLD}{color.RED}(valid samples {valid_ratio}%){color.END}")

    #===========================================================================
    # Check statistical similarity (columns used in the generation + relational features)
    #===========================================================================

    print("-"*75 + f"\nChecking statistical similarity ({color.BOLD}{color.BLUE}columns used in generation + relational features - after cleaning{color.END})...")
    quality_report = evaluate_quality(
        real_div_with_relational,
        synthetic_flight_information,
        metadata_with_relational
    )
    #===========================================================================
    # Save synthetic
    #===========================================================================
    synthetic_flight_information.to_pickle('../data/outputs/synthetic/synthetic_{}_diversions_{}_{}.pkl'.format(synth_type.upper(), synth_state, size))
    print("="*75)
    print("\n")


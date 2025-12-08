import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder

# Class for terminal text colors and styles
class color:
    PURPLE = '\033[95m'  # Purple text
    CYAN = '\033[96m'  # Cyan text
    DARKCYAN = '\033[36m'  # Dark cyan text
    BLUE = '\033[94m'  # Blue text
    GREEN = '\033[92m'  # Green text
    YELLOW = '\033[93m'  # Yellow text
    RED = '\033[91m'  # Red text
    BOLD = '\033[1m'  # Bold text
    UNDERLINE = '\033[4m'  # Underlined text
    END = '\033[0m'  # Reset to default style

# Fix seed for reproducibility
def set_seed(seed=45):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Uncomment if using CUDA

def simplify(df, df_name):
    """
    Remove canceled flight, causes of delays and cancellation and weather for simplicity
    Add "Diversion Label"
    """
    # Working on copies to avoid modifying the original DataFrames
    # Ensuring original DataFrames passed into the function remain unchanged outside the function scope.
    df = df.copy()
    
    # To be kept or removed after import (before the synthetic generation) for simplicity:
    features = ["Status", "Cancellation Code", 
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

    # Delete rows for cancelled flights
    print("=" * 75 + f"\n{color.BOLD}{color.GREEN}Simplifying '{df_name}':{color.END}")
    print("Shape before --> ", df.shape)
    n_cancelled = df[df["Status"] == "C"].shape[0]
    print("-"*75, "\nDeleting {} cancelled flights.".format(n_cancelled))

    df.drop(df[df["Status"] == "C"].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Adding new column "Diversion Label"
    df["Diversion Label"] = np.where(df["Status"] == "D", 1, 0)
    
    # Delete columns of status, causes of delays and cancellation and weather for simplicity
    print("-"*75, "\nDeleting {} columns {} for simplicity.".format(len(features), features))
    df.drop(features, axis=1, inplace=True)

    print("-"*75)
    print("Shape after --> ", df.shape)
    print("="*75, "\n")
    return df

def create_diversions_df():
    """
    Import preprocessed data, simplify & create diversions DataFrame
    """
    # Preprocessed DataFrame (full with all features for evaluation)
    df_utc = pd.read_pickle('../data/preprocessed_data/df_utc_weather.pkl')

    # Remove more features for simplicity & add "Diversion Label" (Diverted = 1, Not Diverted = 0)
    real_div_notdiv_with_relational = simplify(df_utc, "df_utc")
    #===========================================================================
    # Preprocessed DataFrame (without relational features to train gen. models)
    df_utc_durations_2 = pd.read_pickle('../data/preprocessed_data/df_utc_durations_2.pkl')
    
    # Remove more features for simplicity & add "Diversion Label" (Diverted = 1, Not Diverted = 0)
    df_utc_durations_2_simplified = simplify(df_utc_durations_2, "df_utc_durations_2")

    # Get only diverted flights
    real_div_no_relational = df_utc_durations_2_simplified[df_utc_durations_2_simplified["Diversion Label"] == 1].reset_index(drop=True)
    
    # Add relational features and clean invalid routes if they exist
    real_div_with_relational = add_relatioanal_features_clean_inv_routes(real_div_no_relational)
    #===========================================================================
    print("="*75)
    print(f"{color.BOLD}{color.GREEN}'real_div_notdiv_with_relational' (for utility evaluation){color.END}")
    print("Shape --> ", real_div_notdiv_with_relational.shape)
    print("-"*75)
    print(f"{color.BOLD}{color.GREEN}'real_div_no_relational' (for generation & fidelity evaluation){color.END}")
    print("Shape --> ", real_div_no_relational.shape)
    print("-"*75)
    print(f"{color.BOLD}{color.GREEN}'real_div_with_relational' (for statistical & diversity evaluations){color.END}")
    print("Shape --> ", real_div_with_relational.shape)
    print("-"*75)
    print(f"{color.BOLD}{color.BLUE}Saving DataFrames in '../data/preprocessed_data/'{color.END}")
    print("="*75)

    # Save
    # Real diverted and not diverted flights (with relational features for utility evaluation)
    real_div_notdiv_with_relational.to_pickle('../data/preprocessed_data/real_div_notdiv_with_relational.pkl')
    # Only diverted flights (without relational features to train the synthesizer & evaluate fidelity)
    real_div_no_relational.to_pickle('../data/preprocessed_data/real_div_no_relational.pkl')
    # Only diverted flights (with relational features for statistical & diversity evaluations)
    real_div_with_relational.to_pickle('../data/preprocessed_data/real_div_with_relational.pkl')


# def clean_inv_routes(synthetic_data):
#     """
#     Post-generation cleaning (removing invalid routes from synthetic data)
#     Invalid routes are those that have null distance (routes do not exist in historical data).
#     """
#     # DataFrame with unique pairs of airport IDs and corresponding distances
#     df_distance = pd.read_pickle('../data/preprocessed_data/df_distance.pkl')

#     # Merge synthetic_data DataFrame with df_distance to add the "Distance (miles)" column
#     synthetic_data = synthetic_data.merge(df_distance, on=["Origin Airport ID", "Destination Airport ID"], how="left")

#     # tot_syn = synthetic_data.shape[0]

#     # Removing invalid routes
#     synthetic_data = synthetic_data.dropna(subset=["Distance (miles)"]).reset_index(drop=True)

#     # valid_syn = synthetic_data.shape[0]
#     # valid_ratio = round((valid_syn / tot_syn) * 100, 1)
    
#     # Remove "Distance (miles)" column
#     synthetic_data = synthetic_data.drop(columns=["Distance (miles)"])

#     # print("-"*75, "\nPost-generation cleaning (removing invalid routes from synthetic data):")
#     # print("   Number of synthetic flights before cleaning = {}".format(tot_syn))
#     # print(f"   Number of synthetic flights after cleaning = {valid_syn} {color.BOLD}{color.RED}(valid samples {valid_ratio}%){color.END}")

#     # Remove invalid routes (those with no distance)
#     return synthetic_data

def add_relatioanal_features_clean_inv_routes(flight_information):
    """
    This function:
    - Adds relational features to the synthetic flight information (removed from df_utc_durations_2).
    - Removes invalid routes (routes that do not exist in historical data).
    
    The final df to be used for utility evaluation.
    """
    # Working on copies to avoid modifying the original DataFrames
    # Ensuring original DataFrames passed into the function remain unchanged outside the function scope.
    flight_information = flight_information.copy()

    # Shape before adding relational features
    before = flight_information.shape

    # Load full DataFrame (to extract airport information)
    df_utc = pd.read_pickle('../data/preprocessed_data/df_utc_weather.pkl')
    
    # DataFrame with unique pairs of airport IDs and corresponding distances
    df_distance = pd.read_pickle('../data/preprocessed_data/df_distance.pkl')

    #========================("Quarter", "Day of Week")=========================
    # Create "Quarter" and "Day of Week" (Monday=1, Sunday=7)
    flight_information["Quarter"] = flight_information["Scheduled Departure Time UTC"].dt.quarter
    flight_information["Day of Week"] = flight_information["Scheduled Departure Time UTC"].dt.weekday + 1

    #===========================("Distance (miles)")============================
    # Merge flight_information DataFrame with df_distance to add the "Distance (miles)" column
    flight_information = flight_information.merge(df_distance, on=["Origin Airport ID", "Destination Airport ID"], how="left")

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
    flight_information = flight_information.merge(
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
    flight_information = flight_information.merge(
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
    flight_information["Departure Delay Label"] = flight_information["Departure ΔT (min)"].apply(lambda x: 1.0 if pd.notna(x) and x >= 15 else (0.0 if pd.notna(x) else np.nan))
    # 2. Wheels Off Time UTC
    flight_information["Wheels Off Time UTC"] = flight_information["Actual Departure Time UTC"] + pd.to_timedelta(flight_information["Taxi Out Time (min)"], unit="m")
    # 3. Wheels On Time UTC
    flight_information["Wheels On Time UTC"] = flight_information["Wheels Off Time UTC"] + pd.to_timedelta(flight_information["Air Time (min)"], unit="m")
    # 4. Scheduled Arrival Time UTC
    flight_information["Scheduled Arrival Time UTC"] = flight_information["Scheduled Departure Time UTC"] + pd.to_timedelta(flight_information["Scheduled Elapsed Time (min)"], unit="m")
    # 5. Actual Arrival Time UTC
    flight_information["Actual Arrival Time UTC"] = flight_information["Actual Departure Time UTC"] + pd.to_timedelta(flight_information["Actual Elapsed Time (min)"], unit="m")
    # 6. Arrival Delay Label (NaN when "Arrival ΔT (min)" = NaN)
    flight_information["Arrival Delay Label"] = flight_information["Arrival ΔT (min)"].apply(lambda x: 1.0 if pd.notna(x) and x >= 15 else (0.0 if pd.notna(x) else np.nan))

    #=============================(Sorting columns)=============================
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
    
    flight_information = flight_information[sorted_features[:-23]]

    #==============================(Print shapes)===============================
    # print("-" * 75, f"\nCalculating the relational features ...")
    # after = flight_information.shape
    # print("   Shape before --> ", before)
    # print("   Shape after ---> ", after)

    #========================(Post-generation cleaning)=========================
    # tot_syn = flight_information.shape[0]

    # Removing invalid routes that have no distance
    flight_information = flight_information.dropna(subset=["Distance (miles)"]).reset_index(drop=True)

    # valid_syn = flight_information.shape[0]
    # valid_ratio = round((valid_syn / tot_syn) * 100, 1)

    # print("-"*75, "\nPost-generation cleaning (removing invalid routes from synthetic data):")
    # print("   Number of synthetic flights before cleaning = {}".format(tot_syn))
    # print(f"   Number of synthetic flights after cleaning = {valid_syn} {color.BOLD}{color.RED}(valid samples {valid_ratio}%){color.END}")

    #===========================================================================
    return flight_information

def prepare_for_prediction(real, synthetic, target_column=None):
    """ 
    Encode categorical and datetime features in the real and synthetic DataFrames.
    If target_column is specified, remove columns that are not needed when predicting the target_column.
    """
    # Working on copies to avoid modifying the original DataFrames
    # Ensuring original DataFrames passed into the function remain unchanged outside the function scope.
    real = real.copy()
    synthetic = synthetic.copy()
    
    # Encode categorical featres: ordinal encoding (to handel unseen categories)
    categorical_columns = [
        "Unique Carrier Code", "Tail Number", 
        "Origin Airport ID", "ICAO Origin Airport",
        "Destination Airport ID", "ICAO Destination Airport"
    ]
    for col in categorical_columns:
        if col in real.columns and col in synthetic.columns:
            # NEW ENCODER for each column
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # Fit the encoder on real data (reshape column to 2D)
            real[col] = encoder.fit_transform(real[[col]])
            # Transform the synthetic dataset using the same encoding
            synthetic[col] = encoder.transform(synthetic[[col]]) #.flatten()
 
    #==============================================================================
    # Encode datetime (convert timestamps to seconds since the Unix epoch)
    datetime_columns= real.select_dtypes(include=['datetime64[ns, UTC]']).columns.tolist()

    for col in datetime_columns:
        real[col] = pd.to_datetime(real[col]).dt.tz_convert(None).astype('int64') // 10**9
        synthetic[col] = pd.to_datetime(synthetic[col]).dt.tz_convert(None).astype('int64') // 10**9
    #==============================================================================
    # Only if target_column is specified
    # Remove columns that are not needed when predicting the diversion label
    if target_column == "Diversion Label":
        remove = [
            "Origin Airport ID", "Origin City", "Origin State Code",
            "Origin State Name", "Destination Airport ID", "Destination City",
            "Destination State Code", "Destination State Name",
            "Departure Delay Label", "Wheels On Time UTC", "Taxi In Time (min)",
            "Actual Arrival Time UTC", "Arrival ΔT (min)", "Arrival Delay Label",
            "Actual Elapsed Time (min)", "Air Time (min)"
        ]
        real = real.drop(columns=remove)
        synthetic = synthetic.drop(columns=remove)
    #==============================================================================
    # Check for non-numeric columns
    real_non_numeric = real.select_dtypes(exclude=[np.number]).columns.tolist()
    synthetic_non_numeric = synthetic.select_dtypes(exclude=[np.number]).columns.tolist()

    if real_non_numeric or synthetic_non_numeric:
        raise ValueError(
            f"\nEncoding error --> there are still some non-numeric columns:\n"
            f" - Real data: {real_non_numeric}\n"
            f" - Synthetic data: {synthetic_non_numeric}"
        )
  
    return real, synthetic
# Trains and saves various synthesizers with default hyperparameters.
# Import necessary libraries
import argparse
import sys
import numpy as np
import pandas as pd
from utils import color, set_seed, create_diversions_df
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

if __name__ == "__main__":
    if len(sys.argv) > 1 and not any(arg.startswith('--') for arg in sys.argv[1:]):
        # Case 1: run via Jupyter with with %run train.py gc tvae
        synth_types = sys.argv[1:]
        # Validate the synth_types
        for synth_type in synth_types:
            if synth_type not in SYNTHESIZER_CLASSES.keys():
                raise ValueError(f"Invalid synth_type: {synth_type}. Choose from {list(SYNTHESIZER_CLASSES.keys())}")
    else:
        # Case 2: execute the script with command line arguments like --synth_type ...
        parser = argparse.ArgumentParser()
        parser.add_argument('--synth_types', type=str, nargs='+', required=True, choices=SYNTHESIZER_CLASSES.keys())
        args = parser.parse_args()
        synth_types = args.synth_types

# Real flight information (e.g. real diversions)
# Only diverted flights (without relational features to train the synthesizer & evaluate fidelity)
real_div_no_relational = pd.read_pickle('../data/preprocessed_data/real_div_no_relational.pkl')

# Metadata (without relational features to train the synthesizer & evaluate fidelity)
metadata_no_relational = Metadata.load_from_json('../data/preprocessed_data/metadata_no_relational.json')
#=======================================================================
# Train Synthesizers
#=======================================================================

# Fixing randomness for reproducibility
set_seed(45)

# Train one or multiple generative models
for synth_type in synth_types:
    if synth_type not in SYNTHESIZER_CLASSES:
        raise ValueError(f"Invalid synthesizer type: '{synth_type}'. Choose from {list(SYNTHESIZER_CLASSES.keys())}")
    SynthClass = SYNTHESIZER_CLASSES[synth_type]

    if synth_type == 'gc':
        # Gaussian Copula Synthesizer
        print(f"{color.BOLD}{color.GREEN}Training:{color.END}")
        # Train
        print("   Training the default Gaussian Copula Synthesizer...")
        synthesizer = SynthClass(metadata_no_relational, default_distribution='gaussian_kde')
        synthesizer.fit(data=real_div_no_relational)
        # Save
        print("   Saving trained synthesizer...")
        synthesizer_path = '../data/outputs/synthesizers/synthesizer_GC_diversions_default.pkl'
        synthesizer.save(filepath=synthesizer_path)
    elif synth_type == 'tvae':
        # TVAE Synthesizer
        print(f"{color.BOLD}{color.GREEN}Training:{color.END}")
        # Train
        print("   Training the default TVAE Synthesizer...")
        synthesizer = SynthClass(metadata_no_relational, verbose=True)
        synthesizer.fit(data=real_div_no_relational)
        # Save
        print("   Saving trained synthesizer...")
        synthesizer_path = '../data/outputs/synthesizers/synthesizer_TVAE_diversions_default.pkl'
        synthesizer.save(filepath=synthesizer_path) 
    elif synth_type == 'ctgan':
        # CTGAN Synthesizer
        print(f"{color.BOLD}{color.GREEN}Training:{color.END}")
        # Train
        print("   Training the default CTGAN Synthesizer...")
        synthesizer = SynthClass(metadata_no_relational, verbose=True)
        synthesizer.fit(data=real_div_no_relational)
        # Save
        print("   Saving trained synthesizer...")
        synthesizer_path = '../data/outputs/synthesizers/synthesizer_CTGAN_diversions_default.pkl'
        synthesizer.save(filepath=synthesizer_path)
    elif synth_type == 'copgan':
        # CopulaGAN Synthesizer
        print(f"{color.BOLD}{color.GREEN}Training:{color.END}")
        # Train
        print("   Training the default CopulaGAN Synthesizer...")
        synthesizer = SynthClass(metadata_no_relational, default_distribution='gaussian_kde', verbose=True)
        synthesizer.fit(data=real_div_no_relational)
        # Save
        print("   Saving trained synthesizer...")
        synthesizer_path = '../data/outputs/synthesizers/synthesizer_COPGAN_diversions_default.pkl'
        synthesizer.save(filepath=synthesizer_path)
print("="*75)

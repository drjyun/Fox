import os
import numpy as np
import mne
from mne.preprocessing import ICA
from scipy.stats import zscore
import pandas as pd

HIGHPASS = 0.3  # Low cutoff 
LOWPASS = 50.0  # High cutoff

# Function to load EEG data from an XDF file
def load_eeg_data(file_path):
    from pyxdf import load_xdf
    
    # Load the XDF file
    data, header = load_xdf(file_path)

    # Find the EEG stream
    eeg_stream = next((stream for stream in data if stream['info']['type'][0] == 'EEG'), None)
    if not eeg_stream:
        raise ValueError('No EEG stream found')

    # Extract data and channel names
    eeg_data = eeg_stream['time_series'][:,:4].T # Only use the first 4 channels
    channel_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
    sfreq = float(eeg_stream['info']['nominal_srate'][0])

    # Create Info object and Raw object
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)

    return raw, sfreq

# Main script
if __name__ == "__main__":
    # Define file paths and load data
    BIDS_ROOT = os.path.join("../")
    DERIV_ROOT = os.path.join(BIDS_ROOT, 'derivatives')
    file_path = os.path.join(BIDS_ROOT, 'sub-001', 'sub-001_ses-S001_task-Default_run-001_eeg.xdf')
    csv_path = os.path.join(BIDS_ROOT, 'sub-001', 'sub-001_task-events.csv')
    raw, sfreq = load_eeg_data(file_path)

    events_df = pd.read_csv(csv_path) # Load the events from the CSV file

    # Set montage and filter
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    raw.filter(HIGHPASS, LOWPASS)

    for index, row in events_df.iterrows():
        onset = row['onset'] / sfreq  # Convert from samples to seconds
        duration = row['duration'] / sfreq  # Convert from samples to seconds
        event_id = {'event': 1}

        # Create an Annotations object for the event
        annotations = mne.Annotations(onset=[onset], duration=[duration], description=['event'])
        raw_temp = raw.copy().set_annotations(annotations)
        events, event_id_map = mne.events_from_annotations(raw_temp, event_id=event_id)

        # Create the Epochs for the event BEFORE ICA
        epochs = mne.Epochs(raw_temp, events, event_id=event_id_map, tmin=-0.5, 
                            tmax=duration, baseline=None, preload=True)

        # Perform ICA to find and remove artifacts on the individual epochs
        ica = ICA(n_components=2, random_state=97)  
        ica.fit(epochs)
        ica.plot_components()

        # Use z-scores to determine which components to exclude
        ica_data = ica.get_sources(epochs).get_data()  # Get the ICA components' time series
        z_scores = zscore(ica_data, axis=1)  # Compute z-scores of ICA components
        z_threshold = 1.96
        ica.exclude = np.where((np.abs(z_scores) > z_threshold).any(axis=1))[0]

        # Apply ICA cleaning to the individual epochs
        epochs_clean = ica.apply(epochs.copy())
        epochs_clean.apply_baseline((-0.5, 0)) # Baseline correction AFTER ICA

        # Save the cleaned epochs
        preprocessing_dir = os.path.join(DERIV_ROOT, 'preprocessing/sub-001')
        os.makedirs(preprocessing_dir, exist_ok=True)
        cleaned_fname = os.path.join(DERIV_ROOT, 'preprocessing/sub-001', f'sub-001_event-{index+1}_epo.fif')
        epochs_clean.save(cleaned_fname, overwrite=True)

        print(f"Cleaned epochs saved to: {cleaned_fname}")

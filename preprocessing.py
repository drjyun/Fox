import os
import numpy as np
import mne
from mne.preprocessing import ICA
from scipy.stats import zscore

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

    # Find the marker stream
    marker_stream = next((stream for stream in data if stream['info']['type'][0] == 'Markers'), None)
    if not marker_stream:
        raise ValueError('No marker stream found')
    
    # Prepare marker info
    markers = marker_stream['time_series']
    marker_info = [(marker[0], timestamp) for marker, timestamp in zip(markers, marker_stream['time_stamps'])]
    print(f'Found {len(marker_info)} markers')

    # Extract data and channel names
    # eeg_data = eeg_stream['time_series'].T
    eeg_data = eeg_stream['time_series'][:,:4].T # Only use the first 4 channels
    channel_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
    sfreq = float(eeg_stream['info']['nominal_srate'][0])

    # Create Info object and Raw object
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)

    return raw, marker_info

# Main script
if __name__ == "__main__":
    # Define file path and load data
    file_path = 'sub-003/sub-003_ses-S001_task-Default_run-001_eeg.xdf'
    raw, marker_info = load_eeg_data(file_path)

    # # Drop the channels not needed BEFORE fitting ICA
    # channels_to_drop = ['T5', 'T6', 'O1', 'O2']
    # raw.drop_channels(channels_to_drop)
    # existing_channels_to_drop = [ch for ch in channels_to_drop if ch in raw.info['ch_names']]
    # print('Channels to drop:', existing_channels_to_drop)

    # if existing_channels_to_drop:
    #     raw.drop_channels(existing_channels_to_drop)
    # print('Channel names after dropping:', raw.info['ch_names'])

    # Optionally, set a montage (source localizations or topo maps)
    # channel_coords_meters = {
    #     'Fp1': (0.808, 0.261, -0.04),  
    #     'Fp2': (0.808, -0.261, -0.04), 
    #     'AF7': (0.687, 0.497, -0.0596), 
    #     'AF8': (0.687, -0.497, -0.0596),
    # }

    # montage = mne.channels.make_dig_montage(ch_pos=channel_coords_meters, coord_frame='head')
    # raw.set_montage(montage)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)

    # Filter the raw data
    raw.filter(HIGHPASS, LOWPASS)

    ## Note that we're not extracting "epochs" before ICA due to small numbers of channels ##
    # Perform ICA to find and remove artifacts
    ica = ICA(n_components=2, random_state=0) # or n_components=3 depending on what data looks like
    ica.fit(raw)
    ica.plot_components()

    ## Note that automatic algorithms like MARA or RANSAC is less appropriate for 4-channel EEG system ##
    ica_data = ica.get_sources(raw).get_data()# Get the source data (the ICA components' time series)

    z_scores = zscore(ica_data, axis=1) # Compute the z-scores of the ICA components

    # Use a z-score threshold to identify components to exclude
    z_threshold = 1.96
    ica.exclude.extend(np.where((np.abs(z_scores) > z_threshold).any(axis=1))[0])

    raw_ica = ica.apply(raw.copy())

    # Save the cleaned data
    raw_ica.save('sub-003/sub-003_eeg_cleaned.fif', overwrite=True)

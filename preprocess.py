import os, re, mne
import numpy as np
import pandas as pd
from pyxdf import load_xdf
from mne.preprocessing import ICA
from scipy.stats import zscore

HIGHPASS = 0.3  # Low cutoff 
LOWPASS = 50.0  # High cutoff 
Z_THRESHOLD = 1.96  # Threshold for z-score to exclude ICA components

def load_eeg_data(file_path, channel_limit=4):
    data, _ = load_xdf(file_path)
    eeg_stream = next((s for s in data if s['info']['type'][0] == 'EEG'), None)
    if eeg_stream is None:
        raise ValueError('No EEG stream found in file: ' + file_path)

    eeg_data = eeg_stream['time_series'][:, :channel_limit].T
    channel_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel'][:channel_limit]]
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
    return mne.io.RawArray(eeg_data, info), sfreq

def preprocess_subject(subject_id, task, bids_root, deriv_root):
    subject_dir = os.path.join(bids_root, f'sub-{subject_id}_{task}')
    xdf_files = [f for f in os.listdir(subject_dir) if f.endswith('.xdf')]

    for file_name in xdf_files:
        file_path = os.path.join(subject_dir, file_name)
        raw, sfreq = load_eeg_data(file_path)
        csv_path = os.path.join(subject_dir, f'sub-{subject_id}_task-events.csv')
        events_df = pd.read_csv(csv_path)
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'), match_case=False)
        raw.filter(HIGHPASS, LOWPASS)
        preprocess_events(raw, sfreq, events_df, subject_id, condition, deriv_root)

def preprocess_events(raw, sfreq, events_df, subject_id, condition, deriv_root):
    for index, row in events_df.iterrows():
        onset = row['onset'] / sfreq
        duration = row['duration'] / sfreq
        annotations = mne.Annotations(onset=[onset], duration=[duration], description=['event'])
        raw_temp = raw.copy().set_annotations(annotations)
        events, event_id_map = mne.events_from_annotations(raw_temp, event_id={'event': 1})

        epochs = mne.Epochs(raw_temp, events, event_id=event_id_map, tmin=-0.5, tmax=duration, 
                            baseline=None, preload=True)
        ica = ICA(n_components=2, random_state=97)
        ica.fit(epochs)
        #ica.plot_components()
        ica_data = ica.get_sources(epochs).get_data()

        z_scores = zscore(ica_data, axis=1)
        ica.exclude = np.where((np.abs(z_scores) > Z_THRESHOLD).any(axis=1))[0]
        epochs_clean = ica.apply(epochs.copy()).apply_baseline((-0.5, 0)) # AFTER ICA

        preprocessing_dir = os.path.join(deriv_root, 'preprocessing', f'sub-{subject_id}_{condition}')
        os.makedirs(preprocessing_dir, exist_ok=True)
        cleaned_fname = os.path.join(preprocessing_dir, f'sub-{subject_id}_{condition}_event-{index+1}_epo.fif')
        epochs_clean.save(cleaned_fname, overwrite=True)
        print(f"Cleaned epochs saved to: {cleaned_fname}")

if __name__ == "__main__":
    BIDS_ROOT = "../"
    DERIV_ROOT = os.path.join(BIDS_ROOT, 'derivatives')
    condition = 'bigmood'  ## CONDITION ##
    subject_folders = [d for d in os.listdir(BIDS_ROOT) if d.startswith('sub-') and condition in d]

    for subject_folder in subject_folders:
        subject_match = re.match(r'sub-(\d{3})_(.*)', subject_folder)
        if subject_match:
            subject_id = subject_match.group(1)
            task = subject_match.group(2)

            try:
                preprocess_subject(subject_id, task, BIDS_ROOT, DERIV_ROOT)
            except FileNotFoundError as e:
                print(e)


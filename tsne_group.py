import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pyxdf
from mne.time_frequency import tfr_morlet
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Define the function outside the loop
# Define the function to process data, now including event extraction based on markers
def process_data_to_alpha_power(raw_cleaned, markers, timestamps, first_timestamp):
    sfreq = raw_cleaned.info['sfreq']  # Sampling frequency

    # Convert marker onset times to samples relative to the first EEG data timestamp
    events = np.array([
        [int((timestamp - first_timestamp) * sfreq), 0, int(marker[0])]
        for marker, timestamp in zip(markers, timestamps) if marker[0].isdigit()  # Ensure marker is a digit
    ])

    # Create epochs
    epochs = mne.Epochs(raw_cleaned, events=events, event_id=None, tmin=-0.2, tmax=10.0, preload=True, reject=None,
                        baseline=(-0.2, 0))

    # Your existing code for frequency analysis and power computation
    alpha_freqs = np.arange(8, 13, 1)
    power = tfr_morlet(epochs, freqs=alpha_freqs, n_cycles=alpha_freqs / 2, return_itc=False, average=False)
    power.apply_baseline((-0.2, 0), mode='logratio')

    af7_index = raw_cleaned.ch_names.index('AF7')
    af8_index = raw_cleaned.ch_names.index('AF8')
    alpha_power_af7 = power.data[:, af7_index, :, :].mean(axis=1)
    alpha_power_af8 = power.data[:, af8_index, :, :].mean(axis=1)

    return alpha_power_af7, alpha_power_af8


tsne_features_list, colors_subject, colors_epoch = [], [], []


for subject_num in [4,6,8]:
    file_path = os.path.join(os.getcwd(), f'sub-00{subject_num}', f'sub-00{subject_num}_ses-S001_task-Default_run-001_eeg.xdf')
    cleaned_eeg_path = os.path.join(f'sub-00{subject_num}', f'sub-00{subject_num}_ses-S001_task-Default_run-001_eeg_cleaned.fif')

    # Load XDF file and extract markers and timestamps
    data, header = pyxdf.load_xdf(file_path)
    markers, timestamps = [], []
    first_timestamp = None
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            markers = stream['time_series']
            timestamps = stream['time_stamps']
        if stream['info']['type'][0] == 'EEG' and first_timestamp is None:
            first_timestamp = stream['time_stamps'][0]

    raw_cleaned = mne.io.read_raw_fif(cleaned_eeg_path, preload=True)

    # Check for the presence of markers and timestamps
    if not markers or first_timestamp is None:
        print("No markers found or first_timestamp not set. Skipping subject.")
        continue

    alpha_power_af7, alpha_power_af8 = process_data_to_alpha_power(raw_cleaned, markers, timestamps, first_timestamp)
    combined_alpha_power = alpha_power_af7 - alpha_power_af8

    scaler = StandardScaler()
    scaled_alpha_power = scaler.fit_transform(combined_alpha_power)

    tsne = TSNE(n_components=2, perplexity=3, random_state=42)
    tsne_features = tsne.fit_transform(scaled_alpha_power)

    tsne_features_list.append(tsne_features)
    colors_subject += [subject_num] * len(tsne_features)
    colors_epoch += list(range(len(tsne_features)))

# Now, you can concatenate all tsne_features and proceed with plotting as you have in your script.
all_tsne_features = np.vstack(tsne_features_list)
colors_subject = np.array(colors_subject)
colors_epoch = np.array(colors_epoch)

# Define subject labels
subject_labels = {4: 'CNN NEWS', 6: 'FOX NEWS', 8: 'Control Condition'}

# Plotting t-SNE features color-coded by subject
plt.figure(figsize=(10, 8))
for i in [4, 6, 8]:  # Loop through subjects 4, 6, 8
    plt.scatter(all_tsne_features[colors_subject == i, 0], all_tsne_features[colors_subject == i, 1],
                label=subject_labels.get(i, f'Subject {i}'))
plt.title('t-SNE by Subject')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.savefig('tsne_by_subject2D_468.png')
plt.show()

# Define epoch labels
epoch_labels = {0: 'LEGO', 1: 'Burger King', 2: 'Chevrolet', 3: 'Mounjaro',
                4: 'Paramount', 5: 'LEGO', 6: 'Rakuten', 7: 'Redbull', 8: 'Samsung', 9: 'Tracfone'}

# Plotting t-SNE features color-coded by epoch
plt.figure(figsize=(10, 8))
num_epochs = np.max(colors_epoch) + 1  # Calculate the total number of epochs
cmap = plt.get_cmap("tab10")  # Get a colormap to color epochs (adjust if more than 10 epochs)
for i in range(num_epochs):
    plt.scatter(all_tsne_features[colors_epoch == i, 0], all_tsne_features[colors_epoch == i, 1],
                color=cmap(i % 10), label=f'{epoch_labels.get(i, f"{i}")}')
plt.title('t-SNE by Epoch')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.savefig('tsne_by_epoch2D_468.png')
plt.show()



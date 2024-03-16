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


for subject_num in range(1, 9):
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

    tsne = TSNE(n_components=3, perplexity=3, random_state=42)
    tsne_features = tsne.fit_transform(scaled_alpha_power)

    tsne_features_list.append(tsne_features)
    colors_subject += [subject_num] * len(tsne_features)
    colors_epoch += list(range(len(tsne_features)))

# Concatenate all tsne_features for 3D plotting
all_tsne_features = np.vstack(tsne_features_list)
colors_subject = np.array(colors_subject)
colors_epoch = np.array(colors_epoch)

# 3D Plotting t-SNE features color-coded by subject
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(1, 9):  # Loop through subjects 1 to 8
    idx = colors_subject == i
    ax.scatter(all_tsne_features[idx, 0], all_tsne_features[idx, 1], all_tsne_features[idx, 2], label=f'Subject {i}')
ax.set_title('3D t-SNE by Subject')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
plt.legend()
plt.savefig('tsne_by_subject3D.png')
plt.show()

# Optionally, 3D Plotting t-SNE features color-coded by epoch
# This approach might become visually cluttered for large numbers of epochs.
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Maximum number of distinct colors available for plotting
max_colors = plt.cm.tab10.colors  # Using the tab10 colormap for up to 10 distinct colors
num_epochs = np.unique(colors_epoch).size

# Plot each epoch with a different color
for epoch in range(num_epochs):
    idx = colors_epoch == epoch
    color = max_colors[epoch % len(max_colors)]  # Cycle through colors if more than 10 epochs
    ax.scatter(all_tsne_features[idx, 0], all_tsne_features[idx, 1], all_tsne_features[idx, 2], label=f'Epoch {epoch + 1}', color=color)

ax.set_title('3D t-SNE by Epoch')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')

# Optional: to avoid clutter, only add a legend if the number of epochs is manageable
if num_epochs <= 10:
    plt.legend()
plt.savefig('tsne_by_epoch3D.png')
plt.show()
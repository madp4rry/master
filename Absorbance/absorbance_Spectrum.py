import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- Matplotlib style ---
plt.style.use('seaborn-v0_8-poster')
from matplotlib import rcParams
rcParams['font.family'] = 'Rubik'
rcParams['axes.labelsize'] = 30
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 22
plt.rcParams['lines.markersize'] = 12


# Load the CSV file
data_dir = 'C:/Users/patri/Coding/Master/Absorbance/'
data = pd.read_csv(data_dir + 'PS1_DETN.csv', sep=';', decimal=',')

# Extract x-axis (first column) and y-axis (remaining columns)
wavelengths = data.iloc[:, 0]  # First column as x-axis
samples = data.iloc[:, 1:]     # Remaining columns as y-axis

# Filter and smooth the data
mask = (wavelengths >= 380) & (wavelengths <= 455)
samples_filtered = samples.copy()
samples_filtered.loc[mask, :] = samples.loc[mask, :].where(samples.loc[mask, :] <= 4.5)
samples_filtered.loc[mask, :] = samples_filtered.loc[mask, :].rolling(window=40, min_periods=1).mean()

# Normalize the data
samples_normalized = samples_filtered / samples_filtered.max().max()

# Calculate means and standard deviationsdd
o_kcl_mean = samples_normalized.iloc[:, 0:2].mean(axis=1)
o_kcl_std = samples_normalized.iloc[:, 0:2].std(axis=1)

kcl_400_mean = samples_normalized.iloc[:, 2:4].mean(axis=1)
kcl_400_std = samples_normalized.iloc[:, 2:4].std(axis=1)

kcl_800_mean = samples_normalized.iloc[:, 4:6].mean(axis=1)
kcl_800_std = samples_normalized.iloc[:, 4:6].std(axis=1)

# Plot the data
plt.figure(figsize=(12, 9))

# Plot O KCL
plt.plot(wavelengths, o_kcl_mean, label='O KCL', color='blue')
plt.fill_between(wavelengths, o_kcl_mean - o_kcl_std, o_kcl_mean + o_kcl_std, color='blue', alpha=0.2)

# Plot 400 KCL
plt.plot(wavelengths, kcl_400_mean, label='400 KCL', color='green')
plt.fill_between(wavelengths, kcl_400_mean - kcl_400_std, kcl_400_mean + kcl_400_std, color='green', alpha=0.2)

# Plot 800 KCL
plt.plot(wavelengths, kcl_800_mean, label='800 KCL', color='red')
plt.fill_between(wavelengths, kcl_800_mean - kcl_800_std, kcl_800_mean + kcl_800_std, color='red', alpha=0.2)

# Add labels, legend, and grid
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative Absorbance')
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.3)

# Show the plot
plt.tight_layout()
plt.savefig('C:/Users/patri/Coding/Master/Absorbance/PS1_DETN_plot.png', dpi=300, bbox_inches='tight')
plt.show()



def plot_o_kcl_and_mean_normalized_current_drop_with_error(data, wavelengths, file1, file2):
    """
    Plots the normalized O KCL data and mean normalized current drop values with error bars.

    Args:
        data (DataFrame): Input absorbance data containing O KCL columns.
        wavelengths (Series): Wavelengths corresponding to the absorbance data.
        file1 (str): Path to the first normalized current drop CSV file.
        file2 (str): Path to the second normalized current drop CSV file.
    """
    # Normalize O KCL data to 1
    o_kcl_mean = data.iloc[:, 0:2].mean(axis=1)
    o_kcl_normalized = o_kcl_mean / o_kcl_mean.max()

    # Load the data from both files
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Merge the data on the 'Wavelength (nm)' column
    merged_data = pd.merge(data1, data2, on='Wavelength (nm)', suffixes=('_file1', '_file2'))

    # Compute the mean and standard error of normalized current drop values
    merged_data['Mean Normalized Current Drop'] = merged_data[['Normalized Current Drop_file1', 'Normalized Current Drop_file2']].mean(axis=1)
    merged_data['Standard Error'] = merged_data[['Normalized Current Drop_file1', 'Normalized Current Drop_file2']].sem(axis=1)

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(18, 8))

    # Plot normalized O KCL on the primary y-axis
    ax1.plot(wavelengths, o_kcl_normalized, label='Normalized O KCL', color='green', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Normalized O KCL', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    # Create a secondary y-axis for Mean Normalized Current Drop with error bars
    ax2 = ax1.twinx()
    ax2.errorbar(merged_data['Wavelength (nm)'], merged_data['Mean Normalized Current Drop'],
                 yerr=merged_data['Standard Error'], fmt='o', color='black', label='Mean Normalized Current Drop',capsize=5, elinewidth=1, capthick=1, alpha=0.7)
    ax2.set_ylabel('Mean Normalized Current Drop', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add labels, legend, and grid
    plt.title('Normalized O KCL and Mean Normalized Current Drop')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
file1 = 'C:/Users/patri/Coding/Master/Characterization/normalized_current_drop.csv'
file2 = 'C:/Users/patri/Coding/Master/Characterization/normalized_current_drop1.csv'
plot_o_kcl_and_mean_normalized_current_drop_with_error(samples_normalized, wavelengths, file1, file2)
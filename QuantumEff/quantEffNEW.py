from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import glob
import os
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def in_mikroamper(df, column_name):
    df_adj = df.copy()
    for _ in range(3):
        df_adj[column_name] = df_adj[column_name] * 100
    return df_adj


directory = r'C:\Users\patri\Coding\Master\Characterization\QuantumEff'
file_paths = glob.glob(os.path.join(directory, '*.txt'))

# Constants
h = 6.626e-34  # Planck constant (J·s)
c = 3e8  # Speed of light (m/s)
e = 1.602e-19  # Elementary charge (C)
area = 0.125e-4  # Example: 0.25 cm² = 0.25e-4 m²
wavelength = 680e-9  # Example: 680 nm = 680e-9 m
surface_coverage = 18.6e-12 * 1e4  # = 2.0e-7 mol/m^2
print(surface_coverage)
absorb_spec_conc = 0.024 # in mM
#absorbance = 0.85  # Example: 85% absorption

#------------------ ABSORBANCE DATA ------------------
# Load absorbance CSV
abs_dir = r'C:\Users\patri\Coding\Master\Characterization\absorbanceRAWQE.csv'

import seaborn as sns

plt.style.use('seaborn-v0_8-poster')

rcParams['axes.labelsize'] = 30
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 22

lightsourceP =  r'C:\Users\patri\Coding\Master\Characterization\lightsource.txt'


def load_spectrum(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip first 3 lines, remove leading '›', and clean up
    cleaned_lines = [line.lstrip('›').strip() for line in lines[3:] if line.strip()]

    # Parse data
    data = [line.split() for line in cleaned_lines]
    df = pd.DataFrame(data, columns=['Wavelength_m', 'Intensity'])
    df = df.astype(float)

    # Convert wavelength to nm and scale intensity
    df['Wavelength_nm'] = df['Wavelength_m'] * 1e9
    df['Intensity'] *= 1000  # Unit conversion

    # Crop to 350–800 nm
    df = df[(df['Wavelength_nm'] >= 350) & (df['Wavelength_nm'] <= 800)]

    # Normalize intensity to area = 1 using trapezoidal integration
    area = np.trapz(df['Intensity'], df['Wavelength_nm'])
    df['Intensity'] /= df['Intensity'].sum()

    return df

def load_absorbance(
    file_path,
    concentration_mM,
    path_length_cm=1.0,
    surface_coverage=20e-12 * 1e4,  # mol/m^2  (20 pmol/cm^2)
    n_chl_per_trimer=288,           # PSI_t has 288 Chls per trimer
    epsilon_basis="per_chl",        # "per_chl" or "per_complex"
    plot=True
):
    """
    Builds ε(λ), A_surface(λ), and f_abs(λ) from a long-format CSV:
      columns: Wellenlänge, Extinktion, Mno
    IMPORTANT:
      - If your concentration is per-PSI trimer, set epsilon_basis="per_complex".
      - If your concentration is per-Chl (or you matched ε≈57e3 at 680 nm), leave epsilon_basis="per_chl".
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Load and average replicates per wavelength
    df = pd.read_csv(file_path, sep=';', decimal=',', encoding='latin1')
    grouped = df.groupby('Wellenlänge', as_index=False)['Extinktion'].mean()
    grouped.rename(columns={'Wellenlänge': 'Wavelength_nm', 'Extinktion': 'Absorbance_avg'}, inplace=True)

    # 2) ε(λ) from A = ε c l   (ε in L mol^-1 cm^-1)
    c_M = concentration_mM / 1000.0
    grouped['Epsilon_L_per_mol_cm'] = grouped['Absorbance_avg'] / (c_M * path_length_cm)

    # 3) Convert ε-basis to "per trimer" if you started from per-Chl ε
    if epsilon_basis.lower() == "per_chl":
        grouped['Epsilon_L_per_mol_cm'] *= n_chl_per_trimer
        basis_note = "ε converted: per-Chl × 288 → per-PSI_trimer"
    else:
        basis_note = "ε treated as per-PSI_trimer (no multiplication)"

    # (Optional) Print ε(680) for sanity
    eps_680 = grouped.loc[grouped['Wavelength_nm'].round()==680, 'Epsilon_L_per_mol_cm']
    if not eps_680.empty:
        print(f"[{basis_note}]  ε(680) = {eps_680.iloc[0]:.3e}  L mol^-1 cm^-1 (per PSI trimer)")

    # 4) A_surface(λ) = ε(λ) * Γ, with ε converted to m^2/mol:  (L mol^-1 cm^-1 → m^2/mol is ×0.1)
    grouped['A_surface'] = grouped['Epsilon_L_per_mol_cm'] * 0.1 * surface_coverage  # unitless

    # 5) f_abs(λ) = 1 - 10^(-A_surface)
    grouped['Fraction_absorbed'] = 1.0 - 10.0 ** (-grouped['A_surface'])

    # (optional plot with 3 y-axes omitted for brevity)
    if plot:
        fig, ax1 = plt.subplots(figsize=(15, 9))
        ax2 = ax1.twinx(); ax3 = ax1.twinx()
        l1, = ax1.plot(grouped['Wavelength_nm'], grouped['Epsilon_L_per_mol_cm'], color='orange', label='ε(λ) [L mol^-1 cm^-1]')
        l2, = ax2.plot(grouped['Wavelength_nm'], grouped['A_surface'],             color='blue',   label='A_surface(λ) [–]')
        l3, = ax3.plot(grouped['Wavelength_nm'], grouped['Fraction_absorbed'],     color='green',  label='f_abs(λ) [fraction]')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('ε(λ) [L mol⁻¹ cm⁻¹]', color='orange'); ax1.tick_params(axis='y', labelcolor='orange')
        ax1.set_ylim(0, grouped['Epsilon_L_per_mol_cm'].max() * 1.1)
        ax2.set_ylabel('A_surface(λ) [–]',     color='blue');   ax2.tick_params(axis='y', labelcolor='blue')
        ax3.set_ylabel('f_abs(λ) [fraction]',  color='green');  ax3.tick_params(axis='y', labelcolor='green')
        ax3.spines['right'].set_position(("axes", 1.14))  # Increase the value from 1.1 to 1.2
        ax1.grid(True, alpha=0.2)
        ax1.legend(handles=[l1,l2,l3], loc='upper right')
        plt.tight_layout()
        plt.savefig(r'C:\Users\patri\Coding\Master\figures\QuantumAbsorbance.png', dpi=300, bbox_inches='tight')
        plt.show()

    return grouped

def plot_raw_current(data, time_column='Time/s', current_column='Current/A', intensity_column='Intensity'):
    light_phases = []
    is_dark = data['IsDark']
    start_idx = None  # Initialize start_idx

    plt.figure(figsize=(12, 6))
    plt.scatter(data[time_column], data[current_column], label='Raw Current', color='blue', alpha=0.7)

    for start_idx, end_idx in light_phases:
        plt.axvspan(data[time_column][start_idx], data[time_column][end_idx], color='yellow', alpha=0.3)

    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.title('Raw Current Data with Light Phases')
    plt.legend()
    plt.tight_layout()
    #plt.show()


# Crop the dataset to the specified time window
def crop_data_to_time_window(data, time_column='Time/s', start_time=100, end_time=650):
    """
    Crops the dataset to include only rows within the specified time window.

    Args:
        data (DataFrame): The input dataset.
        time_column (str): The name of the time column.
        start_time (float): The start of the time window.
        end_time (float): The end of the time window.

    Returns:
        DataFrame: The cropped dataset.
    """
    cropped_data = data[(data[time_column] > start_time) & (data[time_column] < end_time)]
    return cropped_data


def baseline_correct_and_plot(
        data, current_column='Current/A', intensity_column='Intensity', time_column='Time/s', threshold=4, degree=3
):
    light_mask = data[intensity_column] > threshold
    time_light = data[time_column][light_mask]
    current_light = data[current_column][light_mask]

    valid_range_mask = (time_light > 80) & (time_light < 650)
    time_light = time_light[valid_range_mask]
    current_light = current_light[valid_range_mask]

    # Safety check
    if time_light.empty or current_light.empty:
        raise ValueError("No light-phase data found above threshold. Check 'threshold' value or input data.")

    poly_coeffs = np.polyfit(time_light, current_light, deg=degree)
    baseline = np.polyval(poly_coeffs, data[time_column])
    data['CorrectedCurrent'] = data[current_column] - baseline

    return data


def plot_absolute_corrected_current_with_light_highlight(data, time_column='Time/s',
                                                         corrected_column='CorrectedCurrent', is_dark_column='IsDark'):
    import matplotlib.pyplot as plt

    # Offset corrected current to start from 0
    corrected_min = data[corrected_column].min()
    data['AbsCorrectedCurrent'] = data[corrected_column] - corrected_min

    # Prepare light regions (where IsDark is False)
    light_regions = []
    in_light = False
    start = None

    for i in range(len(data)):
        if not data[is_dark_column].iloc[i] and not in_light:
            start = data[time_column].iloc[i]
            in_light = True
        elif data[is_dark_column].iloc[i] and in_light:
            end = data[time_column].iloc[i]
            light_regions.append((start, end))
            in_light = False
    # Handle case where light continues to end
    if in_light:
        light_regions.append((start, data[time_column].iloc[-1]))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data[time_column], data['AbsCorrectedCurrent'], color='blue', label='Absolute Corrected Current')

    # Shade light regions
    for start, end in light_regions:
        plt.axvspan(start, end, color='orange', alpha=0.2)

    plt.xlabel('Time (s)')
    plt.ylabel('Abs. Corrected Current (A)')
    plt.title('Baseline-Corrected Absolute Current with Light Phases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_EQE_IQE_from_spectrum(
        current_df,
        photo_df,
        absorb_df,  # DataFrame with columns: Wavelength_nm, Fraction_absorbed
        electrode_area=area,
        poly_degree=2
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d

    # Crop spectrum to 350–750 nm
    photo_df = photo_df[(photo_df['Wavelength_nm'] >= 350) & (photo_df['Wavelength_nm'] <= 750)].copy()
    wavelength_resolution = np.mean(np.diff(photo_df['Wavelength_nm']))


    # Constants
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 299792458       # Speed of light (m/s)
    e = 1.602176634e-19 # Elementary charge (C)

    # PSI parameters
    epsilon = 57.1  # L·mol⁻¹·cm⁻¹ (at 680 nm)


    # Energy of photon
    photo_df['photon_energy'] = h*c/photo_df['Wavelength_m']

    from scipy.interpolate import interp1d

    # Set up f_abs interpolator ONCE before the loop
    abs_interp = interp1d(
        absorb_df['Wavelength_nm'],
        absorb_df['Fraction_absorbed'],
        bounds_error=False,
        fill_value=0
    )

    EQEs = []
    IQEs = []

    for _, row in current_df.iterrows():
        intensity = row['MeanIntensity']  # in W/m²

        # Scale total intensity to spectrum
        photo_df['ScaledIntensity'] = photo_df['Intensity'] * intensity

        # Convert to photon flux
        photo_df['PhotonFlux'] = photo_df['ScaledIntensity'] / photo_df['photon_energy']

        # Total incident photons per m² per second
        total_photons = photo_df['PhotonFlux'].sum()

        # Compute EQE from photocurrent
        current = np.abs(row['MeanDrop']) * 1e-6 * 10000  # A/m²
        electrons = current / e  # 1/s/m²
        EQE = (electrons / total_photons) * 100  # %

        # --- IQE computation using interpolated f_abs ---
        photo_df['f_abs'] = abs_interp(photo_df['Wavelength_nm'])
        photo_df['AbsorbedPhotonFlux'] = photo_df['PhotonFlux'] * photo_df['f_abs']
        absorbed_photons = photo_df['AbsorbedPhotonFlux'].sum()

        # Compute IQE
        IQE = (electrons / absorbed_photons) * 100 if absorbed_photons > 0 else 0

        # Append results
        EQEs.append(EQE)
        IQEs.append(IQE)

    current_df['EQE'] = EQEs
    print("EQE values:", EQEs)
    current_df['IQE'] = IQEs
    print("IQE values:", IQEs)


    # Fit curves
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    def log_growth(x, a, b, c):
        return a * np.log(np.maximum(b * x + 1, 1e-9)) + c

    xdata = current_df['MeanIntensity'].values
    y_eqe = current_df['EQE'].values
    y_iqe = current_df['IQE'].values
    y_current = current_df['MeanDrop'].values

    try:
        popt_eqe, _ = curve_fit(exp_decay, xdata, y_eqe, p0=(max(y_eqe), 0.01, min(y_eqe)))
        popt_iqe, _ = curve_fit(exp_decay, xdata, y_iqe, p0=(max(y_iqe), 0.01, min(y_iqe)))
    except RuntimeError:
        popt_eqe = (0, 0, np.mean(y_eqe))
        popt_iqe = (0, 0, np.mean(y_iqe))

    try:
        sigma = np.ones_like(xdata)
        sigma[xdata > 40] = 0.97
        popt_curr, _ = curve_fit(log_growth, xdata, np.abs(y_current), p0=[1, 0.1, 0], sigma=sigma, absolute_sigma=True)
    except RuntimeError:
        popt_curr = (1, 0.1, 0)

    x_fit = np.linspace(min(xdata), max(xdata), 200)
    y_fit_eqe = exp_decay(x_fit, *popt_eqe)
    y_fit_iqe = exp_decay(x_fit, *popt_iqe)
    y_fit_curr = -log_growth(x_fit, *popt_curr)

    # Plot
    fig, ax1 = plt.subplots(figsize=(19, 10))
    ax1.set_xlabel('Mean Intensity (W/m²)')
    ax1.set_ylabel('EQE / IQE (%)',)
    ax1.scatter(xdata, y_eqe, color='blue', label='EQE data')
    ax1.scatter(xdata, y_iqe, color='purple', label='IQE data')
    ax1.plot(x_fit, y_fit_eqe, 'b--', alpha= 0.4)
    ax1.plot(x_fit, y_fit_iqe , color='purple', linestyle='--', alpha = 0.4)
    # ax1.scatter(xdata, y_iqe, color='purple', marker='x', label='IQE data')
    ax1.legend(loc='upper right' ,bbox_to_anchor=(1, 0.85) ) # Shift legend slightly)
    ax1.grid(True, alpha=0.15)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Photocurrent (µA)', color='green')
    ax2.scatter(xdata, y_current, color='green', label='Photocurrent data')
    ax2.plot(x_fit, y_fit_curr, 'g--', label='Photocurrent fit')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.invert_yaxis()



    fig.tight_layout()
    plt.savefig(r'C:\Users\patri\Coding\Master\figures\Quantum.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Σ ScaledIntensity =", photo_df['ScaledIntensity'].sum())
    print(photo_df['f_abs'].min(), photo_df['f_abs'].max())
    return current_df


def process_single_file(filepath, lightsource_df, plot=True):
    # Step 1: Load and process
    raw = pd.read_csv(filepath, skiprows=18, delim_whitespace=True)
    raw = in_mikroamper(raw, 'Current/A')
    raw['Current/A'] = raw['Current/A'] / 0.011
    raw.rename(columns={raw.columns[4]: 'Intensity'}, inplace=True)
    raw['IsDark'] = raw['Intensity'] <= 0.5

    # Step 2: Crop and baseline correct
    raw = crop_data_to_time_window(raw, time_column='Time/s', start_time=100, end_time=680)

    raw = baseline_correct_and_plot(raw, threshold=0.8, degree=4)

    # Step 3: Offset corrected current
    corrected_min = raw['CorrectedCurrent'].min()
    raw['AbsCorrectedCurrent'] = raw['CorrectedCurrent'] - corrected_min

    # Plot everything in one go
    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        axs[0].plot(raw['Time/s'], raw['Current/A'], alpha=0.5)
        axs[0].set_title('Raw Current')

        axs[1].plot(raw['Time/s'], raw['CorrectedCurrent'])
        axs[1].set_title('Baseline Corrected Current')

        axs[2].plot(raw['Time/s'], raw['AbsCorrectedCurrent'])
        axs[2].set_title('Abs Corrected Current')

        for ax in axs:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Current (A)')
        plt.tight_layout()
        plt.show()

        # Optionally: plot with light-phase shading
        # plot_raw_current(raw)

    return raw


def compute_and_plot_current_drop_vs_intensity(
    data,
    current_column='CorrectedCurrent',
    intensity_column='Intensity',
    is_dark_column='IsDark',
    dark_points=4,
    dark_gap=1,
    poly_degree=3
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- Subtract background light intensity ---
    background_intensity = np.abs(data.loc[data[is_dark_column], intensity_column]).mean()
    data[intensity_column] = data[intensity_column] - background_intensity

    results = []
    in_light = False
    start_idx = None
    light_phases = []

    # Identify light phases
    for i in range(len(data)):
        is_dark = data[is_dark_column].iloc[i]
        if not is_dark and not in_light:
            start_idx = i
            in_light = True
        elif is_dark and in_light:
            light_phases.append((start_idx, i))
            in_light = False
    if in_light:
        light_phases.append((start_idx, len(data)))

    # Compute drops
    for start, end in light_phases:
        if start < dark_gap + dark_points:
            continue

        dark_start = start - dark_gap - dark_points
        dark_end = start - dark_gap

        dark_current = data[current_column].iloc[dark_start:dark_end]
        light_current = data[current_column].iloc[start:end]
        light_intensity = data[intensity_column].iloc[start:end]

        drop = light_current.mean() - dark_current.mean()
        drop_err = np.sqrt(light_current.std()**2 + dark_current.std()**2)

        results.append({
            'MeanIntensity': light_intensity.mean(),
            'IntensityStd': light_intensity.std(),
            'CurrentDrop': drop,
            'CurrentError': drop_err
        })

    stats_df = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        stats_df['MeanIntensity'], stats_df['CurrentDrop'],
        xerr=stats_df['IntensityStd'], yerr=stats_df['CurrentError'],
        fmt='o', color='green', ecolor='gray', capsize=5, label='Data'
    )

    # Polynomial regression fit
    x = stats_df['MeanIntensity'].values
    y = stats_df['CurrentDrop'].values
    coeffs = np.polyfit(x, y, deg=poly_degree)
    poly_fn = np.poly1d(coeffs)

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = poly_fn(x_fit)
    plt.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=2, label=f'Poly Fit (deg {poly_degree})')

    plt.xlabel('Mean Intensity (W/m²)')
    plt.ylabel('Current Drop (µA)')
    plt.title('Photocurrent Drop vs Light Intensity')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()  # ✅ Invert y-axis for upward negative current
    plt.tight_layout()
    plt.show()

    return stats_df


if __name__ == '__main__':

    lightsource_df = load_spectrum(lightsourceP)

    all_dfs = []
    file_errors = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                raw_df = process_single_file(filepath, lightsource_df, plot=False)
                stats_df = compute_and_plot_current_drop_vs_intensity(raw_df)

                stats_df['File'] = filename
                all_dfs.append(stats_df)

                # Store per-file current error for combining later
                file_errors.append(stats_df['CurrentError'].values)

            except Exception as e:
                print(f"Error with {filepath}: {e}")

    # Combine all results
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Set bin size here
    bin_width = 0.8

    # Create bins based on the bin_width
    combined_df['IntensityBin'] = (combined_df['MeanIntensity'] / bin_width).round() * bin_width
    grouped = combined_df.groupby('IntensityBin')

    # Compute mean, std, and total error
    mean_intensity = grouped['MeanIntensity'].mean()
    intensity_error = grouped['MeanIntensity'].std()
    mean_drop = grouped['CurrentDrop'].mean()
    drop_std = grouped['CurrentDrop'].std()

    # Compute mean of within-file error (by intensity group)
    mean_within_file_error = grouped['CurrentError'].mean()

    # Combine into total error
    total_error = np.sqrt(drop_std**2 + mean_within_file_error**2)

    # Final DataFrame
    avg_stats_df = pd.DataFrame({
        'MeanIntensity': grouped['MeanIntensity'].mean(),  # center of each bin
        'IntensityError': grouped['MeanIntensity'].std(),
        'MeanDrop': grouped['CurrentDrop'].mean(),
        'DropSTD': grouped['CurrentDrop'].std(),
        'WithinFileError': grouped['CurrentError'].mean()
    })
    avg_stats_df['TotalError'] = np.sqrt(avg_stats_df['DropSTD']**2 + avg_stats_df['WithinFileError']**2)

    def power_fn(x, a, b, c):
        return a * np.power(x, b) + c
    # Prepare data
    x_data = avg_stats_df['MeanIntensity'].values
    y_data = avg_stats_df['MeanDrop'].values

    # Fit power law
    popt, _ = curve_fit(power_fn, x_data, y_data, maxfev=10000)

    # Generate fit line
    x_fit = np.linspace(x_data.min(), x_data.max(), 200)
    y_fit = power_fn(x_fit, *popt)

    # Plot
    plt.figure(figsize=(14, 10))

    plt.errorbar(
        avg_stats_df['MeanIntensity'], avg_stats_df['MeanDrop'],
        xerr=avg_stats_df['IntensityError'],
        yerr=avg_stats_df['TotalError'],
        fmt='o', capsize=5, capthick=3, color='green', ecolor='gray', label='Mean Current Drop'
    )

    # Dotted power fit
    plt.plot(x_fit, y_fit, linestyle=':', color='black', linewidth=2, label='Power Fit')

    plt.xlabel('Mean Intensity (W/m²)')
    plt.ylabel('Current Drop (µA)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(r'C:\Users\patri\Coding\Master\figures\Quantum_AverageCurrents.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Compute EQE and IQE

    absorb_df = load_absorbance(
        file_path=abs_dir,
        concentration_mM=absorb_spec_conc,
        path_length_cm=1,
        surface_coverage=surface_coverage,
        plot=True
    )


    eqe_iqe_df = compute_EQE_IQE_from_spectrum(current_df=avg_stats_df, photo_df=lightsource_df, absorb_df=absorb_df, electrode_area=area)

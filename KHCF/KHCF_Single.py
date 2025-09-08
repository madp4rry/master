import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import peak_widths
from scipy.stats import linregress
import os

# Set the style of matplotlib to 'ggplot'
plt.style.use('seaborn-v0_8-poster')
from matplotlib import rcParams
# Set the font to "Rubik" and adjust font sizes
rcParams['font.family'] = 'Rubik'
rcParams['axes.labelsize'] = 20
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 14

def in_mikroamper(df, column_name):
    df_adj = df.copy()
    for _ in range(3):
        df_adj[column_name] = df_adj[column_name] * 100
    return df_adj

def process_file(filepath, A, figure_dir):
    # Define column names explicitly
    col_names = ['Number', 'Time/s', 'Potential/V', 'Current/A']

    # Preprocessing
    dataR = pd.read_csv(
        filepath,
        skiprows=19,  # Skip metadata and header rows
        sep=r'\s+',  # Handle space/tab delimited
        names=col_names,  # Force correct column names
        header=None  # Treat all rows as data
    )

    # Debugging: Check for non-numeric values in 'Current/A'
    if not pd.api.types.is_numeric_dtype(dataR['Current/A']):
        non_numeric_rows = dataR[pd.to_numeric(dataR['Current/A'], errors='coerce').isna()]
        print(f"Non-numeric rows in 'Current/A':\n{non_numeric_rows}")
        raise ValueError(f"Non-numeric values found in 'Current/A' column of file: {filepath}")

    # Convert 'Current/A' to numeric, replacing non-numeric values with NaN
    dataR['Current/A'] = pd.to_numeric(dataR['Current/A'], errors='coerce')

    # Drop rows with NaN in 'Current/A'
    dataR = dataR.dropna(subset=['Current/A'])

    dataR['Current/A'] = dataR['Current/A'].astype('float32')  # Optimize memory usage
    dataC = in_mikroamper(dataR, 'Current/A')
    dataC['Current/A'] = dataC['Current/A'] / A

    # Assign cycle numbers
    cycle_no = 0
    cycles = []
    highest_voltage = dataC['Potential/V'].max()
    for i in range(len(dataC)):
        if i > 0 and dataC.loc[i, 'Potential/V'] == highest_voltage and dataC.loc[i - 1, 'Potential/V'] != highest_voltage:
            cycle_no += 1
        cycles.append(cycle_no)
    dataC['Cycle_No'] = cycles

    # Filter cycles
    max_cycle_no = dataC['Cycle_No'].max()
    dataF = dataC[(dataC['Cycle_No'] >= 1) & (dataC['Cycle_No'] < max_cycle_no)].reset_index(drop=True)

    # Save raw plot
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    figure_path = os.path.join(figure_dir, f"{base_filename}_raw.png")
    plt.figure(figsize=(10, 6))
    plt.plot(dataF['Potential/V'], dataF['Current/A'], color='blue', linewidth=1.5, label='Cyclic Voltammetry')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (µA/cm²)')
    plt.title('Cyclic Voltammetry Measurement')
    plt.legend(loc='upper left')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()

    return dataF


def compute_combined_stats(dataF,filepath, figure_dir):
    # Initialize variables
    half_cycle_dfs = []
    cycles = dataF['Cycle_No'].unique()

    # Split cycles into half cycles
    for c in cycles:
        cdataF = dataF[dataF['Cycle_No'] == c].reset_index(drop=True)
        # Get indices of minimum voltage
        min_v_cdataF = cdataF[cdataF['Potential/V'] == cdataF['Potential/V'].min()].dropna()
        cutting_point = min_v_cdataF.index.max()
        # Split the DataFrame into two halves
        first_half = cdataF.iloc[:cutting_point + 1].reset_index(drop=True)
        second_half = cdataF.iloc[cutting_point + 1:].reset_index(drop=True)
        # Add cycle number to each half
        first_half['half_Cycle_No'] = c
        second_half['half_Cycle_No'] = c + 0.5
        # Append the halves to the list
        half_cycle_dfs.append(first_half)
        half_cycle_dfs.append(second_half)

    # Concatenate all half-cycle DataFrames into one
    dataF = pd.concat(half_cycle_dfs, ignore_index=True)

    # Filter full cycles (integer cycle numbers)
    full_cycles = dataF[dataF['half_Cycle_No'] % 1 == 0]

    # Filter half cycles (non-integer cycle numbers)
    half_cycles = dataF[dataF['half_Cycle_No'] % 1 != 0]

    # Group full cycles by 'Potential/V' and calculate mean and standard deviation of 'Current/A'
    full_cycle_stats = full_cycles.groupby('Potential/V')['Current/A'].agg(['mean', 'std']).reset_index()

    # Group half cycles by 'Potential/V' and calculate mean and standard deviation of 'Current/A'
    half_cycle_stats = half_cycles.groupby('Potential/V')['Current/A'].agg(['mean', 'std']).reset_index()

    # Invert the half_cycle_stats DataFrame
    half_cycle_stats = half_cycle_stats.iloc[::-1].reset_index(drop=True)

    # Extract the last n rows of half_cycle_stats
    n = 5
    last_half_cycle_rows = half_cycle_stats.iloc[-n:]
    full_cycle_stats = pd.concat([last_half_cycle_rows, full_cycle_stats], ignore_index=True)

    # Combine full and half cycle stats
    combined_stats = pd.concat([full_cycle_stats, half_cycle_stats], ignore_index=True)

    # Get time into combined_stats
    start_potential = combined_stats.iloc[0]['Potential/V']
    start_index = dataF[dataF['Potential/V'] == start_potential].index[0]
    reference_times = dataF.loc[start_index: start_index + len(combined_stats) - 1, 'Time/s'].reset_index(drop=True)
    # Normalize time to 0s
    normalized_times = reference_times - reference_times.iloc[0]
    combined_stats = combined_stats.copy()
    combined_stats['Time/s'] = normalized_times

    # Apply rolling average to smooth the data
    window_size = 3
    window_sizeS = 5
    combined_stats['mean'] = combined_stats['mean'].rolling(window=window_size, center=True).mean()
    combined_stats['std'] = combined_stats['std'].rolling(window=window_sizeS, center=True).mean()

    # Drop rows with NaN values resulting from rolling
    combined_stats = combined_stats.dropna().reset_index(drop=True)

    combined_halfcycles = []


    # Plot the combined data with standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(combined_stats['Potential/V'], combined_stats['mean'], label='Combined Cycles', color='purple',
             linewidth=1.5)
    plt.fill_between(combined_stats['Potential/V'],
                     combined_stats['mean'] - 3*combined_stats['std'],
                     combined_stats['mean'] + 3*combined_stats['std'],
                     color='purple', alpha=0.2)

    # Add labels, title, and legend
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (µA/cm²)')
    plt.title('Averaged Currents with Standard Deviation (Combined)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    figure_path = os.path.join(figure_dir, f"{base_filename}_smooth.png")
    # Save and show the plot
    plt.savefig(figure_path, bbox_inches='tight', dpi=300)
    plt.show()

    return combined_stats, dataF


filepath =  "C:/Users/patri/Coding/Master/CV/KHCF/500m_800kcl.txt"
def analyze_peaks(voltage, current, std, figure_dir, filename="peak_analysis.png",
                  anodic_window=(0.08, 0.4), cathodic_window=(-0.24, 0.15), gapL=110, gapR=120):
    print(figure_dir)

    def estimate_peak_width(voltage, current, peak_idx, is_anodic=True):
        width_results = peak_widths(current if is_anodic else -current, [peak_idx], rel_height=0.5)
        return max(5, int(width_results[0][0]))

    def estimate_peak_width(voltage, current, peak_idx, is_anodic=True):
        width_results = peak_widths(current if is_anodic else -current, [peak_idx], rel_height=0.5)
        return max(5, int(width_results[0][0]))

    def fit_baseline_one_sided(voltage, current, peak_idx, width, is_anodic, gapL, gapR, manual_slope=40):
        scan_direction = np.sign(voltage[peak_idx + 1] - voltage[peak_idx]) if peak_idx < len(voltage) - 1 else -1
        if is_anodic:
            fit_idx = np.arange(max(0, peak_idx - width - gapL), peak_idx - gapL) if scan_direction > 0 else \
                      np.arange(peak_idx + gapL, min(len(voltage), peak_idx + width + gapL))
        else:
            fit_idx = np.arange(peak_idx + gapR, min(len(voltage), peak_idx + width + gapR)) if scan_direction > 0 else \
                      np.arange(max(0, peak_idx - width - gapR), peak_idx - gapR)
        x_fit, y_fit = voltage[fit_idx], current[fit_idx]
        if is_anodic and manual_slope is not None:
            mean_x, mean_y = np.mean(x_fit), np.mean(y_fit)
            intercept = mean_y - manual_slope * mean_x
            slope = manual_slope
        else:
            slope, intercept, *_ = linregress(x_fit, y_fit)
        return slope, intercept

    def corrected_peak(voltage, current, peak_idx, is_anodic):
        width = estimate_peak_width(voltage, current, peak_idx, is_anodic)
        slope, intercept = fit_baseline_one_sided(voltage, current, peak_idx, width, is_anodic, gapL, gapR)
        baseline = slope * voltage[peak_idx] + intercept
        return current[peak_idx] - baseline, slope, intercept

    # Find peak indices in specified voltage windows
    anodic_mask = (voltage >= anodic_window[0]) & (voltage <= anodic_window[1])
    cathodic_mask = (voltage >= cathodic_window[0]) & (voltage <= cathodic_window[1])

    anodic_index = np.where(anodic_mask)[0][np.argmax(current[anodic_mask])]
    cathodic_index = np.where(cathodic_mask)[0][np.argmin(current[cathodic_mask])]

    # Baseline-corrected peaks
    Ipa, a_slope, a_int = corrected_peak(voltage, current, anodic_index, is_anodic=True)
    Ipc, c_slope, c_int = corrected_peak(voltage, current, cathodic_index, is_anodic=False)
    delta_Ep = voltage[anodic_index] - voltage[cathodic_index]
    # Calculate anodic and cathodic peak potentials
    Epa = voltage[anodic_index]
    print(f"Anodic peak potential (Epa): {Epa:.2f} V")
    Epc = voltage[cathodic_index]
    print(f"Cathodic peak potential (Epc): {Epc:.2f} V")
    # Plot
    cv_label = os.path.splitext(os.path.basename(filepath))[0]
    plt.figure(figsize=(10, 6))
    plt.plot(voltage, current, label=f'CV: {cv_label}', color='black',linewidth=1.5, alpha=0.9)
    plt.fill_between(voltage, current - std, current + std, color='gray', alpha=0.2)

    # Baselines
    plt.plot(voltage, a_slope * voltage + a_int, 'r--', label='Anodic Baseline', linewidth=0.8, alpha=0.4)
    plt.plot(voltage, c_slope * voltage + c_int, 'b--', label='Cathodic Baseline', linewidth=0.8, alpha=0.4)

    # Peaks
    plt.plot(voltage[anodic_index], current[anodic_index], 'ro', label='Anodic Peak')
    plt.plot(voltage[cathodic_index], current[cathodic_index], 'bo', label='Cathodic Peak')

    # Anodic vertical line (baseline to peak)
    baseline_anodic_value = a_slope * voltage[anodic_index] + a_int
    plt.vlines(x=voltage[anodic_index],
               ymin=baseline_anodic_value,
               ymax=current[anodic_index],
               colors='red', linestyles='dotted', alpha=0.5)

    # Cathodic vertical line (baseline to peak)
    baseline_cathodic_value = c_slope * voltage[cathodic_index] + c_int
    plt.vlines(x=voltage[cathodic_index],
               ymin=baseline_cathodic_value,
               ymax=current[cathodic_index],
               colors='blue', linestyles='dotted', alpha=0.5)
    # Text annotations
    plt.text(voltage[anodic_index], current[anodic_index] + 5, f'Ipa = {Ipa:.1f} µA/cm²', color='red')
    plt.text(voltage[cathodic_index], current[cathodic_index] - 10, f'Ipc = {Ipc:.1f} µA/cm²', color='blue')
    plt.text((voltage[anodic_index] + voltage[cathodic_index]) / 2,
             (current[anodic_index] + current[cathodic_index]) / 2,
             f'Ep = {delta_Ep:.2f} V', ha='center', color='gray', fontsize=12)

    # Midpoint current between anodic and cathodic peaks (for ΔEp line)
    mid_current = (current[anodic_index] + current[cathodic_index]) / 2

    # Horizontal line to indicate ΔEp
    plt.hlines(y=mid_current,
               xmin=voltage[cathodic_index],
               xmax=voltage[anodic_index],
               colors='gray', linestyles='dashed', linewidth=1.5)

    # Compute average and std of absolute peak currents
    Ipa_abs = abs(Ipa)
    Ipc_abs = abs(Ipc)
    I_avg = (Ipa_abs + Ipc_abs) / 2
    I_std = np.std([Ipa_abs, Ipc_abs], ddof=1)  # sample std (n-1)

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (µA/cm²)")
    plus_minus = "\u00B1"
    plt.title(f"Peak Current = {I_avg:.2f} {plus_minus} {I_std:.2f} µA/cm²")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    os.makedirs(figure_dir, exist_ok=True)
    #plt.savefig(figure_dir + "analyzed/" + cv_label, bbox_inches='tight', dpi=300)
    print(f"Saved plot to: {figure_dir + cv_label}.png")
    plt.show()
    plt.close()



    return {
        "Filename": os.path.basename(filename),
        "Ipa (µA/cm²)": Ipa,
        "Ipc (µA/cm²)": Ipc,
        "Epa (V)": Epa,           # <--- Add Epa here
        "Epc (V)": Epc,           # <--- Add Epc here
        "Ep (V)": delta_Ep,
        "I_avg (µA/cm²)": I_avg,
        "I_std (µA/cm²)": I_std,
    }



# Path to your raw .txt data file


# Electrode area in cm²
A = 0.282

# Path to your baseline CSV (with 'Potential/V' and 'mean' columns)
#baseline_path = r"C:/Users/patri/Coding/Master/CV/KHF/baseline.csv"

# Directory to save plots
figure_dir = "C:/Users/patri/Coding/Master/figures/CV/KHCF/"
os.makedirs(figure_dir, exist_ok=True)  # Create directory if it doesn't exist

# ==== RUN PROCESSING ====
# Step 1: Process the raw file
dataF = process_file(filepath, A, figure_dir)

# Step 2: Compute stats and apply baseline correction
combined_stats, full_data_with_halfcycles = compute_combined_stats(
    dataF, filepath, figure_dir
)

# Optional: Print a preview
print("Processed data preview:")
print(combined_stats.head())


# Example using your `combined_stats` DataFrame:
voltage = combined_stats["Potential/V"].values
current = combined_stats["mean"].values
std = combined_stats["std"].values
import numpy as np
from scipy.stats import linregress

def _label_by_direction_per_cycle(cdf, win=7, eps=None):
    c = cdf.sort_values('Time/s' if 'Time/s' in cdf.columns else cdf.index).reset_index(drop=True)
    E = c['Potential/V'].to_numpy()
    dE = np.diff(E, prepend=E[0])
    win = max(3, int(win) | 1)                 # odd window
    kernel = np.ones(win)/win
    dE_sm = np.convolve(dE, kernel, mode='same')
    if eps is None:
        step_est = np.median(np.abs(np.diff(E))) if len(E) > 1 else 1e-3
        eps = max(1e-6, 0.2*step_est)
    lab = []
    cur = 'anodic' if (len(dE_sm)>1 and dE_sm[1] >= 0) else 'cathodic'
    for g in dE_sm:
        if g > +eps: cur = 'anodic'
        elif g < -eps: cur = 'cathodic'
        lab.append(cur)
    c = c.copy(); c['half_label'] = lab
    return c

def _mean_std(arr):
    arr = np.asarray(arr, float)
    good = np.isfinite(arr)
    if not good.any():
        return np.nan, 0.0
    mean = np.nanmean(arr)
    n = good.sum()
    std = np.nanstd(arr, ddof=1) if n > 1 else 0.0
    return float(mean), float(std)

def per_cycle_peak_stats(dataF,
                         baseline_windows={'anodic': (-0.20, -0.10),
                                           'cathodic': (0.25, 0.40)},
                         peak_windows={'anodic': (0.08, 0.40),
                                       'cathodic': (-0.24, 0.15)}):
    """
    Compute Epa/Epc and baseline-corrected Ipa/Ipc per *cycle* (anodic/cathodic
    determined by sweep direction), then return their mean & std across cycles.
    """
    Epa_list, Epc_list, Ipa_list, Ipc_list, Ep_list = [], [], [], [], []

    for _, cdf in dataF.groupby('Cycle_No'):
        lab = _label_by_direction_per_cycle(cdf)

        # --- Anodic baseline fit in fixed window (direction-aware) ---
        Emin, Emax = baseline_windows['anodic']; Emin, Emax = min(Emin, Emax), max(Emin, Emax)
        msk = (lab['half_label']=='anodic') & (lab['Potential/V'].between(Emin, Emax))
        vx = lab.loc[msk, 'Potential/V'].to_numpy(); iy = lab.loc[msk, 'Current/A'].to_numpy()
        if len(vx) >= 3:
            ma, ba, *_ = linregress(vx, iy)
        else:
            ma, ba = 0.0, (float(np.nanmean(iy)) if len(iy) else np.nan)

        # Anodic peak (max in anodic phase within window)
        pEmin, pEmax = peak_windows['anodic']; pEmin, pEmax = min(pEmin, pEmax), max(pEmin, pEmax)
        mskp = (lab['half_label']=='anodic') & (lab['Potential/V'].between(pEmin, pEmax))
        vA = lab.loc[mskp, 'Potential/V'].to_numpy(); iA = lab.loc[mskp, 'Current/A'].to_numpy()
        if len(iA):
            k = int(np.argmax(iA))
            Epa = float(vA[k]); Ipa_raw = float(iA[k])
            Ipa = Ipa_raw - (ma*Epa + ba) if np.isfinite(ba) else np.nan
        else:
            Epa, Ipa = np.nan, np.nan

        # --- Cathodic baseline fit ---
        Emin, Emax = baseline_windows['cathodic']; Emin, Emax = min(Emin, Emax), max(Emin, Emax)
        msk = (lab['half_label']=='cathodic') & (lab['Potential/V'].between(Emin, Emax))
        vx = lab.loc[msk, 'Potential/V'].to_numpy(); iy = lab.loc[msk, 'Current/A'].to_numpy()
        if len(vx) >= 3:
            mc, bc, *_ = linregress(vx, iy)
        else:
            mc, bc = 0.0, (float(np.nanmean(iy)) if len(iy) else np.nan)

        # Cathodic peak (min in cathodic phase within window)
        pEmin, pEmax = peak_windows['cathodic']; pEmin, pEmax = min(pEmin, pEmax), max(pEmin, pEmax)
        mskp = (lab['half_label']=='cathodic') & (lab['Potential/V'].between(pEmin, pEmax))
        vC = lab.loc[mskp, 'Potential/V'].to_numpy(); iC = lab.loc[mskp, 'Current/A'].to_numpy()
        if len(iC):
            k = int(np.argmin(iC))
            Epc = float(vC[k]); Ipc_raw = float(iC[k])
            Ipc = Ipc_raw - (mc*Epc + bc) if np.isfinite(bc) else np.nan
        else:
            Epc, Ipc = np.nan, np.nan

        Epa_list.append(Epa); Ipa_list.append(Ipa)
        Epc_list.append(Epc); Ipc_list.append(Ipc)
        Ep_list.append((Epa - Epc) if np.isfinite(Epa) and np.isfinite(Epc) else np.nan)

    # aggregate
    Epa_mu, Epa_sd = _mean_std(Epa_list)
    Epc_mu, Epc_sd = _mean_std(Epc_list)
    Ipa_mu, Ipa_sd = _mean_std(Ipa_list)
    Ipc_mu, Ipc_sd = _mean_std(Ipc_list)
    Ep_mu,  Ep_sd  = _mean_std(Ep_list)

    return {
        # per-file means (if you want to use them)
        "Epa (V) mean": Epa_mu, "Epa (V) std": Epa_sd,
        "Epc (V) mean": Epc_mu, "Epc (V) std": Epc_sd,
        "Ipa (µA/cm²) mean": Ipa_mu, "Ipa (µA/cm²) std": Ipa_sd,
        "Ipc (µA/cm²) mean": Ipc_mu, "Ipc (µA/cm²) std": Ipc_sd,
        "Ep (V) mean": Ep_mu, "Ep (V) std": Ep_sd,
        "N cycles": int(len(Ep_list))
    }


results = analyze_peaks(voltage, current, std, figure_dir = figure_dir)

print(results)


def plot_all_cond(folder, A, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)

    all_combined = []  # List of (label, df)
    cond = "000kcl"  # Condition to filter files
    for filename in os.listdir(folder):
        if cond in filename.lower() and filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            try:
                # Step 1: Process and compute combined stats
                dataF = process_file(filepath, A, figure_dir)
                combined_stats, _ = compute_combined_stats(dataF, filepath, figure_dir)

                # Step 2: Label from filename
                label = os.path.splitext(filename)[0]
                all_combined.append((label, combined_stats))

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Step 3: Plot all together
    if all_combined:
        plt.figure(figsize=(10, 6))
        for label, df in all_combined:
            plt.plot(df['Potential/V'], df['mean'], label=label)

        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (µA/cm²)")
        plt.title("Combined CVs for " + cond + " Condition")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save combined plot
        combined_plot_path = os.path.join(figure_dir, "combined_"+ cond + "_plot.png")
        plt.savefig(combined_plot_path, bbox_inches='tight')
        plt.show()
    else:
        print("No valid files found for plotting.")


data_dir=  "C:/Users/patri/Coding/Master/CV/KHCF/"
output_csv = figure_dir +"peak_summary.csv"
#plot_all_cond(folder=data_dir, A=A, figure_dir=figure_dir)
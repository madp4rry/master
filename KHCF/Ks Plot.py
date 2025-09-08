import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
plt.style.use('seaborn-v0_8-poster')
from matplotlib import rcParams
# Set the font to "Rubik" and adjust font sizes
rcParams['font.family'] = 'Rubik'
rcParams['axes.labelsize'] = 30
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 22
plt.rcParams['lines.markersize'] = 12  # Default marker size (points)

figure_dir = "C:/Users/patri/Coding/Master/figures/CV/KHCF/"
data = figure_dir +"peak_summarry.csv"
# Read the CSV (adjust the filename/path if needed)
df = pd.read_csv(data, sep=";", decimal=",")
print(df.head())

# Add √Scan column
df["sqrt_Scan"] = np.sqrt(df["Scan"])
df["Ip_std"] = df["Ip_std"]/3
# Plot Ip ± Ip_std vs √Scan for each concentration
plt.figure(figsize=(10, 6))
for conc in sorted(df["Conc"].unique()):
    sub_df = df[df["Conc"] == conc]
    plt.errorbar(
        sub_df["sqrt_Scan"],
        sub_df["Ip"],
        yerr=sub_df["Ip_std"],
        fmt='o--',                # circle markers with dashed lines
        alpha=0.9,                # slightly transparent
        label=f"{int(conc)} mM",
        capsize=4,
        capthick=2 # little lines on error bars
    )

plt.xlabel("√Scan rate (√mV/s)")
plt.ylabel("Peak current Ip (µA)")
plt.title("Ip vs √Scan rate with standard deviations")
plt.legend(title="Concentration")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Constants ---
n = 1          # Number of electrons transferred
A = 0.282      # cm² - Electrode area
C = 3e-6       # mol/cm³ - Concentration used in Ip calculation
F = 96485      # C/mol - Faraday constant
R = 8.314      # J/mol·K - Ideal gas constant
T = 298        # K (25 °C) - Absolute temperature
RS_const = 2.69e5 # Randles-Sevcik constant (for 25°C)


df["sqrt_Scan"] = np.sqrt(df["Scan"])

# Prepare figure
plt.figure(figsize=(12, 7))

# Store diffusion coefficients
diffusion_results = []

for conc in sorted(df["Conc"].unique()):
    sub_df = df[df["Conc"] == conc]
    x = sub_df["sqrt_Scan"]
    y = sub_df["Ip"]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Calculate D using rearranged Randles–Sevcik equation
    D = (slope / (RS_const * n**1.5 * A * C))**2
    # Calculate error for D from standard error of the slope
    # error_D = |dD/dslope| * std_err_slope = |2 * D / slope| * std_err_slope
    D_error = abs(2 * D / slope) * std_err if slope != 0 else np.nan
    # Save results
    diffusion_results.append({
        "Conc": conc,  # Use "Conc" for consistent merging later
        "Slope (uA/sqrt(V/s))": slope,
        "D (cm^2/s)": D,
        "D_error (cm^2/s)": D_error,  # Store the calculated error for D
        "R^2": r_value ** 2
    })

for conc in sorted(df["Conc"].unique()):
    sub_df = df[df["Conc"] == conc]
    plt.errorbar(
        sub_df["sqrt_Scan"],
        sub_df["Ip"],
        yerr=sub_df["Ip_std"],
        fmt='o--',                # circle markers with dashed lines
        alpha=0.9,                # slightly transparent
        label=f"{int(conc)} mM",
        capsize=4,
        capthick=1.2, # little lines on error bars
        elinewidth = 0.9

    )
'''
    # Plot with regression
    line_color = plt.errorbar(
        x, y, yerr=sub_df["Ip_std"], fmt='o', label=f'{int(conc)} mM', elinewidth=0.8, capsize=4, capthick=0.8, alpha=1
    ).lines[0].get_color()  # Extract the color of the data points
    plt.plot(x, slope * x + intercept, '--', color=line_color, alpha=1, linewidth=2)
'''
# Finalize plot
plt.xlabel("√Scan rate (√mV/s)")
plt.ylabel("Peak current Ip (µA/cm²)")
#legend location offset to the right
plt.legend(title="Concentration", loc='upper left', bbox_to_anchor=(0.1, 0.99))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir + "Ip_vs_sqrt_Scan_with_regression.png", dpi=300, bbox_inches='tight')
plt.show()

# Show diffusion coefficients
df_D = pd.DataFrame(diffusion_results)
print(df_D)



# --- Prepare DataFrame for ks Calculation ---
# Merge D values into the main DataFrame 'df'
df_merged = pd.merge(df, df_D[["Conc", "D (cm^2/s)"]], on='Conc', how='left')
df_merged.rename(columns={"D (cm^2/s)": "D_calculated"}, inplace=True)

# --- CRITICAL: Use 'Ep' directly as Delta_Ep and assume it's in Volts ---
# Assume 'Ep' column in df_merged is already the peak separation (Delta_Ep) in Volts.
if 'Ep' not in df_merged.columns:
    print("\nError: 'Ep' column not found in 'df'. This column is assumed to be the peak separation (Delta_Ep).")
    print("Please ensure your 'df' DataFrame contains an 'Ep' column with the peak separation data.")
    exit()

# DIRECTLY assign Ep (which is in Volts) to Delta_Ep_V
df_merged['Delta_Ep_V'] = df_merged['Ep']

# Convert scan rate (mV/s to V/s)
df_merged['Scan_V_per_s'] = df_merged['Scan'] / 1000

# --- Standard Heterogeneous Electron Transfer Rate Constant (ks) Calculation (Nicholson Method) ---

# Empirical Psi (Ψ) approximation function
def get_psi_direct_approximation(n_delta_ep_mV):
    if n_delta_ep_mV <= 59.2: # For n=1 at 25C, reversible limit for Delta_Ep is 59.2mV
        return np.inf # Effectively reversible

    p = n_delta_ep_mV
    # This is an empirical fit suitable for n*Delta_Ep from ~60 to ~200 mV.
    if 60 <= p <= 200:
        return (-0.6288 + 0.0021 * p) / (1 - 0.017 * p)
    return np.nan # Outside the valid range for this specific approximation

df_merged2= df_merged.copy()  # Keep a copy of the original DataFrame for plotting
# Filter scan rates to focus on your selected range (e.g., 250 to 1500 mV/s)
df_filtered = df_merged[(df_merged["Scan"] >= 250) & (df_merged["Scan"] <= 1500)].copy()

# Then do all calculations on df_filtered instead of df_merged
df_filtered["n_Delta_Ep_mV"] = n * df_filtered["Delta_Ep_V"] * 1000
df_filtered["Psi"] = df_filtered["n_Delta_Ep_mV"].apply(get_psi_direct_approximation)

df_filtered["ks_calc_per_scan"] = df_filtered.apply(
    lambda row: row["Psi"] * np.sqrt(
        (np.pi * n * row["D_calculated"] * row["Scan_V_per_s"]) / (R * T)
    ) if pd.notna(row["Psi"]) and pd.notna(row["D_calculated"]) else np.nan,
    axis=1
)
#renaming again, because lazy
df_merged = df_filtered.copy()

# Summarize ks by concentration from filtered data
ks_summary = df_filtered.groupby("Conc").agg(
    Avg_ks_cm_s=('ks_calc_per_scan', 'mean'),
    Std_Dev_ks_cm_s=('ks_calc_per_scan', 'std'),
    Num_Measurements=('ks_calc_per_scan', 'count')
).reset_index()

print("\n--- Calculated Standard Heterogeneous Electron Transfer Rate Constants (ks) (filtered) ---")
print(ks_summary)

# --- Diagnostic Plots ---

# Plot Peak Separation (Ep) vs. Scan Rate
plt.figure(figsize=(12, 7))
for conc in sorted(df_merged2["Conc"].unique()):
    sub_df = df_merged2[df_merged2["Conc"] == conc]
    plt.errorbar(
        sub_df["Scan"],
        sub_df["Ep"] * 1000,          # Convert V to mV for y-values
        yerr=sub_df["Ep_std"] * 1000,  # Convert V to mV for error bars
        fmt='o--',
        capsize=4,
        capthick=1.2, # little lines on error bars
        elinewidth = 0.9,
        label=f"{int(conc)} mM"
    )

plt.xlabel("Scan rate (mV/s)")
plt.ylabel(r"Peak separation $\Delta E_p$ (mV)")
plt.axhline(y=59.2/n, color='r', linestyle=':', label=r'Rev. limit (59 mV)')
plt.legend(title="Concentration", loc='upper left', bbox_to_anchor=(0.1, 0.99))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir + "Peak_Separation_vs_Scan.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot Psi vs. 1/√Scan rate
plt.figure(figsize=(12, 7))
for conc in sorted(df_merged["Conc"].unique()):
    sub_df = df_merged[df_merged["Conc"] == conc].dropna(subset=['Psi', 'Scan_V_per_s'])
    if not sub_df.empty:
        x_plot = 1 / np.sqrt(sub_df["Scan_V_per_s"])
        plt.plot(x_plot, sub_df["Psi"], 'o--', label=f"{int(conc)} mM")

plt.xlabel("1 / √Scan rate (1/√V/s)")
plt.ylabel(r"Dimensionless Parameter ($\Psi$)") # Use raw string r"" for LaTeX-like syntax

plt.legend(title="Concentration", loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir + "Psi_vs_1_sqrt_Scan.png", dpi=300, bbox_inches='tight')
plt.show()



# --- FINAL PLOTTING: Diffusion Coefficients and ks on a Dual Y-axis Plot with Error Bars ---

# Create a figure and a primary axes
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot Diffusion Coefficients on ax1 with error bars
color = 'tab:blue'
ax1.set_xlabel('Concentration (mM)')
ax1.set_ylabel('Diffusion Coefficient $D$ ($cm^2/s$)', color=color, fontsize=26)

# Check if 'D_error (cm^2/s)' column exists in df_D before plotting error bars
if 'D_error (cm^2/s)' in df_D.columns:
    ax1.errorbar(
        df_D['Conc'],
        df_D['D (cm^2/s)'],
        yerr=df_D['D_error (cm^2/s)'], # Error bars for D
        fmt='o-',
        color=color,
        label='Diffusion Coefficient $D$',
        capsize=5, # Add caps to error bars
        capthick=1,
        elinewidth=0.6,
        #alpha=0.3
    )
else:
    # If D_error column is missing, plot without error bars and print a warning
    print("\nWarning: 'D_error (cm^2/s)' column not found in df_D. Plotting Diffusion Coefficient without error bars.")
    ax1.plot(
        df_D['Conc'],
        df_D['D (cm^2/s)'],
        'o-',
        color=color,
        label='Diffusion Coefficient $D$'
    )

ax1.tick_params(axis='y', labelcolor=color)

ax1.grid(True, alpha=0.3)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()

# Plot Standard Heterogeneous Electron Transfer Rate Constants ($k_s$) on ax2 with error bars
color = 'tab:red'
ax2.set_ylabel('st. e Transfer Rate Constant $k_s$ ($cm/s$)', color=color, fontsize=23)
# Use 'Std_Dev_ks_cm_s' from ks_summary for error bars
ax2.errorbar(
    ks_summary['Conc'],
    ks_summary['Avg_ks_cm_s'],
    yerr=ks_summary['Std_Dev_ks_cm_s'], # Error bars for ks
    fmt='s-',
    color=color,
    label='Avg. $k_s$',
    capsize=5, # Add caps to error bars
    capthick=1,
    elinewidth=0.6,
    #alpha=0.3
)
ax2.tick_params(axis='y', labelcolor=color)

# Add legends for both axes (needs to be done manually for twinx)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2,loc='lower right')

# Adjust layout and save the figure
plt.tight_layout()

plt.savefig(figure_dir + 'D_and_ks_vs_Concentration.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot 'D_and_ks_vs_Concentration.png' created.")

plt.figure(figsize=(10,6))
for conc in sorted(df["Conc"].unique()):
    sub_df = df[df["Conc"] == conc]
    plt.plot(sub_df["Scan"], sub_df["Epa"], 'o-', label=f'Epa {int(conc)} mM')
    plt.plot(sub_df["Scan"], sub_df["Epc"], 's--', label=f'Epc {int(conc)} mM')

plt.xlabel("Scan rate (mV/s)")
plt.ylabel("Peak Potential (V)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

concentrations = sorted(df["Conc"].unique())
n_plots = len(concentrations)

fig, axs = plt.subplots(2, 2, figsize=(16, 15), sharey=True)
axs = axs.flatten()

colors = cm.get_cmap('tab10', n_plots)

for i, conc in enumerate(concentrations):
    ax = axs[i]
    sub_df = df[df["Conc"] == conc].copy()
    color = colors(i)

    # Plot Epa and Epc points
    ax.plot(sub_df["Scan"], sub_df["Epa"], 'o', color=color, alpha=0.7, label='Epa')
    ax.plot(sub_df["Scan"], sub_df["Epc"], 's', color=color, alpha=0.7, label='Epc')

    # Calculate ΔEp
    delta_ep = sub_df["Epa"] - sub_df["Epc"]
    ax.plot(sub_df["Scan"], delta_ep, '^-', color=color, alpha=1, label='ΔEp')

    # Highlight linear region points (250–500 mV/s)
    linear_region = sub_df[(sub_df["Scan"] >= 250) & (sub_df["Scan"] <= 500)]

    # Plot linear region points on ΔEp curve with bigger black circles
    ax.plot(linear_region["Scan"], linear_region["Epa"] - linear_region["Epc"], 'o',
            color='black', markersize=9, label='Linear region ΔEp')

    # Linear fit on linear region ΔEp vs Scan
    if len(linear_region) >= 2:
        slope, intercept = np.polyfit(linear_region["Scan"], linear_region["Epa"] - linear_region["Epc"], 1)
        # Plot fit line over full scan range
        scan_range = np.linspace(sub_df["Scan"].min(), sub_df["Scan"].max(), 100)
        ax.plot(scan_range, slope * scan_range + intercept, '--', color='gray', label='Linear fit (250–500 mV/s)')
        # Mark y-intercept with red horizontal line and point
        ax.axhline(intercept, color='red', linestyle=':', linewidth=1.5, label=f'Intercept = {intercept:.3f} V')
        ax.plot(0, intercept, 'ro')

    ax.set_title(f"{int(conc)} mM")
    ax.set_xlabel("Scan rate (mV/s)")
    if i % 2 == 0:
        ax.set_ylabel("Potential (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

# Remove any unused axes if concentrations < 4
for j in range(n_plots, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(figure_dir + "Epa_Epc_DeltaEp_vs_Scan.png", dpi=300, bbox_inches='tight')
plt.show()

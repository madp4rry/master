# ============================ KHCF ks/D analysis (cleaned) ============================
import re
import ast
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# --- stdout encoding (Windows consoles) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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

# --- Paths ---
figure_dir = r"C:/Users/patri/Coding/Master/figures/CV/KHCF/autolab/"
data = figure_dir + "peak_summary.csv"   # German CSV with ; and decimal ,

# --- Read CSV ---
df = pd.read_csv(data, sep=";", decimal=",", encoding="cp1252")
print(df.head())

# ============== Parse Scan (mV/s) & Conc (mM) from filename =================
def parse_file_meta(fname: str):
    s = fname.lower()
    m_scan = re.search(r'(\d+(?:[.,]\d+)?)\s*m[v]?', s, flags=re.I)
    m_conc = re.search(r'(\d+(?:[.,]\d+)?)\s*kcl', s, flags=re.I)
    scan = float(m_scan.group(1).replace(',', '.')) if m_scan else np.nan
    conc = float(m_conc.group(1).replace(',', '.')) if m_conc else np.nan
    return pd.Series({'Scan': scan, 'Conc': conc})

df[['Scan', 'Conc']] = df['File'].apply(parse_file_meta)
df['sqrt_Scan'] = np.sqrt(df['Scan'])

# (Optional) shrink I_std like before
df['I_std (µA/cm²)'] = df['I_std (µA/cm²)'] / 3.0

# ============== Plot Iavg ± Istd vs √Scan ===================================
plt.figure(figsize=(10, 6))
for conc in sorted(df['Conc'].dropna().unique()):
    sub = df[df['Conc'] == conc].sort_values('sqrt_Scan')
    plt.errorbar(
        sub['sqrt_Scan'], sub['I_avg (µA/cm²)'],
        yerr=sub['I_std (µA/cm²)'],
        fmt='o--', alpha=0.9, label=f"{int(conc)} mM",
        capsize=4, capthick=2
    )
plt.xlabel("√Scan rate (√mV/s)")
plt.ylabel("Peak current $I_{avg}$ (µA/cm²)")
plt.title("Iavg vs √Scan rate with standard deviations")
plt.legend(title="Concentration")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============== Randles–Ševčík: D from low scans ============================
# Constants (current density form; v in V/s; C in mol/cm³)
n = 1
A = 0.282
C = 3e-6
F = 96485
R = 8.314
T = 298
RS_CONST_J = 2.69e11  # µA·cm⁻² = (2.69×10¹¹) n^(3/2) C D^(1/2) v^(1/2)

WINDOW_SCANS = [50, 100, 200]  # change if needed

diffusion_results, fit_lines = [], []
for conc in sorted(df['Conc'].dropna().unique()):
    sub_all = df[df['Conc'] == conc].copy()
    have = sorted(sub_all['Scan'].unique())
    scans_used = [s for s in WINDOW_SCANS if s in have]
    if len(scans_used) < 2:
        scans_used = have[:3]  # fallback
    sub = sub_all[sub_all['Scan'].isin(scans_used)].sort_values('Scan')
    if len(sub) < 2:
        continue

    x_rs = np.sqrt(sub['Scan'].to_numpy() / 1000.0)  # √(V/s)
    y = sub['I_avg (µA/cm²)'].to_numpy()

    slope, intercept, r, p, se = linregress(x_rs, y)
    D = (slope / (RS_CONST_J * (n**1.5) * C))**2
    D_error = abs(2 * D / slope) * se if slope != 0 else np.nan

    slope0 = float((x_rs * y).sum() / (x_rs * x_rs).sum())
    yhat0 = slope0 * x_rs
    r2_0 = 1 - np.sum((y - yhat0)**2) / np.sum((y - y.mean())**2) if len(y) > 2 else np.nan

    diffusion_results.append({
        "Conc": conc,
        "Scans used (mV/s)": ",".join(map(str, scans_used)),
        "Slope free (µA/cm² / √(V/s))": slope,
        "Intercept free (µA/cm²)": intercept,
        "R² free": r**2,
        "Slope origin (µA/cm² / √(V/s))": slope0,
        "R² origin": r2_0,
        "D (cm^2/s)": D,
        "D_error (cm^2/s)": D_error,
    })

    # Store regression line to overlay on √(mV/s) axis
    x_plot = np.sqrt(sub['Scan'].to_numpy())         # √(mV/s)
    yhat = slope * x_rs + intercept                  # model uses √(V/s)
    fit_lines.append((conc, x_plot, yhat))

# Overlay the RS fits
plt.figure(figsize=(12, 7))
for conc in sorted(df['Conc'].dropna().unique()):
    sub = df[df['Conc'] == conc].sort_values('sqrt_Scan')
    h = plt.errorbar(
        sub['sqrt_Scan'], sub['I_avg (µA/cm²)'],
        yerr=sub['I_std (µA/cm²)'],
        fmt='o--', alpha=0.9, label=f"{int(conc)} mM",
        capsize=4, capthick=1.2, elinewidth=0.9
    )
    for c, x_plot, yhat in fit_lines:
        if c == conc:
            plt.plot(x_plot, yhat, '--', linewidth=2, alpha=0.9, color=h.lines[0].get_color())

plt.xlabel("√Scan rate (√mV/s)")
plt.ylabel("Peak current $I_{avg}$ (µA/cm²)")
plt.legend(title="Concentration", loc='upper left', bbox_to_anchor=(0.1, 0.99))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir + "Ip_vs_sqrt_Scan_with_regression.png", dpi=300, bbox_inches='tight')
plt.show()

df_D = pd.DataFrame(diffusion_results)
print(df_D)

# ============== Peak separation vs scan (for reference) =====================
plt.figure(figsize=(12, 7))
for conc in sorted(df['Conc'].dropna().unique()):
    sub = df[df["Conc"] == conc].sort_values('Scan')
    y = sub["Ep (V)"] * 1000.0
    yerr = (sub["Ep (V) std"] * 1000.0) if "Ep (V) std" in sub.columns else None
    plt.errorbar(sub["Scan"], y, yerr=yerr, fmt='o--', capsize=4, capthick=1.2,
                 elinewidth=0.9, label=f"{int(conc)} mM")
plt.xlabel("Scan rate (mV/s)")
plt.ylabel(r"Peak separation $\Delta E_p$ (mV)")
plt.axhline(y=59.2/n, color='r', linestyle=':', label=r'Rev. limit (59 mV)')
plt.legend(title="Concentration", loc='upper left', bbox_to_anchor=(0.1, 0.99))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir + "Peak_Separation_vs_Scan.png", dpi=300, bbox_inches='tight')
plt.show()

# ================= Nicholson mapping & window picking ======================
def get_psi_direct_approximation(delta_ep_mV, n=1):
    """Nicholson empirical fit: maps ΔEp (mV) -> Psi (dimensionless)."""
    p = float(n * delta_ep_mV)
    if p <= 59.2:
        return np.inf
    if 60 <= p <= 200:
        return (-0.6288 + 0.0021 * p) / (1 - 0.017 * p)
    return np.nan

# Merge D into df for per-row ks later
df = pd.merge(
    df,
    df_D[["Conc", "D (cm^2/s)"]].rename(columns={"D (cm^2/s)": "D_calculated"}),
    on="Conc",
    how="left",
)

def suggest_nicholson_windows(
    df,
    v_bounds=(100, 900),          # scan window to consider (mV/s)
    dep_bounds=(90, 200),         # keep points whose ΔEp is within [mV]
    min_pts=4,                    # prefer 4; raise/lower as needed
    n=1, R=8.314, T=298.0
):
    need = {"Conc","Scan","Ep (V)","D_calculated"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"df missing columns: {missing}")

    rows = []
    for conc in sorted(df["Conc"].dropna().unique()):
        sub = df[df["Conc"] == conc].copy().sort_values("Scan")
        sub = sub[(sub["Scan"] >= v_bounds[0]) & (sub["Scan"] <= v_bounds[1])]
        sub = sub[(sub["Ep (V)"]*1000 >= dep_bounds[0]) & (sub["Ep (V)"]*1000 <= dep_bounds[1])]
        if len(sub) < min_pts:
            print(f"{int(conc)} mM: no candidate window (after bounds/filter).")
            continue

        scans = sub["Scan"].to_numpy()
        best = None
        for i in range(0, len(scans)-min_pts+1):
            for j in range(i+min_pts-1, len(scans)):
                win = sub.iloc[i:j+1]
                v = win["Scan"].to_numpy()/1000.0
                dep_mV = (win["Ep (V)"]*1000.0).to_numpy()
                D = win["D_calculated"].to_numpy()

                Psi = np.array([get_psi_direct_approximation(x, n=n) for x in dep_mV], dtype=float)
                if np.any(~np.isfinite(Psi)) or np.any(~np.isfinite(D)) or np.any(D<=0):
                    continue
                ks = Psi * np.sqrt((np.pi * n * D * v) / (R * T))
                if np.any(~np.isfinite(ks)) or np.any(ks<=0):
                    continue

                m_log, _ = np.polyfit(np.log(v), np.log(ks), 1)
                x_lin = 1/np.sqrt(v)
                _, interceptPsi, rPsi, *_ = linregress(x_lin, Psi)
                r2Psi = rPsi**2
                rel_intercept = abs(interceptPsi)/max(1e-12, np.nanmedian(Psi))

                score = -abs(m_log) - 0.2*np.std(np.log(ks)) + 0.02*len(win) + 0.5*(r2Psi-0.95) - 0.5*rel_intercept
                if (best is None) or (score > best[0]):
                    best = (score, win["Scan"].tolist(), m_log, r2Psi, rel_intercept,
                            float(np.nanmean(ks)), float(np.nanstd(ks)),
                            float(dep_mV.min()), float(dep_mV.max()))
        if best is None:
            print(f"{int(conc)} mM: no valid window after scoring.")
            continue

        score, scans_used, m_log, r2Psi, rel_int, ks_mean, ks_std, dep_min, dep_max = best
        print(f"{int(conc)} mM -> use scans [{','.join(map(str, scans_used))}] "
              f"(n={len(scans_used)}), ΔEp {dep_min:.0f}-{dep_max:.0f} mV | "
              f"slope log(ks) vs log(v) = {m_log:+.2f} | "
              f"R2(Psi vs 1/sqrt(v)) = {r2Psi:.3f} | "
              f"Psi intercept ~ {100*rel_int:.1f}% of median | "
              f"ks = {ks_mean:.2e} ± {ks_std:.2e} cm/s")

        rows.append({
            "Conc_mM": int(conc),
            "Scans_mV_s": scans_used,
            "logks_vs_logv_slope": m_log,
            "Psi_vs_1/sqrt(v)_R2": r2Psi,
            "Psi_intercept_rel": rel_int,
            "ks_mean_cm_s": ks_mean,
            "ks_std_cm_s": ks_std,
            "DeltaEp_range_mV": (dep_min, dep_max),
        })

    return pd.DataFrame(rows)

summary_df = suggest_nicholson_windows(
    df, v_bounds=(100, 900), dep_bounds=(90, 200), min_pts=4
)
print("\nSummary table:\n", summary_df.to_string(index=False))

# ================= Epa/Epc/ΔEp panel w/ window & metrics ====================
def _to_list(x):
    if isinstance(x, (list, tuple, np.ndarray)): return list(x)
    if isinstance(x, str) and x.strip().startswith('['):
        try: return list(ast.literal_eval(x))
        except Exception: return [float(y) for y in x.split(',')]
    return [x]

win = {}
for _, r in summary_df.iterrows():
    c = int(r["Conc_mM"])
    scans = sorted(float(s) for s in _to_list(r["Scans_mV_s"]))
    win[c] = dict(scans=scans,
                  mlog=float(r["logks_vs_logv_slope"]),
                  r2=float(r["Psi_vs_1/sqrt(v)_R2"]),
                  psi_int=float(r["Psi_intercept_rel"]),
                  dep_rng=r.get("DeltaEp_range_mV", None))

concs = sorted([int(x) for x in df["Conc"].dropna().unique()])
fig, axs = plt.subplots(2, 2, figsize=(16, 15))
axs = axs.flatten()
cmap = plt.colormaps.get_cmap('tab10')
custom_ylim = {200: (0.0, 0.30), 400: (0.0, 0.30), 800: (0.0, 0.30)}

for i, conc in enumerate(concs[:4]):
    ax = axs[i]
    color = cmap(i % 10)
    sub = df[df["Conc"]==conc].sort_values("Scan").copy()

    ax.plot(sub['Scan'], sub['Epa (V)'], 'o', color=color, alpha=0.85, label='Epa')
    ax.plot(sub['Scan'], sub['Epc (V)'], 's', color=color, alpha=0.85, label='Epc')
    dep_all = (sub['Epa (V)'] - sub['Epc (V)'])
    ax.plot(sub['Scan'], dep_all, '^-', color=color, lw=2, label=r'$\Delta E_p$')

    metrics_text = None
    if conc in win and win[conc]["scans"]:
        scans = win[conc]["scans"]
        sel = sub[sub["Scan"].isin(scans)].copy()

        style = {'ls': '--', 'color': 'gray', 'lw': 2.5}
        tag = "ΔEp linear fit (window)"
        if len(sel) >= 4:
            style.update(ls='-', color='gray')
        else:
            tag += " (3-pt provisional)"

        ax.plot(sel['Scan'], (sel['Epa (V)']-sel['Epc (V)']),
                'o', color='black', ms=9, label='Selected window')

        if len(sel) >= 3:
            xw = sel['Scan'].to_numpy()
            yw = (sel['Epa (V)'] - sel['Epc (V)']).to_numpy()
            slope, intercept = np.polyfit(xw, yw, 1)
            xr = np.linspace(sub['Scan'].min(), sub['Scan'].max(), 200)
            ax.plot(xr, slope*xr + intercept, label=tag, **style)

        dep_rng = win[conc]["dep_rng"]
        dep_txt = f"{dep_rng[0]:.0f}–{dep_rng[1]:.0f} mV" if isinstance(dep_rng,(list,tuple)) and len(dep_rng)==2 else "n/a"
        metrics_text = (
            f"win: {int(min(scans))}-{int(max(scans))} mV/s\n"
            f"ΔEp: {dep_txt}\n"
            f"slope log($k_s$) vs log($v$) = {win[conc]['mlog']:+.2f}\n"
            f"$R^2(\\Psi)$ = {win[conc]['r2']:.3f}; "
            f"$\\Psi$ int = {100*win[conc]['psi_int']:.0f}%"
        )

    ax.set_title(f"{conc} mM")
    ax.set_xlabel("Scan rate (mV/s)")
    if i % 2 == 0:
        ax.set_ylabel("Potential / V")
    if conc in custom_ylim:
        ax.set_ylim(*custom_ylim[conc])
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if metrics_text:
        handles.append(plt.Line2D([], [], color='none'))
        labels.append(metrics_text)
    ax.legend(handles, labels, loc='lower right', fontsize=10, handlelength=1.6,
              borderpad=0.8, labelspacing=0.7)

for j in range(len(concs), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout(rect=[0, 0, 1, 0.76])
plt.savefig(figure_dir + "Epa_Epc_DeltaEp_vs_Scan.png", dpi=300, bbox_inches='tight')
plt.show()

# ============== Windowed Ψ & ks (adds 0 mM with 100–400 mV/s) ==============
# Build window dict (from summary_df); add 0 mM generic window
win_info = {int(r["Conc_mM"]): sorted(float(s) for s in _to_list(r["Scans_mV_s"]))
            for _, r in summary_df.iterrows()}

if 0 in df["Conc"].unique() and 0 not in win_info:
    scans0 = sorted(df.loc[df["Conc"]==0, "Scan"].unique())
    scans0 = [s for s in scans0 if 100 <= s <= 400]
    if scans0:
        win_info[0] = scans0

def _psi_from_deltaEp_V(dep_V):
    return get_psi_direct_approximation(dep_V*1000.0, n=n)

rows = []
for conc, scans in win_info.items():
    if not scans: continue
    sub = df[(df["Conc"]==conc) & (df["Scan"].isin(scans))].copy()
    if sub.empty: continue
    sub["Scan_V_per_s"] = sub["Scan"] / 1000.0
    sub["DeltaEp_V"] = sub["Epa (V)"] - sub["Epc (V)"]
    sub["Psi"] = sub["DeltaEp_V"].apply(_psi_from_deltaEp_V)
    sub = sub[np.isfinite(sub["Psi"])]
    if sub.empty:
        rows.append(sub)
        continue
    if "D_calculated" in sub.columns:
        sub["ks_calc_per_scan"] = sub.apply(
            lambda r: r["Psi"] * np.sqrt((np.pi * n * r["D_calculated"] * r["Scan_V_per_s"]) / (R * T)),
            axis=1
        )
    rows.append(sub)

df_win = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ===== Ψ vs 1/√Scan (using selected windows) — show fits & Ψ-intercepts =====
plt.figure(figsize=(11, 7))
cmap = plt.colormaps.get_cmap('tab10')

for i, conc in enumerate(sorted(df["Conc"].dropna().unique())):
    sub = df_win[df_win["Conc"] == conc].dropna(subset=["Psi", "Scan_V_per_s"])
    if sub.empty:
        continue

    x = 1.0 / np.sqrt(sub["Scan_V_per_s"].to_numpy())  # 1/√(V/s)
    y = sub["Psi"].to_numpy()
    color = cmap(i % 10)

    # data points
    plt.plot(x, y, 'o', color=color, label=f"{int(conc)} mM")

    # linear fit of Ψ vs 1/√v  → intercept at x→0 is Ψ0 = b
    if len(x) >= 2:
        m, b, r, p, se = linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 200)
        plt.plot(xr, m * xr + b, '--', color=color, alpha=0.85)

        # Ψ-intercept line (y = b) over the data range
        plt.hlines(b, xmin=x.min(), xmax=x.max(), colors=color, linestyles=':', alpha=0.35)

        # compute the SAME metric as in the summary table
        median_psi = np.median(y)
        psi_intercept_rel = abs(b) / max(1e-12, median_psi)  # matches "Psi_intercept_rel"

        # annotate: raw Ψ0, relative (% of median), and R²
        plt.text(
            x.min(), b,
            rf"$\Psi_0$={b:.2f}  |  "
            rf"$|\Psi_0|/\mathrm{{median}}(\Psi)$={100 * psi_intercept_rel:.0f}%"
            rf"\n$R^2$={r ** 2:.3f}",
            color=color, fontsize=13, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.6)
        )

plt.xlabel(r"$1 / \sqrt{\mathrm{Scan\ rate}}\ \mathrm{(1/\sqrt{V/s})}$")
plt.ylabel(r"Dimensionless parameter $\Psi$")
plt.legend(title="Concentration", loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir + "Psi_vs_1_sqrt_Scan.png", dpi=300, bbox_inches='tight')
plt.show()

# ---- ks summary from windows ----
ks_summary_win = (
    df_win.dropna(subset=["ks_calc_per_scan"])
         .groupby("Conc", as_index=False)
         .agg(Avg_ks_cm_s=("ks_calc_per_scan","mean"),
              Std_Dev_ks_cm_s=("ks_calc_per_scan","std"),
              Num_Measurements=("ks_calc_per_scan","count"))
)
if 0 in df["Conc"].unique() and 0 not in ks_summary_win["Conc"].values:
    ks_summary_win = pd.concat(
        [ks_summary_win, pd.DataFrame([{"Conc": 0, "Avg_ks_cm_s": np.nan,
                                        "Std_Dev_ks_cm_s": np.nan, "Num_Measurements": 0}])],
        ignore_index=True
    )
ks_summary_win = ks_summary_win.sort_values("Conc")

# ---- Dual-axis: D and ks vs concentration (windowed ks) ----
fig, ax1 = plt.subplots(figsize=(12, 7))

color = 'tab:blue'
ax1.set_xlabel('Concentration (mM)')
ax1.set_ylabel(r'Diffusion Coefficient $D$ (cm$^2$/s)', color=color, fontsize=24)
if 'D_error (cm^2/s)' in df_D.columns:
    ax1.errorbar(df_D['Conc'], df_D['D (cm^2/s)'],
                 yerr=df_D['D_error (cm^2/s)'],
                 fmt='o-', color=color, label=r'Diffusion Coefficient $D$',
                 capsize=5, capthick=1, elinewidth=0.6)
else:
    ax1.plot(df_D['Conc'], df_D['D (cm^2/s)'], 'o-', color=color, label=r'Diffusion Coefficient $D$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel(r'Standard rate constant $k_s$ (cm/s)', color=color, fontsize=22)
ax2.errorbar(ks_summary_win['Conc'], ks_summary_win['Avg_ks_cm_s'],
             yerr=ks_summary_win['Std_Dev_ks_cm_s'],
             fmt='s-', color=color, label=r'Avg. $k_s$ (windowed)',
             capsize=5, capthick=1, elinewidth=0.6)
ax2.tick_params(axis='y', labelcolor=color)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')

plt.tight_layout()
plt.savefig(figure_dir + 'D_and_ks_vs_Concentration.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nPlot 'D_and_ks_vs_Concentration.png' created (using selected windows).")
# ===========================================================================


# ---- Dual-axis: D and ks vs concentration (windowed ks)  FOR RESULTS----
fig, ax1 = plt.subplots(figsize=(12, 10))

color = 'tab:blue'
ax1.set_xlabel('KCL Concentration [mM]', labelpad=20, fontsize=34)
ax1.set_ylabel(r'Diffusion Coefficient $D$ [cm$^2$/s]', color=color, fontsize=30)
if 'D_error (cm^2/s)' in df_D.columns:
    ax1.errorbar(df_D['Conc'], df_D['D (cm^2/s)'],
                 yerr=df_D['D_error (cm^2/s)'],
                 fmt='o-', color=color, label=r'Diffusion Coefficient $D$',
                 capsize=5, capthick=1, elinewidth=0.6)
else:
    ax1.plot(df_D['Conc'], df_D['D (cm^2/s)'], 'o-', color=color, label=r'Diffusion Coefficient $D$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3, linestyle='--')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel(r'Standard rate constant $k_s$ [cm/s]', color=color, fontsize=30)
ax2.errorbar(ks_summary_win['Conc'], ks_summary_win['Avg_ks_cm_s'],
             yerr=ks_summary_win['Std_Dev_ks_cm_s'],
             fmt='s-', color=color, label=r'Avg. $k_s$ (windowed)',
             capsize=5, capthick=1, elinewidth=0.6)
ax2.tick_params(axis='y', labelcolor=color)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')

plt.tight_layout()
plt.savefig(figure_dir + 'D_and_ks_vs_Concentration_FR.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nPlot 'D_and_ks_vs_Concentration.png' created (using selected windows).")
# ===========================================================================

# =================== Irreversible ln|Ip| vs (E − E°) analysis ===================

from typing import Dict, Optional, Tuple

# --- Helpers: estimate E° and prep tidy data for fits ---

def estimate_E0_per_conc(df: pd.DataFrame) -> Dict[float, float]:
    """
    E°(formal) per concentration from the *slowest* scan: E° ≈ (Epa+Epc)/2.
    If multiple rows at the slowest scan exist, returns their average.
    """
    need = {"Conc", "Scan", "Epa (V)", "Epc (V)"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"Cannot estimate E°; df missing columns: {missing}")

    E0_map = {}
    for conc, sub in df.groupby("Conc"):
        if sub.empty or sub["Scan"].isna().all():
            continue
        vmin = sub["Scan"].min()
        slow = sub[sub["Scan"] == vmin]
        e0 = 0.5 * (slow["Epa (V)"] + slow["Epc (V)"]).mean()
        E0_map[float(conc)] = float(e0)
    return E0_map


def _choose_current_column(df: pd.DataFrame, which: str) -> Optional[str]:
    """
    Prefer peak-specific columns if available; fallback to I_avg with a warning.
    which ∈ {'cathodic','anodic'}
    """
    if which == "cathodic":
        if "Ipc (µA/cm²)" in df.columns:
            return "Ipc (µA/cm²)"
    else:
        if "Ipa (µA/cm²)" in df.columns:
            return "Ipa (µA/cm²)"
    if "I_avg (µA/cm²)" in df.columns:
        print(f"[WARN] Using 'I_avg (µA/cm²)' for {which} analysis (no Ipa/Ipc column found).")
        return "I_avg (µA/cm²)"
    return None


from typing import Dict, Optional
# ---------------- PROF CONVENTION: x = Ep - E° for BOTH branches ----------------
from typing import Dict, Optional

def prepare_lnIp_dataset(
    df: pd.DataFrame,
    E0_map: Dict[float, float],
    which: str = "cathodic",
    dep_min_mV: Optional[float] = None
) -> pd.DataFrame:
    """
    Tidy rows for ln|Ip| vs x fits.

    Columns returned:
      Conc, Scan, E0, E_peak, ln_j (A/cm²), DeltaEp_V,
      eta_V (branch-positive overpotential, kept for reference),
      x_prof = Ep - E°  (NEG for cathodic, POS for anodic),
      used_current_col
    """
    need = {"Conc","Scan","Epa (V)","Epc (V)"}
    if not need.issubset(df.columns):
        raise ValueError(f"df missing columns: {need - set(df.columns)}")

    # pick current column (prefers Ipa/Ipc; falls back to I_avg)
    cur_col = None
    if which.lower() == "cathodic" and "Ipc (µA/cm²)" in df.columns:
        cur_col = "Ipc (µA/cm²)"
    elif which.lower() == "anodic" and "Ipa (µA/cm²)" in df.columns:
        cur_col = "Ipa (µA/cm²)"
    elif "I_avg (µA/cm²)" in df.columns:
        cur_col = "I_avg (µA/cm²)"
        print(f"[WARN] {which}: using I_avg (µA/cm²)")
    else:
        raise ValueError("Need Ipa/Ipc or I_avg in the CSV.")

    rows = []
    for _, r in df.iterrows():
        conc = float(r["Conc"])
        if conc not in E0_map:
            continue
        E0  = float(E0_map[conc])
        Epa = float(r["Epa (V)"])
        Epc = float(r["Epc (V)"])
        dEp = Epa - Epc

        if dep_min_mV is not None and 1000.0*dEp < dep_min_mV:
            continue

        # branch-positive overpotentials (kept for info)
        eta_c = max(0.0, E0 - Epc)
        eta_a = max(0.0, Epa - E0)

        if which.lower() == "cathodic":
            Epk = Epc
            eta = eta_c
            x_prof = Epc - E0         # <-- Ep - E°
        else:
            Epk = Epa
            eta = eta_a
            x_prof = Epa - E0         # <-- Ep - E°

        val = r.get(cur_col, np.nan)
        if pd.isna(val):
            continue
        j = abs(float(val)) * 1e-6    # A/cm²
        if j <= 0:
            continue

        rows.append({
            "Conc": float(conc),
            "Scan": float(r["Scan"]),
            "E0": E0,
            "E_peak": Epk,
            "ln_j": float(np.log(j)),
            "DeltaEp_V": dEp,
            "eta_V": eta,             # for reference
            "x_prof": float(x_prof),  # Ep - E°
            "used_current_col": cur_col,
        })

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        print(f"[WARN] No rows prepared for {which}.")
    return tidy


def fit_lnIp_vs_profX(
    tidy: pd.DataFrame,
    which: str,
    n: int, C_star_mol_per_cm3: float,
    R: float, T: float, F: float,
    min_points_per_conc: int = 3
) -> pd.DataFrame:
    """
    Fit ln(jp) = b + m * x with x = Ep - E° (sign-preserving) for both branches.

    α mapping for this convention:
      cathodic: α = - (m RT)/(nF)
      anodic:   α = 1 + (m RT)/(nF)

    Intercept → ks: ks = exp(b) / (0.227 n F C*)
    """
    expected = ["Conc","which","N","slope","intercept","R2",
                "alpha","alpha_err_est","ks_cm_s","ks_err_est","used_current_col"]
    rows = []
    if tidy is None or tidy.empty:
        return pd.DataFrame(columns=expected)

    for conc, sub in tidy.groupby("Conc"):
        sub = sub.dropna(subset=["x_prof","ln_j"])
        if len(sub) < min_points_per_conc:
            continue

        x = sub["x_prof"].to_numpy(float)  # Ep - E°
        y = sub["ln_j"].to_numpy(float)
        npts = len(x)
        xbar, ybar = x.mean(), y.mean()
        Sxx = ((x - xbar)**2).sum()
        if Sxx <= 0 or npts < 2:
            continue
        Sxy = ((x - xbar)*(y - ybar)).sum()

        m = Sxy / Sxx
        b = ybar - m * xbar
        yhat = m * x + b

        rss = ((y - yhat)**2).sum()
        tss = ((y - ybar)**2).sum()
        R2  = 1 - rss/tss if tss > 0 else np.nan
        s2  = rss/(npts - 2) if npts > 2 else np.nan
        se_m = np.sqrt(s2/Sxx) if np.isfinite(s2) and Sxx > 0 else np.nan
        se_b = np.sqrt(s2*(1/npts + xbar**2/Sxx)) if np.isfinite(s2) and Sxx > 0 else np.nan

        RT_nF = (R*T)/(n*F)
        if which.lower() == "cathodic":
            alpha = - m * RT_nF
        else:
            alpha = 1.0 + m * RT_nF
        alpha_err = abs(se_m) * RT_nF if np.isfinite(se_m) else np.nan

        denom = 0.227 * n * F * C_star_mol_per_cm3
        ks = np.exp(b) / denom if denom > 0 else np.nan
        ks_err = ks * se_b if np.isfinite(se_b) else np.nan

        used = sub["used_current_col"].mode().iat[0] if "used_current_col" in sub.columns else ""

        rows.append({
            "Conc": float(conc), "which": which, "N": int(npts),
            "slope": float(m), "intercept": float(b), "R2": float(R2),
            "alpha": float(alpha), "alpha_err_est": float(alpha_err),
            "ks_cm_s": float(ks), "ks_err_est": float(ks_err),
            "used_current_col": used
        })

    df_out = pd.DataFrame(rows, columns=expected)
    return df_out.sort_values(["which","Conc"]) if not df_out.empty else df_out


def plot_lnIp_vs_profX(tidy: pd.DataFrame, fits: pd.DataFrame, which: str, savepath: Optional[str] = None):
    if tidy.empty or fits.empty:
        print(f"[WARN] Nothing to plot for {which}."); return
    plt.figure(figsize=(11,7))
    for conc, sub in tidy.groupby("Conc"):
        sub = sub.sort_values("x_prof")
        plt.plot(sub["x_prof"], sub["ln_j"], 'o', label=f"{int(conc)} mM")
        row = fits[(fits["Conc"]==conc) & (fits["which"]==which)]
        if not row.empty:
            m = row["slope"].iat[0]; b = row["intercept"].iat[0]
            xr = np.linspace(sub["x_prof"].min(), sub["x_prof"].max(), 200)
            plt.plot(xr, m*xr + b, '--', alpha=0.9)
            alpha = row["alpha"].iat[0]; ks = row["ks_cm_s"].iat[0]; r2 = row["R2"].iat[0]
            label = f"$\\alpha$ = {alpha:.2f}\\n$k_s$ = {ks:.2e} cm/s\\n$R^2$ = {r2:.3f}"
            plt.text(xr.min(), m*xr.min()+b, label, fontsize=12, va='bottom',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.6, edgecolor='none'))
    plt.xlabel(r"$E_p - E^\circ$ (V)")
    plt.ylabel(r"$\ln |j_p|$ (A/cm$^2$)")
    title_side = "cathodic" if which=="cathodic" else "anodic"
    plt.title(fr"$\ln |I_p|$ vs $(E_p - E^\circ)$  ({title_side})")
    plt.grid(True, alpha=0.3); plt.legend(title="Concentration"); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()
# -------------------------------------------------------------------------------

# USAGE (replace your previous ln|Ip| block):
E0_map = estimate_E0_per_conc(df)  # your existing helper

tidy_c = prepare_lnIp_dataset(df[df["Conc"]>0], E0_map, which="cathodic", dep_min_mV=None)
tidy_a = prepare_lnIp_dataset(df[df["Conc"]>0], E0_map, which="anodic",   dep_min_mV=None)

# (Optional) Prefer peak currents but allow fallback
def _keep_if_any(tidy, pat):
    if tidy.empty: return tidy
    t2 = tidy[ tidy["used_current_col"].str.contains(pat, na=False) ]
    return t2 if not t2.empty else tidy

tidy_c = _keep_if_any(tidy_c, r"Ipc")
tidy_a = _keep_if_any(tidy_a, r"Ipa")

fits_c = fit_lnIp_vs_profX(tidy_c, "cathodic", n=n, C_star_mol_per_cm3=C, R=R, T=T, F=F)
fits_a = fit_lnIp_vs_profX(tidy_a, "anodic",   n=n, C_star_mol_per_cm3=C, R=R, T=T, F=F)

print(fits_c.to_string(index=False))
print(fits_a.to_string(index=False))

#plot_lnIp_vs_profX(tidy_c, fits_c, "cathodic", savepath=figure_dir+"lnIp_vs_(Ep-E0)_cathodic.png")
#plot_lnIp_vs_profX(tidy_a, fits_a, "anodic",   savepath=figure_dir+"lnIp_vs_(Ep-E0)_anodic.png")


# ---------- RS helper ----------
def rs_D_from_slope(slope_uAcm2_per_sqrtV, n, C, RS_CONST_J):
    """Return D [cm^2/s] from slope of j_p (µA/cm²) vs sqrt(v [V/s])."""
    return (slope_uAcm2_per_sqrtV / (RS_CONST_J * (n**1.5) * C))**2

# ---------- D_O / D_R from low scans ----------
def compute_DO_DR(df, v_bounds=(50, 200), min_points=2):
    """
    Compute D_O (from |Ipc|) and D_R (from Ipa) per concentration,
    using only scans within v_bounds [mV/s].
    """
    rows = []
    for conc, sub_all in df.groupby("Conc"):
        sub = sub_all[(sub_all["Scan"] >= v_bounds[0]) & (sub_all["Scan"] <= v_bounds[1])].copy()
        if sub.empty:
            continue

        x_Vs = np.sqrt(sub["Scan"].to_numpy()/1000.0)   # √(V/s)

        # cathodic → |Ipc| → D_O
        y_ipc = sub.get("Ipc (µA/cm²)", pd.Series(index=sub.index, dtype=float)).astype(float).abs().to_numpy()
        DO = np.nan; DO_se = np.nan
        if np.isfinite(y_ipc).sum() >= min_points:
            m_ipc, _, _, _, se = linregress(x_Vs, y_ipc)
            DO = rs_D_from_slope(m_ipc, n=n, C=C, RS_CONST_J=RS_CONST_J)
            DO_se = abs(2*DO/m_ipc)*se if m_ipc != 0 else np.nan

        # anodic → Ipa → D_R
        y_ipa = sub.get("Ipa (µA/cm²)", pd.Series(index=sub.index, dtype=float)).astype(float).to_numpy()
        DR = np.nan; DR_se = np.nan
        if np.isfinite(y_ipa).sum() >= min_points:
            m_ipa, _, _, _, se = linregress(x_Vs, y_ipa)
            DR = rs_D_from_slope(m_ipa, n=n, C=C, RS_CONST_J=RS_CONST_J)
            DR_se = abs(2*DR/m_ipa)*se if m_ipa != 0 else np.nan

        if not (np.isfinite(DO) and np.isfinite(DR)):
            print(f"[WARN] {int(conc)} mM: insufficient low-v points for D_O/D_R.")
            continue

        rows.append({
            "Conc": float(conc),
            "v_window_mV_s": f"{v_bounds[0]}–{v_bounds[1]}",
            "D_O_cm2_s": DO, "D_O_err": DO_se,
            "D_R_cm2_s": DR, "D_R_err": DR_se,
            "DeltaD_cm2_s": DO-DR,
            "D_ratio_O_over_R": DO/DR if DR>0 else np.nan
        })
    return pd.DataFrame(rows).sort_values("Conc")

# ---------- Literature: Konopka & McDuffie (1970), Table III ----------
def make_konopka_mcduffie_df():
    """
    Average diffusion coefficients at 25 °C in 0.10 M and 1.00 M KCl.
    Values are av ± std dev of group (Table III).
    Units: cm^2/s; Conc in mM.
    """
    data = [
        (100.0,  0.720e-5, 0.018e-5, 0.666e-5, 0.013e-5),  # 0.10 M KCl
        (1000.0, 0.726e-5, 0.011e-5, 0.667e-5, 0.014e-5),  # 1.00 M KCl
    ]
    arr = np.array(data, float)
    df_lit = pd.DataFrame({
        "Conc": arr[:,0],
        "D_O_cm2_s": arr[:,1],
        "D_O_err":   arr[:,2],
        "D_R_cm2_s": arr[:,3],
        "D_R_err":   arr[:,4],
    })
    df_lit["DeltaD_cm2_s"] = df_lit["D_O_cm2_s"] - df_lit["D_R_cm2_s"]
    return df_lit

def plot_D_with_secondary_pct(
    Dsep: pd.DataFrame,
    figure_dir: str,
    show_ratio_pct: bool = True,   # legacy arg (ignored)
    show_delta_pct: bool = False,  # legacy arg (ignored)
    fname: str = "DO_DR_vs_conc_abs_ratio.png"
):
    """
    Left axis:  D_O (from |Ipc|) & D_R (from Ipa) with errors, plus Konopka–McDuffie (1970) hollow markers.
    Right axis: ABSOLUTE deviation |D_O/D_R - 1| in percent (red).  (No ΔD/Ḋ curve.)
    The 'show_ratio_pct'/'show_delta_pct' args are accepted for compatibility but ignored.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # --- your data ---
    conc = np.asarray(Dsep["Conc"], float)
    DO   = np.asarray(Dsep["D_O_cm2_s"], float)
    DR   = np.asarray(Dsep["D_R_cm2_s"], float)
    DOe  = np.asarray(Dsep.get("D_O_err", np.full_like(DO, np.nan)), float)
    DRe  = np.asarray(Dsep.get("D_R_err", np.full_like(DR, np.nan)), float)

    fig, ax1 = plt.subplots(figsize=(14, 8.5))

    h1 = ax1.errorbar(conc, DO, yerr=DOe, fmt="o-", capsize=4, capthick=2, elinewidth=1.5,  label=r"$D_O$ from $|I_{pc}|$")
    h2 = ax1.errorbar(conc, DR, yerr=DRe, fmt="s-", capsize=4, capthick=2, elinewidth=1.5, label=r"$D_R$ from $I_{pa}$")
    ax1.set_xlabel("KCl concentration (mM)")
    ax1.set_ylabel(r"Diffusion coefficient $D$ (cm$^2$/s)")
    ax1.grid(True, alpha=0.25, linestyle="--")

    # --- Konopka–McDuffie points (0.10 M and 1.00 M KCl) ---
    km_conc = np.array([100.0, 1000.0])
    km_DO   = np.array([7.20e-6, 7.26e-6])
    km_DR   = np.array([6.66e-6, 6.67e-6])

    km_ms, km_mew = 11, 2.2
    ax1.scatter(km_conc, km_DO, marker="o", s=km_ms**2, facecolors="none",
                edgecolors=h1[0].get_color(), linewidths=km_mew, zorder=6, alpha=0.85, label="_nolegend_")
    ax1.scatter(km_conc, km_DR, marker="s", s=km_ms**2, facecolors="none",
                edgecolors=h2[0].get_color(), linewidths=km_mew, zorder=6, alpha=0.85, label="_nolegend_")

    # axis ranges include all series
    x_all = np.concatenate([conc.ravel(), km_conc])
    ax1.set_xlim(np.nanmin(x_all) - 20, np.nanmax(x_all) + 20)

    y_all = np.concatenate([DO.ravel(), DR.ravel(), km_DO, km_DR])
    y_all = y_all[np.isfinite(y_all)]
    pad = 0.12 * (np.nanmax(y_all) - np.nanmin(y_all)) if y_all.size else 0
    ax1.set_ylim(np.nanmin(y_all) - pad, np.nanmax(y_all) + pad)

    # --- right axis: absolute ratio only ---
    ax2 = ax1.twinx()
    red = "tab:red"
    ax2.spines["right"].set_color(red)
    ax2.tick_params(axis="y", colors=red)
    ax2.yaxis.label.set_color(red)

    ratio_abs_pct = 100.0 * np.abs(DO/DR - 1.0)
    h_ratio, = ax2.plot(conc, ratio_abs_pct, "d--", lw=2.0, color=red, alpha=0.4,
                        label=r"$|D_O/D_R - 1|$ (%)")

    km_ratio_abs_pct = 100.0 * np.abs(km_DO/km_DR - 1.0)
    ax2.plot(km_conc, km_ratio_abs_pct, "D", mfc="none", mec=red, mew=2.0, ms=10, alpha=0.4, label="_nolegend_")

    ymax = np.nanmax(np.concatenate([ratio_abs_pct, km_ratio_abs_pct]))
    ax2.set_ylim(0, ymax * 1.25 if np.isfinite(ymax) and ymax > 0 else 1)
    ax2.set_ylabel(r"Diff. coef. deviation $D_O/D_R $ (%)")

    # legend with proxies for K–M hollow markers
    km_do_proxy = Line2D([0],[0], marker="o", linestyle="None", mfc="none",
                         mec=h1[0].get_color(), mew=2.0, markersize=9, label=r"$D_O$ (Konopka-McDuffieM, 1970)")
    km_dr_proxy = Line2D([0],[0], marker="s", linestyle="None", mfc="none",
                         mec=h2[0].get_color(), mew=2.0, markersize=9, label=r"$D_R$ (K–M, 1970)")

    ax1.legend([h1, h2, km_do_proxy, km_dr_proxy, h_ratio],
               [r"$D_O$ from $|I_{pc}|$", r"$D_R$ from $I_{pa}$",
                r"$D_O$ (K–M, 1970)", r"$D_R$ (Konopka-McDuffie, 1970)",
                r"$|D_O/D_R - 1|$ (%)"],
               loc="lower right", framealpha=0.9)

    out = os.path.join(figure_dir, fname)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out)




# ---------- RUN ----------
# 1) D_O / D_R from low scans (adjust window if desired)
Dsep = compute_DO_DR(df, v_bounds=(50, 200))
print("\nD_O/D_R from low scan rates:\n", Dsep.to_string(index=False))

plot_D_with_secondary_pct(Dsep, figure_dir, show_ratio_pct=True, show_delta_pct=False)



# ==== ks overlay: your D vs. literature D (Konopka–McDuffie) =================

def _ks_from_Psi_with_given_Dmaps(df_win, Dmap_O, Dmap_R, n, R, T, F,
                                  alpha_default=0.5, alpha_map=None):
    """Generic ks calculator using provided DO/DR maps."""
    rows = []
    for _, r in df_win.iterrows():
        conc = float(r["Conc"])
        if conc not in Dmap_O or conc not in Dmap_R:
            continue
        DO = float(Dmap_O[conc]); DR = float(Dmap_R[conc])
        v  = float(r["Scan_V_per_s"])
        Psi = float(r["Psi"])
        alpha = (alpha_map or {}).get(conc, alpha_default)

        # prof's definition
        factor = np.sqrt(np.pi * DO * v * (n * F / (R * T)))
        ks = Psi * factor * (DR/DO)**alpha
        rows.append({**r.to_dict(), "alpha_used": alpha, "ks_prof_cm_s": ks})

    out = pd.DataFrame(rows)
    if out.empty:
        return out, pd.DataFrame()
    summary = (out.groupby("Conc", as_index=False)
                  .agg(Avg_ks_cm_s=("ks_prof_cm_s","mean"),
                       Std_Dev_ks_cm_s=("ks_prof_cm_s","std"),
                       N=("ks_prof_cm_s","count"),
                       alpha_used=("alpha_used","mean")))
    return out, summary.sort_values("Conc")

def interp_lit_DO_DR(concs_mM):
    """
    Linear interpolation of Konopka–McDuffie (1970) Table III
    between 0.10 M and 1.00 M KCl for requested concs (mM).
    """
    lit = make_konopka_mcduffie_df().sort_values("Conc")
    x = lit["Conc"].to_numpy()
    DOx = lit["D_O_cm2_s"].to_numpy()
    DRx = lit["D_R_cm2_s"].to_numpy()
    concs = np.array(sorted(concs_mM), float)
    DOi = np.interp(concs, x, DOx)
    DRi = np.interp(concs, x, DRx)
    return pd.DataFrame({"Conc": concs, "D_O_cm2_s": DOi, "D_R_cm2_s": DRi})

def plot_ks_overlay(ks_meas, ks_lit, figure_dir, fname="ks_vs_conc_meas_vs_lit.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    # your ks (with your D_O/D_R)
    plt.errorbar(ks_meas["Conc"], ks_meas["Avg_ks_cm_s"],
                 yerr=ks_meas["Std_Dev_ks_cm_s"], fmt="o-", capthick=2, elinewidth=1.5,
                 capsize=5, label=r"$k_s$ (after Nicholson, using measured $D_O,D_R$, this work)")
    # ks using literature D_O/D_R
    plt.errorbar(ks_lit["Conc"], ks_lit["Avg_ks_cm_s"],
                 yerr=ks_lit["Std_Dev_ks_cm_s"], fmt="s--", capthick=2, elinewidth=1.5,
                 capsize=5, label=r"$k_s$ (using literature $D_O,D_R$ Konopka-McDuffie, 1970)")
    plt.xlabel("KCl concentration (mM)")
    plt.ylabel(r"$k_s$ (cm/s)")
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir + fname, dpi=300)
    plt.show()

# --- Build ks using your measured D_O/D_R (you already did this earlier) ---
# Dsep from compute_DO_DR(df, v_bounds=(50,200))
Dmap_O_meas = dict(zip(Dsep["Conc"], Dsep["D_O_cm2_s"]))
Dmap_R_meas = dict(zip(Dsep["Conc"], Dsep["D_R_cm2_s"]))
_, ks_meas_summary = _ks_from_Psi_with_given_Dmaps(
    df_win, Dmap_O_meas, Dmap_R_meas, n=n, R=R, T=T, F=F, alpha_default=0.5
)

# --- Now ks using Konopka–McDuffie D_O/D_R interpolated to your concs ---
concs_needed = sorted(df_win["Conc"].unique().tolist())
Dsep_lit = interp_lit_DO_DR(concs_needed)
Dmap_O_lit = dict(zip(Dsep_lit["Conc"], Dsep_lit["D_O_cm2_s"]))
Dmap_R_lit = dict(zip(Dsep_lit["Conc"], Dsep_lit["D_R_cm2_s"]))
_, ks_lit_summary = _ks_from_Psi_with_given_Dmaps(
    df_win, Dmap_O_lit, Dmap_R_lit, n=n, R=R, T=T, F=F, alpha_default=0.5
)

print("\nks (your D):\n", ks_meas_summary.to_string(index=False))
print("\nks (Konopka–McDuffie D):\n", ks_lit_summary.to_string(index=False))

# --- Overlay plot (keeps your original line, adds literature-based line) ---
plot_ks_overlay(ks_meas_summary, ks_lit_summary, figure_dir,
                fname="ks_vs_conc_measuredD_vs_literatureD.png")

def plot_ks_overlay_Res(ks_meas, ks_lit, figure_dir, fname="ks_vs_conc_meas_vs_lit.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11,9.2))
    # your ks (with your D_O/D_R)
    plt.errorbar(ks_meas["Conc"], ks_meas["Avg_ks_cm_s"],
                 yerr=ks_meas["Std_Dev_ks_cm_s"], fmt="o-", capthick=2, elinewidth=1.5,
                 capsize=5, label=r"$k_s$ (after Nicholson, using measured $D_O,D_R$, this work)")
    # ks using literature D_O/D_R
    plt.errorbar(ks_lit["Conc"], ks_lit["Avg_ks_cm_s"],
                 yerr=ks_lit["Std_Dev_ks_cm_s"], fmt="s--", capthick=2, elinewidth=1.5,
                 capsize=5, label=r"$k_s$ (using literature $D_O,D_R$ Konopka-McDuffie, 1970)")
    plt.xlabel("KCl concentration (mM)", labelpad=24, fontsize=34)
    plt.ylabel(r"Standard heterogeneous rate constant $k_s$ (cm/s)", labelpad=10, fontsize=22)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(loc='lower right', fontsize=19)
    plt.tight_layout()
    plt.savefig(figure_dir + fname, dpi=300)
    plt.show()

plot_ks_overlay_Res(ks_meas_summary, ks_lit_summary, figure_dir,
                fname="ks_vs_conc_measuredD_vs_literatureD_results.png")
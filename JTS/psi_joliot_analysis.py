import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-poster')
from matplotlib import rcParams
rcParams['font.family'] = 'Rubik'
rcParams['axes.labelsize'] = 32
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 24
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ========= Helpers =========
def parse_kcl_columns(cols):
    pat = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*M\s*KCl\s*$", re.IGNORECASE)
    out = []
    for c in cols:
        m = pat.match(c)
        if m:
            out.append((c, float(m.group(1))))
    out.sort(key=lambda x: x[1])
    return out

def interp_at_time(t, y, t0):
    if t0 <= t[0]:  return float(y[0])
    if t0 >= t[-1]: return float(y[-1])
    j = int(np.searchsorted(t, t0))
    tL, tR = t[j-1], t[j]
    yL, yR = y[j-1], y[j]
    return float(yL + (yR - yL) * (t0 - tL) / (tR - tL))

def compute_minima_custom(df, min_search_start_ms=0.0, min_search_end_ms=None,
                          manual_t0_by_conc=None):
    t = df["time"].to_numpy()
    start_idx = int(np.searchsorted(t, min_search_start_ms, side="left"))
    end_idx   = len(t) if min_search_end_ms is None else int(np.searchsorted(t, min_search_end_ms, side="right"))
    mins, tmins = {}, {}
    for name, conc in parse_kcl_columns(df.columns):
        y = df[name].to_numpy()
        if manual_t0_by_conc and conc in manual_t0_by_conc:
            t0 = float(manual_t0_by_conc[conc])
            y0 = interp_at_time(t, y, t0)
            mins[name], tmins[name] = y0, t0
        else:
            seg = y[start_idx:end_idx]
            idxseg = int(np.nanargmin(seg))
            idx = start_idx + idxseg
            mins[name], tmins[name] = float(y[idx]), float(t[idx])
    return mins, tmins

def compute_preflash_baselines(df, pre_start_ms=None, pre_end_ms=-5050.0):
    """Mean before actinic ON; by default all times <= -5050 ms."""
    t = df["time"].to_numpy()
    if pre_start_ms is None:
        mask = (t <= pre_end_ms)
    else:
        mask = (t >= pre_start_ms) & (t <= pre_end_ms)
    bases = {}
    for name, _ in parse_kcl_columns(df.columns):
        bases[name] = float(df.loc[mask, name].mean())
    return bases

def interp_crossing_time(t, y, target, start_index):
    for i in range(start_index, len(y)-1):
        y0, y1 = y[i], y[i+1]
        if (y0 <= target <= y1) or (y1 <= target <= y0):
            if y1 == y0: return float(t[i])
            f = (target - y0) / (y1 - y0)
            return float(t[i] + f*(t[i+1] - t[i]))
    return np.nan

def compute_txx_to_baseline(df, baselines, mins, tmins, fraction=0.10):
    """Time to recover 'fraction' of the step from min to the DARK pre-flash baseline."""
    out = {}
    t = df["time"].to_numpy()
    for name, _ in parse_kcl_columns(df.columns):
        y = df[name].to_numpy()
        y_min = mins[name]
        y_base = baselines[name]
        t0 = tmins[name]
        # If baseline is below min (shouldn't happen, but be safe) skip
        if np.isclose(y_base, y_min, atol=1e-9):
            out[name] = np.nan
            continue
        target = y_min + fraction * (y_base - y_min)
        start_idx = int(np.searchsorted(t, t0, side="right"))
        tcross = interp_crossing_time(t, y, target, start_idx)
        out[name] = np.nan if np.isnan(tcross) else float(tcross - t0)
    return out

def extract_metrics_preflashref(df,
                                pre_start_ms=None, pre_end_ms=-5050.0,
                                min_search_start_ms=2.0, min_search_end_ms=None,
                                fractions=(0.10, 0.50, 0.63)):
    """Compute t_f for each fraction toward the pre-flash baseline + tau from t10%."""
    baselines = compute_preflash_baselines(df, pre_start_ms, pre_end_ms)
    mins, tmins = compute_minima_custom(df, min_search_start_ms, min_search_end_ms)
    # Compute all requested fractions
    tf_dicts = {f: compute_txx_to_baseline(df, baselines, mins, tmins, fraction=f) for f in fractions}
    rows = []
    for name, conc in parse_kcl_columns(df.columns):
        rec = {
            "condition": name,
            "KCl_M": conc,
            "baseline_preflash": baselines[name],
            "y_min": mins[name],
            "t_min_ms": tmins[name],
        }
        for f in fractions:
            rec[f"t{int(f*100)}_ms_preflashref"] = tf_dicts[f][name]
        # Optional tau from t10% assuming single exponential
        t10 = rec.get("t10_ms_preflashref", np.nan)
        rec["tau_ms_from_t10"] = (t10 / 0.10536051565782628) if not np.isnan(t10) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("KCl_M")

# ========= Load your CSV (long file) =========
df = pd.read_csv("jts_data_long.csv", decimal=",", sep=";")

# ========= Metrics: use DARK pre-flash baseline & 10% crossing =========
# Pre-flash window ends at -5050 ms (adjust if your actinic ON time is slightly different)
metrics = extract_metrics_preflashref(
    df,
    pre_start_ms=None,        # or e.g. -5600
    pre_end_ms=-5050.0,
    min_search_start_ms=2.0,  # ignore any pre-0 minima
    fractions=(0.10, 0.50, 0.63)
)

print(metrics)
metrics.to_csv("psi_metrics_preflashref_t10.csv", index=False)
print("Saved: psi_metrics_preflashref_t10.csv")

# ======= Plotting =======
# 1) Overlay of traces
plt.figure(figsize=(12, 8))
for name, _ in parse_kcl_columns(df.columns):
    plt.plot(df["time"], df[name], label=name)
plt.xlabel("Time [ms]", labelpad=16)
plt.ylabel(r'$\Delta$ OD [a.u.]')
plt.legend(fontsize=22, loc="upper left", bbox_to_anchor=(0.12, 1.0))
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("psi_traces.png", dpi=300)
plt.show()

# 2) t10% vs [KCl] and tau estimate from t10%
plt.figure(figsize=(10, 8))
plt.plot(metrics["KCl_M"], metrics["t10_ms_preflashref"], marker="o")
plt.xlabel("[KCl] (M)")
plt.ylabel(r"$\tau_{{10\%}}$  to pre-flash (ms)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("psi_t10_vs_kcl_preflashref.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(metrics["KCl_M"], metrics["tau_ms_from_t10"], marker="o")
plt.xlabel("[KCl] (M)")
plt.ylabel(r"τ estimate from $t_{10\%}$ (ms)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("psi_tau_from_t10_vs_kcl.png", dpi=300)
plt.show()

# 3) Diagnostic plot: baseline & 10% target for selected concentrations
# ---- if not already defined in your file ----
def compute_preflash_baselines(df, pre_start_ms=None, pre_end_ms=-5050.0):
    """Mean in a dark pre-flash window; by default all times <= -5050 ms."""
    t = df["time"].to_numpy()
    if pre_start_ms is None:
        mask = (t <= pre_end_ms)
    else:
        mask = (t >= pre_start_ms) & (t <= pre_end_ms)
    bases = {}
    for name, _ in parse_kcl_columns(df.columns):
        bases[name] = float(df.loc[mask, name].mean())
    return bases
# ---------------------------------------------
# ----------------- Diagnostic plot with auto vertical spacing -----------------
check_concs = [0.0, 0.3, 0.75, 1.0]  # 0.2 -> 0.3
colors = {0.0:"tab:blue", 0.3:"tab:orange", 0.75:"tab:green", 1.0:"tab:red"}

baselines = compute_preflash_baselines(df, pre_start_ms=None, pre_end_ms=-5050.0)
mins, tmins = compute_minima_custom(df, min_search_start_ms=2.0)

fig, ax = plt.subplots(figsize=(11, 8))
t = df["time"].to_numpy()

# plot traces and reference lines; collect intersection points for labeling later
baseline_labeled = False
target_labeled = False
label_points = []   # will hold tuples (x, y_target, tau10, color, name)

for name, conc in parse_kcl_columns(df.columns):
    if conc not in check_concs:
        continue

    y = df[name].to_numpy()
    col = colors[conc]
    y_min = mins[name]
    t0    = tmins[name]
    y_base = baselines[name]
    target10 = y_min + 0.10 * (y_base - y_min)

    # trace
    ax.plot(t, y, color=col, lw=2, label=name)

    # pre-flash baseline (dashed gray) – label once
    ax.axhline(
        y_base, color="0.35", lw=2, ls=(0, (6, 4)),
        label="pre-flash baseline" if not baseline_labeled else None
    )
    baseline_labeled = True

    # 10% target (dotted gray) – label once
    ax.axhline(
        target10, color="0.55", lw=2, ls=(0, (2, 3)),
        label="10% toward pre-flash" if not target_labeled else None
    )
    target_labeled = True

    # intersection with 10% target
    start_idx = int(np.searchsorted(t, t0, side="right"))
    tcross = interp_crossing_time(t, y, target10, start_idx)
    if tcross is not None and not np.isnan(tcross):
        tau10 = (tcross - t0) / 0.10536051565782628  # τ ≈ t10 / 0.10536 (single-exp)
        # draw the dot now; texts later (after we compute vertical spacing)
        ax.plot(tcross, target10, "o", ms=8, color=col, zorder=5)
        label_points.append((tcross, target10, tau10, col, name))
    else:
        ax.text(t0 + 20, y_min, "no 10% crossing", color=col,
                fontsize=16, ha="left", va="center")

# ------- auto vertical spacing for labels (keep x centered at the dot) -------
# minimum vertical gap between labels (in data units)
ymin_plot, ymax_plot = ax.get_ylim()
y_range = ymax_plot - ymin_plot
min_gap = 0.055 * y_range  # tweak as needed

# sort labels by their natural y (target10) from low to high
label_points.sort(key=lambda p: p[1])

placed_y = []
for x, y_target, tau10, col, name in label_points:
    y_lbl = y_target
    if placed_y:
        # ensure at least min_gap above the last placed label
        y_lbl = max(y_lbl, placed_y[-1] + min_gap)
    # clamp to top with a small margin
    y_lbl = min(y_lbl, ymax_plot - 0.02 * y_range)
    placed_y.append(y_lbl)

    # centered text at same x, with a light bbox for readability
    ax.text(
        x, y_lbl, fr"$\tau_{{10}} \approx {tau10:.0f}\ \mathrm{{ms}}$",
        color=col, fontsize=18, weight="bold", ha="center", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2.5)
    )

# cosmetics
ax.set_xlabel("Time [ms]", labelpad=16)
ax.set_ylabel(r"$\Delta$ OD [a.u.]")

ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right", fontsize=20)
fig.tight_layout()
fig.savefig("psi_selected_traces_preflashref_t10_annotated.png", dpi=300)
plt.show()

# -------- Zoomed plot: ONE global 10% line, big dots, yellow light window --------
def plot_zoom_single_t10(
    df,
    pre_end_ms=-5050.0,                 # end of DARK pre-flash window
    min_search_start_ms=2.0,            # ignore any minima before 2 ms
    check_concs=None,                   # tuple/list of concentrations to show; None -> all
    xlim=(-7000, 20000),
    ylim=(-1.0, -0.8),
    light_window=(-5000, 0),            # shaded illumination window
    dot_size=12,
    outfile="psi_zoom_single_t10.png",
):
    # reference: DARK pre-flash baseline per trace, and post-flash minima
    baselines = compute_preflash_baselines(df, pre_start_ms=None, pre_end_ms=pre_end_ms)
    mins, tmins = compute_minima_custom(df, min_search_start_ms=min_search_start_ms)

    # choose which traces to plot
    parsed = parse_kcl_columns(df.columns)
    if check_concs is None:
        show = parsed
    else:
        # keep original ordering from parsed but filter to requested concs
        req = set(check_concs)
        show = [(name, c) for (name, c) in parsed if c in req]

    # ONE global 10% target: use global mean of baseline and min across shown traces
    y_base_global = np.mean([baselines[name] for name, _ in show])
    y_min_global  = np.mean([mins[name]       for name, _ in show])
    target10 = y_min_global + 0.10 * (y_base_global - y_min_global)

    fig, ax = plt.subplots(figsize=(12, 8))
    t = df["time"].to_numpy()

    # yellow background for the illumination window
    ax.axvspan(light_window[0], light_window[1], facecolor="#FFEB3B", alpha=0.25, zorder=0)

    # draw the single 10% dotted line across the full x-range and label it inline
    ax.hlines(target10, xlim[0], xlim[1], color="0.35", lw=2.0, linestyles=(0, (2, 3)))
    yspan = ylim[1] - ylim[0]
    ax.text(
        x=xlim[0] + 0.11 * (xlim[1] - xlim[0]),
        y=target10 + 0.015 * yspan,
        s="10% toward pre-flash",
        color="0.25",
        fontsize=12,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        zorder=3,
    )



    # plot traces + intersection dots, and put tau10 in the legend
    legend_entries = []
    for name, conc in show:
        y = df[name].to_numpy()
        t0 = tmins[name]

        # crossing of this trace with the single global 10% line
        start_idx = int(np.searchsorted(t, t0, side="right"))
        tcross = interp_crossing_time(t, y, target10, start_idx)
        if tcross is not None and not np.isnan(tcross):
            t10_ms = float(tcross - t0)
            tau10_ms = t10_ms / 0.10536051565782628  # τ ≈ t10 / 0.10536 (single-exp)
            leg = fr"{name} — $\tau_{{10}}\approx {tau10_ms:.0f}\ \mathrm{{ms}}$"
        else:
            leg = fr"{name} — $\tau_{{10}}:$ n/a"

        # trace
        line, = ax.plot(t, y, lw=2.2, label=leg)
        col = line.get_color()

        # intersection dot
        if tcross is not None and not np.isnan(tcross):
            ax.plot(tcross, target10, "o", ms=dot_size, color=col, zorder=5)

    # axes + cosmetics
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time [ms]", labelpad=16)
    ax.set_ylabel(r"$\Delta$ OD [a.u.]")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="lower right", fontsize=15, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.show()

# Example: include all concentrations
all_concs = tuple(conc for _, conc in parse_kcl_columns(df.columns))
plot_zoom_single_t10(df, check_concs=all_concs)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_methods_schematic_like_data(
    xlim=(-7000, 12000),
    ylim=(-1.05, 0.05),
    light_on=(-5000, 0),   # actinic ON window
    y_base=0.0,            # pre-flash baseline
    y_min=-1.0,            # level during illumination (after fast depletion)
    y_inf=-0.15,           # long-time level after light OFF (keeps it below baseline)
    tau_drop=200.0,        # fast decay at light ON (ms)
    tau_recover=6500.0,    # slow recovery after light OFF (ms)
    outfile="methods_t63_like_data.png",
):
    # time grid
    t = np.linspace(xlim[0], xlim[1], 4000)

    # piecewise trace: baseline -> fast drop at light ON -> flat -> slow recovery after OFF
    y = np.full_like(t, y_base, dtype=float)
    m_on  = (t >= light_on[0]) & (t < light_on[1])
    m_off = (t >= light_on[1])

    # fast approach to y_min during the ON window
    y[m_on] = y_min + (y_base - y_min) * np.exp(-(t[m_on] - light_on[0]) / tau_drop)
    # slow recovery after OFF toward y_inf
    y[m_off] = y_inf + (y_min - y_inf) * np.exp(-(t[m_off] - light_on[1]) / tau_recover)

    # 63% toward the PRE-FLASH baseline (0)
    target63 = y_min + 0.63 * (y_base - y_min)

    # crossing time during the OFF exponential segment (t>=0)
    ratio = (target63 - y_inf) / (y_min - y_inf)
    if not (0 < ratio < 1):
        raise RuntimeError("63% target not reachable with these parameters; tweak y_inf/tau_recover.")
    tcross = -tau_recover * np.log(ratio)   # since OFF is at 0 ms
    t63 = tcross                            # time since minimum

    # ---------- plot ----------
    plt.style.use('seaborn-v0_8-poster')
    fig, ax = plt.subplots(figsize=(11, 8))

    # yellow illumination window
    ax.axvspan(light_on[0], light_on[1], facecolor="#FFEB3B", alpha=0.25, zorder=0)

    # reference lines + labels
    ax.axhline(y_base, color="0.35", lw=1.8, ls=(0, (6, 4)))
    ax.text(xlim[0] + 0.03*(xlim[1]-xlim[0]), y_base + 0.02*(ylim[1]-ylim[0]),
            "pre-flash baseline", color="0.35", fontsize=12,
            ha="left", va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.2))

    ax.axhline(target63, color="0.25", lw=2.0, ls=(0, (2, 3)))
    ax.text(xlim[0] + 0.03*(xlim[1]-xlim[0]), target63 + 0.02*(ylim[1]-ylim[0]),
            "63% toward pre-flash", color="0.25", fontsize=12,
            ha="left", va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.2))

    # smooth trace + vertical cues
    ax.plot(t, y, lw=3.0, color="tab:blue")
    ax.axvline(light_on[0], color="0.4", lw=1.2)
    ax.axvline(light_on[1], color="0.4", lw=1.2)

    # intersection dot
    ax.plot(tcross, target63, "o", ms=10, color="tab:red", zorder=5)

    # --- t63 arrow/text INSIDE plot using mathtext for tau ---
    yspan = ylim[1] - ylim[0]
    y_br = target63 - 0.12 * yspan
    y_br = np.clip(y_br, ylim[0] + 0.05*yspan, ylim[1] - 0.05*yspan)

    ax.annotate("", xy=(0, y_br), xytext=(tcross, y_br),
                arrowprops=dict(arrowstyle="<->", lw=2, color="0.2"))
    ax.text(0.5*tcross, y_br + 0.03*yspan,
            fr"$\tau_{{63}}\  \;=\;{t63:.0f}\ \mathrm{{ms}}$",
            ha="center", va="bottom", fontsize=14, weight="bold", color="0.15")

    # axes & labels (use mathtext for Delta)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(r"$\Delta$OD [a.u.]")  # Greek Δ via mathtext
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.show()

# Run it
plot_methods_schematic_like_data()



# -*- coding: utf-8 -*-
# CV parsing + DC-centering (single-offset) peak extraction + summary export

import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import linregress

# ---------- console unicode ----------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- units ----------
def in_mikroamper(df, column_name):
    """Convert A -> µA by ×1e6 (done as 3×100 to keep dtype stable)."""
    df = df.copy()
    for _ in range(3):
        df[column_name] = df[column_name] * 100
    return df

# ======================================================================
#                           FILE PARSERS
# ======================================================================
def _process_classic_file(filepath, A, figure_dir):
    """Fallback reader for old 'Time/Potential/Current' ASCII export."""
    col_names = ['Number', 'Time/s', 'Potential/V', 'Current/A']
    dataR = pd.read_csv(filepath, skiprows=19, sep=r'\s+', names=col_names, header=None)

    # ensure numeric
    dataR['Current/A'] = pd.to_numeric(dataR['Current/A'], errors='coerce')
    dataR = dataR.dropna(subset=['Current/A'])
    dataR['Current/A'] = dataR['Current/A'].astype('float32')

    # A -> µA and normalize by area (µA/cm²)
    dataC = in_mikroamper(dataR, 'Current/A')
    dataC['Current/A'] = dataC['Current/A'] / A

    # derive cycles from voltage apex (original logic)
    cycle_no = 0
    cycles = []
    highest_voltage = dataC['Potential/V'].max()
    for i in range(len(dataC)):
        if i > 0 and dataC.loc[i, 'Potential/V'] == highest_voltage and dataC.loc[i-1, 'Potential/V'] != highest_voltage:
            cycle_no += 1
        cycles.append(cycle_no)
    dataC['Cycle_No'] = cycles

    # keep interior cycles only
    max_cycle = dataC['Cycle_No'].max()
    dataF = dataC[(dataC['Cycle_No'] >= 1) & (dataC['Cycle_No'] < max_cycle)].reset_index(drop=True)

    # raw plot
    base = os.path.splitext(os.path.basename(filepath))[0]
    figp = os.path.join(figure_dir, f"{base}_raw.png")
    plt.figure(figsize=(10, 6))
    plt.plot(dataF['Potential/V'], dataF['Current/A'], color='blue', lw=1.3, label='CV')
    plt.xlabel('Voltage (V)'); plt.ylabel('Current (µA/cm²)')
    plt.title('Cyclic Voltammetry (raw)')
    plt.legend(); plt.grid(True, alpha=0.3)
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(figp, dpi=200, bbox_inches='tight'); plt.close()
    return dataF

def _process_scanblock_file(filepath, A, figure_dir, file_text=None):
    """
    Parse 'Potential   Scan N' blocks (Autolab-style).
    Assign Cycle_No from N; export a raw overlay for sanity checking.
    """
    if file_text is None:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            file_text = f.read()

    lines = file_text.splitlines()

    # try to infer dt ~ step/scanrate (optional)
    step_val = None; scan_rate = None
    for ln in lines[:120]:
        m = re.search(r'Step\s*potential\s*\(V\)\s*=\s*([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)', ln, re.I)
        if m: step_val = float(m.group(1))
        m = re.search(r'Scan\s*rate\s*\(V/s\)\s*=\s*([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)', ln, re.I)
        if m: scan_rate = float(m.group(1))
    dt = (step_val / scan_rate) if (step_val is not None and scan_rate not in (None, 0)) else None

    recs = []  # Potential/V, Current/A, Cycle_No, Time/s
    current_scan = None
    k_in_scan = 0

    for ln in lines:
        h = re.search(r'^\s*Potential\s+Scan\s+(\d+)', ln, re.I)
        if h:
            current_scan = int(h.group(1))
            k_in_scan = 0
            continue
        if current_scan is None or not ln.strip():
            continue

        parts = ln.strip().split()
        if len(parts) < 2:
            continue
        p0 = parts[0].replace('D','E'); p1 = parts[1].replace('D','E')
        try:
            pot = float(p0); cur = float(p1)
        except ValueError:
            continue
        t = (k_in_scan * dt) if dt is not None else np.nan
        recs.append((pot, cur, current_scan, t))
        k_in_scan += 1

    if not recs:
        raise ValueError("No numeric data rows found under 'Potential  Scan N' headers.")

    df = pd.DataFrame(recs, columns=['Potential/V','Current/A','Cycle_No','Time/s'])
    df['Current/A'] = pd.to_numeric(df['Current/A'], errors='coerce')
    df = df.dropna(subset=['Current/A'])
    df['Current/A'] = df['Current/A'].astype('float32')

    # A -> µA; normalize by area
    df = in_mikroamper(df, 'Current/A')
    df['Current/A'] = df['Current/A'] / A

    dataF = df.reset_index(drop=True)

    # overlay by scan
    base = os.path.splitext(os.path.basename(filepath))[0]
    figp = os.path.join(figure_dir, f"{base}_raw.png")
    plt.figure(figsize=(10, 6))
    for c, sub in dataF.groupby('Cycle_No'):
        plt.plot(sub['Potential/V'], sub['Current/A'], lw=1.0, alpha=0.85, label=f'Scan {c}')
    plt.xlabel('Voltage (V)'); plt.ylabel('Current (µA/cm²)')
    plt.title('CV (parsed by Scan header)')
    plt.legend(ncol=2, fontsize=9); plt.grid(True, alpha=0.3)
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(figp, dpi=200, bbox_inches='tight'); plt.close()
    return dataF

def process_file(filepath, A, figure_dir):
    """Use scan-block parser if present, else classic CSV parser."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    if re.search(r'^\s*Potential\s+Scan\s+\d+', txt, re.M | re.I):
        return _process_scanblock_file(filepath, A, figure_dir, file_text=txt)
    return _process_classic_file(filepath, A, figure_dir)

# ======================================================================
#                    LOOP SPLIT + PER-PHASE AVERAGING
# ======================================================================
def _split_into_half_cycles(df, include_return=True):
    """
    Split each cycle into 2–3 monotonic halves using potential vertices.
    Label halves as 'anodic' (dE>0) or 'cathodic' (dE<0).
    """
    halves = []
    for c, cdf in df.groupby('Cycle_No'):
        key = 'Time/s' if 'Time/s' in cdf.columns else None
        cdf = cdf.sort_values(key).reset_index(drop=True)
        E = cdf['Potential/V'].to_numpy()
        if len(E) < 5:
            continue
        dE = np.diff(E)
        nlook = max(5, min(50, len(dE)//10 or 5))
        init_dir = np.sign(np.nanmedian(dE[:nlook])) if len(dE) else 1.0
        imax = int(np.nanargmax(E)); imin = int(np.nanargmin(E))
        if init_dir >= 0:
            i_first, i_second = imax, imin
        else:
            i_first, i_second = imin, imax
        i_first  = int(np.clip(i_first, 1, len(cdf)-2))
        i_second = int(np.clip(i_second, i_first+1, len(cdf)-1))

        hA = cdf.iloc[:i_first+1].copy()
        hA['half_label'] = 'anodic' if (hA['Potential/V'].iloc[-1] > hA['Potential/V'].iloc[0]) else 'cathodic'
        hA['half_idx']   = f"{int(c)}A"; halves.append(hA)

        hB = cdf.iloc[i_first:i_second+1].copy()
        hB['half_label'] = 'anodic' if (hB['Potential/V'].iloc[-1] > hB['Potential/V'].iloc[0]) else 'cathodic'
        hB['half_idx']   = f"{int(c)}B"; halves.append(hB)

        if include_return and i_second < len(cdf)-1:
            hC = cdf.iloc[i_second:].copy()
            hC['half_label'] = 'anodic' if (hC['Potential/V'].iloc[-1] > hC['Potential/V'].iloc[0]) else 'cathodic'
            hC['half_idx']   = f"{int(c)}C"; halves.append(hC)
    return pd.concat(halves, ignore_index=True)

def _common_grid(halves_phase, step=None, strategy="union"):
    if step is None:
        dEs = []
        for _, h in halves_phase:
            e = h['Potential/V'].to_numpy()
            if len(e) > 1:
                d = np.diff(e)
                dEs.extend(np.abs(d[np.isfinite(d) & (np.abs(d) > 0)]))
        step = np.median(dEs) if len(dEs) else 0.001

    mins = [h['Potential/V'].min() for _, h in halves_phase]
    maxs = [h['Potential/V'].max() for _, h in halves_phase]
    if strategy == "union":
        emin, emax = min(mins), max(maxs)
    else:
        emin, emax = max(mins), min(maxs)
        if emin >= emax:
            emin, emax = min(mins), max(maxs)

    n = max(5, int(round((emax - emin)/step)) + 1)
    grid = np.linspace(emin, emax, n)
    return grid, step

def _interp_matrix(halves_phase, grid):
    mats = []
    for _, h in halves_phase:
        x = h['Potential/V'].to_numpy()
        y = h['Current/A'].to_numpy()
        order = np.argsort(x)
        x = x[order]; y = y[order]
        yi = np.interp(grid, x, y, left=np.nan, right=np.nan)
        mats.append(yi)
    return np.vstack(mats)

def _phase_stats(half_df, phase):
    halves_phase = list(half_df.groupby('half_idx'))
    grid, _ = _common_grid(halves_phase)
    M = _interp_matrix(halves_phase, grid)
    mean = np.nanmean(M, axis=0)
    std  = np.nanstd(M, axis=0, ddof=1)
    out = pd.DataFrame({"Potential/V": grid, "mean": mean, "std": std})
    out['half_label'] = phase
    out['Time/s'] = np.arange(len(out), dtype=float)  # synthetic
    return out

def compute_combined_stats(dataF, filepath, figure_dir, smooth_win=3, smooth_win_std=5):
    halves = _split_into_half_cycles(dataF, include_return=True)
    anodic_df   = _phase_stats(halves[halves['half_label']=='anodic'], 'anodic')
    cathodic_df = _phase_stats(halves[halves['half_label']=='cathodic'], 'cathodic')
    combined_stats = pd.concat([anodic_df, cathodic_df], ignore_index=True)

    # mild smoothing (keeps endpoints)
    if smooth_win and smooth_win > 1:
        combined_stats['mean'] = combined_stats.groupby('half_label')['mean'] \
            .transform(lambda s: s.rolling(smooth_win, center=True, min_periods=1).mean())
    if smooth_win_std and smooth_win_std > 1:
        combined_stats['std'] = combined_stats.groupby('half_label')['std'] \
            .transform(lambda s: s.rolling(smooth_win_std, center=True, min_periods=1).mean())

    combined_stats = combined_stats[combined_stats['mean'].notna()].reset_index(drop=True)

    # quick visual
    base = os.path.splitext(os.path.basename(filepath))[0]
    fig_path = os.path.join(figure_dir, f"{base}_smooth.png")
    plt.figure(figsize=(10,6))
    for phase, sub in combined_stats.groupby('half_label', sort=False):
        asc = (phase == 'anodic')
        subp = sub.sort_values('Potential/V', ascending=asc)
        plt.plot(subp['Potential/V'], subp['mean'], lw=1.5, label=f'{phase.capitalize()} mean')
        plt.fill_between(subp['Potential/V'], subp['mean']-3*subp['std'], subp['mean']+3*subp['std'], alpha=0.16)
    plt.xlabel('Voltage (V)'); plt.ylabel('Current (µA/cm²)')
    plt.title('Averaged currents ±3σ'); plt.legend(); plt.grid(True, alpha=0.3)
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    return combined_stats, halves

# ======================================================================
#                      DC-CENTERING (single offset)
# ======================================================================
def _center_by_E0(combined_stats, E0, window_V=0.02):
    """
    Compute a single offset I0 from a small window around E0 using both branches,
    subtract it from the 'mean' current, and return (centered_df, I0).
    """
    cs = combined_stats.copy()
    vals = []
    for name in ("anodic", "cathodic"):
        sub = cs[cs["half_label"] == name].sort_values("Potential/V")
        if sub.empty:
            continue
        v = sub["Potential/V"].to_numpy()
        i = sub["mean"].to_numpy()
        if window_V > 0:
            m = (v >= (E0 - window_V/2)) & (v <= (E0 + window_V/2))
            if m.sum() >= 3:
                vals.append(float(np.nanmean(i[m])))
            else:
                vals.append(float(np.interp(E0, v, i)))
        else:
            vals.append(float(np.interp(E0, v, i)))
    I0 = float(np.nanmean(vals)) if len(vals) else 0.0
    cs["mean"] = cs["mean"] - I0
    return cs, I0

# ======================================================================
#                  PEAK PICKING on CENTERED CURVE
# ======================================================================
def analyze_peaks_auto(
    combined_stats,
    figure_dir,
    filename_hint="peak_analysis.png",
    # peak picking
    k_sigma=3.0, min_distance_pts=30,
    # DC-centering
    center_mode="E0_window",   # "none", "E0_point", "E0_window"
    center_width_V=0.02,       # window width for E0_window
    # (no baseline modeling: we take Ip directly from centered curve)
):
    """
    Returns dict with Epa/Epc (raw), single offset removed near E°, and centered Ip.
    """
    out = {}
    fig, ax = plt.subplots(figsize=(10, 6))

    # split phases, ascending potential
    phases = {}
    for phase_name in ("anodic", "cathodic"):
        ph = combined_stats[combined_stats['half_label']==phase_name].copy()
        if not ph.empty:
            phases[phase_name] = ph.sort_values('Potential/V').reset_index(drop=True)
    if not phases:
        return out

    # raw peak positions (constant offset won't change Ep indices)
    def _pick_raw_peak(ph_name):
        ph = phases[ph_name]
        v = ph['Potential/V'].to_numpy()
        i = ph['mean'].to_numpy()
        s = np.nanmedian(ph['std'].to_numpy())
        is_anodic = (ph_name == 'anodic')
        sig = i if is_anodic else (-i)
        prom = max(1e-12, k_sigma * (s if np.isfinite(s) else np.nanstd(i)))
        idxs, props = find_peaks(sig, prominence=prom, distance=min_distance_pts)
        idx = int(idxs[np.argmax(props['prominences'])]) if len(idxs) else int(np.nanargmax(sig))
        return v[idx], i[idx], idx

    have_a = "anodic" in phases
    have_c = "cathodic" in phases
    if have_a:
        Epa_raw, Ipa_raw, idx_a = _pick_raw_peak("anodic")
        out["Epa (V)"] = float(Epa_raw)
        out["Ipa_raw (µA/cm²)"] = float(Ipa_raw)
    if have_c:
        Epc_raw, Ipc_raw, idx_c = _pick_raw_peak("cathodic")
        out["Epc (V)"] = float(Epc_raw)
        out["Ipc_raw (µA/cm²)"] = float(Ipc_raw)

    # estimate E° from same scan
    E0 = None
    if have_a and have_c:
        E0 = 0.5*(Epa_raw + Epc_raw)

    # DC-centering
    I0 = 0.0
    centered = combined_stats.copy()
    if center_mode != "none" and E0 is not None:
        if center_mode == "E0_window":
            centered, I0 = _center_by_E0(combined_stats, E0, window_V=center_width_V)
        elif center_mode == "E0_point":
            centered, I0 = _center_by_E0(combined_stats, E0, window_V=0.0)
        else:
            centered = combined_stats.copy()
    out["I_offset_center (µA/cm²)"] = float(I0)
    out["centering_mode"] = center_mode

    # replace phases with centered ones for plotting & Ip evaluation
    phases = {}
    for phase_name in ("anodic","cathodic"):
        ph = centered[centered['half_label']==phase_name].copy()
        if not ph.empty:
            phases[phase_name] = ph.sort_values('Potential/V').reset_index(drop=True)

    # centered Ip at Ep
    if have_a and "anodic" in phases:
        vA = phases["anodic"]["Potential/V"].to_numpy()
        iA = phases["anodic"]["mean"].to_numpy()
        out["Ipa (µA/cm²)"] = float(np.interp(Epa_raw, vA, iA))
    if have_c and "cathodic" in phases:
        vC = phases["cathodic"]["Potential/V"].to_numpy()
        iC = phases["cathodic"]["mean"].to_numpy()
        out["Ipc (µA/cm²)"] = float(np.interp(Epc_raw, vC, iC))

    if "Epa (V)" in out and "Epc (V)" in out:
        out["Ep (V)"] = float(out["Epa (V)"] - out["Epc (V)"])
        Ipa_abs = abs(out.get("Ipa (µA/cm²)", np.nan))
        Ipc_abs = abs(out.get("Ipc (µA/cm²)", np.nan))
        if np.isfinite(Ipa_abs) and np.isfinite(Ipc_abs):
            out["I_avg (µA/cm²)"] = float((Ipa_abs + Ipc_abs)/2)
            out["I_std (µA/cm²)"] = float(np.std([Ipa_abs, Ipc_abs], ddof=1))

    # plot centered curves + zero line
    x_all_min = min(p["Potential/V"].min() for p in phases.values())
    x_all_max = max(p["Potential/V"].max() for p in phases.values())
    for name, ph in phases.items():
        v = ph["Potential/V"].to_numpy(); i = ph["mean"].to_numpy()
        ax.plot(v, i, lw=1.8, label=f"{name.capitalize()} (centered)")
        ax.fill_between(v, i - ph['std'], i + ph['std'], alpha=0.10)

    ax.axhline(0.0, color='k', linestyle='--', alpha=0.5, label="DC offset removed")
    if have_a and "anodic" in phases:
        ax.plot([out["Epa (V)"]], [out["Ipa (µA/cm²)"]], 'o', label="Anodic peak")
    if have_c and "cathodic" in phases:
        ax.plot([out["Epc (V)"]], [out["Ipc (µA/cm²)"]], 'o', label="Cathodic peak")

    ax.set_xlim(x_all_min, x_all_max)
    ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current (µA/cm²)")
    ttl = f"ΔEp = {out.get('Ep (V)', np.nan):.3f} V | DC-centering around E0"
    ax.set_title(ttl); ax.legend(ncol=2); ax.grid(True, alpha=0.3)

    base = os.path.splitext(os.path.basename(filename_hint))[0]
    outpath = os.path.join(figure_dir, f"{base}_peaks.png")
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight'); plt.close()
    return out

# ======================================================================
#                         BATCH PROCESSOR
# ======================================================================
def _parse_conc_scan_from_name(name: str):
    s = name.lower()
    m_conc = re.search(r'(\d+)\s*kcl', s)
    m_scan = re.search(r'(\d+)\s*m[v]?(?:/?s)?\b', s)  # 50mV, 50mV/s, 50mv_s, ...
    conc = int(m_conc.group(1)) if m_conc else None
    scan = int(m_scan.group(1)) if m_scan else None
    return conc, scan

def batch_process_cv(
    folder,
    A,
    figure_dir,
    center_mode="E0_window",
    center_width_V=0.02,
    summary_basename="peak_summary_centered"
):
    """
    Parse all .txt files, compute DC-centered Ipa/Ipc & Ep,
    save per-file plots and export a summary (xlsx + DE.csv).
    """
    os.makedirs(figure_dir, exist_ok=True)
    rows = []

    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".txt"):
            continue
        fp = os.path.join(folder, fn)
        try:
            dataF = process_file(fp, A, figure_dir)
            combined_stats, _ = compute_combined_stats(dataF, fp, figure_dir)

            # point estimates from DC-centered averaged curve
            res = analyze_peaks_auto(
                combined_stats,
                figure_dir=figure_dir,
                filename_hint=fn,
                center_mode=center_mode,
                center_width_V=center_width_V
            )

            row = {"File": fn}
            row.update(res)

            conc, scan = _parse_conc_scan_from_name(fn)
            if conc is not None: row["Conc"] = conc
            if scan is not None: row["Scan"] = scan

            rows.append(row)
            print("[OK]", fn)
        except Exception as e:
            print("[ERR]", fn, ":", e)

    if not rows:
        print("No results to save.")
        return None

    df = pd.DataFrame(rows)

    # preferred column order
    cols = [
        "File", "Conc", "Scan",
        "Epa (V)", "Ipa_raw (µA/cm²)", "Ipa (µA/cm²)",
        "Epc (V)", "Ipc_raw (µA/cm²)", "Ipc (µA/cm²)",
        "Ep (V)", "I_avg (µA/cm²)", "I_std (µA/cm²)",
        "I_offset_center (µA/cm²)", "centering_mode"
    ]
    df = df[[c for c in cols if c in df.columns]]

    base = os.path.join(figure_dir, summary_basename)
    df.to_excel(base + ".xlsx", index=False)
    df.to_csv(base + "_DE.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print("Saved:", base + ".xlsx")
    print("Saved:", base + "_DE.csv")
    return df

# ======================================================================
#                         OVERLAY PLOT (optional)
# ======================================================================
def _combined_to_loop_df(combined_stats: pd.DataFrame) -> pd.DataFrame:
    anod = combined_stats[combined_stats['half_label']=='anodic'] \
           .sort_values('Potential/V', ascending=True)[['Potential/V','mean','std']]
    cath = combined_stats[combined_stats['half_label']=='cathodic'] \
           .sort_values('Potential/V', ascending=False)[['Potential/V','mean','std']]
    if (not anod.empty and not cath.empty
        and np.isclose(anod['Potential/V'].iloc[-1], cath['Potential/V'].iloc[0], atol=1e-9)):
        cath = cath.iloc[1:]
    loop = pd.concat([anod, cath], ignore_index=True)
    loop.columns = ['E','I','sigma']
    return loop

def plot_overlay_2x2_meanloops(
    data_dir, A, figure_dir,
    concentrations=None, save_name='overlay_meanloops_2x2.png',
    show_sigma=False, exclude_scans_global=None, legend_loc='lower right',
    apply_centering=True, center_width_V=0.02
):
    """
    Overlay averaged loops. If apply_centering=True, a single offset around E0
    is removed before plotting (visual symmetry).
    """
    exclude_scans_global = set(exclude_scans_global or [])
    os.makedirs(figure_dir, exist_ok=True)

    bag = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith('.txt'):
            continue
        conc, scan = _parse_conc_scan_from_name(fn)
        if conc is None or scan is None or scan in exclude_scans_global:
            continue
        fp = os.path.join(data_dir, fn)
        try:
            df = process_file(fp, A, figure_dir)
            combined_stats, _ = compute_combined_stats(df, fp, figure_dir)

            if apply_centering:
                # estimate E0 from peaks on the averaged curve
                # (we reuse analyze_peaks_auto peak picking to get Epa/Epc quickly)
                tmp = analyze_peaks_auto(combined_stats, figure_dir, filename_hint=fn,
                                         center_mode="none")  # no centering yet
                if "Epa (V)" in tmp and "Epc (V)" in tmp:
                    E0 = 0.5*(tmp["Epa (V)"] + tmp["Epc (V)"])
                    combined_stats, _ = _center_by_E0(combined_stats, E0, window_V=center_width_V)

            loop = _combined_to_loop_df(combined_stats)
            bag.append((conc, scan, loop))
        except Exception as e:
            print(f"[WARN] {fn}: {e}")

    if not bag:
        print("No curves found.")
        return

    all_concs = sorted({c for c,_,_ in bag})
    if concentrations is None:
        concentrations = all_concs[:4]
    else:
        concentrations = [c for c in concentrations if c in all_concs][:4]
    if not concentrations:
        print("No matching concentrations present.")
        return

    scan_rates = sorted({s for _, s, _ in bag})
    base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    color_map = {sr: base_colors[i % len(base_colors)] for i, sr in enumerate(scan_rates)}

    xlim = (-0.3, 0.6); ylim = (-2200, 2500)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (ax, conc) in enumerate(zip(axes, concentrations)):
        items = sorted([(scan, loop) for c, scan, loop in bag if c == conc], key=lambda x: x[0])
        for scan, loop in items:
            ax.plot(loop['E'], loop['I'], lw=2.0, label=f"{scan} mV/s", color=color_map[scan])
            if show_sigma and 'sigma' in loop.columns:
                ax.fill_between(loop['E'], loop['I']-3*loop['sigma'], loop['I']+3*loop['sigma'],
                                alpha=0.10, edgecolor='none', color=color_map[scan])
        ax.set_title(f"{conc} mM (centered)" if apply_centering else f"{conc} mM")
        if idx in (2,3): ax.set_xlabel("Voltage (V)")
        if idx in (0,2): ax.set_ylabel("Current (µA/cm²)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.legend(loc=legend_loc, fontsize=9, ncol=1)

    for k in range(len(concentrations), 4):
        fig.delaxes(axes[k])

    fig.tight_layout()
    out = os.path.join(figure_dir, save_name)
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show()
    print("Saved:", out)

# ======================================================================
#                              EXAMPLE RUN
# ======================================================================
if __name__ == "__main__":
    # paths
    data_dir   = r"C:/Users/patri/Coding/Master/CV/KHCF/new/"
    figure_dir = r"C:/Users/patri/Coding/Master/figures/CV/KHCF/autolab/"
    A = 0.282  # electrode area [cm²]

    # ---- batch (writes peak_summary_centered.xlsx and _DE.csv) ----
    summary = batch_process_cv(
        folder=data_dir,
        A=A,
        figure_dir=figure_dir,
        center_mode="E0_window",   # single offset from a tiny window around E0
        center_width_V=0.02,       # ±10 mV window (tweak if your grid is coarse)
        summary_basename="peak_summary_centered"
    )

    # ---- optional overlay figure (nice for symmetry check) ----
    plot_overlay_2x2_meanloops(
        data_dir=data_dir,
        A=A,
        figure_dir=figure_dir,
        concentrations=[0, 200, 400, 800],
        exclude_scans_global=set(),   # e.g. {1000}
        show_sigma=False,
        apply_centering=True,         # show the DC-centered loops
        center_width_V=0.02,
        save_name='overlay_meanloops_2x2_centered.png'
    )

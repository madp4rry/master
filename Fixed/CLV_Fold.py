#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import math
import numpy as np
import warnings
import os
import glob
from matplotlib import rcParams
# Set the font to "Rubik" and adjust font sizes
plt.style.use('seaborn-v0_8-poster')
rcParams['font.family'] = 'Rubik'
rcParams['axes.labelsize'] = 34
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 26



data_dir = "C:/Users/patri/Coding/Master/data/Measure_Chopped_LV/2024_08_23/"
figure_dir = "C:/Users/patri/Coding/Master/figures/CLV_2024_08_23/"
#file = "2024_08_23_0032_old_powder.txt"
#pfad = data_dir + file

# if figure dir are not existing, create them
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

PS1c="22"
Layer="8"
Doping="1,2"
Buffer="KPP  (5 mM; pH 7.0)"
# Set the style of matplotlib to 'ggplot'


txt_files = glob.glob(data_dir + "*.txt")


def in_mikroamper(df, column_name):
    """
    Multipliziert die Werte in einer bestimmten Spalte eines Pandas DataFrame mit 100 (insgesamt drei mal).

    Args:
        df (DataFrame): Das Eingabe-Datenframe.
        column_name (str): Der Name der Spalte, deren Werte multipliziert werden sollen.

    Returns:
        DataFrame: Ein neues DataFrame mit den multiplizierten Werten.
    """
    df_adj = df.copy()
    for _ in range(3):
        df_adj[column_name] = df_adj[column_name] * 100
    return df_adj

#Preprocessing




def plot_striped_voltage_current(data, xlabel='Voltage (V)', ylabel='Current (A)', caption=None):

    # Step 2: Fit a baseline to the dark data ----------------------------------------------------------------
    N = 20  # Data Points to exclude from the fit
    dark_data = data[data['IsDark']][N:]
    degree = 7  # Change this to adjust the degree of the polynomial
    coeffs = np.polyfit(dark_data['Voltage/V'], dark_data['Current/A'], degree)

    # Use the same slicing for np.polyval
    baseline = np.polyval(coeffs, data['Voltage/V'][N:])
    data['BaselineCorrectedCurrent/A'] = data['Current/A'][N:] - baseline
    #crop ONLY current  column to start at N
    data.loc[:N - 1, 'Current/A'] = np.nan


    phases = []
    for i in range(int(len(data)/100)):
        phases.extend([i]*100)
    data['Phase'] = phases

    if len(phases) == len(data):
        data['Phase'] = phases
    else:
        print("Length of phases does not match length of data.")

    photocurrents = []
    # split df by phases
    phase_groups = data.groupby('Phase')
    for name, group in phase_groups:
        # count rows with IsDark == True, if not 50 warn
        if len(group[group['IsDark']]) != 50:
            warnings.warn(f"Phase {name} doesn't have 50 dark rows!!")

        # separate dark and light rows
        dark = group[group['IsDark']]
        light = group[~group['IsDark']]

        # get maximum value of dark rows
        max_dark = dark['BaselineCorrectedCurrent/A'].max()
        #get corresponding voltage
        max_dark_voltage = dark[ dark['BaselineCorrectedCurrent/A'] == max_dark]['Voltage/V'].values[0]

        # get minimum value of light rows
        min_light = light['BaselineCorrectedCurrent/A'].min()
        # get corresponding voltage
        min_light_voltage = light[ light['BaselineCorrectedCurrent/A'] == min_light]['Voltage/V'].values[0]

        # calculate photocurrent
        PC = max_dark - min_light
        photocurrents.append({
            'Phase': name,
            'Photocurrent': PC,
            'MaxDark': max_dark,
            'MaxDarkVoltage': max_dark_voltage,
            'MinLight': min_light,
            'MinLightVoltage': min_light_voltage
        })

    # make df with columns phase and photocurrent
    pc_df = pd.DataFrame(photocurrents)

    # add rank column
    pc_df['Rank'] = pc_df['Photocurrent'].rank(ascending=False)
    print(pc_df)
    pc_Max = pc_df['Photocurrent'].max()


    # Step 5: Plot the data -----------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.plot(data['Voltage/V'], data['Current/A'], color='black', linewidth=0.75, label='31uM PS1 + 1mM CytC| Photocurrent = ' + str(pc_Max) + ' µA/cm²')
    plt.plot(data['Voltage/V'][N:], baseline, label='Fitted Baseline', color='grey', linewidth=0.5)
    plt.plot(data['Voltage/V'], data['BaselineCorrectedCurrent/A'], label='Corrected Data', color='green', linewidth=2.5)


    # add ranks 1, 2 and 3 to phases in plot
    ax = plt.gca()
    ylims = ax.get_ylim()
    yspan = ylims[1] - ylims[0]

    # if ties are possible, sort instead of == rank
    top3 = pc_df.sort_values('Photocurrent', ascending=False).head(3).reset_index(drop=True)

    offsets = [0.08, 0.03, -0.02]  # fractions of y-span for ranks 1,2,3

    for i in range(len(top3)):
        row = top3.iloc[i]
        x_mid = 0.5 * (row['MaxDarkVoltage'] + row['MinLightVoltage'])
        y_mid = 0.5 * (row['MaxDark'] + row['MinLight'])
        y_text = y_mid + offsets[i] * yspan

        #ax.text(x_mid, y_text, f"{i + 1}.", ha='center', va='bottom', fontweight='bold', fontsize=16)
        #ax.text(x_mid, y_text - 0.02 * yspan, f"{row['Photocurrent']:.2f} µA/cm²",
                #ha='center', va='top', fontweight='bold', fontsize=16)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.02)
    plt.ticklabel_format(style='plain')

    farben = ['#FFA500', '#FFFFFF']
    #  farben = farben[::-1]

    min_voltage = data['Voltage/V'].min()
    max_voltage = data['Voltage/V'].max()
    # Define the width of each stripe
    streifen_breite = 25 * 10 ** -3

    # Calculate the total range of your voltage data
    total_range = max_voltage - min_voltage

    # Calculate the number of stripes
    num_streifen = int(total_range / streifen_breite)

    # Create each stripe
    for i in range(num_streifen):
        # Calculate the start and end of each stripe
        start = min_voltage + i * streifen_breite
        end = min_voltage + (i + 1) * streifen_breite
        # Determine the color of the stripe
        farbe_index = i % len(farben)
        # Add the stripe to the plot
        plt.axvspan(start, end, color=farben[farbe_index], alpha=0.25)

    plt.xlim(min_voltage, max_voltage)

    if caption:
        fig.text(0.5, 0.08, caption, ha='center', va='top', fontsize=14)

        # Adjust the layout to make room for the caption
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # adjust the bottom parameter to make room for the caption

    base_name = os.path.basename(txt_file)
    plt.savefig(figure_dir + base_name  + ".png",bbox_inches='tight', dpi=300)
    plt.show()






txt_files = glob.glob(data_dir + "*.txt")
for txt_file in txt_files:
    print(txt_file)
    filename = txt_file.split("\\")[-1]
    print(filename)
    dataR = pd.read_csv(txt_file, skiprows=13, sep='\s+')

    initial_length = len(dataR)
    while len(dataR) % 100 != 0:
        dataR = dataR.drop(dataR.index[0])
    final_length = len(dataR)
    rows_removed = initial_length - final_length
    print(f"{rows_removed} rows were removed to make the data length divisible by 100.")

    dataC = in_mikroamper(dataR, 'Current/A')
    dataC['Current/A'] = dataC['Current/A'] / 0.047
    # Step 1: Identify the dark periods
    dataC.rename(columns={dataC.columns[4]: 'Intensity'}, inplace=True)
    dataC['IsDark'] = dataC['Intensity'] == 0
    caption = f"""Abbildung_{os.path.basename(txt_file)}: Chopped-Light-Voltammetrie einer ITO-PS1-CytC-Elektrode ({PS1c} µM PS1, 1 mM CytC);  hergestellt aus {Layer} Schichten ITO-Precursor mit {Doping} % Tin; Messung in {Buffer}; Lichtperiode: 10 s; Scanrate: 5 mV/s; Lichtintensität: 1000 W/m²; Startpotential: 0,1 V; Endpotential: -0,5 V.)"""
    wrapped_caption = "\n".join(textwrap.wrap(caption, width=175))

    plot_striped_voltage_current(
        dataC,
        xlabel='Potential [V]',
        ylabel='Photocurrent  [µA/cm²]',
        #caption=wrapped_caption
    )





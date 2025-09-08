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

data_dir = "C:/Users/patri/Coding/Master/data/Measure_Chopped_LV/"
figure_dir = "C:/Users/patri/Coding/Master/figures/"
file = "2024_08_23_0032_old_powder.txt"
pfad = data_dir + file

# if figure dir are not existing, create them
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# Set the style of matplotlib to 'ggplot'
plt.style.use('ggplot')

"""
files = glob.glob(data_dir + "*.txt")
for file in files:
    print(file)
    filename = file.split("\\")[-1]
    print(filename)
"""

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

dataR = pd.read_csv(pfad, skiprows=13, delim_whitespace=True)

dataC = in_mikroamper(dataR, 'Current/A')
dataC['Current/A'] = dataC['Current/A'] / 0.031
 # Step 1: Identify the dark periods
dataC.rename(columns={dataC.columns[4]: 'Intensity'}, inplace=True)
dataC['IsDark'] = dataC['Intensity'] == 0


def plot_striped_voltage_current(data, xlabel='Voltage (V)', ylabel='Current (A)', caption=None):

    # Step 2: Fit a baseline to the dark data ----------------------------------------------------------------
    dark_data = data[data['IsDark']][30:]
    degree = 8  # Change this to adjust the degree of the polynomial
    coeffs = np.polyfit(dark_data['Voltage/V'], dark_data['Current/A'], degree)

    # Step 3: Subtract the fitted baseline from the entire dataset
    baseline = np.polyval(coeffs, data['Voltage/V'])
    data['BaselineCorrectedCurrent/A'] = data['Current/A'] - baseline

    phases = []
    for i in range(int(len(data)/100)):
        phases.extend([i]*100)
    data['Phase'] = phases

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

    # Step 4: Calculate the photocurrent ---------------------------------------------------------------------
    x_min = -0.38  # your minimum x-value
    x_max = -0.366  # your maximum x-value

    # Filter the DataFrame to only include rows where the x-value is within the desired range
    filtered_data = data[(data['Voltage/V'] >= x_min) & (data['Voltage/V'] <= x_max)]

    # Calculate the difference between the maximum and minimum y-values within this filtered DataFrame
    PC = filtered_data['BaselineCorrectedCurrent/A'].max() - filtered_data['BaselineCorrectedCurrent/A'].min()

    print(f"Photocurrent between x={x_min} and x={x_max} is {PC} µA/cm²")


    # Step 5: Plot the data -----------------------------------------------------------------------------------
    plt.figure(figsize=(20, 10))
    plt.plot(data['Voltage/V'], data['Current/A'], color='black', linewidth=0.75, label='31uM PS1 + 1mM CytC | Photocurrent = ' + str(PC) + ' µA/cm²')
    plt.plot(data['Voltage/V'], baseline, label='Fitted Baseline', color='grey', linewidth=0.5)
    plt.plot(data['Voltage/V'], data['BaselineCorrectedCurrent/A'], label='Corrected Data', color='green', linewidth=1)
    plt.axvline(x=x_min, color='g', linestyle='--', linewidth=0.2)
    plt.axvline(x=x_max, color='g', linestyle='--', linewidth=0.2)

    # add ranks 1, 2 and 3 to phases in plot
    for i in range(3):
        rank = pc_df[pc_df['Rank'] == i+1]
        ylims = plt.gca().get_ylim()
        y_position = ylims[0] + 0.5 * (ylims[1] - ylims[0])
        x_position = np.linspace(rank['MaxDarkVoltage'], rank['MinLightVoltage'], 3)[1]
        plt.text(x_position, y_position, f"{i+1}.", ha='center', va='center', fontweight='bold', color='k')
        # add text with photocurrent below rank
        y_position_pc = ylims[0] + 0.48 * (ylims[1] - ylims[0])
        plt.text(x_position, y_position_pc, f"{rank['Photocurrent'].values[0]:.2f} µA/cm²", ha='center', va='center', fontweight='bold', color='k')

        # add points at the max dark and min light values
        plt.scatter(rank['MaxDarkVoltage'], rank['MaxDark'], color='red', s=60, zorder=10)
        plt.scatter(rank['MinLightVoltage'], rank['MinLight'], color='blue', s=60, zorder=10)
        # add lines between points
        plt.plot([rank['MaxDarkVoltage'], rank['MinLightVoltage']], [rank['MaxDark'], rank['MinLight']], color='black', linewidth=0.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')

    if caption:
        plt.text(-0.6, -1345, caption, ha='left', wrap=True)
    plt.grid(True)
    plt.ticklabel_format(style='plain')

    farben = ['#FFA500', '#DDDDDD']
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
        plt.axvspan(start, end, color=farben[farbe_index], alpha=0.4)

    plt.xlim(min_voltage, max_voltage)




    plt.savefig(figure_dir + file  + ".png")
    plt.show()








# In[40]:




caption = """Abbildung_X: Chopped-Light-Voltammetrie einer ITO-PS1-CytC-Elektrode (31,3 µM PS1, 1 mM CytC);  hergestellt aus 8 Schichten ITO-Precursor mit 1,2 % Tin; Messung in KPP  (5 mM; pH 7.0); Lichtperiode: 10 s; Scanrate: 5 mV/s; Lichtintensität: 1000 W/m²; Startpotential: 0,2 V; Endpotential: -0,6 V.)"""


wrapped_caption = "\n".join(textwrap.wrap(caption, width=230))

plot_striped_voltage_current(dataC, xlabel='Spannung (V)', ylabel='Stromstärke pro cm² (µA/cm²)', caption=wrapped_caption)



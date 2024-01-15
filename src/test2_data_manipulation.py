import pandas as pd
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent.parent
dm_file = base / "Datos3/Datos Maestros VF.xlsx"

df = pd.read_excel(dm_file, sheet_name = 'Master Data Oficial', 
                   usecols=['AGENTE (OFEI)', 
                            'CENTRAL (dDEC, dSEGDES, dPRU…)',
                            'Tipo de central (Hidro, Termo, Filo, Menor)'])

# For easier manipulation, rename the columns
df.columns = ['Agente', 'Central', 'Tipo']

# Filter the data that corresponds to agente = EMGESA ó EMGESA S.A. and Tipo de Central ‘H’ or ‘T’.
df = df[(df['Agente'].isin(['EMGESA', 'EMGESA S.A.'])) & (df['Tipo'].isin(['H', 'T']))]

# Load the data from the file dDEC1204.TXT which has data in the format Central, Hora_1, Hora_2, …, Hora_24
txt_file = base / "Datos3/dDEC1204.TXT"
# Define the columns names
column_names = ['Central'] + [f'Hora_{i}' for i in range(1, 25)]
df2 = pd.read_csv(txt_file, sep = ',', names = column_names, encoding='ISO-8859-1')

# Merge the dataframes
df_mod = pd.merge(df, df2, on = 'Central')

# Group the data by Agente, Central, Tipo and sum the values of the columns Hora_i
df_mod = df_mod.groupby(['Agente', 'Central', 'Tipo']).sum()
df_mod['Total'] = df_mod.sum(axis = 1)

# Filter the data that has Total > 0
df_mod = df_mod[df_mod['Total'] > 0]

df_mod.to_excel(base / "output/oferta_plantas_emgesa.xlsx", sheet_name='ofertas')
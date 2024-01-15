import re

import pandas as pd
from pathlib import Path
from pandas import ExcelWriter

base = Path(__file__).resolve().parent.parent


# Read the txt file
txt_file = base / "Datos3/OFEI1204.txt"
with open(txt_file, "r", encoding ='utf-8') as f:
    data = f.readlines()

# Function to process the txt file
def process_txt(txt):
    # Remove '\n', change ':' to ',' and remove ' '
    txt = [s.strip('\n').replace(':', ',').replace(" ", '') for s in txt]
    
    # Filter out empty elements
    txt = list(filter(None, txt))
    
    # Separate header and data
    header = txt.pop(0)
    
    # Combine elements starting with 'AGENTE' with the previous one
    agente = ''
    for i in range(len(txt)): 
        if re.match('^AGENTE,', txt[i]):
            agente = txt[i]
            txt[i] = ''
        else:
            txt[i] = agente + ',' + txt[i]
    
    # # Filter out empty elements
    txt = list(filter(None, txt))
    # Remove 'AGENTE' from the elements
    txt = [s.replace('AGENTE,', '') for s in txt]
    txt = [s.split(",") for s in txt]
    
    return txt


data = process_txt(data)


# Creation of the dataframe
# Define the columns names
column_names = ['Agente', 'Planta', 'Tipo'] + [f'Hora_{i}' for i in range(1, 25)]
# Convert the data to a list of lists

# Read the data as a dataframe
df = pd.DataFrame(data, columns = column_names)

# Filter the data that corresponds to Tipo = 'D'
df_tipoD = df[df['Tipo'] == 'D']
# Drop the column Tipo
df_tipoD = df_tipoD.drop(['Tipo'], axis = 1)
# Convert the columns Hora_i to numeric
df_tipoD.iloc[:,2:] = df_tipoD.iloc[:,2:].apply(pd.to_numeric)

# Save the dataframe as a xlsx file
df_tipoD.to_excel(base / "output/oferta_plantas_tipo_d.xlsx", index = False)
import pandas as pd
import numpy as np
from pathlib import Path
import re

base = Path(__file__).resolve().parent.parent
address = base / "data_address.csv"

df = pd.read_csv(address, delimiter='\t')

def normalize_address(address):
    def extract_number_and_cardinal(number):
        number_match = re.match(r'(\d+)\s*([A-Za-z]+)?', number)
        if number_match:
            number_value = number_match.group(1).strip()
            cardinal_direction = number_match.group(2).strip() if number_match.group(2) else ''
            
            # Check if the cardinal direction is a known direction
            dir = is_cardinal_direction(cardinal_direction)
            return f'{number_value} {dir}' if dir else f'{number_value}'
        else:
            return ''

    def is_cardinal_direction(string):
        known_directions = ['sur', 'norte', 'este', 'oriente', 'Sur', 'Norte', 'Oriente', 'Este', 'S']

        for direction in known_directions:
            if direction in string:
                rest_of_string = string.replace(direction, '', 1).strip()
                return f'{rest_of_string} {direction}' 
        return ''
    
    # Extract street and numbers using regular expression
    def is_good_address(address):
        good1 = re.match(r'^([^0-9]+)\s(\d+\s+[A-Za-z\s*]*)\s*#\s(\d+\s+[A-Za-z\s]*)\s-\s(\d+\s*[A-Za-z\s]*)\s(.*)', address)
        good2 = re.match(r'^([^0-9]+\s)(\d+\s)\s*#\s*(\d+)\s*-\s*(\d+)\s*(.*)$', address)
        if not good1 and not good2:
            return False
        elif good1:
            return good1
        else:
            return good2
        
    good_address = is_good_address(address)

    match = re.match(r'([^\d]+)([\d\w#]+)\D*([\d\w#]+)?\D*([\d\w#]+)?(.*)', address)

    if not good_address:
        street = match.group(1).strip()
        number1 = extract_number_and_cardinal(match.group(2)) if match.group(2) else ''
        number2 = extract_number_and_cardinal(match.group(3)) if match.group(3) else ''
        number3 = extract_number_and_cardinal(match.group(4)) if match.group(4) else ''
        extra = match.group(5).strip()

        # Adjust the normalization based on the presence of letters in street or numbers
        street_normalized = re.sub(r'(\d)([A-Za-z])', r'\1 \2', street)

        # Format the normalized address
        normalized_address = f'{street_normalized} {number1} # {number2} - {number3}' + f' |{extra}' if extra else '' 
        return normalized_address.strip()
    else:
        street = good_address.group(1).strip()
        number1 = good_address.group(2)
        number2 = good_address.group(3)
        number3 = good_address.group(4)
        extra = good_address.group(5).strip()
        go_address = f'{street} {number1} # {number2} - {number3} |{extra}'
        return go_address  # Return the original address if no match


# # Version Power BI
# dataset = pd.DataFrame(dataset)
# dataset['address_description'] = dataset['address_description'].apply(normalize_address)

# Apply the normalize_address function to the 'address_description' column
df = df['address_description'].apply(normalize_address)

# Display the original and normalized dataframes
print(df)

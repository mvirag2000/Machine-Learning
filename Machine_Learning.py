import numpy as np
import pandas as pd

def lookup(VIN):
    url = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/" + VIN + "?format=json"
    data = pd.read_json(url)
    return data

# Run first one and retain table of Variable IDs
VIN = "4T1G11AK4LU952610"
data = lookup(VIN)
data = pd.json_normalize(data['Results'])
variables = data.drop(columns=['Value', 'ValueId']).set_index('VariableId').sort_values('VariableId') # Starts with 1 (Battery Info) not zero 
data = data.sort_values('VariableId').set_index('Variable').drop(columns=['VariableId', 'ValueId']).T # Maybe later use IDs instead of values 


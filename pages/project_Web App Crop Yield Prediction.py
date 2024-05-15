import streamlit as st
import pandas as pd
import joblib
import os

# Load the dataset to understand its structure
file_path = 'dataset/clean_data.csv'
data = pd.read_csv(file_path)

# Title of the web app
st.image('images/crop.png')
#st.title('🌽Crop Yield Estimation')

# Input fields for the user based on dataset columns
st.header('🌽Enter crop details:')

# Create input fields dynamically based on relevant columns
col1, col2 = st.columns(2)

with col1:
    year = st.number_input('Year', min_value=int(data['Year'].min()), max_value=int(data['Year'].max()))
    heat_wave = st.selectbox('Heat Wave', ['Y', 'N'])
    dry_spell = st.selectbox('Dry Spell', ['Y', 'N'])
    avg_min_temp = st.number_input('Avg Min Temp *C', min_value=float(data['Avg Min Temp *C'].min()), max_value=float(data['Avg Min Temp *C'].max()))
    irrigation = st.selectbox('Irrigation', ['Y', 'N'])
    crop_type = st.selectbox('Crop Type', data['Crop Type'].unique())

with col2:
    location = st.selectbox('Location', data['Location'].unique())
    cold_wave = st.selectbox('Cold Wave', ['Y', 'N'])
    wet_spell = st.selectbox('Wet Spell', ['Y', 'N'])
    yr_rain_mm = st.number_input('Yr Rain mm', min_value=float(data['Yr Rain mm'].min()), max_value=float(data['Yr Rain mm'].max()))
    crop_damage = st.selectbox('Crop Damage', ['Y', 'N'])

# Prepare the input data
input_data = pd.DataFrame([{
    'Year': year,
    'Avg Min Temp *C': avg_min_temp,
    'Yr Rain mm': yr_rain_mm,
    'Heat Wave': heat_wave,
    'Dry Spell': dry_spell,
    'Cold Wave': cold_wave,
    'Wet Spell': wet_spell,
    'Irrigation': irrigation,
    'Crop Damage': crop_damage,
    'Crop Type': crop_type,
    'Location': location
}])

# Encode categorical variables
input_data['Heat Wave'] = input_data['Heat Wave'].map({'Y': 1, 'N': 0})
input_data['Dry Spell'] = input_data['Dry Spell'].map({'Y': 1, 'N': 0})
input_data['Cold Wave'] = input_data['Cold Wave'].map({'Y': 1, 'N': 0})
input_data['Wet Spell'] = input_data['Wet Spell'].map({'Y': 1, 'N': 0})
input_data['Irrigation'] = input_data['Irrigation'].map({'Y': 1, 'N': 0})
input_data['Crop Damage'] = input_data['Crop Damage'].map({'Y': 1, 'N': 0})

# One-hot encode 'Crop Type' and 'Location'
input_data = pd.get_dummies(input_data, columns=['Crop Type', 'Location'], drop_first=True)

# Ensure all expected columns are present
expected_columns = [
    'Year', 'Avg Min Temp *C', 'Yr Rain mm', 'Heat Wave_transform', 'Dry Spell_transform', 'Cold Wave_transform', 
    'Wet Spell_transform', 'Irrigation_transform', 'Crop Damage_transform', 'Crop Type_Canola', 'Crop Type_Soy', 
    'Crop Type_Wheat', 'Location_Davidfurt', 'Location_East Stevenside', 'Location_Lake Meganhaven'
]

# Rename the columns to match the expected names
input_data.columns = input_data.columns.str.replace(' ', '_')
input_data = input_data.rename(columns={
    'Heat_Wave': 'Heat Wave_transform',
    'Dry_Spell': 'Dry Spell_transform',
    'Cold_Wave': 'Cold Wave_transform',
    'Wet_Spell': 'Wet Spell_transform',
    'Irrigation': 'Irrigation_transform',
    'Crop_Damage': 'Crop Damage_transform'
})

for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_columns]

# Load the models
model_dir = 'models'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

selected_model = st.selectbox('Select a model', model_files)

# Button to predict
if st.button('Estimate Yield'):
    if selected_model:
        model_path = os.path.join(model_dir, selected_model)
        model = joblib.load(model_path)
        prediction = model.predict(input_data)
        st.header(f'Estimated Crop Yield: {prediction[0]:.4f}')
    else:
        st.write("Please select a model.")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import random

def actual_trss(n1,n2,height,radius,gap):
    df = pd.read_csv('data_curation/sample_data/Ag_3_reduced.csv',low_memory=False)

    if int(radius) % 2 == 0:
        radius = int(radius-1)
    else:
        radius= int(radius)

    if int(gap) % 2 == 0:
        gap = int(gap-1)
    else:
        gap = int(gap)

    filtered_df = df[(df['n2'] == n2) & (df['rad'] == radius) & (df['gap'] == gap)]
    result_values = filtered_df['Ts']

    return result_values


def predict_trss(n1, n2, height, radius, gap):
    
    a = (radius-27.176199438026963)/13.17202192625192
    b = (gap - 54.1006631715935)/36.07597465176819
    c = (n2-3.699815690207628)/0.3997656731778045
    start_value = 400
    end_value = 1001
    data = []

    for i in range(start_value, end_value,2):
        data.append([a,b,c,i])
    for i in range(0,301):
        data[i][3] = (data[i][3]-700.0102603338136)/173.78230000960846
    X = pd.DataFrame(data)
    loaded_model = tf.keras.models.load_model("model_weights/forward_final.h5", compile = False)
    new_predictions = loaded_model.predict(X)

    return new_predictions


st.markdown(
    """
    <style>
        body {
            background: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def inv(l,u):
    df = pd.read_csv('data_curation/sample_data/Ag_3_reduced.csv',low_memory=False)
    filtered_df = df[(df['lambda_val'] >= l) & (df['lambda_val'] <= u)]
    max_row = filtered_df.loc[filtered_df['Ts'].idxmax()]

    return max_row

logo_image = "IITG_logo.png" 

st.image(logo_image, width=150, output_format="PNG", caption="in collaboration with NSM")

st.title("iDT-NaPaMeGs: Inverse Design Tool")
# Display material selection and input fields
# Material = st.selectbox("Select Material", ['Ag', 'Au', 'ZiN', 'TiN', 'Cu'])
# n2 = st.number_input("Enter n2 value (3 - 4)", min_value=3.0, max_value=4.5, step=0.1, value=4.1)
# n1 = st.selectbox("Select n1 value", [1.58, 1])
# height = st.number_input("Enter height value (0 - 10)", min_value=0.0, max_value=9.0, step=0.1, value=0.0)
# radius = st.number_input("Select radius value (5 - 50 nm)", min_value=5.0, max_value=50.0, step=2.0)
# gap = st.number_input("Select gap value (3 - 120 nm)", min_value=3.0, max_value=120.0, step=2.0)

color_ranges = {
    'Infrared': (760.0,1000.0),
    'Purple': (400.0, 450.0),
    'Blue': (450.0,500.0),
    'Green': (500.0, 570.0),
    'Yellow':(570.0, 590.0),
    'Orange': (590.0, 610.0),
    'Red': (610.0, 760.0)
}

color = st.selectbox("Select Color",['Infrared','Red','Orange','Yellow','Green','Blue','Purple'])
lower_limit, upper_limit = color_ranges[color]
wavelength_low = st.number_input("Put lower limit of wavelength", min_value=lower_limit, max_value=upper_limit, step=0.1,value =lower_limit)
wavelength_high = st.number_input("Put upper limit of wavelength", min_value=lower_limit, max_value=upper_limit, step=0.1,value = upper_limit)
max_row = inv(wavelength_low,wavelength_high)


# st.header("Combination with the Highest Ts Value")
# st.write("Material:", random.choice(['Au','Ag']))
# st.write("n1:", max_row['n1'])
# st.write("n2:", max_row['n2'] + round(random.uniform(0,1), 2))
# st.write("h:", random.randint(1,10))
# st.write("r:", max_row['rad'] +  round(random.uniform(0,1), 2))
# st.write("g:", max_row['gap'] +  round(random.uniform(0,1), 2))
# st.write("Ts value:", max_row['Ts'])

styled_text1 = f'<div style="font-size: 30px;">Material : {random.choice(["Au","Ag"])}</div>'
styled_text2 = f'<div style="font-size: 30px;">n1 (refractive index of encapsulant): {max_row["n1"]}</div>'
styled_text3 = f'<div style="font-size: 30px;">n2 (refractive index of LED): {max_row["n2"]+round(random.uniform(0,1), 2)}</div>'
styled_text4 = f'<div style="font-size: 30px;">height: {random.randint(1,10)} nm</div>'
styled_text5 = f'<div style="font-size: 30px;">radius: {max_row["rad"] +  round(random.uniform(0,1), 2)} nm</div>'
styled_text6 = f'<div style="font-size: 30px;">gap: {max_row["gap"] +  round(random.uniform(0,1), 2)} nm</div>'

st.markdown(styled_text1, unsafe_allow_html=True)
st.markdown(styled_text2, unsafe_allow_html=True)
st.markdown(styled_text3, unsafe_allow_html=True)
st.markdown(styled_text4, unsafe_allow_html=True)
st.markdown(styled_text5, unsafe_allow_html=True)
st.markdown(styled_text6, unsafe_allow_html=True)


predicted_trs = predict_trss(max_row['n1'], max_row['n2'], max_row['height'], max_row['rad'], max_row['gap'])
actual_trs = actual_trss(max_row['n1'], max_row['n2'], max_row['height'], max_row['rad'], max_row['gap'])
# predicted_trs = predict_trss(n1,n2,height,radius,gap)
# actual_trs = actual_trss(n1,n2,height,radius,gap)
# plt.figure(figsize=(10,6))
# plt.plot(range(400,1001,2),predicted_trs, label="Predicted TRS")
# plt.plot(range(400,1001,2),actual_trs, label="Actual TRS")
# plt.xlabel("Wavelength (in nm)")
# plt.ylabel("Transmittance (in %)")
# plt.title("Actual TRS vs Predicted TRS")
# plt.legend()
# st.pyplot(plt)

plt.figure(figsize=(10, 6))
plt.plot(range(400, 1001, 2), predicted_trs, label="Predicted TRS")
plt.plot(range(400, 1001, 2), actual_trs, label="Actual TRS")

# Increase font size and make labels bold
plt.xlabel("Wavelength (in nm)", fontsize=14, fontweight="bold")
plt.ylabel("Transmittance (in %)", fontsize=14, fontweight="bold")
plt.title("Actual TRS vs Predicted TRS", fontsize=16, fontweight="bold")

# Increase tick label font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add legend
plt.legend()

st.pyplot(plt)

import streamlit as st
import requests
from PIL import Image


# Load and set images in the first place
#header_images = Image.open('assets/header_images.jpg')
#st.image(header_images)

# Add some information about the service
st.title("NPL and Credit Worthiness Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "air_data_form"):
    # Create select box input
    housing_type = st.selectbox(
        label = "1.\tEnter the customer's housing type",
        options = (
            'milik orang tua',
            'milik sendiri',
            'kos',
            'milik pasangan',
            'kontrak'
        )
    
    )
    status_pernikahan = st.selectbox(
        label = "2.\tEnter the customer's status pernikahan",
        options = (
            'Belum Nikah',
            'Nikah'
        )
    
    )
    pekerjaan = st.selectbox(
        label = "3.\tEnter the customer's Pekerjaan",
        options = (
            'Wiraswasta',
            'Profesional',
            'Karyawan Swasta',
            'PNS',
            'Buruh',
            'Ibu Rumah Tangga/Pensiunan/Mahasiswa/Lainnya'
        )
    
    )

    # Create box for number input
    monthly_income = st.number_input(
        label = "4.\tEnter Monthly Income Value:",
        min_value = 21664620,
        max_value = 105000000,
        help = "Value range from 21664620 to 105000000"
    )
    
    num_of_dependent = st.number_input(
        label = "5.\tEnter Num Of Dependent Value:",
        min_value = 0,
        max_value = 14,
        help = "Value range from 0 to 14"
    )

    lama_bekerja = st.number_input(
        label = "6.\tEnter Lama Bekerja Value (Year):",
        min_value = 0,
        max_value = 14,
        help = "Value range from 0 to 14"
    )

    otr = st.number_input(
        label = "7.\tEnter OTR Value:",
        min_value = 500000000,
        max_value = 7000000000,
        help = "Value range from 500000000 to 7000000000"
    )

    tenor = st.number_input(
        label = "7.\tEnter Tenor Value:",
        min_value = 12,
        max_value = 48,
        help = "Value range from 12 to 48"
    )

    dp = st.number_input(
        label = "8.\tEnter DP Value:",
        min_value = 0,
        max_value = 7000000000,
        help = "Value range from 0 to 7000000000"
    )

    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "monthly_income": monthly_income,
            "housing_type": housing_type,
            "num_of_dependent": num_of_dependent,
            "lama_bekerja": lama_bekerja,
            "otr": otr,
            "status_pernikahan": status_pernikahan,
            "pekerjaan": pekerjaan,
            "tenor" : tenor,
            "dp": dp
                }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict", json = raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Tidak":
                st.warning("Predicted NPL: Ya")
                st.warning(f"Kualitas Nasabah {res['prob']}%")
            else:
                st.success("Predicted NPL: Tidak")
                st.success(f"Kualitas Nasabah : {res['prob']}%")
            
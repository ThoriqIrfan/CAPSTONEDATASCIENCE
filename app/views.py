from app import app
from flask   import render_template, request, redirect,url_for , session
from jinja2  import TemplateNotFound
import numpy as np
import csv
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import io
from sklearn.metrics.pairwise import euclidean_distances
import base64


@app.route('/')
def index():
    return render_template('form.html')

@app.route('/formsubmit', methods=['POST'])
def model_data_info():
    # Get the form data
    gender = request.form.get('gender')
    age = request.form.get('age')
    annual_income = request.form.get('annual_income')
    membership_tier = request.form.get('membership_tier')
    weekly_income = request.form.get('weekly_income')
    region = request.form.get('region')

    # Check if all the fields have been filled
    if not all([gender, age, annual_income, membership_tier, weekly_income, region]):
        return "Please fill all the form fields", 400
    # Process or print the data
    print(gender, age, annual_income, membership_tier, weekly_income, region)
    new_data = np.array([[annual_income, membership_tier]])

    with open('A:\DTS\CAPSTONEDATASCIENCE\dbscan_model.pkl', 'rb') as f:
        loaded_dbscan = pickle.load(f)
    with open('A:\DTS\CAPSTONEDATASCIENCE\scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    new_labels, distances = test_new_data(new_data, loaded_dbscan, loaded_scaler)
    results = []
    for i, (label, distance) in enumerate(zip(new_labels, distances)):
        if(label == 1 or label == 3):
            reward = "Mendapat Reward"
        else:
            reward = "Tidak Mendapat Reward"
        
        if label != -1:
            text = f"Data point {i} termasuk dalam cluster {label} dengan jarak {distance:.4f} ke titik inti terdekat {reward}"
        else:
            text = f"Data point {i} diklasifikasikan sebagai noise. Cluster terdekat berjarak {distance:.4f} {reward}"
        results.append(text)
    
    return redirect(url_for('modeldatainfo', results=','.join(results)))

@app.route('/modeldatainfo')
def modeldatainfo():
    results = request.args.get('results', '').split(',')
    return results

@app.route ('/customerdata')
def customer_data_list():
    return 'Customer Data List'

@app.route('/process_csv', methods=['GET'])
def model_data_info_csv():
    # Read the CSV file
    df = pd.read_csv('https://raw.githubusercontent.com/Fatikhaaa/data_science/main/Pengunjung_Mall_Dataset.csv')
    
    # Process each row in the CSV file
    results = []
    for index, row in df.iterrows():
        # Extract data from the row
        CustomerID = row['CustomerID']
        gender = row['Gender']
        age = row['Age']
        annual_income = row['Annual_Income']
        membership_tier = row['Membership_Tier']
        spending_score = row['Spending_Score']
        weekly_income = row['Weekly_Income']
        region = row['Region']

        # Check if all the fields have values
        if pd.isna(gender) or pd.isna(age) or pd.isna(annual_income) or pd.isna(membership_tier) or pd.isna(weekly_income) or pd.isna(region):
            results.append(f"Baris {index + 2}: Data tidak lengkap. Pastikan semua kolom terisi.")
            continue

        # Process the data
        print(gender, age, annual_income, membership_tier, weekly_income, region)
        new_data = np.array([[spending_score, membership_tier]])

        with open('A:\\DTS\\CAPSTONEDATASCIENCE\\dbscan_model.pkl', 'rb') as f:
            loaded_dbscan = pickle.load(f)
        with open('A:\\DTS\\CAPSTONEDATASCIENCE\\scaler.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)

        new_labels, distances = test_new_data(new_data, loaded_dbscan, loaded_scaler)
        
        for i, (label, distance) in enumerate(zip(new_labels, distances)):
            if(label == 1 or label == 3):
                reward = "Mendapat Reward"
            else:
                reward = "Tidak Mendapat Reward"
            if label != -1:
                text = f"Baris {index + 2}: Data point {i} termasuk dalam cluster {label} dengan jarak {distance:.4f} ke titik inti terdekat {reward}"
            else:
                text = f"Baris {index + 2}: Data point {i} diklasifikasikan sebagai noise. Cluster terdekat berjarak {distance:.4f} {reward}"
            results.append(text)

    return redirect(url_for('modeldatainfo', results=','.join(results)))

def test_new_data(new_data, dbscan_model, scaler):
    new_data_scaled = scaler.transform(new_data)
    distances = euclidean_distances(new_data_scaled, dbscan_model.components_)
    min_distances = distances.min(axis=1)
    nearest_core_indices = distances.argmin(axis=1)
    
    new_labels = np.where(min_distances <= dbscan_model.eps, 
                          dbscan_model.labels_[dbscan_model.core_sample_indices_][nearest_core_indices], 
                          -1)
    
    return new_labels, min_distances
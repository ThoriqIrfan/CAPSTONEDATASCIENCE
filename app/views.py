from app import app
from flask import render_template

@app.route('/')
def index():
    return render_template('form.html')

@app.route ('/formsubmit', methods=['POST'])
def model_data_info():
    return 'form is submitted'

@app.route ('/modeldatainfo')
def capstone_model():
    return 'Model'

@app.route ('/customerdata')
def customer_data_list():
    return 'Customer Data List'


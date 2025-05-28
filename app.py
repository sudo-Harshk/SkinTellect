import os
from dotenv import load_dotenv
import sqlite3
from flask import Flask, jsonify, render_template, request, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from roboflow import Roboflow
import json
import supervision as sv
import uuid  
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2
import pandas as pd
from joblib import load
import numpy as np
from PIL import Image
import base64
import io
from flask_mail import Mail, Message
from datetime import datetime, timedelta
import random
import re
import unicodedata
from twilio.rest import Client

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', '4545')  # Use environment variable with fallback
DATABASE = os.environ.get('DATABASE_PATH', 'app.db')

# Twilio configuration
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Email config
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = (
    os.environ.get('MAIL_DEFAULT_SENDER_NAME', 'SkinTellect'),
    os.environ.get('MAIL_DEFAULT_SENDER_EMAIL', os.environ.get('MAIL_USERNAME'))
)
app.config['MAIL_MAX_EMAILS'] = int(os.environ.get('MAIL_MAX_EMAILS', 5))
app.config['MAIL_ASCII_ATTACHMENTS'] = os.environ.get('MAIL_ASCII_ATTACHMENTS', 'False').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'False').lower() == 'true'

# Initialize mail
mail = Mail(app)

# Skin detection model initialization
rf_skin = Roboflow(api_key=os.environ.get('ROBOFLOW_API_KEY'))
project_skin = rf_skin.workspace().project("skin-detection-pfmbg")
model_skin = project_skin.version(2).model

# Oiliness detection model initialization
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ.get('ROBOFLOW_INFERENCE_API_KEY')
)

# Store unique classes
unique_classes = set()

# Mapping for oily skin class
class_mapping = {
    "Jenis Kulit Wajah - v6 2023-06-17 11-53am": "oily skin",
    "-": "normal/dry skin"  
}

prediction_column_mapping = {
    "dryness": "dryness",
    "Dark Circle": "Dark Circle",
    "oily skin": "oily skin",
    "normal/dry skin": "normal/dry skin",
    "whitehead": "whitehead",
    "blackhead": "blackhead",
    "papule": "papule",
    "pustule": "pustule",
    "PIH": "PIH"
}

fallback_recommendations = {
    "dryness": [
        {"Brand": "CeraVe", "Name": "Moisturizing Cream", "Ingredients": "Ceramides, Hyaluronic Acid"},
        {"Brand": "Neutrogena", "Name": "Hydro Boost", "Ingredients": "Hyaluronic Acid, Glycerin"}
    ],
    "Dark Circle": [
        {"Brand": "The Ordinary", "Name": "Caffeine Solution", "Ingredients": "Caffeine, EGCG"},
        {"Brand": "Cetaphil", "Name": "Hydrating Eye Gel-Cream", "Ingredients": "Hyaluronic Acid, Licorice Extract"}
    ],
    "oily skin": [
        {"Brand": "La Roche-Posay", "Name": "Effaclar Gel", "Ingredients": "Zinc PCA, Salicylic Acid"},
        {"Brand": "Paula's Choice", "Name": "BHA Liquid", "Ingredients": "Salicylic Acid"}
    ],
    "normal/dry skin": [
        {"Brand": "Aveeno", "Name": "Daily Moisturizing Lotion", "Ingredients": "Colloidal Oatmeal, Glycerin"},
        {"Brand": "Clinique", "Name": "Dramatically Different Moisturizing Lotion", "Ingredients": "Glycerin, Urea"}
    ]
}

def normalize_string(s):
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return ' '.join(s.split()).lower()

def recommend_products_based_on_classes(classes):
    recommendations = []
    used_products = set()

    json_path = r"D:\skin-care-complete\dataset\skincare_products.json"
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            product_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        product_data = {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        product_data = {}

    product_counts = [1, 2, 3]
    random.shuffle(product_counts)

    for idx, skin_condition in enumerate(classes):
        mapped_condition = prediction_column_mapping.get(skin_condition, skin_condition)
        if mapped_condition in product_data:
            products_list = product_data[mapped_condition]
            available_products = [
                product for product in products_list
                if (product["Brand"], product["Name"]) not in used_products
            ]
            if available_products:
                num_products_to_recommend = product_counts[idx % len(product_counts)]
                num_products_to_recommend = min(num_products_to_recommend, len(available_products))
                random.shuffle(available_products)
                selected_products = available_products[:num_products_to_recommend]
                for product in selected_products:
                    used_products.add((product["Brand"], product["Name"]))
                recommendations.append((skin_condition, selected_products))
            else:
                fallback_product = [{
                    "Brand": "N/A",
                    "Name": "No unique product available",
                    "Image_URL": "",
                    "Product_URL": ""
                }]
                recommendations.append((skin_condition, fallback_product))
        else:
            fallback_product = [{
                "Brand": "N/A",
                "Name": "No matching product",
                "Image_URL": "",
                "Product_URL": ""
            }]
            recommendations.append((skin_condition, fallback_product))
    return recommendations

@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = request.form['login_id']
        password = request.form['password']
        user = get_user_by_login_id(login_id)

        if user and check_password_hash(user[3], password):
            session['username'] = user[1]
            session['name'] = user[1]  # Default to username
            # Try to fetch latest age from appointment table
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT age FROM appointment WHERE username = ? ORDER BY id DESC LIMIT 1", (user[1],))
                row = cursor.fetchone()
                if row and row[0]:
                    session['age'] = row[0]
                else:
                    session['age'] = ''
            flash('Login successful!', 'success')
            if user[1] == 'doctor':
                return redirect(url_for('allappoint'))
            return redirect('/predict')
        flash('Invalid username/email or password', 'danger')
        return render_template('login.html')
    return render_template('login.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'username' not in session:
        return redirect('/')
    if request.method == 'POST':
        unique_classes = set()

        image_file = request.files.get('image')
        if not image_file or image_file.filename == '':
            return render_template('face_analysis.html', data=[], error="⚠️ Please select an image to analyze.")
        
        if not image_file.mimetype.startswith('image/'):
            return render_template('face_analysis.html', data=[], error="⚠️ Please upload a valid image file.")

        image_filename = str(uuid.uuid4()) + '.jpg'
        image_path = os.path.join('static', image_filename)
        image_file.save(image_path)

        # Skin prediction
        skin_result = model_skin.predict(image_path, confidence=15, overlap=30).json()
        skin_labels = [item["class"] for item in skin_result.get("predictions", [])]
        unique_classes.update(skin_labels)

        # Oiliness detection
        custom_configuration = InferenceConfiguration(confidence_threshold=0.3)
        with CLIENT.use_configuration(custom_configuration):
            oilyness_result = CLIENT.infer(image_path, model_id="oilyness-detection-kgsxz/1")

        if not oilyness_result['predictions']:
            unique_classes.add("dryness")
        else:
            oilyness_classes = [class_mapping.get(pred['class'], pred['class']) for pred in oilyness_result['predictions'] if pred['confidence'] >= 0.3]
            unique_classes.update(oilyness_classes)

        image = cv2.imread(image_path)
        if skin_result.get("predictions"):
            detections = sv.Detections(
                xyxy=np.array([
                    [
                        pred["x"] - pred["width"] / 2,
                        pred["y"] - pred["height"] / 2,
                        pred["x"] + pred["width"] / 2,
                        pred["y"] + pred["height"] / 2
                    ] for pred in skin_result["predictions"]
                ]),
                class_id=np.array([0] * len(skin_result["predictions"])),
                confidence=np.array([pred["confidence"] for pred in skin_result["predictions"]]),
                data={"class_name": [pred["class"] for pred in skin_result["predictions"]]}
            )
            label_annotator = sv.LabelAnnotator()
            bounding_box_annotator = sv.BoxAnnotator()
            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        else:
            annotated_image = image.copy()

        annotated_image_path = os.path.join('static', 'annotations_0.jpg')
        cv2.imwrite(annotated_image_path, annotated_image)

        recommended_products = recommend_products_based_on_classes(list(unique_classes))
        prediction_data = {
            'classes': list(unique_classes),
            'recommendations': recommended_products
        }

        return render_template('face_analysis.html', data=prediction_data, message=request.args.get('message'))
    return render_template('face_analysis.html', data=[])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        hashed_password = generate_password_hash(password)

        if get_user(username):
            return render_template('register.html', error="⚠️ Username already exists. Please choose a different one.")
        
        if get_user_by_email(email):
            return render_template('register.html', error="⚠️ Email already registered. Please log in or reset your password.")

        insert_user(username, email, hashed_password)
        session['name'] = username 
        session['age'] = age
        return redirect('/')
    return render_template('register.html')

@app.route('/bookappointment')
def bookappointment():
    if 'username' not in session:
        return redirect('/')
    return render_template('bookappointment.html', name=session.get('name'), age=session.get('age'))

@app.route("/appointment", methods=["POST"])
def appointment():
    if 'username' not in session:
        return redirect('/')
    name = request.form.get("name")
    email = request.form.get("email")
    date = request.form.get("date")
    skin = request.form.get("skin")
    phone = request.form.get("phone", "").strip()  # Get and clean phone number
    age = request.form.get("age")
    address = request.form.get("address")
    concerns = ",".join(request.form.getlist("concerns"))
    skin_type = request.form.get("skin_type")
    acne_frequency = request.form.get("acne_frequency")
    first_concern = request.form.get("first_concern")
    username = session['username']
    status = False
    
    # Clean and format phone number
    if phone:
        # Remove any non-digit characters
        phone = ''.join(filter(str.isdigit, phone))
        # Add +91 prefix if not present
        if not phone.startswith('91'):
            phone = '91' + phone
        # Add + prefix for Twilio
        if not phone.startswith('+'):
            phone = '+' + phone
        print(f"Processed appointment phone number: {phone}")
    
    # Insert appointment data
    insert_appointment_data(name, email, date, skin, phone, age, address, status, username, concerns, skin_type, acne_frequency, first_concern)
    
    # Send confirmation email only (no SMS on initial booking)
    email_sent = send_email(
        to_email=email,
        subject='Appointment Request Received - SkinTellect',
        template_name='appointment_confirmation_email.html',
        name=name,
        status='PENDING',
        date=date,
        time=date,
        service=skin
    )
    
    return redirect(url_for('appointment_success'))

@app.route('/appointment_success')
def appointment_success():
    return render_template('appointment_success.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'username' not in session:
        return redirect('/')
    username = session['username']

    if request.method == 'POST':
        # Save the updated profile data
        name = request.form.get('name')
        age = request.form.get('age')
        concerns = ",".join(request.form.getlist('concerns'))
        skin_type = request.form.get('skin_type')
        acne_frequency = request.form.get('acne_frequency')
        first_concern = request.form.get('first_concern')

        # Update session for name and age
        session['name'] = name
        session['age'] = age

        # Update the latest appointment for this user
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM appointment WHERE username=? ORDER BY id DESC LIMIT 1
            """, (username,))
            row = cursor.fetchone()
            if row:
                latest_id = row[0]
                cursor.execute("""
                    UPDATE appointment
                    SET name=?, age=?, concerns=?, skin_type=?, acne_frequency=?, first_concern=?
                    WHERE id=?
                """, (name, age, concerns, skin_type, acne_frequency, first_concern, latest_id))
                conn.commit()
        flash('Profile updated successfully!', 'success')
        return redirect('/profile')

    # GET request: show profile (edit mode if ?edit=1)
    name = session.get('name')
    age = session.get('age')
    edit = request.args.get('edit') == '1'

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT concerns, skin_type, acne_frequency, first_concern
            FROM appointment
            WHERE username = ?
            ORDER BY id DESC LIMIT 1
        """, (username,))
        appointment = cursor.fetchone()

    if appointment:
        concerns, skin_type, acne_frequency, first_concern = appointment
        return render_template(
            'profile.html',
            name=name,
            age=age,
            concerns=concerns or "Not specified",
            skin_type=skin_type or "Not specified",
            acne_frequency=acne_frequency or "Not specified",
            first_concern=first_concern or "Not specified",
            edit=edit
        )
    else:
        return render_template(
            'profile.html',
            name=name,
            age=age,
            error="No appointment data available. Book an appointment to provide skin details.",
            edit=edit
        )

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        phone = request.form.get('phone', '').strip()  # Get phone number from form and clean it
        user = get_user_by_email(email)

        if not user:
            return render_template('forgot_password.html', error="Email not found.")

        # Validate phone number
        if phone:
            # Remove any spaces or special characters
            phone = ''.join(filter(str.isdigit, phone))
            if not phone.startswith('+'):
                phone = '+' + phone
            print(f"Processed phone number: {phone}")

        # Check if there's an existing unused OTP
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM otps WHERE user_id = ? AND used = 0 AND expires_at > datetime('now', 'utc')", 
                         (user[0],))
            existing_otp = cursor.fetchone()
            
            if existing_otp:
                # If there's an existing valid OTP, use it instead of generating a new one
                otp = existing_otp[2]
                expires_at = existing_otp[3]
                print(f"Using existing OTP: {otp}")
            else:
                # Generate new OTP
                otp = f"{random.randint(100000, 999999)}"
                expires_at = (datetime.utcnow() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("INSERT INTO otps (user_id, otp, expires_at) VALUES (?, ?, ?)", 
                             (user[0], otp, expires_at))
                conn.commit()
                print(f"Generated new OTP: {otp}")

        # Send OTP via SMS if phone number is provided
        sms_sent = False
        if phone:
            print(f"Attempting to send SMS to {phone}")
            sms_message = f"Your SkinTellect password reset OTP is: {otp}. It will expire in 5 minutes."
            sms_sent = send_sms(phone, sms_message)
            if not sms_sent:
                print(f"Failed to send SMS to {phone}")
                flash('Failed to send OTP via SMS. Please check your phone number.', 'error')
                return render_template('forgot_password.html', error="Failed to send OTP via SMS.")
            print(f"SMS sent successfully to {phone}")

        # Also send email as backup
        msg = Message("SkinTellect Password Reset OTP", 
                     sender=app.config['MAIL_USERNAME'], 
                     recipients=[email])
        
        html_content = render_template('email_template.html', otp=otp)
        msg.html = html_content
        msg.body = f"Your OTP for password reset is: {otp}. It will expire in 5 minutes."
        
        try:
            mail.send(msg)
            print(f"Email sent successfully to {email}")
            if sms_sent:
                flash('OTP has been sent to your email and phone. Please check both.', 'success')
            else:
                flash('OTP has been sent to your email. Please check your inbox.', 'success')
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            if sms_sent:
                flash('Failed to send OTP via email. Please check your SMS.', 'warning')
            else:
                flash('Failed to send OTP. Please try again.', 'error')
                return render_template('forgot_password.html', error="Failed to send OTP. Please try again.")

        session['reset_user_id'] = user[0]
        return redirect('/verify_otp')
    return render_template('forgot_password.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if 'reset_user_id' not in session:
        flash('Please request a password reset first.', 'error')
        return redirect('/forgot_password')
        
    if request.method == 'POST':
        entered_otp = request.form['otp']
        user_id = session.get('reset_user_id')

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM otps WHERE user_id = ? AND otp = ? AND used = 0 AND expires_at > datetime('now', 'utc')", 
                          (user_id, entered_otp))
            otp_record = cursor.fetchone()

        if otp_record:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE otps SET used = 1 WHERE id = ?", (otp_record[0],))
                cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
                user_email = cursor.fetchone()[0]
                conn.commit()
            
            session['reset_email'] = user_email
            flash('OTP verified successfully. Please set your new password.', 'success')
            return redirect('/reset_password')
            
        flash('Invalid or expired OTP. Please try again.', 'error')
        return render_template('verify_otp.html', error="Invalid or expired OTP")
    return render_template('verify_otp.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_email' not in session:
        flash('Please complete the OTP verification first.', 'error')
        return redirect('/forgot_password')
        
    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Password validation
        if len(password) < 8:
            return render_template('reset_password.html', 
                                 error="Password must be at least 8 characters long.")
        if not re.search(r"[A-Z]", password):
            return render_template('reset_password.html', 
                                 error="Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", password):
            return render_template('reset_password.html', 
                                 error="Password must contain at least one lowercase letter.")
        if not re.search(r"\d", password):
            return render_template('reset_password.html', 
                                 error="Password must contain at least one number.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return render_template('reset_password.html', 
                                 error="Password must contain at least one special character.")
        
        if password != confirm_password:
            return render_template('reset_password.html', 
                                 error="Passwords do not match.")
        
        hashed_password = generate_password_hash(password)
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET password = ? WHERE email = ?", 
                         (hashed_password, session['reset_email']))
            conn.commit()
            
        # Clear all reset-related session data
        session.pop('reset_email', None)
        session.pop('reset_user_id', None)
        
        flash('Password has been reset successfully. Please login with your new password.', 'success')
        return redirect('/?message=Password+reset+successful')
    return render_template('reset_password.html')

@app.route("/reject_appointment", methods=["POST"])
def reject_appointment():
    try:
        appointment_id = request.form.get("appointment_id")
        reason = request.form.get("reason", "No reason provided")
        
        if not appointment_id:
            return jsonify(success=False, error="Missing appointment ID")
        
        with sqlite3.connect(DATABASE) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get appointment details before updating
            c.execute("SELECT * FROM appointment WHERE id = ?", (appointment_id,))
            appointment = c.fetchone()
            
            if appointment:
                # Update status
                c.execute("UPDATE appointment SET rejected = 1, status = 0 WHERE id = ?", (appointment_id,))
                conn.commit()
                
                # Send rejection email
                email_sent = send_email(
                    to_email=appointment['email'],
                    subject='Appointment Update - SkinTellect',
                    template_name='appointment_confirmation_email.html',
                    name=appointment['name'],
                    status='REJECTED',
                    date=appointment['date'],
                    time=appointment['date'],
                    service=appointment['skin'],
                    reason=reason
                )
                
                # Send SMS notification if phone number is provided
                if appointment['phone']:
                    print(f"Attempting to send appointment rejection SMS to {appointment['phone']}")
                    sms_message = f"Dear {appointment['name']},\n\nWe regret to inform you that we cannot accommodate your appointment request for {appointment['date']}.\n\nPlease feel free to book another appointment at a more convenient time.\n\nBest regards,\nSkinTellect Team"
                    sms_sent = send_sms(appointment['phone'], sms_message)
                    if not sms_sent:
                        print(f"Failed to send appointment rejection SMS to {appointment['phone']}")
                
                return jsonify(success=True)
            else:
                return jsonify(success=False, error="Appointment not found")
    except Exception as e:
        print(f"Error in reject_appointment: {str(e)}")
        return jsonify(success=False, error=str(e))

@app.route('/test_email')
def test_email():
    test_email = '18nu1a0598@nsrit.edu.in'  # Use your email for testing
    success = send_email(
        to_email=test_email,
        subject='Test Email from SkinTellect',
        template_name='appointment_confirmation_email.html',
        name='Test User',
        status='PENDING',
        date='2024-03-20',
        time='10:00 AM',
        service='Test Service'
    )
    
    if success:
        return "✅ Test email sent successfully! Check your inbox and spam folder."
    else:
        return "❌ Failed to send test email. Check the console for error messages."

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('name', None)
    session.pop('age', None)
    return redirect('/')

@app.route("/allappointments")
def allappoint():
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM appointment")
        appointments = c.fetchall()
        appointments = [dict(row) for row in appointments]
    return render_template('doctor.html', appointments=appointments)

@app.route("/update_status", methods=["POST"])
def update_status():
    try:
        appointment_id = request.form.get("appointment_id")
        if not appointment_id:
            return jsonify(success=False, error="Missing appointment ID")
        
        with sqlite3.connect(DATABASE) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Get appointment details before updating
            c.execute("SELECT * FROM appointment WHERE id = ?", (appointment_id,))
            appointment = c.fetchone()
            
            if appointment:
                # Update status
                c.execute("UPDATE appointment SET status = 1 WHERE id = ?", (appointment_id,))
                conn.commit()
                
                # Send confirmation email
                email_sent = send_email(
                    to_email=appointment['email'],
                    subject='Appointment Confirmed - SkinTellect',
                    template_name='appointment_confirmation_email.html',
                    name=appointment['name'],
                    status='ACCEPTED',
                    date=appointment['date'],
                    time=appointment['date'],
                    service=appointment['skin']
                )
                
                # Send SMS confirmation if phone number is provided
                if appointment['phone']:
                    print(f"Attempting to send appointment acceptance SMS to {appointment['phone']}")
                    sms_message = f"Dear {appointment['name']},\n\nYour appointment has been confirmed for {appointment['date']}.\n\nPlease arrive 10 minutes before your scheduled time.\n\nBest regards,\nSkinTellect Team"
                    sms_sent = send_sms(appointment['phone'], sms_message)
                    if not sms_sent:
                        print(f"Failed to send appointment acceptance SMS to {appointment['phone']}")
                
                return jsonify(success=True)
            else:
                return jsonify(success=False, error="Appointment not found")
    except Exception as e:
        print(f"Error in update_status: {str(e)}")
        return jsonify(success=False, error=str(e))

@app.route("/delete_user_request", methods=["POST"])
def delete_user_request():
    try:
        appointment_id = request.form.get("id")
        if not appointment_id:
            return jsonify(success=False, error="Missing appointment ID")
            
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM appointment WHERE id = ?", (appointment_id,))
            conn.commit()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route("/userappointment")
def userappointment():
    if 'username' not in session:
        return redirect('/')
    
    user = session['username']
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        c = conn.cursor()
        c.execute("SELECT * FROM appointment WHERE username = ? ORDER BY date DESC", (user,))
        appointments = [dict(row) for row in c.fetchall()]
    
    print(f"Found {len(appointments)} appointments for user {user}")  # Debug print
    return render_template('userappointment.html', all_appointments=appointments)

@app.route("/doctor")
def doctor():
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM appointment")
        appointments = [dict(row) for row in c.fetchall()]
    return render_template('doctor.html', appointments=appointments)

@app.route('/delete_all_appointments', methods=['POST'])
def delete_all_appointments():
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM appointment")
            conn.commit()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/appointments/<int:appointment_id>/update', methods=['POST'])
def update_appointment(appointment_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    status = request.form.get('status')
    reason = request.form.get('reason', '')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Update appointment status
        cursor.execute('''
            UPDATE appointments 
            SET status = ?, reason = ? 
            WHERE id = ?
        ''', (status, reason, appointment_id))
        
        # Get appointment details for email
        cursor.execute('''
            SELECT a.*, u.name, u.email, s.name as service_name
            FROM appointments a
            JOIN users u ON a.user_id = u.id
            JOIN services s ON a.service_id = s.id
            WHERE a.id = ?
        ''', (appointment_id,))
        appointment = cursor.fetchone()
        
        if appointment:
            # Send email notification
            msg = Message(
                subject=f'Appointment {status} - SkinTellect',
                sender=('SkinTellect', 'noreply@skintellect.com'),
                recipients=[appointment['email']]
            )
            
            # Render email template
            msg.html = render_template(
                'appointment_confirmation_email.html',
                name=appointment['name'],
                status=status,
                date=appointment['date'],
                time=appointment['time'],
                service=appointment['service_name'],
                reason=reason if status == 'REJECTED' else None
            )
            
            # Add plain text version
            msg.body = f"""
            Dear {appointment['name']},
            
            Your appointment request has been {status.lower()}.
            
            Appointment Details:
            Date: {appointment['date']}
            Time: {appointment['time']}
            Service: {appointment['service_name']}
            Status: {status}
            {f'Reason: {reason}' if status == 'REJECTED' else ''}
            
            {f'We look forward to seeing you at your appointment. Please arrive 10 minutes before your scheduled time.' if status == 'ACCEPTED' else 'We apologize for any inconvenience. Please feel free to book another appointment at a more convenient time.'}
            
            If you have any questions, please don't hesitate to contact us.
            
            Best regards,
            SkinTellect Team
            """
            
            mail.send(msg)
        
        conn.commit()
        flash(f'Appointment {status.lower()} successfully!', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error updating appointment: {str(e)}', 'error')
    finally:
        conn.close()
    
    return redirect(url_for('admin_appointments'))

def send_email(to_email, subject, template_name, **kwargs):
    try:
        msg = Message(
            subject=subject,
            recipients=[to_email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        
        # Render email template
        msg.html = render_template(template_name, **kwargs)
        
        # Add plain text version
        msg.body = f"""
        Dear {kwargs.get('name', 'User')},
        
        Your appointment request has been {kwargs.get('status', 'PENDING').lower()}.
        
        Appointment Details:
        Date: {kwargs.get('date', 'N/A')}
        Service: {kwargs.get('service', 'N/A')}
        Status: {kwargs.get('status', 'PENDING')}
        {f'Reason: {kwargs.get("reason")}' if kwargs.get('reason') else ''}
        
        {f'We look forward to seeing you at your appointment. Please arrive 10 minutes before your scheduled time.' if kwargs.get('status') == 'ACCEPTED' else 'We will review your appointment request and send you a confirmation email shortly.' if kwargs.get('status') == 'PENDING' else 'We apologize for any inconvenience. Please feel free to book another appointment at a more convenient time.'}
        
        If you have any questions, please don't hesitate to contact us.
        
        Best regards,
        SkinTellect Team
        """
        
        # Send email with error handling
        try:
            mail.send(msg)
            print(f"Email sent successfully to {to_email}")
            return True
        except Exception as e:
            print(f"Error sending email to {to_email}: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error preparing email: {str(e)}")
        return False

def send_sms(to_number, message):
    try:
        print(f"Attempting to send SMS to {to_number} using Twilio account {TWILIO_ACCOUNT_SID}")
        print(f"Message content: {message}")
        
        # Verify phone number format
        if not to_number.startswith('+'):
            to_number = '+' + to_number
            print(f"Added + prefix to phone number: {to_number}")
        
        # Create and send message
        message = twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        
        print(f"SMS sent successfully to {to_number}")
        print(f"Message SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending SMS to {to_number}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return False

def get_user_by_email(email):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        return cursor.fetchone()

def get_user_by_login_id(login_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (login_id, login_id))
        return cursor.fetchone()

def create_tables():
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE NOT NULL,
                          email TEXT UNIQUE NOT NULL,
                          password TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS appointment (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT,
                          email TEXT,
                          date TEXT,
                          skin TEXT,
                          phone TEXT,
                          age TEXT,
                          address TEXT,
                          status BOOLEAN,
                          rejected BOOLEAN DEFAULT 0,
                          username TEXT,
                          concerns TEXT,
                          skin_type TEXT,
                          acne_frequency TEXT,
                          first_concern TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS otps (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id INTEGER NOT NULL,
                          otp TEXT NOT NULL,
                          expires_at TEXT NOT NULL,
                          used BOOLEAN DEFAULT 0,
                          FOREIGN KEY (user_id) REFERENCES users(id))''')
        connection.commit()

def insert_user(username, email, password):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
        connection.commit()

def insert_appointment_data(name, email, date, skin, phone, age, address, status, username, concerns, skin_type, acne_frequency, first_concern):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO appointment 
                    (name, email, date, skin, phone, age, address, status, username, concerns, skin_type, acne_frequency, first_concern)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (name, email, date, skin, phone, age, address, status, username, concerns, skin_type, acne_frequency, first_concern))
        conn.commit()
        print(f"Inserted appointment for user: {username}")  # Debug print

def findappointment(user):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM appointment WHERE username = ?", (user,))
        users = c.fetchall()
        return users

def findallappointment():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM appointment")
        users = c.fetchall()
        return users

def get_user(username):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

def init_app():
    create_tables()

if __name__ == '__main__':
    init_app()
    app.run(debug=True)
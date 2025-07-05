import sys
import threading
import sqlite3
import pickle
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_session import Session
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import secrets
import numpy as np

# Function to get resource path for PyInstaller
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Flask app setup
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = secrets.token_hex(32)
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions')
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)

# SQLite database setup
DB_PATH = resource_path("health_risk.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            date TEXT NOT NULL,
            prediction TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            details TEXT,
            doctor_recommendation TEXT,
            risk_factors TEXT,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )''')
        conn.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        conn.close()


# Load ML model and related files
MODEL_PATH = resource_path("random_forest_model.pkl")
FEATURES_PATH = resource_path("feature_names.pkl")
LABELENCODERS_PATH = resource_path("label_encoder.pkl")

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(LABELENCODERS_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(FEATURES_PATH, "rb") as features_file:
        feature_names = pickle.load(features_file)
    with open(LABELENCODERS_PATH, "rb") as labels_file:
        label_encoder = pickle.load(labels_file)
else:
    model = None
    feature_names = []
    label_encoder = None
    print(f"Warning: Model or Feature Files Missing! Paths checked: {MODEL_PATH}, {FEATURES_PATH}, {LABELENCODERS_PATH}")

# Routes
# Shutdown server function
def shutdown_server():
    os._exit(0)  # Forcefully exit the process

@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route('/')
def frontpage():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering frontpage: {e}")
        return "Error loading welcome page", 500

@app.route('/index')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index: {e}")
        return "Error loading home page", 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        try:
            return render_template('signup.html')
        except Exception as e:
            print(f"Error rendering signup: {e}")
            return "Error loading signup page", 500
    
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')

    if not all([name, email, password, role]):
        return jsonify({"success": False, "message": "All fields are required"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT email FROM users WHERE email = ?", (email,))
        if c.fetchone():
            return jsonify({"success": False, "message": "Email already exists"}), 400
        c.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                  (name, email, password, role))
        conn.commit()
        return jsonify({"success": True, "message": "Signup successful, please login"})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Email already exists"}), 400
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({"success": False, "message": "An error occurred during signup"}), 500
    finally:
        conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        try:
            return render_template('login.html')
        except Exception as e:
            print(f"Error rendering login: {e}")
            return "Error loading login page", 500
    
    data = request.json
    email = data.get("email")
    password = data.get("password")
    role = data.get("role")

    if not all([email, password, role]):
        return jsonify({"success": False, "message": "All fields are required"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT name, email, password, role FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        if user and user[2] == password and user[3] == role:
            session['user'] = user[1]
            session['role'] = user[3]
            session['name'] = user[0]
            return jsonify({"success": True, "role": role, "message": "Login successful"})
        return jsonify({"success": False, "message": "Invalid email, password, or role"}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"success": False, "message": "An error occurred during login"}), 500
    finally:
        conn.close()

@app.route('/user_dashboard')
def user_dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Fetch the latest record
        c.execute("""
            SELECT prediction, recommendation, date, details, doctor_recommendation, risk_factors
            FROM records
            WHERE user_email = ?
            ORDER BY date DESC LIMIT 1
        """, (session['user'],))
        latest_record = c.fetchone()
        prediction = None
        if latest_record:
            prediction = {
                "risk": latest_record[0],
                "recommendation": latest_record[1],
                "date": latest_record[2],
                "details": json.loads(latest_record[3]) if latest_record[3] else {},
                "doctor_recommendation": latest_record[4],
                "risk_factors": json.loads(latest_record[5]) if latest_record[5] else []
            }

        # Fetch all previous records
        c.execute("""
            SELECT date, prediction, recommendation, details, doctor_recommendation, risk_factors
            FROM records
            WHERE user_email = ?
            ORDER BY date DESC
        """, (session['user'],))
        records = [
            {
                "date": r[0],
                "risk": r[1],
                "recommendation": r[2],
                "details": json.loads(r[3]) if r[3] else {},
                "doctor_recommendation": r[4],
                "risk_factors": json.loads(r[5]) if r[5] else []
            } for r in c.fetchall()
        ]

        return render_template('user_dashboard.html',
                               prediction=prediction,
                               records=records,
                               error=None,
                               name=session.get('name', 'User'))
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return render_template('user_dashboard.html',
                               prediction=None,
                               records=[],
                               error="Error retrieving data",
                               name=session.get('name', 'User'))
    finally:
        conn.close()


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/newdata')
def newdata():
    if 'user' not in session:
        return redirect(url_for('login'))
    try:
        return render_template('newdata.html')
    except Exception as e:
        print(f"Error rendering newdata: {e}")
        return "Error loading data entry page", 500

def generate_recommendation(age, bs, systolic_bp, diastolic_bp, heart_rate, hemoglobin_level, body_temp, thyroid):
    recs = []

    # Blood Sugar (bs in mmol/L)
    if bs > 9.0:
        recs.append("High blood sugar detected. Follow a diabetic-friendly diet and consult a diabetologist.")
       # recs.append("Include daily physical activity like brisk walking or prenatal yoga.")
    elif bs < 4.0:
        recs.append("Low blood sugar. Eat small, balanced meals regularly and carry healthy snacks to avoid hypoglycemia.")
    elif 4.0 <= bs <= 8.0:
        recs.append("Blood sugar is normal. Maintain a balanced diet with whole grains.")

    # Blood Pressure
    if systolic_bp > 140 or diastolic_bp > 90:
        recs.append("High blood pressure. Limit salt, and focus on potassium-rich foods.")
        recs.append("Practice stress management.")
    elif systolic_bp < 90 or diastolic_bp < 60:
        recs.append("Low blood pressure. Stay hydrated, and consult your healthcare provider.")
    elif 90 <= systolic_bp <= 120 and 60 <= diastolic_bp <= 80:
        recs.append("Blood pressure is normal.")

    # Heart Rate
    if heart_rate > 110:
        recs.append("Tachycardia detected. Reduce caffeine and stress. Seek medical evaluation if persistent.")
    elif heart_rate < 60:
        recs.append("Bradycardia. If not an athlete, consult a doctor to rule out any cardiac condition.")
    elif 60 <= heart_rate <= 100:
        recs.append("Heart rate is normal.")

    # Hemoglobin
    if hemoglobin_level < 7.0:
        recs.append("Severe anemia. Begin iron supplements and eat iron-rich foods.")
    elif 7.0 <= hemoglobin_level < 9.9:
        recs.append("Moderate anemia. Increase dietary iron with foods.")
    elif 9.9 <= hemoglobin_level < 11.0:
        recs.append("Mild anemia. Maintain an iron-rich diet and monitor regularly.")
    elif 11.0 <= hemoglobin_level <= 14.0:
        recs.append("Hemoglobin level is normal. Continue a balanced diet with adequate iron and vitamin C.")
    elif hemoglobin_level > 14.0:
        recs.append("High hemoglobin level. May be due to dehydration. Increase hydration and consult your doctor.")

    # Body Temperature
    if body_temp <= 97.0:
        recs.append("Low body temperature detected. Consult a doctor to rule out hypothermia or circulation issues.")
    elif body_temp >= 101.0:
        recs.append("Fever detected. Stay hydrated, and consult a doctor if persistent.")
    elif 97.0 < body_temp < 101.0:
        recs.append("Body temperature is normal.")

    # Age
    if age <= 15:
        recs.append("Early maternal age. Receive additional support and education regarding prenatal care and nutrition.")
    elif age >= 35:
        recs.append("Advanced maternal age. Ensure frequent prenatal checkups and regular monitoring.")
    elif 16 <= age <= 34:
        recs.append("Maternal age is optimal. Follow standard prenatal care and maintain a healthy lifestyle.")

    # Thyroid
    if thyroid == "Yes":
        recs.append("Thyroid condition detected. Follow up with an endocrinologist.")
    else:
        recs.append("Thyroid condition not reported. Continue a balanced diet with adequate iodine.")

    # Default if no specific issues
    if not recs:
        recs.append("All health parameters appear normal. Continue regular prenatal checkups and maintain a balanced diet and active lifestyle.")

    return " ".join(recs)


def generate_risk_factors(age, bs, systolic_bp, diastolic_bp, heart_rate, hemoglobin_level, body_temp, thyroid):
    risk_factors = []

    if bs > 9 or bs < 4:
        risk_factors.append("Abnormal Blood Sugar")

    if systolic_bp > 140 or diastolic_bp > 90:
        risk_factors.append("Hypertension")
    elif systolic_bp < 90 or diastolic_bp < 60:
        risk_factors.append("Hypotension")

    if heart_rate > 110:
        risk_factors.append("Tachycardia")
    elif heart_rate < 60:
        risk_factors.append("Bradycardia")

    if hemoglobin_level < 8.8:
        risk_factors.append("Severe Anemia")
    elif 8.8 <= hemoglobin_level < 10.8:
        risk_factors.append("Moderate Anemia")
    elif 10.8 <= hemoglobin_level < 12.6:
        risk_factors.append("Mild Anemia")
    elif hemoglobin_level > 14:
        risk_factors.append("High Hemoglobin")

    if age <= 15:
        risk_factors.append("Early Maternal Age")
    if age >= 35:
        risk_factors.append("Advanced Maternal Age")

    if body_temp<=97 or body_temp>=101:
        risk_factors.append("Body temperature")

    if thyroid == "Yes":
        risk_factors.append("Thyroid Dysfunction")

    return risk_factors if risk_factors else ["None"]


def generate_health_guidance(risk_factors):
    plans = []

    # Risk factor grouping
    rf_map = {
        "Abnormal Blood Sugar": "Blood Sugar",
        "Hypertension": "Blood Pressure",
        "Hypotension": "Blood Pressure",
        "Tachycardia": "Heart Rate",
        "Bradycardia": "Heart Rate",
        "Severe Anemia": "Hemoglobin",
        "Moderate Anemia": "Hemoglobin",
        "Mild Anemia": "Hemoglobin",
        "High Hemoglobin": "Hemoglobin",
        "Advanced Maternal Age": "Advanced Maternal Age",
        "Thyroid Dysfunction": "Thyroid Condition"
    }

    normalized = {rf_map[rf] for rf in risk_factors if rf in rf_map}

    # Diet Plan
    diet = []
    if "Blood Sugar" in normalized:
        diet.append("Blood Sugar Control:\n• Eat whole grains and fiber-rich veggies\n• Avoid sweets and refined carbs")
    if "Blood Pressure" in normalized:
        diet.append("Blood Pressure Management:\n• Reduce salt and processed food\n• Eat potassium-rich foods (e.g., banana, spinach)")
    if "Hemoglobin" in normalized:
        diet.append("Improve Hemoglobin:\n• Eat spinach, dates, and red meat\n• Pair iron-rich foods with vitamin C sources")
    if "Thyroid Condition" in normalized:
        diet.append("Thyroid Care:\n• Avoid raw cruciferous veggies in excess\n• Include selenium-rich foods like Brazil nuts")
    if "Advanced Maternal Age" in normalized:
        diet.append("Maternal Age Nutrition:\n• Take prenatal vitamins\n• Include antioxidant-rich foods and stay hydrated")

    plans.append("**Diet Plan:**\n" + ("\n\n".join(diet) if diet else "Follow a balanced diet suitable for pregnancy."))

    # Exercise Plan
    exercise = []
    if "Blood Sugar" in normalized:
        exercise.append("• Walk 20–30 mins daily\n• Prenatal yoga")
    if "Blood Pressure" in normalized:
        exercise.append("• Deep breathing\n• Moderate walking")
    if "Heart Rate" in normalized:
        exercise.append("• Reduce stimulants\n• Try meditation")
    if "Hemoglobin" in normalized:
        exercise.append("• Gentle movement")
    if "Thyroid Condition" in normalized:
        exercise.append("• Light yoga\n• Consistent routine")
    if "Advanced Maternal Age" in normalized:
        exercise.append("• Pelvic floor exercises\n• Light prenatal movement")

    plans.append("\n\n**Exercise Plan:**\n" + ("\n\n".join(exercise) if exercise else "Continue regular light exercise as advised."))

    # Preventive Measures
    preventive = []
    if "Blood Sugar" in normalized:
        preventive.append("• Monitor glucose regularly\n• Avoid sugary snacks")
    if "Blood Pressure" in normalized:
        preventive.append("• Monitor BP weekly\n• Limit stress and salt")
    if "Heart Rate" in normalized:
        preventive.append("• Avoid caffeine\n• Sleep 7–8 hours")
    if "Hemoglobin" in normalized:
        preventive.append("• Recheck hemoglobin monthly\n• Maintain iron intake\n• Stay well-hydrated")
    if "Thyroid Condition" in normalized:
        preventive.append("• Take medication consistently\n• Check TSH every trimester")
    if "Advanced Maternal Age" in normalized:
        preventive.append("• Frequent prenatal checkups are  suggested")

    plans.append("\n\n**Preventive Measures:**\n" + ("\n\n".join(preventive) if preventive else "Maintain prenatal checkups and a healthy routine."))

    return "\n\n".join(plans)

import pandas as pd  # Add at the top of app.py

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        print("Redirecting to login: No user in session")
        return redirect(url_for('login'))
    
    if model is None:
        print(f"Error: Model is None. Check file paths: {MODEL_PATH}, {FEATURES_PATH}, {LABELENCODERS_PATH}")
        session['error'] = "Model not loaded. Please contact support."
        session['prediction'] = None
        return redirect(url_for('result'))

    try:
        data = request.json
        if not data:
            print("Error: No JSON data received in request")
            session['error'] = "No input data provided. Please fill out the form."
            session['prediction'] = None
            return redirect(url_for('result'))
        
        #print(f"Received JSON data: {data}")
        
        # Extract and validate inputs
        required_fields = ['age', 'systolic_bp', 'diastolic_bp', 'bs', 'body_temp', 'heart_rate', 'hemoglobin_level', 'thyroid']
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        if missing_fields:
            print(f"Error: Missing fields: {missing_fields}")
            session['error'] = f"Missing required fields: {', '.join(missing_fields)}."
            session['prediction'] = None
            return redirect(url_for('result'))
        
        age = float(data.get("age"))
        systolic_bp = float(data.get("systolic_bp"))
        diastolic_bp = float(data.get("diastolic_bp"))
        bs_mg_dl = float(data.get("bs"))
        bs = bs_mg_dl / 18.0
        body_temp = float(data.get("body_temp"))
        heart_rate = float(data.get("heart_rate"))
        hemoglobin_level = float(data.get("hemoglobin_level"))
        thyroid_value = data.get("thyroid")
        thyroid_mapped = 1 if thyroid_value == "Yes" else 0
        
        # print(f"Processed inputs: age={age}, systolic_bp={systolic_bp}, diastolic_bp={diastolic_bp}, "
        #       f"bs_mg_dl={bs_mg_dl}, bs={bs}, body_temp={body_temp}, heart_rate={heart_rate}, "
        #       f"hemoglobin_level={hemoglobin_level}, thyroid={thyroid_value}")
        
        # Feature engineering
        bp_ratio = systolic_bp / diastolic_bp if diastolic_bp != 0 else 0
        risk_score = (bs * 0.3) + (age * 0.2) + (systolic_bp * 0.15)
        age_bp_interaction = age * (systolic_bp + diastolic_bp)
        bs_heart_rate_interaction = bs * heart_rate
        
        hemoglobin_low = 1 if hemoglobin_level <= 8.78 else 0
        hemoglobin_medium = 1 if 8.78 < hemoglobin_level <= 10.74 else 0
        hemoglobin_high = 1 if 10.74 < hemoglobin_level <= 12.5975 else 0
        hemoglobin_very_high = 1 if hemoglobin_level > 12.5975 else 0
        
        features = [
            age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate, hemoglobin_level, thyroid_mapped,
            bp_ratio, risk_score, age_bp_interaction, bs_heart_rate_interaction,
            hemoglobin_low, hemoglobin_medium, hemoglobin_high, hemoglobin_very_high
        ]
        
        # print(f"Features: {features}")
        # print(f"Feature names: {feature_names}")
        
        # Convert to DataFrame
        if len(feature_names) != len(features):
            print(f"Error: Feature names length ({len(feature_names)}) does not match features length ({len(features)})")
            session['error'] = "Model configuration error: Feature mismatch."
            session['prediction'] = None
            return redirect(url_for('result'))
        
        features_df = pd.DataFrame([features], columns=feature_names)
        #print(f"Features DataFrame: {features_df.to_dict()}")
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        print(f"Raw prediction: {prediction}")
        risk_levels = ["Low", "Medium", "High"]
        risk = risk_levels[int(prediction)]
        
        # Generate recommendation and risk factors
        recommendation = generate_recommendation(age, bs, systolic_bp, diastolic_bp, heart_rate, hemoglobin_level, body_temp, thyroid_value)
        risk_factors = generate_risk_factors(age, bs, systolic_bp, diastolic_bp, heart_rate, hemoglobin_level, body_temp, thyroid_value)
        health_guidance = generate_health_guidance(risk_factors)
        
        # print(f"Recommendation: '{recommendation}'")  # Log with quotes to show empty string
        # print(f"Risk factors: {risk_factors}")
        # print(f"Health guidance: {health_guidance}")
        
        # Store prediction in session
        session['prediction'] = {
            "risk": risk,
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "health_guidance": health_guidance,
            "details": {
                "age": age,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "bs": bs_mg_dl,
                "body_temp": body_temp,
                "heart_rate": heart_rate,
                "hemoglobin_level": hemoglobin_level,
                "thyroid": thyroid_value
            }
        }
        #print(f"Stored in session: {session['prediction']}")
        session.pop('error', None)
        return redirect(url_for('result'))
    
    except ValueError as e:
        print(f"ValueError in predict: {e}")
        session['error'] = f"Invalid input data: {str(e)}. Please check your entries."
        session['prediction'] = None
        return redirect(url_for('result'))
    except Exception as e:
        print(f"Unexpected error in predict: {e}")
        session['error'] = f"Prediction failed: {str(e)}. Please try again or contact support."
        session['prediction'] = None
        return redirect(url_for('result'))

@app.route('/save_result', methods=['POST'])
def save_result():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    prediction_data = session.get('prediction')
    if not prediction_data:
        return redirect(url_for('user_dashboard'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO records (
                user_email, date, prediction, recommendation, details, risk_factors
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session['user'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            prediction_data['risk'],
            prediction_data['recommendation'],
            json.dumps(prediction_data['details']),
            json.dumps(prediction_data['risk_factors'])  # <- NEW
        ))
        conn.commit()
        session.pop('prediction', None)
        return redirect(url_for('user_dashboard'))
    except Exception as e:
        print(f"Error saving result: {e}")
        return redirect(url_for('user_dashboard'))
    finally:
        conn.close()


@app.route('/result')
def result():
    if 'user' not in session:
        return redirect(url_for('login'))
    prediction = session.get('prediction', None)
    error = session.get('error', None)
    try:
        return render_template('result.html', prediction=prediction, error=error)
    except Exception as e:
        print(f"Error rendering result: {e}")
        return "Error loading result page", 500

@app.route('/doctor_recommendation/<int:record_id>/<user_email>')
def doctor_recommendation(record_id, user_email):
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT name FROM users WHERE email = ?", (user_email,))
        user_result = c.fetchone()
        username = user_result[0] if user_result else "Unknown"
        
        c.execute("SELECT id, doctor_recommendation FROM records WHERE user_email = ? ORDER BY date DESC LIMIT 10", (user_email,))
        records = c.fetchall()
        if not records or record_id >= len(records):
            return render_template('doctor_recommendation.html', error="Invalid record ID", username=username, record_id=record_id, current_recommendation='', user_email=user_email)
        
        current_recommendation = records[record_id][1] or ''
        return render_template('doctor_recommendation.html', username=username, record_id=record_id, current_recommendation=current_recommendation, error=None, user_email=user_email)
    except Exception as e:
        print(f"Database error: {e}")
        return render_template('doctor_recommendation.html', error="Error retrieving data", username="Unknown", record_id=record_id, current_recommendation='', user_email=user_email)
    finally:
        conn.close()

@app.route('/save_doctor_recommendation/<int:record_id>/<user_email>', methods=['POST'])
def save_doctor_recommendation(record_id, user_email):
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    doctor_recommendation = request.form.get('doctor_recommendation')
    selected_user = request.form.get('selected_user', user_email)  # Preserve filter state
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT id FROM records WHERE user_email = ? ORDER BY date DESC LIMIT 10", (user_email,))
        records = c.fetchall()
        if not records or record_id >= len(records):
            return redirect(url_for('admin_dashboard', user_email=selected_user))
        
        record_id_to_update = records[record_id][0]
        c.execute("UPDATE records SET doctor_recommendation = ? WHERE id = ?", (doctor_recommendation, record_id_to_update))
        conn.commit()
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        conn.close()
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
        total_users = c.fetchone()[0]
        c.execute("SELECT email FROM users WHERE role = 'user'")
        users = [row[0] for row in c.fetchall()]
        
        selected_user = request.args.get('user_email', None)
        selected_user_name = None
        if selected_user:
            c.execute("SELECT name FROM users WHERE email = ?", (selected_user,))
            result = c.fetchone()
            selected_user_name = result[0] if result else "Unknown"

        # Include risk_factors column
        query = """
            SELECT user_email, date, prediction, recommendation, details,
                   doctor_recommendation, risk_factors
            FROM records
        """
        params = ()
        if selected_user:
            query += " WHERE user_email = ?"
            params = (selected_user,)
        query += " ORDER BY date DESC LIMIT 10"
        
        c.execute(query, params)
        records = []
        for row in c.fetchall():
            c.execute("SELECT name FROM users WHERE email = ?", (row[0],))
            user_result = c.fetchone()
            user_name = user_result[0] if user_result else "Unknown"

            details = json.loads(row[4]) if row[4] else {}
            risk_factors = json.loads(row[6]) if row[6] else []

            records.append({
                "user_name": user_name,
                "user_email": row[0],
                "date": row[1],
                "risk": row[2],
                "recommendation": row[3],
                "details": details,
                "doctor_recommendation": row[5],
                "risk_factors": risk_factors  # ✅ added
            })

        return render_template(
            'admin_dashboard.html',
            prediction=None,
            records=records,
            error=None,
            name=session.get('name', 'Admin'),
            total_users=total_users,
            users=users,
            selected_user=selected_user,
            selected_user_name=selected_user_name
        )
    except Exception as e:
        print(f"Database error: {e}")
        return render_template(
            'admin_dashboard.html',
            prediction=None,
            records=[],
            error="Error retrieving data",
            name=session.get('name', 'Admin'),
            total_users=0,
            users=[],
            selected_user=None,
            selected_user_name=None
        )
    finally:
        conn.close()

@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except Exception as e:
        print(f"Error rendering about: {e}")
        return "Error loading about page", 500


# Flask runner
def run_flask():
    init_db()
    app.run(port=5003, debug=True, use_reloader=False, threaded=True)

# PyQt5 GUI
def run_gui():
    qt_app = QApplication(sys.argv)
    window = QWebEngineView()
    window.setWindowTitle("Maternal Health Risk Predictor")
    window.resize(800, 600)
    window.load(QUrl("http://localhost:5003/"))
    window.show()
    sys.exit(qt_app.exec_())

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    run_gui()
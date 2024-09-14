from flask import Flask, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///contacts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Set the system prompt for the Groq chatbot
system_prompt = {
    "role": "system",
    "content": "You are a career counselor. Provide guidance and advice on career choices and related questions only."
}

# Define database models
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<Contact {self.name}>'

class Subscribe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False, unique=True)

    def __repr__(self):
        return f'<Subscribe {self.email}>'

# Load the trained model
linear_svc = joblib.load('linear_svc_model.joblib')

# Skill mapping and career mapping
skill_map = {
    "Not Interested": 0,
    "Poor": 1,
    "Beginner": 2,
    "Average": 3,
    "Intermediate": 4,
    "Advanced": 5,
    "Professional": 6
}

career_map = {
    0: "Data Scientist",
    1: "IoT Developer",
    2: "Application Developer",
    3: "Web Developer",
    4: "Security Professional",
    5: "Data Analyst",
    6: "Business Analyst",
    7: "Database Administrator",
    8: "Networking Engineer",
    9: "Software Tester",
    10: "Software Developer",
    11: "Technical Writer",
    12: "Project Manager",
    13: "Graphics Designer",
    14: "System Administrator",
    15: "Hardware Engineer",
    16: "AI ML Specialist"
}

# Load your dataset to get career counts
df = pd.read_csv('D:/dessertation/app/backend/dataset9000.csv', sep=',')
career_counts = df['Role'].value_counts()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    skills = data.get('skills', [])

    if len(skills) != 17:
        return jsonify({'error': 'Invalid number of skills'}), 400

    try:
        skill_values = [skill_map[skill] for skill in skills]
    except KeyError:
        return jsonify({'error': 'Invalid skill rating'}), 400

    input_data = np.array(skill_values).reshape(1, -1)
    prediction = linear_svc.predict(input_data)

    predicted_career = career_map.get(prediction[0], "Unknown Career")

    # Save the graph as an image file
    img_path = f'static/career_graph_{prediction[0]}.png'

    plt.figure(figsize=(12, 8))  # Adjust the figure size here

    careers = career_counts.index
    counts = career_counts.values

    plt.plot(careers, counts, marker='o', linestyle='-', color='blue')

    if predicted_career in careers:
        index = list(careers).index(predicted_career)
        plt.plot(careers[index], counts[index], marker='o', color='red', markersize=10, label=predicted_career)

    plt.xlabel('Career')
    plt.ylabel('Number of People')
    plt.title(f"Number of People Who Chose the Career: {predicted_career}")
    plt.xticks(rotation=45)

    if predicted_career in careers:
        plt.legend()

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(img_path)
    plt.close()

    return jsonify({
        'career': predicted_career,
        'image_url': f'http://127.0.0.1:5000/{img_path}'
    })

@app.route('/static/<path:filename>')
def serve_image(filename):
    return send_file(os.path.join('static', filename))

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    sendMessage = data.get('message')  # Update this to match the payload

    if not name or not email or not sendMessage:
        return jsonify({'error': 'Missing fields'}), 400

    contact = Contact(name=name, email=email, message=sendMessage)  # Map to 'sendMessage'
    db.session.add(contact)
    db.session.commit()

    return jsonify({'message': 'Contact submitted successfully'})

@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # Save the email to the database
    subscription = Subscribe(email=email)
    db.session.add(subscription)
    db.session.commit()

    return jsonify({'message': 'Subscription successful'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Set the chat history with the system prompt
    chat_history = [system_prompt, {"role": "user", "content": message}]

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_history,
            max_tokens=100,
            temperature=1.2
        )
        reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'reply': reply})

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    with app.app_context():
        db.create_all()  # Create database tables if not exist
    app.run(debug=True)

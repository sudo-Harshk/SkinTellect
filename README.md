# SkinTellect - AI-Powered Skin Care Analysis

SkinTellect is a web application that uses artificial intelligence to analyze skin conditions and provide personalized skincare recommendations.

## Features

- AI-powered skin condition analysis
- Personalized skincare product recommendations
- Appointment booking system
- User authentication and profile management
- Email and SMS notifications
- Doctor's dashboard for appointment management

## Tech Stack

- Python 3.12
- Flask
- SQLite
- Roboflow AI
- Twilio (SMS)
- Flask-Mail
- HTML/CSS/JavaScript

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/sudo-Harshk/Skintelite.git
cd Skintelite
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:
```env
# Flask Configuration
FLASK_SECRET_KEY=your_secret_key
DATABASE_PATH=app.db

# Twilio Credentials
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number

# Email Configuration
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_email_password
MAIL_DEFAULT_SENDER_NAME=SkinTellect
MAIL_DEFAULT_SENDER_EMAIL=your_email@gmail.com
MAIL_MAX_EMAILS=5
MAIL_ASCII_ATTACHMENTS=False
MAIL_USE_SSL=False

# Roboflow API Keys
ROBOFLOW_API_KEY=your_roboflow_api_key
ROBOFLOW_INFERENCE_API_KEY=your_roboflow_inference_api_key
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Roboflow for AI model hosting
- Twilio for SMS services
- Flask community for the web framework

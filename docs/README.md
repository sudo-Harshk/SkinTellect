# SkinTellect - AI-Powered Skin Care Analysis

SkinTellect is a Flask-based web application that uses AI to analyze skin conditions and provide personalized skincare product recommendations.

## Features

- **AI Skin Analysis** - Dual-model detection using Roboflow + HuggingFace
- **Product Recommendations** - Personalized skincare products based on detected conditions
- **Appointment Booking** - Schedule consultations with dermatologists
- **User Authentication** - Secure login, registration, OTP verification
- **Email & SMS Notifications** - Twilio SMS + Flask-Mail integration
- **Doctor Dashboard** - Manage appointments and patient consultations

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.12, Flask |
| Database | SQLite |
| AI Models | Roboflow (skin conditions), HuggingFace (skin type) |
| SMS | Twilio |
| Email | Flask-Mail (Zoho SMTP) |
| Frontend | HTML, CSS, JavaScript, Tailwind CSS |

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/sudo-Harshk/Skintelite.git
cd Skintelite

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
# Copy .env.example to .env and fill in your API keys

# 5. Run the application
python app.py
```

Application runs at: `http://localhost:5000`

## Documentation

| Document | Description |
|----------|-------------|
| [SETUP.md](docs/SETUP.md) | Detailed installation guide |
| [API.md](docs/API.md) | API endpoints documentation |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture overview |

## AI Detection Models

### Primary: Roboflow
- Detects: whitehead, blackhead, papule, pustule, freckles, PIH, dark circles
- Confidence threshold: 15%
- Annotates images with bounding boxes

### Secondary: HuggingFace
- Model: `dima806/skin_types_image_detection`
- Detects: oily, dry, normal, combination skin types
- Used for skin type classification

## Environment Variables

```env
# Required API Keys
ROBOFLOW_API_KEY=your_key
ROBOFLOW_INFERENCE_API_KEY=your_key
HUGGINGFACE_API_KEY=hf_xxxxx

# Email (Zoho SMTP)
MAIL_SERVER=smtp.zoho.in
MAIL_USERNAME=your_email
MAIL_PASSWORD=your_password

# Twilio SMS
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=+1234567890
```

## Project Structure

```
skin-care-complete/
├── app.py                 # Main Flask application
├── templates/             # HTML templates (26 files)
├── static/                # CSS, JS, images
├── dataset/
│   └── skincare_products.json  # Product database
├── model/
│   └── final_model.h5     # Local Keras model (optional)
├── docs/                   # Documentation
└── requirements.txt        # Python dependencies
```

## License

MIT License

## Acknowledgments

- Roboflow for AI model hosting
- HuggingFace for skin type detection model
- Twilio for SMS services
- Flask community

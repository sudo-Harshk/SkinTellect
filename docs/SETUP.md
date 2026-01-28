# Setup Guide - SkinTellect

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git
- Internet connection (for API calls)

## Step 1: Clone Repository

```bash
git clone https://github.com/sudo-Harshk/Skintelite.git
cd Skintelite
```

## Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Email Configuration (Zoho SMTP)
MAIL_SERVER=smtp.zoho.in
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USE_SSL=False
MAIL_USERNAME=your_email@domain.com
MAIL_PASSWORD=your_password
MAIL_DEFAULT_SENDER_NAME=SkinTellect
MAIL_DEFAULT_SENDER_EMAIL=your_email@domain.com
MAIL_MAX_EMAILS=5
MAIL_ASCII_ATTACHMENTS=False

# Twilio SMS
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# Roboflow API Keys
ROBOFLOW_API_KEY=your_roboflow_api_key
ROBOFLOW_INFERENCE_API_KEY=your_roboflow_inference_key

# HuggingFace API Key (for fallback skin type detection)
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx
```

## Step 5: Get API Keys

### Roboflow
1. Go to [roboflow.com](https://roboflow.com)
2. Create account and project
3. Get API key from Settings

### HuggingFace
1. Go to [huggingface.co](https://huggingface.co)
2. Create account
3. Settings → Access Tokens → Create new token (read permissions)

### Twilio
1. Go to [twilio.com](https://www.twilio.com)
2. Create account
3. Get Account SID, Auth Token, and phone number

## Step 6: Run Application

```bash
python app.py
```

Application will be available at: `http://localhost:5000`

## Step 7: Verify Installation

1. Open browser to `http://localhost:5000`
2. Register a new account
3. Go to Skin Analysis (`/predict`)
4. Upload a face image
5. Verify detection results and product recommendations appear

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Roboflow connection error | Check API keys in `.env` |
| HuggingFace 410 error | Update API URL to `router.huggingface.co` |
| Email not sending | Verify Zoho SMTP credentials |
| Port 5000 in use | Change port in `app.py` or kill process |

### Test HuggingFace API

```bash
python test_huggingface.py
```

Expected output: `✅ SUCCESS! Top prediction: oily (XX.X%)`

## Development Mode

The app runs in debug mode by default. For production:

```python
# In app.py, change:
app.run(debug=False, host='0.0.0.0', port=5000)
```

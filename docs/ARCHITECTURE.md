# System Architecture - SkinTellect

## Overview

SkinTellect is a Flask-based web application that combines multiple AI services for skin analysis and provides personalized skincare product recommendations.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Browser  │────▶│   Flask Server   │────▶│   SQLite DB     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌───────────┐  ┌────────────┐  ┌───────────┐
        │ Roboflow  │  │ HuggingFace│  │  Twilio   │
        │    API    │  │    API     │  │    SMS    │
        └───────────┘  └────────────┘  └───────────┘
```

## Components

### 1. Flask Application (`app.py`)

**Core Functions:**
- User authentication (login, register, password reset)
- Session management
- Image upload and processing
- AI model integration
- Product recommendation engine

### 2. AI Detection Pipeline

```
Image Upload
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  ROBOFLOW (Primary)                                    │
│  - Model: skin-detection-pfmbg                         │
│  - Detects: whitehead, blackhead, acne, freckles, etc  │
│  - Returns: bounding boxes + labels                    │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  HUGGINGFACE (Secondary)                               │
│  - Model: dima806/skin_types_image_detection           │
│  - Detects: oily, dry, normal, combination             │
│  - Returns: skin type classification                   │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  ROBOFLOW OILINESS (Additional)                        │
│  - Model: oilyness-detection-kgsxz                     │
│  - Fine-grained oiliness detection                     │
└────────────────────────────────────────────────────────┘
    │
    ▼
Combine Results → Product Recommendations
```

### 3. Product Recommendation Engine

**File:** `dataset/skincare_products.json`

**Logic:**
1. Collect detected skin conditions
2. Map conditions to product categories
3. Fetch matching products from database
4. Return 1-3 products per condition

**Mapping:**
| Detection | Product Category |
|-----------|-----------------|
| oily (HuggingFace) | oily skin |
| dry (HuggingFace) | dryness |
| whitehead (Roboflow) | whitehead |
| blackhead (Roboflow) | blackhead |

### 4. Database Schema

**Tables:**
- `users` - User accounts
- `appointments` - Dermatologist bookings
- `surveys` - Skin questionnaire responses

### 5. External Services

| Service | Purpose | Config |
|---------|---------|--------|
| Roboflow | Object detection | `ROBOFLOW_API_KEY` |
| HuggingFace | Skin type classification | `HUGGINGFACE_API_KEY` |
| Twilio | SMS notifications | `TWILIO_ACCOUNT_SID` |
| Zoho SMTP | Email notifications | `MAIL_USERNAME` |

## File Structure

```
skin-care-complete/
├── app.py                  # Main application (1200+ lines)
├── .env                    # Environment variables (secrets)
├── requirements.txt        # Python dependencies
├── app.db                  # SQLite database
│
├── templates/              # HTML templates
│   ├── layout.html         # Base template
│   ├── index.html          # Landing page
│   ├── face_analysis.html  # Skin analysis results
│   ├── login.html
│   ├── register.html
│   └── ...
│
├── static/
│   ├── css/style.css
│   └── assets/
│
├── dataset/
│   ├── skincare_products.json  # Product database
│   └── cosmetics.csv
│
├── model/
│   └── final_model.h5      # Local Keras model (unused)
│
└── docs/
    ├── README.md
    ├── SETUP.md
    ├── API.md
    └── ARCHITECTURE.md
```

## Data Flow

```
1. User uploads face image
        ↓
2. Image saved to static/ folder
        ↓
3. Roboflow API called (detect conditions)
        ↓
4. HuggingFace API called (detect skin type)
        ↓
5. Roboflow Oiliness model called
        ↓
6. All detections combined into unique_classes set
        ↓
7. Product recommendations fetched from JSON
        ↓
8. Annotated image created (bounding boxes)
        ↓
9. Results rendered in face_analysis.html
```

## Error Handling

**Fallback Strategy:**
1. If Roboflow fails → Continue with HuggingFace
2. If HuggingFace fails → Use default skin type
3. If both fail → Show generic recommendations

## Performance

| Operation | Expected Time |
|-----------|--------------|
| Roboflow API | 2-4 seconds |
| HuggingFace API | 2-5 seconds |
| Total analysis | 5-10 seconds |

## Security

- Passwords hashed with Werkzeug
- Session-based authentication
- API keys stored in `.env` (not committed)
- Input validation on uploads

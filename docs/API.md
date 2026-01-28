# API Documentation - SkinTellect

## Base URL
```
http://localhost:5000
```

## Authentication

Most endpoints require user authentication via session cookies.

---

## Public Routes

### `GET /`
**Description:** Landing page  
**Auth Required:** No

### `GET /login`
**Description:** Login page  
**Auth Required:** No

### `POST /login`
**Description:** Authenticate user  
**Auth Required:** No  
**Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

### `GET /register`
**Description:** Registration page  
**Auth Required:** No

### `POST /register`
**Description:** Create new user account  
**Auth Required:** No

---

## Authenticated Routes

### `GET /predict`
**Description:** Skin analysis page (upload form)  
**Auth Required:** Yes

### `POST /predict`
**Description:** Analyze uploaded face image  
**Auth Required:** Yes  
**Content-Type:** `multipart/form-data`  
**Body:**
```
image: File (JPEG, PNG)
```
**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `force_fallback` | boolean | Force HuggingFace detection only |

**Response:** Renders `face_analysis.html` with:
- Detected skin conditions
- Annotated image with bounding boxes
- Product recommendations

**Example:**
```
POST /predict?force_fallback=true
```

### `GET /profile`
**Description:** User profile page  
**Auth Required:** Yes

### `GET /bookappointment`
**Description:** Book dermatologist appointment  
**Auth Required:** Yes

### `POST /bookappointment`
**Description:** Submit appointment request  
**Auth Required:** Yes

---

## Doctor Routes

### `GET /allappoint`
**Description:** View all appointments (doctor dashboard)  
**Auth Required:** Yes (doctor role)

---

## AI Detection Endpoints

### Roboflow (Primary)
- **Model:** `skin-detection-pfmbg`
- **Confidence:** 15%
- **Detects:** whitehead, blackhead, papule, pustule, freckles, PIH, dark circles

### HuggingFace (Secondary)
- **Endpoint:** `https://router.huggingface.co/hf-inference/models/dima806/skin_types_image_detection`
- **Method:** POST
- **Headers:** `Authorization: Bearer {API_KEY}`, `Content-Type: image/jpeg`
- **Detects:** oily, dry, normal, combination

**Response Format:**
```json
[
  {"label": "oily", "score": 0.85},
  {"label": "dry", "score": 0.10},
  {"label": "normal", "score": 0.05}
]
```

---

## Product Recommendations

Products are fetched from `dataset/skincare_products.json`

**Categories:**
- `dryness`
- `oily skin`
- `normal/dry skin`
- `whitehead`
- `blackhead`
- `papule`
- `pustule`
- `PIH`
- `Dark Circle`

**Product Object:**
```json
{
  "Brand": "Neutrogena",
  "Name": "Hydro Boost Serum",
  "Image_URL": "https://...",
  "Product_URL": "https://amazon.in/..."
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
| 302 | Redirect to login (not authenticated) |
| 400 | Invalid image format |
| 500 | Server error |

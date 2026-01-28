"""
Test script to verify HuggingFace API works for skin type detection
Run this: python test_huggingface.py
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/hf-inference/models/dima806/skin_types_image_detection"
API_KEY = os.getenv('HUGGINGFACE_API_KEY')

print("=" * 50)
print("HuggingFace API Test for SkinTellect")
print("=" * 50)

# Check API key
if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
    print("❌ ERROR: HUGGINGFACE_API_KEY not set in .env")
    print("   Please add your API key to the .env file")
    exit(1)

print(f"✅ API Key found: {API_KEY[:10]}...")

headers = {"Authorization": f"Bearer {API_KEY}"}

# Check for test image
test_images = [
    "static/test_face.jpg",
    "static/test.jpg",
    "static/face.jpg",
    "static/test_face.png"
]

test_image = None
for img in test_images:
    if os.path.exists(img):
        test_image = img
        break

if not test_image:
    print("⚠️  No test image found. Creating a simple test...")
    print("   For proper testing, add a face image to static/test_face.jpg")
    
    # Try with a simple API ping
    response = requests.get(
        "https://api-inference.huggingface.co/models/dima806/skin_types_image_detection",
        headers=headers,
        timeout=10
    )
    print(f"📊 API Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ API key is valid!")
        print("   Add a test face image to static/test_face.jpg and run again")
    else:
        print(f"❌ Error: {response.text}")
    exit(0)

print(f"📷 Using test image: {test_image}")

# Send request
with open(test_image, "rb") as f:
    data = f.read()

print("📤 Sending request to HuggingFace...")
try:
    if test_image.endswith('.png'):
        content_type = 'image/png'
    else:
        content_type = 'image/jpeg'
    
    request_headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": content_type
    }
    
    response = requests.post(
        API_URL, 
        headers=request_headers, 
        data=data,
        timeout=15
    )
    
    print(f"📊 Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"📄 Response: {results}")
        
        if results and len(results) > 0:
            top = max(results, key=lambda x: x.get("score", 0))
            print("=" * 50)
            print(f"✅ SUCCESS! Top prediction: {top['label']} ({top['score']*100:.1f}%)")
            print("=" * 50)
        else:
            print("⚠️  Empty response from API")
    else:
        print(f"❌ API Error: {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ Request timed out (>15 seconds)")
except Exception as e:
    print(f"❌ Error: {e}")

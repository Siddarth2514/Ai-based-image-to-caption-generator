import streamlit as st
from groq import Groq
from PIL import Image
import requests
import os
import base64

# ---------------------------
# 🔐 LOAD API KEYS
# ---------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY not set in environment variables.")
    st.stop()

if not HF_API_KEY:
    st.error("🚨 HUGGINGFACE_API_KEY not set in environment variables.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# 🤖 HUGGINGFACE IMAGE CAPTION API
# ---------------------------

HF_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

def generate_image_description(image_bytes):
    try:
        # Convert image to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": image_base64
            },
            timeout=60
        )

        print("HF STATUS:", response.status_code)
        print("HF RESPONSE:", response.text[:300])

        if response.status_code != 200:
            return f"⚠ HF Error {response.status_code}"

        result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"]

        if "error" in result:
            return f"⚠ {result['error']}"

        return "⚠ Unexpected response"

    except Exception as e:
        return f"⚠ Request failed: {str(e)}"

# ---------------------------
# 🧠 GROQ TEXT GENERATION
# ---------------------------

def generate_text_with_groq(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a creative social media assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq Error: {str(e)}"

# ---------------------------
# ✨ CAPTION + HASHTAG
# ---------------------------

def caption_generator(description):
    prompt = f"Generate 3 creative Instagram captions for: {description}"
    return generate_text_with_groq(prompt)

def hashtag_generator(description):
    prompt = f"Generate 10 trending Instagram hashtags for: {description}"
    return generate_text_with_groq(prompt)

# ---------------------------
# 🎯 SAMPLE SECTION
# ---------------------------

def sample():
    sp_images = {
        "Beach": "image/beach.png",
        "Coffee": "image/coffee.png",
        "Footballer": "image/footballer.png",
        "Mountain": "image/mountain.jpg"
    }

    cols = st.columns(4)

    for idx, (name, path) in enumerate(sp_images.items()):
        with cols[idx]:
            st.image(path, width=150)
            if st.button(f"Generate - {name}", key=f"sample_{idx}"):

                with open(path, "rb") as f:
                    image_bytes = f.read()

                description = generate_image_description(image_bytes)

                st.subheader("📄 Description")
                st.write(description)

                st.subheader("✨ Captions")
                st.write(caption_generator(description))

                st.subheader("#️⃣ Hashtags")
                st.write(hashtag_generator(description))

# ---------------------------
# 📤 UPLOAD SECTION
# ---------------------------

def upload():
    images = st.file_uploader(
        "Upload Images",
        accept_multiple_files=True,
        type=["jpg", "png", "jpeg"]
    )

    if images:
        if st.button("Generate"):
            for i, image in enumerate(images):
                image_bytes = image.getvalue()

                st.image(image, width=250)

                description = generate_image_description(image_bytes)

                st.subheader(f"📄 Description for Image {i+1}")
                st.write(description)

                st.subheader("✨ Captions")
                st.write(caption_generator(description))

                st.subheader("#️⃣ Hashtags")
                st.write(hashtag_generator(description))

# ---------------------------
# 🎨 MAIN UI
# ---------------------------

def main():
    st.set_page_config(
        page_title="AI Caption & Hashtag Generator",
        layout="centered"
    )

    st.title("📸 AI Caption & Hashtag Generator")
    st.write("Upload an image and generate captions + hashtags using AI 🚀")

    tab1, tab2 = st.tabs(["Upload Image", "Sample Images"])

    with tab1:
        upload()

    with tab2:
        sample()

if __name__ == "__main__":
    main()




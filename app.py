import streamlit as st
from groq import Groq
from PIL import Image
import os
import base64

# ---------------------------
# 🔐 LOAD API KEY
# ---------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY not set in environment variables.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# 👁️ IMAGE DESCRIPTION USING GROQ VISION
# ---------------------------

def generate_image_description(image_bytes):
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image clearly in one sentence."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Vision Error: {str(e)}"


# ---------------------------
# 🧠 CAPTION + HASHTAG USING GROQ
# ---------------------------

def generate_text(prompt):
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


def caption_generator(description):
    return generate_text(f"Generate 3 creative Instagram captions for: {description}")


def hashtag_generator(description):
    return generate_text(f"Generate 10 trending Instagram hashtags for: {description}")


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

                with st.spinner("Analyzing image..."):
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

    upload()


if __name__ == "__main__":
    main()


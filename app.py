import streamlit as st
from groq import Groq
from PIL import Image
import torch
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------
# 🔐 LOAD API KEY FROM STREAMLIT SECRETS
# ---------------------------


if "GROQ_API_KEY" not in st.secrets:
    st.error("🚨 GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# ---------------------------
# 🚀 CACHE MODEL
# ---------------------------

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    device = torch.device("cpu")  # Force CPU
    model.to(device)

    return processor, model, device


processor, model, device = load_model()

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
# 🖼 IMAGE CAPTION MODEL
# ---------------------------

def prediction(img_list):
    max_length = 30
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    images = []

    for image in img_list:
        img = Image.open(image)

        if img.mode != "RGB":
            img = img.convert("RGB")

        st.image(img, width=250)
        images.append(img)

    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output = model.generate(pixel_values, **gen_kwargs)
    captions = processor.batch_decode(output, skip_special_tokens=True)

    return [caption.strip() for caption in captions]

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
                description = prediction([path])[0]

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
            descriptions = prediction(images)

            for i, description in enumerate(descriptions):
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

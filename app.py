import os
from groq import Groq
import streamlit as st
import requests
from itertools import cycle
from tqdm import tqdm
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

client = Groq(api_key="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")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text_with_groq(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # You can change this to "llama2-70b-chat" if preferred
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def caption_generator(description):
    caption_prompt = f"Please generate three unique and creative captions for a photo that shows {description}."
    return generate_text_with_groq(caption_prompt)

def hashtag_generator(description):
    hashtag_prompt = f"Please generate ten relevant hashtags for a photo that shows {description}."
    return generate_text_with_groq(hashtag_prompt)

def prediction(img_list):
    max_length = 30
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    img = []

    for image in tqdm(img_list):
        i_image = Image.open(image)
        st.image(i_image, width=200)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        img.append(i_image)

    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)

    output = model.generate(pixel_val, **gen_kwargs)
    predict = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
    predict = [pred.strip() for pred in predict]

    return predict

def sample():
    sp_images = {
        'Sample 1': 'image/beach.png',
        'Sample 2': 'image/coffee.png',
        'Sample 3': 'image/footballer.png',
        'Sample 4': 'image/mountain.jpg'
    }

    colms = cycle(st.columns(4))

    for sp in sp_images.values():
        next(colms).image(sp, width=150)

    for i, sp in enumerate(sp_images.values()):
        if next(colms).button("Generate", key=i):
            description = prediction([sp])
            st.subheader("Description for the Image:")
            st.write(description[0])

            st.subheader("Captions for this image are:")
            captions = caption_generator(description[0])
            st.write(captions)

            st.subheader("#Hashtags")
            hashtags = hashtag_generator(description[0])
            st.write(hashtags)

def upload():
    with st.form("uploader"):
        images = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        submit = st.form_submit_button("Generate")

        if submit and images:
            descriptions = prediction(images)

            for i, description in enumerate(descriptions):
                st.subheader(f"Description for Image {i+1}:")
                st.write(description)

                st.subheader("Captions for this image are:")
                captions = caption_generator(description)
                st.write(captions)

                st.subheader("#Hashtags")
                hashtags = hashtag_generator(description)
                st.write(hashtags)

def main():
    st.set_page_config(page_title="Caption and Hashtag generation")
    st.title("Get Captions and Hashtags for your Image")

    tab1, tab2 = st.tabs(["Upload Image", "Sample"])

    with tab1:
        upload()

    with tab2:
        sample()

if __name__ == '__main__':
    main()

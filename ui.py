import streamlit as st
import requests
import numpy as np
from PIL import Image
import json
import io
import ast

API_ENDPOINT = "https://face-embedding-generator.herokuapp.com/predict"

# Create the header page content
st.title("Face Similarity Score Calculator")
st.markdown("### Calculate a similarity score between two images with faces in them!",
            unsafe_allow_html=True)

st.text("Please upload images of the people you want to compare")


def predict(img):
    bytes_image = img.getvalue()
    np_array = np.array(Image.open(io.BytesIO(bytes_image)))
    np_array = np_array[...,:3]

    # Send the image to the API
    response = requests.post(
        API_ENDPOINT, 
        headers={'Content-type': 'application/json'}, 
        data=json.dumps(np_array.tolist())
    )

    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Error: ", response.content)


def main():
    img1_file = st.file_uploader("Person 1:", type=["jpg", "png"], key="img1")
    img2_file = st.file_uploader("Person 2:", type=["jpg", "png"], key="img2")

    if img1_file is not None and img2_file is not None:
        with st.spinner("Calculating Similarity Score..."):
            img1_embedding = np.array(ast.literal_eval(predict(img1_file)))
            img2_embedding = np.array(ast.literal_eval(predict(img2_file)))
            score = numpy.linalg.norm(img1_embedding - img2_embedding)
            st.success(f"Similarity Score: {score}")

if __name__ == "__main__":
    main()

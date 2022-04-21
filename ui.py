import streamlit as st
import requests
import numpy as np
from PIL import Image
import json
import io

API_ENDPOINT = "https://face-embedding-generator.herokuapp.com/predict"

# Create the header page content
st.title("Face Similarity Service")
st.markdown("### Calculate a similarity score between two faces",
            unsafe_allow_html=True)

st.text("Please upload an image of each of the people you want to compare")


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
    img1_file = st.file_uploader("Upload an image", type=["jpg", "png"], key="img1")
    img2_file = st.file_uploader("Upload an image", type=["jpg", "png"], key="img2")

    if img1_file is not None and img2_file is not None:
        with st.spinner("Calculating Similarity Score..."):
            img1_embedding = predict(img1_file)
            img2_embedding = predict(img2_file)
            # print(type(img1_embedding), type(img2_embedding))
            st.success(f"Your similarity score is {img1_embedding}")

    # camera_input = st.camera_input("Or take a picture")
    # if camera_input is not None:
    #     with st.spinner("Predicting..."):
    #         prediction = float(predict(camera_input).strip("[").strip("]"))
    #         st.success(f"Your pet's cuteness score is {prediction:.3f}")


if __name__ == "__main__":
    main()

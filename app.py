import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the saved model
loaded_model = load_model(r"C:\Users\Jiya Sharma\Dropbox\PC\Desktop\fasion_rec\fashion_recommender_model_01.keras")

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) #resizing the image 
    x = image.img_to_array(img) #the image is converted into a numpy array
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = loaded_model.predict(x)
    return features.flatten()

# Function to recommend similar products
def recommend_similar_products(input_img_path, inventory_features, inventory_img_paths):
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_model.fit(inventory_features)
    
    input_features = extract_features(input_img_path)
    distances, indices = nn_model.kneighbors([input_features])
    similar_products = [inventory_img_paths[i] for i in indices[0]]
    return similar_products

# Function to load images from a directory
def load_inventory(inventory_dir, num_images=100):
    features_list = []
    img_paths = []
    for img_file in os.listdir(inventory_dir):
        img_path = os.path.join(inventory_dir, img_file)
        features = extract_features(img_path)
        features_list.append(features)
        img_paths.append(img_path)
    return np.array(features_list), img_paths

# Streamlit UI
st.title("Fashion Recommender System")
st.write("Enter the path of the image to find similar fashion products.")

# Input image path
input_img_path = st.text_input("Enter image path:")

if input_img_path:
    # Display input image
    st.subheader("Input Image:")
    input_img = image.load_img(input_img_path, target_size=(224, 224))
    st.image(input_img, caption='Input Image', width=90)

    # Load inventory features and image paths
    inventory_dir = r"C:\Users\Jiya Sharma\Dropbox\PC\Desktop\fasion_rec\valid"
    inventory_features, inventory_img_paths = load_inventory(inventory_dir)

    # Recommend similar products
    similar_products = recommend_similar_products(input_img_path, inventory_features, inventory_img_paths)

    # Display similar products
    st.subheader("Recommended similar products:")
    for i, img_path in enumerate(similar_products):
        st.write(f"Similar Product {i+1}")
        img = mpimg.imread(img_path)
        st.image(img, caption=f"Similar Product {i+1}", width=90)

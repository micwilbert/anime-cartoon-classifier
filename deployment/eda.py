import os
import zipfile
import shutil
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# Path dataset
archive_path = "./archive.zip"
dataset_path = "./Training Data"
anime_path = os.path.join(dataset_path, "Anime")
cartoon_path = os.path.join(dataset_path, "Cartoon")

# Function to extract and organize dataset
def extract_and_organize_dataset():
    if not os.path.exists(dataset_path):
        st.write("Extracting dataset...")
        if not os.path.exists(archive_path):
            st.error(f"Dataset file not found: {archive_path}")
            return
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.write(f"Dataset extracted to: {dataset_path}")

    st.write("Reorganizing dataset...")
    for category_path in [anime_path, cartoon_path]:
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    destination_path = os.path.join(category_path, file)
                    if os.path.exists(destination_path):
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(destination_path):
                            destination_path = os.path.join(category_path, f"{base}_{counter}{ext}")
                            counter += 1
                    shutil.move(file_path, destination_path)
                os.rmdir(subfolder_path)
    st.write("Dataset reorganized.")

# Function to get data distribution
def get_data_distribution():
    anime_images = len(os.listdir(anime_path))
    cartoon_images = len(os.listdir(cartoon_path))
    return {"Anime": anime_images, "Cartoon": cartoon_images}

# Function to get image dimensions
def get_image_dimensions(folder_path):
    dimensions = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        dimensions.append(img.size)
    return dimensions

# Function to show sample images
def show_sample_images(folder_path, category):
    sample_files = os.listdir(folder_path)[:5]
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for ax, img_file in zip(axes, sample_files):
        img = Image.open(os.path.join(folder_path, img_file))
        ax.imshow(img)
        ax.set_title(category)
        ax.axis('off')
    st.pyplot(fig)

# Display EDA page
def display_eda():
    # Title and author
    st.title("Exploratory Data Analysis: Anime vs Cartoon Classifier")
    st.markdown("**Created by:** Michael Wilbert Puradisastra")
    st.markdown("---")

    # Definition and differences
    st.header("What Are Anime and Cartoons?")
    st.markdown("""
    - **Anime:** Anime refers to Japanese animated productions that are characterized by detailed artwork, imaginative themes, and complex narratives. ([Source](https://www.nfi.edu/what-is-anime/))
    - **Cartoons:** Cartoons, primarily originating from Western cultures, are known for their exaggerated features, comedic tones, and often simpler designs. ([Source](https://www.merriam-webster.com/dictionary/cartoon))
    """)

    st.header("Why Know the Difference?")
    st.markdown("""
    - **Audience:** Anime often caters to a broader age range, with genres targeting children, teens, and adults. Cartoons primarily focus on younger audiences, though exceptions exist (e.g., "Rick and Morty").
    - **Market:** The anime industry generates billions globally, led by merchandise, conventions, and streaming services like Crunchyroll and Funimation. Cartoons dominate Western streaming platforms like Disney+ and Cartoon Network.
    - **Artistic Style:** Anime emphasizes realistic details, vibrant colors, and unique storytelling techniques. Cartoons often feature simple, vibrant designs with a focus on humor.
    """)

    st.image(
        "naruto.jpg", 
        caption="Example of Anime (Naruto Uzumaki)", 
        use_container_width=True
    )
    st.image(
        "mickey.jpg", 
        caption="Example of Cartoon (Mickey Mouse)", 
        use_container_width=True
    )

    # App objectives
    st.header("Objectives of This App")
    st.markdown("""
    - **Education:** Highlight the distinctions between anime and cartoons.
    - **Engagement:** Provide an interactive tool for image classification.
    - **Insight:** Showcase how machine learning models can identify visual styles and characteristics.
    """)

    # Extract and organize dataset
    extract_and_organize_dataset()

    # Data distribution
    st.header("Data Distribution")
    distribution = get_data_distribution()
    st.bar_chart(data=distribution)

    # Image dimensions
    st.header("Average Image Dimensions")
    anime_sizes = get_image_dimensions(anime_path)
    cartoon_sizes = get_image_dimensions(cartoon_path)
    anime_mean = np.mean(anime_sizes, axis=0)
    cartoon_mean = np.mean(cartoon_sizes, axis=0)
    st.write(f"Average dimensions for Anime: {anime_mean}")
    st.write(f"Average dimensions for Cartoon: {cartoon_mean}")

    # Sample images
    st.header("Sample Images")
    st.subheader("Anime")
    show_sample_images(anime_path, "Anime")
    st.subheader("Cartoon")
    show_sample_images(cartoon_path, "Cartoon")

    # Insights and references
    st.header("Insights")
    st.markdown("""
    - Anime markets to diverse age groups and interests, often exploring deeper, more mature themes compared to cartoons.
    - Cartoons are globally recognized for their humor and child-friendly content, making them a staple of family entertainment.
    - Both forms of animation contribute significantly to their respective cultural identities and global economies.
    """)
    st.markdown("---")
    st.markdown("### References")
    st.markdown("""
    - [NFI: What is Anime?](https://www.nfi.edu/what-is-anime/)
    - [Merriam-Webster: Cartoon](https://www.merriam-webster.com/dictionary/cartoon)
    """)

import streamlit as st
from eda import display_eda
from prediction import predict_image

# Set page configuration
st.set_page_config(
    page_title="Anime vs Cartoon Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Sidebar customization
st.sidebar.title("🎨 Anime vs Cartoon App")
st.sidebar.subheader("Navigate:")
page = st.sidebar.radio("Choose a Page:", ["EDA", "Prediction"])

# Title banner
st.markdown("<h1 class='stTitle'>Anime vs Cartoon Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")  # Divider line

# Page rendering
if page == "EDA":
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    display_eda()

elif page == "Prediction":
    st.markdown("<h2 style='text-align: center; color: #FF5722;'>Upload your image here (only Anime or Cartoon & .jpg or .png)</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📤 Upload your image here:", type=["jpg", "png"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_column_width=True,
            clamp=True,
        )

        # Perform prediction
        result = predict_image(uploaded_file)

        # Display prediction results
        st.markdown(f"<h3 style='color: #4CAF50;'>Predicted Class: <b>{result['class']}</b></h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>Confidence: <b>{result['confidence']:.2f}</b></h4>", unsafe_allow_html=True)

import streamlit as st
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Load the model and feature extractor
extractor = AutoFeatureExtractor.from_pretrained("yangy50/garbage-classification")
model = AutoModelForImageClassification.from_pretrained("yangy50/garbage-classification")

# Define the function to classify the image
def classify_image(image):
    # Open the uploaded image
    img = Image.open(image)
    # Preprocess the image
    inputs = extractor(images=img, return_tensors="pt")
    # Classify the image
    outputs = model(**inputs)
    predictions = outputs.logits[0]
    # Get the labels and scores
    labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    scores = predictions.softmax(dim=0)
    # Return the labels and scores
    return dict(zip(labels, scores.tolist()))

# Define the Streamlit app
def app():
    # Set the app title
    st.title("Garbage Detector")
    # Upload an image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # Check if an image file was uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file)
        # Classify the image
        results = classify_image(uploaded_file)
        # Print the results
        st.write("Detected items:")
        has_garbage = False
        for label, score in results.items():
            if score > 0.1:  # Show only items with score > 0.1
                st.write(f"- {label}: {score:.2f}")
                if label != "trash" and score > 0.41:
                    has_garbage = True
        # Print the final result
        if has_garbage:
            st.write("Garbage detected")
            # Plot the scores in a horizontal bar chart
            st.bar_chart(results)
        else:
            st.write("No garbage detected")

# Run the app
if __name__ == "__main__":
    app()


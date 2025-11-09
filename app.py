import streamlit as st
from PIL import Image
from filters import apply_filter_sequential, apply_filter_parallel
import matplotlib.pyplot as plt
import os

st.title("Parallel Image Processing")

st.write("Hamza Ahmed Khan (2212341)")
st.write("Sibtain Ahmed (2212271)")
st.write("Munesh Kumar (2212260)")
st.write("Pawan Mahesh (2212263)")
st.write("Ahmed Ali Khokhar (2212243)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
filter_type = st.selectbox("Select a filter", ["Grayscale", "Blur", "Edge Detection"])
num_threads = st.slider("Threads", 1, 8, 4)

if not os.path.exists("output"):
    os.makedirs("output")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original Image", width=400)

    # Sequential
    seq_img, seq_time = apply_filter_sequential(image, filter_type)
    seq_path = f"output/seq_{filter_type}.jpg"
    seq_img.save(seq_path)

    # Parallel
    par_img, par_time = apply_filter_parallel(image, filter_type, num_threads)
    par_path = f"output/par_{filter_type}.jpg"
    par_img.save(par_path)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sequential Result")
        st.image(seq_img, width=400)
        st.write(f"Time: {seq_time:.4f} seconds")

    with col2:
        st.subheader("Parallel Result")
        st.image(par_img, width=400)
        st.write(f"Time: {par_time:.4f} seconds")

    # Chart
    st.write("---")
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.bar(["Sequential", "Parallel"], [seq_time, par_time], color=["gray", "skyblue"])
    ax.set_ylabel("Time (s)")
    st.pyplot(fig)

    if par_time > 0:
        st.write(f"**Speedup:** {seq_time / par_time:.2f}x faster")
else:
    st.info("Upload an image to begin.")

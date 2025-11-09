# Parallel Image Processing

This project demonstrates how **parallel processing** can speed up image filtering tasks using **Python threads**.  
It applies filters such as **Grayscale**, **Blur**, and **Edge Detection** on songle image — both sequentially and in parallel — and compares their performance visually.

---

## 🧠 Project Description

Image processing tasks like applying filters can often be done independently on different parts of an image.  
This project splits an image into chunks and processes them simultaneously using multiple threads to achieve faster results.

The app uses **Streamlit** for the graphical interface so users can:
- Upload an image
- Choose a filter type
- Adjust the number of threads
- Compare sequential vs parallel processing time

---

## ⚙️ Features
- Sequential and parallel implementations of image filters  
- Adjustable number of threads  
- Visual side-by-side comparison of results  
- Performance graph (bar chart) for timing comparison  
- Simple Streamlit-based UI  

---

## 🛠️ Tools and Libraries
- **Python 3**
- **Streamlit** (for the UI)
- **Pillow (PIL)** (for image processing)
- **Matplotlib** (for timing graph)

---

## 📦 Installation

### 1️⃣ Clone or download the repository:
```bash
git clone https://github.com/yourusername/Parallel_Image_Processing.git
cd Parallel_Image_Processing

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main app background - pure black */
    .stApp {
        background-color: #000000 !important;
    }
    
    /* Text colors */
    h1, h2, h3, p, .stMarkdown {
        color: white !important;
    }
    
    /* Success message styling */
    .stAlert {
        background-color: #1a1a1a !important;
        color: #4CAF50 !important;
        border: 1px solid #333 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #4CAF50 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888 !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #4CAF50 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: white !important;
        background-color: #1a1a1a !important;
    }
    
    .streamlit-expanderContent {
        background-color: #0a0a0a !important;
        border: 1px solid #333 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #4CAF50 !important;
    }
    
    .stButton > button:hover {
        background-color: #4CAF50 !important;
        color: black !important;
    }
    
    /* Slider styling */
    .stSlider {
        color: white !important;
    }
    
    /* Canvas container */
    .canvas-container {
        background-color: #000000;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #333;
    }
    
    /* Big digit display */
    .big-digit {
        font-size: 10rem !important;
        text-align: center;
        color: #4CAF50;
        margin: 0;
        padding: 0;
        line-height: 1;
        text-shadow: 0 0 20px rgba(76, 175, 80, 0.3);
    }
    
    /* Confidence text */
    .confidence-text {
        text-align: center;
        color: #888;
        font-size: 1.5rem;
        margin-top: 0;
    }
    
    /* Stats boxes */
    .stat-box {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    .stat-number {
        color: #4CAF50;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .stat-label {
        color: #888;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_best.h5')

# Header
st.markdown("<h1 style='text-align: center; color: white;'>🔢 MNIST Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>High Accuracy Deep Learning Model</p>", unsafe_allow_html=True)

# Stats row
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown('<div class="stat-box"><div class="stat-number">99.2%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
with col_s2:
    st.markdown('<div class="stat-box"><div class="stat-number">60K</div><div class="stat-label">Images</div></div>', unsafe_allow_html=True)
with col_s3:
    st.markdown('<div class="stat-box"><div class="stat-number">10</div><div class="stat-label">Digits</div></div>', unsafe_allow_html=True)
with col_s4:
    st.markdown('<div class="stat-box"><div class="stat-number"><0.1s</div><div class="stat-label">Speed</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Load model
with st.spinner("🎯 Loading 99.2% accuracy model..."):
    model = load_model()
st.success("✅ Model loaded successfully!")

# Main interface
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h3 style='color: white;'>✏️ Drawing Area</h3>", unsafe_allow_html=True)
    
    # Drawing controls
    col_brush, col_clear = st.columns([3, 1])
    with col_brush:
        stroke_size = st.slider("Brush Size", 10, 40, 25, key="brush")
    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)
        clear = st.button("🗑️ Clear", use_container_width=True)
        if clear:
            st.rerun()
    
    # Canvas - larger size
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_size,
        stroke_color="white",
        background_color="#000000",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips
    with st.expander("💡 Drawing Tips"):
        st.markdown("""
        - Draw **large and centered**
        - Use **thick strokes** (20-25 brush size)
        - Make digits **fill most of the box**
        - Clear before drawing new digit
        """)

with col2:
    st.markdown("<h3 style='color: white;'>🔍 Recognition Result</h3>", unsafe_allow_html=True)
    
    if canvas_result.image_data is not None:
        # Process image
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        
        # Invert if needed (MNIST expects white digit on black)
        if img_array.mean() > 0.5:
            img_array = 1 - img_array
        
        if img_array.max() > 0.1:
            # Prepare for model (CNN expects 28x28x1)
            img_input = img_array.reshape(1, 28, 28, 1)
            
            # Predict
            predictions = model.predict(img_input, verbose=0)[0]
            digit = np.argmax(predictions)
            confidence = predictions[digit] * 100
            
            # Display digit
            st.markdown(f'<p class="big-digit">{digit}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence-text">Confidence: {confidence:.1f}%</p>', unsafe_allow_html=True)
            
            # Show what the AI sees
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("**Your drawing:**")
                st.image(img.resize((100, 100)), width=100)
            with col_img2:
                st.markdown("**What AI sees:**")
                st.image(img_array.reshape(28, 28), width=100, clamp=True)
            
            # Top 3 predictions
            st.markdown("---")
            st.markdown("### 🏆 Top 3 Predictions")
            top3_idx = np.argsort(predictions)[-3:][::-1]
            top3_probs = predictions[top3_idx] * 100
            
            cols = st.columns(3)
            for i, (idx, prob) in enumerate(zip(top3_idx, top3_probs)):
                with cols[i]:
                    st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{idx}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>{prob:.1f}%</p>", unsafe_allow_html=True)
                    st.progress(int(prob))
            
            # All probabilities
            with st.expander("📊 All Digit Probabilities"):
                for i in range(10):
                    prob = predictions[i] * 100
                    if i == digit:
                        st.markdown(f"**Digit {i}:** {prob:.1f}% ⭐")
                    else:
                        st.markdown(f"Digit {i}: {prob:.1f}%")
                    st.progress(int(prob))
        else:
            st.info("👆 Draw a digit in the canvas to see prediction")
            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #444;'>Powered by TensorFlow & Streamlit | 99.2% Accuracy on MNIST</p>", unsafe_allow_html=True)

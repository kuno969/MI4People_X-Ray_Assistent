from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st

import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_tensor, to_pil_image

from torchcam import methods
from torchcam.methods._utils import locate_candidate_layer
from torchcam.utils import overlay_mask

from PIL import Image

from src.model_library import *

CAM_METHODS = ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM", "LayerCAM"]
MODEL_SOURCES = ["XRV"]

def main():
    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("Chest X-ray Investigation")
    # For newline
    st.write("\n")

    # Layout
    text1 = st.empty()
    col1, col2 = st.columns(2)

    # Sidebar
    # File selection
    st.sidebar.title("Input selection")

    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["png", "jpeg", "jpg"])

    # Model selection
    st.sidebar.title("Setup")

    model_source = st.sidebar.selectbox(
        "Classification model source",
        MODEL_SOURCES,
        help="Supported models from Torchxrayvision",
    )

    if "model_choice_disabled" not in st.session_state:
        st.session_state.model_choice_disabled = True

    model_lib = AbstractModelLibrary()
    if model_source is not None:
        if model_source=="XRV":
            model_lib = XRVModelLibrary()
            st.session_state.model_choice_disabled = False
    
    model_choice = st.sidebar.selectbox(
        "Model choice",
        model_lib.CHOICES,
        disabled = st.session_state.model_choice_disabled,
        )

    model = None
    target_layer = None
    if st.session_state is not None:
        with st.spinner("Loading model..."):
            model = model_lib.get_model(model_choice).eval()
            target_layer = model_lib.get_target_layer(model)

    cam_method = st.sidebar.selectbox(
        "CAM method",
        CAM_METHODS,
        help="The way your class activation map will be computed",
    )

    cam_extractor = None
    if cam_method is not None:
        cam_extractor = methods.__dict__[cam_method](model, target_layer)

    class_choices = []
    if model is not None:
        class_choices = [f"{idx + 1} - {class_name}" for idx, class_name in enumerate(model_lib.LABELS)]
    class_selection = st.sidebar.selectbox("Class selection", ["No selection"] + class_choices)

    # For newline
    st.sidebar.write("\n")

    if st.sidebar.button("Diagnose"):
        if uploaded_file is None:
            st.sidebar.error("Please upload an image first")
        else:
            # Load image
            img = Image.open(BytesIO(uploaded_file.read()), mode="r").convert("RGB")
            img_tensor = to_tensor(img)

            # Show imputs
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.axis("off")
                ax1.imshow(to_pil_image(img_tensor))
                st.header("Input X-ray image")
                st.pyplot(fig1)

            if model is None:
                st.sidebar.error("Please select a classification model")
            else:
                with st.spinner("Analyzing..."):
                    # Preprocess image
                    transformed_img, normalized_img = model_lib.preprocess(img_tensor)

                    # Forward
                    if torch.cuda.is_available():
                        model = model.cuda()
                        normalized_img = normalized_img.cuda()

                    out = model(normalized_img.unsqueeze(0))

                    # Select the target class
                    class_idx = out.squeeze(0).argmax().item()
                    diagnosis_label = model_lib.LABELS[class_idx]

                    with text1:
                        st.write("Based on the inputs, the diagnosis is "+diagnosis_label)

                    class_label = diagnosis_label
                    if class_selection != "No selection":
                        class_label = class_selection.split("-")[-1].strip()
                        class_idx = model_lib.LABELS.index(class_label)

                    activation_maps = cam_extractor(class_idx, out)

                    # Show results
                    result = overlay_mask(to_pil_image(transformed_img.expand(3,-1,-1)), \
                                        to_pil_image(activation_maps[0].squeeze(0), mode='F'), \
                                        alpha=0.7)

                    with col2:
                        fig2, ax2 = plt.subplots()
                        ax2.axis("off")
                        ax2.imshow(result)
                        st.header("Overlay : "+class_label)
                        st.pyplot(fig2)

if __name__ == "__main__":
    main()

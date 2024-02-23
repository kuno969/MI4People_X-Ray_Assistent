from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from torchcam import methods
from torchcam.utils import overlay_mask

from PIL import Image

from src.model_library import XRVModelLibrary, AbstractModelLibrary
from src.feedback_utils import Feedback
from src.db_interface import MetadataStore, get_image_from_azure, setup_container_client


# All supported CAM
# CAM_METHODS = ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM", "LayerCAM"]

# Supported CAMs for multiple target layers
CAM_METHODS = [
    "GradCAM",
    "GradCAMpp",
    "SmoothGradCAMpp",
    "XGradCAM",
    "LayerCAM",
    "ScoreCAM",
    "SSCAM",
    "ISCAM",
]
MODEL_SOURCES = ["XRV"]
NUM_RESULTS = 5
N_IMAGES = 10


def main():
    feedback = Feedback()

    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("Chest X-ray Investigation")
    # For newline
    st.write("\n")

    # Layout
    input_col, general_feedback_col = st.columns(2)

    # Sidebar
    # File selection
    st.sidebar.title("Input selection")

    # Enter access key
    account_key = st.sidebar.text_input("account_key", value=None)

    # Choose image
    metadata = MetadataStore()

    img = None

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if account_key is not None:
        container_client = setup_container_client(account_key)

        metadata.read_from_azure(container_client)

        filter_label = st.sidebar.selectbox(
            "Filter Label", metadata.get_unique_labels()
        )

        img = None
        if filter_label is not None:
            image_filenames = metadata.get_random_image_filenames(filter_label, N_IMAGES)

        
        # if filter_label is not None:
        #     image_filename = st.sidebar.selectbox(
        #         "Image Filename", metadata.get_image_filenames(filter_label)
        #     )

        #     if image_filename is not None:
        #         

    # # Upload image
    # uploaded_file = st.sidebar.file_uploader(
    #     "Upload files", type=["png", "jpeg", "jpg"]
    # )

    # if uploaded_file is not None:
    #     # Load image
    #     img = Image.open(BytesIO(uploaded_file.read()), mode="r").convert("RGB")

    # Model selection
    st.sidebar.title("Setup")

    model_source = st.sidebar.selectbox(
        "Classification model source",
        MODEL_SOURCES,
        help="Supported models from Torchxrayvision",
    )

    model_lib = AbstractModelLibrary()
    if model_source is not None:
        if model_source == "XRV":
            model_lib = XRVModelLibrary()

    model_choice = st.sidebar.selectbox(
        "Model choice",
        model_lib.CHOICES,
    )

    model = None
    if model_source is not None:
        with st.spinner("Loading model..."):
            model = model_lib.get_model(model_choice)

    for p in model.parameters():
        p.requires_grad_(False)

    # CAM selection
    cam_method = st.sidebar.selectbox(
        "CAM method",
        CAM_METHODS,
        help="The way your class activation map will be computed",
    )

    # For newline
    st.sidebar.write("\n")

    feedback_submitted = None
    feedback_comment = [None for i in range(NUM_RESULTS)]
    feedback_ok = [False for i in range(NUM_RESULTS)]

    if st.sidebar.button("Diagnose Next"):

        image_filename = image_filenames[st.session_state.current_index]

        st.write("Store label : " + metadata.get_full_label(image_filename))

        blob_data = get_image_from_azure(container_client, image_filename)
        img = Image.open(BytesIO(blob_data.read()), mode="r").convert("RGB")

        if img is not None:
            img_tensor = to_tensor(img)

            # Show imputs
            with input_col:
                fig1, ax1 = plt.subplots()
                ax1.axis("off")
                ax1.imshow(to_pil_image(img_tensor))
                st.header("Input X-ray image")
                st.pyplot(fig1)
        if img is None:
            st.sidebar.error("Please upload an image first")
        else:
            if model is None:
                st.sidebar.error("Please select a classification model")
            elif cam_method is None:
                st.sidebar.error("Please select CAM method.")
            else:
                with st.spinner("Analyzing..."):
                    result_cols = [None for i in range(NUM_RESULTS)]
                    feedback_cols = [None for i in range(NUM_RESULTS)]

                    # Preprocess image
                    transformed_img, rescaled_img = model_lib.preprocess(img_tensor)

                    rescaled_img.requires_grad_(True)

                    if torch.cuda.is_available():
                        model = model.cuda()
                        rescaled_img = rescaled_img.cuda()

                    # Show results
                    with st.form("form"):
                        for i in range(NUM_RESULTS):
                            result_cols[i], feedback_cols[i] = (
                                st.container(),
                                st.container(),
                            )
                            st.divider()

                        # Somehow this returns the same activation map disregarding the class per method
                        # Why is this happening?

                        for i in range(NUM_RESULTS):

                            cam_extractors = []
                            # Initialize CAM

                            for cam_method in CAM_METHODS:
                                cam_extractor_method = methods.__dict__[cam_method](
                                    model,
                                    target_layer=model_lib.TARGET_LAYER,
                                    enable_hooks=False,
                                )
                                cam_extractors.append(cam_extractor_method)

                            fig2, ax2 = plt.subplots(
                                ncols=len(cam_extractors), figsize=(20, 5)
                            )

                            for idx, cam_extractor in enumerate(cam_extractors):
                                # Forward
                                cam_extractor._hooks_enabled = True

                                model.zero_grad()
                                out = model(rescaled_img.unsqueeze(0))

                                print(CAM_METHODS[idx])
                                print(out.requires_grad)

                                # Select the target class
                                class_ids = torch.topk(
                                    out.squeeze(0), NUM_RESULTS
                                ).indices

                                activation_maps = cam_extractor(
                                    class_idx=class_ids[i].item(), scores=out
                                )

                                # Fuse the CAMs if there are several
                                activation_map = (
                                    activation_maps[0]
                                    if len(activation_maps) == 1
                                    else cam_extractor.fuse_cams(activation_maps)
                                )

                                cam_extractor.remove_hooks()
                                cam_extractor._hooks_enabled = False

                                result = overlay_mask(
                                    to_pil_image(transformed_img.expand(3, -1, -1)),
                                    to_pil_image(activation_map.squeeze(0), mode="F"),
                                    alpha=0.7,
                                )

                                # plot result on respective axis
                                ax2[idx].set_title(CAM_METHODS[idx])
                                ax2[idx].axis("off")
                                ax2[idx].imshow(result)
                                # ax2.axis("off")
                                # ax2.imshow(result)
                                

                            with result_cols[i]:
                                class_label = model_lib.LABELS[class_ids[i].item()]
                                st.header("Result %d : %s" % (i + 1, class_label))
                                st.pyplot(fig2)

                            with feedback_cols[i]:
                                st.write("")
                                probability = out.squeeze(0)[class_ids[i]].item() * 100
                                st.write("**Probability** : %0.2f %%" % (probability,))
                                feedback_ok[i] = st.checkbox(
                                    "Confirm", key="Confirm-%d" % (i + 1)
                                )
                                feedback_comment[i] = st.text_area(
                                    "Comment", key="Comment-%d" % (i + 1)
                                )

                        feedback_submitted = st.form_submit_button("Submit")

                    if feedback_submitted:
                        feedback = Feedback()
                        for i in range(NUM_RESULTS):
                            feedback.insert(
                                "result%d_confirm" % (i), str(feedback_ok[i])
                            )
                            feedback.insert(
                                "result%d_comment" % (i), str(feedback_comment[i])
                            )

                        with general_feedback_col:
                            st.subheader("Feedback summary")
                            st.write(str(feedback.get_data()))

        if st.session_state.current_index < N_IMAGES:
            st.session_state.current_index += 1
        else:
            st.write("No more images to diagnose")

if __name__ == "__main__":
    main()

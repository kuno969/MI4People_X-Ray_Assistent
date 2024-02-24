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
    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("Chest X-ray Investigation")
    # For newline
    st.write("\n")

    # Sidebar
    # File selection
    st.sidebar.title("Input selection")

    # Enter access key
    account_key = st.sidebar.text_input("account_key", value=None)

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if "images" not in st.session_state:

        metadata = MetadataStore()

        if account_key is not None:
            container_client = setup_container_client(account_key)

            metadata.read_from_azure(container_client)

            filter_label = st.sidebar.selectbox(
                "Filter Label", metadata.get_unique_labels()
            )

            img = None
            if filter_label is not None:
                image_filenames = metadata.get_random_image_filenames(
                    filter_label, N_IMAGES
                )

            images = []
            for image_filename in image_filenames:
                img = {
                    "filename": image_filename,
                    "label": metadata.get_full_label(image_filename),
                }
                images.append(img)

            st.session_state["images"] = images
            st.session_state["container_client"] = container_client

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
    # cam_method = st.sidebar.selectbox(
    #     "CAM method",
    #     CAM_METHODS,
    #     help="The way your class activation map will be computed",
    # )

    # For newline
    st.sidebar.write("\n")

    # st.sidebar.multiselect(
    #     "CAM choices",
    #     CAM_METHODS,
    #     help="The way your class activation map will be computed",
    #     max_selections=3,
    #     key="cam_choices",
    #     default=["GradCAM", "GradCAMpp"]
    # )

    # cam_choices = st.session_state["cam_choices"]

    cam_choices = CAM_METHODS

    if "images" in st.session_state:
        image = st.session_state.images[st.session_state.current_index]

        diagnose(image, model, cam_choices, model_lib)


def diagnose(
    img: dict, model: torch.nn.Module, cam_choices: list, model_lib: XRVModelLibrary
):
    input_col, general_feedback_col = st.columns(2)

    blob_data = get_image_from_azure(st.session_state["container_client"], img["filename"])
    img_data = Image.open(BytesIO(blob_data.read()), mode="r").convert("RGB")

    img_tensor = to_tensor(img_data)

    with input_col:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.axis("off")
        ax1.imshow(to_pil_image(img_tensor))
        st.header("Input X-ray image")
        st.pyplot(fig1)

    st.write("Store label : " + img["label"])

    feedback = Feedback()

    feedback_comment = [None for i in range(NUM_RESULTS)]
    feedback_ok = [False for i in range(NUM_RESULTS)]

    if model is None:
        st.sidebar.error("Please select a classification model")
    # elif cam_method is None:
    #     st.sidebar.error("Please select CAM method.")
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

                for i in range(NUM_RESULTS):
                    cam_extractors = []
                    # Initialize CAM

                    for cam_method in cam_choices:
                        cam_extractor_method = methods.__dict__[cam_method](
                            model,
                            target_layer=model_lib.TARGET_LAYER,
                            enable_hooks=False,
                        )
                        cam_extractors.append(cam_extractor_method)

                    fig2, ax2 = plt.subplots(ncols=len(cam_extractors), figsize=(20, 5))

                    for idx, cam_extractor in enumerate(cam_extractors):
                        # Forward
                        cam_extractor._hooks_enabled = True

                        model.zero_grad()
                        out = model(rescaled_img.unsqueeze(0))

                        # Select the target class
                        class_ids = torch.topk(out.squeeze(0), NUM_RESULTS).indices

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

                st.form_submit_button(
                    "Next Patient",
                    on_click=lambda: give_feedback(
                        feedback, feedback_ok, feedback_comment
                    ),
                )


def give_feedback(feedback: Feedback, feedback_ok: list, feedback_comment: list):

    for i in range(NUM_RESULTS):
        feedback_dict = {
            "confirm": str(feedback_ok[i]),
            "comment": str(feedback_comment[i]),
        }

        feedback.insert(f"result{i}", feedback_dict)

    if st.session_state.current_index < N_IMAGES:
        st.session_state.current_index += 1
    else:
        st.write("No more images to diagnose")

    print(feedback.get_data())


if __name__ == "__main__":
    main()

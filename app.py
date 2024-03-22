from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

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
NUM_RESULTS = 3
N_IMAGES = 10
RESULTS_PER_ROW = 3


def main():
    # Wide mode
    st.set_page_config(page_title="Chest X-ray Investigation", page_icon="🚑", layout="centered", initial_sidebar_state="expanded")

    # Designing the interface
    st.title("Chest X-ray Investigation")

    # Sidebar
    # File selection
    st.sidebar.title("Input selection")

    st.sidebar.write("This is a tool to evaluate AI predictions on chest X-ray images. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla nec purus feugiat, molestie ipsum et, consequat nibh. Ut sit amet odio eu est aliquet euismod a ante")

    # Enter access key
    account_key = st.sidebar.text_input("account_key", value=None)

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if "images" not in st.session_state:

        metadata = MetadataStore()

        if account_key is not None:
            container_client = setup_container_client(account_key)

            metadata.read_from_azure(container_client)

            # TODO: Add possibility to filter by label or not (maybe through checkboxes or a multiselectbox)
            # We need metadataStore for that, so load beforehand
            # Also we need to reload image_filenames when we change the filter
            # Is there a on_change event for selectbox?

            # filter_label = st.sidebar.selectbox(
            #     "Filter Label", metadata.get_unique_labels()
            # )

            # img = None
            # if filter_label is not None:
            #     image_filenames = metadata.get_random_image_filenames(
            #         N_IMAGES, filter_label
            #     )

            image_filenames = metadata.get_random_image_filenames(N_IMAGES)

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
    st.sidebar.title("How-To")

    # model_source = st.sidebar.selectbox(
    #     "Classification model source",
    #     MODEL_SOURCES,
    #     help="Supported models from Torchxrayvision",
    # )

    # model_lib = AbstractModelLibrary()
    # if model_source is not None:
    #     if model_source == "XRV":
    #         model_lib = XRVModelLibrary()

    # model_choice = st.sidebar.selectbox(
    #     "Model choice",
    #     model_lib.CHOICES,
    # )

    model_source = "XRV"
    model_lib = XRVModelLibrary()
    model_choice = "densenet121-res224-all"

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
    # input_col = st.columns(1)

    blob_data = get_image_from_azure(st.session_state["container_client"], img["filename"])
    img_data = Image.open(BytesIO(blob_data.read()), mode="r").convert("RGB")

    img_tensor = to_tensor(img_data)

    # with input_col:
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    ax1.axis("off")
    ax1.imshow(to_pil_image(img_tensor))
    st.header("Input X-ray image")
    st.pyplot(fig1)

    st.write(f"Store label: {img['label']}")

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

                result_tabs = st.tabs([f"Result {i+1}" for i in range(NUM_RESULTS)])

                for i in range(NUM_RESULTS):
                    with result_tabs[i]:
                        result_cols[i], feedback_cols[i] = (
                            st.container(),
                            st.container(),
                        )

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

                    # fig2 = plt.figure()
                    # fig2.tight_layout()
                    # plt.rcParams['figure.facecolor'] = st.get_option("theme.backgroundColor")
                        
                    figs = []

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
                            # to_pil_image(transformed_img.expand(3, -1, -1)),
                            img_data,
                            to_pil_image(activation_map.squeeze(0), mode="F"),
                            alpha=0.7,
                        )

                        fig = px.imshow(result)
                        fig.update_xaxes(visible=False)
                        fig.update_yaxes(visible=False)
                        figs.append(fig)

                        # row = idx // RESULTS_PER_ROW
                        # col = idx % RESULTS_PER_ROW

                        # ax2 = fig2.add_axes([0.1 + col * 0.3, 0.1 + row * 0.3, 0.25, 0.25])
                        # ax2.set_title(cam_choices[idx])
                        # ax2.axis("off")
                        # ax2.imshow(result)

                    with result_cols[i]:
                        class_label = model_lib.LABELS[class_ids[i].item()]
                        st.header(f"Finding: {class_label}")
                        tabs = st.tabs(cam_choices)
                        for idx, tab in enumerate(tabs):
                            with tab:
                                st.plotly_chart(figs[idx], use_container_width=True, theme="streamlit")
                        # st.pyplot(fig2)
                        # components.html(mpld3.fig_to_html(fig2), height=500, width=500)

                    with feedback_cols[i]:
                        probability = out.squeeze(0)[class_ids[i]].item() * 100
                        st.write(f"**Probability** : {probability:.2f}%")
                        feedback_ok[i] = st.checkbox("Confirm Finding", key=f"confirm{i}")
                        feedback_comment[i] = st.text_area("Comment", key=f"comment{i}")

                st.form_submit_button(
                    "Next Patient",
                    use_container_width=True,
                    type="primary",
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

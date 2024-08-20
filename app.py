from io import BytesIO
import json
import os
import requests

import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots

from PIL import Image

from src.feedback_utils import Feedback
from src.db_interface import MetadataStore, get_image_from_azure, setup_container_client, write_data_to_azure_blob


# All supported CAM
# CAM_METHODS = ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM", "LayerCAM"]

# Supported CAMs for multiple target layers
CAM_METHODS = [
    "gradcam",
    "gradcampp" ,
    # "isc" ,
    "xgradcam" ,
    "layercam",
]
MODEL_SOURCES = ["XRV"]
NUM_RESULTS = 3
N_IMAGES = 10
RESULTS_PER_ROW = 2

FUNCTION_URL = os.environ["FUNCTION_URL"] + "?code=" + os.environ["FUNCTION_KEY"]


def main():
    st.set_page_config(page_title="Chest X-ray Investigation", page_icon="🚑", layout="wide", initial_sidebar_state="collapsed")

    st.sidebar.title("How-To")

    st.sidebar.write("""
        Welcome to X-Ray Insight, an innovative platform designed to democratize access to advanced medical technologies through the power of artificial intelligence. Our platform is dedicated to MI4People's commitment to "AI for Good," aiming to make cutting-edge medical diagnostics accessible worldwide, especially in developing countries where such advancements can significantly transform healthcare.
        Purpose of This Website:
        This website is designed for the evaluation of our diagnostic model. Your insights and feedback are invaluable in enhancing the accuracy and effectiveness of this tool.
        How to Provide Feedback:
        1. Confirming Diagnoses: Below each X-ray image, you'll see the actual medical indication. On the right side of the screen is the predicted indication generated by our AI. Please confirm the accuracy of the prediction by ticking the checkbox if it matches the true indication.
        2. Selecting the Best CAM Method: Examine the set of 4 images displaying different CAM (Class Activation Mapping) methods. Each image uses a heat map to highlight critical areas relevant to the diagnosis. If the indication fits, select the CAM method that best highlights the critical areas by its red colouring. Otherwise, provide a comment or go to the next step.
        3. Additional Observations: If you notice any other significant features, please enter them into the comment box.
        4. Navigation: Click on "Next" to either view the next best predicted indication with associated heatmaps for the same X-ray, or to move on to the next patient. The top 3 predicted indications are shown for each patient.

        Note: Each time you refresh the page, 20 randomly selected patients are displayed.
        
        Thank you for your support in improving healthcare through technology!
        
        MI4People 2024"""
    )

    account_key = os.environ["SAS_TOKEN"]

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if "images" not in st.session_state:

        metadata = MetadataStore()
        st.session_state["feedback"] = Feedback()
        

        if account_key is not None:
            container_client = setup_container_client(account_key)

            metadata.read_from_azure(container_client)

            image_filenames = metadata.get_random_image_filenames(N_IMAGES, "No Finding", inverse_label=True)

            images = []
            for image_filename in image_filenames:
                img = {
                    "filename": image_filename,
                    "label": metadata.get_full_label(image_filename),
                }
                images.append(img)

            st.session_state["images"] = images
            st.session_state["container_client"] = container_client
            st.session_state.num_result = 0

    st.sidebar.write("\n")

    cam_choices = CAM_METHODS

    if "images" in st.session_state and st.session_state.current_index < N_IMAGES:

        # if st.session_state.num_result == NUM_RESULTS:
        image = st.session_state.images[st.session_state.current_index]

        diagnose(image, cam_choices)
    else:
        st.write("No more images to diagnose")


def diagnose(
    img: dict, cam_choices: list
):
    # input_col = st.columns(1)

    input_col, result_col = st.columns([0.5, 0.5], gap="medium")

    blob_data = get_image_from_azure(st.session_state["container_client"], img["filename"]).read()
    img_data = Image.open(BytesIO(blob_data), mode="r").convert("RGB")

    with st.spinner("Analyzing..."):
        if st.session_state.num_result == 0:
            st.session_state["model_result"] = {}

            for cam_method in cam_choices:
                data = {
                    "method": cam_method,
                    "k": "5",
                }

                files = {
                    "image": BytesIO(blob_data)
                }

                response = requests.post(FUNCTION_URL, data=data, files=files)
                response = response.json()
                
                if "predictions" not in st.session_state["model_result"]:
                    st.session_state["model_result"]["predictions"] = response["predictions"]

                if "cam" not in st.session_state["model_result"]:
                    st.session_state["model_result"]["cam"] = response["cam"]

                st.session_state["model_result"]["cam"].update(response["cam"])
            
            print(st.session_state["model_result"])
                

    class_label = list(st.session_state["model_result"]["predictions"].keys())[st.session_state.num_result]
    probability = st.session_state["model_result"]["predictions"][class_label]

    cam_heatmaps = {}
    for cam_method in cam_choices:
        cam_heatmaps[cam_method] = get_image_from_azure(st.session_state["container_client"], st.session_state["model_result"]["cam"][cam_method][class_label], prefix="")

    cam_fig = draw_cam(cam_heatmaps, cam_choices)


    with input_col:
        with st.container(border=True):
            fig = plt.figure(figsize=(6, 6))
            fig.tight_layout()
            ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax1.axis("off")
            ax1.imshow(img_data)
            st.header("Input X-ray image")
            st.pyplot(fig)

            st.write(f"Store label: {img['label']}")

    with result_col:
            with st.form("form"):
                result_container = st.container()
                feedback_container = st.container()


                with result_container:
                    st.header(f"Finding: {class_label} ({(st.session_state.num_result + 1)}/{NUM_RESULTS})")
                    st.session_state["finding"] = class_label
                    st.plotly_chart(cam_fig, use_container_width=True, theme="streamlit")

                with feedback_container:

                    cols = st.columns(2)

                    with cols[0]:
                        st.session_state["probability"] = probability
                        st.write(f"**Probability** : {(probability * 100):.2f}%")
                        st.checkbox("Confirm Finding", key=f"confirm{st.session_state.num_result}")
                        st.text_area("Comment", key=f"comment{st.session_state.num_result}")
                    
                    with cols[1]:
                        st.selectbox(
                            "Best CAM method",
                            cam_choices,
                            index=None,
                            help="The best CAM method for this image",
                            key=f"best_cam_method{st.session_state.num_result}",
                            placeholder="Select the best CAM method",
                        )

                if st.session_state.num_result == NUM_RESULTS - 1:
                    submit_label = "Next Patient"
                else:
                    submit_label = "Next Result"

                st.form_submit_button(
                    submit_label,
                    use_container_width=True,
                    type="primary",
                    on_click=give_feedback,
                )


def activate_feedback(feedback: Feedback):

    if st.session_state[f"best_cam_method{st.session_state.num_result}"] is not None:
        st.session_state["submit_button"].disabled = False


def draw_cam(cam_heatmaps: dict, cam_choices) -> None:
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=cam_choices, horizontal_spacing=0.05, vertical_spacing=0.05)

    for idx, cam in enumerate(cam_choices):
        
        result = Image.open(BytesIO(cam_heatmaps[cam].read()), mode="r").convert("RGB")
        row = idx // RESULTS_PER_ROW
        col = idx % RESULTS_PER_ROW

        fig.add_trace(px.imshow(result).data[0], row=row + 1, col=col + 1)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    
    fig.update_layout(height=500, width=800, margin=dict(l=20, r=20, t=50, b=20))

    return fig


def give_feedback():

    selection_dict = {
        "confirm": st.session_state[f"confirm{st.session_state.num_result}"],
        "comment": st.session_state[f"comment{st.session_state.num_result}"],
        "best_cam_method": st.session_state[f"best_cam_method{st.session_state.num_result}"],
        "probability": st.session_state["probability"],
    }

    image_name = st.session_state.images[st.session_state.current_index]["filename"]

    feedback_dict = {
        "result": str(st.session_state.num_result) + "_" + st.session_state["finding"],
        "selection": selection_dict,
    }

    st.session_state["feedback"].insert(image_name, feedback_dict)

    if st.session_state.num_result == NUM_RESULTS - 1:
        st.session_state.current_index += 1
        st.session_state.num_result = 0
        feedback_json = json.dumps(dict(st.session_state["feedback"].get_data()), indent=4)
        write_data_to_azure_blob(
            st.session_state["container_client"],
            f"feedback/feedback_{_get_session().id}.json",
            feedback_json,
        )
    else:
        st.session_state.num_result += 1

def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session


if __name__ == "__main__":
    main()


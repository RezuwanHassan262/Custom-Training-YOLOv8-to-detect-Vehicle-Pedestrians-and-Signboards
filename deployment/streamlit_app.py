import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import torch
import numpy as np
import math
from ultralytics import YOLO
import time
import supervision as sv

st.set_page_config(
    page_title="Pedestrians, vehicles and signboard tracking",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
demo_img = os.path.join(ROOT, 'images', 'streamlit_image.png')


def get_example_images():
    folder = os.path.join(ROOT, 'images', 'inference_images')
    exts = ('.png', '.jpg', '.jpeg')
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)
    )

# Image part
def image_app(image, st, conf):
    if torch.cuda.is_available():
        device = torch.device('cuda')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    else:
        device = torch.device('cpu')


    model = YOLO(os.path.join(ROOT, "models_and_weights", "best.pt"))  # This correctly loads the YOLO model
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to device

    clsDict = model.names


    result = model(image, conf=conf)[0]
    bbox_xyxys = np.array(result.boxes.xyxy.cpu(), dtype = 'int')
    confidences = result.boxes.conf.cpu()
    # labels = np.array(result.boxes.cls.cpu(), dtype='int')
    labels = result.boxes.cls.tolist()

    for (bbox_xyxy, conf, cls) in zip(bbox_xyxys, confidences, labels):
        (x1, y1, x2, y2) = bbox_xyxy
        class_name = clsDict[cls]
        label = f"{class_name} {conf:.03}"
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

        c2 = x1 + t_size[0], y1 - t_size[1] - 3

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 225), 2) # bbox
        cv2.rectangle(image, (x1, y1), c2, (150, 0, 0), -1, cv2.LINE_AA) # rec for class names

        cv2.putText(image, label, (x1, y1-2), 0, 1, (0, 0, 225), 1, lineType = cv2.LINE_AA)
        
    st.subheader(':green[Output Image]')
    st.image(image, channels='BGR')
    # st.markdown(
    #         """

    #     :green[This model has also been deployed as a web app on HuggingFace Spaces also.]

    #     :green[[Click here](https://huggingface.co/spaces/Rezuwan/Road_and_Pedestrian_Detection) to check out the HuggingFace spaces version of this web app.]

    #         """
    # )



def main():
    st.title(":green[Pedestrians, vehicles and signboard tracking]")
    #st.title("_Streamlit_ is :blue[cool] :sunglasses:")
    # st.sidebar.title('Settings')
    # st.sidebar.subheader('Parameter')

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("https://images.adsttc.com/media/images/58f8/f346/e58e/ceac/3100/0990/large_jpg/201001_NY_Times_Square_Sn_hetta_N58_publication.jpg?1492710206");
            background-size: cover;
        }
        [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html = True
    )

    app_mode = st.sidebar.selectbox('Choose the App Mode', ['App Description', 'Run on Image'], index=1)

    if app_mode == 'App Description':
        st.markdown(":green[This app markdowns pedestrians, vehicles and signboards from urban street images]")   
        st.image(demo_img)
        # st.markdown(
        #     """

        # :green[This model has also been deployed as a web app on HuggingFace spaces also.]

        # :green[[Click here](https://huggingface.co/spaces/Rezuwan/Road_and_Pedestrian_Detection) to check out the HuggingFace spaces version of this web app.]

        #     """
        # )
        img = cv2.imread(demo_img)
        image = np.array(Image.open(demo_img))


    elif app_mode == 'Run on Image':
        st.sidebar.title('Settings')
        st.sidebar.subheader('Parameter')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        st.sidebar.markdown('---')
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

        example_paths = get_example_images()
        if img_file_buffer is not None:
            img = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), 1)
        else:
            names = [os.path.basename(p) for p in example_paths]
            choice = st.sidebar.selectbox('Or pick an example image', names)
            img = cv2.imread(example_paths[names.index(choice)])

        st.sidebar.text('Example Images')
        for p in example_paths:
            st.sidebar.image(Image.open(p), caption=os.path.basename(p))

        image_app(img, st, confidence) 


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

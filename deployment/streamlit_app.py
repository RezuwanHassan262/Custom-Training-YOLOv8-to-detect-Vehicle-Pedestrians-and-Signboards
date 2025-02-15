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

demo_img = 'images/streamlit_image.png'
# Image part
def image_app(image, st, conf):
    if torch.cuda.is_available():
        device = torch.device('cuda')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    else:
        device = torch.device('cpu')


    model = YOLO("models_and_weights/best.pt")  # This correctly loads the YOLO model
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
    st.markdown(
            """

        :green[This model has also been deployed as a web app.]

        :green[[Click here](https://huggingface.co/spaces/Rezuwan/Road_and_Pedestrian_Detection) to check out the WebApp version of this app.]

            """
    )



def main():
    st.title(":green[Pedestrians, vehicles and signboard tracking]")
    #st.title("_Streamlit_ is :blue[cool] :sunglasses:")
    # st.sidebar.title('Settings')
    # st.sidebar.subheader('Parameter')

    # st.markdown(
    #     """
    #     <style>
    #     [data-testid="stAppViewContainer"] {
    #         background-image: url("https://images.adsttc.com/media/images/58f8/f346/e58e/ceac/3100/0990/large_jpg/201001_NY_Times_Square_Sn_hetta_N58_publication.jpg?1492710206");
    #         background-size: cover;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html = True
    # )

    app_mode = st.sidebar.selectbox('Choose the App Mode', ['App Description', 'Run on Image'])

    if app_mode == 'App Description':
        st.markdown(":green[This app markdowns pedestrians, vehicles and signboards from urban street images]")   
        st.image(demo_img)
        st.markdown(
            """

        :green[This model has also been deployed as a web app.]

        :green[[Click here](https://huggingface.co/spaces/Rezuwan/Road_and_Pedestrian_Detection) to check out the WebApp version of this app.]

            """
        )
        img = cv2.imread(demo_img)
        image = np.array(Image.open(demo_img))


    elif app_mode == 'Run on Image':
        st.sidebar.title('Settings')
        st.sidebar.subheader('Parameter')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        st.sidebar.markdown('---')
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

        # if img_file_buffer is not None:
        #     img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
        #     image = np.array(Image.open(img_file_buffer))
        
        # else:
        #     img = cv2.imread(demo_img)
        #     image = np.array(Image.open("images/inference_images/eg_3.png"))
        
        # st.sidebar.text('Input Image')
        # st.sidebar.image(np.array(Image.open("images/inference_images/eg_3.png")))
        
        img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
        image = np.array(Image.open(img_file_buffer))
        image_app(img, st, confidence) 


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

<h1 align='center' style=color:#fe5e21;><strong>Custom Training YOLOv8n to detect Vehicle, Pedestrians and Signboards</strong></h1>

This object detection model was trained on a custom dataset that can detect Vehicle, Pedestrians and Signboards from images.

<br/>

<h2 style=color:#fe5e21;>Dataset Development</h2>

The dataset was developed by extracting only 150 frames by setting the `stride=100` from this YouTube [video](https://www.youtube.com/watch?v=7HaJArMDKgI&ab_channel=JUtah) and was annotated mostly using **Roboflow's auto-labeling** feature in both using python scripts and roboflow platform.
Some samples of the dataset are given below.

<h4 style=color:#fe5e21;>Roboflow Platform</h4>

![roboflow_platform_annotated_dataset](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/roboflow_annotated_data.PNG)


<h4 style=color:#fe5e21;>Python Script</h4>

![python_script_annotated_dataset](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/annotated_data_batch.png)


<h4 style=color:#fe5e21;>Dataset Links</h4>

[HuggingFace](https://www.youtube.com/watch?v=7HaJArMDKgI&ab_channel=JUtah) | [Roboflow](https://app.roboflow.com/bondstein-technologies-limited/bondstein_project/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)


<h2 style=color:#fe5e21;>Model Development</h2>

I finetuned a pre-trained model, YOLOv8n to be exact in 2 different environments with the same set of hyperparameter tunings. One on Kaggle and one on the Roboflow platform and then reported the Kaggle model since it yielded better results.


<h4 style=color:#fe5e21;>Model Architecture</h4>

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/YOLOv8n_architecture.jpg)



<h4 style=color:#fe5e21;>Kaggle Training Result</h4>

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/results.png)

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/cr.PNG)

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/cf.png)

<h4 style=color:#fe5e21;>Roboflow Platform Training Result</h4>

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/graphs.PNG)

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/x1.PNG)

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/x2.PNG)

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/x3.PNG)


##  Project Overview
- **Model**: YOLOv8 (fine-tuned)
- **Dataset**: Extracted frames from a video and labeled using Roboflow
- **Training**: Trained using Ultralytics YOLOv8 framework
- **Application**: Custom object detection

## Note: couldn't upload everything related to this project on github due to space restriction. Please find everything [here](https://drive.google.com/drive/folders/1Mf7FGdRDhd3JZC-tb-gghykeM1qa4cc6?usp=sharing)

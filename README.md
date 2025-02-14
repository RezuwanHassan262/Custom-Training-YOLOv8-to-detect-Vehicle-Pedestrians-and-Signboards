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

I finetuned the model on 2 different platforms, The results below are for the Kaggle platform.

![training_curves](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/results.png)

The training appears healthy and effective, with no strong signs of overfitting. The model is improving consistently, and validation metrics align well with training progress.

The figure above illustrates the training losses (train/box_loss, train/cls_loss, train/dfl_loss) decrease smoothly indicating the model is learning effectively and the validation losses (val/box_loss, val/cls_loss, val/dfl_loss) also decrease, following a similar pattern to the training losses, meaning the model generalizes well. As we see in the subplots, the Precision, Recall, and mAP scores are increasing consistently indicating the model is improving its detection capabilities. Some fluctuations in precision and recall are visible but they generally trend upwards suggesting the model is not memorizing the training data excessively and not overfitting. The overall trend suggests effective learning and performance improvement over time.


![classification_report](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/cr.PNG)

The table is summarizing precision (P), recall (R), and mean Average Precision (mAP50 & mAP50-95) of the model across different classes: pedestrians, signposts, and vehicles.
The model shows good overall performance with no strong signs of overfitting, as precision and recall are balanced, and validation mAP is reasonable. Vehicles are detected well (P: 0.805, R: 0.868, mAP50: 0.903), while pedestrians perform decently (P: 0.73, R: 0.692, mAP50: 0.758), but signposts struggle (P: 0.581, R: 0.48, mAP50: 0.535), likely due to class imbalance or insufficient training data. The drop in mAP50-95 (0.501 overall) suggests bounding box localization could improve. 


![confusion_matrix](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/cf.png)

The confusion matrix above illustrates a strong diagonal presence indicates good overall classification, but significant misclassifications exist. Pedestrians and signposts show weaker performance. The background class, which should not exist, absorbs many incorrect predictions, suggesting the model struggles with uncertain detections. 


<h4 style=color:#fe5e21;>Roboflow Platform Training Result</h4>

As mentioned earlier, I finetuned the model on 2 different platforms, The results below are for the RoboFlow platform.

![training_curves](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/graphs.PNG)

Over the 300 epochs during the training session of the model, the progression of mAP (mean Average Precision) and loss functions. The mAP graph (top) indicates steady improvement suggesting decent model performance. The loss graphs show a sharp initial decline stabilizing at lower values indicating effective learning without major signs of overfitting. The slight increase in object loss* towards the later epochs may suggest minor instability. Overall trends are suggesting the model is learning well.


![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/x1.PNG)

<!--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$-->


![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/x2.PNG)

![yolov8_model_architecture](https://raw.githubusercontent.com/RezuwanHassan262/YOLOv8-Custom-Training-Object-Detection/main/images/x3.PNG)


##  Project Overview
- **Model**: YOLOv8 (fine-tuned)
- **Dataset**: Extracted frames from a video and labeled using Roboflow
- **Training**: Trained using Ultralytics YOLOv8 framework
- **Application**: Custom object detection

## Note: couldn't upload everything related to this project on github due to space restriction. Please find everything [here](https://drive.google.com/drive/folders/1Mf7FGdRDhd3JZC-tb-gghykeM1qa4cc6?usp=sharing)

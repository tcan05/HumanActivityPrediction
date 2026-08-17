# Human Activity Prediction

## About This Project
This is a simple project about classifying human activity based on images which scores around 90% accuracy. System first resizes every image to 320x320 pixels and then augments the images for increased training data. It then uses EfficientNetB1 for training the model. Before running the application, you must determine:

- the path to your dataset
- path to where you want to save the model
- your json path
- your test image path

The model classifies human activity into 6 categories: control, sitting, stairs, standing, sleeping, walking. Also keep in mind that you must find an image for testing and that image must be in .jpg format.

## Requirements
- Python 3.10+
- tensorflow
- numpy

## Model

The trained EfficientNetB1 model is available on Hugging Face:

[Hugging Face model](https://huggingface.co/tcan05/HumanActivityPredictor/tree/main)

## Dataset

The dataset is available at:

- [Original Dataset](https://www.kaggle.com/datasets/jithinnambiarj/human-activity-detection-dataset/versions/1/data)
- [Modified Dataset](https://huggingface.co/datasets/tcan05/HumanActivityDataset/tree/main)

Modified dataset is the dataset I used for training the model. Only difference is that in modified dataset every image is in .jpg format.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/tcan05/HumanActivityPrediction.git 
```

2. Navigate to the project directory:
```bash
cd HumanActivityPrediction/HAP Project
```

3. Run the application:
```bash
python main.py
```

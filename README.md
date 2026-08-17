# Human Activity Prediction

## About This Project
This is a simple project about classifying human activity based on images which scores around 90% accuracy. System first resizes every image to 320x320 pixels and then augments the images for increased training data. It then uses EfficientNetB1 for training the model. Before running the application, you must determine:

- the path to your dataset
- path to where you want to save the model
- your json path
- your test image path

Also keep in mind that you must find an image for testing.

## Requirements
- Python 3.10+

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

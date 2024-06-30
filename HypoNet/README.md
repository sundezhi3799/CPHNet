# HypoNet: Hypoxia Scoring Network

## Overview
HypoNet is a deep learning tool designed to classify cells into hypoxic or non-hypoxic states. It extends the ResNet-50 model to handle six-channel Cell Painting image data (DNA, AG, ER, Mito, PM, and RNA), enabling effective scoring of hypoxia and screening for potential anti-hypoxic agents.

## Key Features
- **Cell State Classification**: Classifies cells based on hypoxia status.
- **Multi-Channel Image Processing**: Tailored to process six-channel Cell Painting images.
- **AI-driven Drug Screening**: Utilizes deep learning to score potential anti-hypoxic drugs.

## Dependencies
- Python 3.8 or higher
- PyTorch 1.13.1
- Other dependencies listed in `requirements.txt`

## File Structure
- `custom_resnet.py`: Custom network model.
- `custom_train.py`: Model training script.
- `custom_test.py`: Model testing script.
- `hypoxia_scoring.py`: Script for scoring cellular hypoxia.
- ... (brief explanations for other files)

## Usage Instructions
### Training the Model
```bash
python custom_train.py -net resnet50 -gpu 0
```
### Testing the Model
```bash
python custom_test.py -net resnet50 -weights path_to_resnet50_weights_file
```
### Hypoxia Scoring
```bash
python hypoxia_scoring.py -net resnet50 -weights path_to_weights.pth -gpu
```
### Prediction
```bash
python predict.py -net resnet50 -weights path_to_weights.pth -gpu
```

## Developers and Contributors
- Dezhi Sun



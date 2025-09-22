# Sentiment Analysis on Movie Reviews

## Table of Contents
1. [Context]
2. [How to Use]
3. [Data]

## Context
This repository contains the full development and evaluation of a binary sentiment classification system, along with the saved models for deployment.

## How to Use
### Installation
This repository requires Python 3.12.0

Clone and change to repository
```bash
git clone https://github.com/LuisBarbosa02/Sentiment-Analysis-on-Movie-Reviews.git
cd Sentiment-Analysis-on-Moview-Reviews 
```

[Optional] Create and activate a virtual environment
```bash
python3.12 -m venv venv
source venv/bin/activate
```

Install the dependencies
```bash
pip install -r requirements.txt
```

### Running Scripts
The trained models are already saved in the *models* folder. If deployment is needed, simply import a model using the *joblib* Python module.

To train the model on a specific dataset, run the following command from the main directory:
```bash
python train_eval_(dataset_name).py
```
Replace (dataset_name) with the name of the dataset used for training.

To evaluate the models on 100 artificially created samples, run:
```bash
python predict.py
```

## Data
- The *Polarity Dataset v2.0* and *Sentence Polarity Dataset v1.0* were obtained from:
https://www.cs.cornell.edu/people/pabo/movie-review-data/
- The *IMDb Large Movie Review Dataset v1.0* was obtained from:
https://ai.stanford.edu/~amaas/data/sentiment/
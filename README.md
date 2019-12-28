# Language Identification using Deep Convolutional Recurrent Neural Networks

This repository contains the code for the paper "Language Identification Using Deep Convolutional Recurrent Neural
Networks" by Bartz, C., Herold, T., Yang, H., and Meinel, C. [1]

This project is developed as a partial fulfillment of the requirements for the Deep Learning (CSE 676) course at the
University at Buffalo.

## Structure of the Repository

- **/config**
  - `sources.yml`: source file for downloading data from YouTube
  - **/pre-trained models**
    - `CRNN_EN_DE_FR.model`: pre-trained model for English, German and French languages
    - `CRNN_EN_HI_DE.model`: pre-trained model for English, Hindi and German languages
- **/scripts**
  - `models.py`: python script for the defined models (Convolutional Recurrent Neural Network)
  - `train.py`: python script for training a models
  - `predict.py`: python script for using a pre-trained model and checking model performance on real audio samples
  - **/tools**
    - `audio_to_image.py`: python script to convert audio data to image representations (Spectrogram representation)
    - `build_data.py`: python script to generate training, validation and test CSVs
    - `data_loader.py`: python script to load the data in CSV
    - `download_youtube.py`: python script to download audio from YouTube. Uses `config/sources.yml`

## Requirements

Library requirements for this project can be found in `requirements.txt` and can be installed using
```
pip install -r requirements.txt
```

## Models

This repository contains two models. One for English, German and French languages and the other for English, Hindi and
German languages. The models can be found in the folder `config/pre-trained models`

#### Data sourcing and pre-processing

Update `config/sources.yml` as per requirement and run
```
python scripts/tools/download_youtube.py --output <path_to_data_directory>
```

This will:
- Download audio files from the respective YouTube channel/playlist videos
- Convert the downloaded audio file into WAVE format
- Segment them into parts of 10 seconds each
- Generate the CSV for training, validation and test datasets

#### Training & Prediction

To start training, run the command:
```
python train.py --dir <path_to_segmented_data_directory>
```

To predict a single audio file, run the command:
```
python predict.py --model <path_to_model> --input <path_to_audio_file>
```

### References:

Bartz, C., Herold, T., Yang, H., and Meinel, C.: ‘Language identification using deep convolutional recurrent neural
networks’, in Editor (Ed.)^(Eds.): ‘Book Language identification using deep convolutional recurrent neural networks’
(Springer, 2017, edn.), pp. 880-889 [[PDF]](https://arxiv.org/pdf/1708.04811v1.pdf)

# Sign Language Detection Project

This project implements a sign language detection pipeline using video inputs and deep learning models. It processes video data through several stages including preprocessing, feature extraction, transformer-based recognition, and optional segmentation to improve detection performance.


## Getting Started

### Prerequisites
- Preprocessed dataset link-https://drive.google.com/drive/folders/15Gqg2LTVP5iSEfP0z6LQkkLysO4FKofy?usp=drive_link
- Sapiens weights-https://drive.google.com/file/d/1o-PEVf6EDfxK8n4nPWZw218ud7ko4ANg/view?usp=drive_link
- train.csv and test.csv has been attached but can be downloaded from https://github.com/hoangchunghien/Sign-Language-Recognition/blob/master/data
- Python 3.8 or above
- PyTorch 2.6.0
- Other dependencies listed in `requirements.txt`
- setup the sapiens lite for segmentation masks and change path in scripts/train.sh and scripts/test.sh

Change paths wherever required while running the scripts.
### Installation

1. Clone the repository:
   ```
   git clone git@github.com:b22237/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. **For training**

   ```
   bash scripts/train.sh (or train_lstm.sh for lstm model with attention)
   ```
1. **For testing(real time)**

   ```
   bash scripts/test.sh
   ```
synthetic data has been provided in data/videos. 
## Notes

- Ensure that the label mapping file is correctly aligned with your training data.
- Check the log files in the `logs/` directory for detailed run-time information.
- For advanced segmentation, the project can integrate with the Sapiens framework for human-centric vision tasks.

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgements

This project integrates parts of the Sapiens framework for human-centric vision tasks. Special thanks to the developers behind Sapiens for their open-source contributions. 
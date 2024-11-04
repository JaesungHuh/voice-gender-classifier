# Voice gender classifier 
[![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FJaesungHuh%2Fvoice-gender-classifier%3Fexpand%255B%255D%3Ddownloads%26expand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=%F0%9F%A4%97%20Downloads)](https://huggingface.co/JaesungHuh/voice-gender-classifier)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-blue)](https://huggingface.co/spaces/JaesungHuh/voice-gender-classifier)

- This repo contains the inference code to use pretrained human voice gender classifier.

## Installation
First, clone this repository
```
git clone https://github.com/JaesungHuh/voice-gender-classifier.git
```

and install the packages via pip.

```
cd voice-gender-classifier
pip install -r requirements.txt
```

## Usage
```
import torch

from model import ECAPA_gender

# You could directly download the model from the huggingface model hub
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval()

# If you are using gpu .... 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the audio file and use predict function to directly get the output
example_file = "data/00001.wav"
with torch.no_grad():
    output = model.predict(example_file, device=device)
    print("Gender : ", output)
```

## Pretrained weights
For those who need pretrained weights, please download them [here](https://drive.google.com/file/d/1ojtaa6VyUhEM49F7uEyvsLSVN3T8bbPI/view?usp=sharing).

## Training details
State-of-the-art speaker verification model already produces good representation of the speaker's gender.

I used the pretrained ECAPA-TDNN from [TaoRuijie's](https://github.com/TaoRuijie/ECAPA-TDNN) repository, added one linear layer to make a two-class classifier, and finetuned the model with the VoxCeleb2 dev set.

The model achieved **98.7%** accuracy on the VoxCeleb1 identification test split.

## Caveat
I would like to note that the training dataset I've used for this model (VoxCeleb) may not represent the global human population. Please be careful of unintended biases when using this model.

## References
- 🤗 [Huggingface Hub link](https://huggingface.co/JaesungHuh/voice-gender-classifier)
- I modified the model architecture from [TaoRuijie's](https://github.com/TaoRuijie/ECAPA-TDNN) repository.
- For more details about ECAPA-TDNN, check the [paper](https://arxiv.org/abs/2005.07143).

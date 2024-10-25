import torch

from model import ECAPA_gender

# You could directly download the model from the huggingface model hub
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval()

# If you are using gpu or not.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the example file and use predict function to directly get the output
example_file = "data/00001.wav"
with torch.no_grad():
    output = model.predict(example_file, device=device)
    print("Gender : ", output)
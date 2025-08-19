
# wav2face
Direct audio to 3D facial landmark generation (no phoneme conversion) with a Temporal Convolutional Network (TCN).

---

## Setup

Create Conda Environment:

```bash
conda create -n audio2landmark python=3.10 -y
conda activate audio2landmark
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Structure

The script expects the dataset to be preprocessed with the following structure:



```
data_root/
    video_01/
        audio_chunk/video_01_mfcc.npy       # MFCC features (N, 40)
        landmarks/video_01_landmarks.npy    # Landmarks (N, 1434) where 1434 = 478*3
    video_02/
        audio_chunk/video_02_mfcc.npy
        landmarks/video_02_landmarks.npy
    ...
```

---

## Usage

### Train the model

```bash
python train_tcn.py -n my_model -d /path/to/processed_video
```

**Arguments:**

* `-n`, `--model_name` : Name for the trained model (used in checkpoint naming)
* `-d`, `--data_path` : Path to the processed dataset

---

## Gradio Application 

```bash

python landmark_gradio.py







```
## Demo
https://github.com/user-attachments/assets/022288f2-7e86-4eea-86ba-66d4011fe493

allows for paralanguage events (sighs etc) to be captured

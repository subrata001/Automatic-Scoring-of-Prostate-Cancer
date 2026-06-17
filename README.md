# Automatic Scoring of Prostate Cancer

Inference code for the accepted Elsevier *Intelligence Medicine* paper:

**Leveraging Efficient Transfer Learning and Heuristic Algorithms for Gigapixel Histopathology Image Analysis and Automatic Scoring of Prostate Biopsy: A Multicenter Risk Stratification Study**

## Overview

This repository provides the initial inference-only implementation of an Efficient Feature Distillation Convolutional Neural Network (EFDCNN) framework for prostate histopathology whole-slide image (WSI) analysis.

The method is designed for automatic patch-level Gleason pattern prediction and slide-level scoring of prostate biopsy WSIs. The full pipeline follows the three-stage binary classification strategy described in the paper:

1. **Stage 1:** Normal tissue vs. cancer tissue  
2. **Stage 2:** Low-grade vs. high-grade cancer  
3. **Stage 3:** Gleason pattern 4 vs. Gleason pattern 5  

The current release is intended to support paper publication and reproducibility of inference. Training scripts and extended experimental evaluation code will be added in a later update.

## Key Features

- Whole-slide image loading using OpenSlide
- Tissue preprocessing and patch extraction
- Three-stage EFDCNN-based inference
- Patch-level Gleason pattern prediction
- JSON-based prediction output
- Support for slide-level scoring and risk stratification based on predicted patch distributions

## Repository Structure

```text
Automatic-Scoring-of-Prostate-Cancer/
│
├── README.md
├── requirements.txt
├── .gitignore
├── run.py
│
├── dataloader/
│   ├── __init__.py
│   ├── load.py
│   └── patch.py
│
├── model/
│   ├── __init__.py
│   ├── infer.py
│   ├── load.py
│   └── weights/
│       ├── model_nc.tf
│       ├── model_lh.tf
│       └── model_45.tf
│
├── utils/
│   ├── __init__.py
│   ├── contour.py
│   ├── mask.py
│   ├── preprocess.py
│   └── save.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/subrata001/Automatic-Scoring-of-Prostate-Cancer.git
cd Automatic-Scoring-of-Prostate-Cancer
```

### 2. Create a Python environment

```bash
conda create -n efdcnn_prostate python=3.9.12
conda activate efdcnn_prostate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install OpenSlide system library

This code uses `openslide-python`, which requires the OpenSlide system library.

For Ubuntu/Linux:

```bash
sudo apt-get update
sudo apt-get install -y openslide-tools
```

For Windows, install OpenSlide binaries and add the OpenSlide `bin` folder to the system PATH.

## Requirements

The recommended `requirements.txt` file is:

```text
numpy==1.23.5
scipy==1.9.3
tensorflow==2.10.1
h5py==3.7.0
opencv-python==4.6.0.66
Pillow==9.2.0
tifffile==2022.8.12
openslide-python==1.2.0
matplotlib==3.7.0
```

## Model Weights

Place the trained model weights inside:

```text
model/weights/
```

The current inference code expects the following model names:

```text
model_nc.tf   # Stage 1: normal tissue vs. cancer tissue
model_lh.tf   # Stage 2: low-grade vs. high-grade cancer
model_45.tf   # Stage 3: Gleason pattern 4 vs. Gleason pattern 5
```

If the model weight names are changed, update the model-name tuple in `run.py`.

Example:

```python
model_file_names = ("model_nc.tf", "model_lh.tf", "model_45.tf")
```

Depending on how the models were saved, each model may be a TensorFlow SavedModel folder or a single `.h5`/`.keras` file. The code loads models using:

```python
tensorflow.keras.models.load_model()
```

## Running Inference

### 1. Prepare a WSI file

Place your prostate WSI file in an accessible local folder. Supported formats depend on OpenSlide and may include `.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs`, and related pathology slide formats.

### 2. Set the WSI path in `run.py`

Edit the WSI path:

```python
wsi_path = "path/to/your/prostate_slide.tiff"
```

### 3. Run inference

```bash
python run.py
```

## Output

Prediction results are saved under:

```text
result/<slide_name>/
```

Typical output files include:

```text
pattern3.json
pattern4.json
pattern5.json
<slide_name>_count.json
```

The JSON files store patch-level prediction coordinates or contour information for predicted Gleason patterns. The count file summarizes the number of predicted patches for each class and can be used for slide-level Gleason scoring and risk stratification.

## Method Summary

The EFDCNN inference pipeline performs WSI analysis as follows:

1. Load the WSI.
2. Preprocess and identify tissue-containing regions.
3. Extract patches from the WSI.
4. Apply the three trained EFDCNN models.
5. Assign patch-level predictions.
6. Save predicted Gleason pattern outputs.
7. Summarize prediction counts for slide-level interpretation.

## Data Availability

This repository does not include raw WSIs or private clinical data.

Users should download public datasets from their original sources and follow the corresponding data-use agreements, licenses, and citation requirements. Private or institutional WSIs should not be shared without appropriate IRB, institutional, and ethical approval.

## Current Release Status

This repository currently provides the initial inference-only code release for the accepted paper. Future updates may include:

- Training code
- Self-supervised pretraining scripts
- Full EFDCNN model-building scripts
- Evaluation scripts
- Example WSI inference tutorial
- Additional documentation for slide-level Gleason scoring

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{Bhattacharjee_EFDCNN_Prostate_WSI,
  title   = {Leveraging Efficient Transfer Learning and Heuristic Algorithms for Gigapixel Histopathology Image Analysis and Automatic Scoring of Prostate Biopsy: A Multicenter Risk Stratification Study},
  author  = {Bhattacharjee, Subrata and Armand, Tagne Poupi Theodore and Hwang, Yeong-Byn and Hayat, Mansoor and Kim, Hee-Cheol and Kang, Suki and Cho, Nam-Hoon and Ryu, Wi-Sun and Kim, Dong Min and Choi, Heung-Kook and Ajakwe, Simeon Okechukwu},
  journal = {Intelligence Medicine},
  year    = {2026},
  note    = {Accepted}
}
```

The final DOI and publication details will be added after online publication.

## Disclaimer

This software is provided for research purposes only. It is not intended for direct clinical diagnosis, treatment planning, or medical decision-making without appropriate validation, regulatory approval, and review by qualified pathologists.

## License

A license should be added after approval from all co-authors and the affiliated institution. Until a license is added, reuse rights may be restricted.

## Contact

For questions, please open an issue on this GitHub repository or contact the corresponding authors listed in the paper.

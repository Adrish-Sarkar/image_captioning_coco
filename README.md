# image_captioning_coco

# Image Captioning with COCO Dataset

A deep learning project that generates captions for images using a CNN-RNN architecture (ResNet Encoder + LSTM Decoder) trained on the COCO 2014 dataset.

## 📋 Overview

This project implements an image captioning model that can automatically generate descriptive captions for images. The model uses:
- **Encoder**: Pre-trained ResNet CNN to extract image features
- **Decoder**: LSTM network to generate captions word-by-word

## 🚀 Features

- Cross-platform data downloading (works on Windows, Linux, macOS)
- Automated vocabulary building from COCO captions
- Custom PyTorch Dataset and DataLoader implementation
- Image preprocessing and augmentation
- Visualization tools to verify data loading

## 📦 Requirements

```bash
pip install torch torchvision pycocotools nltk matplotlib pillow
```

## 🗂️ Project Structure

```
ML Projects/
├── Image_Captioning_COCO.ipynb    # Main notebook with complete pipeline
├── data/                          # Dataset directory (created automatically)
│   ├── train2014/                # Training images
│   ├── val2014/                  # Validation images
│   └── annotations/              # COCO annotations
└── README.md                      # This file
```

## 💻 Usage

### Running the Notebook

1. **Open the notebook** in Jupyter or Google Colab:
   ```bash
   jupyter notebook Image_Captioning_COCO.ipynb
   ```

2. **Run cells sequentially**:
   - **Step 1**: Download COCO dataset (~19GB total)
   - **Step 2**: Extract dataset files
   - **Step 3**: Clean up zip files
   - **Step 4**: Build vocabulary from captions
   - **Step 5**: Define custom Dataset class
   - **Step 6**: Create DataLoader
   - **Step 7**: Verify data loading (visualize samples)

### Important Notes

- The dataset download may take significant time depending on your internet connection
- Ensure you have at least **25GB of free disk space**
- The import cell (with `import torch.utils.data as data`) must be run before Step 5

## 🔧 Configuration

Key hyperparameters (defined in Step 6):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embed_size` | 256 | Word embedding dimension |
| `hidden_size` | 512 | LSTM hidden state size |
| `num_layers` | 1 | Number of LSTM layers |
| `batch_size` | 128 | Training batch size |
| `num_epochs` | 5 | Number of training epochs |
| `learning_rate` | 0.001 | Optimizer learning rate |
| `threshold` | 4 | Minimum word frequency for vocabulary |
| `crop_size` | 224 | Image crop size |

## 📊 Dataset

**COCO 2014 Dataset**:
- Training images: ~82,000 images
- Validation images: ~40,000 images
- Each image has 5 human-annotated captions
- Total captions: ~400,000+

## 🛠️ Troubleshooting

### NameError: name 'data' is not defined
**Solution**: Run the import cell that contains `import torch.utils.data as data`

### wget command not found (Windows)
**Solution**: The notebook now uses Python's `urllib` for downloads - no external tools needed

### Out of memory errors
**Solution**: Reduce `batch_size` in the configuration

## 📝 License

This project uses the COCO dataset, which is licensed under Creative Commons Attribution 4.0 License.

## 🙏 Acknowledgments

- COCO Dataset: [https://cocodataset.org/](https://cocodataset.org/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)

## 📧 Contact

For questions or issues, please open an issue in the repository.

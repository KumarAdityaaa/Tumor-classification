# Brain Tumor MRI Classification

Research code for multi-class brain tumor classification from MRI images, comparing a custom CNN with pretrained backbones. The repository focuses on reproducible evaluation across four classes: glioma, meningioma, pituitary tumor, and no tumor.

## 📊 Dataset
(The dataset is currently not availabe will upload soon, contact for details)
The dataset consists of MRI brain images categorized into four classes:
- **Glioma Tumor** (glioma_tumor)
- **Meningioma Tumor** (meningioma_tumor)
- **No Tumor** (no_tumor)
- **Pituitary Tumor** (pituitary_tumor)

Images are automatically resized to 224×224 pixels during preprocessing. The dataset is split into train/validation/test sets with stratified sampling to maintain class balance.

## 🏗️ Project Structure

```
brain_tumor_classification/
├── config.yaml              # Configuration file with hyperparameters
├── requirements.txt         # Python dependencies
├── train.py                 # Training script for all models
├── evaluate.py              # Evaluation script with metrics and plots
├── data/
│   ├── raw/                 # Raw MRI images organized by class
│   ├── processed/           # Processed images (if any)
│   └── splits/              # Train/val/test CSV files
├── models/
│   ├── custom_cnn.py        # Custom CNN architecture
│   └── pretrained.py        # Pretrained model implementations
├── notebooks/
│   └── eda.py               # Exploratory data analysis
├── results/
│   ├── figures/             # Generated plots and reports
│   └── saved_models/        # Trained model weights
├── scripts/
│   ├── flatten_data.py      # Data preprocessing utilities
│   └── generate_figures.py  # Figure generation scripts
└── src/
    └── dataset.py           # Data loading and preprocessing utilities
```

## 🚀 Installation

1. Download or clone the repository to your local environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Place your MRI dataset in the `data/raw/` directory with subfolders for each class.

## 📋 Usage

### Training

Train any of the available models:

```bash
# Custom CNN
python train.py --model custom_cnn

# Pretrained models
python train.py --model vgg16
python train.py --model vgg19
python train.py --model resnet50
python train.py --model efficientnetb0
```

### Evaluation

Evaluate trained models:

```bash
# Evaluate custom CNN
python evaluate.py --model custom_cnn

# Evaluate pretrained models
python evaluate.py --model vgg16
# ... etc
```

### Data Preparation

If you need to regenerate the train/val/test splits:

```bash
python -c "from src.dataset import load_config, build_split_csv; build_split_csv(load_config())"
```

## 🧠 Models

### Custom CNN
- 4 convolutional blocks with increasing filters (32→64→128→256)
- Batch normalization after each convolution
- Global average pooling
- Dropout regularization
- Configurable dense layer units and dropout rate

### Pretrained Models
- **VGG16/VGG19**: Feature extraction followed by fine-tuning
- **ResNet50**: Residual network with transfer learning
- **EfficientNetB0**: Efficient scaled architecture

All pretrained models use:
- Frozen base layers initially
- Gradual unfreezing for fine-tuning
- Warmup training on classification head
- Lower learning rates for fine-tuning

## ⚙️ Configuration

All hyperparameters are defined in `config.yaml`:

- **Data**: Image size, train/val/test splits, class names
- **Augmentation**: Horizontal flip, rotation, zoom, brightness adjustments
- **Training**: Batch size, epochs, learning rate, early stopping
- **Models**: Architecture-specific parameters

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 (Macro) |
|-------|----------|-----------|--------|------------|
| VGG16 | 0.9346 | 0.9401 | 0.9298 | 0.9340 |
| VGG19 | 0.9198 | 0.9246 | 0.9175 | 0.9199 |
| ResNet50 | 0.8966 | 0.8992 | 0.8928 | 0.8950 |
| EfficientNetB0 | 0.8333 | 0.8512 | 0.8369 | 0.8367 |
| Custom CNN | 0.7954 | 0.8095 | 0.7704 | 0.7811 |

### Generated Outputs

For each model, the evaluation generates:
- **Classification Report**: Detailed precision, recall, F1-scores per class
- **Confusion Matrix**: Both count and normalized versions
- **ROC Curves**: One-vs-rest curves for each class with AUC scores
- **Training Logs**: CSV files with loss/accuracy curves

All outputs are saved in `results/figures/<model_name>/`

## 🔧 Key Features

- **Stratified Splitting**: Maintains class balance across train/val/test sets
- **Data Augmentation**: Multiple augmentation techniques to prevent overfitting
- **Class Weighting**: Balanced class weights for imbalanced datasets
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Learning Rate Scheduling**: Automatic reduction on plateau
- **Transfer Learning**: Two-stage training for pretrained models
- **Comprehensive Evaluation**: Multiple metrics and visualizations

## 📚 Dependencies

- TensorFlow 2.15.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- OpenCV, Pillow
- PyYAML, tqdm

## 📄 License

This repository is provided for research and publication purposes. Please refer to the license file for usage terms.

## 📌 Research note

Intended for academic reference and reproducible evaluation. Please cite the associated research work if this code is used in a publication.</content>
<parameter name="filePath">d:\Research\brain_tumor_classification\README.md
# TDCE: Tabular Diffusion for Counterfactual Explanations

This is the implementation of TDCE (Tabular Diffusion for Counterfactual Explanations), which generates counterfactual explanations for tabular data using diffusion models with classifier guidance.

## Overview

TDCE is a method for generating counterfactual explanations for tabular data by:
1. Using Gaussian diffusion for numerical features
2. Using Gumbel-Softmax diffusion for categorical features
3. Applying classifier gradient guidance to ensure counterfactuals flip the target label
4. Supporting immutable features (features that cannot be changed)

## Key Features

- **Gumbel-Softmax Diffusion**: Handles categorical features with gradient-based updates
- **Classifier Guidance**: Uses classifier gradients to guide counterfactual generation
- **U-Net Architecture**: Uses U-Net (adapted for tabular data) as the denoising network
- **Immutable Features**: Supports masking features that should not be changed
- **Inverse Transformation**: Converts generated counterfactuals back to original data space

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TDCE
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml  # if available
# or
conda activate tdce
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

Prepare your dataset in the following format:
```
data/
└── [dataset_name]/
    ├── X_num_train.npy
    ├── X_cat_train.npy
    ├── y_train.npy
    ├── X_num_val.npy
    ├── X_cat_val.npy
    ├── y_val.npy
    ├── X_num_test.npy
    ├── X_cat_test.npy
    ├── y_test.npy
    └── info.json
```

### 2. Train Diffusion Model

```bash
python scripts/train.py \
    --config exp/[dataset_name]/config.toml \
    --data_path data/[dataset_name] \
    --parent_dir exp/[dataset_name] \
    --device cuda:0
```

### 3. Train Classifier

```bash
python scripts/train_classifier.py \
    --config exp/[dataset_name]/config.toml \
    --data_path data/[dataset_name] \
    --output_path exp/[dataset_name]/classifier.pt \
    --device cuda:0
```

### 4. Generate Counterfactuals

```bash
python scripts/sample_counterfactual.py \
    --config exp/[dataset_name]/config.toml \
    --original_data exp/[dataset_name]/original_samples.npy \
    --classifier_path exp/[dataset_name]/classifier.pt \
    --output exp/[dataset_name]/counterfactuals.npy \
    --target_y 1 \
    --lambda_guidance 1.0 \
    --device cuda:0
```

### 5. Evaluate Counterfactuals

```bash
python scripts/evaluate_counterfactuals.py \
    --original_samples exp/[dataset_name]/original_samples.npy \
    --counterfactual_samples exp/[dataset_name]/counterfactuals.npy \
    --classifier_path exp/[dataset_name]/classifier.pt \
    --config exp/[dataset_name]/config.toml \
    --data_path data/[dataset_name] \
    --target_y 1 \
    --device cuda:0 \
    --output exp/[dataset_name]/evaluation_results.txt
```

## Configuration

The configuration file (`config.toml`) includes:
- Model parameters (U-Net architecture, hidden dimensions, etc.)
- Training parameters (learning rate, batch size, epochs)
- Diffusion parameters (number of timesteps, noise schedule)
- Gumbel-Softmax parameters (temperature, annealing schedule)

See `CONFIG_DESCRIPTION.md` for detailed configuration options.

## Architecture

- **Denoising Network**: U-Net (adapted for tabular data with fully connected layers)
- **Classifier**: U-Net (same architecture as denoising network)
- **Numerical Features**: Gaussian diffusion
- **Categorical Features**: Gumbel-Softmax diffusion

## References

- TDCE Paper: [Paper Title/URL]
- Related Work: This implementation uses diffusion models for tabular data generation

## License

See LICENSE.md for details.

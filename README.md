# HEP Binary Classifier — PyTorch and TensorFlow

Two independent backends share ROOT readers, weight preprocessing, statistics and plots. Model definitions contain **no YAML access**.

## Repository

```text
train.py / test.py                 PyTorch
train_tf.py / test_tf.py           TensorFlow
tools/models/mlp_torch.py          PyTorch architecture
tools/models/mlp_tensorflow.py     TensorFlow architecture
tools/models/loader.py             Python module loader
tools/data.py                      ROOT loading and MeV-to-GeV conversion
tools/preprocess.py                weight policy, balancing, split/scaling
tools/metrics.py                   performance metrics and significance
tools/plotting.py                  independent plots
config.yaml / config_tf.yaml       train configs
test_config.yaml / test_config_tf.yaml independent inference configs
data/saved_models/                model and scaler checkpoints
output/                            reports and plots
```

## Run

```bash
python train.py -c config.yaml
python train_tf.py -c config_tf.yaml
python test.py -c test_config.yaml
python test_tf.py -c test_config_tf.yaml
```

Train configs use `training.model_name` (folder naming) and `model.model_path` (Python source). Test configs use `model.model_path` (trained `.pt`/`.keras` file) and `model.scaler_path` (`scaler.npz`). The same key is intentionally interpreted differently between train and test: in training it selects **source**, in testing **saved checkpoint**.

`training.use_weights` and `training.balanced_sample_weight` control only loss weighting. `training.drop_negative_weight` and `training.absolute_negative_weight` are mutually exclusive; choose exactly one for datasets with negatives. With `absolute_negative_weight: true`, signed weights remain intact for yield plots and reports, while training and ML metrics use absolute weights. Input data in MeV are converted by sample-specific `mev_to_gev` lists, exactly as in the original project.

Saved output: `data/saved_models/<training.model_name>/model.pt` (or `model.keras`), `scaler.npz`, and `metadata.json`. Each training run **overwrites the previous model and scaler with the same model name**. Plots and reports remain in timestamped `output/training_<timestamp>/` folders.

**Important:** A split test sample must be different from an external test sample; do not reuse training events for independent evaluation. Signed-weight stacked plots are only meaningful if MC weights carry proper physics normalization.

### Editing the architecture

Edit `tools/models/mlp_torch.py` or `tools/models/mlp_tensorflow.py` directly. Each file contains a small explicit `build_model()` definition, so there is no `hidden_layers` option in YAML. Training and inference load that Python file. **When editing a model architecture after training, keep the saved model and its implementation in sync**; the PyTorch checkpoint only contains the weights and the model source path.

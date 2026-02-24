# EEG Notebooks Guide

This folder contains a 2-stage EEG pipeline:

1. `EEG_J_01.ipynb` / `eeg_j_01.py`: convert EEG `.mat` signals into 2D map data and labels.
2. `EEG_J_02.ipynb` / `eeg_j_02.py`: train a deep model on generated 2D frames.

## Files In This Folder

- `EEG_J_01.ipynb`: notebook version of Stage 1 conversion.
- `eeg_j_01.py`: script export of Stage 1.
- `EEG_J_02.ipynb`: notebook version of Stage 2 training.
- `eeg_j_02.py`: script export of Stage 2.

## End-to-End Flow

1. Download preprocessed EEG dataset (`DATA_preproc.zip` from Zenodo).
2. Read each subject `.mat` file (`S*_data_preproc.mat`).
3. Sample `(subject, trial, channel)` signals.
4. Convert each 1D EEG signal to 2D map frames using SMM (`generate_both_maps`).
5. Save metadata:
   - `results.csv`
   - `labels.csv`
6. Build frame-level training records from generated `.npz` files.
7. Train `microsoft/cvt-13` for binary attention classification (`attended_1` vs `attended_2`).

## Models / Methods Used

### Stage 1 (Feature Generation)

- Library: `SMM` (`generate_both_maps`)
- Time-frequency method: `cwt`
- Spatial method: EEG topomap
- Output: combined 2D representations saved per signal folder

Note: this stage is signal-to-image transformation, not a trainable classifier.

### Stage 2 (Classifier)

- Backbone: `microsoft/cvt-13` (`CvtForImageClassification`)
- Task: binary classification (`num_labels=2`)
- Label mapping:
  - `0 -> attended_1`
  - `1 -> attended_2`
- The original 1000-class classifier head from the checkpoint is reinitialized for 2 classes (confirmed in notebook output).

## Key Config And Why

### Stage 1 (`eeg_j_01.py`)

- `TF_METHOD = "cwt"`
- `WINDOW_SEC = 4.0`
- `STEP_SEC = 0.5`
- `OUTPUT_MODE = "medium"`
- Default sampling strategy in run: `strategy="tenth"` (`10%`)

Why these values:

- EEG sampling rate is `64 Hz`.
- SMM uses internal `nperseg=256`.
- So chunk length must satisfy: `window_sec * fs >= 256`.
- `4.0 * 64 = 256`, so `WINDOW_SEC=4.0` is the minimum safe value.
- `STEP_SEC=0.5` increases temporal coverage without shrinking window size.
- `OUTPUT_MODE="medium"` reduces storage while keeping key combined files.
- `tenth` sampling limits compute cost for conversion.

### Stage 2 (`eeg_j_02.py`)

Default training config (`TrainConfig`):

- `model_name = "microsoft/cvt-13"`
- `epochs = 10`
- `batch_size = 16`
- `lr = 3e-5`
- `weight_decay = 1e-4`
- `val_ratio = 0.2` (stratified by signal label)
- `max_frames_per_signal = 16`
- `num_workers = 2`
- `seed = 42`

Why these values:

- `cvt-13` is the smallest CvT checkpoint, practical as a baseline.
- 10 epochs gives a quick first benchmark.
- Frame cap (`16`) avoids over-representing long signals and controls memory/time.
- Stratified split helps preserve class balance between train/val.
- Fixed seed improves reproducibility.

## Observed Results In Notebook Outputs

### Stage 1 (`EEG_J_01.ipynb`)

Observed logs:

- `SMM library loaded`
- `Found 18 subject files`
- Sampling: `tenth (10.0%)`
- Selected signals: `7,128` (`396 per subject x 18`)

Run status shown in notebook output:

- Conversion reached `3069/7128` (`43%`) and then stopped with `KeyboardInterrupt`.

Observed label summary from `labels.csv`:

- `attended = 1`: `1536`
- `attended = 2`: `1533`

Also shown:

- `/content/eeg_2d_outputs/labels.csv` exists in output cell (`42985` bytes in that run).

### Stage 2 (`EEG_J_02.ipynb`)

Observed logs:

- Train records: `4,912`
- Val records: `1,226`
- Train signals: `2456`
- Val signals: `613`
- Device: `cuda`
- Model: `microsoft/cvt-13`

Training progression (first and last epoch):

- Epoch 1: `train_acc=0.5063`, `val_acc=0.5065`
- Epoch 10: `train_acc=0.6610`, `val_acc=0.6085`

Final result:

- `Training complete. Best val_acc=0.6085`
- Saved to: `/content/Data/Training`

## Expected Folder Structure (Generated Data)

After Stage 1 conversion:

```text
eeg_2d_outputs/
  labels.csv
  results.csv
  SXX_TYY_CZZ/
    combined_freq.npz
    combined_topo.npz
    combined_frames_preview.png   # or combined_preview.png (depends on SMM version)
    ... other SMM files by output mode
```

After Stage 2 training:

```text
Training/
  best_model/
  best_state.pt
  history.csv
  learning_curves.png
  train_config.json
  val_predictions.csv
  val_confusion_matrix.csv
  val_confusion_matrix_normalized.csv
  val_confusion_matrix.png
  val_classification_report.csv
  val_classification_report.json
```

## How To Run

### Stage 1

Use notebook `EEG_J_01.ipynb` or script:

```python
results = run_conversion(
    data_dir="EEG_data",
    output_dir="eeg_2d_outputs",
    strategy="tenth",
)
```

### Stage 2

Use notebook `EEG_J_02.ipynb` or script:

```bash
python notebooks/eeg_j_02.py \
  --data-root /path/to/eeg_2d_outputs \
  --labels-csv /path/to/eeg_2d_outputs/labels.csv \
  --output-dir /path/to/output_root
```

## Dependencies (Main)

- Stage 1: `numpy`, `scipy`, `pandas`, `tqdm`, `soundfile`, `librosa`, `matplotlib`, SMM repo/package.
- Stage 2: `torch`, `torchvision`, `transformers`, `Pillow`, plus the above data libs.

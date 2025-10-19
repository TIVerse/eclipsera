# Eclipsera CLI Usage

Eclipsera includes a powerful command-line interface for training, predicting, and evaluating models.

## Commands

### Info

Show version and system information:

```bash
eclipsera info
# or
python -m eclipsera.cli.main info
```

Output:
```
Eclipsera version: 1.1.0
Python version: 3.13.7
Platform: Linux-6.12.52-1-lts-x86_64-with-glibc2.42

Dependencies:
  numpy: 2.3.4
  scipy: 1.16.2
  ...
```

### Train

Train a model using AutoML:

```bash
eclipsera train \
  --data training_data.npy \
  --target target_data.npy \
  --output model.pkl \
  --task classification \
  --cv 5
```

**Arguments:**
- `--data`: Training data path (.npy or .csv file)
- `--target`: Target values (column name for CSV, or .npy file path)
- `--output`: Output model path (default: model.pkl)
- `--task`: Task type - `classification` or `regression` (default: classification)
- `--cv`: Number of cross-validation folds (default: 5)

**Example with CSV:**
```bash
eclipsera train \
  --data data.csv \
  --target target_column \
  --output best_model.pkl \
  --task classification \
  --cv 10
```

**Output:**
```
Loading data from training_data.npy...
Data shape: (1000, 20), Target shape: (1000,)

Training classifier with AutoML...
AutoClassifier: Evaluating 6 algorithms...
  Trying logistic_regression... Score: 0.8524
  Trying random_forest... Score: 0.9123
  Trying gradient_boosting... Score: 0.9087
  ...

Best algorithm: random_forest (score: 0.9123)

✓ Training complete!
  Best algorithm: random_forest
  CV score: 0.9123
  Model saved to: model.pkl
```

### Predict

Make predictions with a trained model:

```bash
eclipsera predict \
  --model model.pkl \
  --data new_data.npy \
  --output predictions.npy
```

**Arguments:**
- `--model`: Trained model path (.pkl file)
- `--data`: Input data path (.npy or .csv file)
- `--output`: Output predictions path (default: predictions.npy)

**Output:**
```
Loading model from model.pkl...
Loading data from new_data.npy...
Data shape: (500, 20)
Making predictions...
Saving predictions to predictions.npy...

✓ Predictions complete!
  Predictions shape: (500,)
  Saved to: predictions.npy
```

### Evaluate

Evaluate a trained model:

```bash
eclipsera evaluate \
  --model model.pkl \
  --data test_data.npy \
  --target test_target.npy
```

**Arguments:**
- `--model`: Trained model path (.pkl file)
- `--data`: Test data path (.npy or .csv file)
- `--target`: Target values (column name for CSV, or .npy file path)

**Output:**
```
Loading model from model.pkl...
Loading data from test_data.npy...
Data shape: (200, 20), Target shape: (200,)
Evaluating model...

✓ Evaluation complete!
  Score: 0.9250
  Accuracy: 0.9250
```

## File Formats

### NumPy Arrays (.npy)

Save your data as NumPy arrays:

```python
import numpy as np

# Training data
X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 2, 1000)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
```

Then use in CLI:
```bash
eclipsera train --data X_train.npy --target y_train.npy
```

### CSV Files (.csv)

Use CSV files with column headers:

```csv
feature1,feature2,feature3,target
0.5,1.2,0.3,0
-0.2,0.8,1.1,1
...
```

Then use in CLI:
```bash
eclipsera train --data data.csv --target target
```

## Complete Workflow Example

```bash
# 1. Prepare your data
python prepare_data.py  # Creates train.npy, y_train.npy, test.npy, y_test.npy

# 2. Train with AutoML
eclipsera train \
  --data train.npy \
  --target y_train.npy \
  --output classifier.pkl \
  --task classification \
  --cv 5

# 3. Make predictions
eclipsera predict \
  --model classifier.pkl \
  --data test.npy \
  --output predictions.npy

# 4. Evaluate performance
eclipsera evaluate \
  --model classifier.pkl \
  --data test.npy \
  --target y_test.npy
```

## Advanced Usage

### Regression Task

```bash
eclipsera train \
  --data X.npy \
  --target y.npy \
  --task regression \
  --cv 10
```

### More Cross-Validation Folds

```bash
eclipsera train \
  --data data.csv \
  --target price \
  --task regression \
  --cv 10 \
  --output regressor.pkl
```

### Custom Output Names

```bash
eclipsera predict \
  --model my_classifier.pkl \
  --data new_samples.npy \
  --output my_predictions.npy
```

## Tips

1. **Use consistent file formats**: Stick with either .npy or .csv for your workflow
2. **Cross-validation**: More folds (`--cv 10`) give more reliable scores but take longer
3. **Model files**: Save your models with descriptive names (e.g., `rf_classifier_v1.pkl`)
4. **CSV support**: Requires pandas (`pip install pandas`)

## Error Handling

The CLI provides clear error messages:

```bash
$ eclipsera train --data missing.npy --target y.npy
Error during training: [Errno 2] No such file or directory: 'missing.npy'

$ eclipsera train --data data.csv --target wrong_column
Error during training: Target column 'wrong_column' not found in CSV
```

## Integration

Use the CLI in scripts:

```bash
#!/bin/bash
set -e

echo "Training model..."
eclipsera train --data data.csv --target label --output model.pkl

echo "Making predictions..."
eclipsera predict --model model.pkl --data new_data.csv --output pred.npy

echo "Done!"
```

Or check exit codes:

```bash
if eclipsera train --data X.npy --target y.npy; then
    echo "Training succeeded!"
else
    echo "Training failed!"
    exit 1
fi
```

## Python API Alternative

If you prefer Python code over CLI:

```python
# Instead of CLI
from eclipsera.automl import AutoClassifier
import numpy as np

X = np.load('X.npy')
y = np.load('y.npy')

model = AutoClassifier(cv=5, verbose=1)
model.fit(X, y)

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

---

**Need help?** Run `eclipsera <command> --help` for detailed usage information.

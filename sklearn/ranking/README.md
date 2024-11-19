
# BoostLR Algorithm

BoostLR is a supervised learning algorithm specifically designed for label ranking tasks. It enhances predictive performance by using boosting techniques, leveraging the power of Weka through Python integration with `JPype1`.

## Features

- **Boosting for Label Ranking**: Implements a boosting-based approach tailored for label ranking problems.
- **Seamless Integration**: Designed to work seamlessly with the `scikit-learn` ecosystem.
- **Highly Customizable**: Offers parameters such as the number of boosting iterations and a random seed for reproducibility.

## Requirements

- **Java Version**: The BoostLR algorithm requires **Java 8**. Make sure your `JAVA_HOME` is set to a valid Java 8 installation.
- **Python Dependencies**:
  - `JPype1`: Used to interface with Weka's Java implementation.
  - `scikit-learn`: For data preprocessing and integration.

### Additional Notes
- This implementation does **not** use `python-weka-wrapper3`.
- Proper installation and configuration of Weka are essential.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/BoostLR.git
   cd BoostLR
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that Java 8 is installed and configured:
   ```bash
   export JAVA_HOME=/path/to/java8
   export PATH=$JAVA_HOME/bin:$PATH
   ```

4. Verify Weka files are correctly placed under the `weka` and `lib` directories:
   - `lib` contains essential JAR files for Weka.
   - `weka` includes compiled class files required for BoostLR.

## Usage Example

```python
from sklearn.ranking import BoostingLRWrapper, load_dataset_as_Instances

# Start the JVM
from sklearn.ranking.utils import start_jvm
start_jvm()

# Load datasets
train_data = load_dataset_as_Instances("datasets/train_data.xarff")
test_data = load_dataset_as_Instances("datasets/test_data.xarff")

# Initialize BoostLR
model = BoostingLRWrapper(max_iterations=100, seed=42)

# Train the model
model.fit(train_data)

# Make predictions
predictions = model.predict(test_data)
print("Predicted Rankings:", predictions)

# Evaluate the model
score = model.score(test_data)
print("Model Score:", score)
```

## Parameters

- **`max_iterations`** (`int`, default=50): Maximum number of boosting iterations.
- **`seed`** (`int`): Seed for random number generator (for reproducibility).

## Methods

- **`fit(train_data)`**: Trains the model on Weka Instances.
- **`predict(test_data)`**: Predicts rankings for test data.
- **`score(test_data)`**: Evaluates the model using a ranking evaluation metric.

## API Reference

### `BoostingLRWrapper`

- **Description**: A wrapper class for the BoostLR algorithm that interacts with Weka's Java-based implementation through `JPype1`.

#### Attributes
- `max_iterations`: Maximum boosting iterations.
- `seed`: Random seed for reproducibility.

#### Methods
- **`fit(train_data)`**: Fits the model to training data.
- **`predict(test_data)`**: Predicts rankings for test data.
- **`score(test_data)`**: Evaluates the model's ranking performance.

## Dataset Format

BoostLR expects datasets in the XARFF format. Use the `load_xarff` function provided in `utils.py` to load datasets and convert them to Pandas DataFrames. Here's an example of splitting and saving datasets:

```python
from sklearn.ranking.utils import load_xarff, save_to_xarff
from sklearn.model_selection import train_test_split

# Load the dataset
df, attribute_info = load_xarff("datasets/data.xarff")

# Split into training and testing
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the splits
save_to_xarff(train_data, "datasets/train_data.xarff", relation_name="train_data", attribute_info=attribute_info)
save_to_xarff(test_data, "datasets/test_data.xarff", relation_name="test_data", attribute_info=attribute_info)
```

## Notes

1. **Java Setup**: Ensure Java 8 is installed and set as the default environment for BoostLR to function properly.
2. **Directory Structure**: Ensure the `lib` and `weka` directories contain the required Weka files.
3. **Error Handling**: If you encounter issues loading classes from Weka, ensure the classpath is correctly set when starting the JVM.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

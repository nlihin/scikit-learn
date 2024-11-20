
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
   pip install git+https://github.com/oriazadok/scikit-learn.git
   pip install JPype1
   pip install pandas
   ```

2. Ensure that Java 8 is installed and configured:
   ```bash
   export JAVA_HOME=/path/to/java8
   export PATH=$JAVA_HOME/bin:$PATH
   ```

## Usage Example

```python
from sklearn.ranking import BoostingLRWrapper
from sklearn.ranking.utils import *
from sklearn.ranking.datasets import *
from sklearn.model_selection import train_test_split

from sklearn import __path__ as sklearn_path


start_jvm()

# Locate the datasets directory in the installed package
DATASETS_DIR = os.path.join(sklearn_path[0], "ranking", "datasets")

# Specify the dataset name
dataset_name = "iris_dense.xarff"

# Construct the full path to the dataset
dataset_path = os.path.join(DATASETS_DIR, dataset_name)

# Extract the base name (without extension)
base_name = os.path.basename(dataset_path).replace(".xarff", "")

if not os.path.exists("tmp"):
    os.makedirs("tmp")

# Create the full paths and the desired string
train_base_name = f"tmp/{base_name}_train"
test_base_name = f"tmp/{base_name}_test"

train_dataset = f"{train_base_name}.xarff"
test_dataset =  f"{test_base_name}.xarff"

# Load the dataset and get attribute information
df, attribute_info = load_xarff(dataset_path)


# Split the data into training and test sets using Pandas
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the split datasets to XARFF files with the original attribute info
save_to_xarff(train_data, train_dataset, relation_name=train_base_name, attribute_info=attribute_info)
save_to_xarff(test_data, test_dataset, relation_name=test_base_name, attribute_info=attribute_info)

print("Training and test datasets saved to XARFF files.")

# Load dataset as Instances
train_data_Instances = load_dataset_as_Instances(train_dataset)
test_data_Instances = load_dataset_as_Instances(test_dataset)

# Initialize model
model = BoostingLRWrapper(max_iterations=50, seed=7, dist_algo=kendalls_tau, dist_score=ndcg)

# Train and score the model
model.fit(train_data_Instances)

predictions = model.predict(test_data_Instances)

score = model.score(test_data_Instances)

print(f"Score: {score * 100:.2f}%")

stop_jvm()

```

## Parameters

- **`max_iterations`** (`int`, default=50): Maximum number of boosting iterations.
- **`seed`** (`int`): Seed for random number generator (for reproducibility).
- **`dist_algo`** (`function`): Distance algo for trin
- **`dist_score`** (`function`): Distance algo for test

## Methods

- **`fit(train_data)`**: Trains the model on Weka Instances.
- **`predict(test_data)`**: Predicts rankings for test data.
- **`score(test_data)`**: Evaluates the model using a ranking evaluation metric.

## API Reference

### `BoostingLRWrapper`

- **Description**: A wrapper class for the BoostLR algorithm that interacts with Weka's Java-based implementation through `JPype1`.


### Methods
- **`fit(train_data)`**: Fits the model to training data.
- **`predict(test_data)`**: Predicts rankings for test data.
- **`score(test_data)`**: Evaluates the model's ranking performance.

## Dataset Format

BoostLR expects datasets in the XARFF format.

## Notes

1. **Java Setup**: Ensure Java 8 is installed and set as the default environment for BoostLR to function properly.
2. **Directory Structure**: Ensure the `lib` and `weka` directories contain the required Weka files.
3. **Error Handling**: If you encounter issues loading classes from Weka, ensure the classpath is correctly set when starting the JVM.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

# Import necessary libraries
import pandas as pd
from pycaret.classification import *

# Load the Iris dataset
from sklearn.datasets import load_iris
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target

# Initialize the PyCaret setup
classification_setup = setup(data=iris_df, target='species', session_id=42)

# Compare different models
compare_models()

# Use Random Forrest
best_model = create_model('rf')

# Tune the model
tuned_best_model = tune_model(best_model)

# Evaluate the tuned model
evaluate_model(tuned_best_model)

# Make predictions on new data
predictions = predict_model(tuned_best_model, data=iris_df)

# Finalize the model
final_model = finalize_model(tuned_best_model)

# Save the model
save_model(final_model, 'iris_rf_model')

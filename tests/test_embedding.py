import pytest
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.models import load_model
from graph_representation_learning.embedding import from_observations


@pytest.mark.parametrize(
    "model_path",
    [
        Path("embedding_models/testing_model/test.keras"),
        # Add more Path objects as needed
    ],
)
def test_from_observations(model_path):
    # Load the model
    model = load_model(str(model_path))

    # Load the metadata
    with open(Path("embedding_models/metadata.json")) as f:
        metadata = json.load(f)

    # Extract the expected input and output shapes
    model_name = model_path.parent.name
    dataset_name = model_path.stem
    input_shape = metadata["shapes"][model_name][dataset_name]["input"]
    output_shape = metadata["shapes"][model_name][dataset_name]["output"]

    # Generate a batch of observations for testing
    # The shape of the observations is (batch_size, *input_shape)
    observations = np.random.rand(10, *input_shape)

    # Call the function with the test observations
    embeddings = from_observations(observations, model_path)

    # Use the model to generate expected embeddings
    expected_embeddings = model.predict(observations)

    # Check that the output has the expected shape
    # The expected shape is (batch_size, *output_shape)
    assert expected_embeddings.shape == (10, *output_shape)

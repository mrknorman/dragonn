import random
from copy import deepcopy
import gravyflow as gf

from collections import deque

def ensure_path_to_output(graph, start_layer):
    """
    Ensure that there is a path from the start layer to the output layer.
    """
    visited = set()
    queue = deque([start_layer])

    while queue:
        current_layer = queue.popleft()
        if current_layer == 'output':
            return True
        if current_layer not in visited:
            visited.add(current_layer)
            queue.extend(graph[current_layer]["connections"])

    return False

def create_random_graph(input_layers, max_connections=3):
    """
    Create a random graph for a neural network ensuring no dead-end layers.

    :param input_layers: A list of tuples with layer names and their hyperparameters.
    :param max_connections: Maximum number of connections a layer can have.
    :return: A dictionary representing the graph architecture.
    """
    num_layers = len(input_layers)

    if num_layers < 2:
        raise ValueError("The number of layers must be at least 2 (including input and output layers).")

    # Initialize layers
    layers = ['input'] + [f'layer_{i}' for i in range(1, num_layers)] + ['output']

    # Create connections
    graph = {layer: {"parameters": hp, "connections": []} for layer, hp in zip(layers, input_layers)}
    for i, layer in enumerate(layers[:-1]):
        while True:
            # Random number of connections for each layer (at least 1)
            num_connections = random.randint(1, min(max_connections, len(layers) - i - 1))
            connections = random.sample(layers[i + 1:], num_connections)
            graph[layer]["connections"] = connections

            # Check if the current layer is part of a path to the output layer
            if ensure_path_to_output(graph, layer):
                break

    return graph

max_num_inital_layers = 10
num_examples_per_batch = 32

# Setup hyperparameters
optimizer = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CONSTANT, 
            value="adam"
        )
    )
num_layers = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=2, 
            max_=max_num_inital_layers+1, 
            dtype=int
        )
    )
batch_size = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CONSTANT, 
            value=num_examples_per_batch
        )
    )
activations = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE, 
            possible_values=['relu', 'elu', 'sigmoid', 'tanh']
        )
    )
d_units = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=1, 
            max_=128, 
            dtype=int
        )
    )
filters = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM,
            min_=1, 
            max_=128, 
            dtype=int
        )
    )
kernel_size = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=1, 
            max_=128, 
            dtype=int
        )
    )
strides = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=1, 
            max_=16, 
            dtype=int
        )
    )
learning_rate = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.LOG, 
            min_=10E-7, 
            max_=10E-3
        )
    )
pool_size = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=1, 
            max_=32, 
            dtype=int
        )
    )
pool_stride = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=1, 
            max_=32, 
            dtype=int
        )
    )
dropout_value = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM,
            min_=0, 
            max_=1
        )
    )
default_layer_type = gf.HyperParameter(
    gf.Distribution(
        type_=gf.DistributionType.CHOICE,
        possible_values=[
            gf.DenseLayer(d_units, activations),
            gf.ConvLayer(filters, kernel_size, activations, strides),
            gf.PoolLayer(pool_size, pool_stride),
            gf.DropLayer(dropout_value),
        ]
    ),
)
whiten_layer = gf.HyperParameter(
    gf.Distribution(
        type_=gf.DistributionType.CHOICE,
        possible_values=[gf.WhitenLayer()]
    ),
)

layers = [whiten_layer]
layers += [
    deepcopy(default_layer_type) for i in range(max_num_inital_layers)
]

random_graph = create_random_graph(layers)

for key, value in random_graph.items():
    print(value["connections"])
from pathlib import Path
from copy import deepcopy
import logging
import os

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Local imports:
import gravyflow as gf

def test_model(
        num_tests : int = 32
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    patience = 4
    max_epochs = 1000
    minimum_snr = 8
    maximum_snr = 15
    cache_segments = False
    ifos = [gf.IFO.L1]
    
    max_populaton : int = 100
    max_num_inital_layers : int = 10
    
    num_train_examples : int = int(1.0E5)
    num_validation_examples : int = int(1.0E4)
    
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    injection_directory_path : Path = \
        Path(current_dir / "injection_parameters")
    
    # Intilise Scaling Method:
    scaling_method : gf.ScalingMethod = gf.ScalingMethod(
        gf.Distribution(
            min_= minimum_snr,
            max_= maximum_snr,
            type_=gf.DistributionType.UNIFORM
        ),
        gf.ScalingTypes.SNR
    )

    # Load injection config:
    phenom_d_generator : gf.cuPhenomDGenerator = gf.WaveformGenerator.load(
        injection_directory_path / "baseline_phenom_d.json", 
        scaling_method=scaling_method,    
        network = None # Single detector
    )
    phenom_d_generator.injection_chance = 0.5

    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3, 
        gf.DataQuality.BEST, 
        [
            gf.DataLabel.NOISE,
            gf.DataLabel.GLITCHES
        ],
        gf.SegmentOrder.RANDOM,
        cache_segments=cache_segments,
        force_acquisition=False,
        logging_level=logging.ERROR
    )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
        data_directory_path=Path(f"{current_dir}/../../generator_data"),
        ifo_data_obtainer=ifo_data_obtainer,
        noise_type=gf.NoiseType.REAL,
        ifos=ifos
    )

    # Set requested data to be used as model input:
    input_variables = [
        gf.ReturnVariables.ONSOURCE,
        gf.ReturnVariables.OFFSOURCE
    ]
    
    # Set requested data to be used as model output:
    output_variables = [
        gf.ReturnVariables.INJECTION_MASKS
    ]

    dataset_arguments : Dict = {
        # Noise: 
        "noise_obtainer" : noise_obtainer,
        # Injections:
        "injection_generators" : phenom_d_generator, 
        # Output configuration:
        "input_variables" : input_variables,
        "output_variables": output_variables,
    }

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

    genome = gf.ModelGenome(
        optimizer=optimizer,
        num_layers=num_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        layer_genomes=layers
    )

    genome.randomize()
    genome.mutate(0.05)

    training_config = {
        "num_examples_per_epoc" : num_train_examples,
        "num_validation_examples" : num_validation_examples,
        "patience" : patience,
        "max_epochs" : max_epochs
    }
    
    population = gf.Population(
        max_populaton, 
        max_populaton, 
        genome,
        training_config,
        dataset_arguments
    )
    population.train(
        100, 
        dataset_arguments,
        num_validation_examples,
        num_examples_per_batch
    )
        
if __name__ == "__main__":

    gf.Defaults.set(
        seed=1000,
        num_examples_per_generation_batch=256,
        num_examples_per_batch=32,
        sample_rate_hertz=8196.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        crop_duration_seconds=0.5,
        scale_factor=1.0E21
    )
    
    # ---- User parameters ---- #
    # Set logging level:
    logging.basicConfig(level=logging.INFO)

    memory_to_allocate_tf = 8000    
    # Test Genetic Algorithm Optimiser:
    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf,
            gpus="5"
        ):

        test_model()
    
        os._exit(1)
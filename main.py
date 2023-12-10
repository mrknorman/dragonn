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
    max_epochs = 4
    minimum_snr = 8
    maximum_snr = 15
    cache_segments = True
    ifos = [gf.IFO.L1]
    
    max_populaton : int = 10
    max_num_inital_layers : int = 10
    
    num_train_examples : int = int(1.0E5)
    num_validation_examples : int = int(1.0E2)
    
    # Intilise gf.Scaling Method:
    scaling_method = \
        gf.ScalingMethod(
            gf.Distribution(min_=8.0,max_=15.0,type_=gf.DistributionType.UNIFORM),
            gf.ScalingTypes.SNR
        )
    
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    injection_directory_path : Path = \
        Path(current_dir / "injection_parameters")
    model_path = Path(current_dir / "./models/")

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
        "output_variables": output_variables
    }

    optimizer = gf.HyperParameter(
            {"type" : "list", "values" : ['adam']}
        )
    num_layers = gf.HyperParameter(
            {"type" : "int_range", "values" : [1, max_num_inital_layers]}
        )
    batch_size = gf.HyperParameter(
            {"type" : "list", "values" : [num_examples_per_batch]}
        )
    activations = gf.HyperParameter(
            {"type" : "list", "values" : ['relu', 'elu', 'sigmoid', 'tanh']}
        )
    d_units = gf.HyperParameter(
            {"type" : "power_2_range", "values" : [16, 256]}
        )
    filters = gf.HyperParameter(
            {"type" : "power_2_range", "values" : [16, 256]}
        )
    kernel_size = gf.HyperParameter(
            {"type" : "int_range", "values" : [1, 7]}
        )
    strides = gf.HyperParameter(
            {"type" : "int_range", "values" : [1, 7]}
        )
    learning_rate = gf.HyperParameter(
            {"type" : "log_range", "values" : [1E-7, 1E-3]}
        )

    param_limits = {
        "Dense" : gf.DenseLayer(d_units,  activations),
        "Convolutional": gf.ConvLayer(
            filters, kernel_size, activations, strides
        ),
        "Whiten" : gf.WhitenLayer()
    }

    layers = [(["Whiten"], param_limits) ]
    layers += [
        (["Dense", "Convolutional", "Pooling"], deepcopy(param_limits)) for i in range(max_num_inital_layers)
    ]

    genome_template = {
        'base' : {
            'optimizer'  : optimizer,
            'num_layers' : num_layers,
            'batch_size' : batch_size
        },
        'layers' : layers
    }

    training_config = {
        "num_examples_per_epoc" : num_train_examples,
        "num_validation_examples" : num_validation_examples,
        "patience" : patience,
        "learning_rate" : learning_rate,
        "max_epochs" : max_epochs,
        "model_path" : model_path
    }
    
    population = gf.Population(
        100, 
        100, 
        genome_template,
        training_config,
        dataset_arguments
    )
    quit()
    population.train(
        100, 
        dataset_arguments
    )
        
if __name__ == "__main__":
    
    # ---- User parameters ---- #
    # Set logging level:
    logging.basicConfig(level=logging.INFO)

    memory_to_allocate_tf = 8000    
    # Test Genetic Algorithm Optimiser:
    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf
        ):

        test_model()
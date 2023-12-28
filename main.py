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
    patience = 1
    max_epochs = 1000
    minimum_snr = 8
    maximum_snr = 15
    cache_segments = False
    ifos = [gf.IFO.L1]
    
    max_population : int = 100
    max_num_inital_layers : int = 10
    
    num_train_examples : int = int(512)
    num_validation_examples : int = int(512)
    
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

    # Load glitch config:
    wnb_generator : gf.WNBGenerator = gf.WaveformGenerator.load(
        injection_directory_path / "baseline_wnb.json", 
        scaling_method=scaling_method,    
        network = None # Single detector
    )
    wnb_generator.injection_chance = 0.5

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
        "injection_generators" : [phenom_d_generator, wnb_generator], 
        # Output configuration:
        "input_variables" : input_variables,
        "output_variables": output_variables,
    }

    # Setup hyperparameters

    # Training genes:
    optimizer = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.CONSTANT, 
                value="adam"
            )
        )
    batch_size = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.POW_TWO, 
                min_=16,
                max_=128,
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
    
    # Injection genes:
    injection_generators = [gf.HyperInjectionGenerator(
        min_ = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=0, 
                max_=100
            ) 
        ),
        max_ = gf.HyperParameter(
            gf.Distribution(
                    type_=gf.DistributionType.UNIFORM, 
                    min_=0, 
                    max_=100
                )
        ),
        mean = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=0, 
                max_=100
            )
        ),
        std = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=0, 
                max_=100
            )
        ),
        distribution = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.CHOICE, 
                possible_values=[
                    gf.DistributionType.UNIFORM,
                    gf.DistributionType.LOG,
                    gf.DistributionType.NORMAL
                ]
            )
        ),
        chance = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=0, 
                max_=1
            )
        ),
        generator = gf.HyperParameter(
                gf.Distribution(
                    type_=gf.DistributionType.CONSTANT, 
                    value = gen
            )
        )
    ) for gen in [phenom_d_generator, wnb_generator] ]

    # Noise genes:
    noise_type = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE, 
            possible_values=[
                gf.NoiseType.WHITE,
                gf.NoiseType.COLORED,
                gf.NoiseType.PSEUDO_REAL,
                gf.NoiseType.REAL,
            ],
        )
    )
    exclude_real_glitches = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE, 
            possible_values=[
                True,
                False
            ]
        )
    )

    # Temporal Genes:
    onsource_duration_seconds = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=0.2,
            max_=4.0
        )
    )
    offsource_duration_seconds = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=1.0,
            max_=32.0
        )
    )
    sample_rate_hertz = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.POW_TWO, 
            min_=512,
            max_=8196,
            dtype=int
        )
    )

    #Feature engineering layers:

    num_layers = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=2, 
                max_=max_num_inital_layers+1, 
                dtype=int
            )
        )
    activations = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.CHOICE, 
                possible_values=[
                    'relu', 
                    'elu', 
                    'sigmoid', 
                    'tanh', 
                    'selu', 
                    'gelu',
                    'swish',
                    'softmax'
                ]
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
                max_=128, 
                dtype=int
            )
        )
    dilation = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=0, 
                max_=64, 
                dtype=int
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
                gf.ConvLayer(filters, kernel_size, activations, strides, dilation),
                gf.PoolLayer(pool_size, pool_stride),
                gf.DropLayer(dropout_value),
            ]
        ),
    )
    whiten_layer = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE,
            possible_values=[gf.WhitenLayer(), gf.WhitenPassLayer()]
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
        injection_generators=injection_generators,
        noise_type=noise_type,
        exclude_glitches=exclude_real_glitches,
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        sample_rate_hertz=sample_rate_hertz,
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
        max_population, 
        max_population, 
        genome,
        training_config,
        deepcopy(dataset_arguments)
    )
    population.train(
        100, 
        deepcopy(dataset_arguments),
        num_validation_examples,
        num_examples_per_batch
    )
        
if __name__ == "__main__":

    gf.Defaults.set(
        seed=1000,
        num_examples_per_generation_batch=256,
        num_examples_per_batch=32,
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        crop_duration_seconds=0.5,
        scale_factor=1.0E21
    )
    
    # ---- User parameters ---- #
    # Set logging level:
    logging.basicConfig(level=logging.INFO)

    memory_to_allocate_tf = 5000    
    # Test Genetic Algorithm Optimiser:
    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf,
            gpus="6"
        ):

        test_model()
    
        os._exit(1)
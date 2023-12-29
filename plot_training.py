from pathlib import Path
import os
import sys
from itertools import cycle

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.palettes import Category20, interp_palette  # A palette with 20 distinct colors
from bokeh.embed import components, file_html
from bokeh.io import export_png, output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import (ColumnDataSource, CustomJS, Dropdown, HoverTool, 
                          Legend, LogAxis, LogTicker, Range1d, Slider, Select,
                         Div)
from bokeh.plotting import figure, show
from bokeh.resources import INLINE, Resources
from bokeh.palettes import Bright

# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

import numpy as np
import gravyflow as gf

def snake_case_to_capitalised_first_with_spaces(text):
    """
    Convert a string from snake_case to Capitalised First With Spaces format.
    """

    text.replace('val', 'validate') if 'val' in text else text
    # Split the string at underscores
    words = text.split('_')
    
    # Capitalise the first letter of each word and join them with spaces
    return ' '.join(word.capitalize() for word in words)

def transform_string(s):
    # Remove the 'perceptron_' prefix and split by underscore
    name = s.replace('model_', '')

    return snake_case_to_capitalised_first_with_spaces(name)

def plot_metrics(metrics_dict):
    plot_widths = [1200] * 4  # Default width for each plot

    # Create a figure for each metric
    figures = []

    keys = ['binary_accuracy', 'loss', 'val_binary_accuracy', 'val_loss']
    labels = [
        "Accuracy: Training Data (Per Cent)", 
        "Loss: Training Data (Per Cent)", 
        "Accuracy: Validation Data (Per Cent)", 
        "Loss: Validation Data (Per Cent)"
    ]

    for i, (key, label) in enumerate(zip(keys, labels)):
        # Use specified width for each plot, or default if not enough widths are specified
        width = plot_widths[i] if i < len(plot_widths) else plot_widths[-1]

        
        p = figure(x_axis_label='Epochs', y_axis_label=label, width=width)
        
        if "accuracy" in key:
            p.y_range.start = 45
            p.y_range.end = 100
        else: 
            p.y_range.start = 0.4
            p.y_range.end = 1.0

        # Increase font sizes
        p.axis.axis_label_text_font_size = "14pt"  # Increase axis label font size
        p.axis.major_label_text_font_size = "12pt"  # Increase tick label font size

        # If you have titles
        p.title.text_font_size = '16pt'

        all_sources = {}

        # Plot each series in the list
        for idx, (name, metrics) in enumerate(metrics_dict.items()):
            if key in metrics:
                color = interp_palette(Category20[20], len(metrics_dict.items()))[idx]

                y = metrics[key]
                if "accuracy" in key:
                    y = [100*i for i in y]

                source = ColumnDataSource(
                        data={"x": np.arange(len(metrics[key])), "y" : y, "name" : [name] * len(metrics[key])}
                )
                all_sources[name] = source
                line = p.line(
                    x='x', 
                    y='y', 
                    source=source, 
                    line_color=color,
                    legend_label=name
                )

         # If you have a legend
        p.legend.location = "top_right"
        p.legend.label_text_font_size = "12pt"
        p.legend.click_policy = "hide"
        
        hover = HoverTool()
        if "accuracy" in key:
            hover.tooltips = [("Name", "@name"), ("Epoch", "@x"), ("Accuracy", "@y")]
        else:
            hover.tooltips = [("Name", "@name"), ("Epoch", "@x"), ("Loss", "@y")]

        p.add_tools(hover)
        
        figures.append(p)

    # Layout the plots in a grid
    grid = gridplot(figures, ncols=2)

    # Output file
    output_file(f"{current_dir}/population/training_plot.html")
    save(grid)

histories = {}

directory_path = Path(f'{current_dir}/population/')
for entry in directory_path.iterdir():
    if entry.name.startswith("model_"):
        name = transform_string(entry.name)

        histories[name] = gf.load_history(entry)

plot_metrics(histories)
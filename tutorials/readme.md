# README

This folder is used for ipynb files which will be generated as tutorials in the site.

Only code cells in each notebook that start with `# unittest` will also be run as unit
tests.

## Interactive plotting in tutorials

### Quick instructions

- Ensure your notebook is fully executed.
- Save and close the notebook.
- Strip metadata out of your notebook.
  ```bash
  jupyter nbconvert --to notebook --inplace --ClearMetadataPreprocessor.enabled=True Name_of_notebook.ipynb`
  ```
- Execute `make tutorials` in the top level directory of Bean Machine.
- `cd website` and run `yarn` then `yarn build` finally `yarn start` and navigate
  to `localhost:3000`.

### Full instructions

You can create interactive plots in a notebook using either Bokeh or Plotly. If you use
Bokeh for creating a plot, then the script `scripts/convert_ipynb_to_mdx.py` expects the
last line of the cell executing the plot to be `show(NAME_OF_PLOT)`. Below is a simple
self contained example showing a sine wave that has tooltip interactivity using Bokeh.

```python
import numpy as np
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show
output_notebook()
N = 1000
x = np.linspace(0, 2 * np.pi, N)
y = np.sin(x)
cds = ColumnDataSource({"x": x, "y": y})
sine_wave = figure()
glyph = sine_wave.line(x="x", y="y", source=cds, line_color="steelblue")
tips = HoverTool(renderers=[glyph], tooltips=[("x", "@x{0.000}"), ("y", "@y{0.000}")])
sine_wave.add_tools(tips)
show(sine_wave)
```

The script will take the last line of the above code, search for the token `show(`, and
use the name of the variable in `show` for: 1) the name of the React component for
display in the mdx file and 2) as the name of the JSON output for the Bokeh figure. This
means we will have a JSON file with the name `sine-wave.json` and a React component
called `SineWave`. It is important to note that if you overwrite the name of a figure
further down in your notebook, then the first figure will be lost when converting the
notebook to an mdx file. Thus you will need to **ensure you name all Bokeh figures
uniquely**, otherwise you will not get an mdx file with all the figures you created in
your notebook.

The output for both Bokeh and Plotly are embedded in the notebook as long, as the cell
has been executed. The script uses this fact to parse the HTML/JavaScript produced by
the tools so it can save the data used for the plot to a JSON file. These JSON files are
later read into memory by React from the mdx file, and rendered using the React
components found in `website/src/components/Plotting.jsx`.

If you use Plotly to generate an interactive figure, then you do not have to pay
attention to naming it something unique. The script will use a UUID to generate both the
name for the JSON file and the React component. The file names and component names are
not as easy to read as the Bokeh figures, since they do not contain semantic information
about the plot, but you also do not have to worry about creating a unique name for the
plot if you use Plotly.

### Creating mdx files

Follow these step when you are ready to create mdx files from an ipynb tutorial. The
script uses `website/tutorials.json` as a catalog of tutorials it should convert into
mdx files. If you do not update this file, then the script will not build your new
tutorial. The structure for adding a new tutorial that needs to be built looks like the
following.

```json
"Name_of_notebook": {
  "title": "Example sine wave",
  "sidebar_label": "Sine wave",
  "path": "overview/tutorials/Name_of_notebook",
  "nb_path": "tutorials/Name_of_notebook.ipynb",
  "github": "https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Name_of_notebook.ipynb",
  "colab": "https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Name_of_notebook.ipynb"
}
```

Do not leave out the "sidebar_label" key/value pair and do not overwrite an already
existing "sidebar_label" key/value pair. No checks are done to ensure you do not
overwrite a name that already exists, so make sure you look at the JSON file to ensure
your name for the new tutorial has not been taken already. Without a proper sidebar
label, the documentation will fail to build. The information in the JSON file is used in
the frontmatter of the mdx file, and for creating links to GitHub and Colab.

After updating `website/tutorials.json`, navigate to the top level directory of Bean
Machine and execute the following.

```bash
make tutorials
```

The output of the above command will look like the following.

```bash
python scripts/convert_ipynb_to_mdx.py
--------------------------------------------
Converting tutorial notebooks into mdx files
--------------------------------------------
Coin_flipping
Hierarchical_regression
Name_of_notebook
```

The script will use the catalog found in `website/tutorials.json` to create a new
tutorial in `docs/overview/tutorials/` with the following structure.

```bash
tutorials
├── Coin_flipping
├── Hierarchical_regression
└── Name_of_notebook
    ├── assets
    │   ├── img
    │   └── plot_data
    │       ├── PlotlyFigure1211d834-9867-43c2-9bd8-e8b08e0199fe.json
    │       └── sine-wave.json
    ├── NameOfNotebook.jsx
    └── NameOfNotebook.mdx
```

Image assets in the notebook will be placed in the `img` folder, and JSON plot data will
be placed in the `plot_data` folder both of which are located in the `assets` folder.
The script will automatically generate the `NameOfNotebook.{jsx|mdx}` files and place
any React components needed for the mdx file in the jsx file. All React components
needed for executing the mdx file correctly will be imported just below the frontmatter
of the file.

If you have matplotlib images in your tutorial, then the script will base64 encode the
png output and embed it in the mdx file. The script does not know how to handle
ipywidgets at this time, so if you have need for them, please make an issue so the
script can be updated for your use case.

After the tutorials have been created, navigate to `website` and build it using
`yarn build`. After the build is complete, you can see how your new tutorial rendered by
running `yarn start` and navigating to `localhost:3000`.

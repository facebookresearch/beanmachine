# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import shutil
import uuid
from os import PathLike
from pathlib import Path
from textwrap import wrap
from typing import Dict, Tuple, Union

import nbformat
import pandas as pd
from nbformat.notebooknode import NotebookNode

try:
    from libfb.py.fbcode_root import get_fbcode_dir
except ImportError:
    SCRIPTS_DIR = Path(__file__).parent.resolve()
    LIB_DIR = SCRIPTS_DIR.parent.parent.resolve()
else:
    LIB_DIR = (Path(get_fbcode_dir()) / "beanmachine").resolve()

WEBSITE_DIR = LIB_DIR.joinpath("website")
DOCS_DIR = LIB_DIR.joinpath("docs")
OVERVIEW_DIR = DOCS_DIR.joinpath("overview")
TUTORIALS_DIR = OVERVIEW_DIR.joinpath("tutorials")


def load_nbs_to_convert() -> Dict[str, Dict[str, str]]:
    """Load the metadata and list of notebooks to convert to mdx.

    Args:
        None

    Returns:
        Dict[str, Dict[str, str]]: A dictionary of metadata needed to convert notebooks
        to mdx.

    """
    tutorials_json_path = WEBSITE_DIR.joinpath("tutorials.json")
    with open(str(tutorials_json_path), "r") as f:
        tutorials_data = json.load(f)

    return tutorials_data


def load_notebook(path: Union[PathLike, str]) -> NotebookNode:
    """Load the given notebook into memory.

    Args:
        path (Union[PathLike, str]): Path to the notebook.

    Returns:
        NotebookNode: `nbformat` object.

    """
    if isinstance(path, PathLike):
        path = str(path)
    with open(path, "r") as f:
        nb_str = f.read()
        nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)

    return nb


def transform_markdown_cell(
    cell: NotebookNode,
    img_folder: Union[PathLike, str],
) -> str:
    """Transform the given Jupyter markdown cell.

    Args:
        cell (NotebookNode): Jupyter markdown cell object.
        img_folder (Union[PathLike, str]): Image folder path.

    Returns:
        str: Transformed markdown object suitable for inclusion in mdx files.

    """
    img_folder = Path(img_folder)
    cell_source = cell["source"]

    # Check if the cell is displaying an image.
    if cell_source[0] == "!":
        # Change the path to always be `assets/img/...`
        start = cell_source.find("(") + 1
        stop = cell_source.find(")")
        old_img_path = (LIB_DIR / "tutorials" / Path(cell_source[start:stop])).resolve()
        name = old_img_path.name
        img_path_str = f"assets/img/{name}"
        cell_source = cell_source[:start] + img_path_str + cell_source[stop:]
        # Copy the image to the folder where the markdown can access it.
        new_img_path = str(img_folder.joinpath(name))
        shutil.copy(str(old_img_path), new_img_path)

    # Wrap lines using black's default of 88 characters. Used to help debug issues from
    # the tutorials. Note that all tables (lines that start with |) are not wrapped as
    # well as any LaTeX formulas that start new block math sections ($$).
    cell_source = cell_source.splitlines()
    math_lines = [i for i, line in enumerate(cell_source) if line.startswith("$$")]
    math_lines = [(i, j) for i, j in zip(math_lines, math_lines[1:])]
    math_line_check = [False] * len(cell_source)
    for start, stop in math_lines:
        math_line_check[start : stop + 1] = [True] * (stop + 1 - start)
    new_cell_source = ""
    for i, line in enumerate(cell_source):
        if math_line_check[i] or line.startswith("|"):
            new_cell_source += f"{line}\n"
        elif not math_line_check[i]:
            md = "\n".join(wrap(line, width=88, replace_whitespace=False))
            new_cell_source += f"{md}\n"

    return f"{new_cell_source}\n\n"


def transform_code_cell(  # noqa: C901 (flake8 too complex)
    cell: NotebookNode,
    plot_data_folder: Union[PathLike, str],
    filename: Union[PathLike, str],
) -> Dict[str, Union[str, bool]]:
    """Transform the given Jupyter code cell.

    Args:
        cell (NotebookNode): Jupyter code cell object.
        plot_data_folder (Union[PathLike, str]): Path to the `plot_data` folder for the
            tutorial.
        filename (str): File name to use for the mdx and jsx output.

    Returns:
        Tuple[str, str]: First object is for mdx inclusion, and the second is for jsx if
        a bokeh plot was found.

    """
    plot_data_folder = Path(plot_data_folder).resolve()
    # Data display priority.
    priorities = [
        "text/markdown",
        "application/javascript",
        "image/png",
        "image/jpeg",
        "image/svg+xml",
        "image/gif",
        "image/bmp",
        "text/latex",
        "text/html",
        "application/vnd.jupyter.widget-view+json",  # tqdm progress bars
        "text/plain",
    ]

    bokeh_flag = False
    plotly_flag = False

    mdx_output = ""
    jsx_output = ""
    link_btn = "../../../../website/src/components/LinkButtons.jsx"
    cell_out = "../../../../website/src/components/CellOutput.jsx"
    plot_out = "../../../../website/src/components/Plotting.jsx"
    components_output = f'import LinkButtons from "{link_btn}";\n'
    components_output += f'import CellOutput from "{cell_out}";\n'

    # Handle cell input.
    cell_source = cell.get("source", "")
    mdx_output += f"```python\n{cell_source}\n```\n\n"

    # Handle cell outputs.
    cell_outputs = cell.get("outputs", [])
    if cell_outputs:
        # Create a list of all the data types in the outputs of the cell. These values
        # are similar to the ones in the priorities variable.
        cell_output_dtypes = [
            list(cell_output.get("data", {}).keys()) for cell_output in cell_outputs
        ]

        # Order the output of the cell's data types using the priorities list.
        ordered_cell_output_dtypes = [
            sorted(
                set(dtypes).intersection(set(priorities)),
                key=lambda dtype: priorities.index(dtype),
            )
            for dtypes in cell_output_dtypes
        ]

        # Create a list of the cell output types. We will handle each one differently
        # for inclusion in the mdx string. Types include:
        # - "display_data"
        # - "execute_result"
        # - "stream"
        # - "error"
        cell_output_types = [cell_output["output_type"] for cell_output in cell_outputs]

        # We handle bokeh and plotly figures differently, so check to see if the output
        # contains on of these plot types.
        if "plotly" in str(cell_output_dtypes):
            plotly_flag = True
        if "bokeh" in str(cell_output_dtypes):
            bokeh_flag = True

        # Cycle through the cell outputs and transform them for inclusion in the mdx
        # string.
        display_data_outputs = []
        for i, cell_output in enumerate(cell_outputs):
            data_object = (
                ordered_cell_output_dtypes[i][0]
                if ordered_cell_output_dtypes[i]
                # Handle "stream" cell output type.
                else "text/plain"
            )
            data_category, data_type = data_object.split("/")
            cell_output_data = cell_output.get("data", {}).get(data_object, "")
            cell_output_type = cell_output_types[i]

            # Handle "display_data".
            if cell_output_type == "display_data":
                if not bokeh_flag and not plotly_flag:
                    # Handle binary images.
                    if data_category == "image":
                        if data_type in ["png", "jpeg", "gif", "bmp"]:
                            mdx_output += (
                                f"![](data:{data_object};base64,{cell_output_data})\n\n"
                            )
                    # TODO: Handle svg images.

                    # Handle tqdm progress bars.
                    if data_type == "vnd.jupyter.widget-view+json":
                        cell_output_data = cell_output["data"]["text/plain"]
                        display_data_outputs.append(cell_output_data)

                # Handle plotly images.
                if plotly_flag:
                    components_output += f'import {{PlotlyFigure}} from "{plot_out}";\n'
                    cell_output_data = cell_output["data"]
                    for key, value in cell_output_data.items():
                        if key == "application/vnd.plotly.v1+json":
                            # Save the plotly JSON data.
                            file_name = "PlotlyFigure" + str(uuid.uuid4())
                            file_path = str(
                                plot_data_folder.joinpath(f"{file_name}.json")
                            )
                            with open(file_path, "w") as f:
                                json.dump(value, f, indent=2)

                            # Add the Plotly figure to the MDX output.
                            path_to_data = f"./assets/plot_data/{file_name}.json"
                            mdx_output += (
                                f"<PlotlyFigure data={{require('{path_to_data}')}} "
                                "/>\n\n"
                            )

                # Handle bokeh images.
                if bokeh_flag:
                    components_output += f'import {{BokehFigure}} from "{plot_out}";\n'
                    # Ignore any HTML data objects. The bokeh object we want is a
                    # `application/javascript` object. We will also ignore the first
                    # bokeh output, which is an image indicating that bokeh is loading.
                    bokeh_ignore = (
                        data_object == "text/html"
                        or "HTML_MIME_TYPE" in cell_output_data
                    )
                    if bokeh_ignore:
                        continue
                    if data_object == "application/javascript":
                        # Parse the cell source to create a name for the component. This
                        # will be used as the id for the div as well as it being added
                        # to the JSON data.
                        plot_name = cell_source.split("\n")[-1]
                        token = "show("
                        plot_name = plot_name[plot_name.find(token) + len(token) : -1]
                        div_name = plot_name.replace("_", "-")
                        # Parse the javascript for the bokeh JSON data.
                        flag = "const docs_json = "
                        json_string = list(
                            filter(
                                lambda line: line.startswith(flag),
                                [
                                    line.strip()
                                    for line in cell_output_data.splitlines()
                                ],
                            )
                        )[0]
                        # Ignore the const definition and the ending ; from the line.
                        json_string = json_string[len(flag) : -1]
                        json_data = json.loads(json_string)
                        # The js from bokeh in the notebook is nested in a single key,
                        # hence the reason why we do this.
                        json_data = json_data[list(json_data.keys())[0]]
                        js = {}
                        js["target_id"] = div_name
                        js["root_id"] = json_data["roots"]["root_ids"][0]
                        js["doc"] = {
                            "defs": json_data["defs"],
                            "roots": json_data["roots"],
                            "title": json_data["title"],
                            "version": json_data["version"],
                        }
                        js["version"] = json_data["version"]
                        # Save the bokeh JSON data.
                        file_path = str(plot_data_folder.joinpath(f"{div_name}.json"))
                        with open(file_path, "w") as f:
                            json.dump(js, f, indent=2)

                            # Add the Bokeh figure to the MDX output.
                        path_to_data = f"./assets/plot_data/{div_name}.json"
                        mdx_output += (
                            f"<BokehFigure data={{require('{path_to_data}')}} />\n\n"
                        )

            # Handle "execute_result".
            if cell_output_type == "execute_result":
                if data_category == "text":
                    # Handle HTML.
                    if data_type == "html":
                        # Handle pandas DataFrames. There is a scoped style tag in the
                        # DataFrame output that uses the class name `dataframe` to style
                        # the output. We will use this token to determine if a pandas
                        # DataFrame is being displayed.
                        if "dataframe" in cell_output_data:
                            df = pd.read_html(cell_output_data, flavor="lxml")
                            # NOTE: The return is a list of dataframes and we only care
                            #       about the first one.
                            md_df = df[0]
                            for column in md_df.columns:
                                if column.startswith("Unnamed"):
                                    md_df.rename(columns={column: ""}, inplace=True)
                            # Remove the index if it is just a range, and output to
                            # markdown.
                            md = ""
                            if isinstance(md_df.index, pd.RangeIndex):
                                md = md_df.to_markdown(showindex=False)
                            elif not isinstance(md_df.index, pd.RangeIndex):
                                md = md_df.to_markdown()
                            mdx_output += f"\n{md}\n\n"

                    # Handle plain text.
                    if data_type == "plain":
                        cell_output_data = "\n".join(
                            [line for line in cell_output_data.splitlines() if line]
                        )
                        display_data_outputs.append(cell_output_data)
                    # Handle markdown.
                    if data_type == "markdown":
                        mdx_output += f"{cell_output_data}\n\n"

            # Handle "stream".
            if cell_output_type == "stream":
                # Ignore if the output is an error.
                if cell_output["name"] == "stderr":
                    continue
                cell_output_data = cell_output.get("text", None)
                if cell_output_data is not None:
                    cell_output_data = "\n".join(
                        [line for line in cell_output_data.splitlines() if line]
                    )
                    display_data_outputs.append(cell_output_data)

        if display_data_outputs:
            cell_output_data = "\n".join(display_data_outputs)
            mdx_output += f"<CellOutput>\n{{`{cell_output_data}`}}\n</CellOutput>\n\n"

    return {
        "mdx": mdx_output,
        "jsx": jsx_output,
        "components": components_output,
        "bokeh": bokeh_flag,
        "plotly": plotly_flag,
    }


def find_frontmatter_ending(mdx: str, stop_looking_after: int = 10) -> int:
    """Find the line number where the mdx frontmatter ends.

    Args:
        mdx (str): String representation of the mdx file.
        stop_looking_after (int): Optional, default is 10. Number of lines to stop
        looking for the end of the frontmatter.

    Returns:
        int: The next line where the frontmatter ending is found.

    Raises:
        IndexError: No markdown frontmatter was found.

    """
    indices = []
    still_looking = 0
    lines = mdx.splitlines()
    for i, line in enumerate(lines):
        still_looking += 1
        if still_looking >= stop_looking_after:
            break
        if line == "---":
            indices.append(i)
            still_looking = 0
        if i == len(line) - 1:
            break

    if not indices:
        msg = "No markdown frontmatter found in the tutorial."
        raise IndexError(msg)

    return max(indices) + 1


def transform_notebook(path: Union[str, PathLike]) -> Tuple[str, str]:
    """Transform the given Jupyter notebook into strings suitable for mdx and jsx files.

    Args:
        path (Union[str, PathLike]): Path to the Jupyter notebook.

    Returns:
        Tuple[str, str]: mdx string, jsx string

    """
    # Ensure the given path is a pathlib.PosixPath object.
    path = Path(path).resolve()

    # Load all metadata for notebooks that should be included in the documentation.
    nb_metadata = load_nbs_to_convert()

    # Create the assets folder for the given tutorial.
    tutorial_folder_name = path.stem
    filename = "".join([token.title() for token in tutorial_folder_name.split("_")])
    tutorial_folder = TUTORIALS_DIR.joinpath(tutorial_folder_name)
    assets_folder = tutorial_folder.joinpath("assets")
    img_folder = assets_folder.joinpath("img")
    plot_data_folder = assets_folder.joinpath("plot_data")
    if not tutorial_folder.exists():
        tutorial_folder.mkdir(parents=True, exist_ok=True)
    if not img_folder.exists():
        img_folder.mkdir(parents=True, exist_ok=True)
    if not plot_data_folder.exists():
        plot_data_folder.mkdir(parents=True, exist_ok=True)

    # Load the notebook.
    nb = load_notebook(path)

    # Begin to build the mdx string.
    mdx = ""
    # Add the frontmatter to the mdx string. This is the part between the `---` lines
    # that define the tutorial sidebar_label information.
    frontmatter = "\n".join(
        ["---"]
        + [
            f"{key}: {value}"
            for key, value in nb_metadata.get(
                tutorial_folder_name,
                {
                    "title": "",
                    "sidebar_label": "",
                    "path": "",
                    "nb_path": "",
                    "github": "",
                    "colab": "",
                },
            ).items()
        ]
        + ["---"]
    )
    frontmatter_line = len(frontmatter.splitlines())
    mdx += f"{frontmatter}\n"

    # Create the JSX and components strings.
    jsx = ""
    components = set()

    # Cycle through each cell in the notebook.
    bokeh_flags = []
    plotly_flags = []
    for cell in nb["cells"]:
        cell_type = cell["cell_type"]

        # Handle markdown cell objects.
        if cell_type == "markdown":
            mdx += transform_markdown_cell(cell, img_folder)

        # Handle code cell objects.
        if cell_type == "code":
            tx = transform_code_cell(cell, plot_data_folder, filename)
            mdx += str(tx["mdx"])
            jsx += str(tx["jsx"])
            bokeh_flags.append(tx["bokeh"])
            plotly_flags.append(tx["plotly"])
            for component in str(tx["components"]).splitlines():
                components.add(component)

    # Add the JSX template object to the jsx string. Only include the plotting component
    # that is needed.
    bokeh_flag = any(bokeh_flags)
    plotly_flag = any(plotly_flags)
    plotting_fp = "../../../../website/src/components/Plotting.jsx"
    JSX_TEMPLATE = ["import React from 'react';"]
    if bokeh_flag:
        JSX_TEMPLATE.append(f"import {{ BokehFigure }} from '{plotting_fp}';")
    if plotly_flag:
        JSX_TEMPLATE.append(f"import {{ PlotlyFigure }} from '{plotting_fp}';")

    jsx = "\n".join([item for item in JSX_TEMPLATE if item]) + "\n\n" + jsx
    # Remove the last line since it is blank.
    jsx = "\n".join(jsx.splitlines()[:-1])

    # Add the react components needed to display bokeh objects in the mdx string.
    mdx = mdx.splitlines()
    mdx[frontmatter_line:frontmatter_line] = list(components) + [""]
    # Add the react components needed to display links to GitHub and Colab.
    idx = frontmatter_line + len(components) + 1
    glk = nb_metadata[tutorial_folder_name]["github"]
    clk = nb_metadata[tutorial_folder_name]["colab"]
    mdx[idx:idx] = (
        f'<LinkButtons\n  githubUrl="{glk}"\n  colabUrl="{clk}"\n/>\n\n'
    ).splitlines()
    mdx = "\n".join(mdx)

    # Write the mdx file to disk.
    mdx_filename = str(tutorial_folder.joinpath(f"{filename}.mdx"))
    with open(mdx_filename, "w") as f:
        f.write(mdx)

    # Write the jsx file to disk.
    jsx_filename = str(tutorial_folder.joinpath(f"{filename}.jsx"))
    with open(jsx_filename, "w") as f:
        f.write(jsx)

    # Return the mdx and jsx strings for debugging purposes.
    return mdx, jsx


if __name__ == "__main__":
    tutorials_metadata = load_nbs_to_convert()
    print("--------------------------------------------")
    print("Converting tutorial notebooks into mdx files")
    print("--------------------------------------------")
    for _, value in tutorials_metadata.items():
        path = (LIB_DIR / value["nb_path"]).resolve()
        print(f"{path.stem}")
        mdx, jsx = transform_notebook(path)
    print("")

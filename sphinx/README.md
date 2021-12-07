# Sphinx
The API is extracted from the Python docstrings and converted to HTML with [Sphinx](https://www.sphinx-doc.org/en/master/).

### Configuration
Bean Machine uses a [custom script](https://github.com/facebookresearch/beanmachine/blob/master/sphinx/source/docs.py) to autogenerate the `.rst` files for Sphinx. It is run inside [`conf.py`](https://github.com/stefanwebb/beanmachine/blob/master/sphinx/source/conf.py) and so is hidden from the end-user.

A configuration file, [`configuration.toml`](https://github.com/facebookresearch/beanmachine/blob/master/website/documentation.toml) specifies which symbols (modules, classes, members, attributes, functions, and variables) to include and exclude from the Sphinx output based on filtering by Python regular expressions.

### Build
Run Sphinx like so:
```
$ make html
```
Then, the docs can be viewed under `website/static/api`.

### Docstrings
By convention, Bean Machine docstrings are written in the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and we treat Sphinx warnings as errors.

Some common mistakes to avoid when writing docstrings are:

* Use a double colon :: rather than a single one before starting a code example or another block that you want treated as a text literal
* Put a non-text symbol after ticked sections. E.g. \`\`param\`\`s will error, but \`\`params\`\`'s is fine
* If breaking e.g. a parameter description across lines, indent the lines after the first one

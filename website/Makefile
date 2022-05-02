SHELL := /bin/bash

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = ./sphinx
MODULEDIR     = ../src/beanmachine
MODULEIGNOREDIRS     = ../src/beanmachine/{applications,graph,tutorials}* ../src/beanmachine/ppl/{conftest.py,diagnostics,examples,experimental,inference/utils,legacy,testlib,utils}
BUILDDIR      = ./static
ALLSPHINXOPTS = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) $(SOURCEDIR)


apidocs:
	sphinx-apidoc -e -o $(SOURCEDIR) $(MODULEDIR) $(MODULEIGNOREDIRS)

apihtml: apidocs
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/api

clean:
	rm -rf build/ $(SOURCEDIR)/beanmachine*.rst

all: clean apidocs apihtml
	yarn build

prebuild:
	# Only run prebuild script with Buck internally
	if [ -f "../TARGETS" ]; then \
		buck run //beanmachine/scripts:ipynb-to-mdx; \
	fi

FORCE:

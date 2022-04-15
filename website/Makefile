# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= 
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = sphinx
MODULEDIR     = ../src/beanmachine
MODULEIGNOREDIRS     = ../src/beanmachine/{applications,graph,tutorials}* ../src/beanmachine/ppl/{compiler,diagnostic,examples,experimental,inference/utils,legacy,testlib,utils}
BUILDDIR      = ./static
ALLSPHINXOPTS = -q -d $(BUILDDIR)/doctrees $(SPHINXOPTS) $(SOURCEDIR)

# NOTE: Since we cannot ignore the no-duplicates warning specifically
# we use a Bash hack for the same purpose. Note, ! is to invert error
# status of grep
BASHHACK	  = 2>&1 >/dev/null | grep -v -E "(use :noindex:)|(more than one target found)|(warnings.warn)|(experimental)"

apidocs:
	sphinx-apidoc -e -o $(SOURCEDIR) $(MODULEDIR) $(MODULEIGNOREDIRS)

apihtml: apidocs
	-$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/api $(BASHHACK)

clean:
	rm -rf build/ $(SOURCEDIR)/beanmachine*.rst $(SOURCEDIR)/modules.rst

all: clean apidocs apihtml
	yarn build

FORCE:

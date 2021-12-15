all: lint docs test

install: FORCE
	pip install -e .[dev]

lint: FORCE
	flake8 . && usort check . &&  black --check .

test: lint FORCE
	pytest -vs .

docs: FORCE
	$(MAKE) -C sphinx html

tutorials: FORCE
	python scripts/convert_ipynb_to_mdx.py; wait; cd website; wait; yarn; wait; yarn build;

format-notebook: FORCE
ifndef nb
	find tutorials -name "*.ipynb" | xargs jupyter nbconvert --to notebook \
	--inplace --ClearMetadataPreprocessor.enabled=True
else
	jupyter nbconvert --to notebook --inplace \
		--ClearMetadataPreprocessor.enabled=True $(nb)
endif

website: docs FORCE
	$(MAKE) -C website 

FORCE:

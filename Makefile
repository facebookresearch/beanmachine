all: lint docs test

install: FORCE
	pip install -e .[dev]

lint: FORCE
	flake8 . && usort check . &&  black --check .

test: lint FORCE
	pytest -vs .

docs: FORCE
	$(MAKE) -C sphinx html

FORCE:

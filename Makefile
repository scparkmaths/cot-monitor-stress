PYTHON := uv run python

.PHONY: test lint rd rewrite validate features structure length content combined scores analyze full reproduce

test:
	uv run pytest -q

lint:
	uv run ruff check .

rd:
	$(PYTHON) -m cms.monitors.actions_only

rewrite:
	$(PYTHON) -m cms.attack.obfuscator

validate:
	$(PYTHON) -m cms.attack.validate

features:
	$(PYTHON) -m cms.features.extract

structure:
	$(PYTHON) -m cms.monitors.structure_monitor

length:
	$(PYTHON) -m cms.monitors.length_only

content:
	$(PYTHON) -m cms.monitors.content_monitor

combined:
	$(PYTHON) -m cms.monitors.combined

scores:
	$(PYTHON) -m cms.monitors.run_all

analyze:
	$(PYTHON) -m cms.analyze.report

full: rd rewrite validate features structure length content combined scores analyze

# Recompute aggregate tables from private precomputed scores; no API calls.
reproduce: analyze

# cadinho — one-command reproduction. Fixture mode by default (synthetic data).
# Switch to real data via config/config.yaml (data_sources.mode: live|provided).

UV ?= uv
RUN := $(UV) run cadinho

.PHONY: all setup ingest normalize train evaluate report report-variant test lint clean

setup:            ## create env from pinned uv.lock
	$(UV) sync --extra dev

all:              ## ingest -> normalize -> train -> evaluate -> report
	$(RUN) all

ingest:
	$(RUN) ingest

normalize:
	$(RUN) normalize

train:
	$(RUN) train

evaluate:
	$(RUN) evaluate

report:
	$(RUN) report

report-variant:   ## make report-variant HGVS="c.229C>T"
	$(RUN) report-variant --hgvs "$(HGVS)"

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests

clean:            ## remove generated data/models/reports (keep raw manifest dir)
	rm -rf data/interim/* data/processed/* models/*.pkl reports/*.png reports/*.md
	find data -name '.gitkeep' -o -type f -path '*/raw/*' -delete 2>/dev/null || true

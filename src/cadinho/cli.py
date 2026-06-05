"""cadinho command-line interface.

  cadinho info | ingest | normalize | train | evaluate | report | report-variant | all

Local-first, deterministic. In fixture mode every artifact is synthetic.
"""

from __future__ import annotations

import json

import click

from cadinho.config import get_config, load_config, set_global_seeds


def _cfg(path):
    cfg = load_config(path) if path else get_config()
    set_global_seeds(cfg.project.seed)
    return cfg


@click.group(help=__doc__)
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def main(ctx, config_path):
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@main.command(help="Show config + data manifest summary.")
@click.pass_context
def info(ctx):
    cfg = _cfg(ctx.obj["config_path"])
    click.echo(f"project={cfg.project.name} build={cfg.project.genome_build} "
               f"seed={cfg.project.seed}")
    click.echo(f"mode={cfg.data_sources.mode}  panel={cfg.genes.panel}  focus={cfg.genes.focus}")
    man = cfg.path("raw", "manifest.json", mkdir=False)
    if man.exists():
        m = json.loads(man.read_text())
        for name, e in m.get("sources", {}).items():
            click.echo(f"  {name:8s} mode={e['mode']} synthetic={e['synthetic']} "
                       f"n={e.get('n_rows')} sha256={(e.get('sha256') or '')[:12]}")
    else:
        click.echo("  (no manifest yet — run `cadinho ingest`)")


@main.command(help="Pull/generate raw sources (per data_sources.mode) + manifest.")
@click.option("--force", is_flag=True, help="Regenerate even if present.")
@click.pass_context
def ingest(ctx, force):
    from cadinho.ingest.run import run_ingest
    cfg = _cfg(ctx.obj["config_path"])
    paths = run_ingest(cfg, force=force)
    for k, p in paths.items():
        click.echo(f"  {k:8s} -> {p}")


@main.command(help="Normalise + reconcile sources into the unified variant table.")
@click.option("--force", is_flag=True)
@click.pass_context
def normalize(ctx, force):
    from cadinho.normalize.reconcile import build_interim
    cfg = _cfg(ctx.obj["config_path"])
    df = build_interim(cfg, force=force)
    click.echo(f"  interim: {len(df)} variants "
               f"(missense={int((df.variant_class=='missense').sum())}, "
               f"truncating={int((df.variant_class=='truncating').sum())})")


@main.command(help="Train the calibrated missense model (deterministic).")
@click.pass_context
def train(ctx):
    from cadinho.model.train import train_model
    from cadinho.normalize.reconcile import load_interim
    cfg = _cfg(ctx.obj["config_path"])
    bundle, _ = train_model(cfg, load_interim(cfg))
    click.echo(f"  trained {bundle.backend} on {bundle.metadata['n_train']} labelled "
               f"missense -> {cfg.path('models','model.pkl')}")


@main.command(help="Leakage-aware evaluation: CV + ablations + baselines + calibration.")
@click.option("--repeats", default=None, type=int, help="Override CV repeats.")
@click.pass_context
def evaluate(ctx, repeats):
    from cadinho.evaluate.run import run_evaluation
    from cadinho.normalize.reconcile import load_interim
    cfg = _cfg(ctx.obj["config_path"])
    r = run_evaluation(cfg, load_interim(cfg), repeats=repeats)
    full = r["ablations"]["full"]["auc"]
    click.echo(f"  full AUC={full['mean']:.3f} [{full['lo']:.3f}-{full['hi']:.3f}]; "
               f"reports/evaluation.md written")


@main.command(help="Score the focus-gene VUS set and write the report.")
@click.pass_context
def report(ctx):
    from cadinho.ingest import uniprot
    from cadinho.ingest.layout import raw_paths
    from cadinho.model.train import load_bundle
    from cadinho.normalize.reconcile import load_interim
    from cadinho.report.variant_report import run_vus_report
    cfg = _cfg(ctx.obj["config_path"])
    df = load_interim(cfg)
    bundle = load_bundle(cfg.path("models", "model.pkl"))
    uni = uniprot.parse(raw_paths(cfg)["uniprot"])
    vus = run_vus_report(cfg, bundle, df, uni)
    click.echo(f"  scored {len(vus)} {cfg.genes.focus} VUS -> reports/vus_predictions.md")


@main.command("report-variant", help="Interpretable report for one HGVS variant.")
@click.option("--hgvs", required=True, help='e.g. "c.229C>T" or "p.Arg77Cys"')
@click.option("--gene", default=None, help="Gene symbol (default: focus gene).")
@click.pass_context
def report_variant(ctx, hgvs, gene):
    from cadinho.ingest import uniprot
    from cadinho.ingest.layout import raw_paths
    from cadinho.model.train import load_bundle
    from cadinho.normalize.reconcile import load_interim
    from cadinho.report.variant_report import report_variant as _rv
    cfg = _cfg(ctx.obj["config_path"])
    df = load_interim(cfg)
    bundle = load_bundle(cfg.path("models", "model.pkl"))
    uni = uniprot.parse(raw_paths(cfg)["uniprot"])
    rep = _rv(cfg, bundle, df, uni, hgvs, gene)
    click.echo(rep["markdown"])


@main.command(help="Run the whole pipeline: ingest -> normalize -> train -> evaluate -> report.")
@click.option("--repeats", default=None, type=int)
@click.pass_context
def all(ctx, repeats):
    from cadinho.evaluate.run import run_evaluation
    from cadinho.ingest import uniprot
    from cadinho.ingest.layout import raw_paths
    from cadinho.ingest.run import run_ingest
    from cadinho.model.train import train_model
    from cadinho.normalize.reconcile import build_interim
    from cadinho.report.variant_report import run_vus_report
    cfg = _cfg(ctx.obj["config_path"])
    run_ingest(cfg)
    df = build_interim(cfg, force=True)
    bundle, _ = train_model(cfg, df)
    run_evaluation(cfg, df, repeats=repeats)
    uni = uniprot.parse(raw_paths(cfg)["uniprot"])
    run_vus_report(cfg, bundle, df, uni)
    click.echo("  done: data/ + models/ + reports/ populated.")


if __name__ == "__main__":
    main()

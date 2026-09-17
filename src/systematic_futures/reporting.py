"""The rendered documents: how a rule's outcome and a table's numbers are written down.

The rule itself is `decision.verdict`, and the prose a phase pre-declared stays in the phase script —
this module owns the *derived* text, the parts a reader would otherwise have to trust: the markdown
body of the outcome table and the sentence that reports which variants won, both rendered from the same
read of the same numbers that `DECISION.md` and `OOT.md` commit. The notebook renders from here too, so
a verdict cannot be written three ways.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def family_names(variants: Sequence[str], labels: Mapping[str, str]) -> str:
    """The variants as the docs name them: `6a \\`ml_defaults\\`, 6b \\`ml_tuned\\``."""
    return ", ".join(f"{labels[v]} `{v}`" for v in variants)


def outcome_split(primary: Mapping, variants: Sequence[str]) -> tuple[list[str], list[str]]:
    """(winners, failed challengers) — the partition every sentence about the rule is written from."""
    winners = [v for v in variants if primary["beats"][v]]
    return winners, [v for v in variants if not primary["beats"][v]]


def benchmark_line(primary: Mapping, benchmark: str) -> str:
    """The line each outcome document opens its numbers with."""
    return (
        f"Benchmark `{benchmark}`: Sharpe {primary['benchmark_sharpe']:.2f}, "
        f"DSR {primary['benchmark_dsr']:.4g}."
    )


def outcome_table(
    primary: Mapping,
    sensitivity: Mapping[str, Mapping],
    variants: Sequence[str],
    labels: Mapping[str, str],
    *,
    extra_headers: Sequence[str] = (),
    extra_cells: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """The outcome document's table: the primary read and the robustness read side by side.

    One row per variant — Sharpe, the primary DSR and whether it beats the benchmark, the like-for-like
    DSR and whether that read agrees — plus any extra columns a phase's document carries (the
    out-of-sample record adds coverage). `extra_cells` maps a variant to its already-formatted cells.
    """
    headers = ["variant", "Sharpe", "DSR (primary)", "beats benchmark", "DSR (own dates)"]
    headers += ["beats benchmark", *extra_headers]
    separator = "|" + "---|" * len(headers)
    rows = [
        "| {label} `{variant}` | {sharpe:.2f} | {primary_dsr:.4g} | {beats} | {own_dsr:.4g} | {own_beats} |{extra}".format(
            label=labels[v],
            variant=v,
            sharpe=primary["variant_sharpe"][v],
            primary_dsr=primary["variant_dsr"][v],
            beats="yes" if primary["beats"][v] else "no",
            own_dsr=sensitivity[v]["variant_dsr"][v],
            own_beats="yes" if sensitivity[v]["beats"][v] else "no",
            extra="".join(f" {cell} |" for cell in (extra_cells or {}).get(v, ())),
        )
        for v in variants
    ]
    return "\n".join(["| " + " | ".join(headers) + " |", separator, *rows])

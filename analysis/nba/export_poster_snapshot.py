"""Export poster-ready model snapshot tables for the final capstone poster."""

from pathlib import Path

from feature_spec import poster_model_snapshot_frame


DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "poster_model_snapshot.csv"
MD_PATH = DATA_DIR / "poster_model_snapshot.md"


def to_markdown_table(df) -> str:
    header = "| Concept | NBA variables | NHL variables | Why it matters |"
    divider = "| --- | --- | --- | --- |"
    rows = [
        "| "
        + " | ".join(str(value).replace("\n", " ").replace("|", "/") for value in row)
        + " |"
        for row in df[
            ["concept", "nba_variables", "nhl_variables", "why_it_matters"]
        ].itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = poster_model_snapshot_frame()
    snapshot.to_csv(CSV_PATH, index=False)
    MD_PATH.write_text(to_markdown_table(snapshot), encoding="utf-8")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()

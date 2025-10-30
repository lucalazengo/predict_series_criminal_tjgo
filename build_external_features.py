import os
import json
import unicodedata
from typing import Dict, List

import pandas as pd


def slugify(text: str) -> str:
    """Convert a string to a slug: ascii, lowercase, underscores, no accents."""
    if text is None:
        return ""
    # Normalize accents
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    # Clean up: lowercase, spaces->underscores, remove non-alphanum except underscore
    ascii_text = ascii_text.lower().strip().replace(" ", "_")
    cleaned = []
    for ch in ascii_text:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        else:
            # collapse other punctuations into underscore
            cleaned.append("_")
    # collapse multiple underscores
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def build_external_features(
    input_csv: str = "data/raw_data/df_go_seg_pub_2015-2025.csv",
    output_csv: str = "data/processed/criminal/external_features_2015_2024.csv",
    output_metadata: str = "data/processed/criminal/external_features_metadata.json",
    uf_filter: str = "GO",
    start_year: int = 2015,
    end_year: int = 2024,
) -> Dict:
    """
    Build monthly external variables dataset for modeling (2015–2024).

    Steps:
    - Load raw CSV
    - Filter by UF and statewide scope
    - Parse date and clip to [start_year, end_year]
    - Coalesce numeric values (sum of total_vitima, total, total_peso)
    - Aggregate by month and event
    - Pivot to wide format with slugified event names
    - Reindex to full monthly range and fill missing with 0
    - Save CSV and metadata JSON
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_csv}")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Read CSV (likely UTF-8)
    df = pd.read_csv(input_csv, encoding="utf-8")

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Parse date
    if "data_referencia" not in df.columns:
        raise KeyError("Coluna 'data_referencia' não encontrada no CSV.")
    df["data_referencia"] = pd.to_datetime(df["data_referencia"], errors="coerce")

    # Filter UF
    if "uf" in df.columns and uf_filter:
        df = df[df["uf"].astype(str).str.upper() == uf_filter.upper()]

    # Filter statewide scope
    if "abrangencia" in df.columns:
        abr = df["abrangencia"].astype(str).str.strip().str.lower().fillna("")
        df = df[abr == "estadual"]

    # Clip date range
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    df = df[(df["data_referencia"] >= start) & (df["data_referencia"] <= end)]

    # Numeric columns coalescing
    num_cols = [c for c in ["total_vitima", "total", "total_peso"] if c in df.columns]
    if not num_cols:
        raise KeyError("Nenhuma coluna numérica encontrada (total_vitima/total/total_peso).")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["valor"] = df[num_cols].sum(axis=1)

    # Keep positive values only (events with actual counts)
    df = df[df["valor"] > 0]

    # Month key
    df["mes"] = df["data_referencia"].dt.to_period("M").dt.to_timestamp()

    # Event key
    if "evento" not in df.columns:
        raise KeyError("Coluna 'evento' não encontrada no CSV.")
    df["evento"] = df["evento"].astype(str).str.strip()

    # Aggregate
    g = df.groupby(["mes", "evento"], as_index=False)["valor"].sum()

    if g.empty:
        raise ValueError("Agregação resultou vazia. Verifique se há valores numéricos no período.")

    # Pivot wide
    wide = g.pivot(index="mes", columns="evento", values="valor")

    # Ensure full monthly index
    full_index = pd.period_range(f"{start_year}-01", f"{end_year}-12", freq="M").to_timestamp()
    wide = wide.reindex(full_index)

    # Slugify columns
    original_cols: List[str] = list(wide.columns)
    slug_cols: List[str] = [slugify(c) for c in original_cols]
    wide.columns = slug_cols

    # Fill missing with zero
    wide = wide.fillna(0)

    # Reset index and rename date column
    wide = wide.reset_index().rename(columns={"index": "data", "mes": "data"})

    # Save CSV
    wide.to_csv(output_csv, index=False)

    # Metadata
    metadata = {
        "uf": uf_filter,
        "period_start": f"{start_year}-01",
        "period_end": f"{end_year}-12",
        "rows": len(wide),
        "columns": ["data"] + slug_cols,
        "event_mapping": {slugify(orig): orig for orig in original_cols},
        "value_definition": f"sum({', '.join(num_cols)})",
        "source_csv": os.path.abspath(input_csv),
        "output_csv": os.path.abspath(output_csv),
    }

    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


if __name__ == "__main__":
    meta = build_external_features()
    print("Dataset de variáveis externas gerado com sucesso:")
    print(json.dumps({
        "periodo": f"{meta['period_start']} a {meta['period_end']}",
        "linhas": meta["rows"],
        "colunas": len(meta["columns"]),
        "arquivo": meta["output_csv"],
    }, ensure_ascii=False, indent=2))
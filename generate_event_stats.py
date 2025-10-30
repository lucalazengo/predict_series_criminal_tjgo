import os
import json
import unicodedata
import pandas as pd


def slugify(text: str) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower().strip().replace(" ", "_")
    cleaned = []
    for ch in ascii_text:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def load_filtered(
    input_csv: str,
    uf_filter: str = "GO",
    scope_filter: str = "Estadual",
    start_year: int = 2015,
    end_year: int = 2024,
) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    if "data_referencia" not in df.columns:
        raise KeyError("Coluna 'data_referencia' não encontrada no CSV.")
    df["data_referencia"] = pd.to_datetime(df["data_referencia"], errors="coerce")
    if uf_filter and "uf" in df.columns:
        df = df[df["uf"].astype(str).str.upper() == uf_filter.upper()]
    if scope_filter and "abrangencia" in df.columns:
        df = df[df["abrangencia"].astype(str).str.strip().str.lower() == scope_filter.lower()]
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    df = df[(df["data_referencia"] >= start) & (df["data_referencia"] <= end)]
    # Clean evento
    if "evento" not in df.columns:
        raise KeyError("Coluna 'evento' não encontrada no CSV.")
    df["evento"] = df["evento"].astype(str).str.strip()
    df["mes"] = df["data_referencia"].dt.to_period("M").dt.to_timestamp()
    return df


def main():
    input_csv = "data/raw_data/df_go_seg_pub_2015-2025.csv"
    out_dir = "data/processed/criminal"
    os.makedirs(out_dir, exist_ok=True)

    df = load_filtered(input_csv)

    # Contagem de linhas por mês e classe de evento
    counts = (
        df.groupby(["mes", "evento"]).size().reset_index(name="row_count")
    )
    # Pivot para wide e padroniza nomes
    wide_counts = counts.pivot(index="mes", columns="evento", values="row_count")
    full_index = (
        pd.period_range("2015-01", "2024-12", freq="M").to_timestamp()
    )
    wide_counts = wide_counts.reindex(full_index).fillna(0)
    original_cols = list(wide_counts.columns)
    slug_cols = [slugify(c) for c in original_cols]
    wide_counts.columns = slug_cols
    wide_counts = wide_counts.reset_index().rename(columns={"index": "data", "mes": "data"})
    wide_counts.to_csv(os.path.join(out_dir, "event_monthly_row_counts_2015_2024.csv"), index=False)

    # Sumário total de contagens por classe
    summary = counts.groupby("evento")["row_count"].sum().reset_index()
    summary["slug"] = summary["evento"].apply(slugify)
    summary = summary.sort_values("row_count", ascending=False)
    summary.to_csv(os.path.join(out_dir, "event_row_counts_summary_2015_2024.csv"), index=False)

    # Mapeamento slug -> original
    mapping = {slugify(orig): orig for orig in original_cols}
    with open(os.path.join(out_dir, "event_slug_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print("Arquivos gerados:")
    print("- data/processed/criminal/event_monthly_row_counts_2015_2024.csv")
    print("- data/processed/criminal/event_row_counts_summary_2015_2024.csv")
    print("- data/processed/criminal/event_slug_mapping.json")


if __name__ == "__main__":
    main()
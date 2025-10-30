import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Inspeciona classes únicas na coluna 'evento' e salva a lista."
    )
    parser.add_argument(
        "--input",
        default="data/raw_data/df_go_seg_pub_2015-2025.csv",
        help="Caminho do CSV bruto",
    )
    parser.add_argument(
        "--uf",
        default="GO",
        help="Filtro de UF (ex.: GO). Use vazio para não filtrar",
    )
    parser.add_argument(
        "--scope",
        default="Estadual",
        help="Filtro de abrangência (ex.: Estadual). Use vazio para não filtrar",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Ano inicial para recorte",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Ano final para recorte",
    )
    parser.add_argument(
        "--output",
        default="data/processed/criminal/event_classes_GO_Estadual_2015_2024.txt",
        help="Arquivo de saída para lista de classes",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    # Parse date
    if "data_referencia" not in df.columns:
        raise KeyError("Coluna 'data_referencia' não encontrada no CSV.")
    df["data_referencia"] = pd.to_datetime(df["data_referencia"], errors="coerce")

    # Filters
    if args.uf and "uf" in df.columns:
        df = df[df["uf"].astype(str).str.upper() == args.uf.upper()]
    if args.scope and "abrangencia" in df.columns:
        df = df[df["abrangencia"].astype(str).str.strip().str.lower() == args.scope.lower()]

    # Clip by date
    start = pd.Timestamp(f"{args.start_year}-01-01")
    end = pd.Timestamp(f"{args.end_year}-12-31")
    df = df[(df["data_referencia"] >= start) & (df["data_referencia"] <= end)]

    # Evento classes
    if "evento" not in df.columns:
        raise KeyError("Coluna 'evento' não encontrada no CSV.")
    eventos = (
        df["evento"].astype(str).str.strip().replace({"": pd.NA}).dropna().unique()
    )
    classes = sorted(set(eventos))

    # Output directory
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save list
    with open(args.output, "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")

    # Print summary
    print(f"Total de classes únicas em 'evento': {len(classes)}")
    preview = classes[:20]
    print("Prévia (20 primeiras):")
    for i, cls in enumerate(preview, 1):
        print(f"{i:02d}. {cls}")
    print(f"Lista completa salva em: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
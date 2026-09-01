#!/usr/bin/env python3
"""
pdf_to_eps.py — Convertit un ou plusieurs PDF en EPS (vectoriel, idéal pour LaTeX).

Méthode principale : `pdftops -eps` (poppler-utils).
Méthode de secours  : Ghostscript (`gs`), si pdftops n'est pas installé.

Installation des dépendances (si besoin) :
    Ubuntu/Debian : sudo apt install poppler-utils ghostscript
    macOS (brew)  : brew install poppler ghostscript

Usage :
    python pdf_to_eps.py figure.pdf
    python pdf_to_eps.py figure.pdf -o figure.eps
    python pdf_to_eps.py dossier_de_pdfs/          # convertit tous les .pdf du dossier
    python pdf_to_eps.py *.pdf                     # plusieurs fichiers à la fois
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def convert_with_pdftops(pdf_path: Path, eps_path: Path) -> None:
    subprocess.run(
        ["pdftops", "-eps", str(pdf_path), str(eps_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def convert_with_ghostscript(pdf_path: Path, eps_path: Path) -> None:
    subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-dEPSCrop",
            "-sDEVICE=eps2write",
            f"-sOutputFile={eps_path}",
            str(pdf_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def convert_pdf_to_eps(pdf_path: Path, eps_path: Path | None = None) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")
    eps_path = eps_path or pdf_path.with_suffix(".eps")

    if shutil.which("pdftops"):
        try:
            convert_with_pdftops(pdf_path, eps_path)
            return eps_path
        except subprocess.CalledProcessError as e:
            print(f"  pdftops a échoué, essai avec Ghostscript... ({e.stderr.strip()})")

    if shutil.which("gs"):
        convert_with_ghostscript(pdf_path, eps_path)
        return eps_path

    raise RuntimeError(
        "Aucun outil de conversion trouvé. Installez poppler-utils (pdftops) "
        "ou ghostscript (gs)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertit des PDF en EPS.")
    parser.add_argument(
        "inputs", nargs="+", help="Fichier(s) PDF ou dossier(s) contenant des PDF"
    )
    parser.add_argument(
        "-o", "--output", help="Chemin de sortie (uniquement si un seul fichier PDF)"
    )
    args = parser.parse_args()

    # Construit la liste des PDF à traiter (fichiers ou dossiers)
    pdf_files: list[Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            pdf_files.extend(sorted(p.glob("*.pdf")))
        else:
            pdf_files.append(p)

    if not pdf_files:
        print("Aucun fichier PDF trouvé.")
        sys.exit(1)

    if args.output and len(pdf_files) > 1:
        print("Erreur : -o/--output ne peut être utilisé qu'avec un seul fichier PDF.")
        sys.exit(1)

    for pdf_file in pdf_files:
        out_path = Path(args.output) if args.output else None
        try:
            eps_file = convert_pdf_to_eps(pdf_file, out_path)
            print(f"✓ {pdf_file} → {eps_file}")
        except Exception as e:
            print(f"✗ {pdf_file} : {e}")


if __name__ == "__main__":
    main()
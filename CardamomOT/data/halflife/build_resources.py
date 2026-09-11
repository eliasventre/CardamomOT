"""
build_resources.py
------------------
Rebuild the half-life reference tables shipped in ``CardamomOT/data/halflife``.

Nothing in this script runs during a normal CardamomOT analysis: it downloads
the public sources below once, and writes compact tables that
``CardamomOT.inference.halflife_db`` reads at runtime (no network needed).

Sources
~~~~~~~
Half-lives, in hours:

* Mouse, mRNA and protein — Schwanhäusser et al., "Global quantification of
  mammalian gene expression control", Nature 473:337-342 (2011), Supplementary
  Table 3 **as replaced by the 2013 corrigendum** (Nature 495:126-127,
  doi:10.1038/nature11848). NIH3T3 fibroblasts.
* Human, mRNA — RNADecayCafe v1.1 (Vock et al. 2025, bioRxiv
  10.1101/2025.08.19.671151; Zenodo 16884513, CC-BY 4.0): uniformly reprocessed
  SLAM-seq / TimeLapse-seq data from 11 human cell lines. Per gene, the
  geometric mean over cell lines of the dropout-normalised half-life, each cell
  line first recentred on a common median.
* Human, protein — Mathieson et al., "Systematic analysis of protein turnover in
  primary cells", Nat Commun 9:689 (2018), Supplementary Data 2 ("high qual"):
  B cells, NK cells, hepatocytes, monocytes. Per gene, the geometric mean over
  replicates, then over cell types, each recentred on a common median (NK cells
  turn over ~2x slower than the others).

Gene relations:

* Nomenclature — MGI ``MRK_List2.rpt`` / ``MRK_ENSEMBL.rpt``; HGNC complete set
  (symbols, previous/alias symbols, Ensembl and UniProt ids, gene groups).
* Mouse ↔ human orthology — MGI ``HOM_MouseHumanSequence.rpt`` (Alliance).
* Paralogy — Ensembl BioMart within-species paralogues with protein identity.
* Function — Gene Ontology annotations (``mgi.gaf``, ``goa_human.gaf``) on
  ``go-basic.obo``; functional neighbours ranked by simGIC (Pesquita et al.
  2008), the information-content-weighted Jaccard index of ancestor-propagated
  GO term sets.

Usage
~~~~~
    python CardamomOT/data/halflife/build_resources.py --raw <download_dir>

Raw files already present in ``--raw`` are not downloaded again. Reading the
.xls/.xlsx sources needs ``xlrd`` and ``openpyxl``.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

URLS = {
    "schwanhausser2011_S3.xls": "https://media.springernature.com/original/springer-static/esm/"
                                "art%3A10.1038%2Fnature10098/MediaObjects/41586_2011_BFnature10098_MOESM304_ESM.xls",
    "mathieson2018_S2.xlsx": "https://media.springernature.com/original/springer-static/esm/"
                             "art%3A10.1038%2Fs41467-018-03106-1/MediaObjects/41467_2018_3106_MOESM5_ESM.xlsx",
    "rnadecaycafe_AvgKdegs_genes_v1.1.csv": "https://zenodo.org/api/records/16884513/files/"
                                            "AvgKdegs_genes_v1.1.csv/content",
    "MRK_List2.rpt": "https://www.informatics.jax.org/downloads/reports/MRK_List2.rpt",
    "MRK_ENSEMBL.rpt": "https://www.informatics.jax.org/downloads/reports/MRK_ENSEMBL.rpt",
    "HOM_MouseHumanSequence.rpt": "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt",
    "hgnc_complete_set.txt": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    "mgi.gaf.gz": "https://current.geneontology.org/annotations/mgi.gaf.gz",
    "goa_human.gaf.gz": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
    "go-basic.obo": "https://current.geneontology.org/ontology/go-basic.obo",
}

BIOMART_MIRRORS = ["https://asia.ensembl.org", "https://useast.ensembl.org", "https://www.ensembl.org"]
BIOMART_DATASET = {"mouse": "mmusculus", "human": "hsapiens"}
BIOMART_LABEL = {"mouse": "Mouse", "human": "Human"}

# Paralog pairs below this protein identity (%, averaged over both directions)
# are not kept: at that level they share little more than a domain.
MIN_PARALOG_IDENTITY = 20.0
# Number of GO functional neighbours stored per gene (runtime filters further).
N_GO_NEIGHBORS = 30
MIN_GO_SIMILARITY = 0.05
# Gene names such as "NA" or "Nan" must stay strings: only empty cells are missing.
NO_NA = dict(keep_default_na=False, na_values=[""])


def _download(url, path):
    print(f"[build] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
        f.write(r.read())


def _biomart_paralog_query(species):
    ds = BIOMART_DATASET[species]
    return (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" '
        'completionStamp="1" datasetConfigVersion="0.6">'
        f'<Dataset name="{ds}_gene_ensembl" interface="default">'
        '<Filter name="biotype" value="protein_coding"/>'
        '<Attribute name="ensembl_gene_id"/>'
        f'<Attribute name="{ds}_paralog_ensembl_gene"/>'
        f'<Attribute name="{ds}_paralog_perc_id"/>'
        f'<Attribute name="{ds}_paralog_perc_id_r1"/>'
        '</Dataset></Query>'
    )


def fetch_raw(raw):
    os.makedirs(raw, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            _download(url, path)
    for species in ("mouse", "human"):
        path = os.path.join(raw, f"paralogs_{species}.tsv")
        if os.path.exists(path):
            continue
        query = urllib.parse.urlencode({"query": _biomart_paralog_query(species)})
        for host in BIOMART_MIRRORS:
            try:
                _download(f"{host}/biomart/martservice?{query}", path)
            except Exception as e:  # mirror down: try the next one
                print(f"[build]   {host} failed: {e}")
                continue
            with open(path) as f:
                last = f.read().rstrip().rsplit("\n", 1)[-1]
            if last == "[success]":
                break
            print(f"[build]   {host} returned a truncated file, retrying elsewhere")
            os.remove(path)
        else:
            sys.exit(f"[build] could not download the complete {species} paralog table from BioMart")


# ---------------------------------------------------------------------------
# Nomenclature
# ---------------------------------------------------------------------------

def build_mouse_index(raw):
    mrk = pd.read_csv(os.path.join(raw, "MRK_List2.rpt"), sep="\t", dtype=str, **NO_NA)
    mrk = mrk[mrk["Marker Type"].isin(["Gene", "Pseudogene"])]
    rows = []
    for mgi, sym, syn in zip(mrk["MGI Accession ID"], mrk["Marker Symbol"],
                             mrk["Marker Synonyms (pipe-separated)"]):
        rows.append((sym, mgi, "symbol"))
        if isinstance(syn, str):
            rows.extend((s, mgi, "synonym") for s in syn.split("|") if s)
    ens = pd.read_csv(os.path.join(raw, "MRK_ENSEMBL.rpt"), sep="\t", header=None, dtype=str, **NO_NA)
    rows.extend((e, m, "ensembl") for m, e in zip(ens[0], ens[5]) if isinstance(e, str))
    idx = pd.DataFrame(rows, columns=["name", "gene_id", "kind"]).drop_duplicates()
    info = mrk[["MGI Accession ID", "Marker Symbol", "Feature Type"]]
    info.columns = ["gene_id", "symbol", "feature_type"]
    return idx, info, dict(zip(ens[5], ens[0]))


def build_human_index(raw):
    h = pd.read_csv(os.path.join(raw, "hgnc_complete_set.txt"), sep="\t", dtype=str, **NO_NA)
    h = h[h["status"] == "Approved"]
    rows, uniprot = [], {}
    for _, r in h.iterrows():
        rows.append((r["symbol"], r["hgnc_id"], "symbol"))
        for col, kind in (("prev_symbol", "previous"), ("alias_symbol", "synonym")):
            if isinstance(r[col], str):
                rows.extend((s, r["hgnc_id"], kind) for s in r[col].split("|") if s)
        if isinstance(r["ensembl_gene_id"], str):
            rows.append((r["ensembl_gene_id"], r["hgnc_id"], "ensembl"))
        if isinstance(r["uniprot_ids"], str):
            for u in r["uniprot_ids"].split("|"):
                uniprot[u] = r["hgnc_id"]
    idx = pd.DataFrame(rows, columns=["name", "gene_id", "kind"]).drop_duplicates()
    info = h[["hgnc_id", "symbol", "locus_type"]].rename(columns={"hgnc_id": "gene_id", "locus_type": "feature_type"})
    groups = []
    for hid, gids, gnames in zip(h["hgnc_id"], h["gene_group_id"], h["gene_group"]):
        if isinstance(gids, str):
            groups.extend((hid, int(g), n) for g, n in zip(gids.split("|"), gnames.split("|")))
    groups = pd.DataFrame(groups, columns=["hgnc_id", "group_id", "group_name"])
    ens2hgnc = dict(zip(h["ensembl_gene_id"], h["hgnc_id"]))
    return idx, info, groups, ens2hgnc, uniprot


def build_orthologs(raw):
    hom = pd.read_csv(os.path.join(raw, "HOM_MouseHumanSequence.rpt"), sep="\t", dtype=str, **NO_NA)
    mouse = hom[hom["NCBI Taxon ID"] == "10090"][["DB Class Key", "Mouse MGI ID"]]
    human = hom[hom["NCBI Taxon ID"] == "9606"][["DB Class Key", "HGNC ID"]]
    orth = mouse.merge(human, on="DB Class Key").dropna()
    return orth.rename(columns={"Mouse MGI ID": "mgi_id", "HGNC ID": "hgnc_id"})[["mgi_id", "hgnc_id"]].drop_duplicates()


class NameResolver:
    """Resolve published gene names to current MGI / HGNC ids (unambiguous only)."""

    def __init__(self, idx):
        self.maps = {k: sub.groupby("name").gene_id.apply(set).to_dict() for k, sub in idx.groupby("kind")}

    def __call__(self, name):
        for kind in ("symbol", "ensembl", "previous", "synonym"):
            ids = self.maps.get(kind, {}).get(name, set())
            if ids and (kind in ("symbol", "ensembl") or len(ids) == 1):
                return ids
        return set()


# ---------------------------------------------------------------------------
# Half-lives
# ---------------------------------------------------------------------------

def _geomean(v):
    v = v.dropna()
    return float(np.exp(np.log(v).mean())) if len(v) else np.nan


def build_halflives_mouse(raw, resolve, valid_mgi):
    """Schwanhäusser 2011 (corrigendum 2013), NIH3T3."""
    x = pd.read_excel(os.path.join(raw, "schwanhausser2011_S3.xls"), **NO_NA)
    records, n_unmapped = [], 0
    for _, r in x.iterrows():
        mgi = set()
        if isinstance(r["MGI ID"], str):
            mgi = {m.strip() for m in r["MGI ID"].split(";") if m.strip() in valid_mgi}
        if not mgi and isinstance(r["Gene Names"], str):
            # MGI ids missing or retired: resolve the 2011 names against current nomenclature.
            for name in r["Gene Names"].split(";"):
                mgi |= resolve(name)
        if not mgi:
            n_unmapped += 1
            continue
        for m in mgi:
            records.append((m, pd.to_numeric(r["mRNA half-life average [h]"], errors="coerce"),
                            pd.to_numeric(r["Protein half-life average [h]"], errors="coerce"), r["Gene Names"]))
    df = pd.DataFrame(records, columns=["gene_id", "mrna_half_life_h", "prot_half_life_h", "measurement_group"])
    # A gene hit by several protein groups: geometric mean of its measurements.
    agg = df.groupby("gene_id").agg(mrna_half_life_h=("mrna_half_life_h", _geomean),
                                    prot_half_life_h=("prot_half_life_h", _geomean),
                                    measurement_group=("measurement_group", "first")).reset_index()
    print(f"[build] Schwanhäusser: {len(x)} rows -> {len(agg)} mouse genes ({n_unmapped} rows unmapped)")
    return agg


def _centered_mean(log_values):
    """
    Per-gene mean of log values over conditions (columns), each condition first
    recentred on a common median. Cell types / lines differ in overall turnover
    (e.g. NK cells vs hepatocytes), so a gene measured in only some of them must
    not inherit their scale: it keeps its relative stability within each one.
    """
    medians = log_values.median()
    return (log_values - medians).mean(axis=1) + medians.mean()


def build_halflives_human(raw, resolve):
    """RNADecayCafe v1.1 (mRNA) and Mathieson 2018 (protein)."""
    k = pd.read_csv(os.path.join(raw, "rnadecaycafe_AvgKdegs_genes_v1.1.csv"), **NO_NA)
    log_kdeg = k.pivot_table(index="feature_ID", columns="cell_line", values="avg_donorm_log_kdeg")
    per_gene = pd.DataFrame({"log_kdeg": _centered_mean(log_kdeg),
                             "n_mrna_cell_lines": log_kdeg.notna().sum(axis=1)})
    rows, n_unmapped = [], 0
    for name, lk, n in zip(per_gene.index, per_gene.log_kdeg, per_gene.n_mrna_cell_lines):
        ids = resolve(name)
        n_unmapped += not ids
        rows.extend((h, np.log(2) / np.exp(lk), n) for h in ids)
    mrna = pd.DataFrame(rows, columns=["gene_id", "mrna_half_life_h", "n_mrna_cell_lines"])
    mrna = mrna.groupby("gene_id").agg(mrna_half_life_h=("mrna_half_life_h", _geomean),
                                       n_mrna_cell_lines=("n_mrna_cell_lines", "max")).reset_index()
    print(f"[build] RNADecayCafe: {len(per_gene)} genes -> {len(mrna)} HGNC genes ({n_unmapped} unmapped)")

    m = pd.read_excel(os.path.join(raw, "mathieson2018_S2.xlsx"), sheet_name="protein half lives high qual")
    cells = ["Bcells", "NK cells", "Hepatocytes", "Monocytes"]
    # Replicates are recentred within each cell type, then cell types on a common median.
    log_cells = pd.DataFrame({
        c: _centered_mean(np.log(m[[f"{c} replicate 1 half_life", f"{c} replicate 2 half_life"]])) for c in cells})
    m = pd.DataFrame({"name": m.gene_name.astype(str), "prot_half_life_h": np.exp(_centered_mean(log_cells)),
                      "n_prot_cell_types": log_cells.notna().sum(axis=1)}).dropna(subset=["prot_half_life_h"])
    rows, n_unmapped = [], 0
    for name, hl, n in zip(m.name, m.prot_half_life_h, m.n_prot_cell_types):
        ids = resolve(name)
        n_unmapped += not ids
        rows.extend((h, hl, n) for h in ids)
    prot = pd.DataFrame(rows, columns=["gene_id", "prot_half_life_h", "n_prot_cell_types"])
    prot = prot.groupby("gene_id").agg(prot_half_life_h=("prot_half_life_h", _geomean),
                                       n_prot_cell_types=("n_prot_cell_types", "max")).reset_index()
    print(f"[build] Mathieson 2018: {len(m)} proteins -> {len(prot)} HGNC genes ({n_unmapped} unmapped)")

    df = mrna.merge(prot, on="gene_id", how="outer")
    df["measurement_group"] = df.gene_id  # one measurement per gene
    return df


# ---------------------------------------------------------------------------
# Biological neighbours
# ---------------------------------------------------------------------------

def build_paralogs(raw, species, ens2id, measured):
    label = BIOMART_LABEL[species]
    p = pd.read_csv(os.path.join(raw, f"paralogs_{species}.tsv"), sep="\t", comment="[")
    p = p[["Gene stable ID", f"{label} paralogue gene stable ID",
           f"Paralogue %id. target {label} gene identical to query gene",
           f"Paralogue %id. query gene identical to target {label} gene"]]
    p.columns = ["gene", "paralog", "pid", "pid_r1"]
    p = p.dropna(subset=["paralog"])
    p["gene_id"] = p.gene.map(ens2id)
    p["paralog_id"] = p.paralog.map(ens2id)
    p["identity"] = (p.pid + p.pid_r1) / 2
    p = p.dropna(subset=["gene_id", "paralog_id"])
    p = p[(p.identity >= MIN_PARALOG_IDENTITY) & p.paralog_id.isin(measured) & (p.gene_id != p.paralog_id)]
    p = p.groupby(["gene_id", "paralog_id"], as_index=False).identity.max().round({"identity": 2})
    p.insert(0, "species", species)
    print(f"[build] {species} paralogs: {len(p)} pairs towards measured genes")
    return p


def _parse_obo(path):
    parents, obsolete, term = defaultdict(set), set(), None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                term = None
            elif line.startswith("[") and line.endswith("]"):
                term = "skip"
            elif term != "skip" and line.startswith("id: GO:"):
                term = line[4:]
            elif term and term != "skip":
                if line.startswith("is_a: "):
                    parents[term].add(line[6:16])
                elif line.startswith("relationship: part_of "):
                    parents[term].add(line[22:32])
                elif line == "is_obsolete: true":
                    obsolete.add(term)
    return parents, obsolete


def build_go_neighbors(gaf_path, species, to_gene_id, measured, ontology):
    parents, obsolete = ontology
    roots = {"GO:0008150", "GO:0003674", "GO:0005575"}
    ancestors_cache = {}

    def ancestors(t):
        if t not in ancestors_cache:
            out = {t}
            for p in parents.get(t, ()):
                out |= ancestors(p)
            ancestors_cache[t] = out
        return ancestors_cache[t]

    sys.setrecursionlimit(10000)
    gaf = pd.read_csv(gaf_path, sep="\t", comment="!", header=None, dtype=str, usecols=[1, 2, 3, 4, 6], **NO_NA)
    gaf.columns = ["object_id", "symbol", "qualifier", "go", "evidence"]
    gaf = gaf[~gaf.qualifier.str.contains("NOT", na=False) & (gaf.evidence != "ND") & ~gaf.go.isin(obsolete)]
    gene_terms = defaultdict(set)
    for obj, sym, t in zip(gaf.object_id, gaf.symbol, gaf.go):
        g = to_gene_id(obj, sym)
        if g:
            gene_terms[g] |= ancestors(t)
    genes = sorted(gene_terms)
    terms = sorted(set().union(*gene_terms.values()) - roots)
    tix = {t: i for i, t in enumerate(terms)}
    rows, cols = [], []
    for i, g in enumerate(genes):
        for t in gene_terms[g]:
            if t in tix:
                rows.append(i)
                cols.append(tix[t])
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(genes), len(terms)))
    freq = np.asarray(X.sum(axis=0)).ravel() / len(genes)
    ic = -np.log(freq).astype(np.float32)
    XW = X @ sp.diags(ic)
    total = np.asarray(XW.sum(axis=1)).ravel()

    target_idx = np.array([i for i, g in enumerate(genes) if g in measured])
    XT = X[target_idx].T.tocsc()
    total_t = total[target_idx]
    out = []
    for start in range(0, len(genes), 2000):
        stop = min(start + 2000, len(genes))
        shared = (XW[start:stop] @ XT).toarray()
        union = total[start:stop, None] + total_t[None, :] - shared
        sim = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)
        for k, i in enumerate(range(start, stop)):
            sim[k, target_idx == i] = 0.0  # never a neighbour of itself
        top = np.argsort(-sim, axis=1)[:, :N_GO_NEIGHBORS]
        for k in range(stop - start):
            for j in top[k]:
                if sim[k, j] >= MIN_GO_SIMILARITY:
                    out.append((species, genes[start + k], genes[target_idx[j]], float(sim[k, j])))
    df = pd.DataFrame(out, columns=["species", "gene_id", "neighbor_id", "simgic"]).round({"simgic": 4})
    print(f"[build] {species} GO: {len(genes)} annotated genes, {len(terms)} terms, {len(df)} neighbour links")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="directory holding (or receiving) the raw downloads")
    args = ap.parse_args()
    raw = args.raw
    fetch_raw(raw)

    mouse_idx, mouse_info, ens2mgi = build_mouse_index(raw)
    human_idx, human_info, groups, ens2hgnc, uniprot2hgnc = build_human_index(raw)
    orth = build_orthologs(raw)

    hl_mouse = build_halflives_mouse(raw, NameResolver(mouse_idx), set(mouse_idx.gene_id))
    hl_human = build_halflives_human(raw, NameResolver(human_idx))
    for hl, info in ((hl_mouse, mouse_info), (hl_human, human_info)):
        hl.insert(1, "symbol", hl.gene_id.map(dict(zip(info.gene_id, info.symbol))))

    human_symbols = dict(zip(human_info.symbol, human_info.gene_id))
    ontology = _parse_obo(os.path.join(raw, "go-basic.obo"))
    paralogs = pd.concat([build_paralogs(raw, "mouse", ens2mgi, set(hl_mouse.gene_id)),
                          build_paralogs(raw, "human", ens2hgnc, set(hl_human.gene_id))])
    go = pd.concat([
        build_go_neighbors(os.path.join(raw, "mgi.gaf.gz"), "mouse",
                           lambda obj, sym: obj if obj.startswith("MGI:") else None,
                           set(hl_mouse.gene_id), ontology),
        build_go_neighbors(os.path.join(raw, "goa_human.gaf.gz"), "human",
                           lambda obj, sym: uniprot2hgnc.get(obj) or human_symbols.get(sym),
                           set(hl_human.gene_id), ontology),
    ])

    mouse_idx.insert(0, "species", "mouse")
    human_idx.insert(0, "species", "human")
    mouse_info.insert(0, "species", "mouse")
    human_info.insert(0, "species", "human")

    def save(df, name):
        path = os.path.join(OUT_DIR, name)
        df.to_csv(path, sep="\t", index=False, compression="gzip" if name.endswith(".gz") else None)
        print(f"[build] wrote {name} ({len(df)} rows, {os.path.getsize(path) / 1e6:.2f} MB)")

    save(hl_mouse, "halflife_mouse.tsv")
    save(hl_human, "halflife_human.tsv")
    save(pd.concat([mouse_idx, human_idx]), "gene_index.tsv.gz")
    save(pd.concat([mouse_info, human_info]), "gene_info.tsv.gz")
    save(orth, "orthologs_mouse_human.tsv.gz")
    save(groups, "gene_groups_hgnc.tsv.gz")
    save(paralogs, "paralogs.tsv.gz")
    save(go, "go_neighbors.tsv.gz")

    # Weights of the estimation sources, fitted by leave-one-out on the measured genes.
    sys.path.insert(0, os.path.abspath(os.path.join(OUT_DIR, "..", "..", "..")))
    from CardamomOT.inference.halflife_db import HalfLifeDB, fit_models
    model = fit_models(HalfLifeDB(OUT_DIR, load_model=False))
    with open(os.path.join(OUT_DIR, "neighbor_model.json"), "w") as f:
        json.dump(model, f, indent=2)
    print("[build] estimation model:\n" + json.dumps(model, indent=2))


if __name__ == "__main__":
    main()

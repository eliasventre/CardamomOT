"""
Literature mRNA / protein half-lives and degradation rates (hour⁻¹).

Each species has its own reference table, in hours — the same time unit as
``adata.obs['time']`` and as ``NetworkModel``'s default rates:

* **mouse** — Schwanhäusser et al. 2011 (Nature 473:337, corrected table of the
  2013 corrigendum), NIH3T3 fibroblasts, mRNA and protein;
* **human** — mRNA: RNADecayCafe v1.1 (Vock et al. 2025), SLAM-seq/TimeLapse-seq
  in 11 cell lines; protein: Mathieson et al. 2018 (Nat Commun 9:689), primary
  B cells, NK cells, hepatocytes and monocytes.

The tables are built by ``CardamomOT/data/halflife/build_resources.py``.

For each query gene:

1. **Species.** Mouse (MGI, e.g. ``Gata1``) or human (HGNC, e.g. ``GATA1``)
   nomenclature is detected from the gene list (``species='auto'``) or given.
2. **Name resolution** in that species: official symbol, Ensembl id, previous
   symbol, unambiguous synonym, then the same case-insensitively.
3. **Measured.** If the gene is in the table of its own species, that value is
   used.
4. **Estimated.** Otherwise the half-life is predicted, on the scale of the
   query species, from:

   * *ortholog* — the value of its ortholog(s) in the other species' table;
   * *paralogs* — measured same-species paralogs, weighted by protein identity;
   * *family* — the most specific curated HGNC gene group containing the gene;
   * *function* — the measured genes with the most similar Gene Ontology
     annotation (simGIC).

   Each source gives a (weighted geometric) mean, and the sources are combined
   as ``log t½ = m + Σ_s β_s · (log t½_s − m_s)``, where ``m`` is the median of
   the query species' table and ``m_s`` the median of the table the source was
   read from. The weights ``β_s`` are fitted by leave-one-out least squares on
   the measured genes of the query species (``neighbor_model.json``), with one
   set of weights for genes whose ortholog is measured and one for the others.
   For the ortholog source this is a linear recalibration of the other
   species' table onto the query species' table, learnt on their common
   orthologs.
5. **Global median** of the query species' table when nothing else is known.
"""
import json
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from CardamomOT.logging import get_logger

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "halflife")
SPECIES = ("mouse", "human")
OTHER = {"mouse": "human", "human": "mouse"}
REFERENCES = {
    "mouse": {
        "mrna": "Schwanhausser et al. 2011, Nature 473:337 (corrigendum 2013); mouse NIH3T3 fibroblasts",
        "protein": "Schwanhausser et al. 2011, Nature 473:337 (corrigendum 2013); mouse NIH3T3 fibroblasts",
    },
    "human": {
        "mrna": "RNADecayCafe v1.1 (Vock et al. 2025, bioRxiv 10.1101/2025.08.19.671151); "
                "11 human cell lines, dropout-normalised NR-seq",
        "protein": "Mathieson et al. 2018, Nat Commun 9:689; human primary B cells, NK cells, "
                   "hepatocytes, monocytes",
    },
}
QUANTITIES = {"mrna": "mrna_half_life_h", "protein": "prot_half_life_h"}
NEIGHBOR_SOURCES = ("paralog_high", "paralog_mid", "paralog_low", "family", "function")
SOURCES = ("ortholog",) + NEIGHBOR_SOURCES

# Neighbour settings — the same values are used when fitting the weights.
PARALOG_TOP = 5            # closest measured paralogs averaged
PARALOG_POWER = 4          # weight = (identity / 100) ** PARALOG_POWER
PARALOG_TIERS = (60.0, 40.0)  # % identity separating high / mid / low paralogs
FAMILY_MAX_SIZE = 300      # HGNC groups larger than this are not informative
FAMILY_MIN_MEASURED = 2
GO_TOP = 20                # most similar measured genes (simGIC) averaged
_UNIQUE_SUFFIX = re.compile(r"[-.]\d+$")  # var_names_make_unique suffixes
_ENSEMBL_VERSION = re.compile(r"^(ENS[A-Z]*G\d+)\.\d+$")


class HalfLifeDB:
    """In-memory view of the shipped half-life references and gene relations."""

    def __init__(self, data_dir=DATA_DIR, load_model=True):
        read = lambda name: pd.read_csv(os.path.join(data_dir, name), sep="\t", dtype=str,
                                        keep_default_na=False, na_values=[""])
        self.values, self.median, self.measurement_of, self.same_measurement = {}, {}, {}, {}
        for s in SPECIES:
            hl = read(f"halflife_{s}.tsv")
            self.values[s], self.median[s] = {}, {}
            for q, col in QUANTITIES.items():
                v = pd.to_numeric(hl[col], errors="coerce")
                self.values[s][q] = dict(zip(hl.gene_id[v.notna()], np.log(v[v.notna()])))
                self.median[s][q] = float(np.median(list(self.values[s][q].values())))
            # Genes whose value comes from the same measurement (e.g. a histone
            # protein group) must not predict each other when fitting.
            self.measurement_of[s] = dict(zip(hl.gene_id, hl.measurement_group))
            self.same_measurement[s] = _multimap(hl.measurement_group, hl.gene_id)

        self.index, self.index_upper = _gene_index(data_dir)
        info = read("gene_info.tsv.gz")
        self.symbol = dict(zip(info.gene_id, info.symbol))

        orth = read("orthologs_mouse_human.tsv.gz")
        self.orthologs = {"mouse": _multimap(orth.mgi_id, orth.hgnc_id),
                          "human": _multimap(orth.hgnc_id, orth.mgi_id)}

        groups = read("gene_groups_hgnc.tsv.gz")
        self.group_members = _multimap(groups.group_id, groups.hgnc_id)
        self.group_name = dict(zip(groups.group_id, groups.group_name))
        self.gene_groups = _multimap(groups.hgnc_id, groups.group_id)

        par = read("paralogs.tsv.gz")
        par = par.assign(identity=par.identity.astype(float)).sort_values("identity", ascending=False)
        go = read("go_neighbors.tsv.gz")
        go = go.assign(simgic=go.simgic.astype(float)).sort_values("simgic", ascending=False)
        self.paralogs, self.go_neighbors = {}, {}
        for s in SPECIES:
            p, g = par[par.species == s], go[go.species == s]
            self.paralogs[s] = _multimap(p.gene_id, zip(p.paralog_id, p.identity))
            self.go_neighbors[s] = _multimap(g.gene_id, zip(g.neighbor_id, g.simgic))

        path = os.path.join(data_dir, "neighbor_model.json")
        if not load_model:
            self.model = None
        elif os.path.exists(path):
            with open(path) as f:
                self.model = json.load(f)
        else:
            logger.warning("neighbor_model.json missing: fitting estimation weights now")
            self.model = fit_models(self)

    # ------------------------------------------------------------------
    # Species and names
    # ------------------------------------------------------------------
    def detect_species(self, genes):
        """Return ``('mouse' | 'human', match counts)`` from the nomenclature of ``genes``."""
        return _detect_species(self.index, genes)

    def resolve(self, gene, species):
        """Map a gene name to ``(gene_ids, how)`` in ``species`` nomenclature."""
        name = _ENSEMBL_VERSION.sub(r"\1", str(gene))
        for candidate in (name, _UNIQUE_SUFFIX.sub("", name)):
            for kind in ("symbol", "ensembl", "previous", "synonym"):
                ids = set(self.index.get((species, kind), {}).get(candidate, ()))
                # A previous symbol or synonym shared by several genes is ambiguous.
                if ids and (kind in ("symbol", "ensembl") or len(ids) == 1):
                    return sorted(ids), kind
            for kind in ("symbol", "previous", "synonym"):
                ids = set(self.index_upper.get((species, kind), {}).get(candidate.upper(), ()))
                if len(ids) == 1:
                    return sorted(ids), f"{kind} (case-insensitive)"
        return [], "unresolved"

    def ortholog_ids(self, ids, species):
        return sorted({o for i in ids for o in self.orthologs[species].get(i, ())})

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def estimates(self, ids, species, quantity, exclude=()):
        """Per-source log half-life estimates for genes ``ids`` of ``species``, with the genes used."""
        values = self.values[species][quantity]
        skip = set(ids) | set(exclude)
        out = {}

        other = OTHER[species]
        orth = [o for o in self.ortholog_ids(ids, species) if o in self.values[other][quantity]]
        if orth:
            est = float(np.mean([self.values[other][quantity][o] for o in orth]))
            out["ortholog"] = (est, [f"{self.symbol.get(o, o)} ({other})" for o in orth])

        cand = {}
        for g in ids:
            for p, ident in self.paralogs[species].get(g, ()):
                if p in values and p not in skip:
                    cand[p] = max(cand.get(p, 0.0), ident)
        if cand:
            top = sorted(cand.items(), key=lambda kv: -kv[1])[:PARALOG_TOP]
            w = np.array([(i / 100.0) ** PARALOG_POWER for _, i in top])
            est = float(np.sum(w * [values[p] for p, _ in top]) / w.sum())
            best = top[0][1]
            tier = ("paralog_high" if best >= PARALOG_TIERS[0] else
                    "paralog_mid" if best >= PARALOG_TIERS[1] else "paralog_low")
            out[tier] = (est, [f"{self.symbol.get(p, p)} ({i:.0f}%)" for p, i in top])

        # HGNC families are defined on human genes; mouse genes go through orthology.
        hgnc_ids = ids if species == "human" else self.ortholog_ids(ids, species)
        best = None
        for h in hgnc_ids:
            for gid in self.gene_groups.get(h, ()):
                members = self.group_members[gid]
                if len(members) > FAMILY_MAX_SIZE or (best and len(members) >= best[0]):
                    continue
                member_ids = members if species == "human" else self.ortholog_ids(members, "human")
                measured = {m for m in member_ids if m in values and m not in skip}
                if len(measured) >= FAMILY_MIN_MEASURED:
                    best = (len(members), gid, sorted(measured))
        if best:
            est = float(np.mean([values[m] for m in best[2]]))
            out["family"] = (est, [f"{self.group_name[best[1]]}: " +
                                   ",".join(self.symbol.get(m, m) for m in best[2][:8])])

        cand = {}
        for g in ids:
            for n, s in self.go_neighbors[species].get(g, ()):
                if n in values and n not in skip:
                    cand[n] = max(cand.get(n, 0.0), s)
        if cand:
            top = sorted(cand.items(), key=lambda kv: -kv[1])[:GO_TOP]
            w = np.array([s for _, s in top])
            est = float(np.sum(w * [values[n] for n, _ in top]) / w.sum())
            out["function"] = (est, [f"{self.symbol.get(n, n)} ({s:.2f})" for n, s in top[:8]])
        return out

    def _centered(self, estimates, species, quantity):
        """Source estimates minus the median of the table each one was read from."""
        centers = {s: self.median[species][quantity] for s in NEIGHBOR_SOURCES}
        centers["ortholog"] = self.median[OTHER[species]][quantity]
        return {s: e - centers[s] for s, (e, _) in estimates.items()}

    def combine(self, estimates, species, quantity):
        pattern = "with_ortholog" if "ortholog" in estimates else "without_ortholog"
        beta = self.model[species][quantity][pattern]["beta"]
        centered = self._centered(estimates, species, quantity)
        return self.median[species][quantity] + sum(beta[s] * d for s, d in centered.items())

    def lookup(self, genes, species="auto"):
        """Half-lives (h) and degradation rates (h⁻¹) for ``genes``, one row per gene."""
        genes = [str(g) for g in genes]
        if species == "auto":
            species, hits = self.detect_species(genes)
            logger.info("detected species '%s' (official-name matches: %s)", species, hits)
        elif species not in SPECIES:
            raise ValueError(f"species must be 'auto', 'mouse' or 'human', got {species!r}")

        rows = []
        for gene in genes:
            ids, how = self.resolve(gene, species)
            row = {"gene": gene, "species": species, "match": how,
                   "matched_symbol": ",".join(self.symbol.get(i, i) for i in ids),
                   "ortholog": ",".join(self.symbol.get(o, o) for o in self.ortholog_ids(ids, species))}
            for q in QUANTITIES:
                measured = [self.values[species][q][i] for i in ids if i in self.values[species][q]]
                if measured:
                    log_hl, source, detail = float(np.mean(measured)), "measured", ""
                else:
                    est = self.estimates(ids, species, q)
                    if est:
                        log_hl = self.combine(est, species, q)
                        source = "estimated:" + "+".join(s for s in SOURCES if s in est)
                        detail = " | ".join(f"{s}: {', '.join(n)}" for s, (_, n) in est.items())
                    else:
                        log_hl, source, detail = self.median[species][q], "global_median", ""
                row[f"{q}_half_life_h"] = float(np.exp(log_hl))
                row[f"{q}_source"] = source
                row[f"{q}_neighbors"] = detail
            rows.append(row)
        df = pd.DataFrame(rows)
        df["d0"] = np.log(2) / df["mrna_half_life_h"]
        df["d1"] = np.log(2) / df["protein_half_life_h"]
        return df


def _detect_species(index, genes):
    genes = [_ENSEMBL_VERSION.sub(r"\1", str(g)) for g in genes]
    hits = {s: sum(g in index[s, "symbol"] or g in index[s, "ensembl"] for g in genes) for s in SPECIES}
    if hits["mouse"] == hits["human"]:
        # Nothing official matched: fall back on synonyms, then on letter case.
        hits = {s: sum(g in index[s, "synonym"] for g in genes) for s in SPECIES}
    if hits["mouse"] == hits["human"]:
        lower = np.mean([bool(re.search(r"[a-z]", g.replace("orf", ""))) for g in genes]) if genes else 0
        species = "mouse" if lower > 0.5 else "human"
        logger.warning("species not identifiable from gene names, guessing '%s' from letter case", species)
        return species, hits
    return max(hits, key=hits.get), hits


def detect_species(genes, data_dir=DATA_DIR):
    """
    Detect whether ``genes`` (e.g. ``adata.var_names``) follow mouse (MGI, ``Gata1``)
    or human (HGNC, ``GATA1``) nomenclature; Ensembl ids (``ENSMUSG``/``ENSG``) work too.

    Returns:
        ``(species, hits)`` with ``species`` in ``{'mouse', 'human'}`` and ``hits`` the
        number of genes matching each species' official symbols / Ensembl ids.
    """
    return _detect_species(_gene_index(data_dir)[0], genes)


@lru_cache(maxsize=2)
def _gene_index(data_dir=DATA_DIR):
    """Gene names → ids per (species, kind), exact and upper-cased."""
    idx = pd.read_csv(os.path.join(data_dir, "gene_index.tsv.gz"), sep="\t", dtype=str,
                      keep_default_na=False, na_values=[""])
    groups_by_key = list(idx.groupby(["species", "kind"]))
    index = {key: _multimap(sub.name, sub.gene_id) for key, sub in groups_by_key}
    index_upper = {key: _multimap([n.upper() for n in sub.name], sub.gene_id) for key, sub in groups_by_key}
    return index, index_upper


def _multimap(keys, values):
    """``{key: [values...]}`` preserving input order (fast alternative to groupby)."""
    out = {}
    for k, v in zip(keys, values):
        out.setdefault(k, []).append(v)
    return out


@lru_cache(maxsize=2)
def get_halflife_db(data_dir=DATA_DIR):
    return HalfLifeDB(data_dir)


def _loo_design(db, species, quantity):
    """Leave-one-out centred source estimates for every measured gene of ``species``."""
    values = db.values[species][quantity]
    genes = sorted(values)
    X = np.zeros((len(genes), len(SOURCES)))
    has = np.zeros((len(genes), len(SOURCES)), dtype=bool)
    for i, g in enumerate(genes):
        group = db.same_measurement[species].get(db.measurement_of[species].get(g), ())
        est = db.estimates([g], species, quantity, exclude=group)
        for s, d in db._centered(est, species, quantity).items():
            X[i, SOURCES.index(s)] = d
            has[i, SOURCES.index(s)] = True
    y = np.array([values[g] for g in genes]) - db.median[species][quantity]
    return X, has, y


def _fit(X, y, n_folds=5, seed=0):
    """Least-squares weights and their 5-fold cross-validated predictions."""
    folds = np.random.default_rng(seed).integers(0, n_folds, len(y))
    pred = np.zeros_like(y)
    for k in range(n_folds):
        tr = folds != k
        pred[~tr] = X[~tr] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    return np.linalg.lstsq(X, y, rcond=None)[0], pred


def fit_models(db):
    """Fit the source weights β per species, quantity and ortholog availability."""
    rmse = lambda e: round(float(np.sqrt(np.mean(e ** 2))), 4)
    corr = lambda a, b: round(float(np.corrcoef(a, b)[0, 1]), 3)
    model = {}
    for s in SPECIES:
        model[s] = {}
        for q in QUANTITIES:
            X, has, y = _loo_design(db, s, q)
            w_orth = has[:, 0]
            nb_cols = [SOURCES.index(c) for c in NEIGHBOR_SOURCES]
            # Genes whose ortholog is measured: all sources. Otherwise: neighbours
            # only, fitted on every measured gene as if its ortholog were unknown.
            beta_with, pred_with = _fit(X[w_orth], y[w_orth])
            beta_without, pred_without = _fit(X[:, nb_cols], y)
            has_any = has[:, nb_cols].any(axis=1)
            model[s][q] = {
                "reference": REFERENCES[s][q],
                "n_measured": int(len(y)),
                "median_half_life_h": round(float(np.exp(db.median[s][q])), 3),
                "with_ortholog": {
                    "beta": {c: round(float(b), 4) for c, b in zip(SOURCES, beta_with)},
                    "n": int(w_orth.sum()),
                    "cv_pearson_r": corr(pred_with, y[w_orth]),
                    "cv_rmse_log": rmse(pred_with - y[w_orth]),
                    "global_median_rmse_log": rmse(y[w_orth]),
                    "ortholog_alone_pearson_r": corr(X[w_orth, 0], y[w_orth]),
                },
                "without_ortholog": {
                    "beta": {c: round(float(b), 4) for c, b in zip(NEIGHBOR_SOURCES, beta_without)},
                    "n": int(len(y)),
                    "cv_pearson_r": corr(pred_without[has_any], y[has_any]),
                    "cv_rmse_log": rmse(pred_without - y),
                    "global_median_rmse_log": rmse(y),
                },
                "coverage": {c: round(float(has[:, i].mean()), 3) for i, c in enumerate(SOURCES)},
            }
    return model

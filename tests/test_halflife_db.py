"""
Tests for the literature half-life lookup (CardamomOT.inference.halflife_db).

Run with:
    python -m pytest tests/test_halflife_db.py -v
or just:
    python tests/test_halflife_db.py
"""
import numpy as np

from CardamomOT.inference.halflife_db import detect_species, get_halflife_db
from CardamomOT.inference.pretreatment import extract_degradation_rates


def test_units_are_hours():
    """Actb is measured in Schwanhäusser 2011 (corrected table): 11.2 h mRNA, 61.9 h protein."""
    df = get_halflife_db().lookup(["Actb"], species="mouse")
    assert df.mrna_source[0] == "measured" and df.protein_source[0] == "measured"
    assert np.isclose(df.mrna_half_life_h[0], 11.21, atol=0.01)
    assert np.isclose(df.protein_half_life_h[0], 61.92, atol=0.01)
    assert np.isclose(df.d0[0], np.log(2) / 11.21, rtol=1e-3)  # h^-1


def test_reference_medians():
    """Mouse medians match NetworkModel defaults (ln2/9 h, ln2/46 h), from the same study."""
    db = get_halflife_db()
    assert 8 < np.exp(db.median["mouse"]["mrna"]) < 11
    assert 40 < np.exp(db.median["mouse"]["protein"]) < 55
    assert 1.5 < np.exp(db.median["human"]["mrna"]) < 4       # RNADecayCafe, dropout-normalised
    assert 50 < np.exp(db.median["human"]["protein"]) < 100   # Mathieson 2018, primary cells


def test_species_detection():
    db = get_halflife_db()
    assert db.detect_species(["Gata1", "Sox2", "Nanog", "Actb"])[0] == "mouse"
    assert db.detect_species(["GATA1", "SOX2", "NANOG", "ACTB"])[0] == "human"
    assert db.detect_species(["ENSMUSG00000031162", "ENSMUSG00000074637"])[0] == "mouse"
    assert db.detect_species(["ENSG00000102145.15", "ENSG00000181449"])[0] == "human"


def test_standalone_detect_species():
    """Used by get_proliferation_rates without loading the half-life tables."""
    assert detect_species(["Mki67", "Top2a", "Casp3"])[0] == "mouse"
    assert detect_species(["MKI67", "TOP2A", "CASP3"])[0] == "human"


def test_each_species_uses_its_own_table():
    db = get_halflife_db()
    human = db.lookup(["ACTB"], species="human")
    assert human.mrna_source[0] == "measured" and human.protein_source[0] == "measured"
    assert np.isclose(np.log(human.mrna_half_life_h[0]), db.values["human"]["mrna"]["HGNC:132"])
    assert human.ortholog[0] == "Actb"


def test_ortholog_is_recalibrated_on_query_scale():
    """A mouse gene measured only in the human table gets a value on the mouse scale."""
    db = get_halflife_db()
    mouse_meas, human_meas = db.values["mouse"]["mrna"], db.values["human"]["mrna"]
    orth = db.orthologs["mouse"]
    gene = next(g for g, hs in orth.items() if g not in mouse_meas and any(h in human_meas for h in hs))
    df = db.lookup([db.symbol[gene]], species="mouse")
    assert "ortholog" in df.mrna_source[0]
    beta = db.model["mouse"]["mrna"]["with_ortholog"]["beta"]["ortholog"]
    assert 0 < beta < 1
    # Human mRNA half-lives are ~4x shorter; the estimate must not inherit that scale.
    assert df.mrna_half_life_h[0] > 2 * np.exp(db.median["human"]["mrna"])


def test_name_resolution():
    db = get_halflife_db()
    assert db.resolve("Hist1h4a", "mouse")[1] == "synonym"          # renamed H4c1 in 2020
    assert db.resolve("Actb-1", "mouse")[1] == "symbol"             # var_names_make_unique suffix
    assert db.resolve("NKX2-1", "human")[1] == "symbol"             # real hyphen kept
    assert db.resolve("ENSG00000136997.15", "human")[1] == "ensembl"
    assert db.symbol[db.resolve("OCT4", "human")[0][0]] == "POU5F1"
    assert db.resolve("NotAGene123", "mouse") == ([], "unresolved")


def test_unmeasured_gene_uses_biological_neighbors():
    """Gata1 was not measured in mouse: its value must come from related genes, not the median."""
    db = get_halflife_db()
    df = db.lookup(["Gata1"], species="mouse")
    assert df.mrna_source[0].startswith("estimated:")
    assert "function" in df.mrna_source[0]
    assert df.mrna_neighbors[0]
    # Transcription factors are less stable than the typical gene.
    assert df.protein_half_life_h[0] < np.exp(db.median["mouse"]["protein"])


def test_unknown_gene_falls_back_to_median():
    db = get_halflife_db()
    df = db.lookup(["NotAGene123"], species="mouse")
    assert df.mrna_source[0] == "global_median"
    assert np.isclose(df.mrna_half_life_h[0], np.exp(db.median["mouse"]["mrna"]))


def test_estimation_beats_global_median():
    for species, by_q in get_halflife_db().model.items():
        for q, m in by_q.items():
            for pattern in ("with_ortholog", "without_ortholog"):
                assert m[pattern]["cv_rmse_log"] < m[pattern]["global_median_rmse_log"], (species, q, pattern)
                assert m[pattern]["cv_pearson_r"] > 0.3, (species, q, pattern)


def test_extract_degradation_rates_shape():
    genes = ["Gata1", "Actb", "Sox2"]
    deg, details = extract_degradation_rates(genes, return_details=True)
    assert deg.shape == (2, 3)
    assert np.all(deg > 0) and np.all(np.isfinite(deg))
    assert list(details.gene) == genes


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")

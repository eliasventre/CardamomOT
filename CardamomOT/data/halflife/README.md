# Half-life reference data

Tables read by `CardamomOT.inference.halflife_db` (used by `get_degradation_rates.py`)
to assign mRNA (`d0`) and protein (`d1`) degradation rates, in **hour⁻¹**
(`d = ln 2 / t½`, with `t½` in hours — the time unit of `adata.obs['time']`).
Rebuild them with `python CardamomOT/data/halflife/build_resources.py --raw <download_dir>`.

## Reference half-lives (hours)

| Species | Quantity | Source | Genes | Median t½ |
|---|---|---|---|---|
| mouse | mRNA | Schwanhäusser et al. 2011, *Nature* 473:337, Suppl. Table 3 as replaced by the 2013 corrigendum (*Nature* 495:126) — NIH3T3 fibroblasts | 4,695 | 9.9 h |
| mouse | protein | same | 5,023 | 47.9 h |
| human | mRNA | RNADecayCafe v1.1 (Vock et al. 2025, bioRxiv [10.1101/2025.08.19.671151](https://doi.org/10.1101/2025.08.19.671151); [Zenodo 16884513](https://zenodo.org/records/16884513), CC-BY 4.0) — uniformly reprocessed SLAM-seq/TimeLapse-seq, 11 cell lines; dropout-normalised half-lives | 16,998 | 2.5 h |
| human | protein | Mathieson et al. 2018, *Nat Commun* 9:689, Suppl. Data 2 ("high qual") — primary B cells, NK cells, hepatocytes, monocytes | 6,843 | 81 h |

When a source reports several cell types / lines (or replicates), each one is
first recentred on a common median, then the per-gene geometric mean is taken
over those where the gene was measured. Cell types differ in overall turnover
(NK cells ~2× slower than the other three), so without recentring a gene
measured in only some of them would inherit their scale.

The two species' tables are **not on the same scale**: they come from different
cell types and methods (human NR-seq mRNA half-lives are ~4× shorter than
Schwanhäusser's; human primary-cell protein half-lives ~1.7× longer). Each
dataset is therefore put on the scale of its own species' table.

## Files

| File | Content | Source |
|---|---|---|
| `halflife_mouse.tsv`, `halflife_human.tsv` | half-lives (h) per MGI / HGNC gene | see above |
| `gene_index.tsv.gz` | symbols, synonyms, previous symbols, Ensembl ids → MGI / HGNC ids | MGI `MRK_List2.rpt`, `MRK_ENSEMBL.rpt`; HGNC complete set |
| `gene_info.tsv.gz` | MGI / HGNC id → official symbol | same |
| `orthologs_mouse_human.tsv.gz` | mouse ↔ human orthologs | MGI `HOM_MouseHumanSequence.rpt` (Alliance of Genome Resources) |
| `gene_groups_hgnc.tsv.gz` | curated gene families | HGNC gene groups |
| `paralogs.tsv.gz` | within-species paralogs with protein identity (≥ 20 %), towards measured genes | Ensembl BioMart |
| `go_neighbors.tsv.gz` | 30 most GO-similar measured genes per gene (simGIC), per species | MGI / GOA human annotations + `go-basic.obo` |
| `neighbor_model.json` | weights of the estimation sources and their cross-validated accuracy | fitted by `build_resources.py` |

## How a gene gets its half-life

1. The species is detected from the nomenclature (`Gata1` → mouse, `GATA1` → human),
   or forced with `--species`.
2. The name is resolved in that species (official symbol, Ensembl id, previous
   symbol, unambiguous synonym, then case-insensitive).
3. If the gene is in its own species' table, that value is used (`measured`).
4. Otherwise it is **estimated on the scale of its own species** from:
   - *ortholog*: the value of its ortholog(s) in the other species' table;
   - *paralogs*: measured same-species paralogs, weighted by protein identity
     (tiers ≥ 60 %, 40–60 %, < 40 %);
   - *family*: the most specific HGNC gene family;
   - *function*: the 20 measured genes with the most similar GO annotation.

   Sources are combined as `log t½ = m + Σ β_s (log t½_s − m_s)`, where `m` is the
   median of the query species' table and `m_s` the median of the table the
   source comes from. The weights `β_s` are fitted by least squares
   (leave-one-out) on the measured genes of the query species, separately for
   genes whose ortholog is measured and for the others. For the ortholog this is
   a linear recalibration of the other species' table onto the query species'
   scale, learnt on their common orthologs — mouse data use the human table
   recalibrated on Schwanhäusser, human data use Schwanhäusser recalibrated on
   the human tables.
5. Otherwise the median of the query species' table is used.

`Data/degradation_rates_report.csv` lists, for each gene of a project, the
match, the ortholog, the half-lives, the source used and the genes behind it.

## Accuracy of the estimation

Five-fold cross-validation on measured genes (`neighbor_model.json`), Pearson r
of log t½ and RMSE of log t½ (in parentheses: RMSE when using the median):

| | ortholog measured | ortholog not measured |
|---|---|---|
| mouse mRNA | r = 0.65, 0.36 (0.46) | r = 0.41, 0.42 (0.46) |
| mouse protein | r = 0.65, 0.73 (0.96) | r = 0.51, 0.89 (1.03) |
| human mRNA | r = 0.68, 0.56 (0.81) | r = 0.58, 0.69 (0.83) |
| human protein | r = 0.69, 0.51 (0.71) | r = 0.53, 0.66 (0.78) |

The ortholog alone correlates at r ≈ 0.6 across species; paralogs, family and
function add to it. Estimates are shrunk towards the median: they point in the
right direction (e.g. transcription factors come out less stable than average)
but do not reproduce extreme values.

## Sources considered and not used

- Li et al. 2021 (*Mol Cell*), cycloheximide-chase protein half-lives in U2OS,
  HEK293T, HCT116, RPE1 (1,476 genes, 831 absent from Mathieson, mostly
  short-lived regulators). On the 645 genes shared with Mathieson it correlates
  at r = 0.28 overall and r ≈ 0 for its short-lived (< 8 h) proteins; added as an
  extra recalibrated source it barely changed the cross-validated error
  (RMSE 0.645 → 0.643).
- An RNA-seq / PRO-seq ratio (relative mRNA stability, Blumberg et al. 2021
  method): r = 0.68 with RNADecayCafe, but only 16 genes absent from it.

## Caveats

- Turnover varies between cell types; these are references from a few cell
  types, not measurements in your cells.
- Proteins quantified by mass spectrometry are biased towards abundant,
  long-lived ones.
- Genes with no measured ortholog, paralog, family or GO annotation (mostly
  lncRNAs) get the median.

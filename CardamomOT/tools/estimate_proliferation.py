"""
Estimate per-cell net proliferation rates (birth - death) from curated
proliferation/death/senescence gene signatures.

Follows the growth-rate estimation approach used by Waddington-OT
(Schiebinger et al. 2019, Cell) and moscot: score each cell against a
proliferation gene set and a death gene set, then map each score to a
birth/death rate via a shifted logistic curve. The net rate `birth - death`
is what CardamomOT actually consumes (as `adata.obs['proliferation_net_rate']`) to
reweight OT marginals during trajectory inference.

Named `death_genes` rather than `apoptosis_genes`: apoptosis is moscot's own
default choice of death-signal gene set, but any process that removes cells
from the population (necrosis, senescence-driven clearance, etc.) is a valid
substitute here — pass whatever gene set is the right proxy for cell loss in
your system.

--------------------------------------------------------------------------
2026-07 update: senescence/quiescence gating
--------------------------------------------------------------------------
`MOSCOT_PROLIFERATION_GENES` / `MOSCOT_DEATH_GENES` below are copied
verbatim from moscot's shipped marker lists (``moscot.utils.data.
proliferation_markers`` / ``apoptosis_markers``, as of 2026-07; see
https://github.com/theislab/moscot/tree/main/src/moscot/utils/_data) and are
kept unmodified for strict parity / reference. Per moscot's own
documentation: proliferation markers come from Tirosh et al. 2016 (human:
Science; mouse: Nature) cell-cycle gene lists; death markers are MSigDB
Hallmark Apoptosis for human, but MSigDB Hallmark P53 Pathway for mouse
(moscot does not use a symmetric gene set across species).

`DEFAULT_PROLIFERATION_GENES` / `DEFAULT_DEATH_GENES` / `DEFAULT_SENESCENCE_GENES`
are CardamomOT's own curated defaults, derived from the moscot lists above,
and are what `estimate_growth_rates` actually uses unless you override them.
They differ from moscot's raw lists in three ways, motivated by using this
module on stress-perturbation data (cell-cycle arrest / DNA-damage response /
senescence WITHOUT death, which the raw moscot mouse "death" set — Hallmark
P53 Pathway — cannot distinguish from real apoptosis):

1. Human death genes no longer overlap with the proliferation set (`TOP2A`,
   `HMGB2` removed — these are cell-cycle markers, not apoptosis markers,
   and their presence in both sets waters down the contrast between the two
   scores).
2. Human death genes no longer include genes that are mainly *anti*-
   apoptotic/pro-survival despite being part of the Hallmark Apoptosis gene
   set (`BCL2L1`, `BCL2L2`, `MCL1`, `XIAP`, `CFLAR`, `BIRC3`, `CLU` removed).
3. The mouse Hallmark P53 Pathway list is split into two signatures instead
   of being used as-is for "death":
   - A short "apoptosis core" subset of genes with a direct, literature-
     supported role in triggering/executing apoptosis (`Bax`, `Bak1`,
     `Fas`, `Casp1`, `Apaf1`, `Perp`, `Aen`, `Ei24`, `Ddit3`, `Dram1`).
   - Everything else in the Hallmark P53 list — cell-cycle arrest,
     DNA-damage response, senescence genes (`Cdkn1a`, `Cdkn2a`, `Cdkn2b`,
     `Rb1`, `Gadd45a`, `Ddb2`, `Xpc`, ...) — becomes
     `DEFAULT_SENESCENCE_GENES["mouse"]`. `DEFAULT_SENESCENCE_GENES["human"]`
     is obtained by upper-casing this same mouse list, using the same
     human<->mouse capitalization convention moscot itself relies on for
     the Tirosh proliferation genes. This is a reasonable approximation,
     NOT a verified human Hallmark P53 Pathway extraction — a handful of
     older/renamed mouse symbols (e.g. `Gnb2l1`, historically renamed to
     `RACK1` in some human annotations; `Hist3h2a` legacy histone naming)
     may not match current human gene symbols. Genes absent from
     `adata.var_names` are silently dropped by `score_gene_sets`
     regardless, so this only affects how much of the signature is
     actually usable, not correctness.
4. The mouse "apoptosis core" above is extended with real human->mouse
   orthologs of the cleaned human death set, rather than left at 10 genes.
   These orthologs come from **HCOP** (HGNC Comparison of Orthology
   Predictions, Eyre et al. 2007; Yates et al. 2021 — the standard curated
   cross-method ortholog resource, not a naive case-conversion), filtered
   to `evidence >= 3` (number of independent prediction methods agreeing).
   One 1:1 pick was made per human gene (highest-evidence mouse candidate),
   with a single manual override: `GSTM1 -> Gstm1` (the correctly *named*
   ortholog within a heavily duplicated glutathione-S-transferase gene
   cluster; `Gstm2` had nominally higher evidence but is a distinct
   paralog, not the 1:1 ortholog).
   Cross-checking these orthologs against `DEFAULT_SENESCENCE_GENES`
   surfaced 17 more human genes (`APP`, `ATF3`, `BMP2`, `BTG2`, `CCND2`,
   `CDKN1A`, `F2R`, `FDXR`, `GADD45A`, `HMOX1`, `IER3`, `IL1A`, `JUN`,
   `RETSAT`, `SAT1`, `TAP1`, `TXNIP`) whose mouse orthologs already sit in
   the arrest/senescence list — i.e. Hallmark Apoptosis and Hallmark P53
   Pathway independently agree these are stress/arrest genes, not death
   effectors. This is exactly the kind of noise flagged for the earlier,
   manually-removed genes (`TOP2A`/`HMGB2`, the anti-apoptotic regulators),
   so these 17 were removed from `DEFAULT_DEATH_GENES["human"]` too (kept
   only in `DEFAULT_SENESCENCE_GENES`), bringing the final human death set
   to 135 genes and giving `DEFAULT_DEATH_GENES["mouse"]` 141 genes with
   zero residual overlap against proliferation or senescence in either
   species (checked programmatically, see tests below).
5. `Pcna` (DNA-damage-response role in Hallmark P53 Pathway, but also a
   core Tirosh cell-cycle gene) is excluded from `DEFAULT_SENESCENCE_GENES`
   for the same reason as point 1 above.

The senescence score feeds into a soft **gate on the birth (proliferation)
rate only**: `birth_effective = birth * (1 - gate(senescence_score))`, so a
strongly senescent/quiescent cell gets `proliferation_net_rate -> 0` even if
it still carries residual cell-cycle gene expression, while an independently
elevated death score is left untouched (so real apoptosis occurring in a
cell that also happens to score on a couple of shared stress genes is not
masked). Disable via `senescence_gating=False` to recover the original,
un-gated moscot-style `birth - death`. The gate's `gamma_center`/
`gamma_width` are NOT calibrated against any reference (unlike `beta_*`/
`delta_*`, which are moscot's own published defaults) — they are a
starting point only; inspect your own senescence score distribution and
tune them (see `estimate_growth_rates` docstring).
--------------------------------------------------------------------------

`adata` is expected to hold raw counts, as everywhere else in CardamomOT.
`score_gene_sets`/`estimate_growth_rates` internally normalise (total-count)
and log1p-transform a *copy* of the data before scoring — `scanpy.tl.score_genes`
assumes that scale, same as moscot's own tutorials — and never modify or
return the transformed values; only the original raw `adata` is ever read
elsewhere in the pipeline or written back to disk.
"""
import warnings
import numpy as np

# --------------------------------------------------------------------------
# Raw moscot marker lists (verbatim, unmodified) — kept for strict parity /
# reference, and as the source from which the curated DEFAULT_* sets below
# are derived.
# --------------------------------------------------------------------------

MOSCOT_PROLIFERATION_GENES = {
    "human": sorted([
        "ANLN", "ANP32E", "ATAD2", "AURKA", "AURKB", "BIRC5", "BLM", "BRIP1",
        "BUB1", "CASP8AP2", "CBX5", "CCNB2", "CCNE2", "CDC20", "CDC25C",
        "CDC45", "CDC6", "CDCA2", "CDCA3", "CDCA7", "CDCA8", "CDK1", "CENPA",
        "CENPE", "CENPF", "CHAF1B", "CKAP2", "CKAP2L", "CKAP5", "CKS1B",
        "CKS2", "CLSPN", "CTCF", "DLGAP5", "DSCC1", "DTL", "E2F8", "ECT2",
        "EXO1", "FAM64A", "FEN1", "G2E3", "GAS2L3", "GINS2", "GMNN", "GTSE1",
        "HELLS", "HJURP", "HMGB2", "HMMR", "HN1", "KIF11", "KIF20B", "KIF23",
        "KIF2C", "LBR", "MCM2", "MCM4", "MCM5", "MCM6", "MKI67", "MLF1IP",
        "MSH2", "NASP", "NCAPD2", "NDC80", "NEK2", "NUF2", "NUSAP1", "PCNA",
        "POLA1", "POLD3", "PRIM1", "PSRC1", "RAD51", "RAD51AP1", "RANGAP1",
        "RFC2", "RPA2", "RRM1", "RRM2", "SLBP", "SMC4", "TACC3", "TIPIN",
        "TMPO", "TOP2A", "TPX2", "TTK", "TUBB4B", "TYMS", "UBE2C", "UBR7",
        "UHRF1", "UNG", "USP1", "WDR76",
    ]),
    "mouse": sorted([
        "Mcm4", "Smc4", "Gtse1", "Ttk", "Rangap1", "Ccnb2", "Cenpa", "Cenpe",
        "Cdca8", "Ckap2", "Rad51", "Pcna", "Ube2c", "Lbr", "Cenpf", "Birc5",
        "Dtl", "Dscc1", "Cbx5", "Usp1", "Hmmr", "Wdr76", "Ung", "Hn1", "Cks2",
        "Kif20b", "Cdk1", "Slbp", "Aurkb", "Kif11", "Cks1b", "Blm", "Msh2",
        "Gas2l3", "Tyms", "Hjurp", "Hells", "Prim1", "Uhrf1", "Ndc80", "Mcm6",
        "Rrm1", "Mlf1ip", "Top2a", "Hmgb2", "Ccne2", "G2e3", "Tmpo", "Nusap1",
        "Ncapd2", "Mcm2", "Kif2c", "Cdca2", "Nasp", "Gmnn", "Cdc6", "Pold3",
        "Ckap2l", "Fam64a", "Ubr7", "Fen1", "Bub1", "Brip1", "Atad2", "Psrc1",
        "Rrm2", "Tipin", "Casp8ap2", "Tubb4b", "Kif23", "Exo1", "Rfc2",
        "Pola1", "Mki67", "Tpx2", "Aurka", "Anln", "Chaf1b", "Tacc3", "Mcm5",
        "Anp32e", "Dlgap5", "Ect2", "Nuf2", "Cdc45", "Ckap5", "Ctcf", "Clspn",
        "Cdca7", "Cdca3", "Rpa2", "Gins2", "E2f8", "Cdc25c", "Nek2", "Cdc20",
        "Rad51ap1",
    ]),
}

MOSCOT_DEATH_GENES = {
    # MSigDB Hallmark Apoptosis
    "human": sorted([
        "ADD1", "AIFM3", "ANKH", "ANXA1", "APP", "ATF3", "AVPR1A", "BAX",
        "BCAP31", "BCL10", "BCL2L1", "BCL2L10", "BCL2L11", "BCL2L2", "BGN",
        "BID", "BIK", "BIRC3", "BMF", "BMP2", "BNIP3L", "BRCA1", "BTG2",
        "BTG3", "CASP1", "CASP2", "CASP3", "CASP4", "CASP6", "CASP7", "CASP8",
        "CASP9", "CAV1", "CCNA1", "CCND1", "CCND2", "CD14", "CD2", "CD38",
        "CD44", "CD69", "CDC25B", "CDK2", "CDKN1A", "CDKN1B", "CFLAR", "CLU",
        "CREBBP", "CTH", "CTNNB1", "CYLD", "DAP", "DAP3", "DCN", "DDIT3",
        "DFFA", "DIABLO", "DNAJA1", "DNAJC3", "DNM1L", "DPYD", "EBP", "EGR3",
        "EMP1", "ENO2", "ERBB2", "ERBB3", "EREG", "ETF1", "F2", "F2R", "FAS",
        "FASLG", "FDXR", "FEZ1", "GADD45A", "GADD45B", "GCH1", "GNA15",
        "GPX1", "GPX3", "GPX4", "GSN", "GSR", "GSTM1", "GUCY2D", "H1-0",
        "HGF", "HMGB2", "HMOX1", "HSPB1", "IER3", "IFITM3", "IFNB1",
        "IFNGR1", "IGF2R", "IGFBP6", "IL18", "IL1A", "IL1B", "IL6", "IRF1",
        "ISG20", "JUN", "KRT18", "LEF1", "LGALS3", "LMNA", "LUM", "MADD",
        "MCL1", "MGMT", "MMP2", "NEDD9", "NEFH", "PAK1", "PDCD4", "PDGFRB",
        "PEA15", "PLAT", "PLCB2", "PLPPR4", "PMAIP1", "PPP2R5B", "PPP3R1",
        "PPT1", "PRF1", "PSEN1", "PSEN2", "PTK2", "RARA", "RELA", "RETSAT",
        "RHOB", "RHOT2", "RNASEL", "ROCK1", "SAT1", "SATB1", "SC5D",
        "SLC20A1", "SMAD7", "SOD1", "SOD2", "SPTAN1", "SQSTM1", "TAP1",
        "TGFB2", "TGFBR3", "TIMP1", "TIMP2", "TIMP3", "TNF", "TNFRSF12A",
        "TNFSF10", "TOP2A", "TSPO", "TXNIP", "VDAC2", "WEE1", "XIAP",
    ]),
    # MSigDB Hallmark P53 Pathway (moscot uses this, not Hallmark Apoptosis,
    # as its mouse "apoptosis" set)
    "mouse": sorted([
        "Ercc5", "Serpinb5", "Inhbb", "Steap3", "Btg2", "Phlda3", "Tnni1",
        "Rgs16", "Ier5", "Slc19a2", "Adck3", "Ephx1", "Ptpn14", "Atf3",
        "Notch1", "Rxra", "Ralgds", "Ak1", "Stom", "Ddb2", "Cd82", "Il1a",
        "Pcna", "Bmp2", "Trib3", "Procr", "Blcap", "Ada", "Fgf13", "Irak1",
        "Tspyl2", "Sat1", "Zmat3", "Hspa4l", "Slc7a11", "Tm4sf1", "Rap2b",
        "Fbxw7", "S100a4", "S100a10", "Txnip", "Nhlh2", "Dnttip2", "Clca2",
        "Wwp1", "Klf4", "Ikbkap", "Cdkn2a", "Cdkn2b", "Jun", "Slc35d1",
        "Plk3", "Rnf19b", "Sfn", "Fuca1", "Epha2", "Wrap73", "Mxd4", "Rchy1",
        "Iscu", "Triap1", "Prkab1", "Trafd1", "Pom121", "Pdgfa", "Gadd45a",
        "Vamp8", "Retsat", "Tprkb", "Tgfa", "Mxd1", "Sec61a1", "Xpc",
        "Ccnd2", "H2afj", "Ldhb", "Lrmp", "Tm7sf3", "Tgfb1", "Sertad3",
        "Cebpa", "Klk8", "Bax", "Ppp1r15a", "Rpl18", "Aen", "Rrp8", "Ccp110",
        "Nupr1", "Ptpre", "Hras", "Eps8l2", "Ctsd", "Cd81", "Perp", "Rps12",
        "Tpd52l1", "Sesn1", "Foxo3", "Ddit4", "Zfp365", "Prmt2", "Mknk2",
        "Dram1", "Apaf1", "Btg1", "Mdm2", "Ddit3", "Gls2", "Dgka",
        "Cdkn2aip", "Hmox1", "Rrad", "Cdh13", "Osgin1", "Cgrrf1", "Abhd4",
        "Kif13b", "Rb1", "Nudt15", "Tsc22d1", "Casp1", "St14", "Ei24",
        "Vwa5a", "Zbtb16", "Rps27l", "Mapkapk3", "Ip6k2", "Tcn2", "Lif",
        "Upp1", "Ccng1", "Cyfip2", "Gnb2l1", "Hint1", "Gm2a", "Hist3h2a",
        "Alox8", "Trp53", "Tax1bp3", "Traf4", "Cdk5r1", "Ppm1d", "Rad51c",
        "Tob1", "Krt17", "Hexim1", "Fdxr", "Itgb4", "Sphk1", "Rhbdf2",
        "Baiap2", "Dcxr", "Hist1h1c", "Ninj1", "Nol8", "F2r", "Ankra2",
        "Plk2", "Sdc1", "Gpx2", "Zfp36l1", "Fos", "Ccnk", "Jag2", "Ndrg1",
        "Pmm1", "Plxnb2", "Vdr", "Csrnp2", "Acvr1b", "Sp1", "Abat", "Socs1",
        "Abcc5", "Trp63", "Fam162a", "App", "Rab40c", "Bak1", "Def6",
        "Cdkn1a", "Tap1", "Ier3", "Polh", "Ccnd3", "Hbegf", "Hdac3", "Rad9a",
        "Ctsf", "Slc3a2", "Fas",
    ]),
}

# --------------------------------------------------------------------------
# Curated defaults derived from the moscot lists above (see module
# docstring for the rationale behind each change).
# --------------------------------------------------------------------------

# (1) genes shared between the human proliferation and death lists —
# cell-cycle markers, not apoptosis markers.
_HUMAN_DEATH_OVERLAP_WITH_PROLIF = {"TOP2A", "HMGB2"}

# (2) genes in Hallmark Apoptosis that are mainly anti-apoptotic/pro-survival
# regulators rather than death effectors.
_HUMAN_DEATH_ANTIAPOPTOTIC_REGULATORS = {
    "BCL2L1", "BCL2L2", "MCL1", "XIAP", "CFLAR", "BIRC3", "CLU",
}

# (3) genes within the mouse Hallmark P53 Pathway list with a direct,
# literature-supported role in triggering/executing apoptosis. Everything
# else in that list is cell-cycle arrest / DNA-damage response / senescence.
_MOUSE_APOPTOSIS_CORE = {
    "Bax", "Bak1", "Fas", "Casp1", "Apaf1", "Perp", "Aen", "Ei24", "Ddit3",
    "Dram1",
}

# `Pcna` sits in the Hallmark P53 Pathway list (DNA-damage-response role)
# but is also a core Tirosh cell-cycle/proliferation gene (`DEFAULT_
# PROLIFERATION_GENES`) -- excluded here for the same reason as the
# proliferation/death overlaps above (see module docstring point 1).
_MOUSE_SENESCENCE_PROLIF_OVERLAP = {"Pcna"}

_MOUSE_SENESCENCE_ARREST = sorted(
    g for g in MOSCOT_DEATH_GENES["mouse"]
    if g not in _MOUSE_APOPTOSIS_CORE and g not in _MOUSE_SENESCENCE_PROLIF_OVERLAP
)

# (4) 17 human death genes whose mouse ortholog (see _HCOP_HUMAN_TO_MOUSE_DEATH
# below) turned out to already sit in the arrest/senescence list -- i.e. two
# independent Hallmark gene sets agree these are stress/arrest genes, not
# death effectors (see point 4 in the module docstring).
_HUMAN_DEATH_SENESCENCE_OVERLAP = {
    "APP", "ATF3", "BMP2", "BTG2", "CCND2", "CDKN1A", "F2R", "FDXR",
    "GADD45A", "HMOX1", "IER3", "IL1A", "JUN", "RETSAT", "SAT1", "TAP1",
    "TXNIP",
}

# Human -> mouse orthologs for the cleaned human death set (161 genes minus
# the proliferation-overlap, anti-apoptotic-regulator, and
# senescence-overlap exclusions above -> 135 genes), from HCOP (HGNC
# Comparison of Orthology Predictions), evidence >= 3, one best-evidence
# mouse gene kept per human gene (manual override: GSTM1 -> Gstm1, see
# module docstring point 4). Source file: human_mouse_hcop_fifteen_column.txt
# (HCOP "fifteen column" export), processed 2026-07.
_HCOP_HUMAN_TO_MOUSE_DEATH = {
    "ADD1": "Add1", "AIFM3": "Aifm3", "ANKH": "Ank", "ANXA1": "Anxa1",
    "AVPR1A": "Avpr1a", "BAX": "Bax", "BCAP31": "Bcap31", "BCL10": "Bcl10",
    "BCL2L10": "Bcl2l10", "BCL2L11": "Bcl2l11", "BGN": "Bgn", "BID": "Bid",
    "BIK": "Bik", "BMF": "Bmf", "BNIP3L": "Bnip3l", "BRCA1": "Brca1",
    "BTG3": "Btg3", "CASP1": "Casp1", "CASP2": "Casp2", "CASP3": "Casp3",
    "CASP4": "Casp4", "CASP6": "Casp6", "CASP7": "Casp7", "CASP8": "Casp8",
    "CASP9": "Casp9", "CAV1": "Cav1", "CCNA1": "Ccna1", "CCND1": "Ccnd1",
    "CD14": "Cd14", "CD2": "Cd2", "CD38": "Cd38", "CD44": "Cd44",
    "CD69": "Cd69", "CDC25B": "Cdc25b", "CDK2": "Cdk2", "CDKN1B": "Cdkn1b",
    "CREBBP": "Crebbp", "CTH": "Cth", "CTNNB1": "Ctnnb1", "CYLD": "Cyld",
    "DAP": "Dap", "DAP3": "Dap3", "DCN": "Dcn", "DDIT3": "Ddit3",
    "DFFA": "Dffa", "DIABLO": "Diablo", "DNAJA1": "Dnaja1",
    "DNAJC3": "Dnajc3", "DNM1L": "Dnm1l", "DPYD": "Dpyd", "EBP": "Ebp",
    "EGR3": "Egr3", "EMP1": "Emp1", "ENO2": "Eno2", "ERBB2": "Erbb2",
    "ERBB3": "Erbb3", "EREG": "Ereg", "ETF1": "Etf1", "F2": "F2",
    "FAS": "Fas", "FASLG": "Fasl", "FEZ1": "Fez1", "GADD45B": "Gadd45b",
    "GCH1": "Gch1", "GNA15": "Gna15", "GPX1": "Gpx1", "GPX3": "Gpx3",
    "GPX4": "Gpx4", "GSN": "Gsn", "GSR": "Gsr", "GSTM1": "Gstm1",
    "GUCY2D": "Gucy2e", "H1-0": "H1f0", "HGF": "Hgf", "HSPB1": "Hspb1",
    "IFITM3": "Ifitm3", "IFNB1": "Ifnb1", "IFNGR1": "Ifngr1",
    "IGF2R": "Igf2r", "IGFBP6": "Igfbp6", "IL18": "Il18", "IL1B": "Il1b",
    "IL6": "Il6", "IRF1": "Irf1", "ISG20": "Isg20", "KRT18": "Krt18",
    "LEF1": "Lef1", "LGALS3": "Lgals3", "LMNA": "Lmna", "LUM": "Lum",
    "MADD": "Madd", "MGMT": "Mgmt", "MMP2": "Mmp2", "NEDD9": "Nedd9",
    "NEFH": "Nefh", "PAK1": "Pak1", "PDCD4": "Pdcd4", "PDGFRB": "Pdgfrb",
    "PEA15": "Pea15a", "PLAT": "Plat", "PLCB2": "Plcb2",
    "PLPPR4": "Plppr4", "PMAIP1": "Pmaip1", "PPP2R5B": "Ppp2r5b",
    "PPP3R1": "Ppp3r1", "PPT1": "Ppt1", "PRF1": "Prf1", "PSEN1": "Psen1",
    "PSEN2": "Psen2", "PTK2": "Ptk2", "RARA": "Rara", "RELA": "Rela",
    "RHOB": "Rhob", "RHOT2": "Rhot2", "RNASEL": "Rnasel", "ROCK1": "Rock1",
    "SATB1": "Satb1", "SC5D": "Sc5d", "SLC20A1": "Slc20a1",
    "SMAD7": "Smad7", "SOD1": "Sod1", "SOD2": "Sod2", "SPTAN1": "Sptan1",
    "SQSTM1": "Sqstm1", "TGFB2": "Tgfb2", "TGFBR3": "Tgfbr3",
    "TIMP1": "Timp1", "TIMP2": "Timp2", "TIMP3": "Timp3", "TNF": "Tnf",
    "TNFRSF12A": "Tnfrsf12a", "TNFSF10": "Tnfsf10", "TSPO": "Tspo",
    "VDAC2": "Vdac2", "WEE1": "Wee1",
}

DEFAULT_PROLIFERATION_GENES = {
    "human": MOSCOT_PROLIFERATION_GENES["human"],
    "mouse": MOSCOT_PROLIFERATION_GENES["mouse"],
}

DEFAULT_DEATH_GENES = {
    "human": sorted(
        g for g in MOSCOT_DEATH_GENES["human"]
        if g not in _HUMAN_DEATH_OVERLAP_WITH_PROLIF
        and g not in _HUMAN_DEATH_ANTIAPOPTOTIC_REGULATORS
        and g not in _HUMAN_DEATH_SENESCENCE_OVERLAP
    ),
    "mouse": sorted(_MOUSE_APOPTOSIS_CORE | set(_HCOP_HUMAN_TO_MOUSE_DEATH.values())),
}

assert set(_HCOP_HUMAN_TO_MOUSE_DEATH) == set(DEFAULT_DEATH_GENES["human"]), (
    "_HCOP_HUMAN_TO_MOUSE_DEATH must have exactly one entry per gene in "
    "DEFAULT_DEATH_GENES['human']"
)

DEFAULT_SENESCENCE_GENES = {
    # See module docstring: approximate, obtained by the same human<->mouse
    # capitalization convention used elsewhere in this file, not a
    # separately verified human Hallmark P53 Pathway extraction.
    "human": sorted(g.upper() for g in _MOUSE_SENESCENCE_ARREST),
    "mouse": _MOUSE_SENESCENCE_ARREST,
}


def score_gene_sets(adata, proliferation_genes=None, death_genes=None,
                     senescence_genes=None, species="human",
                     **score_genes_kwargs):
    """
    Score each cell against a proliferation, a death, and a senescence/
    quiescence/arrest gene signature.

    Wraps ``scanpy.tl.score_genes`` (control-gene-bin corrected mean expression,
    as in Waddington-OT/moscot). Genes absent from ``adata.var_names`` are
    silently dropped.

    ``score_genes`` assumes total-count-normalised, log1p-transformed
    expression (moscot's own tutorials normalise and log-transform before
    calling the equivalent ``score_genes_for_marginals``; moscot itself does
    not do this internally, it is left to the caller — see
    ``moscot.base.problems.birth_death.BirthDeathMixin.score_genes_for_marginals``).
    Scoring here is therefore done on an internal, normalised+log1p **copy** of
    ``adata`` — ``adata.X`` (and everything else on the object passed in) is
    left untouched, since CardamomOT's inference requires raw counts.

    If none of the senescence marker genes are found in ``adata.var_names``
    (or an explicit empty list is passed), a warning is issued,
    ``score_senescence`` is returned as an all-zero array, and
    ``senescence_available`` is ``False`` — callers should treat this as
    "no senescence signal available" and skip gating entirely, rather than
    gating on a score of exactly 0 (which is not the same as a disabled
    gate: the shifted-logistic gate is not centered on 0, so `gate(0)` is a
    small but non-zero value, not the "fully open" gate this is meant to
    represent).

    Returns
    -------
    score_prolif, score_death, score_senescence : (N,) arrays of per-cell
    signature scores.
    senescence_available : bool, whether `score_senescence` reflects real
    marker genes (True) or is a zero-filled placeholder (False).
    """
    import scanpy as sc

    prolif_genes = DEFAULT_PROLIFERATION_GENES[species] if proliferation_genes is None else proliferation_genes
    death_genes = DEFAULT_DEATH_GENES[species] if death_genes is None else death_genes
    senescence_genes = DEFAULT_SENESCENCE_GENES[species] if senescence_genes is None else senescence_genes

    prolif_genes = [g for g in prolif_genes if g in adata.var_names]
    death_genes = [g for g in death_genes if g in adata.var_names]
    senescence_genes = [g for g in senescence_genes if g in adata.var_names]
    if len(prolif_genes) == 0:
        raise ValueError(
            "None of the proliferation marker genes were found in adata.var_names; "
            "pass `proliferation_genes` explicitly."
        )
    if len(death_genes) == 0:
        raise ValueError(
            "None of the death marker genes were found in adata.var_names; "
            "pass `death_genes` explicitly."
        )

    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm)
    sc.pp.log1p(adata_norm)

    sc.tl.score_genes(adata_norm, prolif_genes, score_name="_prolif_score_tmp", **score_genes_kwargs)
    sc.tl.score_genes(adata_norm, death_genes, score_name="_death_score_tmp", **score_genes_kwargs)
    score_prolif = adata_norm.obs["_prolif_score_tmp"].to_numpy()
    score_death = adata_norm.obs["_death_score_tmp"].to_numpy()

    if len(senescence_genes) == 0:
        warnings.warn(
            "[score_gene_sets] None of the senescence marker genes were found in "
            "adata.var_names (or an empty list was passed); senescence gating will "
            "be disabled for this call. Pass `senescence_genes` explicitly to fix "
            "this if unintentional."
        )
        score_senescence = np.zeros(adata.n_obs, dtype=float)
        senescence_available = False
    else:
        sc.tl.score_genes(adata_norm, senescence_genes, score_name="_senescence_score_tmp",
                           **score_genes_kwargs)
        score_senescence = adata_norm.obs["_senescence_score_tmp"].to_numpy()
        senescence_available = True

    return score_prolif, score_death, score_senescence, senescence_available


def _shifted_logistic(score, rate_max, rate_min, center, width):
    """Shifted logistic score→rate map, identical in form to moscot's `_gen_logistic`."""
    score = np.asarray(score, dtype=float)
    L = rate_max - rate_min
    k = 4.0 / width
    return rate_min + L / (1.0 + np.exp(-k * (score - center)))


def scores_to_rates(score, rate_max, rate_min, center, width):
    """
    Map a signature score to a rate via a shifted logistic curve.

    Same functional form and default center/width as moscot's `beta`/`delta`
    growth-rate functions (moscot.base.problems.birth_death): a logistic curve
    saturating at `rate_min` for low scores and `rate_max` for high scores,
    with the transition centered on `center` and spanning roughly `width`.
    Unlike a quantile-based rescaling, this is centered on an *absolute* score
    value, not on the data's own score distribution — it assumes
    ``sc.tl.score_genes`` scores are on a comparable scale across datasets.
    """
    return _shifted_logistic(score, rate_max, rate_min, center, width)


def senescence_gate(score_senescence, gamma_center, gamma_width):
    """
    Map a senescence/arrest signature score to a gate in [0, 1] via the same
    shifted-logistic form as `scores_to_rates` (rate_max=1, rate_min=0):
    ``gate -> 0`` (birth left untouched) for low senescence scores,
    ``gate -> 1`` (birth fully suppressed) for high senescence scores.

    Unlike `beta_*`/`delta_*` in `estimate_growth_rates`, `gamma_center`/
    `gamma_width` are NOT calibrated against moscot or any external
    reference — there is no equivalent gating step in moscot/WOT to inherit
    defaults from. Treat the defaults as a starting point and inspect your
    own `score_senescence` distribution (e.g. its histogram, and where your
    known-quiescent/known-arrested cells fall on it) before trusting the gate
    on new data.
    """
    return _shifted_logistic(score_senescence, rate_max=1.0, rate_min=0.0,
                              center=gamma_center, width=gamma_width)


def estimate_growth_rates(adata, proliferation_genes=None, death_genes=None,
                           senescence_genes=None, species="human",
                           beta_max=1.7, beta_min=0.3, beta_center=0.25, beta_width=0.5,
                           delta_max=1.7, delta_min=0.3, delta_center=0.1, delta_width=0.2,
                           senescence_gating=True,
                           gamma_center=0.15, gamma_width=0.3,
                           hours_per_day=24.0,
                           **score_genes_kwargs):
    """
    Estimate a per-cell net growth rate (birth - death) from literature gene
    signatures, with an optional soft gate on birth for senescent/arrested/
    quiescent cells.

    Reproduces moscot's `TemporalProblem.score_genes_for_marginals` /
    `beta`/`delta` growth-rate estimation (itself following Waddington-OT,
    Schiebinger et al. 2019): a proliferation score and a death score are
    each passed through their own shifted-logistic curve (`beta`, `delta`).

    On top of that, if `senescence_gating=True` (default), a third score is
    computed against `senescence_genes` (cell-cycle arrest / DNA-damage
    response / senescence markers — see `DEFAULT_SENESCENCE_GENES`) and
    passed through `senescence_gate` to get a value in [0, 1]. This gate
    multiplies the birth rate only:

        net_rate = beta(score_prolif) * (1 - gate(score_senescence)) - delta(score_death)

    so a strongly senescent/quiescent cell gets `net_rate` pulled toward
    `-delta(score_death)` (i.e. proliferation effectively silenced) even if
    it still expresses residual cell-cycle genes, without also suppressing
    an independently elevated death score — a cell that is both stressed and
    genuinely dying should still show up as dying, not as neutral. This is
    the main motivation for this gate: on stress-perturbation data, the raw
    `beta - delta` from moscot's own defaults cannot distinguish "arrested/
    senescent, not dying" from "actively proliferating", since arrest-
    pathway genes overlap with cell-cycle genes.

    Set `senescence_gating=False` to recover the original, un-gated
    `beta(score_prolif) - delta(score_death)` (moscot-equivalent behaviour).
    Gating is also automatically skipped (regardless of `senescence_gating`)
    if no senescence marker genes are available for this call — e.g. an
    explicit empty `senescence_genes=[]`, or none of them found in
    `adata.var_names` — since gating on an all-zero placeholder score would
    not actually mean "no effect" (see `score_gene_sets`).

    The `beta_*`/`delta_*` defaults are moscot's own published defaults
    (verified against its source), and — like moscot/WOT themselves — express
    a **per-day** rate: moscot's `TemporalProblem` computes elapsed time from a
    `day` obs field and raises the per-timepoint growth score to that many
    days (see `broadinstitute/wot`'s `OTModel.compute_transport_map`:
    ``G = cell_growth_rate ** (t1 - t0)``, with `t0`/`t1` in days). CardamomOT
    instead expresses `adata.obs['time']` and every internal kinetic rate in
    **hours** (e.g. `NetworkModel`'s default degradation rates correspond to a
    9h mRNA / 46h protein half-life) — so the day⁻¹ literature rate is
    converted to hour⁻¹ by dividing by `hours_per_day` (default 24) before
    being returned, making it directly usable as `Δt` in hours without any
    further rescaling downstream (see
    `CardamomOT.model.base.estimate_trajectories_given_model`). Pass
    `hours_per_day=1` to recover the raw moscot-equivalent day⁻¹ rate if your
    own `adata.obs['time']` happens to be in days instead. The `gamma_*`
    defaults have no such external reference — see `senescence_gate`.

    Returns
    -------
    net_rate : (N,) array, in hour⁻¹ (see `hours_per_day` above).
    """
    score_prolif, score_death, score_senescence, senescence_available = score_gene_sets(
        adata, proliferation_genes, death_genes, senescence_genes, species,
        **score_genes_kwargs
    )
    birth = scores_to_rates(score_prolif, beta_max, beta_min, beta_center, beta_width)
    death = scores_to_rates(score_death, delta_max, delta_min, delta_center, delta_width)

    if senescence_gating and senescence_available:
        gate = senescence_gate(score_senescence, gamma_center, gamma_width)
        birth = birth * (1.0 - gate)

    net_rate_per_day = birth - death
    return net_rate_per_day / hours_per_day


def combine_growth_rates_with_reference(lit_rates, group_labels, reference_rates):
    """
    Anchor per-cell literature-derived growth rates to group-level reference rates.

    For each group present in `reference_rates` (e.g. per cell type), the
    literature rate is shifted so its mean over that group matches the
    reference value, while keeping the per-cell heterogeneity coming from the
    gene signature. Groups absent from `reference_rates` keep their raw
    (unanchored) literature estimate.

    No unit conversion is applied here: `reference_rates` is added directly to
    `lit_rates`, so both must already be in the same time unit. If `lit_rates`
    comes from `estimate_growth_rates` (hour^-1 by default, see its
    `hours_per_day` argument), `reference_rates` must be hour^-1 too — growth
    curves reported in the literature are commonly day^-1 and would need
    dividing by 24 first.

    Parameters
    ----------
    lit_rates : (N,) array of per-cell literature-derived net rates.
    group_labels : (N,) array-like of group labels (e.g. adata.obs['cell_type']).
    reference_rates : dict {group_label: rate} of trusted population-level
        rates, in the same time unit as `lit_rates`.

    Returns
    -------
    (N,) array of combined per-cell net rates.
    """
    lit_rates = np.asarray(lit_rates, dtype=float)
    group_labels = np.asarray(group_labels).astype(str)
    reference_rates = {str(k): v for k, v in reference_rates.items()}

    out = lit_rates.copy()
    matched_groups = set()
    for group, ref_rate in reference_rates.items():
        mask = group_labels == group
        if not mask.any():
            continue
        matched_groups.add(group)
        out[mask] = lit_rates[mask] - lit_rates[mask].mean() + ref_rate

    unmatched = set(np.unique(group_labels)) - matched_groups
    if unmatched:
        warnings.warn(
            f"[combine_growth_rates_with_reference] No reference rate for groups "
            f"{sorted(unmatched)}; using unanchored literature estimate for these cells."
        )
    return out
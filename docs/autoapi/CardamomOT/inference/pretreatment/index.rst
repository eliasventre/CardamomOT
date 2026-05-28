CardamomOT.inference.pretreatment
=================================

.. py:module:: CardamomOT.inference.pretreatment

.. autoapi-nested-parse::

   Core functions for selecting the most variable genes according to the
   Zero‑Inflated Negative Binomial (ZiNB) model.



Attributes
----------

.. autoapisummary::

   CardamomOT.inference.pretreatment.logger


Functions
---------

.. autoapisummary::

   CardamomOT.inference.pretreatment.ln2
   CardamomOT.inference.pretreatment.extract_degradation_rates
   CardamomOT.inference.pretreatment.select_DEgenes


Module Contents
---------------

.. py:data:: logger

.. py:function:: ln2(x)

.. py:function:: extract_degradation_rates(df, gene_list, cell_line=None, similarity_threshold=np.linspace(0.99, 0.01, 10))

.. py:function:: select_DEgenes(vect_t, vect_samples_id, vect_celltype_id, proba, list_genes, n_genes_tokeep_temporal=[1000], n_genes_tokeep_celltype=[1000], limit_min=0.01, verb=0)


CardamomOT.config
=================

.. py:module:: CardamomOT.config

.. autoapi-nested-parse::

   Configuration and constants for CARDAMOM pipeline.

   Centralizes all constants, default parameters, and configuration options
   used throughout the CARDAMOM pipeline for easy maintenance and consistency.



Attributes
----------

.. autoapisummary::

   CardamomOT.config.DEFAULT_DATA_FOLDER
   CardamomOT.config.DEFAULT_CARDAMOM_FOLDER
   CardamomOT.config.DEFAULT_RESULTS_FOLDER
   CardamomOT.config.DEFAULT_DATA_FILE
   CardamomOT.config.DEFAULT_GENE_LIST_FILE
   CardamomOT.config.DEFAULT_HALFLIFE_TABLE
   CardamomOT.config.DEFAULT_N_GENES_TEMPORAL
   CardamomOT.config.DEFAULT_N_GENES_CELLTYPE
   CardamomOT.config.DEFAULT_MIN_MEAN_EXPRESSION
   CardamomOT.config.DEFAULT_VAR_THRESHOLD
   CardamomOT.config.DEFAULT_RD_MEDIAN_TOLERANCE
   CardamomOT.config.REQUIRED_OBS_KEYS
   CardamomOT.config.OPTIONAL_OBS_KEYS
   CardamomOT.config.STANDARD_OBS
   CardamomOT.config.DEFAULT_PRIOR_STRENGTH
   CardamomOT.config.DEFAULT_STIM_LEVEL
   CardamomOT.config.DEFAULT_MIXTURE_TOLERANCE
   CardamomOT.config.DEFAULT_MIXTURE_MAX_ITER
   CardamomOT.config.DEFAULT_PROTEIN_HALFLIFE_MIN
   CardamomOT.config.DEFAULT_PROTEIN_HALFLIFE_MAX
   CardamomOT.config.DEFAULT_MRNA_HALFLIFE_MIN
   CardamomOT.config.DEFAULT_MRNA_HALFLIFE_MAX
   CardamomOT.config.CMAP_GENE_EXPRESSION
   CardamomOT.config.CMAP_NETWORK
   CardamomOT.config.CMAP_CELL_TYPES
   CardamomOT.config.DEFAULT_FIGURE_WIDTH
   CardamomOT.config.DEFAULT_FIGURE_HEIGHT
   CardamomOT.config.ERROR_MSG_NO_DATA
   CardamomOT.config.ERROR_MSG_NO_TIMES
   CardamomOT.config.ERROR_MSG_INVALID_SPLIT
   CardamomOT.config.WARNING_MSG_NO_CELL_TYPES
   CardamomOT.config.WARNING_MSG_NO_GENE_LIST


Functions
---------

.. autoapisummary::

   CardamomOT.config.get_project_directories
   CardamomOT.config.get_default_parameters


Module Contents
---------------

.. py:data:: DEFAULT_DATA_FOLDER
   :value: 'Data'


.. py:data:: DEFAULT_CARDAMOM_FOLDER
   :value: 'cardamom'


.. py:data:: DEFAULT_RESULTS_FOLDER
   :value: 'results'


.. py:data:: DEFAULT_DATA_FILE
   :value: 'data.h5ad'


.. py:data:: DEFAULT_GENE_LIST_FILE
   :value: 'gene_list.txt'


.. py:data:: DEFAULT_HALFLIFE_TABLE
   :value: 'table_halflife_mammalian.csv'


.. py:data:: DEFAULT_N_GENES_TEMPORAL
   :value: 5


.. py:data:: DEFAULT_N_GENES_CELLTYPE
   :value: 3


.. py:data:: DEFAULT_MIN_MEAN_EXPRESSION
   :value: 0.01


.. py:data:: DEFAULT_VAR_THRESHOLD
   :value: 1.2


.. py:data:: DEFAULT_RD_MEDIAN_TOLERANCE
   :value: 1e-16


.. py:data:: REQUIRED_OBS_KEYS

.. py:data:: OPTIONAL_OBS_KEYS

.. py:data:: STANDARD_OBS
   :value: ['time', 'cell_type', 'rd', 'd0', 'd1']


.. py:data:: DEFAULT_PRIOR_STRENGTH
   :value: 1.0


.. py:data:: DEFAULT_STIM_LEVEL
   :value: 1.0


.. py:data:: DEFAULT_MIXTURE_TOLERANCE
   :value: 1e-06


.. py:data:: DEFAULT_MIXTURE_MAX_ITER
   :value: 1000


.. py:data:: DEFAULT_PROTEIN_HALFLIFE_MIN
   :value: 30


.. py:data:: DEFAULT_PROTEIN_HALFLIFE_MAX
   :value: 720


.. py:data:: DEFAULT_MRNA_HALFLIFE_MIN
   :value: 5


.. py:data:: DEFAULT_MRNA_HALFLIFE_MAX
   :value: 120


.. py:data:: CMAP_GENE_EXPRESSION
   :value: 'viridis'


.. py:data:: CMAP_NETWORK
   :value: 'coolwarm'


.. py:data:: CMAP_CELL_TYPES
   :value: 'Dark2'


.. py:data:: DEFAULT_FIGURE_WIDTH
   :value: 10


.. py:data:: DEFAULT_FIGURE_HEIGHT
   :value: 8


.. py:data:: ERROR_MSG_NO_DATA
   :value: "No data file found. Create a subfolder 'Data' in your project directory and place a count table...


.. py:data:: ERROR_MSG_NO_TIMES
   :value: "The input data has no temporal information or only one timepoint. Please ensure 'time' column...


.. py:data:: ERROR_MSG_INVALID_SPLIT
   :value: 'Invalid data split specified. Expected splits in: {available_splits}'


.. py:data:: WARNING_MSG_NO_CELL_TYPES
   :value: "No cell type information found in adata.obs['cell_type']. Gene selection will use only temporal...


.. py:data:: WARNING_MSG_NO_GENE_LIST
   :value: 'No external gene list found at {gene_list_path}. Using only data-driven gene selection.'


.. py:function:: get_project_directories(project_path: pathlib.Path) -> Dict[str, pathlib.Path]

   Get all standard subdirectories for a CARDAMOM project.

   :param project_path: Root directory of the project.

   :returns: data, cardamom, results.
   :rtype: Dictionary with keys


.. py:function:: get_default_parameters() -> Dict[str, Any]

   Get all default parameters as a dictionary.

   :returns: Dictionary of all default parameter values.



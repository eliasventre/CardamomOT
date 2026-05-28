CardamomOT.inference.network_final
==================================

.. py:module:: CardamomOT.inference.network_final

.. autoapi-nested-parse::

   PyTorch version of network inference using Adam optimizer
   Simplified version focusing on core inference without refinement



Attributes
----------

.. autoapisummary::

   CardamomOT.inference.network_final.eps_CE
   CardamomOT.inference.network_final.sc
   CardamomOT.inference.network_final.r_elasticnet


Classes
-------

.. autoapisummary::

   CardamomOT.inference.network_final.GRNInference


Functions
---------

.. autoapisummary::

   CardamomOT.inference.network_final.base_kon_torch
   CardamomOT.inference.network_final.smoothed_l1_loss
   CardamomOT.inference.network_final.l2_loss
   CardamomOT.inference.network_final.elasticnet_loss
   CardamomOT.inference.network_final.compute_main_loss
   CardamomOT.inference.network_final.inference_pytorch
   CardamomOT.inference.network_final.inference_network_pytorch


Module Contents
---------------

.. py:data:: eps_CE
   :value: 1e-06


.. py:data:: sc
   :value: 0.001


.. py:data:: r_elasticnet
   :value: 0.5


.. py:function:: base_kon_torch(theta_basal, theta_inter, y_prot) -> torch.Tensor

   PyTorch version of base_kon with numerical stability

   :param theta_basal: (n_networks,) basal parameters
   :param theta_inter: (G, n_networks) interaction parameters
   :param y_prot: (n_cells, G) protein expression values

   :returns: (n_cells, n_networks+1) probabilities for each network state


.. py:function:: smoothed_l1_loss(tensor, l1_weight) -> torch.Tensor

   Smoothed L1 penalization


.. py:function:: l2_loss(tensor, l2_weight)

   L2 penalization


.. py:function:: elasticnet_loss(tensor, weight)

   Elastic net: combination of L1 and L2


.. py:function:: compute_main_loss(y_pred, y_true, loss_type='CE')

   Compute main loss (L1, L2, or Cross-Entropy)

   :param y_pred: predicted values
   :param y_true: true values
   :param loss_type: 'l1', 'l2', or 'CE'


.. py:class:: GRNInference(G, n_networks, inter_init, basal_init, ref_network)

   Bases: :py:obj:`torch.nn.Module`


   PyTorch module for Gene Regulatory Network inference


   .. py:attribute:: G
      :type:  Any


   .. py:attribute:: n_networks
      :type:  Any


   .. py:attribute:: theta_inter


   .. py:attribute:: theta_basal


   .. py:method:: forward(y_prot) -> torch.Tensor

      Forward pass: compute network probabilities

      :param y_prot: (n_cells, G) protein expression

      :returns: (n_cells, n_networks+1) network probabilities
      :rtype: sigma



   .. py:method:: compute_loss(y_samples, y_proba, y_prot, y_prot_prev, y_kon, ks, proba, l_pen, weight_prev=0.5, loss_type='CE')

      Compute total loss including data fitting and regularization

      :param y_samples: (n_cells,) sample/batch indices
      :param y_proba: (n_cells, n_networks+1) target probabilities (if proba=True)
      :param y_prot: (n_cells, G) protein expression current timepoint
      :param y_prot_prev: (n_cells, G) protein expression previous timepoint
      :param y_kon: (n_cells,) target kon values (if proba=False)
      :param ks: (n_networks+1,) network activation rates
      :param proba: bool, whether to predict probabilities or kon
      :param l_pen: penalization weight
      :param weight_prev: weight for previous timepoint loss
      :param loss_type: 'l1', 'l2', or 'CE'
      :param final: bool, use elastic net if True



.. py:function:: inference_pytorch(y_samples, y_kon, y_proba, y_prot, y_prot_prev, ks, inter_init, basal_init, ref_network, inter_ref=None, basal_ref=None, proba=True, scale=100, weight_prev=0.5, loss_type='CE', lr=0.001, n_epochs=1000, verbose=True, device='cuda' if torch.cuda.is_available() else 'cpu')

   PyTorch-based network inference using Adam optimizer

   :param y_samples: (n_cells,) sample/batch indices
   :param y_kon: (n_cells,) target kon values
   :param y_proba: (n_cells, n_networks+1) target network probabilities
   :param y_prot: (n_cells, G) protein expression (current)
   :param y_prot_prev: (n_cells, G) protein expression (previous)
   :param ks: (n_networks+1,) network activation rates
   :param inter_init: (G, n_networks) initial interaction parameters
   :param basal_init: (n_networks,) initial basal parameters
   :param ref_network: (G, n_networks) reference network structure
   :param inter_ref: (G, n_networks) reference interaction parameters (defaults to inter_init)
   :param basal_ref: (n_networks,) reference basal parameters (defaults to basal_init)
   :param proba: bool, predict probabilities (True) or kon values (False)
   :param scale: scaling factor for penalization
   :param weight_prev: weight for previous timepoint in loss
   :param loss_type: 'l1', 'l2', or 'CE'
   :param lr: learning rate for Adam
   :param n_epochs: number of training epochs
   :param final: use elastic net regularization
   :param verbose: print progress
   :param device: 'cuda' or 'cpu'

   :returns: (G, n_networks) inferred interaction parameters
             basal: (n_networks,) inferred basal parameters
             losses: list of loss values during training
   :rtype: inter


.. py:function:: inference_network_pytorch(vect_t, times, y_samples, y_kon, y_proba, y_prot, y_prot_prev, ks, inter_init=np.zeros(2), basal_init=np.zeros(2), ref_network=np.zeros(2), inter_ref=None, basal_ref=None, proba=True, scale=100, weight_prev=0.5, loss='CE', lr=0.01, n_epochs=1000, verbose=True, final=1, device='cuda' if torch.cuda.is_available() else 'cpu') -> Any

   Main entry point for PyTorch network inference
   Simplified version without temporal dynamics and refinement

   :param y_samples: (n_cells,) sample indices
   :param y_kon: (n_cells, G) target kon values for all genes
   :param y_proba: (n_cells, G, n_networks+1) target probabilities for all genes
   :param y_prot: (n_cells, G) protein expression (current)
   :param y_prot_prev: (n_cells, G) protein expression (previous)
   :param ks: (G, n_networks+1) activation rates per gene
   :param inter_init: (G, G, n_networks) initial interaction parameters
   :param basal_init: (G, n_networks) initial basal parameters
   :param ref_network: (G, G, n_networks) reference network structure
   :param inter_ref: (G, G, n_networks) reference interaction parameters
   :param basal_ref: (G, n_networks) reference basal parameters
   :param proba: predict probabilities or kon
   :param scale: penalization scaling
   :param weight_prev: weight for previous timepoint
   :param loss_type: 'l1', 'l2', or 'CE'
   :param lr: learning rate
   :param n_epochs: training epochs
   :param final: use elastic net
   :param verbose: print progress
   :param device: computation device

   :returns: (G, G, n_networks) inferred interaction matrix
             basal: (G, n_networks) inferred basal parameters
             all_losses: dict with losses per gene
   :rtype: inter



---
id: programmable_inference
title: 'Programmable Inference'
sidebar_label: 'Overview'
slug: '/programmable_inference'
---

Bean Machine offers three inference techniques which aim to close the performance gap between general-purpose and model-specific handwritten inference:

* Compositional inference: allows the use of different proposal algorithms for each random variable.
* Block inference: allows multiple variables to be grouped together during inference.
* Custom proposers: allow users to implement a custom proposal algorithm on a per-variable basis.

These techniques together, which we call *programmable inference, *give the inference engine sufficient configurability for users to achieve efficient performance without writing a complete model-specific inference algorithm.

All techniques above are made available through Bean Machine's choice of declarative syntax. Declarative syntax makes the statistical models' dependency structure, directed acyclic graph (DAG), explicit. Random variables are specified independently of the order in which they are sampled during inference and the inference engine has direct access to the code block defining each variable, and can execute these blocks in the order required by the inference algorithm. The fundamental algorithm underlying Bean Machine's inference engine is Single-Site Metropolis Hastings (SSMH) [?], where a new value is proposed for a single variable using a proposal algorithm, and an accept/reject decision is made following the Metropolis Hasting rule. SSMH has been extended in Bean Machine with compositional inference, block inference and custom proposers.

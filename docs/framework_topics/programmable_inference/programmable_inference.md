---
id: programmable_inference
title: 'Programmable Inference'
sidebar_label: 'Overview'
slug: '/programmable_inference'
---

Programmable inference is a key feature of Bean Machine, and is achieved through three key techniques:
- Compositional inference allows you to utilize distinct inference methods for different random variables when fitting a model. Bean Machines's single-site paradigm makes composeability possible as it allows you to modularly mix-and-match inference components to get the most out of your model.
- Block inference allows you to propose updates for several random variables jointly, which can be necessary when dealing with highly-correlated variables.
- Custom proposers allow you to leverage domain-specific transformations or custom proposers on a per-variable basis, which can be especially powerful to avoid worse edge-case performance when running inference over constrained random variables.

These techniques together, which we call *programmable inference, *give the inference engine sufficient configurability for users to achieve efficient performance without writing a complete model-specific inference algorithm, and help close the performance gap between general-purpose and model-specific handwritten inference.  In the rest of this section we expand on each of these concepts.

It is worth noting that supporting these techniques is facilitiated by Bean Machine's choice of declarative syntax, which explicates the statistical models' dependency structure, namely, the directed acyclic graph (DAG). Random variables are specified independently of the order in which they are sampled during inference and the inference engine has direct access to the code block defining each variable, and can execute these blocks in the order required by the inference algorithm. The fundamental algorithm underlying Bean Machine's inference engine is Single Site Metropolis Hastings (SSMH) [?], where a new value is proposed for a single variable using a proposal algorithm, and an accept/reject decision is made following the Metropolis Hasting rule. SSMH has been extended in Bean Machine with compositional inference, block inference and custom proposers.

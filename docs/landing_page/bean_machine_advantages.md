<!-- @import "../header.md" -->

# Bean Machine advantages

By building on top of PyTorch and providing declarative syntax, Bean Machine can be simultaneously performant and intuitive. Bean Machine provides further value by implementing cutting-edge inference algorithms and allowing the user to select and program custom inferences for different problems and subproblems.

## Single-Site Inference

<!-- ### Single-site inference -->

Bean Machine's inference engine uses _single-site inference_. In the single-site paradigm, models are built up from _random variables_ that can be reasoned about individually. This creates a more intuitive user experience, allows for optimizations that can't be achieved in other systems and enables the advanced techniques outlined below.

## Declarative modeling

<!-- Usability improvements -->
In Bean Machine, random variables are implemented as decorated Python functions, which naturally form an interface to the model. Using functions makes it simple to determine a random variable's definition, since it is contained in a function that is usually only a few lines long. This lets you directly refer to random variables to access inferred distributions or when binding data to your model; this is safer and more natural than relying on string identifiers.

<!-- Efficiency improvements -->
Declarative modeling also frees the inference engine to reorder model execution. Foremost, it enables computation of immediate dependencies for random variables. This makes it possible to propose new values for a random variable by examining only its dependencies, saving significant amounts of compute in models with complex structure. Because the language recognizes random variables as a special value, it can accept or reject proposals for each random variable sequentially. Compared to global inference, this single-site paradigm can drastically improve convergence rates, since a single bad proposal will affect only a single random variable.

## Programmable inference

<!-- Compositional inference -->
Bean Machine allows the user to design and apply new and powerful inference methods. Because Bean Machine can propose updates for random variables individually, the user is free to customize the _method_ which it uses to propose those values. Different inference methods can be supplied for different families of random variables. For example, a particular model can leverage gradient information when proposing values for differentiable random variables, and at the same time might sample from discrete ones with a particle filter. The single-site paradigm enables seamless interoperation among any MCMC-based inference strategies.

<!-- Custom proposers -->
Programmable inference extends beyond just the inference customization. MCMC inference methods in Bean Machine are comprised of two main components: the _inference method_, and any _proposers_ it may use. Though Bean Machine comes with a rich set of unconstrained proposers,  it is useful to use constrained proposers for constrained spaces. This can improve performance at boundaries of constrained spaces, where transformations into unconstrained spaces can introduce substantial warping.

<!-- Block inference -->
Lastly, single-site inference is not always the right tool tool for the job. In cases when random variables are strongly correlated, Bean Machine offers tools such as block inference to jointly propose new values simultaneously.

## Advanced methods

Bean Machine supports a variety of classic inference methods such as ancestral sampling and the No U-Turn sampler (NUTS). However, the framework leverages single-site understanding of the model in order to provide efficient methods that take advantage of higher-order gradients and model structure.

Bean Machine includes the first implementation of Newtonian Monte Carlo (NMC) in a more general platform. NMC utilizes second-order gradient information to construct a multivariate Gaussian proposer that takes local curvature into account. As such, it can produce sample very efficiently with no warmup period when the posterior is roughly Gaussian. Bean Machine's structural understanding of the model lets us keep computation relatively cheap by only modeling a subset of the space that is relevant to updating a particular random variable.

Using programmable inference, Bean Machine itself implements numerous enhancements to core inference algorithms. Dynamic learning algorithms enable us to continuously and automatically evolve inference strategies to better model the problem at hand, without paying the hefty cost of a warmup period. A "mixture of proposers" technique lets us blend the benefits of using proposers that precisely describe local curvature, along with less precise ones to enable whole-space generalization. All of these approaches are built upon a highly modular inference framework, enabling you to mix-and-match methods -- or even invent new ones -- with ease.

<!-- ### PyTorch features

Autodiff
GPU support -->

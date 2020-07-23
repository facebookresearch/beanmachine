---
id: landing_page
title: BeanMachine
sidebar_label: BeanMachine
---

<!-- @import "../header.md" -->

Declarative, programmable, efficient probabilistic inference

---

# Key Features

## Declarative modeling

```viz
digraph {
  n1[label="Asset\nrequired"]
}
```

Clean, intuitive syntax that lets you focus on the model and leave performance to the framework

## Programmable inference

```viz
digraph {
  n1[label="Asset\nrequired"]
}
```

Mix-and-match inference methods, proposers, and inference strategies to achieve maximum efficiency

## Built on PyTorch

```viz
digraph {
  n1[label="Asset\nrequired"]
}
```

Leverage native GPU and autograd support and integrate seamlessly with the PyTorch ecosystem

---

# Get started

1\. **Install dependencies:**

  ```sh
  pip install numpy pytest torch
  ```

2\. **Install Bean Machine:**

  <! -- NOTE: I had to use `sudo` when running the setup script. RESPONSE: `sudo` is only needed when not installing in a virtualenv -->

  ```sh
  git clone https://github.com/facebookincubator/BeanMachine.git
  cd BeanMachine
  python setup.py install
  ```

3\. **Define model:**

  ```py
  from torch.distributions import Bernoulli, Beta
  from beanmachine.ppl.model import sample

  @sample
  def p():
      return Beta(1, 1)

  @sample
  def toss(i: int):
      return Bernoulli(p())
  ```

4\. **Bind observations:**

  ```py
  from torch import tensor

  observations = {
      toss(1): tensor(0.0),
      toss(2): tensor(1.0),
      toss(3): tensor(0.0),
      toss(4): tensor(1.0),
      toss(5): tensor(0.0),
      toss(6): tensor(1.0),
  }
  ```

5\. **Run inference:**

  ```py
  from beanmachine.ppl.inference import SingleSiteUniformMetropolisHastings

  samples = SingleSiteUniformMetropolisHastings().infer(
      queries=[p()],
      observations=observations,
      num_samples=1000,
  )
  ```

6\. **Process results:**

  ```py
  from beanmachine.ppl.diagnostics.diagnostics import Diagnostics

  print(Diagnostics(samples).summary())
  ```

  ```py
             avg      std     2.5%      50%    97.5%    r_hat       n_eff
  p()[]   0.5022   0.1683   0.1816   0.5060   0.8229   1.0037   1578.1005
  ```

7\. **Run tests:**

  ```sh
  pytest --pyargs beanmachine
  ```

Take the following sample
$z \sim q_{\theta} (z | x^{(i)}) = N(z; \mu^{(i)}, \sigma^{(i)})$
The function $q_\theta(z | x^{(i)})$ is not back-propagatable, and hence not useful in ML. The reparamaterization trick allows us to make it backpropagatable.

### Reparamaterization of Gaussian
$z = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$
where:
$\boldsymbol{\mu}$ is the "predicted" mean
$\sigma$ is the "predicted" variance
$\epsilon \sim N(0, I)$

This makes $\mu$ and $\sigma$ deterministic and can thus be back-propagated.

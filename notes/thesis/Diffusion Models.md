Diffusion models try to approximate the inverse of the gau
## Notations
| Math Notation | Description |
| - | - |
| $x_0 \sim q(x)$ | $x_0$ is a true sample from our dataset $x$ |
| $\beta_t \in (0,1)^{T}_{t=1}$ | $\beta_t$ is the step size at time step $t$ |

## How they work

Diffusion models work on the basis of two steps:

* Forward diffusion
	Slowly adding noise to an image sampled from training data, using the following step:
	$$ q(x_t | x_{t-1}) = N(x_{t}; \sqrt {1 - \beta_t} \cdot x_{t-1}, \beta_{t} \cdot I) $$	$$ q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1}) $$
	The idea is that when  $T \rightarrow \infty$, $q(x_{T} | x_{0} ) = N(0, 1)$.


* Backward diffusion
	Find the inverse of $q(x_t|x_{t-1})$ i.e. $q(x_{t-1}|x_t)$
	When we have that we can recreate data from the original dataset by sampling $X_T \sim N(0, I)$ and repeatedly applying $q(x_{t-1}|x_t)$.  We learn a model: $p_\theta$ which approximates this function.
	$$
	p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) 
	$$ $$
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$
	Using the [[Reparameterization Trick]] 





# Sources
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
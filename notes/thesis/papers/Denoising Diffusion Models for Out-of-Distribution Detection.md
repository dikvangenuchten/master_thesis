[paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.pdf)
[code](https://github.com/marksgraham/ddpm-ood)

### Abreviations
| short | written-out |
| -| -|
| OOD | Out of Distribution |
|DDPM | Denoising Diffusion Probabilistic Model |

# Main Idea
Detect OOD samples by reconstructing multiple increasingly noised samples, and comparing them to the original.

# Summary

## DDPM

Samples $x_t$ are generated by $q(x_t|x_0)$ same as in [[Diffusion Models]]
A reconstruction is made: $x_{0,t} = q(x_0|x_t)$.
Then the (perceptual) similarity is measured between $S(x_{0,t}, x_t)$ as well  as the $MSE(x_{0,t}, x_t)$.
This is repeated for a $N$ different values for $t$.
These scores are then used to produce an OOD score.
A validation set can be used to determine a good cutoff for OOD score, to differentiate between in-distribution and OOD.

## Results

DDPM is compared with the following baselines:
* Generative
	- Likelihood
	- WAIC
	- Typicality
	- [[Density-of-States Estimation]]
- Reconstruction
	- AutoEncoder
	- AutoEncoder Mahlabonis
	- MemAE
	- AnoDDPM-Mod
	- DDPM (This paper)

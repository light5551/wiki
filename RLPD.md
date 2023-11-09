---
tags:
  - online-RL
  - RL
paper: https://arxiv.org/pdf/2302.02948.pdf
---

## General

Основная идея заключается в обучении агента, используя  50% offline данных и  50% online. В качестве основы был взят алгоритм **SAC**. +Описывается workflow для подбора гиперпараметров и конфигуарция для решения различных задач 
## Алгоритм

Select `LayerNorm`, Large Ensemble Size $E$, Gradient
Steps $G$, and architecture.
Randomly initialize Critic $\theta_i$ (set targets $\theta_i^{\prime}=\theta_i$ ) for $i=1,2, \ldots, E$ and `Actor` $\phi$ parameters. Select discount $\gamma$, temperature $\alpha$ and critic EMA weight $\rho$.
Determine number of `Critic` targets to subset $Z \in\{1,2\}$
Initialize empty `replay buffer` $\mathcal{R}$
Initialize buffer $\mathcal{D}$ with offline data
while True do
	Receive initial observation state $s_0$
	for $\mathrm{t}=0, \mathrm{~T}$ do
		Take action $a_t \sim \pi_\phi\left(\cdot \mid s_t\right)$
		Store transition $\left(s_t, a_t, r_t, s_{t+1}\right)$ in $\mathcal{R}$
		for $g=1, G$ do
			Sample minibatch $b_R$ of $\frac{N}{2}$ from $\mathcal{R}$
			Sample minibatch $b_D$ of $\frac{N}{2}$ from $\mathcal{D}$
			Combine $b_R$ and $b_D$ to form batch $b$ of size $N$
			Sample set $\mathcal{Z}$ of $Z$ indices from $\{1,2, \ldots, E\}$
			With $b$, set
			$y=r+\gamma\left(\min _{i \in \mathcal{Z}} Q_{\theta_i^{\prime}}\left(s^{\prime}, \tilde{a}^{\prime}\right)\right), \quad \tilde{a}^{\prime} \sim \pi_\phi\left(\cdot \mid s^{\prime}\right)$
			Add entropy term $y=y+\gamma \alpha \log \pi_\phi\left(\tilde{a}^{\prime} \mid s^{\prime}\right)$
			for $i=1, E$ do
				Update $\theta_i$ minimizing loss:
				$L=\frac{1}{N} \sum_i\left(y-Q_{\theta_i}(s, a)\right)^2$
			end for
			Update target networks $\theta_i^{\prime} \leftarrow \rho \theta_i^{\prime}+(1-\rho) \theta_i$
		end for
		With $b$, update $\phi$ maximizing objective:
		$\frac{1}{E} \sum_{i=1}^E Q_{\theta_i}(s, \tilde{a})-\alpha \log \pi_\phi(\tilde{a} \mid s), \quad \tilde{a} \sim \pi_\phi(\cdot \mid s)$
	end for
end while


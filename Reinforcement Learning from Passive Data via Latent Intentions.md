---
tags:
  - RL
  - offline-RL
paper: https://arxiv.org/abs/2304.04782
---
## Проблема

Существует оргомное количество контента, которое описывает различные сценарии и так далее. Например, YouTube, но подобные данные являются пассивными, то есть не имеют никакой другой информации, что не подпадает под формализм классического **online** и **offline** RL. А именно не хватает информация  **Actions** и **Rewards**.

Когда идет активное обучение, то обучающий сигнал приходит от *Value*-функцию, которая ***измеряет насколько изменится кумулятивная награда, если будет взято действие A*.** 

## Основная идея
Заменить 2 недостающих компонентов на обучаемое представление намерений -- ???
Основные подидеи:
1. Предсказание возможного будущего состояния вместо получения наград
2. Использование латентного представления для целей вместо действий
## Цель 

Получить pre-train из пассивных данных, чтобы потом дообучить классическм offline RL'ем, но получить "качественную" модель

### Философия #1 
How likely am i to see ___ if I act to do ___ from this state?


### Intention-conditioned value function(ICVF) 

$V\left(s, s_{+}, z\right)=\mathbb{E}_{s_{t+1} \sim P_z\left(\cdot \mid s_t\right)}\left[\sum_t \gamma^t 1\left(s_t=s_{+}\right) \mid s_0=s\right]$ 
где $s$ - нынешнее состояние,  $s_{+}$ - будущее состояние, которое бы мы хотели увидеть, $z$ - желанная цель, которая определяет policy которую надо использовать    
## Алгоритм

Algorithm 1 Learning Intent-Conditioned Value Functions from Passive Data
1. Receive passive dataset of observation sequences $\mathcal{D}=\left(\left(s_0^i, s_1^i, \ldots,\right)\right)_{i=1}^n$
2. Choose intention set $\mathcal{Z}$ to model, e.g. the set of goal-reaching tasks
3. Initialize networks $\phi: S \rightarrow \mathbb{R}^d, \psi: S \rightarrow \mathbb{R}^d, T: \mathcal{Z} \rightarrow \mathbb{R}^{d \times d}$
4. Define ICVF $V_\theta\left(s, s_{+}, z\right)=\phi(s)^{\top} T(z) \psi\left(s_{+}\right)$and derived value model $V_\theta(s, z, z)=\phi(s)^{\top} T(z) \psi\left(r_z\right)$
5. repeat
		Sample transition $\left(s, s^{\prime}\right) \sim \mathcal{D}$, potential future outcome $s_{+} \sim \mathcal{D}$, intent $z \in \mathcal{Z}$
		Determine whether transition $s \rightsquigarrow s^{\prime}$ corresponds to acting with intent $z$ by measuring advantage
		$A=r_z(s)+\gamma V_\theta\left(s^{\prime}, z, z\right)-V_\theta(s, z, z)$
		Regress $V_\theta\left(s, s_{+}, z\right)$ to $1\left(s=s_{+}\right)+\gamma V_{\text {target }}\left(s^{\prime}, s_{+}, z\right)$ when advantage of $s \rightsquigarrow s^{\prime}$ is high under intent $z$
	$\mathcal{L}\left(V_\theta\right)=\mathbb{E}_{\left(s, s^{\prime}\right), z, s_{+}}\left[|\alpha-1(A<0)|\left(V_\theta\left(s, s_{+}, z\right)-1\left(s=s_{+}\right)-\gamma V_{\text {target }}\left(s^{\prime}, s_{+}, z\right)\right)^2\right]$
	until convergence
6. Return $\phi(s)$ as a state representation for use in downstream RL
где  $A$ - текущее предполагаемое преимущество при переходе $s -> s'$ , действуя согласно намерению $z$ 
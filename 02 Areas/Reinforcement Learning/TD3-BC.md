---
tags:
  - RL
  - Behavior-Cloning
  - offline-RL
paper: https://arxiv.org/pdf/2106.06860.pdf
name: A Minimalist Approach to Offline Reinforcement Learning
---
## General

Основная идея заключалась в минимальной модификации существующих алгоритмов для online RL. Был выбран TD3 и добавили BC:
$$\pi=\underset{\pi}{\operatorname{argmax}} \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\lambda Q(s, \pi(s))-(\pi(s)-a)^2\right]$$
От BC добавилось только $(\pi(s)-a)^2$ (Разница между экспертным поведением и политикой)  и $\lambda$ (коэффициент для регулирования насколько сильно мы хотим повторять поведение)

## Полезные ссылки

- https://github.com/sfujim/TD3_BC - PyTorch
---
tags:
  - RL
  - offline-RL
  - exploration
paper: https://arxiv.org/abs/2301.13616
---

Основная мысль заключается в модификации алгоритма SAC с добавлением RND бонуса (обычно RND  бонус зависит от *(s)*, но в данной работе зависит от пары **(s, a)**) и основная роль бонуса в **Anti-Exploration**. В данном случае смысл такой -- насколько действие **a** отличается от другий действий в этом состоянии, то есть насколько оно является OOD.

$$b(s, a)=\left\|f_\psi(s, a)-\bar{f}_{\bar{\psi}}(s, a)\right\|_2^2$$
где $f_\psi(s, a)$ -- predictor network, а $\bar{f}_{\bar{\psi}}(s, a)$ -- prior network (Случайные веса и не меняется)

Temporal difference выглядит следующим образом:
$$r+\gamma \mathbb{E}_{a^{\prime} \sim \pi\left(\cdot \mid s^{\prime}\right)}\left[Q\left(s^{\prime}, a^{\prime}\right)-b\left(s^{\prime}, a^{\prime}\right)\right]$$

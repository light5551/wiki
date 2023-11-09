---
tags:
  - RL
  - offline-RL
paper: https://arxiv.org/abs/2307.11949
name: HIQL Offline Goal-Conditioned RL with Latent States as Actions
---
1. Обучается action free [[IQL]], целью которого оптимизировать  $\min \mathbb{E}\left[\left(-1(s \neq g)+\gamma \max _{s^{\prime}} V\left(s^{\prime}, g\right)-V(s, g)\right)^2\right]$ 
2. Extract a high-level policy $\pi(s|a, g)$ to maximize the value function $\max \mathbb{E}\left[\log \pi^h\left(s_{t+k} \mid s_t, g\right) e^{V\left(s_{t+k}, g\right)-V\left(s_t, g\right)}\right]$ - отвечает за следующий waypoint (subgoal)
3. Extract a low-level policy  $\max \mathbb{E}\left[\log \pi^{\ell}\left(a \mid s_t, s_{t+k}\right) e^{V\left(s_{t+1}, s_{t+k}\right)-V\left(s_t, s_{t+k}\right)}\right]$ -- отвечает за действие
![[Pasted image 20231109142639.png]]

где $s$, $s'$, $g$ - нынешнее состояние, следующее состояние, целевое состояние, соответственно.

##  Основная мысль
Работать с неполностью action-free данными. На стадии pretrain предлагается обучается на версии IQL без действий, а после разбить на две подполитики, одна из которых уже обучается на данных (Поэтому и неполностью action-free данные), но авторы говорят, что подобных данных надо гораздо меньше, чем данных для pretrain. Основной профит статьи заключается в том, что мы сглаживаемм шум, который особенно ощутим от goal'a, который мешает выбрать оптимальную траекторию(Более подробно есть изображение на сайте с зашумленным пространством V-функции (на картинке выше показана уже очищенное пространство)) 
## Недостатки

Как отметили авторы плохо работает в стохастических средах, так как value функцию обучаются предсказывать шум.
## Полезные ссылки
1. https://seohong.me/projects/hiql/
2. https://github.com/seohongpark/HIQL - *Jax*

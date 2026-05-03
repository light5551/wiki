---
Github: https://github.com/amazon-far/fpo-control
tags:
  - RL
  - Robotics
---
Так как мы используем [[Flow Matching]], то есть векторное поле, то проблематично считать log likelihood.

**Проблема:**
В PPO:
$$\rho_t = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\text{old}}(a_t \mid o_t)}$$
что тоже самое как
$$\rho_t = \exp\left( \log \pi_\theta(a_t \mid o_t) - \log \pi_{\text{old}}(a_t \mid o_t) \right)$$
Суррогат FPO — разница логарифмов правдоподобия можно аппроксимировать через разницу в потоках CFM.

---
**CFM loss**
Для $a_t∈A$ и  $o_t∈O$  генерируются шумы $\varepsilon_i$​ и промежуточные моменты $t_i$​. Модель предсказывает скорость $v_θ​$:
$$L_\theta^{(i,t)} = \left\| v_\theta(x_{t_i}, x_t, o_t) - (a_t - \varepsilon_i) \right\|_2^2$$
---
В FP0++ отношение правдоподобий (ratio) вычисляется для каждого сэмпла, путем вычисления нового лога из старого и возведения в экспоненту:
$$\rho_{\text{FPQ++}}^{(i)}(\theta) = \exp\left( L_\theta^{(i,t)} - L_{\theta_{\text{old}}}^{(i,t)} \right)$$

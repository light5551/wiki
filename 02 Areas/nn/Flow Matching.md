**Flow Matching**

$v_t(x)$ — векторное поле
Поток описывается обыкновенным дифференциальным уравнением ODE([[ODE&SDE]]):
$\frac{dx_t}{dt} = v_t(x_t)$
Эволюция плотности вероятности $p_t(x)$:
$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0$
---
**Задача:**
1. $x_0∼p_0$​  
2. $x_1∼p_1$
3. Найти $p_t(x∣x_1)$
---
**Линейная интерполяция:**
$x_t = (1 - t)x_0 + t x_1$ — путь движения
Скорость движения:
$\dot{x}_t = u_t(x_t \mid x_0, x_1) = x_1 - x_0$
---
**Loss (CFM):**
$L_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U(0,1),\, x_0 \sim p_0,\, x_1 \sim p_1}\left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]$
---
tags:
  - RL
  - offline-RL
  - online-RL
paper: https://sites.google.com/view/fastrlap
---

![[Pasted image 20231102182002.png]]

В статье предлагается решить задачу обучения агрессивному стилю вождению по датасету с различных роботов, где есть только медленное движение. В работе используется последовательность алгоритмов для обучения **[[IQL]]** и **[[RLPD]]** для offline и online RL, соответственно. Время на дообучение занимало ~**20** мин, за счет замораживания encoder'a. Checkpoint'ы которые генерируется во время движения агента не должны быть в поле видимости observation. Из интересного, action space имеет динамически границы, а именно - следующее действие не может отличать от предыдущего на коэффициент(гиперпараметр). В качестве входных данных используется RGB + вектор скорости (из колесной одометрии) + вектор ускорения с IMU + вектор до цели(до следующего checkpoint'a).  FSM состоит из состояний:
1. Политика выбирает действие
2. Pseudo-reset когда машинка врезалась -- начинает ехать назад, чтобы выбраться из терминального состояния
## Алгоритм

1. Data: Prior navigation dataset $\mathcal{D}$, slow demo lap $\mathcal{B}_{\text {slow }}$
2. Keys: Pre-Training, Practicing, Online RL
3. while Encoder is not converged do
		$s, a, s^{\prime}, \operatorname{idx} \leftarrow \operatorname{LoadData}(\mathcal{D})$
		$g \leftarrow \operatorname{LoadFutureData}(\mathcal{D}$, idx $+\operatorname{RandomOffset}())$
		$r \leftarrow \operatorname{ComputeReward}(s, a, g)$
		$\operatorname{Train}_{\mathrm{IQL}}\left((s, g), a, r,\left(s^{\prime}, g\right)\right)$
4. while True do
	On Robot
		$s \leftarrow$ Observe ()
		if $s$ near $g$ then
			$g \leftarrow \operatorname{NextCheckpoint}(g)$
		$r \leftarrow \operatorname{ComputeReward}\left(s_{\text {prev }}, a_{\text {prev }}, g\right)$
		SendToWorkstation $\left(s_{\text {prev }}, a_{\text {prev }}, r, s, g\right)$
		$a \sim \pi\left(\phi\left(s_{\text {image }}\right), s_{\text {proprio }}, g\right)$
		Actuate $(a)$
		if Collision or Stuck then
			Execute recovery policy
	On Workstation
		ReceiveFromRobot $(\mathcal{B})$
		$b \leftarrow \operatorname{Sample}(\mathcal{B}), b_{\text {slow }} \leftarrow \operatorname{Sample}\left(\mathcal{B}_{\text {slow }}\right)$
		$\pi, Q \leftarrow \operatorname{Train}_{\mathrm{RLPD}}\left(\pi, Q, b, b_{\text {slow }}\right)$

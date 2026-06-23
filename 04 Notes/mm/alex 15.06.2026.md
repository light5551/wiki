1. **Поставить обучение с VLA+dyn**
2. Cравнение dyn model
	1. Как считать метрика
	2. Почему такая метрика
3. Отчет по VLA+dyn model
4. VLA + goal image
		1. Start train. Branch: `sg/train_goal_image` 

## Сравнение VLA using different dyn models

| Dyn model(data)           | Dyn model type | VLA data      | Trainable VLA | Is Done (dyn model) | Is Done(VLA) | Exp Link |
| ------------------------- | -------------- | ------------- | ------------- | ------------------- | ------------ | -------- |
| WBC-data                  | GRU            | 10h + 4k_tuni | +             | yes                 | yes          |          |
| WBC-data                  | MLP            | 10h + 4k_tuni | +             | yes                 | yes          |          |
| WBC-data                  | Jepa           | -             | -             | yes                 | no           |          |
| 4k_tuni                   | MLP            | -             | -             | yes                 | no           |          |
| 4k_tuni                   | GRU            | 10h + 4k_tuni | +             | yes                 | no           |          |
| Zest                      | GRU            | 10h + 4k_tuni | +             | no                  | no           |          |
| WBC-data + Zest + 4k_tuni | GRU            | 10h + 4k_tuni | +             | no                  | no           |          |

### Что нужно показать?
1. Cравнение метрик дин. моделей на T  step, using cum. mae, mse
	1. Figure (different Dyn model data)
	2. Figure (Diff. arch)
2. Compare VLA using different dyn models
	1. Figure of 5 experiments(Trainable VLA == "+")
	2. 5 checkpoint to tests in vhal/robot
### 1. 📄Reference
Авторы, год, venue + ссылки (paper / code).
### 2.  🎯Task & Inputs
- Тип задачи (point-goal / VLN / social / map-based / exploration)
- Наблюдения (RGB / Depth / Lidar / Language / Map)
- Выход (velocity / waypoints / discrete actions)

### 3. 🧠 Core Method
- Архитектура (e.g. VLM + policy head / Transformer / Diffusion / CNN+LSTM)
- Тип обучения (IL / RL / PPO / offline RL / hybrid)
- Memory / context (если есть)
- Использование foundation models

### 4. 📊 Evidence
- Где тестировали (sim / real / datasets)
- 1–2 ключевых результата
- Есть ли generalization / zero-shot / sim2real
### 5. 🔧Relevance for Our Stack

Инженерная оценка:

- Что можно заимствовать (архитектура / loss / pipeline)
- Совместимость с PPO / RPO / lidar+goal setup
- Требования к данным
- Реалистичность real-time
- Стоит ли экспериментировать
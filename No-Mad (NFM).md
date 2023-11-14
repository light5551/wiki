branch: [nomad v1](https://git.sberrobots.ru/foundation/navigation/vint/-/tree/nomad_v1?ref_type=heads)
### Inputs:
- N images
- goal
### Outputs:
- N local waypoints

### Первая мысль:
Локальный путь в пределах 2 метров по goal(в пределах видимости или почти) изображению и по последовательности K-прошедших кадров. На выход выдает N waypoint'ов, через которые нужно построить траекторию движения.
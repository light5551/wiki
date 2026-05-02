Для отключения касперского

```bash
sudo systemctl stop klnagent64.service
sudo systemctl stop kesl

```

Для рестарта 

```bash
sudo systemctl restart klnagent64.service && sudo systemctl restart kesl
```

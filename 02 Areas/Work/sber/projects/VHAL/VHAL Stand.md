## TODO

- [ ] Добавить информация как запустить и останавливать. Мб в CLI
- [ ] Добавить функционал логов
- [ ] Добавить базовые таски в Vhal
	- [ ] demo zone + box/table
	- [ ] demo zone w/ table manipulation
- [ ] Инструкция для остальных как запустить
- [ ] Статистика как часто пользуются 
## Prerequisites

1. Иметь VPN для сети 10.10.0.0/24
2. Установить `sudo apt install tigervnc-viewer`

## How to connect Vhal Stand

1. Первый терминал: `ssh -L 5901:localhost:5901 dual4090@10.10.0.14` Пароль: `пробел`
2. Второй терминал `xtigervncviewer localhost:5901` . Пароль: `123456`

## Usage
1. В VNC открыть терминал и перейти в папке `./vhal -L`
2. `./isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage --no-sandbox`

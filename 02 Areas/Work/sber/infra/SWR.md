
## Доступ

1. Получить credentials.csv из cloud'a
2. ```
   docker login -u ru-moscow-1@$(cat credentials.csv | awk -F',' 'NR==2 {print $2}') -p $(printf "$(cat credentials.csv | awk -F',' 'NR==2 {print $2}')" | openssl dgst -binary -sha256 -hmac "$(cat credentials.csv | awk -F',' 'NR==2 {print $3}')" | od -An -vtx1 | sed 's/[ \n]//g' | sed 'N;s/\n//') swr.ru-moscow-1.hc.sbercloud.ru

  ```
  
## 1. Установка Docker

```bash
sudo apt install docker.io -y
```

Если сервис не стартовал:

```bash
sudo systemctl start docker
sudo systemctl enable docker
```

## 2. Настройка прав (чтобы запускать docker без sudo)

```bash
sudo usermod -aG docker ruslan
```

> Выйдите и зайдите в систему, чтобы группа применилась.

## 3. Создаём файл `train.py`

Минимальный пример (структура):

```python
def train():
    train_path = '/app/BIG_DATA/train.csv'
    train_data = pd.read_csv(train_path)

    # обучение модели
    # расчет метрик
    # сохранение модели и метрик в /app/output
    with open("output/metrics.json", "w") as f:
        f.write("{...}")

if __name__ == "__main__":
    train()
```

Поместите его в ту же папку, где будет Dockerfile.

## 4. Создаём `Dockerfile`

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY train.py .

RUN pip install pandas scikit-learn numpy

CMD ["python", "train.py"]
```

### Создаём `.dockerignore`

```
BIG_DATA/
```

## 5. Сборка образа

```bash
docker build . -t trainer
```

## 6. Запуск контейнера с монтированием папок

```bash
docker run \
    -v /home/user/BIG_DATA:/app/BIG_DATA \
    -v /home/user/output:/app/output \
    trainer
```

Где:

* `/home/user/BIG_DATA` — локальная папка с данными
* `/home/user/output` — куда сохранять модель/метрики
* `/app/...` — пути внутри контейнера


## 7. Загрузка файлов на удалённый сервер через SCP


Загрузить один файл:
scp train.py user@SERVER_IP:/home/user/

Загрузить целую директорию проекта:
scp -r ./project_dir user@SERVER_IP:/home/user/

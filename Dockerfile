# TimAI - Advanced Multi-Model Trading System
FROM python:3.10-slim

# Метаданные
LABEL maintainer="TimAI Team"
LABEL description="Advanced Multi-Model Trading System"
LABEL version="2.0.0"

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Обновляем систему и устанавливаем зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p data/historical data/live models/production models/experimental logs

# Устанавливаем права доступа
RUN chmod +x api/trading_api.py

# Открываем порт для API
EXPOSE 8000

# Проверка здоровья
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Команда по умолчанию - запуск API
CMD ["python", "api/trading_api.py"] 
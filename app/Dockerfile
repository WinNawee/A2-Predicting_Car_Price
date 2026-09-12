FROM python:3.11-slim

WORKDIR /app

# copy requirements first so this layer is cached unless deps change
COPY code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/ .

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]

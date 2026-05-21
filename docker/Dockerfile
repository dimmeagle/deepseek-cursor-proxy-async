FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/dimmeagle/deepseek-cursor-proxy-async.git . && \
    apt-get remove -y git && apt-get autoremove -y

RUN pip install --no-cache-dir -e .

EXPOSE 9000

CMD ["sh", "-c", "deepseek-cursor-proxy --host 0.0.0.0 --port 9000 --no-ngrok"]
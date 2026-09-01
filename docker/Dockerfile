FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/dimmeagle/deepseek-cursor-proxy-async.git . && \
    apt-get remove -y git && apt-get autoremove -y

RUN pip install --no-cache-dir -e .

EXPOSE 9000

#CMD ["sh", "-c", "deepseek-cursor-proxy --host 0.0.0.0 --port 9000 --no-ngrok"]

CMD ["sh", "-c", "deepseek-cursor-proxy --host 0.0.0.0 --port 9000 --no-ngrok --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen3.8-flash --api-key sk-ws-H.DDHDXEE.hb0D.MEYCIQC34WBCKYiIlNkRron2yPSwpQqD3LsWajIdE8W-zE1htQIhAJ04ZoI6YZeC57sLIZb-z2_-GfCQSY6NZSZz9RALNXhI --thinking disabled"]
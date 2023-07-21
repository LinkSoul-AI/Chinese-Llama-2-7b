FROM nvcr.io/nvidia/pytorch:23.06-py3

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install accelerate==0.21.0 bitsandbytes==0.40.2 gradio==3.37.0 protobuf==3.20.3 scipy==1.11.1 sentencepiece==0.1.99 transformers==4.31.0

WORKDIR /app

COPY example/basic-chat/* ./

CMD ["python", "app.py"]

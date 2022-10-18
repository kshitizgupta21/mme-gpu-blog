FROM 850464037171.dkr.ecr.ap-south-1.amazonaws.com/tritonserver:22.07-py3

RUN  pip install --upgrade pip \
    && pip install \
        torch \
        transformers \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache
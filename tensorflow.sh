docker run -it --rm --gpus all -p 8888:8888 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
       -v ${PWD}:/workspace \
       -w /workspace \
       nvcr.io/nvidia/tensorflow:22.09-tf2-py3

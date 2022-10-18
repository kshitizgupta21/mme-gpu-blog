docker run --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it \
            -v `pwd`/workspace:/workspace nvcr.io/nvidia/tensorflow:22.09-tf2-py3 \
            /bin/bash generate_distilbert_tf.sh

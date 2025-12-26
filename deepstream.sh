docker run --rm --gpus all -it nvcr.io/nvidia/deepstream:8.0-triton-dgx-spark nvidia-smi

docker run -it --rm --gpus all \
    --network=host \
    -v /home/ntust_spark/playbook_gptoss/deepstream_videos:/opt/nvidia/deepstream/deepstream-8.0/videos \
    nvcr.io/nvidia/deepstream:8.0-triton-dgx-spark 


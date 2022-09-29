FROM nvidia/cuda:10.2-cudnn7-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl


RUN apt-get remove --auto-remove python3
RUN apt-get -y install python3.7
RUN apt-get install python3-distutils -y
RUN apt-get install python3-apt -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py

RUN apt-get install unzip

RUN apt-get install make
RUN apt-get update
RUN apt-get install zlib1g-dev -y
RUN apt-get install libjpeg-dev -y
RUN apt-get install git -y


# COPY pyproject.toml .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir poetry && \
#     poetry install


RUN pip install pillow
RUN pip install cmake
RUN pip install wandb
RUN pip install huggingface_hub
RUN pip install omegaconf
RUN pip install matplotlib==3.5.0

RUN pip install flake8==3.8.3
RUN pip install fastapi==0.70.0
RUN pip install hypercorn==0.13.2
RUN pip install celery==5.2.3
RUN pip install redis==4.1.0

RUN apt-get install git -y

RUN pip install "git+https://github.com/affanmehmood/DALLE-pytorch"
# RUN git clone https://github.com/affanmehmood/taming-transformers --quiet
RUN pip install taming-transformers-rom1504

RUN pip install pytorch-lightning==1.7.7

ENV DOCKER=true

EXPOSE 8000
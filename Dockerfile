FROM nvcr.io/nvidia/pytorch:21.07-py3
WORKDIR /usr/home
COPY . .
RUN apt-get update
RUN apt-get install htop
RUN apt-get -y install git
RUN apt-get -y install ssh
RUN apt-get -y install tmux
RUN pip install -U pip
RUN pip install git+git://github.com/carefree0910/carefree-toolkit@dev
RUN pip install git+git://github.com/carefree0910/carefree-data@dev
RUN pip install git+git://github.com/carefree0910/carefree-ml@dev
RUN pip install .
RUN pip install gpustat
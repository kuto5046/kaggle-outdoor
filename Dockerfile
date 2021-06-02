# pytorch versionに注意
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

# 時間設定
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

ENV DEBIAN_FRONTEND=noninteractive
# install basic dependencies
RUN apt-get -y update && apt-get install -y \
    sudo \
    wget \
    cmake \
    vim \
    git \
    tmux \
    zip \
    unzip \
    gcc \
    g++ \
    build-essential \
    ca-certificates \
    software-properties-common \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpng-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libsndfile1 \
    zsh \
    xonsh \
    # neovim \
    nodejs \
    npm \
    curl

RUN apt-get install -y \
    python3.8  \
    python3.8-dev \
    python3-pip \
    python3-ipdb

# node js を最新Verにする
RUN npm -y install n -g && \
    n stable && \
    apt purge -y nodejs npm

# set path
ENV PATH /usr/bin:$PATH

RUN wget https://github.com/neovim/neovim/releases/download/nightly/nvim.appimage
RUN chmod u+x nvim.appimage
# RUN ./nvim.appimage --appimage-extract
# RUN ./squashfs-root/AppRun --version

# Optional: exposing nvim globally
# RUN mv squashfs-root / && ln -s /squashfs-root/AppRun /usr/bin/nvim
RUN /usr/bin/nvim

# install common python packages
COPY ./requirements.txt /
RUN pip3 install --upgrade pip && \
    pip3 install -r /requirements.txt
# https://qiita.com/Hiroaki-K4/items/c1be8adba18b9f0b4cef
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# jupyter用にportを開放
EXPOSE 8888
EXPOSE 5000
EXPOSE 6006

# add user
ARG DOCKER_UID=1000
ARG DOCKER_USER=user
ARG DOCKER_PASSWORD=kuzira
RUN useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
  && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

# for user
RUN mkdir /home/${DOCKER_USER}/.kaggle
COPY ./kaggle.json /home/${DOCKER_USER}/.kaggle/

# set working directory
RUN mkdir /home/${DOCKER_USER}/work
WORKDIR /home/${DOCKER_USER}/work
# 本当はよくないがkaggle cliがuserで使えないので600 -> 666
RUN chmod 666 /home/${DOCKER_USER}/.kaggle/kaggle.json

# switch user
USER ${DOCKER_USER}

RUN git clone https://github.com/kuto5046/dotfiles.git /home/${DOCKER_USER}/dotfiles
RUN bash /home/${DOCKER_USER}/dotfiles/.bin/install.sh }

# jupyter lab
ENV PATH /home/user/.local/bin:$PATH
# RUN wget -q -O - https://linux.kite.com/dls/linux/current
RUN pip install \
    jupyterlab
    # 'jupyterlab-kite>=2.0.2' \
    # jupyterlab-git \ 
    # lckr-jupyterlab-variableinspector \
    # black \
    # jupyterlab_code_formatter
    # jupyterlab-lsp

RUN jupyter labextension install \
    @axlair/jupyterlab_vim \
    @arbennett/base16-nord \
    @jupyterlab/toc
    # @kiteco/jupyterlab-kite
    # @ryantam626/jupyterlab_code_formatter 
# RUN jupyter labextension install jupyterlab-plotly@4.14.3   
# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3
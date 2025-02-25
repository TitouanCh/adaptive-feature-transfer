FROM ubuntu:20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    curl \
    build-essential \
    git \
    nano \
    neovim \
 && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Optionally, set python3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN python3 -m pip install --upgrade pip

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir matplotlib seaborn wandb && \
    pip install --no-cache-dir "transformers[torch]" datasets evaluate scikit-learn && \
    pip install --no-cache-dir fire timm && \
    pip install --no-cache-dir ftfy regex tqdm numba && \
    pip install --no-cache-dir git+https://github.com/openai/CLIP.git

WORKDIR /work

COPY ./work /work

CMD ["/bin/bash"]

FROM cuda-ros-base:noetic

ARG USERNAME=vscode
ARG USER_UID=1000
ENV ROS_DISTRO=noetic
ARG PIXI_VERSION=v0.48.0

ENV DEBIAN_FRONTEND=noninteractive

# install ros-perception dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-perception \
    ros-$ROS_DISTRO-rviz \
    ros-$ROS_DISTRO-realsense2-camera \
    ros-$ROS_DISTRO-rqt-graph \
    ros-$ROS_DISTRO-rqt-tf-tree \
    libmetis-dev libgflags-dev libgoogle-glog-dev googletest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install taskfile
RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

# install ros2 develop tools
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    python3-vcstool \
    python3-rosdep ccache ninja-build && \
    python3-catkin-tools \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl bzip2 ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

# create user
RUN groupadd --gid $USER_UID $USERNAME && \
    useradd --create-home --shell /bin/bash --uid $USER_UID --gid $USER_UID $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN curl -L -o /usr/local/bin/pixi -fsSL --compressed "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl" \
    && chmod +x /usr/local/bin/pixi \
    && pixi info

# change to USER and install rust
USER $USERNAME
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.zshrc && \
    . "$HOME/.cargo/env" && \
    rustup default stable

RUN echo 'eval "$(pixi completion -s zsh)"' >> $HOME/.zshrc

# install Miniforge (Mamba/Conda) for the user
# RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
#     bash /tmp/miniforge.sh -b -p $HOME/miniforge && \
#     rm /tmp/miniforge.sh && \
#     echo 'export PATH="$HOME/miniforge/bin:$PATH"' >> $HOME/.bashrc && \
#     $HOME/miniforge/bin/conda init bash

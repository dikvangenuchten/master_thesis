FROM nvcr.io/nvidia/pytorch:24.02-py3 as base

COPY requirements.tx[t] requirements.txt
RUN touch requirements.txt && pip install -r requirements.txt

FROM base as dev

ARG USERNAME=dev-user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Copy and install development requirements (iff it is present in build-context)
COPY requirements-dev.tx[t] .
# Create the file if it not exists, then install using pip
RUN touch requirements-dev.txt && pip install -r requirements-dev.txt

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo tmux \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

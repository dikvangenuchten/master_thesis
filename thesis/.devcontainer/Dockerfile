FROM ghcr.io/xu-cheng/texlive-full:latest as base

from base as dev

RUN apk --update add imagemagick

ARG USERNAME=dev-user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
#
RUN adduser -D $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apk add --update sudo git-lfs \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ARG ISAACLAB_BASE_IMAGE_ARG

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# # Install whatever you need as additional dependencies.
RUN bash -i -c "source ${HOME}/.bashrc && \
    pip install -e '${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/robot_learning_lab_zoo[isaaclab]' && \
    pip install -e ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/robot_learning_lab_datasets && \
    pip install -e '${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/robot_learning_lab_tasks[isaaclab]' && \
    pip install -e ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/rll_rl"

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace

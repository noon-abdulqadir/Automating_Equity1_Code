# Start your image with a python 3.10-bullseye base image
# syntax=docker/dockerfile:1
# FROM python:3.10-bullseye
FROM condaforge/mambaforge:4.9.2-5 as conda

# The /Automating_Equity1_Code directory should act as the main application directory
WORKDIR /Automating_Equity1_Code

# Copy all files from the current directory to the /Automating_Equity1_Code directory
COPY . .

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# # Copy environment.yml (if found) to a temp location so we update the environment. Also
# # copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# # COPY environment.yml environment.yml requirements.txt requirements.txt /tmp/conda-tmp/
# #/tmp/conda-tmp/
# RUN /opt/conda/bin/conda update --name base --channel conda-forge --yes --file environment.yml
# # RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
# #     && rm -rf /tmp/conda-tmp

# Install python dependencies from requirements.txt
RUN $(which python) -m pip install --upgrade pip
RUN $(which python) -m spacy download en_core_web_sm

# Start from a Miniconda image
FROM continuumio/miniconda3:latest

# Create a working directory
WORKDIR /app

# Copy the environment.yml file into the container
COPY scgpt_bench.yml .

# Create the environment
# Using `mamba` here for faster installations; it's included in newer images.
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -f scgpt_bench.yml

# Activate the environment by default
# The conda environment will be located at /opt/conda/envs/myenv
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Clean up conda cache to reduce image size
RUN conda clean --all --yes

# (Optional) Set a default command to start a shell
CMD ["/bin/bash"]

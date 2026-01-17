FROM mambaorg/micromamba:1.5.8

WORKDIR /app

# Copy environment first for better layer caching
COPY environment.yml /app/environment.yml

# Create conda env
RUN micromamba create -y -n geo -f /app/environment.yml \
  && micromamba clean -a -y

# Use the env for subsequent commands
ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

# Copy app code
COPY app /app/app

EXPOSE 8501

CMD ["bash", "-lc", "echo 'Open http://localhost:8501' && micromamba run -n geo streamlit run app/main.py --server.address=0.0.0.0 --server.port=8501"]


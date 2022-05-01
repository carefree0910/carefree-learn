FROM nvcr.io/nvidia/pytorch:21.07-py3
WORKDIR /usr/home
COPY . .
RUN apt-get update && \
    echo "=============================================================================="  && \
    echo "Freeing up disk space on CI system"  && \
    echo "=============================================================================="  && \
    echo "Listing 100 largest packages"  && \
    dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 100  && \
    df -h  && \
    echo "Removing large packages"  && \
    sudo apt-get remove -y '^ghc-8.*'  && \
    sudo apt-get remove -y '^dotnet-.*'  && \
    sudo apt-get remove -y 'php.*'  && \
    sudo apt-get remove -y azure-cli google-cloud-sdk hhvm google-chrome-stable firefox powershell mono-devel  && \
    sudo apt-get autoremove -y  && \
    sudo apt-get clean  && \
    df -h  && \
    echo "Removing large directories"  && \
    rm -rf /usr/share/dotnet/  && \
    df -h  && \
    apt-get install htop && \
    apt-get -y install git && \
    apt-get -y install ssh && \
    apt-get -y install tmux && \
    pip install -U pip && \
    pip install . && \
    pip install gpustat
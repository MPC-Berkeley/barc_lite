# Start from ROS 2 Foxy base image
FROM ros:foxy-ros-base

RUN apt update

# Text editor
RUN apt install nano

# To test graphical forwarding
RUN apt install -y ros-foxy-turtlesim

RUN apt install -y python3-pip python3-argcomplete
RUN python3 -m pip install -U pip

# Install Python dependencies
RUN python3 -m pip install -U numpy scipy matplotlib
RUN python3 -m pip install casadi==3.5.5 pyqtgraph
RUN python3 -m pip install pyqt5

# Install pyomo and ipopt
RUN apt install -y wget
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$PATH:$CONDA_DIR/bin

RUN python3 -m pip install pyomo
RUN conda install -c conda-forge ipopt  

# Install Jupyter Lab
RUN python3 -m pip install -U jupyter
RUN python3 -m pip install jupyterlab
ENV JUPYTER_ENABLE_LAB yes
ENV JUPYTER_TOKEN docker

# Make default data directory
RUN mkdir /project_data

# Clone barc_lite repo
RUN git clone https://github.com/MPC-Berkeley/barc_lite.git

# Build ros workspace
WORKDIR /barc_lite/workspace
RUN /bin/bash -c "source /opt/ros/foxy/setup.sh && colcon build --symlink-install"

# Source barc_lite packages
RUN touch /root/.bashrc
RUN /bin/bash -c "echo -e 'source /opt/ros/foxy/setup.bash' >> /root/.bashrc && \
                  echo -e 'source /barc_lite/workspace/install/setup.bash' >> /root/.bashrc"

WORKDIR /
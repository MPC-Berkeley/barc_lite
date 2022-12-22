#!/bin/bash

# Copy contents of mounted directory to 
# /barc_lite/workspace/src/mpclab_controllers/mpclab_controlers/lib
export MOUNT_DIRECTORY="/project_code" # Change this to where you mounted the directory containing your code
cp -r ${MOUNT_DIRECTORY}/* /barc_lite/workspace/src/mpclab_controllers/mpclab_controllers/lib

# Rebuild ROS packages
cd /barc_lite/workspace
colcon build --symlink-install
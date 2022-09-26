FROM librealsense/librealsense
MAINTAINER dlcksgur3434@gmail.com
COPY requirements.txt .
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-setuptools \
    python3-pip \
	python3-dev
Run apt-get update -y
RUN apt-get install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
RUN apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
RUN apt-get install gnupg -y
RUN mkdir /home/user/human_display_interaction
RUN apt-get update -y
RUN apt-get install -y build-essential
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get -y install libusb-1.0-0-dev
RUN pip install -r requirements.txt
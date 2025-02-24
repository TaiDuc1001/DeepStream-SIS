sudo apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
    python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev

cd ~/deepstream_python_apps
git submodule update --init

# gstreamer-python
sudo apt-get install -y apt-transport-https ca-certificates -y
sudo update-ca-certificates
cd 3rdparty/gst-python/
./autogen.sh
make
sudo make install

# compiling the bindings
cd ~/deepstream_python_apps/bindings
mkdir build
cd build
cmake .. -DPIP_PLATFORM=linux_aarch64
make

pip3 install ./pyds-1.1.5-py3-none*.whl

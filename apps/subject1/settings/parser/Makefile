CUDA_VER?=11.4
CFLAGS:= -Wall -std=c++11 -shared -fPIC
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include
LIBS:= -lnvinfer -lnvparsers -lstdc++
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

TARGET_LIB:= libnvdsinfer_custom_bbox_parser.so
SRCS:= p.cpp

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
cd custom
make
cd -
GST_DEBUG=3 python3 main.py \
-i \
file:///home/nvidia/deepstream_python_apps/apps/subject1/videos/DongKhoi_MacThiBuoi.mp4 \
file:///home/nvidia/deepstream_python_apps/apps/subject1/videos/RachBungBinh_NguyenThong_2.mp4 \
file:///home/nvidia/deepstream_python_apps/apps/subject1/videos/TranHungDao_NguyenVanCu.mp4 \
file:///home/nvidia/deepstream_python_apps/apps/subject1/videos/TranKhacChan_TranQuangKhai.mp4
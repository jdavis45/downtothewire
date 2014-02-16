cmake_minimum_required(VERSION 2.8)
project( design_challenge )
SET(COMPILE_DEFINITIONS -Werror)
include_directories(/opt/vc/userland/host_applications/linux/libs/bcm_host/include)
include_directories(/opt/vc/userland/interface/vcos)
include_directories(/home/pi/design_challenge)
include_directories(/opt/vc/include)
include_directories(/opt/vc/userland)
include_directories(/opt/vc/userland/interface/vcos/pthreads)
include_directories(/opt/vc/userland/interface/vmcs_host/linux)
include_directories(/usr/local/include/opencv)
include_directories(/usr/local/include/opencv2)

set (GL_SCENE_SOURCES
   gl_scenes/models.c
   gl_scenes/mirror.c
   gl_scenes/yuv.c
   gl_scenes/sobel.c
   gl_scenes/square.c
   gl_scenes/teapot.c)

set (COMMON_SOURCES
   RaspiCamControl.c
   RaspiCLI.c
   RaspiPreview.c
   RaspiTex.c
   RaspiTexUtil.c
   tga.c
   facialrecognizer.cpp)
   
add_executable(design_challenge  ${COMMON_SOURCES} main.cpp ${GL_SCENE_SOURCES} )

set (MMAL_LIBS /opt/vc/lib/libmmal_core.so /opt/vc/lib/libmmal_util.so /opt/vc/lib/libmmal_vc_client.so)
set (OPENCV_LIBS /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_contrib.so)

target_link_libraries(design_challenge ${MMAL_LIBS} /opt/vc/lib/libvcos.so /opt/vc/lib/libbcm_host.so /opt/vc/lib/libGLESv2.so /opt/vc/lib/libEGL.so ${OPENCV_LIBS} m)

#target_link_libraries(design_challenge /opt/vc/lib/libGLESv2.so /opt/vc/lib/libEGL.so /opt/vc/lib/libmmal_core.so /opt/vc/lib/libmmal_util.so /opt/vc/lib/libmmal_vc_client.so #/opt/vc/lib/libvcos.so /opt/vc/lib/libbcm_host.so )

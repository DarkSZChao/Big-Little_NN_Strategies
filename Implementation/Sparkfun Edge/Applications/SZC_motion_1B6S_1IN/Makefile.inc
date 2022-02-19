DEMO_SRCS := \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/main.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/S0_s3.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/S1_s3.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/S2_s3.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/S3_s3.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/S4_s3.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/S5_s3.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/B_s123_1IN_91.cc

DEMO_HDRS := \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/model.h \
tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/model/SparkFun_MOTION_Detector_s123_1IN.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/SZC_motion_1B6S_1IN/*/Makefile.inc)

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,SZC_motion_1B6S_1IN,\
$(DEMO_SRCS),$(DEMO_HDRS)))


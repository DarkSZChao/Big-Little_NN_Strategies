DEMO_SRCS := \
tensorflow/lite/micro/examples/SZC_motion_B_s3/main.cc \
tensorflow/lite/micro/examples/SZC_motion_B_s3/model/multi_motions.cc

DEMO_HDRS := \
tensorflow/lite/micro/examples/SZC_motion_B_s3/model/model.h \
tensorflow/lite/micro/examples/SZC_motion_B_s3/model/SparkFun_MOTION_Detector.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/SZC_motion_B_s3/*/Makefile.inc)

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,SZC_motion_B_s3,\
$(DEMO_SRCS),$(DEMO_HDRS)))


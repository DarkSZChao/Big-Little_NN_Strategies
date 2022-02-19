DEMO_SRCS := \
tensorflow/lite/micro/examples/SZC_motion_1BDD_1IN/main.cc \
tensorflow/lite/micro/examples/SZC_motion_1BDD_1IN/model/B_s123_1IN_91.cc

DEMO_HDRS := \
tensorflow/lite/micro/examples/SZC_motion_1BDD_1IN/model/model.h \
tensorflow/lite/micro/examples/SZC_motion_1BDD_1IN/model/SparkFun_MOTION_Detector_s123_1IN.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/SZC_motion_1BDD_1IN/*/Makefile.inc)

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,SZC_motion_1BDD_1IN,\
$(DEMO_SRCS),$(DEMO_HDRS)))


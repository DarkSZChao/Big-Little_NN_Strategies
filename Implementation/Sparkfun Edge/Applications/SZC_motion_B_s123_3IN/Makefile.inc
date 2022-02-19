DEMO_SRCS := \
tensorflow/lite/micro/examples/SZC_motion_B_s123_3IN/main.cc \
tensorflow/lite/micro/examples/SZC_motion_B_s123_3IN/model/multi_motions_s123.cc

DEMO_HDRS := \
tensorflow/lite/micro/examples/SZC_motion_B_s123_3IN/model/model.h \
tensorflow/lite/micro/examples/SZC_motion_B_s123_3IN/model/10.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/SZC_motion_B_s123_3IN/*/Makefile.inc)

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,SZC_motion_B_s123_3IN,\
$(DEMO_SRCS),$(DEMO_HDRS)))


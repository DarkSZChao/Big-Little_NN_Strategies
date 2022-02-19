DEMO_SRCS := \
tensorflow/lite/micro/examples/SZC_motion_S/main.cc \
tensorflow/lite/micro/examples/SZC_motion_S/model/single_motion.cc

DEMO_HDRS := \
tensorflow/lite/micro/examples/SZC_motion_S/model/model.h \
tensorflow/lite/micro/examples/SZC_motion_S/model/test_data.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/SZC_motion_S/*/Makefile.inc)

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,SZC_motion_S,\
$(DEMO_SRCS),$(DEMO_HDRS)))


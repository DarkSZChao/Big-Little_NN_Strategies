DEMO_SRCS := \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/main.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/single_motion_0.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/single_motion_1.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/single_motion_2.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/single_motion_3.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/single_motion_4.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/single_motion_5.cc \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/multi_motions.cc

DEMO_HDRS := \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/model.h \
tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/model/SparkFun_MOTION_Detector.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/SZC_motion_1B6S_3IN/*/Makefile.inc)

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,SZC_motion_1B6S_3IN,\
$(DEMO_SRCS),$(DEMO_HDRS)))


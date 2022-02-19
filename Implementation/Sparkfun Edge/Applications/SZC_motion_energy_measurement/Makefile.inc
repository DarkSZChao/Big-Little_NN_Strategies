DEMO_SRCS := \
tensorflow/lite/micro/examples/SZC_motion_energy_measurement/main.cc \
tensorflow/lite/micro/examples/SZC_motion_energy_measurement/model/S0_s3.cc

DEMO_HDRS := \
tensorflow/lite/micro/examples/SZC_motion_energy_measurement/model/model.h \
tensorflow/lite/micro/examples/SZC_motion_energy_measurement/model/10.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/SZC_motion_energy_measurement/*/Makefile.inc)

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,SZC_motion_energy_measurement,\
$(DEMO_SRCS),$(DEMO_HDRS)))


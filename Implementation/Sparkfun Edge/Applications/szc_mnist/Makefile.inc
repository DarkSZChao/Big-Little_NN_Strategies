DEMO_SRCS := \
tensorflow/lite/micro/examples/szc_mnist/main.cc \
tensorflow/lite/micro/examples/szc_mnist/model/mnist.cc

DEMO_HDRS := \
tensorflow/lite/micro/examples/szc_mnist/model/model.h \
tensorflow/lite/micro/examples/szc_mnist/model/SparkFun_mnist.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/szc_mnist/*/Makefile.inc)

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,szc_mnist,\
$(DEMO_SRCS),$(DEMO_HDRS)))


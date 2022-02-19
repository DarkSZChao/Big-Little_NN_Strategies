export PROJECTNAME=SZC_motion_NNoM

rm -r tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/obj/tensorflow/lite/micro/examples/${PROJECTNAME}

rm -r tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/${PROJECTNAME}.bin

rm -r tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/main_nonsecure_ota.bin

rm -r tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/main_nonsecure_wire.bin

make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge TAGS="cmsis-nn" ${PROJECTNAME}_bin


python3 tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.2.0/tools/apollo3_scripts/create_cust_image_blob.py \
--bin tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/${PROJECTNAME}.bin \
--load-address 0xC000 \
--magic-num 0xCB \
-o tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/main_nonsecure_ota \
--version 0x0

python3 tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.2.0/tools/apollo3_scripts/create_cust_wireupdate_blob.py \
--load-address 0x20000 \
--bin tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/main_nonsecure_ota.bin \
-i 6 \
-o tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/main_nonsecure_wire \
--options 0x1  


sudo python3 tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.2.0/tools/apollo3_scripts/uart_wired_update.py \
-b 921600 /dev/ttyUSB0 \
-r 1 \
-f tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/main_nonsecure_wire.bin \
-i 6

# model name
export MODELNAME=cifar10


# train the .h5 model
python3 ${MODELNAME}.py


# find input_arrays, output_arrays for tflite converter command
#tflite_convert \
#    --output_file=./Output_Models/${MODELNAME}.dot \
#    --output_format=GRAPHVIZ_DOT \
#    --keras_model_file=./Output_Models/${MODELNAME}.h5
#
#dot -Tpng -O ./Output_Models/${MODELNAME}.dot


# convert .h5 to .tflite, need to confirm input_shapes, input_arrays, output_arrays
tflite_convert \
    --keras_model_file=./Output_Models/${MODELNAME}.h5 \
    --output_file=./Output_Models/${MODELNAME}.tflite \
    --inference_type=QUANTIZED_UINT8 \
    --input_shapes=1,32,32,3 \
    --input_arrays=model_input \
    --output_arrays=model_output/BiasAdd \
    --default_ranges_min=0 --default_ranges_max=255 --mean_values=0 --std_dev_values=1 \
    --change_concat_input_ranges=false \
    --allow_nudging_weights_to_use_fast_gemm_kernel=true \
    --allow_custom_ops


# call tensaiflow compiler to convert .tflite to .cc and add it to application
./tensaiflow_compile \
    --tflite_file ../model_zoo/${MODELNAME}.tflite \
    --out_path ../../../Applications/SZC_cifar10_test/src/ \
    --weights_path ../../../Applications/SZC_cifar10_test/include/

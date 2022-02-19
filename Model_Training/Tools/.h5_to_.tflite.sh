# This shell should be run under Ubuntu only with TF_v1.15


# model name
export MODELNAME=multi_motions_s123


# find input_arrays, output_arrays for tflite converter command
#tflite_convert \
#    --output_file=./${MODELNAME}.dot \
#    --output_format=GRAPHVIZ_DOT \
#    --keras_model_file=./${MODELNAME}.h5

#dot -Tpng -O ./${MODELNAME}.dot


# convert .h5 to .tflite, need to confirm input_shapes, input_arrays, output_arrays # for single input and multiple inputs
tflite_convert \
    --keras_model_file=./${MODELNAME}.h5 \
    --output_file=./${MODELNAME}.tflite \
    --inference_type=QUANTIZED_UINT8 \
    --change_concat_input_ranges=false \
    --allow_nudging_weights_to_use_fast_gemm_kernel=true \
    --allow_custom_ops \
\
    --input_shapes=1,128,9 \
    --input_arrays=input_1 \
    --output_arrays=dense/BiasAdd \
    --default_ranges_min=0 --default_ranges_max=255 --mean_values=128 --std_dev_values=1 
#\
#    --input_shapes=1,128,3:1,128,3:1,128,3 \
#    --input_arrays=model_input1,model_input2,model_input3 \
#    --output_arrays=model_output/BiasAdd \
#    --default_ranges_min=0 --default_ranges_max=255 --mean_values=128,128,128 --std_dev_values=1,1,1
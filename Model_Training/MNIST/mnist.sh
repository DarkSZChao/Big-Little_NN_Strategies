# model name
export MODELNAME=mnist


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
    --input_shapes=1,28,28 \
    --input_arrays=model_input \
    --output_arrays=model_output/BiasAdd \
    --default_ranges_min=0 --default_ranges_max=255 --mean_values=0 --std_dev_values=1 \
    --change_concat_input_ranges=false \
    --allow_nudging_weights_to_use_fast_gemm_kernel=true \
    --allow_custom_ops


# convert .tflite to .cc
xxd -i ./Output_Models/${MODELNAME}.tflite > ./Output_Models/${MODELNAME}.cc


# manualliy reformat the .cc file
python3 ../Tools/.cc_file_modification.py ./Output_Models/${MODELNAME}.cc


# This shell should be run under Ubuntu only
# This shell should be run at: ECM3532\Deployment Tools\


# model name
export MODELNAME=dual_motions
export MARK=D
export TARGETFOLDER=SZC_motion_1B1D_1IN


# call tensaiflow compiler to convert .tflite to .c  and add it to application
./tensaiflow_compile \
    --tflite_file ./${MODELNAME}.tflite \
    --out_path ../../TensaiFlow_rc_alpha2-0.2/Applications/${TARGETFOLDER}/src/ \
    --weights_path ../../TensaiFlow_rc_alpha2-0.2/Applications/${TARGETFOLDER}/include/


# rename the infer.c file name and variable names to guarantee the existence of 2 or more models in one application
python3 .c_var+filename_modification.py -f ../../TensaiFlow_rc_alpha2-0.2/Applications/${TARGETFOLDER}/src/infer.c -s ${MARK}


# rename the weight.bin file name to guarantee the existence of 2 or more models in one application
python3 .bin_filename_modification.py -p ../../TensaiFlow_rc_alpha2-0.2/Applications/${TARGETFOLDER}/include/ -s ${MARK}

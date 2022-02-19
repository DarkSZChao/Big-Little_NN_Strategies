Author: Zichao Shen
Working under TF_1.15

Item List:
1.  .cc_file_modification.py                ---Modify the header of the model.cc converted from model.tflite.
2.  .h_validation_data_generator.py         ---Generate the quantized testing dataset (0, 255) for board Sparkfun Edge.
3.  .h5_summary_read.py                     ---Can read model.h5 in summary.
4.  .h5_to_.pb.py                           ---Convert keras model.h5 to tensorflow model.pb. Can be run in command line: python3 keras_to_tensorflow.py --input_model='./model.h5' --output_model='./model.pb'
5.  .h5_validation.py                       ---Valid the trained model.h5 on computer.
6.  .pb_print_inout_arrays.py               ---Print the input and output name of model.pb which are needed as parameters when using TFconverter in command line.
7   .tflite_summary_read                    ---Can read model.tflite in summary.
8.  .tflite_validation.py                   ---Valid the trained model.tflite on computer.
9.  load_dataset.py                         ---Load data and labels from all datasets.
10. ROOTS_Tools.py                          ---Set which dataset is using.
    
11. Useless for now                         ---Includes .py tool scripts which are not using.
    1.  .h_validation_data_generator_hex.py     ---Generate the quantized testing dataset in hexadecimal for board Sparkfun Edge.
    2.  print_layer_info.py                     ---old script 
    3.  print_pic.py                            ---old script 

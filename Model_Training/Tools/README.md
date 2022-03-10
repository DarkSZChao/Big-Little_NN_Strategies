Author: Zichao Shen
Working under TF_1.15

Item List:
1.  .h_validation_data_generator.py ---Generate the quantized testing dataset for on board testing after deployment
2.  .h5_summary_read.py             ---Can read model.h5 in summary.
3.  .h5_to_.pb.py                   ---Convert keras model.h5 to tensorflow model.pb. Can be run in command line: python3 keras_to_tensorflow.py --input_model='./model.h5' --output_model='./model.pb'
4.  .h5_validation.py               ---Valid the trained model.h5 on computer.
5.  .pb_print_inout_arrays.py       ---Print the input and output name of model.pb which are needed as parameters when using TFconverter in command line.
6   .tflite_summary_read            ---Can read model.tflite in summary.
7.  .tflite_validation.py           ---Valid the trained model.tflite on computer.
8.  load_dataset.py                 ---Load data and labels from all datasets.
9.  ROOTS_Tools.py                  ---Set which dataset is using.
    
10. Useless for now
    1.  .h_validation_data_generator_hex.py     ---Generate the quantized testing dataset in hexadecimal for board Sparkfun Edge.
    2.  print_layer_info.py                     ---old script 
    3.  print_pic.py                            ---old script 
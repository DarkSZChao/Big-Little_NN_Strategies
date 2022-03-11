# Model_Training
Author: Zichao Shen
Working under TF_1.15

This folder only supports:  generation and validation of trained model in h5 format under Windows10
                            generation of trained model in tflite format under Ubuntu20.04
                            validation of trained model in tflite format under Windows10


All the python scripts are using import for pycharm project directory 
except the entrance scripts for each dataset.(using import for both pycharm project directory and Ubuntu non-project directory)


This folder contains:
1.  CIFAR10
2.  MNIST
3.  MOTION_Detector
    1.  Input_Dataset                           ---Original dataset for training and testing on PC.
    2.  Output_Dataset                          ---Quantized dataset for testing on board.
    3.  Output_Models                           ---Trained models in .h5 .tflite.
    4.  Tools
        1.  dataset_changes.py
        2.  dataset_distance.py
        3.  lr_draw.py
    5.  .tflite_BDistance_validation_s123.py    ---Valid Big+Distance system in .tflite on PC.
    6.  .tflite_BDual_validation_s123.py        ---Valid Big+Dual system in .tflite on PC.
    7.  .tflite_BS_validation_s3.py             ---Valid Big+Little system in .tflite using sensor3 only on PC.
    8.  .tflite_BS_validation_s123.py           ---Valid Big+Little system in .tflite using sensor123 on PC.
    9.  .tflite_BS_validation_s123_1IN.py       ---Valid Big+Little system (with Big merged input for NNoM) in .tflite using sensor123 on PC.
    10. B_s3.py                                 ---Train Big using sensor3 only.
    11. B_s123.py                               ---Train Big using sensor123.
    12. B_s123_forNNoM.py                       ---Train Big using sensor123 but merged input.
    13. dual_motions.py                         ---Train Dual.
    14. load_dataset.py                         ---Load data and labels from Input_Dataset folder.
    15. ROOTS.py                                ---Define current directory
    16. S_s3.py                                 ---Train Little using sensor3 only.
    17. S_s3_50_50.py                           ---Train Little using sensor3 only balanced dataset.
    18. S_s3_forNNoM.py                         ---Train Little using sensor3 only adjusted for NNoM.
4.  Tools
    1.  NNoM                                    ---NNoM library files for converting model.h5 to model.h.
        1.  fully_connected_opt_weight_generation.py
        2.  gen_config.py
        3.  nnom.py
    2.  Useless for now
        1.  .h_validation_data_generator_hex.py     ---Generate the quantized testing dataset in hexadecimal for board Sparkfun Edge.
        2.  print_layer_info.py                     ---old script
        3.  print_pic.py                            ---old script
    3.  .h_validation_data_generator.py         ---Generate the quantized testing dataset for on board testing after deployment.
    4.  .h5_draw_graph.py                       ---Draw the network structure for h5 model.
    5.  .h5_load_checkpoint.py                  ---Load checkpoint after training process.
    6.  .h5_summary_read.py                     ---Print model.h5 network summary.
    7.  .h5_to_.h_NNoM.py                       ---Convert keras model.h5 to NNoM model.h.
    8.  .h5_to_.pb.py                           ---Convert keras model.h5 to TF model.pb. Can be run in command line: python3 keras_to_tensorflow.py --input_model='./model.h5' --output_model='./model.pb'
    9.  .h5_to_.tflite.sh                       ---The shell script for converting keras model.h5 to TFLite model.tflite running under Ubuntu20.04.
    10. .h5_validation.py                       ---Valid the trained model.h5 on PC.
    11. .h5_validation_forNNoM.py               ---Valid the trained model.h5 adjusted for NNoM on PC.
    12. .pb_print_inout_arrays.py               ---Print the input and output name of model.pb which are needed as parameters when using TFconverter in command line.
    13. .tflite_summary_read.py                 ---Print model.tflite network summary.
    14. .tflite_validation.py                   ---Valid the trained model.tflite on PC.
    15. .tflite_validation_NNoM.py              ---Valid the trained model.tflite adjusted for NNoM on PC.    
    16. load_dataset.py                         ---Load data and labels from all datasets.
    17. ROOTS_Tools.py                          ---Set which dataset is using.
        

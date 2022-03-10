# Big-Little Adaptive Neural Networks on Low-Power Near-Subthreshold Processors
Authors: Zichao Shen, Neil Howard and Jose Nunez-Yanez 

Abstract: In this paper, we investigate the energy savings that near-subthreshold processors can obtain in Edge AI applications and propose strategies to improve them maintaining the accuracy of the application. The selected processors deploy adaptive voltage scaling techniques in which the frequency and voltage levels of the processor core are determined at run-time. In these systems, embedded RAM and Flash memory size is typically limited to less than 1 Megabyte to save power. This limited memory imposes restrictions on the complexity of the neural networks models that can be mapped to these devices and the required trade-offs between accuracy and battery life. To address these issues, we propose and evaluate alternative ’Big-Little’ neural network strategies to improve battery life while maintaining prediction accuracy. The strategies are applied to a human activity recognition application selected as demonstrator that shows that compared to the original network, the best configurations obtain an energy reduction measured at 80% while maintaining the original level of inference accuracy.  


This project repository contains:
1.  Model_Training      ---contains Python scripts for ‘Big’ and ‘Little’ neural networks training under framework of TensorFlow_v1.15. This folder is extendable, more databases can be added to generate ‘Big’ and ‘Little’ models for future work. (Make sure to maintain the folder structure and subfolder naming)
    1.  CIFAR10                 ---use CIFAR10 database as example.
    2.  MNIST                   ---use MNIST database as example.
    3.  MOTION_Detector         ---use UCI-HAR database as example.
        1.  Input_Dataset                   ---Original dataset for training and testing on computer.
        2.  Output_Dataset                  ---Quantized dataset for testing on board Sparkfun Edge.
        3.  Output_Models                   ---Trained models in .h5 .tflite.
        4.  load_dataset.py                 ---Load data and labels from Input_Dataset folder
        
        
        
        
    4.  Tools                   ---Some utility tools.
        1.  .h_validation_data_generator.py ---Generate the quantized testing dataset for on board testing after deployment
        2.  .h5_summary_read.py             ---Can read model.h5 in summary.
        3.  .h5_to_.pb.py                   ---Convert keras model.h5 to tensorflow model.pb. Can be run in command line: python3 keras_to_tensorflow.py --input_model='./model.h5' --output_model='./model.pb'
        4.  .h5_validation.py               ---Valid the trained model.h5 on computer.
        5.  .pb_print_inout_arrays.py       ---Print the input and output name of model.pb which are needed as parameters when using TFconverter in command line.
        6   .tflite_summary_read            ---Can read model.tflite in summary.
        7.  .tflite_validation.py           ---Valid the trained model.tflite on computer.
        8.  load_dataset.py                 ---Load data and labels from all datasets.
        9.  ROOTS_Tools.py                  ---Set which dataset is using.
            
        10. Useless for now                 ---Includes .py tool scripts which are not using.
            1.  .h_validation_data_generator_hex.py     ---Generate the quantized testing dataset in hexadecimal for board Sparkfun Edge.
            2.  print_layer_info.py                     ---old script 
            3.  print_pic.py                            ---old script 

    
2.  Implementation      ---contains project files for deploying adaptive neural network system on MCUs.
    1.  Apollo2 Blue EVB        ---deployment files for Apollo2 Blue EVB.
    2.  ECM3532                 ---deployment files for ECM3532.
    3.  Sparkfun Edge           ---deployment files for Sparkfun Edge.
    4.  STM32L4R5ZI-P           ---deployment files for STM32L4R5ZI-P.






Used Resources:
    1. UCI-HAR Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones



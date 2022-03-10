# Big-Little Adaptive Neural Networks on Low-Power Near-Subthreshold Processors
Authors: Zichao Shen, Neil Howard and Jose Nunez-Yanez 

Abstract: In this paper, we investigate the energy savings that near-subthreshold processors can obtain in Edge AI applications and propose strategies to improve them maintaining the accuracy of the application. The selected processors deploy adaptive voltage scaling techniques in which the frequency and voltage levels of the processor core are determined at run-time. In these systems, embedded RAM and Flash memory size is typically limited to less than 1 Megabyte to save power. This limited memory imposes restrictions on the complexity of the neural networks models that can be mapped to these devices and the required trade-offs between accuracy and battery life. To address these issues, we propose and evaluate alternative ’Big-Little’ neural network strategies to improve battery life while maintaining prediction accuracy. The strategies are applied to a human activity recognition application selected as demonstrator that shows that compared to the original network, the best configurations obtain an energy reduction measured at 80% while maintaining the original level of inference accuracy.  

-------------------------------------------------------------------------------
This project repository contains:
1.  Model_Training      ---contains Python scripts for ‘Big’ and ‘Little’ neural networks training under framework of TensorFlow_v1.15. This folder is extendable, more databases can be added to generate ‘Big’ and ‘Little’ models for future work. (Make sure to maintain the folder structure and subfolder naming)
    1.  CIFAR10
    2.  MNIST
    3.  MOTION_Detector     ---use UCI-HAR database
    4.  Tools

2.  Implementation      ---contains project files for deploying adaptive neural network system on MCUs.
    1.  Apollo2 Blue EVB
    2.  ECM3532
    3.  Sparkfun Edge
    4.  STM32L4R5ZI-P


Note: The model training process using Model_Training can be done under Window 10 OS, or Ubuntu 20.04 OS if prefer. The deployment for SparkFun and ECM3532 can only be done under Ubuntu 20.04 OS. The deployment for Apollo2 and STM32L4R5ZI can be done by using Keil IDE which only works under Window 10 OS.

Get start with ‘Big’ and ‘Little’ networks training by using Model_Training folder and setting this directory as root in PyCharm (recommended IDE). Then the UCI_HAR database can be found in MOTION_Detector. Run B_s123.py to train ‘Big’ CNN model with 3 sensor input branches. Run S_s3.py to train ‘Little’ model by only using third sensor as input. 
After the training process, h5 files can be found under Output_Models. The h5 model can be converted into quantized tflite format by using .h5_to_.tflite.sh under Tools. 
We use Sparkfun Edge as example here for deployment. Move the tflite models into Deployment Tools and convert them into C code by xxd command. Then manually edit the header in the C model or run the .cc_header_modification.py to match the variables called in main.cc in project. Then the rest steps can be found here: https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow#3 



Used Resources:
    1. UCI-HAR Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones



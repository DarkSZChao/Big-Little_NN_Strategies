# Big-Little Adaptive Neural Networks on Low-Power Near-Subthreshold Processors
 
Authors: Zichao Shen, Neil Howard and Jose Nunez-Yanez 

Abstract: In this paper, we investigate the energy savings that near-subthreshold processors can obtain in Edge AI applications and propose strategies to improve them maintaining the accuracy of the application. The selected processors deploy adaptive voltage scaling techniques in which the frequency and voltage levels of the processor core are determined at run-time. In these systems, embedded RAM and Flash memory size is typically limited to less than 1 Megabyte to save power. This limited memory imposes restrictions on the complexity of the neural networks models that can be mapped to these devices and the required trade-offs between accuracy and battery life. To address these issues, we propose and evaluate alternative ’Big-Little’ neural network strategies to improve battery life while maintaining prediction accuracy. The strategies are applied to a human activity recognition application selected as demonstrator that shows that compared to the original network, the best configurations obtain an energy reduction measured at 80% while maintaining the original level of inference accuracy.  



This project repository contains two main folders:
1.  Model_Training folder   ---contains Python scripts for ‘Big’ and ‘Little’ neural networks training under framework of TensorFlow_v1.15.
2.  Implementation folder   ---contains project files for deploying adaptive neural network system on 4 MCUs. Vendor SDK toolchains and compilers are not included.























Used Resources:
    1. UCI-HAR Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones



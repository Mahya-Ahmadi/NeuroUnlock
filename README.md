# NeuroUnlock

This repository contains the source code of the scripts used for the paper [NeuroUnlock: Unlocking the Architecture of Obfuscated Deep Neural Networks @ IJCNN2022](https://doi.org/10.1109/IJCNN55064.2022.9892545).

# Main concept

The main idea behind NeuroUnlock is that we use [NeurObfuscator](https://github.com/zlijingtao/Neurobfuscator) to obfuscate the neural networks on a specific GPU, but with enough profiling data from the GPU itself we can negate the effects of the different obfuscation knobs created by NeurObfuscator. This happens because NeruObfuscator follows a specific reward function to achieve optimal performance while introducting delays in the network execution. Therefore, with enough profiling and obfuscation data it is enough to reconstruct the original network structure.

# Toolflow

1. Install and run NeurObfuscator docker, which can be found in their GitHub page.
2. Generate the dataset with the random networks through NeurObfuscator. This can be achieved by running the routing for generating random networks for profiling reasons, as they are later used to create a model of the GPU for the Side-channel attack with DeepSniffer. We suggest generating between 1000-5000 networks.
3. Fix the model naming of the random networks using ``neurobfuscator_label_maker.py``, more details in the help of the script. Additionally, this script also fixes the labels for the models to be run in NeurObfuscator.
4. Run all the networks for obfuscation through NeurObfuscator, many might not actually work as they were randomly generated.
    1. This is the most time intensive part, on RTX 2080 Ti this took slightly less than a month of continuous execution. Additionally, it is suggested to reduce the number of offspring and generations to be run in NeurObfuscator, otherwise it might lead to a runtime of several days per network, rathan than 6-8 hours per each.
5. Now it is time to parse all the output from NeurObfuscator, as it generates a log file per model and an overall execution log. They can be parsed using ``neurobfuscator_ga_analysis_log_parser.py`` and ``neurobfuscator_performance_log_parser.py``.
6. Using [openNMT](https://opennmt.net/) we can now train a network on the predictions after the obfuscation with NeurObfuscator, which have been parsed in the previous step.
    1. An example of the used model is available in openNMT format in ``seq-obfuscator-github.yaml``
7. After training and testing on the dataset splits, we can compute the metrics using ``LER_v2.py``. We still need to test the predicted models using DeepSniffer.
7. First, we have to adapt the models for running with DeepSniffer, which is done through ``neurobfuscator_fix_random_nns.py``.
8. Now it is time to run [DeepSniffer](https://github.com/xinghu7788/DeepSniffer), to train the obfuscated and de-obfuscated models, together with the original one, and compare the results of the adversarial attacks. An example script for training/testing is ``train_validate_test.py``
    1. Main change is that the dataset to run the model on must be prepared by splitting it according to the image label, e.g., CIFAR10 creates 10 ``class_{label}.pth`` which are just lists with all the images of the corresponding label. The script is ``cifar10_data_formatter.py``
9. The final results of the DeepSniffer runs are then used to plot the results available in the paper.


# Citation
Use the following Bibtex citation to properly cite the paper.
```
@INPROCEEDINGS{NeuroUnlock_Morid_Ahmadi_ijcnn2022,
  author={Ahmadi, Mahya Morid and Alrahis, Lilas and Colucci, Alessio and Sinanoglu, Ozgur and Shafique, Muhammad},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  title={NeuroUnlock: Unlocking the Architecture of Obfuscated Deep Neural Networks},
  year={2022},
  volume={},
  number={},
  pages={01-10},
  keywords={Deep learning;Training;Machine learning algorithms;Neural networks;Graphics processing units;Intellectual property;Prediction algorithms;Side-channel-based attacks;Deep neural networks;Architecture;Obfuscation;Model extraction},
  doi={10.1109/IJCNN55064.2022.9892545}}
```

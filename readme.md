# Needle-Machine-Roll
This network predicts the roll angle of the needle tip given 5DOF sensor measurements and actuation input.
This uses the same methods as presented in our ISER2020 paper: 

[A Recurrent Neural Network Approach to
Roll Estimation for Needle Steering](https://research.vuse.vanderbilt.edu/MEDLab/sites/default/files/papers/Emerson2020_ISER.pdf).

This repo is meant to be a basic demonstration of Data Prep, Network Training & Eval for this supervised learning problem.

## Data Prep
The sequences consist of input vectors ```X``` that contain the encoded 5DOF measurement and actuation.
The network is tasked with predicting a target vector ```Y``` that contains the encoded roll-angle missing from the tip state information.

The prepared dataset used for ISER2020 Experiments is contained in the ```data/``` folder. If you wish to modify the dataset input/output vectors, you can do so in ```prepare300ptDataset.m```.

![dataset-sequence](resources/dataset-sequence.png)

Another way to interpret the problem, is that the network learns to predict the difference angle between the actuator and the tip, as a function of insertion length. The network learns when actuation is transmitted to the tip, and when the actuation is within some "deadband".

![deadband](resources/diff-ang.png)

## Training Network & Evaluation
This code runs in MATLAB with help from the Deep Learning Toolbox. 

The script ```trainAndEvaluate.m``` will load the dataset, partition it into 'training', 'validation', and 'test' datasets.
The 'training' dataset is then partitioned again ```k```-times to train an ensemble of ```k``` component networks. Because each network is trained on a different subset of the 'training' data, their predictions vary slightly from one another. This can be used to detect 'out-of-distribution' samples and encode a 'confidence' in the mean estimate.

![ensemble](resources/ensemble.png)

*Note:* 
The above is all offline analysis. To deploy this network online (under closed-loop control), the model should be deployed in C++. This can be done with Matlab Coder and the functionality can be implemented within a C++ class. Note that the network can be trained to predict at time ```t``` or for time ```t+1```, depending on how the observer is implemented in the control loop.

## Citing
If you use this in your research, please use the following BibTeX entry.
```shell
@inproceedings{Emerson2020,
    title = {{A Recurrent Neural Network Approach to Roll Estimation for Needle Steering}},
    year = {2021},
    booktitle = {International Symposium on Experimental Robotics},
    author = {Emerson, Maxwell and Ferguson, James M. and Ertop, Tayfun Efe and Rox, Margaret and Granna, Josephine and Lester, Michael and Maldonado, Fabien and Gillaspie, Erin A. and Alterovitz, Ron and III., Robert J. Webster and Kuntz, Alan},
    pages = {1--8},
    url = {http://arxiv.org/abs/2101.04856},
    arxivId = {2101.04856}
}
```
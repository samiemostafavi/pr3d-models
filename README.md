# pr3d-models


This repository contains some examples on how to use the models trained using PR3D project. The code is standalone and only requires json files (distribution parameters) to work.

Run the following example scripts and check how they work.

In the examples, we open the trained model from a json file and sample it. 
Then we draw the CCDF of the samples.
The models are trained with 80K uplink delay measurements taken from a private 5G network. They correspond to Figure 5 of our paper: [Data-Driven Latency Probability Prediction for Wireless Networks: Focusing on Tail Probabilities](https://arxiv.org/abs/2307.10648)

Run the Gaussian mixture model example:
```
python example_gmm.py
```

Run the Gaussian mixture model + Pareto distribution example:
```
python example_gmevm.py
```

They create PNG files in the root folder of the repository.
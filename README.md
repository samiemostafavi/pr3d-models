# pr3d-models


This repository contains some examples on how to use the models trained using PR3D project. The code is standalone and only requires json files (distribution parameters) to work.

Run the following example scripts and check how they work.

In the examples, we open a delay model from a json file and sample it. Then we draw the CCDF of the samples

Run the Gaussian mixture model example:
```
python example_gmm.py
```

Run the Gaussian mixture model + Pareto distribution example:
```
python example_gmevm.py
```

They create PNG files in the root folder of the repository.
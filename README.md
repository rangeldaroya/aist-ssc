# Setup Instructions
1. Activate conda environment (Note this repo uses Python 3.8.13)
2. Install requirements using `pip install -r requirements.txt`
3. Install graphviz for some visualizations in notebooks
    a. On MacOS: `brew install graphviz`
    b. On Ubuntu: `sudo apt-get install graphviz`

4. Install lightgmb. See [this link](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)
    a. For MacOS with M1 chip, do the following:
```
conda install lightgbm
pip install lightgbm
```

5. Install hyperopt for hyperparam search for sklearn in [this link](https://hyperopt.github.io/hyperopt-sklearn/)
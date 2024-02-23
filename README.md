# RAMP starting-kit on party membership 

Party Membership Prediction: Leveraging data from French deputies, our goal is to develop a model that accurately predicts the political party membership of french deputies.

Authors : Lambros Michalopoulos, Vivek Ashokan, Hugo Pointier, Romain Poupon, Nathan Maligeay 

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

## Download data

The required data can be downloaded locally :

```
python3 download_data.py
```
Source: [data.europa.eu](https://data.europa.eu/data/datasets/5fc8b732d30fbf1ed6648aab~~1?locale=fr)

## Preparing data

Before utilizing the data, it needs to undergo processing:

```
python3 prepare_data.py
```

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
# Discord Aware Matrix Profile (DAMP)

Authors:
* Siho Han ([@sihohan](https://github.com/sihohan/DAMP))
* Jihwan Min ([@rtm-jihwan-min](https://github.com/rtm-jihwan-min))
* Taeyeong Heo ([@htyvv](https://github.com/htyvv))
* JuI Ma ([@iju298](https://github.com/iju298))

This repository contains an unofficial Python implementation of Discord Aware Matrix Profile (DAMP), introduced in ["Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions of Datapoints and Ultra-fast Arriving Data Streams" (KDD '22)](https://dl.acm.org/doi/abs/10.1145/3534678.3539271). The official MATLAB implementation can be found [here](https://sites.google.com/view/discord-aware-matrix-profile/documentation).

## Project Organization

    ├── data
    |   └── samples
    |       └── BourkeStreetMall.txt
    ├── .gitignore
    ├── README.md
    ├── damp.py
    └── utils.py

## Requirements

* Python >= 3.6
* matplotlib
* numpy

## Datasets

This repository includes Bourke Street Mall as the default dataset (see the `data` directory), which can be downloaded [here](https://sites.google.com/view/discord-aware-matrix-profile/documentation).

## Run

You can run the code using the following command.
```
python damp.py
```

With `--enable_output`, the resulting plot and DAMP values will be saved in the `./figures` and `./outputs` directories, respectively.

Note that the input time series and its corresponding DAMP scores on the plot are scaled for visualization purposes.

## References

* Lu, Yue, et al. "Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions of Datapoints and Ultra-fast Arriving Data Streams." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.
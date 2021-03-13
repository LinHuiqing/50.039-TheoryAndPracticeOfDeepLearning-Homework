# Week 01: Discriminative ML - Quick Intro

## Setup

1. Ensure that the python version you are using is `>=3.6`.
2. Ensure that you have the followng packages installed:
    - numpy
    - pandas
    - scikit-learn
3. Ensure that the project directory has the following structure:

    ``` utf-8
    imagefeatures/
        {0A3DA3A6-1BED-4AA0-9F27-9A643FC9F617}.jpg_ft.npy
        ...
    dataset_splits/
        train.npy
        val.npy
        test.npy
    gtlabels.txt
    label_info.txt
    W1_HW.ipynb
    ```

## Loading dataset

1. Ensure that are in the project directory.
2. Run use the following line to load the numpy file.

    ``` python
    np.load("<path to .npy file>", allow_pickle=True)
    ```

    The resulting numpy array obtained would have shape (`m`, 3), where `m` is the number of examples. Each row contains the following:

    | filename | Spring | Summer | Autumn | features                        |
    | -------- | ------ | ------ | ------ | ------------------------------- |
    | str      | int    | int    | int    | numpy array of image's features |

## Homework Responses

Responses to homework can be found in `DL_HW_W1.pdf`.

# Attnetion Concept Learning



## Requirements

*   numpy>=1.15.0
*   Pillow>=6.1.0
*   torch==1.2.0
*   torchvision==0.2.2

## Setup

1.  Install PyTorch and other required python libraries with:

    ```
    ./config.sh
    ```

2.  Download CIFAR-10-C and CIFAR-100-C datasets with:

    ```
    mkdir -p ./data/cifar
    curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
    curl -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
    tar -xvf CIFAR-100-C.tar -C data/cifar/
    tar -xvf CIFAR-10-C.tar -C data/cifar/
    ```


## Usage


 `python cifar_attend.py --no-augmix --batch-size 96`
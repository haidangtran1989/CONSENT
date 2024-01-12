# CONSENT: Conversational Search with Tail Entities

## Setup

Please download and unzip [``resource.zip``](https://nextcloud.mpi-klsb.mpg.de/index.php/s/FF8bZgtkzLF4JYS) file to the ``data`` folder.
In the ``resource``, please find ``news-en-documents-2017-20181120.tsv.gz`` and ``wiki-en-documents-20170920.tsv.gz``.
These files contain articles of Wikipedia and news, where the text content and the respective IDs are available.
Please build the BM25 index of this collection, and store such index in the ``data/resource/news-wiki-en-index`` folder.
The structure of the CONSENT directory should look like this:

```.
├── data
│   ├── judges
│   ├── result
│   ├── resource
│   │   ├── news-wiki-en-index
│   │   ├── Box4Types
│   │   └── ...
│   ├── conv_test_set.txt
│   ├── conv_train_set.txt
│   └── ...
└── source
```

## Usage
The run files of main method `CONSENT` are stored in `data/result/consent_test`, please run the following scripts to reproduce the experimental result for `CONSENT` and base method:

    cd source/
    ./run_exp.sh

To generate the benchmark for tail entities, please run:

    cd source/
    ./conv_generator.sh

In addition, please execute the following commands if you want to re-run the main method on test set:

    cd source/
    ./consent.sh 0 224 test

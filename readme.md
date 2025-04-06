# GenAI for Software Development Assignment 1

Benjamin Tremblay, Rowan Miller

- [1 Introduction](#1-introduction)
- [2 Setup](#2-setup)
- [3 Run Model](#3-run-model)
- [4 Report](#4-report)

## **1. Introduction**

We have fine-tuned the T5 Transformer model so that, when given a Python method with a masked if statement, our model will predict the hidden if statement. The code uses the HuggingFace T5 Transformer model maker. To train the model, we mask the if statements for training and validation datasets. We then used the RobertaTokenizer to tokenize the datasets, adding the <MASK> token to take the place of the if statements and then resizing the model. After training the model, we used the train dataset (after masking) to evaluate using multiple different metrics: BLEU4, BLEU score, and exact match. 

## **2. Setup**

This project is implemented in **Python 3** and is compatible with **macOS, Linux, and Windows**.

Clone the repository to your workspace:

```shell
~ $ git clone https://github.com/bentrem38/If-Predicter
```

Navigate into the repository:

```shell
~ $ cd If-Predicter
~/n-grams $
```

Set up a virtual environment and activate it (optional):

For macOS/Linux:

```shell
~/If-Predicter $ python -m venv ./venv/
~/If-Predicters $ source venv/bin/activate
(venv) ~/If-Predicter $
```

When you're finished, use the following command to deactivate the virtual environment:
`(venv) $ deactivate`

## **4. Run Model**

To finetune, evaluate, test, and evaluate the transformer model, simply run the following:

`python If-Predicter.py`

The file `testset-results.csv` will display the results of the model's evaluation on selected tests in the `train.csv` folder.

## 5. Report

Our overall report is available in the file Assignment_Report.pdf.

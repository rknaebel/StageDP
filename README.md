# Two-stage Discourse Parser

Here is a refactoring of the implementation of the RST discourse parser described in [A Two-stage Parsing Method for Text-level Discourse Analysis](http://aclanthology.coli.uni-saarland.de/pdf/P/P17/P17-2029.pdf). 
Due to the licence of RST data corpus, the training data is not included in our project folder. 
To reproduce the result in the paper, download it from the LDC, preprocess the data as stated below.

### Usage:

1. Preprocess the data:
    
    ```
    python3 preprocess.py RST_DATA_DIR RST_DEST_DIR
    ```

2. Train model:
    ```
    python3 main.py --train --train_dir TRAIN_DIR
    ```
    
3. Evaluate model:
    ```
    python3 main.py --eval --eval_dir EVAL_DIR
    ```

### Requirements:

Currently runs under Python 3.7.
The models are rewritten in sklearn. See requirements.txt for more details.

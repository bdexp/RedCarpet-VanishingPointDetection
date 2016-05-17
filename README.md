# RedCarpet Lane Detection

An implementation of a convolutional neural network to predict vanishing points in images from scaled cars.

### Requirements

- Python 2.7
- Virtualenv 15.0.1

### Setup (For Ubuntu/Linux 64-bit)

1. Create a virtual environment: virtualenv env
2. Activate virtual environment: source env/bin/activate
3. Install tensorflow (CPU only mode): pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
4. Install remaining libraries: pip install -r requirements.txt

### Configuration

**Data Settings**
- DATA_PATH (imageRead.py, String): Path to datasets.
- DATA_FOLDERS (imageRead.py, List[String]): List of folders containing data and labels to be used for training.
- TRAINING_RATIO (main.py, 0.0 - 1.0): How much of the dataset should be used in training.
- DISTORTION_RATE (main.py, 0.0 - 1.0): How much of the dataset should be distorted and added to the training set.
- ADD_FLIPPED (main.py, True/False): Add mirrored images to the training set.

**Execution Settings**
- LOAD_MODEL (main.py, True/False): Load existing checkpointfile in /models before training.
- TRAIN_MODEL (main.py, True/False): Execute training step.
- FREEZE_GRAPH (main.py, True/False): Freeze checkpoint and graphdef file to a freezed graph in /models.

**Training Parameters**
- BATCH_SIZE (main.py, Int): The number of training examples in one training step.
- STEP_SIZE_MAX (main.py, Int): The total amount of training steps.
- STEP_SIZE_PRINT (main.py, Int): Every step to print the test set accuracy.
- STEP_SIZE_SAVE (main.py, Int): Every step to save the model to a checkpoint file.

### Running

1. Execute main file: python main.py
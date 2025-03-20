import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  
LOG_FILE = os.path.join(CURRENT_DIR, "train_log.txt")

def clear_log():
    """
    Clear the training log file by overwriting it with a header.

    Parameters:
    None

    Returns:
    None
    """
    with open(LOG_FILE, "w") as f:
        f.write("=== Training Log ===")

def write_log(text):
    """
    Append a line of text to the training log file.

    Parameters:
    text : str
        The text to be written to the log file.

    Returns:
    None
    """
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")

def read_log():
    """
    Read and print the contents of the training log file.

    Parameters:
    None

    Returns:
    None
    """
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                print(line.strip())
    else:
        print("No training log found!")

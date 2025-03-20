import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  
LOG_FILE = os.path.join(CURRENT_DIR, "train_log.txt")

def clear_log():
    with open(LOG_FILE, "w") as f:
        f.write("=== Training Log ===")

def write_log(text):
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")

def read_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                print(line.strip())
    else:
        print("No training log found!")

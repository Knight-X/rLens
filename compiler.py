import subprocess
import numpy as np
import time
def compile():
    subprocess.call(['gcc', '-o', 'convba', 'convba.s'])
    total = 0.0
    for i in range(10):
        end_time = subprocess.check_output(['./convba'])
        total = total + float(end_time)
    total = float(total) / 10.0
    return total 

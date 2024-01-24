import torch
import random
import os, sys
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class Logger(object):
    def __init__(self, file_name='correl_re.log', log=True, stream=sys.stdout) -> None:
        self.terminal = stream
        self.log = log
        if self.log:
            log_dir = "./outlog"
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.log_file = open(f'outlog/{file_name}', "a")
            self.flush()

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log_file.write(message)

    def flush(self):
        self.log_file.seek(0)	# 定位
        self.log_file.truncate()


import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter
import logging
from utils import processed_dir


def met_dict_to_str(md, prefix="", verbose=1):
    """Convert metric dictionary to string for printing."""
    ret_str = prefix
    for k, v in md.items():
        if k == "loss":
            ret_str += k + ": " + "%.5f" % v + " | "
        else:
            ret_str += k + ": " + "%.3f" % v + " | "
    if verbose > 0:
        print("\x1b[40m\x1b[37m" + ret_str[:-1] + "\x1b[0m")

    return ret_str


def met_dict_to_writer(md, step, writer, prefix):
    """Given a dict of eval metrics, write to given Tensorboard writer."""
    for k, v in md.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def print_seperator(strings: list, max_len: int):
    """Print text inside a one-line string with "=" seperation to a max length.

    Args:
        strings (list): List of strings.
        max_len (int): Max length.
    """
    midpoints = int(max_len / len(strings))
    strings = [str(i) for i in strings]
    final_str = ""
    cutoff = max_len + (9 * len(strings))
    for s in strings:
        if "\x1b" in s:
            cutoff += 9
        len_s = len(s.replace("\x1b[32m", "").replace("\x1b[39m", ""))
        final_str += "\x1b[40m"
        final_str += "=" * (int((midpoints / 2) - int(len_s / 2)) - 1)
        final_str += f" {s} "
        final_str += "=" * (int((midpoints / 2) - int(len_s / 2)) - 1)
        final_str += "\x1b[0m"
    print(final_str[:cutoff])
    return final_str[:cutoff]


class LogWriter:
    """Writer class for logging PyTorch model performance."""

    def __init__(
            self,
            model,
            args,
            path
    ):
        """Init writer.

        Args:
            model: Pytorch model.
            path (str): Path to save log files.
        """
        self._model = model
        self._best_val_acc = 0
        self._best_val_f1 = 0
        self._patience = 0
        self._max_patience = args.max_patience
        self._epoch = 0
        self._step = 0
        self._path = Path(path)
        self._writer = SummaryWriter(path)
        self._filer = logging.getLogger(__name__)
        self._log_every = args.log_every if hasattr(args, 'log_every') else 10
        self._val_every = args.val_every if hasattr(args, 'log_every') else args.save_steps
        self.save_attrs = ["_best_val_acc", "_best_val_f1", "_patience", "_epoch", "_step"]

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            filename=self._path / ('train.log' if args.do_train else 'test.log'),
                            filemode='w',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

    def info(self, msg):
        self._filer.info(msg)

    def log(self, train_mets, val_mets):
        """Log information."""
        if not self.log_val() and (self._step + 1) % self._log_every != 0:
            self.step()
            return

        if not self.log_val():
            ret = met_dict_to_str(train_mets, "TR = ")
            self.info(ret)
            met_dict_to_writer(train_mets, self._step, self._writer, "TRN")
            self.step()
            return

        val_f1 = val_mets["f1"]
        if val_f1 > self._best_val_f1:
            self._best_val_f1 = val_f1
            with open(self._path / "best_f1.model", "wb") as f:
                torch.save(self._model.state_dict(), f)
            best_model_string = "Best model(f1) saved: %.3f" % val_f1
            best_model_string = f"\x1b[32m{best_model_string}\x1b[39m"
            self._patience = 0
        else:
            self._patience += 1
            best_model_string = "No improvement."
        ret = print_seperator(
            [
                f"Patience: {self._patience:03d}",
                f"Epoch: {self._epoch:03d}",
                f"Step: {self._step:03d}",
                best_model_string,
            ],
            131,
        )
        self.info(ret)
        ret = met_dict_to_str(train_mets, "TR = ")
        self.info(ret)
        met_dict_to_writer(train_mets, self._step, self._writer, "TRN")
        ret = met_dict_to_str(val_mets, "VA = ")
        self.info(ret)
        met_dict_to_writer(val_mets, self._step, self._writer, "VAL")
        self.step()

    def test(self, test_mets):
        """Helper function to write test mets."""
        print_seperator(["\x1b[36mTest Set\x1b[39m"], 135)
        ret = met_dict_to_str(test_mets, "TS = ")
        self.info(ret)

    def log_val(self):
        """Check whether should validate or not."""
        if (self._step + 1) % self._val_every == 0:
            return True
        return False

    def step(self):
        """Increment step."""
        self._step += 1

    def epoch(self):
        """Increment epoch."""
        self._epoch += 1

    def stop(self):
        """Check if should stop training."""
        return self._patience > self._max_patience

    def load_best_model(self, rule='f1'):
        """Load best model."""
        torch.cuda.empty_cache()
        self._model.load_state_dict(torch.load(self._path / f"best_{rule}.model"))

    def save_logger(self):
        """Save class attributes."""
        with open(self._path / "log.pkl", "wb") as f:
            f.write(pkl.dumps(dict([(i, getattr(self, i)) for i in self.save_attrs])))
        with open(self._path / "current.model", "wb") as f:
            torch.save(self._model.state_dict(), f)

    def load_logger(self):
        """Load class attributes."""
        with open(self._path / "log.pkl", "rb") as f:
            attrs = pkl.load(f)
            for k, v in attrs.items():
                setattr(self, k, v)
        torch.cuda.empty_cache()
        self._model.load_state_dict(torch.load(self._path / "current.model"))

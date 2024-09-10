###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import os
import sys
import time
import logging
import numpy as np

from docplex.mp.progress import ProgressListener, ProgressClock, SolutionListener

def append_to_log(log_path, msg):
    with open(log_path, "a") as f:
        f.write(msg)
        
def write_to_log(log_path, msg):
    with open(log_path, "w") as f:
        f.write(msg)

class BestBoundAborter(ProgressListener):
    """Custom aborter to stop when finding a feasible solution matching the bound.
    see: https://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/progress.html#ProgressClock
    https://dataplatform.cloud.ibm.com/exchange/public/entry/view/6e2bffa5869dacbae6500c7037ecd36f
    
    usage:
    from docplex.mp.model_reader import ModelReader
    mdl = ModelReader.read(path, ignore_names=False)
    log_path = "path_to_log_results.log"
    log_file_obj = open(log_path, "w")
    mdl.add_progress_listener(
            BestBoundAborter(max_best_bound=stops_at_value, log_file_obj=log_file_obj)
        )
    """

    def __init__(
        self, max_best_bound: float = 0, minimize: bool = False, log_file_obj=None
    ):
        super(BestBoundAborter, self).__init__(ProgressClock.BestBound)
        self.max_best_bound = max_best_bound
        self.last_obj = None
        self.minimize = minimize
        self.log_file_obj = log_file_obj

    def notify_start(self):
        super(BestBoundAborter, self).notify_start()
        self.last_obj = None

    def stopping_condition(self):
        if self.minimize:
            return self.last_obj <= self.max_best_bound
        else:
            return self.last_obj >= self.max_best_bound

    def notify_progress(self, pdata):
        super(BestBoundAborter, self).notify_progress(pdata)
        if pdata.has_incumbent:
            self.last_obj = pdata.current_objective
            if self.stopping_condition():
                msg = f"_____ FOUND Feasible solution {self.last_obj} better than stopping condition {self.max_best_bound}\n"
                if self.log_file_obj:
                    self.log_file_obj.write(msg)
                else:
                    print(msg)
                self.abort()

        
class BestFeasibleLogger(ProgressListener):
    """
    Custom logger of time and new best feasible solution in a separate file
    see: https://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/progress.html#ProgressClock
    https://dataplatform.cloud.ibm.com/exchange/public/entry/view/6e2bffa5869dacbae6500c7037ecd36f
    log format csv: time,feasible_solution,best_bound
    
    from docplex.mp.model_reader import ModelReader
    mdl = ModelReader.read(path, ignore_names=False)
    log_path = "path_to_log_results.log"
    mdl.add_progress_listener(
            BestFeasibleLogger(log_path=log_path)
        )
    """

    def __init__(
        self, log_path=None
    ):
        super().__init__(ProgressClock.Gap)
        self.last_obj = None
        self.log_path = log_path
        if self.log_path:
            write_to_log(log_path, "time,feasible_solution,best_bound\n")

    def notify_start(self):
        super().notify_start()
        self.last_obj = None
       
    def is_improving(self, new_obj, eps=1e-4):
        last_obj = self.last_obj
        return last_obj is None or (abs(new_obj- last_obj) >= eps)

    def notify_progress(self, pdata):
        super().notify_progress(pdata) 
        if pdata.has_incumbent and self.is_improving(pdata.current_objective):
            self.last_obj = pdata.current_objective
            msg = f"{pdata.time},{self.last_obj},{pdata.best_bound}\n"
            if self.log_path:
                append_to_log(self.log_path, msg)
            else:
                print(msg)
                

class FirstNodeAborter(ProgressListener):
    """
    Abort at the first node after the root relaxation
    see: https://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/progress.html#ProgressClock
    https://dataplatform.cloud.ibm.com/exchange/public/entry/view/6e2bffa5869dacbae6500c7037ecd36f
    log format csv: time,feasible_solution,best_bound
    
    from docplex.mp.model_reader import ModelReader
    mdl = ModelReader.read(path, ignore_names=False)
    log_path = "path_to_log_results.log"
    mdl.add_progress_listener(
            BestFeasibleLogger(log_path=log_path)
        )
    """

    def __init__(
        self
    ):
        super().__init__(ProgressClock.Gap)
        self.last_obj = None

    def notify_start(self):
        super().notify_start()
        self.last_obj = None
       
    def is_improving(self, new_obj, eps=1e-4):
        last_obj = self.last_obj
        return last_obj is None or (abs(new_obj- last_obj) >= eps)

    def notify_progress(self, pdata):
        super().notify_progress(pdata) 
        if pdata.has_incumbent and self.is_improving(pdata.current_objective):
            self.abort()
from __future__ import absolute_import, print_function
import os
import sys
import time

import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from platoon.channel import Controller


class WaveNetController(Controller):
    def __init__(self, max_mb,saveFreq,default_args):
        """
        Initialize the WaveNetController
        Parameters
        ----------
        max_mb : int
            Max number of minibatches to train on.
        patience: : int
            Training stops when this many minibatches have been trained on
            without any reported improvement.
        valid_freq : int
            Number of minibatches to train on between every monitoring step.
        default_args : dict
            Arguments of default class Controller
        """

        super(WaveNetController, self).__init__(**default_args)
        self.max_mb = int(max_mb)

        self.uidx = 0
        self.eidx = 0
        self.saveFreq = saveFreq

        self._save_params = False
        self.start_time = None
        self._should_stop = False

    def handle_control(self, req, worker_id, req_info):
        """
        Handles a control_request received from a worker
        Parameters
        ----------
        req : str or dict
            Control request received from a worker.
            The control request can be one of the following
            1) "next" : request by a worker to be informed of its next action
               to perform. The answers from the server can be 'train' (the
               worker should keep training on its training data), 'valid' (the
               worker should perform monitoring on its validation set and test
               set) or 'stop' (the worker should stop training).
            2) dict of format {"done":N} : used by a worker to inform the
                server that is has performed N more training iterations and
                synced its parameters. The server will respond 'stop' if the
                maximum number of training minibatches has been reached.
            3) dict of format {"valid_err":x, "test_err":x2} : used by a worker
                to inform the server that it has performed a monitoring step
                and obtained the included errors on the monitoring datasets.
                The server will respond "best" if this is the best reported
                validation error so far, otherwise it will respond 'stop' if
                the patience has been exceeded.
        """
        control_response = ""

        if req == 'next':
            if not self._should_stop:
                if self.start_time is None:
                    self.start_time = time.time()
                if self._save_params:
                    control_response = 'save'
                else:
                    control_response = 'train'
            else:
                control_response = 'stop'
        elif req == 'done':
            self.uidx += req_info['train_len']
            if self.uidx%self.saveFreq==0:
                self._save_params=True

        elif req == 'saved':
            self._save_params=False

        if self.uidx > self.max_mb:
            if not self._should_stop:
                print("Training time {:.4f}s".format(time.time() - self.start_time))
                print("Number of samples:", self.uidx)
            ##NEVER STOPPING!
            self._should_stop = False

        return control_response


def wavenet_control(saveFreq=1110, saveto=None):
    parser = Controller.default_parser()
    parser.add_argument('--max-mb', default=((5000 * 1998) / 10), type=int,
                        required=False, help='Maximum mini-batches to train upon in total.')

    args = parser.parse_args()

    l = WaveNetController(max_mb=10000,saveFreq=1000,
                       default_args=Controller.default_arguments(args))

    print("Controller is ready")
    return l.serve()

if __name__ == '__main__':
    rcode = wavenet_control()
    if rcode != 0:
        sys.exit(rcode)

import os, sys
import tensorflow.keras as keras
import tensorflow.keras.callbacks
import subprocess as sp
import signal

def kill_child_processes():
    parent_id = os.getpid()
    ps_command = sp.Popen("ps -o pid --ppid %d --noheaders" % parent_id, shell=True, stdout=sp.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    for pid_str in ps_output.strip().split(b"\n")[:-1]:
        os.kill(int(pid_str), signal.SIGKILL)
    sys.exit()

class ShutdownCallback(keras.callbacks.Callback):
    def __init__(self, config, on_shutdown):
        self.config = config
        self.on_shutdown = on_shutdown
        self.terminating_ = False
        self.epoch_ = 1

    def on_terminate(self, signo, stack):
        print("Received terminate...")
        self.terminating_ = True
        
    def on_train_begin(self, logs={}):
        # set up the signal handling
        print("Set up signal handling...")
        signal.signal(signal.SIGTERM, self.on_terminate)
        signal.signal(signal.SIGINT, self.on_terminate)
        signal.signal(signal.SIGALRM, self.on_terminate)        
        secs = self.config.PN_CONFIG.shutdown_secs or -1
        if secs > 0:
            mins = secs / 60
            print("Initiating alarm for {} secs ({} mins)".
                           format(secs,mins))
            signal.alarm(secs)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_ = epoch

    def die(self, batchno, logs={}):
        if self.terminating_:
            print("TERMINATING")
            print("Checkpointing the model...")
            self.on_shutdown(self.epoch_, batchno, logs)
            print("Done.")
            print("Exiting.")
            kill_child_processes()
                
    def on_batch_end(self, batchno, logs={}):
        self.die(batchno, logs)

    def on_batch_begin(self, batchno, logs={}):
        self.die(batchno, logs)

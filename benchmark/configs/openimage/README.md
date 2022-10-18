# Open Image Experiment

We initially want to configure this experiment in order to run it on our cluster.
We ahve to keep in mind that FedScale cannot run under the SLURM management, so we'll need to request an interactive shell to SLURM using:

```bash
srun -c <number_of_cpus> --gres=gpu:<gpu_type_name>:<number_of_gpu_of_this_type> --partition=normal --pty bash
```

This command allow us to reserve on SLURM the number of GPUs we need.
If we need to make the experiment run longer than the time we are able to stay connected to the machione via `ssh`, we need to use a terminal multiplexer to keep alive the terminal.
For this we usually open a `screen`, a guide [here](https://linuxize.com/post/how-to-use-linux-screen/), window BEFORE reserving the resources using SLURM, an example could be:

```bash
screen -S <name_of_the_window>
```

In order to submit the FedScale job, following the FedScale tutorial, we should use:

```bash
python /abs/path/to/docker/driver.py submit /abs/path/to/benchmark/configs/openimage/conf.yml # non-local
python /abs/path/to/docker/driver.py start /abs/path/to/benchmark/configs/openimage/conf.yml # local
```

For stopping the job:

```bash
python /abs/path/to/docker/driver.py stop [job_name] # (specified in the yml config)
```

Remember that when stopping the job, `driver.py` is using `ssh`, don't know why.

We can find the job logging `job_name` under the path `log_path` specified in the configuration file.
To check the training loss or test accuracy, we can do:

```bash
cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'
```

Apparently FedScale doesn't like the experiment with only 1 round!!

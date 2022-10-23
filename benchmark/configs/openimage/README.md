
# Open Image Experiment

## Environment

First we need to instatiate te execution environment that will be hardcoded in `conf.yml` files.
In order to do that execute the following commands:

```bash
cd /nfs-share/ls985/FedScale2022

# Please replace ~/.bashrc with ~/.bash_profile for MacOS
FEDSCALE_HOME=$(pwd)
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc # THIS MUST BE DONE IN EVERY MACHINE
echo alias fedscale=\'bash $FEDSCALE_HOME/fedscale.sh\' >> ~/.bashrc 
conda init bash
. ~/.bashrc

conda env create -f  $FEDSCALE_HOME/environment1.yml
conda activate fedscale-11
/nfs-share/ls985/anaconda3/envs/fedscale-11/bin/pip install -e .
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Useful commands

We initially want to configure this experiment in order to run it on our cluster.
We have to keep in mind that FedScale cannot run under the SLURM management, so we'll need to request an interactive shell to SLURM using:

```bash
srun -c <number_of_cpus> --gres=gpu:<gpu_type_name>:<number_of_gpu_of_this_type> --partition=normal --pty bash
srun -c 10 --gres=gpu:v100:1 --mem=80000 --pty bash # mauao v100
srun -c 10 --gres=gpu:a40:1 --mem=80000 --pty bash # mauao a40
srun -c 6 --gres=gpu:rtx2080:1 --mem=80000 --pty bash # ngongotaha
srun -c 6 --gres=gpu:gtx1080:1 --mem=80000 --pty bash # tarawera
```

This command allow us to reserve on SLURM the number of GPUs we need.
If we need to make the experiment run longer than the time we are able to stay connected to the machione via `ssh`, we need to use a terminal multiplexer to keep alive the terminal.
For this we usually open a `screen`, a guide [here](https://linuxize.com/post/how-to-use-linux-screen/), window BEFORE reserving the resources using SLURM, an example could be:

```bash
screen -S <name_of_the_window>
```

In order to submit the FedScale job, following the FedScale tutorial, we should use:

```bash
python </abs/path/to>/docker/driver.py submit </abs/path/to>/benchmark/configs/openimage/conf.yml # non-local
python $FEDSCALE_HOME/docker/driver.py submit $FEDSCALE_HOME/benchmark/configs/openimage/conf.yml
python </abs/path/to>/docker/driver.py start </abs/path/to>/benchmark/configs/openimage/conf.yml # local
python $FEDSCALE_HOME/docker/driver.py start $FEDSCALE_HOME/benchmark/configs/openimage/conf.yml 0
python $FEDSCALE_HOME/docker/driver.py start $FEDSCALE_HOME/benchmark/configs/openimage/conf.yml 1
python $FEDSCALE_HOME/docker/driver.py start $FEDSCALE_HOME/benchmark/configs/openimage/conf.yml 2
python $FEDSCALE_HOME/docker/driver.py start $FEDSCALE_HOME/benchmark/configs/openimage/conf.yml 3
```

For stopping the job:

```bash
python </abs/path/to>/docker/driver.py stop [job_name] [monitor]# (specified in the yml config)
python $FEDSCALE_HOME/docker/driver.py stop openimage monitor
python $FEDSCALE_HOME/docker/driver.py stop all monitor
python </abs/path/to>/docker/driver.py lstop [job_name] [monitor]# (specified in the yml config)
python $FEDSCALE_HOME/docker/driver.py lstop openimage monitor
```

Remember that when stopping the job, `driver.py` is using `ssh`, don't know why.

We can find the job logging `job_name` under the path `log_path` specified in the configuration file.
To check the training loss or test accuracy, we can do:

```bash
cat <job_name>_logging |grep 'Training loss'
cat <job_name>_logging |grep 'FL Testing'
tail -f $FEDSCALE_HOME/openimage_logging
```

Apparently FedScale doesn't like the experiment with only 1 round!!

## Measuring the GPU statistics

For measuring the GPU statistics we need to use the `nvidia-smi` utility because of its speed, other python packages are not as fast as `nvidia-smi`.
The command we need is the following:

```bash
nvidia-smi --query-gpu=timestamp,name,index,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv --filename=</output/file/path.csv> --loop-ms=<period-of-monitoring>
```

I think we need to launch this monitor just before the parameter server, just by adding one line of code to `driver.py`.
We can also modify `shutdown.py` in order to include `nvidia-smi` in the list of processes to shut down.

I've implemented the monitoring, it's configuration is in `conf.yml`.
I did manage to make it start automatically (it starts ~15 seconds before the GPU is being used), but I didn't manage to make it stop automatically.

## Ordering of GPUs in `mauao`

Remember that in `mauao` the `CUDA_ID`s are listed with the all the A40 before the V100, e.g. the first V100 is at `CUDA_ID=6`.
Thanks to Pedro, I've managed to set the same ordering that `nvidia-smi` command has. It was sufficient to put the following lines in my `~.bashrc`.

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
```

## EXPERIMENTS

EXP1: 11 rounds with 1 client per worker per gpu, ps on cpu (4 clients per round)--> count the number of clients that can fit a specific gpu
EXP2: fill the gpus with as much client as we can --> max utilization of GPUs for FedScale --> flower with the automated thing is better for heterogeneous setups
(potEXP3): automated FedScale vs. automated Flower

In order to run "speech" experiment, we need to change the deafult conda package `resampy` from the version `0.4.2` to the version `0.3.1`.


ps on ngongotaha, workers on different gpus --> manipulating driver.py
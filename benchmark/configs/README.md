# Experiments Configuration

In this folder, we are resuming the available experiments proposed in the original FedScale configuration along with the configurations for our experiments.

## Environment

In order to set up the conda environment onto which run the experiments, execute the following commands:

```bash
cd /nfs-share/ls985/FedScale2022

# Please replace ~/.bashrc with ~/.bash_profile for MacOS
FEDSCALE_HOME=$(pwd)
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc # THIS MUST BE DONE IN EVERY MACHINE
echo alias fedscale=\'bash $FEDSCALE_HOME/fedscale.sh\' >> ~/.bashrc 
conda init bash
. ~/.bashrc

# conda env create -f  $FEDSCALE_HOME/environment.yml # ORIGINAL, has issues because of CUDA version on our cluster
conda env create -f  $FEDSCALE_HOME/environment1.yml # MOD, no issues here
conda activate fedscale-11
/nfs-share/ls985/anaconda3/envs/fedscale-11/bin/pip install -e .
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Useful commands and execution

We initially want to configure this experiment in order to run it on our cluster.
We have to keep in mind that FedScale cannot run under the SLURM management using the original script.
We needed to change a bit of the code to made it run on different nodes under a SLURM interactive bash session.
To request such an interactive session use:

```bash
srun -w <node_name> -c <number_of_cpus> --gres=gpu:<gpu_type_name>:<number_of_gpu_of_this_type> --partition=normal --pty bash
```

We suggest to ask SLURM for an interactive session under a terminal multiplexer as the experiments can take a lot.

In order to launch a FedScale job, it's sufficient to execute the script `$FEDSCALE_HOME/docker/driver.py` with the proper parameters.
The first parameter describes the action to take.
The secondo parameter is the path to the configuration `.yml` file.
The third parameter is an integer describing in which node we are executing the script. 
The actions are following:
- `start $FEDSCALE_HOME/benchmark/configs/<exp_name>/<conf_filename>.yml 0`, this execute the configuration in the `.yml` file in the second parameter behaving as if the entire experiment will be executed on the local node; 
- `submit $FEDSCALE_HOME/benchmark/configs/<exp_name>/<conf_filename>.yml 0`, this execute the configuration in the `.yml` file in the second parameter behaving as if the entire experiment will be executed on multiple physical nodes (multiple IPs). 
Two additional actions are available, whose aim is to shutdown the execution of the FedScale experiments.
The shutdown actions don't need the confiration `.yml` file as the second parameter, instead the user could pass to `$FEDSCALE_HOME/docker/driver.py` two addition parameters, namely the 'job_name' (must be the same int he `.yml file`) and flag for shutting down the monitor also.
These parameters presents as follows:
- `stop <job_name> monitor`, this will stop the execution of a FedScale experiment whose execution happens on multiple physical nodes (multiple IPs):
- `lstop <job_name> monitor`, this will stop the execution of a FedScale experiment whose execution happens on the local node.
`<job_name>` could be equal to 'all', meaning that every FedScale program will be shut down.
One could put anything different to 'monitor' in order to keep the monito alive.


We can find the job logging `job_name` under the path `log_path` specified in the configuration file.
To check the training loss or test accuracy, we can do:

```bash
cat <job_name>_logging |grep 'Training loss'
cat <job_name>_logging |grep 'FL Testing'
tail -f $FEDSCALE_HOME/speech_logging
```

Apparently FedScale doesn't like the experiment with only 1 round!!

## Measuring the GPU statistics

For measuring the GPU statistics we need to use the `nvidia-smi` utility because of its speed, other python packages are not as fast as `nvidia-smi`.
I've implemented the monitor to be launched just before the parameter server, it's configuration is in the modified configuration files `.yml`.
The monitor is shut down automatically in every machine after the end of the training.

## Ordering of GPUs in `mauao`

Remember that in `mauao` the `CUDA_ID`s are listed with the all the A40 before the V100, e.g. the first V100 is at `CUDA_ID=6`.
Thanks to Pedro, I've managed to set the same ordering that `nvidia-smi` command has. It was sufficient to put the following lines in my `~/.bashrc`.

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
```

This actually didn't work.

## Additional Notes

In order to run "speech" experiment, we need to change the deafult conda package `resampy` from the version `0.4.2` to the version `0.3.1`.

# Submit job to the remote cluster

import datetime
import os
import pickle
import subprocess
import sys
import time
import yaml


def flatten(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = flatten(subdict).items()
                out.update({key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def process_cmd(yaml_file, local=False, node: int = 0):

    yaml_conf = load_yaml_conf(yaml_file)

    ps_ip = yaml_conf['ps_ip']
    worker_ips, total_gpus = [], []
    
    # cuda for workers
    w_use_cuda = yaml_conf['w_use_cuda']
    cuda_ids = yaml_conf['cuda_ids'] # this is a list of lists
    
    # cuda for server
    ps_use_cuda = yaml_conf['ps_use_cuda']
    ps_cuda_id = yaml_conf['ps_cuda_id']
    
    # monitoring
    monitor = yaml_conf['monitor']['flag']
    monitor_period = yaml_conf['monitor']['period']

    executor_configs = "=".join(yaml_conf['worker_ips'])
    for ip_gpu in yaml_conf['worker_ips']:
        ip, gpu_list = ip_gpu.strip().split(':')
        worker_ips.append(ip)
        total_gpus.append(eval(gpu_list))

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    
    # Defining the running real nodes, using set we can handle repetitions
    running_vms = set()
    running_vms.add(ps_ip)
    [running_vms.add(worker) for worker in worker_ips]
    running_vms = list(running_vms)
    
    job_name = 'fedscale_job'
    log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    job_conf = {'time_stamp': time_stamp,
                'ps_ip': ps_ip,
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)
        
    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
        if conf_name == "job_name":
            job_name = job_conf[conf_name]
        if conf_name == "log_path":
            log_path = os.path.join(
                job_conf[conf_name], 'log', job_name, time_stamp)

    total_gpu_processes = sum([sum(x) for x in total_gpus])
    
    # =========== Opening the logging file ==================
    if node == 0:
        with open(f"{job_name}_logging", 'wb') as fout:
            pass
    
    # =========== Starting monitoring if requested ==========
    if monitor == 'yes':
        monitor_log_dir = os.path.join(log_path, "logs", "monitor")
        if not os.path.isdir(monitor_log_dir):
            os.makedirs(monitor_log_dir, exist_ok=True)
        # Now is one monitor per `driver.py` execution
        monitor_filename = os.path.join(monitor_log_dir, f"/node{node}_{time_stamp}_{job_name}.csv")
        monitor_cmd = f"nvidia-smi --query-gpu=timestamp,name,index,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv --filename={monitor_filename} --loop-ms={monitor_period}"
        with open(f"{job_name}_logging", 'a') as fout:
            if local:
                subprocess.Popen(monitor_cmd, shell=True, stdout=fout, stderr=fout)
            else:
                subprocess.Popen(f'ssh {submit_user}{running_vms[node]} "{setup_cmd} {monitor_cmd}"',
                                shell=True, stdout=fout, stderr=fout)

    # =========== Submit job to parameter server ============
    if node == 0:
        print(f"Starting aggregator on {ps_ip}...")
        ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} {conf_script} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} --use_cuda={ps_use_cuda} --cuda_device=cuda:{ps_cuda_id} "

        with open(f"{job_name}_logging", 'a') as fout:
            if local:
                subprocess.Popen(f'{ps_cmd}', shell=True, stdout=fout, stderr=fout)
            else:
                subprocess.Popen(f'ssh {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"',
                                shell=True, stdout=fout, stderr=fout)
                
        # NOTE: probably we can set a time lower than 10 seconds
        time.sleep(5)
        
    # =========== Submit job to each worker ============
    # This is because we need to keep track of this since it identifies workers
    rank_id = 1 if node == 0 else sum([len(listElem) for listElem in cuda_ids[:int(node)]])+1
    worker = worker_ips[node]
    gpu_ids = cuda_ids[node]
    print(f"Executing worker {worker} in node {node}")
    print(f"Using GPU_IDs {gpu_ids}")
    print(f"Starting from rank {rank_id}")
    
    for cuda_id in gpu_ids: # loop on gpu_ids of the current ip
        print(f"CUDA_ID for this worker in {worker} is {cuda_id}")
        worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} {conf_script} --this_rank={rank_id} --num_executors={total_gpu_processes} --use_cuda={w_use_cuda} --cuda_device=cuda:{cuda_id} "
        rank_id += 1

        with open(f"{job_name}_logging", 'a') as fout:
            if local:
                subprocess.Popen(f'{worker_cmd}',
                                    shell=True, stdout=fout, stderr=fout)
            else:
                subprocess.Popen(f'ssh {submit_user}{worker} "{setup_cmd} {worker_cmd}"',
                                    shell=True, stdout=fout, stderr=fout)

    # Dump the addresses of running workers
    if node == 0:
        current_path = os.path.dirname(os.path.abspath(__file__))
        job_name = os.path.join(current_path, job_name)
        with open(job_name, 'wb') as fout:
            job_meta = {'user': submit_user, 'vms': running_vms, 'use_container': False}
            pickle.dump(job_meta, fout)
    
    print(f"Submitted job, please check your logs {job_conf['log_path']}/logs/{job_conf['job_name']}/{time_stamp} for status")

def terminate(job_name, monitor='', local=False):
    job_meta = dict()
    
    if job_name == "all":
        print("With parameter `all`, this can only shutdown `local` threads.")
        local = True
    else: 
        current_path = os.path.dirname(os.path.abspath(__file__))
        job_meta_path = os.path.join(current_path, job_name)
        if not os.path.isfile(job_meta_path):
            print(f"Fail to terminate {job_name}, as it does not exist")
        with open(job_meta_path, 'rb') as fin:
            job_meta = pickle.load(fin)
        print(job_meta)
        
    cmd_all_fedscale = f"ps -ef | grep python | grep FedScale > {os.path.expandvars('$HOME')}/fedscale_running_temp.txt"
    cmd_job_fedscale = f"ps -ef | grep python | grep job_name={job_name} > '{os.path.expandvars('$HOME')}/fedscale_running_temp.txt'"
    cmd_monitor = f"ps -ef | grep nvidia-smi | grep query | grep monitor >> $HOME/fedscale_running_temp.txt"

    if local:
        print("Shutting down local threads.")
        with open(f"{job_name}_logging", 'a') as fout:
            if job_name == 'all':
                subprocess.Popen([cmd_all_fedscale], shell=True, stdout=fout, stderr=fout)
            else:
                subprocess.Popen([cmd_job_fedscale], shell=True, stdout=fout, stderr=fout)
            # NOTE: probably not needed
            time.sleep(1)
            # added for monitor
            if monitor == 'monitor':
                subprocess.Popen([cmd_monitor], shell=True, stdout=fout, stderr=fout)
            # NOTE: probably not needed
            time.sleep(1)
            [subprocess.Popen([f'kill -9 {str(l.split()[1])} 1>/dev/null 2>&1'], shell=True, stdout=fout, stderr=fout) for l in open(os.path.join(os.getenv("HOME", ""), "fedscale_running_temp.txt")).readlines()]
            subprocess.Popen(["rm $HOME/fedscale_running_temp.txt"], shell=True, stdout=fout, stderr=fout)
    else:
        # TODO: put subprocess.Popen her also
        # print("Shutting down non-local threads.")
        # with open(f"{job_name}_logging", 'a') as fout:
        #     for vm_ip in job_meta['vms']:
        #         print(f"Shutting down job on {vm_ip}")
        #         if job_name == 'all':
        #             subprocess.Popen([f'ssh {job_meta["user"]}{vm_ip} "{cmd_all_fedscale}"'], shell=True, stdin=fout, stdout=fout)
        #         else:
        #             subprocess.Popen([f'ssh {job_meta['user']}{vm_ip} "{cmd_job_fedscale}"'], shell=True, stdin=fout, stdout=fout)
        #         # added for monitor
        #         if monitor == 'monitor':
        #             subprocess.Popen([f'ssh {job_meta['user']}{vm_ip} "{cmd_monitor}"'], shell=True, stdin=fout, stdout=fout)
        #         time.sleep(1)
        #         [subprocess.Popen([f'ssh {job_meta['user']}{vm_ip} "kill -9 {str(l.split()[1])} 1>/dev/null 2>&1"'], shell=True, stdin=fout, stdout=fout) for l in open(os.path.join(os.getenv('HOME', ''), 'fedscale_running_temp.txt')).readlines()]
        #         subprocess.Popen([f'ssh {job_meta['user']}{vm_ip} "rm $HOME/fedscale_running_temp.txt"'], shell=True, stdin=fout, stdout=fout)

print_help: bool = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'submit' or sys.argv[1] == 'start':
        process_cmd(yaml_file=sys.argv[2], local=False if sys.argv[1] == 'submit' else True, node=eval(sys.argv[3]))
    elif sys.argv[1] == 'stop' or sys.argv[1] == 'lstop':
        terminate(sys.argv[2], sys.argv[3], False if sys.argv[1] == 'stop' else True)
    else:
        print_help = True
else:
    print_help = True

if print_help:
    print("\033[0;32mUsage:\033[0;0m\n")
    print("submit $PATH_TO_CONF_YML     # Submit a job")
    print("stop $JOB_NAME               # Terminate a job")
    print()

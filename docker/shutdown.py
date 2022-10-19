import argparse
import os
import sys

# added monitor
job_name = sys.argv[1]
monitor = sys.argv[2]

if job_name == 'all':
    os.system("ps -ef | grep python | grep FedScale > fedscale_running_temp")
else:
    os.system("ps -ef | grep python | grep job_name={} > fedscale_running_temp".format(job_name))
# added for monitor
if monitor == 'monitor':
    os.system("ps -ef | grep nvidia-smi >> fedscale_running_temp")
[os.system("kill -9 "+str(l.split()[1]) + " 1>/dev/null 2>&1") for l in open("fedscale_running_temp").readlines()]
os.system("rm fedscale_running_temp")

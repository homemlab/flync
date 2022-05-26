#!/usr/bin/env python3

import sys
from subprocess import Popen, PIPE, run
import shlex
import os
from pathlib import Path

workdir=Path(sys.argv[1])
sra=Path(sys.argv[2])
runinfo=str(workdir/'results'/'runinfo.csv')

with open(sra, 'r') as sra_file:
    sra_list = sra_file.read().splitlines()

os.chdir(workdir)

run(['echo', '"----- EXTRACTING RUN INFOS -----"'])

if Path.exists(workdir/'results'/'runinfo.csv'):
    run('echo SKIPPING GETTING SRA RUN INFO')
else:
    run(['touch', runinfo])
#    run('echo sra_access,avg_len,strategy,lib_prep,omic,read_layout,bio_proj > ' + runinfo)
    for i in sra_list:
        p1_str="echo -db sra -query " + i
        p1 = Popen(shlex.split(p1_str), stdout=PIPE)
        p2 = Popen(shlex.split("efetch -format runinfo"), stdin=p1.stdout, stdout=PIPE)
        p3 = Popen(shlex.split("cut -f1,7,13,14,15,16,22 -d,"), stdin=p2.stdout)
        print(p3.stdout)
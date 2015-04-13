cat run.sh
#!/bin/bash
module load openmpi
module load intel-mkl
make ass1
qsub batch/batch_job_01
qsub batch/batch_job_02
qsub batch/batch_job_04
qsub batch/batch_job_08
qsub batch/batch_job_16
qsub batch/batch_job_32

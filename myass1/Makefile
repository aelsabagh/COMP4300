HOST=$(shell hostname)
CC_FLAGS = -g -Wall -O2
LD_FLAGS = -mkl

ass1:	ass1.c harness.c summa.c
	mpicc $(CC_FLAGS) harness.c ass1.c summa.c -o ass1 $(LD_FLAGS)
	mpicc $(CC_FLAGS) print.c -o print
	
clean:
	rm ass1 *.o *~

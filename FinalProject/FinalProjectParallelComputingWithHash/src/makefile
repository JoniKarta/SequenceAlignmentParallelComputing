all:  
	mpicxx -fopenmp -c finalProjectParallelComputing.c -o finalProjectParallelComputing.o
	mpicxx -fopenmp -c utility.c -o utility.o
	nvcc -I../inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o finalexec finalProjectParallelComputing.o cudaFunctions.o  utility.o /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt
 
run:  
	mpiexec -np 2 ./finalexec
	
runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./finalexec
	              
clean:
	rm -f *.o finalexec

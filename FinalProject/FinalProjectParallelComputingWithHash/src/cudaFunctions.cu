#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "proto.h"
#include "gpumem.h"
#define NUM_STREAMS 4
#define CUDA_STREAM
//#define CUDA_DEFAULT_STREAM

#ifdef CUDA_STREAM
const int num_streams = NUM_STREAMS;
cudaStream_t streams[num_streams];
#endif


__device__ int hash(char x, char y){
	int hashedIdx =	((x + y)*(x + y + 1)/2) + y;
	return hashedIdx;  
}


__global__  void calc_best_score(char* d_main_seq, char* d_sec_seq,int* d_hash_con,int* d_hash_semi_con, weight_t* weight, int offset,float* results,int res_block_len) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid > res_block_len)
    	return;
	for(int hyphen = 1; hyphen <= res_block_len; hyphen++){
		int delta = tid <= hyphen ? 0 : -1;
		if(tid == hyphen){
			results[tid + (hyphen-1)*res_block_len] = -weight->w4;
		}else if(d_main_seq[tid + offset] == d_sec_seq[tid + delta]){
			results[tid + (hyphen-1)*res_block_len] = weight->w1;
		}else if(d_hash_con[hash(d_main_seq[tid + offset], d_sec_seq[tid + delta])]){
			results[tid + (hyphen-1)*res_block_len] = -weight->w2;
		}else if(d_hash_semi_con[hash(d_main_seq[tid + offset],d_sec_seq[tid + delta])]){
			results[tid + (hyphen-1)*res_block_len] = -weight->w3;
		}else{
			results[tid + (hyphen-1)*res_block_len] = -weight->w4;
		}
	}
	
}

void cudaError(cudaError_t error, const char* message){
	
	if (error != cudaSuccess)
    {
        fprintf(stderr, message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

float* compute_on_gpu(char *d_main_seq, char *d_sec_seq,int* d_hash_con,int* d_hash_semi_con ,weight_t* weight, int offset,int res_len,int thread_id){
	
	// Allocate the result in the gpu's memory
	float *d_result = NULL;
	size_t resultArraySize = sizeof(float) * (res_len * res_len);
	float* res = (float*)malloc(resultArraySize);
	cudaMalloc((void**) &d_result, resultArraySize);
	
	// Set the number of thread in each block
	int threadsPerBlock = 256;

	// Compute the grid size 
	int blocksPerGrid = (res_len*res_len + threadsPerBlock - 1) / threadsPerBlock;
	
	#ifdef CUDA_STREAM
    cudaStreamCreate(&streams[thread_id]);
	calc_best_score<<<blocksPerGrid, threadsPerBlock,0,streams[thread_id]>>>(d_main_seq, d_sec_seq, d_hash_con, d_hash_semi_con,weight,offset, d_result,res_len);
	cudaMemcpyAsync(res , d_result , res_len*res_len*sizeof(float), cudaMemcpyDeviceToHost,streams[thread_id]);

	cudaStreamDestroy(streams[thread_id]);
	#endif

	#ifdef CUDA_DEFAULT_STREAM
	calc_best_score<<<blocksPerGrid, threadsPerBlock>>>(d_main_seq, d_sec_seq, d_hash_con, d_hash_semi_con,weight,offset, d_result,res_len);
	cudaMemcpy(res , d_result , res_len*res_len*sizeof(float), cudaMemcpyDeviceToHost);
	#endif
	
	// Free the current result array from the device 
	cudaError(cudaFree(d_result),"Failed free result device");
	return res;
			
}



char* allocate_main_sequence_on_gpu(char* h_mainSequence){
	char *d_mainSequence = NULL;
	size_t mainSequenceSize = (strlen(h_mainSequence)) * sizeof(char);
	cudaError(cudaMalloc((void**) &d_mainSequence, mainSequenceSize),"Could not allocate main sequence");
    cudaMemcpy(d_mainSequence, h_mainSequence, mainSequenceSize, cudaMemcpyHostToDevice);
	return d_mainSequence;
}

char* allocate_sec_sequence_on_gpu(char* h_sec_seq){
	char *d_sec_seq = NULL;
	size_t sec_seq_len = (strlen(h_sec_seq)) * sizeof(char);
    cudaError(cudaMalloc((void**) &d_sec_seq, sec_seq_len),"Could allocate sec sequence");
	cudaMemcpy(d_sec_seq, h_sec_seq, sec_seq_len, cudaMemcpyHostToDevice);
   	
	return d_sec_seq;
}

int* allocate_hash_conservative_on_gpu(int* hashedCon){
	int* d_hashedCon = NULL;
	size_t hashingSize = (1 << 14) * sizeof(int);
	cudaError(cudaMalloc((void**) &d_hashedCon, hashingSize),"Could not allocate con-hash");
    cudaMemcpy(d_hashedCon, hashedCon, hashingSize, cudaMemcpyHostToDevice);
    return d_hashedCon;
}

int* allocate_hash_semi_conservative_on_gpu(int* hashedSemiCon){
	int* d_hashedSemiCon = NULL;
	size_t hashingSize = (1 << 14) * sizeof(int);
	cudaError(cudaMalloc((void**) &d_hashedSemiCon, hashingSize),"Could not allocate semi-hash");
    cudaMemcpy(d_hashedSemiCon, hashedSemiCon, hashingSize, cudaMemcpyHostToDevice);
	return d_hashedSemiCon;
}

weight_t* allocate_weight_on_gpu(weight_t* weight){
	weight_t* d_weight = NULL;
	size_t weight_size = sizeof(weight_t);
	cudaError(cudaMalloc((void**) &d_weight, weight_size),"Could not allocate weight");
    cudaMemcpy(d_weight, weight, weight_size, cudaMemcpyHostToDevice);
	return d_weight;
}

void free_shared_resources(char* d_mainSequence, int* d_hashedCon, int* d_hashedSemiCon,weight_t* d_weight){
	cudaError(cudaFree(d_hashedCon),"Failed to free hashed from the device");
	cudaError(cudaFree(d_hashedSemiCon),"Failed to free semi-hashed from the device");
	cudaError(cudaFree(d_mainSequence),"Failed to free main sequence from the device");
	cudaError(cudaFree(d_weight),"Failed to free weight from the device");
}

void free_sec_sequence(char* d_sec_seq){
	cudaError(cudaFree(d_sec_seq),"Failed free secondary sequence device");
}


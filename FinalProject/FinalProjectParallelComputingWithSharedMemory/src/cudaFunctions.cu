#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "proto.h"
#include "gpumem.h"
//#define CUDA_STREAM
#define CUDA_DEFAULT_STREAM
#define RES_SIZE 26

#ifdef CUDA_STREAM
#define NUM_STREAMS 4
const int num_streams = NUM_STREAMS;
cudaStream_t streams[num_streams];
#endif

__global__ void calc_best_score(char *d_main_seq, char *d_sec_seq, float *d_score_matrix, weight_t *weight, int offset, float *results, int res_block_len) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ float _score_matrix[RES_SIZE * RES_SIZE];

	for (int i = tid; i < RES_SIZE * RES_SIZE; i++) {
		_score_matrix[i] = d_score_matrix[i];
	}
	
	__syncthreads();
	
	if (tid > res_block_len)
		return;
	for (int j = 0; j < res_block_len; j++) {
		int delta = j >= tid ? 1 : 0;
		results[tid - 1] += _score_matrix[(d_main_seq[j + offset + delta] - 'A') * RES_SIZE + (d_sec_seq[j] - 'A')];
	}
}

void cudaError(cudaError_t error, const char *message) {

	if (error != cudaSuccess) {
		fprintf(stderr, message, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

float* compute_on_gpu(char *d_main_seq, char *d_sec_seq, float *d_score_matrix,
		weight_t *weight, int offset, int res_len, int thread_id) {

	// Allocate the result in the gpu's memory
	float *d_result = NULL;
	size_t resultArraySize = sizeof(float) * res_len;
	float *res = (float*) malloc(resultArraySize);
	cudaMalloc((void**) &d_result, resultArraySize);
	cudaMemset(d_result, 0, resultArraySize);
	
	// Set the number of thread in each block
	int threadsPerBlock = 256;

	// Compute the grid size 
	int blocksPerGrid = (res_len + threadsPerBlock - 1) / threadsPerBlock;

#ifdef CUDA_STREAM
    cudaStreamCreate(&streams[thread_id]);
	calc_best_score<<<blocksPerGrid, threadsPerBlock,0,streams[thread_id]>>> (d_main_seq, d_sec_seq,d_score_matrix,weight,offset, d_result,res_len);
	cudaMemcpyAsync(res , d_result ,res_len*sizeof(float), cudaMemcpyDeviceToHost,streams[thread_id]);
	cudaStreamDestroy(streams[thread_id]);
	#endif

#ifdef CUDA_DEFAULT_STREAM
	calc_best_score<<<blocksPerGrid, threadsPerBlock>>>(d_main_seq, d_sec_seq,d_score_matrix,weight,offset, d_result,res_len);
	cudaMemcpy(res, d_result, res_len * sizeof(float), cudaMemcpyDeviceToHost);
#endif

	// Free the current result array from the device 
	cudaError(cudaFree(d_result), "Failed free result device");
	return res;

}

float* allocate_score_matrix_on_gpu(float score_matrix[RES_SIZE][RES_SIZE]) {
	float *d_score_matrix = NULL;
	size_t score_matrix_size = sizeof(float) * (RES_SIZE * RES_SIZE);
	cudaMalloc((void**) &d_score_matrix, score_matrix_size);
	cudaMemcpy(d_score_matrix, score_matrix, score_matrix_size, cudaMemcpyHostToDevice);
	return d_score_matrix;
}

char* allocate_main_sequence_on_gpu(char *h_mainSequence) {
	char *d_mainSequence = NULL;
	size_t mainSequenceSize = (strlen(h_mainSequence)) * sizeof(char);
	cudaError(cudaMalloc((void**) &d_mainSequence, mainSequenceSize), "Could not allocate main sequence");
	cudaMemcpy(d_mainSequence, h_mainSequence, mainSequenceSize, cudaMemcpyHostToDevice);
	return d_mainSequence;
}

char* allocate_sec_sequence_on_gpu(char *h_sec_seq) {
	char *d_sec_seq = NULL;
	size_t sec_seq_len = (strlen(h_sec_seq)) * sizeof(char);
	cudaError(cudaMalloc((void**) &d_sec_seq, sec_seq_len), "Could allocate sec sequence");
	cudaMemcpy(d_sec_seq, h_sec_seq, sec_seq_len, cudaMemcpyHostToDevice);
	return d_sec_seq;
}

weight_t* allocate_weight_on_gpu(weight_t *weight) {
	weight_t *d_weight = NULL;
	size_t weight_size = sizeof(weight_t);
	cudaError(cudaMalloc((void**) &d_weight, weight_size), "Could not allocate weight");
	cudaMemcpy(d_weight, weight, weight_size, cudaMemcpyHostToDevice);
	return d_weight;
}

void free_shared_resources(char *d_mainSequence, weight_t *d_weight, float *d_score_matrix) {
	cudaError(cudaFree(d_mainSequence), "Failed to free main sequence from the device");
	cudaError(cudaFree(d_weight), "Failed to free weight from the device");
	cudaError(cudaFree(d_score_matrix), "Failed to free score matrix from the device");
}

void free_sec_sequence(char *d_sec_seq) {
	cudaError(cudaFree(d_sec_seq), "Failed free secondary sequence device");
}


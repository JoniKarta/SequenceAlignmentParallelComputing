/*
 * gpumem.h
 *
 *  Created on: 18 Aug 2020
 *      Author: jonathan
 */

#ifndef GPUMEM_H_
#define GPUMEM_H_
#include "proto.h"

float* compute_on_gpu(char *d_main_seq, char *d_sec_seq,int* d_hash_con,int* d_hash_semi_con, weight_t* weight,int offset,int res_len,int thread_id);

char* allocate_main_sequence_on_gpu(char* h_main_seq);

int* allocate_hash_conservative_on_gpu(int* h_hash_con);

int* allocate_hash_semi_conservative_on_gpu(int* h_hash_semi_con);

char* allocate_sec_sequence_on_gpu(char* h_sec_seq);

weight_t* allocate_weight_on_gpu(weight_t* d_weight);

void free_shared_resources(char* d_main_seq, int* d_hash_con, int* d_hash_semi_con, weight_t* d_weight);

void free_sec_sequence(char* d_sec_seq);

void createCudaStream(int thread_id);

#endif /* GPUMEM_H_ */

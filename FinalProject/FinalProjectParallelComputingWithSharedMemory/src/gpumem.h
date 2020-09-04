/*
 * gpumem.h
 *
 *  Created on: 18 Aug 2020
 *      Author: jonathan
 */

#ifndef GPUMEM_H_
#define GPUMEM_H_
#include "proto.h"

float* compute_on_gpu(char *d_main_seq, char *d_sec_seq,float* d_score_matrix, weight_t* weight,int offset,int res_len,int thread_id);

char* allocate_main_sequence_on_gpu(char* h_main_seq);

char* allocate_sec_sequence_on_gpu(char* h_sec_seq);

weight_t* allocate_weight_on_gpu(weight_t* d_weight);

void free_shared_resources(char* d_main_seq, weight_t* d_weight, float* d_score_matrix);

void free_sec_sequence(char* d_sec_seq);

float* allocate_score_matrix_on_gpu(float score_matrix[RES_SIZE][RES_SIZE]);


#endif /* GPUMEM_H_ */

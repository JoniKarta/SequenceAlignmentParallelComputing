/*
 * utility.h
 *
 *  Created on: 24 Jul 2020
 *      Author: jonathan
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include "mpi.h"
#include "proto.h"
#include "float.h"
#include "gpumem.h"

#define BUFFER_SIZE 3000
#define MASTER 0
#define SEMI_CONSERVATIVE 11
#define CONSERVATIVE 9
#define NODES_NUM 2


// Sending the shared data (Weight, Main Sequence).
void send_shared_data(weight_t* weight,char* main_sequence,int num_proc, int tag);

// Receive the shared data (Weight, Main Sequence).
void recieve_shared_data(char** main_sequence, weight_t* weight, int source, int tag, MPI_Status* status);

// Sending all tasks the each process/
int send_tasks(holder_t** holders, int num_proc, int tag);

// Receive tasks formed at sequences array of characters.
char** receive_tasks(int* num_of_sequences,int source,int tag, MPI_Status* status);

// Initializing the holders which hold each process's sequences.
holder_t** init_holders(holder_t** holders,int num_of_holders);

// Read all the data from the file
holder_t** read_from_file(holder_t **holders, char **main_sequence, weight_t *weights, int num_proc,int* num_of_sequences);

// Find the process which current hold the minimum size of sequences are total length.
int find_min(holder_t** holders,int size,int seq_length);

// Sending the results after calculation
void send_result(alignment_t* score,int tag);

// Receive the result from each process.
void receive_result(alignment_t* score, int p, int tag, MPI_Status *status);

// Print the result FOR TESTING ONLY.
void printResult(alignment_t* score);

// Write the result to the file
void write_result_to_file(holder_t** holder, int num_proc, alignment_t *all_scores);


// Calculating the score using OpenMP
void calc_score(char *d_main_seq, int main_seq_len, char *h_sec_seq,float* score_matrix, weight_t *weight, alignment_t *score);

// Setting the best score 
void set_best_score(int res_len, int offset, float *results, float *max_score, alignment_t *score);

// Free all host memeory
void free_host_memory(holder_t** holder,int num_holders, alignment_t* score);

// Create result score matrix
void create_result_score(float score_matrix[RES_SIZE][RES_SIZE], weight_t* weights);

// Check for pair in group 
bool pair_in_group(char a, char b, const char* group[], size_t group_size);

// Custom Weight MPI Data type.
MPI_Datatype weightMPIType();

// Custom alignment score MPI Data type
MPI_Datatype alignmentMPIType();


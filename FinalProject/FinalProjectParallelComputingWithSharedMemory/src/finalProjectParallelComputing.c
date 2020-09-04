/*
 ============================================================================
 Name        : FinalProjectParallelComputing.c
 Author      : Jonathan Karta
 Version     : 1.0.0
 Copyright   : Your copyright notice
 Description : 
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "mpi.h"
#include "utility.h"
//#include "gpumem.h"

int main(int argc, char *argv[]) {
	int my_rank, p, tag = 0, num_of_sequence, num_of_scores = 0;
	char **sequences, *main_sequence, *d_main_seq;
	float score_matrix[RES_SIZE][RES_SIZE], *d_score_matrix;
	double startTime, endTime;
	alignment_t *all_scores, current_score = { 0 };
	weight_t weights = { 0 }, *d_weight;
	holder_t **seq_holder = { 0 };

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if (p != NODES_NUM) {
		printf("The program can run two process only\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}

	if (my_rank == MASTER) {
		startTime = MPI_Wtime();

		// Read the data from file to sequences holder
		seq_holder = read_from_file(seq_holder, &main_sequence, &weights, p, &num_of_sequence);

		// Create score matrix
		create_result_score(score_matrix, &weights);

		// Allocate the matrix on the gpu
		d_score_matrix = allocate_score_matrix_on_gpu(score_matrix);

		// Allocate main memory on the gpu
		d_main_seq = allocate_main_sequence_on_gpu(main_sequence);

		// Allocate weight on the gpu
		d_weight = allocate_weight_on_gpu(&weights);

		// Send the shared data to each process
		send_shared_data(&weights, main_sequence, p, tag);

		// Send each process's tasks
		send_tasks(seq_holder, p, tag);

		// Allocate spaces to all the alignment scores
		all_scores = (alignment_t*) malloc(sizeof(alignment_t) * num_of_sequence);

		// Loop over each tasks and calculate the score using openmp & cuda
		for (int i = 0; i < seq_holder[0]->num_of_sequences; i++, num_of_scores++) {
			calc_score(d_main_seq, strlen(main_sequence), seq_holder[0]->sequences[i], d_score_matrix, d_weight, &current_score);
			memcpy(all_scores + num_of_scores, &current_score, sizeof(alignment_t));
		}

		// Receive all tasks
		for (int i = 1; i < p; i++) {
			for (int j = 0; j < seq_holder[i]->num_of_sequences; j++, num_of_scores++) {
				receive_result(&current_score, i, tag, &status);
				memcpy(all_scores + num_of_scores, &current_score, sizeof(alignment_t));
			}
		}

		endTime = MPI_Wtime();

		// Free all memory from the device
		free_shared_resources(d_main_seq, d_weight, d_score_matrix);

		// Write the result to file
		write_result_to_file(seq_holder, p, all_scores);

		// Free all host's memory
		free_host_memory(seq_holder, p, all_scores);

		printf("TOTAL TIME TAKEN: %f\n", endTime - startTime);

	} else {

		// Receive shared data from the master
		recieve_shared_data(&main_sequence, &weights, MASTER, tag, &status);

		// Receive tasks from the master
		sequences = receive_tasks(&num_of_sequence, MASTER, tag, &status);

		// Allocate main memory on the gpu
		d_main_seq = allocate_main_sequence_on_gpu(main_sequence);

		// Allocate weight on the gpu
		d_weight = allocate_weight_on_gpu(&weights);

		// Creat result matrix 
		create_result_score(score_matrix, &weights);

		// Allocate result matrix on the gpu
		d_score_matrix = allocate_score_matrix_on_gpu(score_matrix);

		// Calculate each score using openmp & cuda and send to the master
		for (int i = 0; i < num_of_sequence; i++) {
			calc_score(d_main_seq, strlen(main_sequence), sequences[i], d_score_matrix, d_weight, &current_score);
			send_result(&current_score, tag);
		}

		// Free all memory from the device
		free_shared_resources(d_main_seq, d_weight, d_score_matrix);
	}

	/* shut down MPI */
	MPI_Finalize();

	return 0;
}


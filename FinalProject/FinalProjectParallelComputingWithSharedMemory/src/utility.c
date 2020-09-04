/*
 * utility.c
 *
 *  Created on: 24 Jul 2020
 *      Author: jonathan
 */
#include "utility.h"


const char* conservative[] = {
		"NDEQ","NEQK","STA",
		"MILV","QHRK","NHQK",
		"FYW","HY","MILF"
};


const char* semiConservative[] = {
		"SAG","ATV","CSA",
		"SGND","STPA","STNK",
		"NEQHRK","NDEQHK","SNDEQK",
		"HFY","FVLIM"
};

holder_t** read_from_file(holder_t **holders, char **main_sequence, weight_t *weights, int num_proc, int *num_of_sequences) {

	// Initialize the buffer to 0.
	char charBuffer[BUFFER_SIZE] = { 0 };

	// Open the file with read only permissions.
	FILE *file = fopen("input.txt", "r");

	// File validation.
	if (!file){
		printf("Could not open the file\n");
		MPI_Abort(MPI_COMM_WORLD,__LINE__);
	}

	// Read the weights from the file.
	fscanf(file, "%f%f%f%f", &weights->w1, &weights->w2, &weights->w3,&weights->w4);

	// Read the main sequence from the file.
	fscanf(file, "%s", charBuffer);

	// Read the number of secondary sequences from the file.
	fscanf(file, "%d", num_of_sequences);

	// Allocate memory for the main sequence.
	*main_sequence = (char*) calloc(strlen(charBuffer) + 1, sizeof(char));
	strcpy(*main_sequence, charBuffer);

	// Initialize the holders which holds all the sequences.
	holders = init_holders(holders, num_proc);

	// Hold the index of holder which the sequence is going to be inserted.
	int idx;

	// Read all sequences from the files
	for (int i = 0; i < *num_of_sequences; i++) {

		// Read the secondary sequence.
		fscanf(file, "%s", charBuffer);

		// Find the holder which the sequence can be inserted to.
		idx = find_min(holders, num_proc, strlen(charBuffer));

		// Allocate new space for the sequence array which all the sequence each process whill have.
		holders[idx]->sequences = (char**) realloc(holders[idx]->sequences, (holders[idx]->num_of_sequences + 1) * sizeof(char*));

		// Allocate space for the sequence in sequences
		holders[idx]->sequences[holders[idx]->num_of_sequences] = (char*) calloc(strlen(charBuffer) + 1, sizeof(char));

		// Copy the buffer to the sequence in the indexed holder
		strcpy(holders[idx]->sequences[holders[idx]->num_of_sequences], charBuffer);

		// Update the number of sequences the current holder have.
		holders[idx]->num_of_sequences++;
	}

	fclose(file);

	return holders;
}

void send_shared_data(weight_t *weight, char *main_sequence, int num_proc, int tag) {

	// Sending to each process the weight and the main sequence
	int sequence_length = strlen(main_sequence);

	for (int i = 1; i < num_proc; i++) {
		// Sending the weight to each process
		MPI_Send(weight, 1, weightMPIType(), i, tag, MPI_COMM_WORLD);

		// Sending the size of the main sequence
		MPI_Send(&sequence_length, 1, MPI_INT, i, tag, MPI_COMM_WORLD);

		// Sending the main sequence
		MPI_Send(main_sequence, strlen(main_sequence), MPI_CHAR, i, tag, MPI_COMM_WORLD);
	}
}

void recieve_shared_data(char **main_sequence, weight_t *weight, int source, int tag, MPI_Status *status) {

	// Allocate the size of the main sequence
	int sequence_length;

	// Receive the weight from the master
	MPI_Recv(weight, 1, weightMPIType(), source, tag, MPI_COMM_WORLD, status);

	//Receive the sequence size for allocation
	MPI_Recv(&sequence_length, 1, MPI_INT, source, tag, MPI_COMM_WORLD, status);

	// Allocate memory for the main sequence
	*main_sequence = (char*) calloc(sequence_length, sizeof(char));

	// Receive the sequence from the master
	MPI_Recv(*main_sequence, sequence_length, MPI_CHAR, source, tag, MPI_COMM_WORLD, status);

}

int send_tasks(holder_t **holders, int num_proc, int tag) {

	int totalTasks = 0;
	for (int i = 1; i < num_proc; i++) {
		// Send the number of sequences each holder have
		MPI_Send(&holders[i]->num_of_sequences, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
		for (int j = 0; j < holders[i]->num_of_sequences; j++, totalTasks++) {

			// Send the length of each sequence
			int sequence_length = strlen(holders[i]->sequences[j]);
			MPI_Send(&sequence_length, 1, MPI_INT, i, tag, MPI_COMM_WORLD);

			// Send the sequence
			MPI_Send(holders[i]->sequences[j], sequence_length,
			MPI_CHAR, i, tag, MPI_COMM_WORLD);
		}
	}
	return totalTasks;
}

char** receive_tasks(int *num_of_sequences, int source, int tag,
		MPI_Status *status) {
	int sequence_length;

	// Receive the number of sequences each process have
	MPI_Recv(num_of_sequences, 1, MPI_INT, source, tag, MPI_COMM_WORLD, status);
	
	// Allocate the the result
	char **result = (char**) malloc(sizeof(char*) * (*num_of_sequences));

	// Loop over number of sequences and gets each one of them
	for (int i = 0; i < *num_of_sequences; i++) {
		MPI_Recv(&sequence_length, 1, MPI_INT, source, tag, MPI_COMM_WORLD, status);
		result[i] = (char*) calloc(sequence_length + 1, sizeof(char));
		MPI_Recv(result[i], sequence_length, MPI_CHAR, source, tag,
		MPI_COMM_WORLD, status);
	}

	return result;
}

int find_min(holder_t **holders, int size, int seqLength) {
	int min = holders[0]->total_sequences_length, idx = 0;
	for (int i = 1; i < size; i++) {
		if (holders[i]->total_sequences_length < min) {
			min = holders[i]->total_sequences_length;
			idx = i;
		}
	}
	holders[idx]->total_sequences_length += seqLength;
	return idx;
}

holder_t** init_holders(holder_t **holders, int numOfHolders) {

	holders = (holder_t**) malloc(sizeof(holder_t*) * numOfHolders);
	if (!holders)
		return 0;

	for (int i = 0; i < numOfHolders; i++) {
		holders[i] = (holder_t*) calloc(1, sizeof(holder_t));
		if (!holders[i]){
			printf("Allocate holder failed\n");
			MPI_Abort(MPI_COMM_WORLD,__LINE__);
		}
	}
	return holders;
}

void send_result(alignment_t *score, int tag) {
	MPI_Send(score, 1, alignmentMPIType(), MASTER, tag, MPI_COMM_WORLD);
}

void receive_result(alignment_t *score, int p, int tag, MPI_Status *status) {
	MPI_Recv(score, 1, alignmentMPIType(), p, tag, MPI_COMM_WORLD, status);
}

void write_result_to_file(holder_t** holder, int num_proc, alignment_t *all_scores) {
	FILE *f = fopen("result.txt", "w+");
	if (!f)
		return;
	for(int i = 0, m = 0; i < num_proc; i++) // Loop over each process
		for (int j = 0; j < holder[i]->num_of_sequences; j++,m++){ // for each sequence 
			fprintf(f, "n = %d, k = %d \t NS2 = %s\n",  all_scores[m].n, all_scores[m].k, holder[i]->sequences[j]);
			printResult(&all_scores[m]);
		}
	fclose(f);
}

void printResult(alignment_t *score) {
	printf("Best score is: offset = %d, hyphen Index = %d\n", score->n,score->k);
}


MPI_Datatype alignmentMPIType() {
	alignment_t score;
	MPI_Datatype AlignmentMPIType;
	MPI_Datatype type[2] = { MPI_INT, MPI_INT };
	int blocklen[2] = { 1, 1 };
	MPI_Aint disp[2];

	disp[0] = (char*) &score.n - (char*) &score;
	disp[1] = (char*) &score.k - (char*) &score;
	MPI_Type_create_struct(2, blocklen, disp, type, &AlignmentMPIType);
	MPI_Type_commit(&AlignmentMPIType);
	return AlignmentMPIType;
}

MPI_Datatype weightMPIType() {
	weight_t weights;
	MPI_Datatype WeightsMPIType;
	MPI_Datatype type[4] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT };
	int blocklen[4] = { 1, 1, 1, 1 };
	MPI_Aint disp[4];

	disp[0] = (char*) &weights.w1 - (char*) &weights;
	disp[1] = (char*) &weights.w2 - (char*) &weights;
	disp[2] = (char*) &weights.w3 - (char*) &weights;
	disp[3] = (char*) &weights.w4 - (char*) &weights;
	MPI_Type_create_struct(4, blocklen, disp, type, &WeightsMPIType);
	MPI_Type_commit(&WeightsMPIType);
	return WeightsMPIType;
}

bool pair_in_group(char a, char b, const char* group[], size_t group_size)
{
	for(int i = 0 ; i < group_size ; i++){
		if(strchr(group[i],a) && strchr(group[i],b))
			return true;
	}
	return false;
}

void create_result_score(float score_matrix[RES_SIZE][RES_SIZE], weight_t* weights){
	for(int i = 0; i < RES_SIZE; i++){
		for(int j = 0; j < RES_SIZE; j++){
			if(i + 'A' == j + 'A'){
				score_matrix[i][j] = weights->w1;
			}else if(pair_in_group(i + 'A', j + 'A',conservative,CONSERVATIVE)) {
				score_matrix[i][j] = -weights->w2;
			}else if(pair_in_group(i + 'A', j + 'A', semiConservative,SEMI_CONSERVATIVE)){
				score_matrix[i][j] = -weights->w3;
			}else{
				score_matrix[i][j] = -weights->w4;
			}
		}
	}

}

void calc_score(char *d_main_seq, int main_seq_len, char *h_sec_seq,float* score_matrix, weight_t *weight, alignment_t *score) {

	char *d_sec_seq = allocate_sec_sequence_on_gpu(h_sec_seq);
	float max_score = FLT_MIN_EXP;
	int sec_seq_len = strlen(h_sec_seq);
	//omp_set_num_threads(12);
	#pragma omp parallel for
	for (int offset = 0; offset < main_seq_len - sec_seq_len; offset++) {
		float *results = compute_on_gpu(d_main_seq, d_sec_seq, score_matrix, weight, offset,sec_seq_len,omp_get_thread_num());
		for (int hyphen_res = 1; hyphen_res <= sec_seq_len; hyphen_res++){
			float current_score = results[hyphen_res - 1];
		//#pragma omp critical
		if (current_score > max_score) {
			max_score = current_score;
			score->n = offset;
			score->k = hyphen_res;
			}
		}
		free(results);
	}

	//printf("%f\n",max_score);
	free_sec_sequence(d_sec_seq);

}


void free_host_memory(holder_t** holder,int num_holders, alignment_t* score){
	free(score);
	
	for(int i = 0; i < num_holders;i++){
		for(int j = 0; j < holder[i]->num_of_sequences; j++){
			free(holder[i]->sequences[j]);
		}
		free(holder[i]->sequences);
	}
	free(holder);
}






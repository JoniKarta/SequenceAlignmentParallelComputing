#pragma once

#define RES_SIZE 26

struct SequenceHolder {
	char** sequences;
	int total_sequences_length;
	int num_of_sequences;
}typedef holder_t;

struct Weight {
	float w1;
	float w2;
	float w3;
	float w4;
}typedef weight_t;

struct AlignmentScore {
	int n;
	int k;
	float score;
}typedef alignment_t;


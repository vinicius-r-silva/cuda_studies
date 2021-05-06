#ifndef PERMUTATION_CLASS
#define PERMUTATION_CLASS

#include <bits/stdc++.h> 

class AllPermutation{
private: 
    uint8_t** arrays;
    uint8_t* base_array;

    uint8_t array_size;
    uint8_t curr_index;
    uint8_t initial_room;
    
    int curr_permut;
    int qtd_permutations;

    void addPermutation();
    void makePermutations(uint8_t* a, int l, int r);

public:
    // Constructor 
    AllPermutation(uint8_t* array, int array_size, uint8_t initial_room);
    ~AllPermutation() ;

    bool hasNext();
    uint8_t* getNext();
};



#endif
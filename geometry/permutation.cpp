// C++ program to print all  
// permutations with duplicates allowed  
// This is code is contributed by rathbhupendra 
// https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/

#include <bits/stdc++.h> 
#include "permutation.h"
  
void AllPermutation::addPermutation(){
    arrays[curr_permut][0] = initial_room;
    memcpy(arrays[curr_permut] + 1, base_array, array_size * sizeof(base_array[0]));
}

void AllPermutation::makePermutations(uint8_t* a, int l, int r){
    // Base case  
    if (l == r){
        addPermutation();
        curr_permut++;
    }
    else{  
        // Permutations made  
        for (int i = l; i <= r; i++)  {  
            // Swapping done  
            std::swap(a[l], a[i]);  

            // Recursion called  
            makePermutations(a, l+1, r);  

            //backtrack  
            std::swap(a[l], a[i]);  
        }  
    }  
}

// Constructor 
AllPermutation::AllPermutation(uint8_t* array, int array_size, uint8_t initial_room){ 
    this->curr_index = 0;
    this->curr_permut = 0;
    this->initial_room = initial_room;

    int l = 0;
    this->array_size = array_size;

    int i = 0;
    int fact = 1;
    for (i = 1; i <= array_size; i++) fact = fact*i;
    
    arrays = (uint8_t**)calloc(fact, sizeof(uint8_t*));
    for(i = 0; i < fact; i++){
        arrays[i] = (uint8_t*)malloc((array_size + 1) * sizeof(uint8_t));
    }

    this->base_array = (uint8_t*)malloc(array_size * sizeof(uint8_t));
    memcpy(base_array, array, array_size * sizeof(array[0]));
    this->qtd_permutations = fact;

    makePermutations(base_array, l, array_size - 1);
} 
AllPermutation::~AllPermutation() 
{ 
    if(arrays != nullptr){
        int i = 0;
        for(i = 0; i < qtd_permutations; i++){
            if(arrays[i] != nullptr){
                free(arrays[i]);
            }
        }
        free(arrays);
    }
    if(base_array != nullptr)
        free(base_array);
} 

bool AllPermutation::hasNext(){
    return curr_index != qtd_permutations;
}

uint8_t* AllPermutation::getNext(){
    return arrays[curr_index++];
}
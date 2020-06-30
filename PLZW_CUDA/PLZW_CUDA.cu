#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
//#include<cuda.h>
#include "..\PLZW_Serial\lzw.h"
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"


__global__ void cuda_hello() {
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<< 1, 1 >>> ();
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include "..\PLZW_Serial\lzw.h"
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"
#define DEFAULT_NPROCS 4


__global__ void encoding(char *input, unsigned int *inputLength, unsigned int *encodedData) {
    short tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("tid = %d\n", tid);
}


using namespace std;
int main()
{
    cudaDeviceProp prop;
    int count, sharedMem_MAX;
    cudaGetDeviceCount(&count);
    if (count > 0) {
        cudaGetDeviceProperties(&prop, 0); // getting first device props
        sharedMem_MAX = prop.sharedMemPerBlock; // 49152 bytes per block for GTX 1070 (capability 6.1)
    }
    else {
        cout << "No device detected" << endl;
        exit(1);
    }

    string input;
    string line;
    ifstream inFile;
    int nProcs;

    nProcs = DEFAULT_NPROCS;
    bool correctness = true;
    inFile.open(IN_PATH);
    if (!inFile) {
        cout << "Unable to open file";
        exit(1);
    }
    while (inFile >> line) {
        input += line;
    }
    inFile.close();

    unsigned int inputLength = input.length();
    unsigned int *dev_encodedData, * dev_inputLength, *encodedData = new unsigned int[inputLength];
    std::chrono::steady_clock::time_point encoding_begin, encoding_end, decoding_begin, decoding_end;

    encoding_begin = std::chrono::steady_clock::now();
    const char *input_point = input.c_str();
    char* dev_input;

    cudaMalloc((void**)&dev_input, inputLength * sizeof(char));
    cudaMalloc((void**)&dev_inputLength, sizeof(unsigned int));
    cudaMalloc((void**)&dev_encodedData, inputLength * sizeof(char));
    cudaMemcpy(dev_input, input_point, inputLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inputLength, &inputLength, sizeof(unsigned int), cudaMemcpyHostToDevice);

    encoding<<< 4, 1 >>>(dev_input, dev_inputLength, dev_encodedData);

    cudaMemcpy(encodedData, dev_input, inputLength * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_encodedData);

    unsigned int encodedLength = encoding_lzw(input_point, input.length(), encodedData);
    encoding_end = std::chrono::steady_clock::now();

    encodedData = (unsigned int*)realloc(encodedData, (encodedLength) * sizeof(unsigned int));
    //for (unsigned int j = 0; j < encodedLength; j++) {
    //    cout << encodedData[j] << " ";
    //}

    decoding_begin = std::chrono::steady_clock::now();
    string decodedData = decoding_lzw(encodedData, encodedLength);
    decoding_end = std::chrono::steady_clock::now();

    // cout << decodedData << "\n\n";

    if (inputLength == decodedData.length()) {
        for (unsigned int j = 0; j < inputLength; j++) {
            correctness = input[j] == decodedData[j];
            if (correctness == 0) {
                break;
            }
        }
    }
    else {
        correctness = 0;
    }

    cout << "Lossless propriety: " << correctness;

    cout <<
        "\nChars: " << inputLength << "  Memory: " << inputLength * sizeof(char) << " bytes" <<
        "\nEncoded: " << encodedLength << "  Memory: " << encodedLength * sizeof(unsigned int) << " bytes" << endl;


    cout << "Encoding time: " << std::chrono::duration_cast<std::chrono::milliseconds> (encoding_end - encoding_begin).count() << "[ms]" << std::endl;
    cout << "Decoding time: " << std::chrono::duration_cast<std::chrono::milliseconds> (decoding_end - decoding_begin).count() << "[ms]" << std::endl;

    delete[] encodedData;
    return 0;
}

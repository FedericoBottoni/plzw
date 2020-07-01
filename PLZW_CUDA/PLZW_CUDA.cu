#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <limits.h>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"
#define ALPHABET_LEN 256
#define DEFAULT_NBLOCKS 4
#define MAX_TOKEN_SIZE 1000

struct unordered_map_node
{
    char* token;
    unsigned int code;
    struct unordered_map_node* next;
};

struct unordered_map_node* unordered_map_head = NULL;

void unordered_map_push(char* token, int code, unsigned int tokenLength) {
    struct unordered_map_node* link = (struct unordered_map_node*)malloc(sizeof(struct unordered_map_node));
    char* token_pointer = (char*)malloc(tokenLength * sizeof(char));
    memcpy(token_pointer, token, tokenLength);
    link->token = token_pointer;
    link->code = code;
    link->next = unordered_map_head;
    //printf("a %s\n", link);
    unordered_map_head = link;
    //printf("b %s\n", unordered_map_head);
}

unsigned int getCodeFromMap(char* token, int tokenLength) {
    struct unordered_map_node* ptr = unordered_map_head;

    while (ptr != NULL) {
        bool equals = true;
        for (unsigned int i = 0; i < tokenLength; i++) {
            char currentTokenChar = ptr->token[i];
            if (currentTokenChar != token[i]) {
                equals = false;
                break;
            }
        }
        if (equals) {
            return ptr->code;
        }
        else {
            ptr = ptr->next;
        }
    }
    return UINT_MAX;
}

char* getTokenFromMap(unsigned int code) {
    struct unordered_map_node* ptr = unordered_map_head;

    while (ptr != NULL) {
        bool equals = true;
        if (ptr->code == code) {
            return ptr->token;
        }
        else {
            ptr = ptr->next;
        }
    }
    return NULL;
}

bool isTokenInMap(char* token, int tokenLength) {
    unsigned int code = getCodeFromMap(token, tokenLength);
    return code != UINT_MAX;
}

bool isCodeInMap(unsigned int code) {
    char* token = getTokenFromMap(code);
    return token != NULL;
}

int encoding_lzw(const char* s1, unsigned int count, unsigned int* objectCode)
{
    int mapLength = ALPHABET_LEN;
    char* ch;
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch = new char[1];
        ch[0] = char(i);
        unordered_map_push(ch, i, 1);
    }
    delete[] ch;

    int out_index = 0, pLength;
    char *p = new char[MAX_TOKEN_SIZE], * pandc = new char[MAX_TOKEN_SIZE], *c = new char[1];
    p[0] = s1[0];
    pLength = 1;
    unsigned int code = ALPHABET_LEN;
    unsigned int i;
    for (i = 0; i < count; i++) {
        if (i != count - 1)
            c[0] = s1[i + 1];
        pandc = strncpy(pandc, p, pLength);
        pandc[pLength] = c[0];
        if (isTokenInMap(pandc, pLength + 1)) {
            strcpy(p, pandc);
            pLength++;
        }
        else {
            objectCode[out_index++] = getCodeFromMap(p, pLength);
            unordered_map_push(pandc, code, pLength + 1);
            code++;
            memset(p, 0, sizeof(p));
            p[0] = c[0];
            pLength = 1;
        }
        memset(c, 0, sizeof(c));
        memset(pandc, 0, sizeof(pandc));
    }
    objectCode[out_index] = getCodeFromMap(p, pLength);
    return out_index;
}


__global__ void encoding(char *input, unsigned int *inputLength, unsigned int *encodedData, unsigned int* nBlocks) {
    unsigned int block = blockIdx.x;
    char thid = threadIdx.x;

    //extern __shared__ unsigned int *cache_shared[];
    //unsigned int* cacheStart = cache_shared[0], *cacheEnd = cache_shared[1], *cache = cache_shared[2];

    //printf("tid = %d\n", thid);
}

using namespace std;
int main()
{
    cudaDeviceProp prop;
    int count, sharedMem_MAX, nBlocks_MAX;
    cudaGetDeviceCount(&count);
    if (count > 0) {
        cudaGetDeviceProperties(&prop, 0); // getting first device props
        sharedMem_MAX = prop.sharedMemPerBlock; // 49152 bytes per block for GTX 1070 (capability 6.1)
        nBlocks_MAX = prop.maxGridSize[0]; // 2147483647 blocks for GTX 1070 (capability 6.1)
    }
    else {
        cout << "No device detected" << endl;
        exit(1);
    }

    string input;
    string line;
    ifstream inFile;
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

    unsigned int nBlocks, inputLength = input.length();
    unsigned int *dev_encodedData, *dev_inputLength, *dev_nBlocks, *encodedData = new unsigned int[inputLength];
    char* dev_input;
    std::chrono::steady_clock::time_point encoding_begin, encoding_end, decoding_begin, decoding_end;

    encoding_begin = std::chrono::steady_clock::now();
    const char *input_point = input.c_str();
    nBlocks = DEFAULT_NBLOCKS;

    cudaMalloc((void**)&dev_input, inputLength * sizeof(char));
    cudaMalloc((void**)&dev_inputLength, sizeof(unsigned int));
    cudaMalloc((void**)&dev_encodedData, inputLength * sizeof(char));
    cudaMalloc((void**)&dev_nBlocks, sizeof(unsigned int));
    cudaMemcpy(dev_input, input_point, inputLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inputLength, &inputLength, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_nBlocks, &nBlocks, sizeof(unsigned int), cudaMemcpyHostToDevice);

    encoding<<< 3, nBlocks >>>(dev_input, dev_inputLength, dev_encodedData, dev_nBlocks);

    cudaMemcpy(encodedData, dev_input, inputLength * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_inputLength);
    cudaFree(dev_encodedData);
    cudaFree(dev_nBlocks);

    unsigned int encodedLength = encoding_lzw(input_point, input.length(), encodedData);
    encoding_end = std::chrono::steady_clock::now();

    encodedData = (unsigned int*)realloc(encodedData, (encodedLength) * sizeof(unsigned int));
    //for (unsigned int j = 0; j < encodedLength; j++) {
    //    cout << encodedData[j] << " ";
    //}

    decoding_begin = std::chrono::steady_clock::now();
    string decodedData = ""; //decoding_lzw(encodedData, encodedLength);
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

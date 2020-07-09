#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <limits.h>
#include <string>
#include <cuda_runtime.h>
#include "../dependencies/uthash.h"
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"
#define ALPHABET_LEN 256
#define DEFAULT_NBLOCKS 4

using namespace std;

struct unsorted_node_map {
    char* id; // key
    unsigned int code;
    short tokenSize;
    UT_hash_handle hh; /* makes this structure hashable */
};

struct unsorted_node_map_dec {
    unsigned int id; // key
    char* token;
    short tokenSize;
    UT_hash_handle hh; /* makes this structure hashable */
};

void push_into_table(unsorted_node_map* table, char* id, short tokenSize, unsigned int code) {
    struct unsorted_node_map* s = (struct unsorted_node_map*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, id, tokenSize);
    curr_token[tokenSize] = '\0';
    s->id = curr_token;
    s->tokenSize = tokenSize;
    s->code = code;
    HASH_ADD_KEYPTR(hh, table, s->id, tokenSize, s);
}

struct unsorted_node_map* find_by_token(unsorted_node_map* table, char* id, short length) {
    struct unsorted_node_map* s;
    id[length] = '\0';
    HASH_FIND_STR(table, id, s);
    return s;
}

struct unsorted_node_map* find_by_code(unsorted_node_map* table, unsigned int code) {
    struct unsorted_node_map* node, * tmp;
    HASH_ITER(hh, table, node, tmp) {
        if (node->code == code) {
            return node;
        }
    }
    return NULL;
}

void dispose_table(unsorted_node_map* table) {
    struct unsorted_node_map* node, * tmp;

    HASH_ITER(hh, table, node, tmp) {
        free(node->id);
        HASH_DEL(table, node);
        free(node);
    }
}

void push_into_table_dec(unsorted_node_map_dec* table, unsigned int id, char* token, short tokenSize) {
    struct unsorted_node_map_dec* s = (struct unsorted_node_map_dec*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, token, tokenSize);
    curr_token[tokenSize] = '\0';
    s->token = curr_token;
    s->tokenSize = tokenSize;
    s->id = id;
    HASH_ADD_INT(table, id, s);
}

struct unsorted_node_map_dec* find_by_code_dec(unsorted_node_map_dec* table, unsigned int id) {
    struct unsorted_node_map_dec* s;
    HASH_FIND_INT(table, &id, s);
    return s;
}

struct unsorted_node_map_dec* find_by_token_dec(unsorted_node_map_dec* table, char* token, short tokenSize) {
    struct unsorted_node_map_dec* node, * tmp;
    bool equals;
    HASH_ITER(hh, table, node, tmp) {
        equals = true;
        if (tokenSize != node->tokenSize) {
            equals = false;
        }
        else {
            for (short i = 0; i < tokenSize; i++) {
                if (token[i] != node->token[i]) {
                    equals = false;
                    break;
                }
            }
        }
        if (equals) {
            return node;
        }
    }
    return NULL;
}

void dispose_table_dec(unsorted_node_map_dec* table) {
    struct unsorted_node_map_dec* node, * tmp;

    HASH_ITER(hh, table, node, tmp) {
        free(node->token);
        HASH_DEL(table, node);
        free(node);
    }
}

int encoding_lzw(const char* s1, unsigned int count, unsigned int* objectCode)
{
    unsorted_node_map* table = NULL;

    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 1; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        push_into_table(table, ch, 1, i);
    }
    free(ch);

    unsorted_node_map* node;
    int out_index = 0, pLength;
    char* p = (char*)malloc(2 * sizeof(char)), * pandc = (char*)malloc(3 * sizeof(char)), * c = new char[1];
    p[0] = s1[0];
    p[1] = '\0';
    pandc[2] = '\0';
    pLength = 1;
    unsigned int code = ALPHABET_LEN;
    unsigned int i;
    for (i = 0; i < count; i++) {
        if (i != count - 1)
            c[0] = s1[i + 1];
        for (unsigned short str_i = 0; str_i < pLength; str_i++) pandc[str_i] = p[str_i];
        pandc[pLength] = c[0];
        unsorted_node_map* node = find_by_token(table, pandc, pLength + 1);
        if (node != NULL) {
            p = (char*)realloc(p, (++pLength + 1) * sizeof(char));
            pandc = (char*)realloc(pandc, (pLength + 2) * sizeof(char));
            p[pLength] = '\0';
            pandc[pLength + 1] = '\0';
            for (unsigned short str_i = 0; str_i < pLength; str_i++) p[str_i] = pandc[str_i];
        }
        else {
            node = find_by_token(table, p, pLength);
            objectCode[out_index++] = node->code;
            push_into_table(table, pandc, pLength + 1, code);
            code++;
            p[0] = c[0];
            if (pLength > 1) {
                p = (char*)realloc(p, 2 * sizeof(char));
                pandc = (char*)realloc(pandc, 3 * sizeof(char));
                p[1] = '\0';
                pandc[2] = '\0';
            }
            pLength = 1;
        }
        c[0] = NULL;
    }
    objectCode[out_index++] = find_by_token(table, p, pLength)->code;

    free(p);
    free(pandc);
    dispose_table(table);
    return out_index;
}

unsigned int decoding_lzw(unsigned int* op, int op_length, char* decodedData)
{
    struct unsorted_node_map_dec* table = NULL;

    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 1; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        push_into_table_dec(table, i, ch, 1);
    }
    free(ch);

    unsigned int old = op[0], decodedDataLength, n;
    struct unsorted_node_map_dec* temp_node, * s_node = find_by_code_dec(table, old);
    int temp_length = 0, s_length = s_node->tokenSize;
    char* s = (char*)malloc((s_length + 1) * sizeof(char)), * temp = (char*)malloc(sizeof(char));
    memcpy(s, s_node->token, s_length);
    s[s_length] = '\0';
    temp[0] = '\0';
    char c = s[0];

    memcpy(decodedData, s, s_length);
    decodedDataLength = 1;
    int count = ALPHABET_LEN;
    for (int i = 0; i < op_length - 1; i++) {
        n = op[i + 1];
        if (find_by_token_dec(table, s, s_length) == NULL) {
            s_node = find_by_code_dec(table, old);
            s_length = s_node->tokenSize;
            s = (char*)realloc(s, (++s_length + 1) * sizeof(char));
            memcpy(s, s_node->token, s_length - 1);
            s[s_length - 1] = c;
            s[s_length] = '\0';
        }
        else {
            s_node = find_by_code_dec(table, n);
            if (s_node->tokenSize != s_length) {
                s_length = s_node->tokenSize;
                s = (char*)realloc(s, (s_length + 1) * sizeof(char));
                s[s_length] = '\0';
            }
            memcpy(s, s_node->token, s_length);
        }
        memcpy(&decodedData[decodedDataLength], s, s_length);
        decodedDataLength += s_length;
        c = s[0];
        temp_node = find_by_code_dec(table, old);
        if (temp_length != temp_node->tokenSize + 1) {
            temp_length = temp_node->tokenSize + 1;
            temp = (char*)realloc(temp, (temp_length + 1) * sizeof(char));
            temp[temp_length] = '\0';
        }
        memcpy(temp, temp_node->token, temp_length - 1);
        temp[temp_length - 1] = c;
        push_into_table_dec(table, count, temp, temp_length);
        count++;
        old = n;
    }
    free(temp);
    free(s);
    dispose_table_dec(table);
    return decodedDataLength;
}



__global__ void encoding(char *input, unsigned int *inputLength, unsigned int *encodedData, unsigned int* nBlocks) {
    unsigned int block = blockIdx.x;
    char thid = threadIdx.x;

    //extern __shared__ unsigned int *cache_shared[];
    //unsigned int* cacheStart = cache_shared[0], *cacheEnd = cache_shared[1], *cache = cache_shared[2];

    //printf("tid = %d\n", thid);
}


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

    unsigned int nBlocks, inputLength = input.length(), inputSize = inputLength * sizeof(char);
    unsigned int *dev_encodedData, *dev_inputLength, *dev_nBlocks;
    unsigned int* encodedData = (unsigned int*)malloc(inputLength * sizeof(unsigned int));
    char* dev_input;
    std::chrono::steady_clock::time_point encoding_begin, encoding_end, decoding_begin, decoding_end;

    encoding_begin = std::chrono::steady_clock::now();
    const char *input_point = input.c_str();
    nBlocks = DEFAULT_NBLOCKS;
    /*
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
    */
    unsigned int encodedLength = encoding_lzw(input_point, input.length(), encodedData);
    encoding_end = std::chrono::steady_clock::now();

    encodedData = (unsigned int*)realloc(encodedData, (encodedLength) * sizeof(unsigned int));
    char* decodedData = (char*)malloc(inputSize);
    //for (unsigned int j = 0; j < encodedLength; j++) {
    //    cout << encodedData[j] << " ";
    //}

    decoding_begin = std::chrono::steady_clock::now();
    unsigned int decodedDataLength = decoding_lzw(encodedData, encodedLength, decodedData);
    decoding_end = std::chrono::steady_clock::now();

    //cout << decodedData << "\n\n";

    if (inputLength == decodedDataLength) {
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

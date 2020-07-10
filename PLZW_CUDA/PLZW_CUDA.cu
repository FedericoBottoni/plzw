#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <limits.h>
#include <string>
#include <cuda_runtime.h>
#include "uthashgpu.h"
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"
#define ALPHABET_LEN 256
#define DEFAULT_NBLOCKS 4
#define SHAREDMEM_MAX 64
#define MAX_TOKEN_SIZE 1000
#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) cerr << "CUDA Error: " << \
    cudaGetErrorString(XXX) << ", at line " << __LINE__ \
    << endl; cudaDeviceSynchronize(); } while (0)


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

__device__ unsorted_node_map* push_into_table(unsorted_node_map* table, char* id, short tokenSize, unsigned int code) {
    struct unsorted_node_map* s = (struct unsorted_node_map*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, id, tokenSize);
    curr_token[tokenSize] = '\0';
    s->id = curr_token;
    s->tokenSize = tokenSize;
    s->code = code;
    HASH_ADD_KEYPTR(hh, table, s->id, tokenSize, s);
    return table;
}

__device__ struct unsorted_node_map* find_by_token(unsorted_node_map* table, char* id, short length) {
    struct unsorted_node_map* s;
    id[length] = '\0';
    HASH_FIND_STR(table, id, s);
    return s;
}

__device__ struct unsorted_node_map* find_by_code(unsorted_node_map* table, unsigned int code) {
    struct unsorted_node_map* node, * tmp;
    HASH_ITER(hh, table, node, tmp) {
        if (node->code == code) {
            return node;
        }
    }
    return NULL;
}

__device__ void dispose_table(unsorted_node_map* table) {
    struct unsorted_node_map* node, * tmp;

    HASH_ITER(hh, table, node, tmp) {
        free(node->id);
        HASH_DEL(table, node);
        free(node);
    }
}

__device__ struct unsorted_node_map_dec* push_into_table_dec(unsorted_node_map_dec* table, unsigned int id, char* token, short tokenSize) {
    struct unsorted_node_map_dec* s = (struct unsorted_node_map_dec*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, token, tokenSize);
    curr_token[tokenSize] = '\0';
    s->token = curr_token;
    s->tokenSize = tokenSize;
    s->id = id;
    HASH_ADD_INT(table, id, s);
    return table;
}

__device__ struct unsorted_node_map_dec* find_by_code_dec(unsorted_node_map_dec* table, unsigned int id) {
    struct unsorted_node_map_dec* s;
    HASH_FIND_INT(table, &id, s);
    return s;
}

__device__ void dispose_table_dec(unsorted_node_map_dec* table) {
    struct unsorted_node_map_dec* node, * tmp;

    HASH_ITER(hh, table, node, tmp) {
        free(node->token);
        HASH_DEL(table, node);
        free(node);
    }
}

__device__ void loadEncodingCache(char* cache, unsigned int stripCacheLength, unsigned int cacheOffset, const char* globalItems, unsigned int globalItemsLength, unsigned int* nThreads, unsigned int thid) {
    unsigned int nitems = stripCacheLength * (cacheOffset + 1) <= globalItemsLength ? stripCacheLength : globalItemsLength - stripCacheLength * cacheOffset;
    //printf("th=%d --- nitems=%d\n", thid, nitems);
    for (unsigned int i = 0; i < nitems; i++) {
        cache[stripCacheLength * thid + i] = globalItems[cacheOffset * *nThreads * stripCacheLength + *nThreads * i + thid];
        //printf("th=%d --- cache[%d] = s1[%d]\n", thid, stripCacheLength * thid + i, cacheOffset * *nThreads * stripCacheLength + *nThreads * i + thid);
    }
    __syncthreads();
}

__device__ void loadDecodingCache(unsigned int* cache, unsigned int stripCacheLength, unsigned int cacheOffset, unsigned int* globalItems, unsigned int globalItemsLength, unsigned int thid) {
    unsigned int nitems = stripCacheLength * (cacheOffset + 1) <= globalItemsLength ? stripCacheLength : globalItemsLength - stripCacheLength * cacheOffset;
    //printf("th=%d --- nitems=%d\n", thid, nitems);
    for (unsigned int i = 0; i < nitems; i++) {
        cache[stripCacheLength * thid + i] = globalItems[cacheOffset * stripCacheLength + i ];
        //printf("th=%d --- cache[%d] = s1[%d]\n", thid, stripCacheLength * thid + i, cacheOffset * stripCacheLength + i);
    }
    __syncthreads();
}

__device__ int encoding_lzw(const char* s1, unsigned int count, unsigned int* objectCode, unsigned int avgRng, unsigned int* nThreads, unsigned int thid, char* cache, unsigned int stripCacheLength)
{
    struct unsorted_node_map* table;
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 1; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        // printf("tbl: %s %d\n", ch, i);
        table = push_into_table(table, ch, 1, i);
    }
    /*
    struct unsorted_node_map* noden1, * tmp1;
    HASH_ITER(hh, table, noden1, tmp1) {
        printf("tbl: %s %d\n", noden1->id, noden1->code);
    }*/
    free(ch);
    unsorted_node_map* node;
    int out_index = 0, pLength;
    char* p = (char*)malloc(MAX_TOKEN_SIZE * sizeof(char)), * pandc = (char*)malloc((MAX_TOKEN_SIZE + 1) * sizeof(char)), * c = new char[1];
    loadEncodingCache(cache, stripCacheLength, 0, s1, count, nThreads, thid);
    p[0] = cache[thid* stripCacheLength];
    p[1] = '\0';
    pandc[2] = '\0';
    pLength = 1;
    unsigned int code = ALPHABET_LEN, cacheOffset = 1, cacheIndex, nextCacheIndex;
    for (unsigned int i = 0; i < count; i++) {
        cacheIndex = i % stripCacheLength;
        nextCacheIndex = (i+1) % stripCacheLength;
        if (cacheIndex == stripCacheLength - 1) {
            loadEncodingCache(cache, stripCacheLength, cacheOffset, s1, count, nThreads, thid);
            cacheOffset++;
        }
        //printf("th=%d index=%d cacheIdx=%d cacheOffset=%d\n", thid, i, stripCacheLength * thid + cacheIndex, cacheOffset);
        if (i != count - 1) {
            c[0] = cache[stripCacheLength * thid + nextCacheIndex];
            //printf("th=%d i=%d accessing cache[%d] = %d\n", thid, i, stripCacheLength * thid + nextCacheIndex, cache[stripCacheLength * thid + nextCacheIndex]);
        }
        for (unsigned short str_i = 0; str_i < pLength; str_i++) pandc[str_i] = p[str_i];
        pandc[pLength] = c[0];
        unsorted_node_map* node = find_by_token(table, pandc, pLength + 1);
        //printf("%d %d, FINO QUI\n", thid, i);
        if (node != NULL) {
            p[++pLength] = '\0';
            pandc[pLength + 1] = '\0';
            for (unsigned short str_i = 0; str_i < pLength; str_i++) p[str_i] = pandc[str_i];
        }
        else {
            node = find_by_token(table, p, pLength);
            objectCode[out_index++] = node->code;
            table = push_into_table(table, pandc, pLength + 1, code);
            code++;
            p[0] = c[0];
            if (pLength > 1) {
                p[1] = '\0';
                pandc[2] = '\0';
            }
            pLength = 1;
        }
        c[0] = NULL;
    }
    objectCode[out_index++] = find_by_token(table, p, pLength)->code;
    /*
    struct unsorted_node_map* noden, * tmp;

    HASH_ITER(hh, table, noden, tmp) {
        if(noden->code > 255)
            printf("tbl: %s %d\n", noden->id, noden->code);
    }*/
    free(p);
    free(pandc);
    dispose_table(table);
    return out_index;
}
__device__ unsigned int decoding_lzw(unsigned int* op, char* decodedData, unsigned int* encodedBuffLengths, unsigned int* nThreads, unsigned int thid, unsigned int* cache, unsigned int stripCacheLength)
{
    struct unsorted_node_map_dec* table;
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 1; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        table = push_into_table_dec(table, i, ch, 1);
    }
    free(ch);

    unsigned int old, decodedDataLength, n, cacheIndex, nextCacheIndex, cacheOffset;
    loadDecodingCache(cache, stripCacheLength, 0, op, encodedBuffLengths[thid], thid);
    old = cache[stripCacheLength * thid];
    cacheOffset = 1;
    struct unsorted_node_map_dec* temp_node, * s_node = find_by_code_dec(table, old);
    int temp_length = 0, s_length = s_node->tokenSize;
    char* s = (char*)malloc(MAX_TOKEN_SIZE * sizeof(char)), * temp = (char*)malloc(sizeof(char));
    memcpy(s, s_node->token, s_length);
    s[s_length] = '\0';
    temp[0] = '\0';
    char c = s[0];

    memcpy(decodedData, s, s_length);
    decodedDataLength = 1;
    int count = ALPHABET_LEN;
    for (int i = 0; i < encodedBuffLengths[thid] - 1; i++) {
        cacheIndex = i % stripCacheLength;
        nextCacheIndex = (i + 1) % stripCacheLength;
        if (cacheIndex == stripCacheLength - 1) {
            loadDecodingCache(cache, stripCacheLength, cacheOffset, op, encodedBuffLengths[thid], thid);
            cacheOffset++;
        }
        n = cache[stripCacheLength * thid + nextCacheIndex];
        //printf("th=%d i=%d accessing cache[%d] = %d\n", thid, i, stripCacheLength * thid + nextCacheIndex, cache[stripCacheLength * thid + nextCacheIndex]);
        if (find_by_code_dec(table, n) == NULL) {
            s_node = find_by_code_dec(table, old);
            s_length = s_node->tokenSize;
            s_length++;
            memcpy(s, s_node->token, s_length - 1);
            s[s_length - 1] = c;
            s[s_length] = '\0';
        }
        else {
            s_node = find_by_code_dec(table, n);
            if (s_node->tokenSize != s_length) {
                s_length = s_node->tokenSize;
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
            temp[temp_length] = '\0';
        }
        memcpy(temp, temp_node->token, temp_length - 1);
        temp[temp_length - 1] = c;
        table = push_into_table_dec(table, count, temp, temp_length);
        count++;
        old = n;
    }
    free(temp);
    free(s);
    dispose_table_dec(table);
    return decodedDataLength;
}

__global__ void encoding(char *input, unsigned int *inputLength, unsigned int *encodedData, unsigned int* encodedBuffLengths, unsigned int* nThreads) {
    unsigned int sharedItems_MAX, thid = threadIdx.x + blockIdx.x * blockDim.x;
    sharedItems_MAX = SHAREDMEM_MAX; //*sharedMem_MAX / (sizeof(char));
    extern __shared__ char enc_cache[SHAREDMEM_MAX];
    sharedItems_MAX /= *nThreads;

    unsigned int encodedLength, dataBuffLength, * encodedBuff,
        avgRng = __double2uint_ru((double)(*inputLength) / (double)(*nThreads)), avgRngRest = *inputLength % *nThreads;
    dataBuffLength = avgRngRest == 0 || thid < avgRngRest ? avgRng : avgRng - 1;

    unsigned int encOffset = 0,* encodedDataBuff = new unsigned int[dataBuffLength];

    encodedBuffLengths[thid] = encoding_lzw(input, dataBuffLength, encodedDataBuff, avgRng, nThreads, thid, enc_cache, sharedItems_MAX);
    __syncthreads();

    for (unsigned int i = 0; i < thid; i++) {
        encOffset += encodedBuffLengths[i];
    }
    for (unsigned int i = 0; i < encodedBuffLengths[thid]; i++) {
        encodedData[encOffset + i] = encodedDataBuff[i];
        //printf("th%d i=%d  %d\n", thid, i, encodedDataBuff[i]);
    }

}

__global__ void decoding(unsigned int* encodedData, unsigned int* encodedBuffLengths, unsigned int* inputLength, char* decodedData, unsigned int* decodedBuffLengths, unsigned int* nThreads) {
    unsigned int sharedItems_MAX, thid = threadIdx.x + blockIdx.x * blockDim.x;
    sharedItems_MAX = SHAREDMEM_MAX; //*sharedMem_MAX / (sizeof(char));

    extern __shared__ unsigned int dec_cache[SHAREDMEM_MAX];
    sharedItems_MAX /= *nThreads;

    unsigned int encodedLength = 0, dataBuffLength, *encodedBuff, encodedOffset = 0,
        avgRng = __double2uint_ru((double)(*inputLength) / (double)(*nThreads)), avgRngRest = *inputLength % *nThreads;
    dataBuffLength = avgRngRest == 0 || thid < avgRngRest ? avgRng : avgRng - 1;

    char* decodedDataBuff = new char[dataBuffLength];

    for (unsigned int i = 0; i < thid; i++) {
        encodedOffset += encodedBuffLengths[i];
    }

    decodedBuffLengths[thid] = decoding_lzw(&encodedData[encodedOffset], decodedDataBuff, encodedBuffLengths, nThreads, thid, dec_cache, sharedItems_MAX);
    __syncthreads();

    for (unsigned int i = 0; i < dataBuffLength; i++) {
        decodedData[*nThreads * i + thid] = decodedDataBuff[i];
        //printf("th%d i=%d  %d\n", thid, i, decodedDataBuff[i]);
    }

    //delete[] decodedDataBuff;
}

int main()
{
    cudaDeviceProp prop;
    int count, sharedMem_MAX, nBlocks_MAX, warpSize;
    cudaGetDeviceCount(&count);
    if (count > 0) {
        cudaGetDeviceProperties(&prop, 0); // getting first device props
        sharedMem_MAX = prop.sharedMemPerBlock; // 49152 bytes per block for GTX 1070 (capability 6.1)
        nBlocks_MAX = prop.maxGridSize[0]; // 2147483647 blocks for GTX 1070 (capability 6.1)
        warpSize = 3; //prop.warpSize;
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

    unsigned int nBlocks, nThreads, inputLength = input.length(), inputSize = inputLength * sizeof(char),
        *dev_encodedData, *dev_inputLength, *dev_encodedBuffLengths, *dev_nThreads, *dev_sharedMem_MAX,
        *encodedData = (unsigned int*)malloc(inputLength * sizeof(unsigned int)), encodedLength = 0,
        *encodedBuffLengths;
    char* dev_input;
    const char* input_point = input.c_str();

    std::chrono::steady_clock::time_point encoding_begin, encoding_end, decoding_begin, decoding_end;
    encoding_begin = std::chrono::steady_clock::now();
    nBlocks = 1; // DEFAULT_NBLOCKS;
    nThreads = nBlocks * warpSize;
    encodedBuffLengths = (unsigned int*)malloc(nThreads * sizeof(unsigned int));
    
    CUDA_WARN(cudaMalloc((void**)&dev_input, inputLength * sizeof(char)));
    CUDA_WARN(cudaMalloc((void**)&dev_inputLength, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_nThreads, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_encodedData, inputLength * sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_encodedBuffLengths, nThreads * sizeof(unsigned int)));

    CUDA_WARN(cudaMemcpy(dev_input, input_point, inputLength * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_inputLength, &inputLength, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_nThreads, &nThreads, sizeof(unsigned int), cudaMemcpyHostToDevice));

    encoding<<< nBlocks, warpSize >>>(dev_input, dev_inputLength, dev_encodedData, dev_encodedBuffLengths, dev_nThreads);

    CUDA_WARN(cudaMemcpy(encodedData, dev_encodedData, inputLength * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_WARN(cudaMemcpy(encodedBuffLengths, dev_encodedBuffLengths, nThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < nThreads; i++) {
        encodedLength += encodedBuffLengths[i];
    }

    CUDA_WARN(cudaFree(dev_input));
    CUDA_WARN(cudaFree(dev_inputLength));
    CUDA_WARN(cudaFree(dev_encodedData));
    CUDA_WARN(cudaFree(dev_encodedBuffLengths));
    CUDA_WARN(cudaFree(dev_nThreads));
    
    encoding_end = std::chrono::steady_clock::now();

    //for (unsigned int i = 0; i < encodedLength; i++) printf("%d ", encodedData[i]);
    printf("\n\n");
    char* decodedData = (char*)malloc(inputSize);
    unsigned int* decodedBuffLengths = (unsigned int*)malloc(nThreads * sizeof(unsigned int));
    decoding_begin = std::chrono::steady_clock::now();
    char* dev_decodedData;
    unsigned int *dev_decodedBuffLengths, decodedDataLength = 0;

    CUDA_WARN(cudaMalloc((void**)&dev_encodedData, encodedLength * sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_encodedBuffLengths, nThreads * sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_inputLength, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_nThreads, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_decodedData, inputSize));
    CUDA_WARN(cudaMalloc((void**)&dev_decodedBuffLengths, nThreads * sizeof(unsigned int)));

    CUDA_WARN(cudaMemcpy(dev_encodedData, encodedData, encodedLength * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_encodedBuffLengths, encodedBuffLengths, nThreads * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_inputLength, &inputLength, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_nThreads, &nThreads, sizeof(unsigned int), cudaMemcpyHostToDevice));

    decoding<<< nBlocks, warpSize >>>(dev_encodedData, dev_encodedBuffLengths, dev_inputLength, dev_decodedData, dev_decodedBuffLengths, dev_nThreads);

    CUDA_WARN(cudaMemcpy(decodedData, dev_decodedData, inputSize, cudaMemcpyDeviceToHost));
    CUDA_WARN(cudaMemcpy(decodedBuffLengths, dev_decodedBuffLengths, nThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < nThreads; i++) {
        decodedDataLength += decodedBuffLengths[i];
    }

    CUDA_WARN(cudaFree(dev_encodedData));
    CUDA_WARN(cudaFree(dev_encodedBuffLengths));
    CUDA_WARN(cudaFree(dev_inputLength));
    CUDA_WARN(cudaFree(dev_nThreads));
    CUDA_WARN(cudaFree(dev_decodedData));
    CUDA_WARN(cudaFree(dev_decodedBuffLengths));

    decoding_end = std::chrono::steady_clock::now();

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

    free(encodedData);
    free(decodedData);
    return 0;
}

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
#define SHAREDMEM_MAX 64
#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) cerr << "CUDA Error: " << \
    cudaGetErrorString(XXX) << ", at line " << __LINE__ \
    << endl; cudaDeviceSynchronize(); } while (0)

using namespace std;
__device__ struct unordered_map_node
{
    char* token;
    short tokenSize;
    unsigned int code;
    struct unordered_map_node* next;
};

__device__ struct unordered_map_node* unordered_map_head = NULL;

__device__ void disposeMap() {
    unordered_map_node* current = unordered_map_head;
    struct unordered_map_node* next;
    while (current != NULL)
    {
        next = current->next;
        free(current->token);
        free(current);
        current = next;
    }
    unordered_map_head = NULL;
}

__device__ void unordered_map_push(char* token, int code, short tokenLength) {
    struct unordered_map_node* link = (struct unordered_map_node*)malloc(sizeof(struct unordered_map_node));
    char* token_pointer = (char*)malloc(tokenLength * sizeof(char));
    memcpy(token_pointer, token, tokenLength);
    link->token = token_pointer;
    link->tokenSize = tokenLength;
    link->code = code;
    link->next = unordered_map_head;
    unordered_map_head = link;
}

__device__ unsigned int getCodeFromMap(char* token, int tokenLength) {
    struct unordered_map_node* ptr = unordered_map_head;
    while (ptr != NULL) {
        bool equals = true;
        if (tokenLength == ptr->tokenSize) {
            for (unsigned int i = 0; i < tokenLength; i++) {
                char currentTokenChar = ptr->token[i];
                if (currentTokenChar != token[i]) {
                    equals = false;
                    break;
                }
            }
        }
        else {
            equals = false;
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

__device__ unordered_map_node* getNodeFromMap(unsigned int code) {
    struct unordered_map_node* ptr = unordered_map_head;

    while (ptr != NULL) {
        bool equals = true;
        if (ptr->code == code) {
            return ptr;
        }
        else {
            ptr = ptr->next;
        }
    }
    return NULL;
}

__device__ bool isTokenInMap(char* token, int tokenLength) {
    unsigned int code = getCodeFromMap(token, tokenLength);
    return code != UINT_MAX;
}

__device__ bool isCodeInMap(unsigned int code) {
    unordered_map_node* token = getNodeFromMap(code);
    return token != NULL;
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
    char* ch;
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch = new char[1];
        ch[0] = char(i);
        unordered_map_push(ch, i, 1);
    }
    delete[] ch;

    int out_index = 0, pLength;
    char *p = new char[MAX_TOKEN_SIZE], * pandc = new char[MAX_TOKEN_SIZE], *c = new char[1];
    loadEncodingCache(cache, stripCacheLength, 0, s1, count, nThreads, thid);
    p[0] = cache[thid* stripCacheLength];
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
        for (unsigned int str_i = 0; str_i < pLength; str_i++) pandc[str_i] = p[str_i];
        pandc[pLength] = c[0];
        if (isTokenInMap(pandc, pLength + 1)) {
            pLength++;
            for (unsigned int str_i = 0; str_i < pLength; str_i++) p[str_i] = pandc[str_i];
        }
        else {
            objectCode[out_index++] = getCodeFromMap(p, pLength);
            unordered_map_push(pandc, code, pLength + 1);
            code++;
            memset(p, 0, sizeof(p));
            p[0] = c[0];
            pLength = 1;
        }
        c[0] = NULL;
        if (pLength > MAX_TOKEN_SIZE - 1) {
            printf("Token-size is not enough big\n");
        }
    }
    objectCode[out_index] = getCodeFromMap(p, pLength);


    delete[] p;
    delete[] pandc;
    //disposeMap();
    
    return out_index;
}
__device__ unsigned int decoding_lzw(unsigned int* op, char* decodedData, unsigned int* encodedBuffLengths, unsigned int* nThreads, unsigned int thid, unsigned int* cache, unsigned int stripCacheLength)
{
    char* ch;
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch = new char[1];
        ch[0] = char(i);
        unordered_map_push(ch, i, 1);
    }
    delete[] ch;

    unsigned int old, decodedDataLength, n, cacheIndex, nextCacheIndex, cacheOffset;
    loadDecodingCache(cache, stripCacheLength, 0, op, encodedBuffLengths[thid], thid);
    old = cache[stripCacheLength * thid];
    cacheOffset = 1;
    unordered_map_node* temp_node, * s_node = getNodeFromMap(old);
    int temp_length, s_length = s_node->tokenSize;
    char* s = new char[MAX_TOKEN_SIZE], *temp = new char[MAX_TOKEN_SIZE];
    memcpy(s, s_node->token, s_length);
    char* c = new char[1];
    c[0] = s[0];
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
        if (!isTokenInMap(s, s_length)) {
            s_node = getNodeFromMap(old);
            s_length = s_node->tokenSize;
            memcpy(s, s_node->token, s_length);
            s[s_length++] = c[0];
        }
        else {
            s_node = getNodeFromMap(n);
            s_length = s_node->tokenSize;
            memcpy(s, s_node->token, s_length);
        }
        if (s_length > MAX_TOKEN_SIZE - 1) {
            printf("Token-size is not enough big");
        }
        memcpy(&decodedData[decodedDataLength], s, s_length);
        decodedDataLength += s_length;
        c[0] = s[0];
        temp_node = getNodeFromMap(old);
        temp_length = temp_node->tokenSize;
        memcpy(temp, temp_node->token, temp_length);
        temp[temp_length] = c[0];
        unordered_map_push(temp, count, temp_length + 1);
        memset(temp, 0, sizeof(temp));
        count++;
        old = n;
    }
    delete[] temp;
    delete[] s;
    //disposeMap();
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
        printf("th%d i=%d  %d\n", thid, i, encodedDataBuff[i]);
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
        printf("th%d i=%d  %d\n", thid, i, decodedDataBuff[i]);
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
    encodedData = (unsigned int*)realloc(encodedData, (encodedLength) * sizeof(unsigned int));
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

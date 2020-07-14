// PLZW_CUDA.exe

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include "../dependencies/uthashgpu.h"
#define IN_PATH "F:\\Dev\\PLZW\\in\\in"
#define ALPHABET_LEN 256
#define THREADS_A_BLOCK 32
#define BLOCKS_GRID 2
#define SHAREDMEM_MAX 256
#define MAX_TOKEN_SIZE 100
#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) cerr << "CUDA Error: " << \
    cudaGetErrorString(XXX) << ", at line " << __LINE__ \
    << endl; cudaDeviceSynchronize(); } while (0)


using namespace std;

// Hashmap struct for encoder
struct unsorted_node_map {
    char* id; // key
    unsigned int code;
    short tokenSize;
    UT_hash_handle hh; /* makes this structure hashable */
};

// Hashmap struct for decoder
struct unsorted_node_map_dec {
    unsigned int id; // key
    char* token;
    short tokenSize;
    UT_hash_handle hh; /* makes this structure hashable */
};

// Push token-code to the existing hashmap
__device__ unsorted_node_map* push_into_table(unsorted_node_map* table, char* id, short tokenSize, unsigned int code) {
    struct unsorted_node_map* s = (struct unsorted_node_map*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, id, sizeof(char)*tokenSize);
    curr_token[tokenSize] = '\0';
    s->id = curr_token;
    s->tokenSize = tokenSize;
    s->code = code;
    HASH_ADD_KEYPTR(hh, table, s->id, tokenSize, s);
    return table;
}

// Hashmap lookup method
__device__ struct unsorted_node_map* find_by_token(unsorted_node_map* table, char* id, short length) {
    struct unsorted_node_map* s;
    id[length] = '\0';
    HASH_FIND_STR(table, id, s);
    return s;
}

// Deallocation of the entire hashmap
__device__ void dispose_table(unsorted_node_map* table) {
    struct unsorted_node_map* node, * tmp;

    HASH_ITER(hh, table, node, tmp) {
        free(node->id);
        HASH_DEL(table, node);
        free(node);
    }
}

// Push token-code to the existing hashmap
__device__ struct unsorted_node_map_dec* push_into_table_dec(unsorted_node_map_dec* table, unsigned int id, char* token, short tokenSize) {
    struct unsorted_node_map_dec* s = (struct unsorted_node_map_dec*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, token, sizeof(char) * tokenSize);
    curr_token[tokenSize] = '\0';
    s->token = curr_token;
    s->tokenSize = tokenSize;
    s->id = id;
    HASH_ADD_INT(table, id, s);
    return table;
}

// Hashmap lookup method
__device__ struct unsorted_node_map_dec* find_by_code_dec(unsorted_node_map_dec* table, unsigned int id) {
    struct unsorted_node_map_dec* s;
    HASH_FIND_INT(table, &id, s);
    return s;
}

// Deallocation of the entire hashmap
__device__ void dispose_table_dec(unsorted_node_map_dec* table) {
    struct unsorted_node_map_dec* node, * tmp;

    HASH_ITER(hh, table, node, tmp) {
        free(node->token);
        HASH_DEL(table, node);
        free(node);
    }
}

// Load cache and provide data to encoder, then synchronize threads
__device__ void loadEncodingCache(char* cache, unsigned int stripCacheLength, unsigned int cacheOffset, const char* globalItems, unsigned int globalItemsLength, unsigned int* nThreads, unsigned int thid, unsigned int thid_block) {
    unsigned int nitems = stripCacheLength * (cacheOffset + 1) <= globalItemsLength ? stripCacheLength : globalItemsLength - stripCacheLength * cacheOffset;
    for (unsigned int i = 0; i < nitems; i++) {
        cache[stripCacheLength * thid_block + i] = globalItems[cacheOffset * *nThreads * stripCacheLength + *nThreads * i + thid];
    }
    __syncthreads();
}

// Load cache and provide data to decoder, then synchronize threads
__device__ void loadDecodingCache(unsigned int* cache, unsigned int stripCacheLength, unsigned int cacheOffset, unsigned int* globalItems, unsigned int globalItemsLength, unsigned int thid, unsigned int thid_block) {
    unsigned int nitems = stripCacheLength * (cacheOffset + 1) <= globalItemsLength ? stripCacheLength : globalItemsLength - stripCacheLength * cacheOffset;
    for (unsigned int i = 0; i < nitems; i++) {
        cache[stripCacheLength * thid_block + i] = globalItems[cacheOffset * stripCacheLength + i ];
    }
    __syncthreads();
}

__device__ unsigned int encoding_lzw(const char* s1, unsigned int count, unsigned int* objectCode, unsigned int avgRng, unsigned int* nThreads, unsigned int thid, unsigned int thid_block, char* cache, unsigned int stripCacheLength)
{
    // Init hashmap with ASCII alphabet
    struct unsorted_node_map* table = NULL;
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        table = push_into_table(table, ch, 1, i);
    }
    free(ch);

    // Definition and assignation of LZW_encoder variables. Vars 'p' and 'pandc' of MAX_TOKEN_SIZE size to avoid
    // the reallocation (not available on devices)
    unsorted_node_map* node;
    unsigned int pLength, cacheOffset = 1, out_index = 0;
    char* p = (char*)malloc(MAX_TOKEN_SIZE * sizeof(char)), * pandc = (char*)malloc((MAX_TOKEN_SIZE + 1) * sizeof(char)), * c = new char[1];

    // Loading cache the first data
    loadEncodingCache(cache, stripCacheLength, 0, s1, count, nThreads, thid, thid_block);
    p[0] = cache[thid_block * stripCacheLength];
    p[1] = '\0';
    pandc[2] = '\0';
    pLength = 1;
    unsigned int code = ALPHABET_LEN, cacheIndex, nextCacheIndex;

    // Iterating through all the input buffer
    for (unsigned int i = 0; i < count; i++) {

        // Calculation of indexes and (if needed) reloading the cache
        cacheIndex = i % stripCacheLength;
        nextCacheIndex = (i+1) % stripCacheLength;
        if (cacheIndex == stripCacheLength - 1) {
            loadEncodingCache(cache, stripCacheLength, cacheOffset, s1, count, nThreads, thid, thid_block);
            cacheOffset++;
        }

        // Gather the next value from cache
        if (i != count - 1) {
            c[0] = cache[stripCacheLength * thid_block + nextCacheIndex];
        }

        // Gather from hashmap 'p_c' concatenated token: if it exists save it and go on with the algorithm; otherwise
        // get just the 'p' value, add it to outcome and add p_c to hashmap
        for (unsigned short str_i = 0; str_i < pLength; str_i++) pandc[str_i] = p[str_i];
        pandc[pLength] = c[0];
        unsorted_node_map* node = find_by_token(table, pandc, pLength + 1);
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

    // Get last value and deallocate
    objectCode[out_index++] = find_by_token(table, p, pLength)->code;
    free(p);
    free(pandc);
    dispose_table(table);
    return out_index;
}
__device__ unsigned int decoding_lzw(unsigned int* op, char* decodedData, unsigned int* encodedBuffLengths, unsigned int* nThreads, unsigned int thid, unsigned int thid_block, unsigned int* cache, unsigned int stripCacheLength)
{
    // Init hashmap with ASCII alphabet
    struct unsorted_node_map_dec* table = NULL;
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        table = push_into_table_dec(table, i, ch, 1);
    }
    free(ch);

    // Loading cache the first data
    unsigned int old, decodedDataLength, n, cacheIndex, nextCacheIndex, cacheOffset;
    loadDecodingCache(cache, stripCacheLength, 0, op, encodedBuffLengths[thid], thid, thid_block);
    cacheOffset = 1;

    // Assignation of "old" variable from cache and s_node from hasmap
    old = cache[stripCacheLength * thid_block];
    struct unsorted_node_map_dec* temp_node, * s_node = find_by_code_dec(table, old);
    int temp_length = 0, s_length = s_node->tokenSize;

    // Definition and assignation of LZW_decoder variables. Var 's' of MAX_TOKEN_SIZE size to avoid the reallocation (not available on devices)
    char* s = (char*)malloc(MAX_TOKEN_SIZE * sizeof(char)), * temp = (char*)malloc(sizeof(char));
    memcpy(s, s_node->token, sizeof(char) * s_length);
    s[s_length] = '\0';
    temp[0] = '\0';
    char c = s[0];
    memcpy(decodedData, s, sizeof(char) * s_length);
    decodedDataLength = 1;
    int count = ALPHABET_LEN;

    // Iterating through all the encoded buffer
    for (int i = 0; i < encodedBuffLengths[thid] - 1; i++) {

        // Calculation of indexes and (if needed) reloading the cache
        cacheIndex = i % stripCacheLength;
        nextCacheIndex = (i + 1) % stripCacheLength;
        if (cacheIndex == stripCacheLength - 1) {
            loadDecodingCache(cache, stripCacheLength, cacheOffset, op, encodedBuffLengths[thid], thid, thid_block);
            cacheOffset++;
        }

        // Gather the next value from cache
        n = cache[stripCacheLength * thid_block + nextCacheIndex];

        // Decoding the old value if the next is new or keep the next one if it's known
        if (find_by_code_dec(table, n) == NULL) {
            s_node = find_by_code_dec(table, old);
            s_length = s_node->tokenSize;
            s_length++;
            memcpy(s, s_node->token, sizeof(char) * (s_length - 1));
            s[s_length - 1] = c;
            s[s_length] = '\0';
        }
        else {
            s_node = find_by_code_dec(table, n);
            if (s_node->tokenSize != s_length) {
                s_length = s_node->tokenSize;
                s[s_length] = '\0';
            }
            memcpy(s, s_node->token, sizeof(char) * s_length);
        }
        
        // Update outcome array
        memcpy(&decodedData[decodedDataLength], s, sizeof(char) * s_length);
        decodedDataLength += s_length;

        // Push the new token to the hashmap
        temp_node = find_by_code_dec(table, old);
        if (temp_length != temp_node->tokenSize + 1) {
            temp_length = temp_node->tokenSize + 1;
            temp[temp_length] = '\0';
        }
        c = s[0];
        memcpy(temp, temp_node->token, sizeof(char) * (temp_length - 1));
        temp[temp_length - 1] = c;
        table = push_into_table_dec(table, count, temp, temp_length);

        // Go on with LZW algorithm
        count++;
        old = n;
    }

    // Deallocation
    free(temp);
    free(s);
    dispose_table_dec(table);
    return decodedDataLength;
}

__global__ void encoding(char *input, unsigned int * avgRng, unsigned int* avgRngRest, unsigned int *encodedData, unsigned int* encodedBuffLengths, unsigned int* nThreads) {
    // Init ids and cache
    unsigned int thid_block = threadIdx.x, thid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ char enc_cache[SHAREDMEM_MAX*THREADS_A_BLOCK];

    // Homogeneous segmentation of input array
    unsigned int encodedLength, dataBuffLength, * encodedBuff;
    dataBuffLength = *avgRngRest == 0 || thid < *avgRngRest ? *avgRng : *avgRng - 1;

    // LZW_encoding call + alignment of encoded array
    encodedBuffLengths[thid] = encoding_lzw(input, dataBuffLength, &encodedData[*avgRng*thid], *avgRng, nThreads, thid, thid_block, enc_cache, SHAREDMEM_MAX);
}

__global__ void decoding(unsigned int* encodedData, unsigned int* encodedBuffLengths, unsigned int* inputLength, char* decodedData, unsigned int* decodedBuffLengths, unsigned int* nThreads) {
    // Init ids and cache
    unsigned int thid_block = threadIdx.x, thid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ unsigned int dec_cache[SHAREDMEM_MAX * THREADS_A_BLOCK];

    // Homogeneous segmentation of decoded array
    unsigned int dataBuffLength, * encodedBuff, encodedLength = 0, encodedOffset = 0,
        avgRng = __double2uint_ru((double)(*inputLength) / (double)(*nThreads)),
        avgRngRest = *inputLength % *nThreads;
    dataBuffLength = avgRngRest == 0 || thid < avgRngRest ? avgRng : avgRng - 1;
    char* decodedDataBuff = (char*)malloc(avgRng * dataBuffLength * sizeof(char));;
    
    // Calculation of the encoded array's alignment
    for (short i = 0; i < thid; i++) {
        encodedOffset += encodedBuffLengths[i];
    }

    // LZW_decoding call + threads' synchronization
    decodedBuffLengths[thid] = decoding_lzw(&encodedData[encodedOffset], decodedDataBuff, encodedBuffLengths, nThreads, thid, thid_block, dec_cache, SHAREDMEM_MAX);
    __syncthreads();

    // Composition of the outcome from buffers
    for (unsigned int i = 0; i < dataBuffLength; i++) {
        decodedData[*nThreads * i + thid] = decodedDataBuff[i];
    }

    // Deallocation of the buffer
    free(decodedDataBuff);
}

__host__ void initGPU() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    if (count > 0) {
        //cudaGetDeviceProperties(&prop, 0);
        // prop.sharedMemPerBlock; // 49152 bytes per block for GTX 1070 (capability 6.1)
        // prop.maxGridSize[0]; // 2147483647 blocks for GTX 1070 (capability 6.1)
        // prop.THREADS_A_BLOCK; // 32

        // Setting the limit of malloc heap size on GPU of 1GB
        CUDA_WARN(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000000000 * sizeof(char)));
    }
    else {
        cout << "No device detected" << endl;
        exit(1);
    }
}

int main()
{
    // GPU check and settings
    initGPU();

    // Read dataset from file
    string line, input;
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

    // Declaration of timers and START
    std::chrono::steady_clock::time_point encoding_begin, encoding_end, decoding_begin, decoding_end;
    encoding_begin = std::chrono::steady_clock::now();

    // Declaration of device variables
    unsigned int* dev_encodedData, * dev_avgRng, * dev_avgRngRest, * dev_encodedBuffLengths, * dev_nThreads, * dev_sharedMem_MAX;
    char* dev_input;

    // Declaration, definition of host variables and assignation of a few of them
    unsigned int nThreads, inputLength, avgRng, avgRngRest, inputSize, encodedLength, * encodedData, * encodedBuffLengths;
    inputLength = input.length();
    inputSize = inputLength * sizeof(char);
    const char* input_point = input.c_str();
    encodedLength = 0;
    nThreads = BLOCKS_GRID * THREADS_A_BLOCK;
    avgRng = ceil((double)(inputLength) / (double)(nThreads)), avgRngRest = inputLength % nThreads;
    encodedData = (unsigned int*)malloc(avgRng * nThreads * sizeof(unsigned int));
    encodedBuffLengths = (unsigned int*)malloc(nThreads * sizeof(unsigned int));
    
    // Definition of device variables
    CUDA_WARN(cudaMalloc((void**)&dev_input, inputLength * sizeof(char)));
    CUDA_WARN(cudaMalloc((void**)&dev_avgRng, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_avgRngRest, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_nThreads, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_encodedData, avgRng * nThreads * sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_encodedBuffLengths, nThreads * sizeof(unsigned int)));

    // Transfer the host variables to the device
    CUDA_WARN(cudaMemcpy(dev_input, input_point, inputLength * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_avgRng, &avgRng, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_avgRngRest, &avgRngRest, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_nThreads, &nThreads, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Kernel encoding call
    encoding<<< BLOCKS_GRID, THREADS_A_BLOCK >>>(dev_input, dev_avgRng, dev_avgRngRest, dev_encodedData, dev_encodedBuffLengths, dev_nThreads);
    // Implicit blocks' synchronization

    // Transfer the device outcomes to the host
    CUDA_WARN(cudaMemcpy(encodedData, dev_encodedData, avgRng * nThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_WARN(cudaMemcpy(encodedBuffLengths, dev_encodedBuffLengths, nThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Fix of the encoded array's alignment and calculation of the encoded array's length
    unsigned int offset = 0;
    for (short i = 0; i < nThreads; i++) {
        memcpy(&encodedData[offset], &encodedData[avgRng * i], sizeof(unsigned int) * encodedBuffLengths[i]);
        offset += encodedBuffLengths[i];
        encodedLength += encodedBuffLengths[i];
    }

    // Deallocation of device variables
    CUDA_WARN(cudaFree(dev_input));
    CUDA_WARN(cudaFree(dev_avgRng));
    CUDA_WARN(cudaFree(dev_avgRngRest));
    CUDA_WARN(cudaFree(dev_encodedData));
    CUDA_WARN(cudaFree(dev_encodedBuffLengths));
    CUDA_WARN(cudaFree(dev_nThreads));

    // Timer STOP
    encoding_end = std::chrono::steady_clock::now();

    // Skipping the export of encoded data and the reimport

    // Timer START
    decoding_begin = std::chrono::steady_clock::now();

    // Declaration and definition of the needed variables (skipping the initialization of input variables)
    char* dev_decodedData, *decodedData;
    unsigned int* decodedBuffLengths, *dev_decodedBuffLengths, decodedDataLength, *dev_inputLength;
    decodedData = (char*)malloc(inputSize);
    decodedBuffLengths = (unsigned int*)malloc(nThreads * sizeof(unsigned int));
    decodedDataLength = 0;

    // Definition of device variables
    CUDA_WARN(cudaMalloc((void**)&dev_encodedData, encodedLength * sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_encodedBuffLengths, nThreads * sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_inputLength, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_nThreads, sizeof(unsigned int)));
    CUDA_WARN(cudaMalloc((void**)&dev_decodedData, inputSize));
    CUDA_WARN(cudaMalloc((void**)&dev_decodedBuffLengths, nThreads * sizeof(unsigned int)));

    // Transfer the host variables to the device
    CUDA_WARN(cudaMemcpy(dev_encodedData, encodedData, encodedLength * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_encodedBuffLengths, encodedBuffLengths, nThreads * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_inputLength, &inputLength, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(dev_nThreads, &nThreads, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Kernel decoding call
    decoding<<< BLOCKS_GRID, THREADS_A_BLOCK >>>(dev_encodedData, dev_encodedBuffLengths, dev_inputLength, dev_decodedData, dev_decodedBuffLengths, dev_nThreads);

    // Transfer the device outcomes to the host
    CUDA_WARN(cudaMemcpy(decodedData, dev_decodedData, inputSize, cudaMemcpyDeviceToHost));
    CUDA_WARN(cudaMemcpy(decodedBuffLengths, dev_decodedBuffLengths, nThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Calculation of the decoded data's length
    for (unsigned int i = 0; i < nThreads; i++) {
        decodedDataLength += decodedBuffLengths[i];
    }

    // Deallocation of device variables
    CUDA_WARN(cudaFree(dev_encodedData));
    CUDA_WARN(cudaFree(dev_encodedBuffLengths));
    CUDA_WARN(cudaFree(dev_inputLength));
    CUDA_WARN(cudaFree(dev_nThreads));
    CUDA_WARN(cudaFree(dev_decodedData));
    CUDA_WARN(cudaFree(dev_decodedBuffLengths));

    // Timer STOP
    decoding_end = std::chrono::steady_clock::now();

    // Checking the correctness of lossless propriety
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

    // Logging the performances
    cout << "Lossless propriety: " << correctness;
    cout <<
        "\nChars: " << inputLength << "  Memory: " << inputLength * sizeof(char) << " bytes (char8)" <<
        "\nEncoded: " << encodedLength << "  Memory: " << encodedLength * sizeof(unsigned int) << " bytes (uint32)" << endl;
    cout << "Encoding time: " << std::chrono::duration_cast<std::chrono::milliseconds> (encoding_end - encoding_begin).count() << "[ms]" << std::endl;
    cout << "Decoding time: " << std::chrono::duration_cast<std::chrono::milliseconds> (decoding_end - decoding_begin).count() << "[ms]" << std::endl;

    // Deallocation of encoded and decoded host arrays
    free(encodedData);
    free(decodedData);
    return 0;
}

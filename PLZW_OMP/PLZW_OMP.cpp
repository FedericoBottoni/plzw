// PLZW_OMP.exe

#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "../dependencies/uthash.h"
#define IN_PATH "F:\\Dev\\PLZW\\in\\in"
#define ALPHABET_LEN 256
#define DEFAULT_NPROCS 3

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

// Declaration of hashmaps and specification of their private scope inside parallel sections
struct unsorted_node_map* table;
#pragma omp threadprivate(table)
struct unsorted_node_map_dec* table_dec;
#pragma omp threadprivate(table_dec)


// Push token-code to the existing hashmap
void push_into_table(char* id, short tokenSize, unsigned int code) {
    struct unsorted_node_map* s = (struct unsorted_node_map*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, id, tokenSize);
    curr_token[tokenSize] = '\0';
    s->id = curr_token;
    s->tokenSize = tokenSize;
    s->code = code;
    HASH_ADD_KEYPTR(hh, table, s->id, tokenSize, s);
}

// Hashmap lookup method
struct unsorted_node_map* find_by_token(char* id, short length) {
    struct unsorted_node_map* s;
    HASH_FIND_STR(table, id, s);
    return s;
}

// Deallocation of the entire hashmap
void dispose_table() {
    struct unsorted_node_map* node, * tmp;

    HASH_ITER(hh, table, node, tmp) {
        free(node->id);
        HASH_DEL(table, node);
        free(node);
    }
}

// Push token-code to the existing hashmap
void push_into_table_dec(unsigned int id, char* token, short tokenSize) {
    struct unsorted_node_map_dec* s = (struct unsorted_node_map_dec*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, token, tokenSize);
    curr_token[tokenSize] = '\0';
    s->token = curr_token;
    s->tokenSize = tokenSize;
    s->id = id;
    HASH_ADD_INT(table_dec, id, s);
}

// Hashmap lookup method
struct unsorted_node_map_dec* find_by_code_dec(unsigned int id) {
    struct unsorted_node_map_dec* s;
    HASH_FIND_INT(table_dec, &id, s);
    return s;
}

// Deallocation of the entire hashmap
void dispose_table_dec() {
    struct unsorted_node_map_dec* node, * tmp;

    HASH_ITER(hh, table_dec, node, tmp) {
        free(node->token);
        HASH_DEL(table_dec, node);
        free(node);
    }
}

unsigned int encoding_lzw(const char* s1, unsigned int count, unsigned int* objectCode)
{
    // Init hashmap with ASCII alphabet
    table = NULL;
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        push_into_table(ch, 1, i);
    }
    free(ch);

    // Definition and assignation of LZW_encoder variables (realloc of dynamic variables is allowed)
    unsorted_node_map* node;
    unsigned int code = ALPHABET_LEN, out_index = 0, pLength = 1;
    char* p = (char*)malloc(2 * sizeof(char)), * pandc = (char*)malloc(3 * sizeof(char)), * c = new char[1];
    p[0] = s1[0];
    p[1] = '\0';
    pandc[2] = '\0';

    // Iterating through all the input buffer
    for (unsigned int i = 0; i < count; i++) {

        // Gather the next value from input
        if (i != count - 1)
            c[0] = s1[i + 1];

        // Gather from hashmap 'p_c' concatenated token: if it exists save it and go on with the algorithm; otherwise
        // get just the 'p' value, add it to outcome and add p_c to hashmap.
        // p and p_c are reallocated in order to take up as little memory as possible
        for (unsigned short str_i = 0; str_i < pLength; str_i++) pandc[str_i] = p[str_i];
        pandc[pLength] = c[0];
        unsorted_node_map* node = find_by_token(pandc, pLength + 1);
        if (node != NULL) {
            p = (char*)realloc(p, (++pLength + 1) * sizeof(char));
            pandc = (char*)realloc(pandc, (pLength + 2) * sizeof(char));
            p[pLength] = '\0';
            pandc[pLength + 1] = '\0';
            for (unsigned short str_i = 0; str_i < pLength; str_i++) p[str_i] = pandc[str_i];
        }
        else {
            node = find_by_token(p, pLength);
            objectCode[out_index++] = node->code;
            push_into_table(pandc, pLength + 1, code);
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

    // Get last value and deallocate
    objectCode[out_index++] = find_by_token(p, pLength)->code;
    free(p);
    free(pandc);
    dispose_table();
    return out_index;
}

unsigned int decoding_lzw(unsigned int* op, int op_length, char* decodedData)
{
    // Init hashmap with ASCII alphabet
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 1; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        push_into_table_dec(i, ch, 1);
    }
    free(ch);

    // Assignation of variables from encoded input and s_node from hasmap
    unsigned int decodedDataLength, n, s_length, old = op[0], temp_length = 0, count = ALPHABET_LEN;
    struct unsorted_node_map_dec* temp_node, * s_node = find_by_code_dec(old);
    s_length = s_node->tokenSize;
    char* s = (char*)malloc((s_length + 1) * sizeof(char)), * temp = (char*)malloc(sizeof(char));
    memcpy(s, s_node->token, s_length);
    s[s_length] = '\0';
    temp[0] = '\0';
    char c = s[0];
    memcpy(decodedData, s, s_length);
    decodedDataLength = 1;

    // Iterating through all the encoded buffer
    for (int i = 0; i < op_length - 1; i++) {

        // Gather the next value from encoded input
        n = op[i + 1];

        // Decoding the old value if the next is new or keep the next one if it's known and realloc s
        if (find_by_code_dec(n) == NULL) {
            s_node = find_by_code_dec(old);
            s_length = s_node->tokenSize;
            s = (char*)realloc(s, (++s_length + 1) * sizeof(char));
            memcpy(s, s_node->token, s_length - 1);
            s[s_length - 1] = c;
            s[s_length] = '\0';
        }
        else {
            s_node = find_by_code_dec(n);
            if (s_node->tokenSize != s_length) {
                s_length = s_node->tokenSize;
                s = (char*)realloc(s, (s_length + 1) * sizeof(char));
                s[s_length] = '\0';
            }
            memcpy(s, s_node->token, s_length);
        }

        // Update outcome array
        memcpy(&decodedData[decodedDataLength], s, s_length);
        decodedDataLength += s_length;

        // Push the new token to the hashmap and realloc temp if length is changed
        c = s[0];
        temp_node = find_by_code_dec(old);
        if (temp_length != temp_node->tokenSize + 1) {
            temp_length = temp_node->tokenSize + 1;
            temp = (char*)realloc(temp, (temp_length + 1) * sizeof(char));
            temp[temp_length] = '\0';
        }
        memcpy(temp, temp_node->token, temp_length - 1);
        temp[temp_length - 1] = c;
        push_into_table_dec(count, temp, temp_length);
        count++;
        old = n;
    }

    // Deallocations
    free(temp);
    free(s);
    dispose_table_dec();
    return decodedDataLength;
}


unsigned int encoding(int nProcs, const char* input, unsigned int inputLength, unsigned int* encodedData, unsigned int* encStartPos) {
    // Assignation of inputs and definition of outputs
    unsigned int avgRng, encodedLength = 0, *encodedBuffLengths = (unsigned int*)malloc(nProcs * sizeof(unsigned int));
    avgRng = inputLength / nProcs;

    // Setting the number of threads and open the parallel section
    // Each variable is shared but the hasmap which is private
    omp_set_num_threads(nProcs);
    #pragma omp parallel shared(nProcs), shared(input), shared(inputLength), shared(avgRng), shared(encodedLength), shared(encodedData), shared(encodedBuffLengths), shared(encStartPos), default(none)
    {
        // Get the thread is and perform segmentation
        char thid = omp_get_thread_num();
        unsigned int dataBuffLength, * encodedBuff;
        dataBuffLength = thid != nProcs - 1 ? avgRng : inputLength - (avgRng * (nProcs - 1));
        encodedBuff = (unsigned int*)malloc(dataBuffLength * sizeof(unsigned int));
        const char* shifted_input_point = &input[avgRng * thid];

        // LZW_encoding call + thread synchronization
        encodedBuffLengths[thid] = encoding_lzw(shifted_input_point, dataBuffLength, encodedBuff);
        #pragma omp barrier

        // Calculation of output alignment
        unsigned int encodedDataOffset = 0;
        for (unsigned short i = 0; i < thid; i++) {
            encodedDataOffset += encodedBuffLengths[i];
        }
        unsigned int* shifted_encodedData = &encodedData[encodedDataOffset];

        // Put the outcomes in arrays
        memcpy(shifted_encodedData, encodedBuff, encodedBuffLengths[thid] * sizeof(unsigned int));
        encStartPos[thid] = encodedDataOffset;

        // Calculation of outcome's length
        #pragma omp single
        {
            for (unsigned short i = 0; i < nProcs; i++) {
                encodedLength += encodedBuffLengths[i];
            }
        }

        // Deallocation
        free(encodedBuff);
    }
    free(encodedBuffLengths);
    return encodedLength;
}


unsigned int decoding(int nProcs, unsigned int* encodedData, unsigned int encodedLength, unsigned int* encStartPos, char* decodedData, unsigned int decodedExpectedLength) {
    // Definition of output
    unsigned int decodedLength = 0;
    unsigned int* decodedBuffLengths = (unsigned int*)malloc(nProcs * sizeof(unsigned int));
    
    // Open the parallel section
    // Each variable is shared but the hasmap which is private
    #pragma omp parallel shared(nProcs), shared(encodedData), shared(encodedLength), shared(encStartPos), shared(decodedData), shared(decodedBuffLengths), shared(decodedLength), shared(decodedExpectedLength), default(none)
    {
        // Get the thread is and perform the output segmentation
        char thid = omp_get_thread_num(), * decodedDataBuff;
        unsigned int dataBuffLength, avgRng = decodedExpectedLength / nProcs;
        dataBuffLength = thid != nProcs - 1 ? avgRng : decodedExpectedLength - (avgRng * (nProcs - 1));
        decodedDataBuff = (char*)malloc(dataBuffLength * sizeof(char));

        // Build the input buffers
        unsigned int encodedBuffLength, * encodedBuffLengths = (unsigned int*)malloc(nProcs * sizeof(unsigned int));
        for (unsigned short p = 0; p < nProcs - 1; p++) {
            encodedBuffLengths[p] = encStartPos[p + 1] - encStartPos[p];
        }
        encodedBuffLengths[nProcs - 1] = encodedLength - encStartPos[nProcs - 1];
        encodedBuffLength = encodedBuffLengths[thid];

        // Input alignment
        unsigned int encodedDataOffset = 0;
        for (unsigned short i = 0; i < thid; i++) {
            encodedDataOffset += encodedBuffLengths[i];
        }
        unsigned int* shifted_encodedData = &encodedData[encodedDataOffset];

        // LZW_decoding call + thread synchronization
        decodedBuffLengths[thid] = decoding_lzw(shifted_encodedData, encodedBuffLength, decodedDataBuff);
        #pragma omp barrier

        // Calculation of output alignment
        unsigned int decodedDataOffset = 0;
        for (unsigned short i = 0; i < thid; i++) {
            decodedDataOffset += decodedBuffLengths[i];
        }
        char* shifted_decodedData = &decodedData[decodedDataOffset];

        // Put the outcomes in array
        memcpy(shifted_decodedData, decodedDataBuff, decodedBuffLengths[thid] * sizeof(char));

        // Calculation of outcome's length
        #pragma omp single
        {
            for (unsigned short i = 0; i < nProcs; i++) {
                decodedLength += decodedBuffLengths[i];
            }
        }

        // Deallocation
        free(encodedBuffLengths);
    }
    free(decodedBuffLengths);
    return decodedLength;
}

int main()
{
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

    // Initialization of input and output variables
    unsigned int inputLength = input.length(), nProcs = DEFAULT_NPROCS;
    unsigned int* encodedData = (unsigned int*)malloc(inputLength * sizeof(unsigned int));
    unsigned int* encStartPos = (unsigned int*)malloc(nProcs * sizeof(unsigned int));

    // Function encoding call
    unsigned int encodedLength = encoding(nProcs, input.c_str(), inputLength, encodedData, encStartPos);

    // Timer STOP
    encoding_end = std::chrono::steady_clock::now();

    // Skipping the export of encoded data and the reimport

    // Timer START
    decoding_begin = std::chrono::steady_clock::now();

    // Declaration and definition of the needed variable (skipping the initialization of input variables)
    char* decodedData = (char*)malloc(inputLength * sizeof(char));

    // Function decoding call
    int decodedLength = decoding(nProcs, encodedData, encodedLength, encStartPos, decodedData, inputLength);

    // Timer STOP
    decoding_end = std::chrono::steady_clock::now();

    // Checking the correctness of lossless propriety
    if (inputLength == decodedLength) {
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

    // Deallocation of encoded and decoded arrays
    free(encodedData);
    free(encStartPos);
    free(decodedData);
    return 0;
}

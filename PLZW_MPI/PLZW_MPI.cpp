// mpiexec -np 4 PLZW_MPI.exe

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "../dependencies/uthash.h"
#define IN_PATH "F:\\Dev\\PLZW\\in\\in"
#define ALPHABET_LEN 256

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

// Declaration of hashmaps
struct unsorted_node_map* table;
struct unsorted_node_map_dec* table_dec;

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

unsigned int encoding(string input, int inputLength, int rank, int size, int root, unsigned int* encodedData, int* encStartPos) {
    // Assignation of inputs and definition of variables
    unsigned int avgRng, receiveCount, * startPoint, encodedLength;
    int* counts = (int*)malloc(size * sizeof(int));
    startPoint = (unsigned int*)malloc(size * sizeof(unsigned int));
    avgRng = inputLength / size;
    receiveCount = avgRng;
    if (rank == size - 1) {
        receiveCount = inputLength - (size - 1) * avgRng;
    }
    char* inputBuf = (char*)malloc(receiveCount * sizeof(char));

    // Calculation of segmentation indexes and lengths
    for (unsigned short p = 0; p < size; p++) {
        startPoint[p] = avgRng * p;
        counts[p] = avgRng;
        encStartPos[p] = avgRng * p;
    }
    counts[size - 1] = inputLength - (size - 1) * avgRng;

    // Perform segmentation
    MPI_Scatterv(
        input.c_str(),
        counts,
        encStartPos,
        MPI_CHAR,
        inputBuf,
        receiveCount,
        MPI_CHAR,
        root,
        MPI_COMM_WORLD);

    // Defininition of output buffers
    unsigned int* encodedDataBuff = (unsigned int*)malloc(receiveCount * sizeof(unsigned int));
    int* encodedLengthBuffs = (int*)malloc(receiveCount * sizeof(int));

    // LZW_encoding call
    unsigned int encodedLengthBuff = encoding_lzw(inputBuf, receiveCount, encodedDataBuff);

    // Spread of the buffer lengths between processes
    for (unsigned short p = 0; p < size; p++) {
        counts[p] = 1;
        encStartPos[p] = p;
    }
    MPI_Allgather(
        &encodedLengthBuff,
        1,
        MPI_INT,
        encodedLengthBuffs,
        1,
        MPI_INT,
        MPI_COMM_WORLD);

    // Calculation of output length and the start position of each strip of codes
    encodedLength = 0;
    for (short p = 0; p < size; p++) {
        encStartPos[p] = encodedLength;
        encodedLength += encodedLengthBuffs[p];
    }

    // Master process gathers the outputs and build the single output array
    MPI_Gatherv(
        encodedDataBuff,
        encodedLengthBuff,
        MPI_INT,
        encodedData,
        encodedLengthBuffs,
        encStartPos,
        MPI_INT,
        root,
        MPI_COMM_WORLD);

    // Deallocations
    free(encodedDataBuff);
    free(encodedLengthBuffs);
    free(startPoint);
    free(counts); 
    free(inputBuf);
    return encodedLength;
}

unsigned int decoding(unsigned int* encodedData, int encodedLength, int* encStartPos, int rank, int size, int root, char* decodedData, unsigned int decodedExpectedLength) {
    // Definition of output and other variables
    int encodedLengthBuff, * decodedLengthBuffs, *counts;
    unsigned int decodedLengthBuff, decodedLength, dataBuffLength, *encodedDataBuff, avgRng = decodedExpectedLength / size;
    dataBuffLength = rank != size - 1 ? avgRng : decodedExpectedLength - (avgRng * (size - 1));
    decodedLengthBuffs = (int*)malloc(size * sizeof(int));
    counts = (int*)malloc(size * sizeof(int));
    char* decodedBuff = (char*)malloc(dataBuffLength * sizeof(char));

    // Calculation of segmentation lengths
    for (unsigned short p = 0; p < size - 1; p++) {
        counts[p] = encStartPos[p + 1] - encStartPos[p];
    }
    counts[size - 1] = encodedLength - encStartPos[size - 1];

    // Definition of input buffers
    encodedLengthBuff = counts[rank];
    encodedDataBuff = (unsigned int*)malloc(encodedLengthBuff * sizeof(unsigned int));

    // Perform segmentation
    MPI_Scatterv(
        encodedData,
        counts,
        encStartPos,
        MPI_INT,
        encodedDataBuff,
        counts[rank],
        MPI_INT,
        root,
        MPI_COMM_WORLD);

    // LZW_decoding call
    decodedLengthBuff = decoding_lzw(encodedDataBuff, counts[rank], decodedBuff);

    // Spread of the buffer lengths between processes
    MPI_Allgather(
        &decodedLengthBuff,
        1,
        MPI_INT,
        decodedLengthBuffs,
        1,
        MPI_INT,
        MPI_COMM_WORLD);


    // Calculation of output length and the start position of each strip of codes
    decodedLength = 0;
    for (short p = 0; p < size; p++) {
        encStartPos[p] = decodedLength;
        decodedLength += decodedLengthBuffs[p];
    }

    // Master process gathers the outputs and build the single output array
    MPI_Gatherv(
        decodedBuff,
        decodedLengthBuff,
        MPI_CHAR,
        decodedData,
        decodedLengthBuffs,
        encStartPos,
        MPI_CHAR,
        root,
        MPI_COMM_WORLD);

    // Deallocations
    free(decodedBuff);
    free(decodedLengthBuffs);
    free(encodedDataBuff);
    free(counts);
    return decodedLength;
}

int main()
{
    // Initialization of MPI
    int size, rank;
    int root = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Input and timer declaration
    string input;
    unsigned int inputLength;
    std::chrono::steady_clock::time_point encoding_begin, encoding_end, decoding_begin, decoding_end;
    bool correctness;

    // Read dataset from file
    if (rank == root) {
        ifstream inFile;
        inFile.open(IN_PATH);
        correctness = true;
        if (!inFile) {
            cout << "Unable to open file";
            exit(1);
        }
        string line;
        while (inFile >> line) {
            input += line;
        }
        line.clear();
        inFile.close();

        // Timer START and calculate input size
        encoding_begin = std::chrono::steady_clock::now();
        inputLength = input.length();
    }

    // Broadcast the input size to each process and definite the output variables
    MPI_Bcast(
        &inputLength,
        1,
        MPI_INT,
        root,
        MPI_COMM_WORLD
    );
    unsigned int* encodedData = (unsigned int*)malloc(inputLength * sizeof(unsigned int));
    int* encStartPos = (int*)malloc(size * sizeof(int));

    // Function encoding call
    int encodedLength = encoding(input, inputLength, rank, size, root, encodedData, encStartPos);

    // Timer STOP
    if (rank == root) {
        encoding_end = std::chrono::steady_clock::now();
    }

    // Skipping the export of encoded data and the reimport

    // Timer START
    if (rank == root) {
        decoding_begin = std::chrono::steady_clock::now();
    }

    // Declaration and definition of the needed variable (skipping the initialization of input variables)
    char* decodedData = (char*)malloc(inputLength * sizeof(char));

    // Function decoding call
    int decodedLength = decoding(encodedData, encodedLength, encStartPos, rank, size, root, decodedData, inputLength);

    // Timer STOP
    if (rank == root) {
        decoding_end = std::chrono::steady_clock::now();
    }

    // Checking the correctness of lossless propriety
    if (rank == root) {
        if (inputLength == decodedLength) {
            const char* input_point = input.c_str(); 
            for (int i = 0; i < input.length(); i++) {
                if (input_point[i] != decodedData[i]) {
                    correctness = 0;
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

    }

    // Deallocation of encoded and decoded arrays + MPI_Finalize
    free(encStartPos);
    free(encodedData);
    free(decodedData);
    MPI_Finalize();
    return 0;
}


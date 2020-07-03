#ifndef LZH
#define LZH
#include <unordered_map>
#define ALPHABET_LEN 256

using namespace std;
inline int encoding_lzw(const char* s1, unsigned int count, unsigned int* objectCode)
{
    unordered_map<string, int> table;
    for (int i = 0; i < ALPHABET_LEN; i++) {
        string ch = "";
        ch += char(i);
        table[ch] = i;
    }
    int out_index = 0;
    string p = "", c = "";
    p += s1[0];
    int code = ALPHABET_LEN;
    unsigned int i;
    for (i = 0; i < count; i++) {
        if (i != count - 1)
            c += s1[i + 1];
        if (table.find(p + c) != table.end()) {
            p = p + c;
        }
        else {
            objectCode[out_index++] = table[p];
            table[p + c] = code;
            code++;
            p = c;
        }
        c = "";
    }
    objectCode[out_index++] = table[p];
    return out_index;
}

inline string decoding_lzw(unsigned int* op, int op_length)
{
    unordered_map<int, string> table;
    for (int i = 0; i < ALPHABET_LEN; i++) {
        string ch = "";
        ch += char(i);
        table[i] = ch;
    }

    int old = op[0], n;
    string result = "";
    string s = table[old];
    string c = "";
    c += s[0];
    result = result.append(s);
    int count = ALPHABET_LEN;
    for (int i = 0; i < op_length - 1; i++) {
        n = op[i + 1];
        if (table.find(n) == table.end()) {
            s = table[old];
            s = s + c;
        }
        else {
            s = table[n];
        }
        result = result.append(s);
        c = "";
        c += s[0];
        table[count] = table[old] + c;
        count++;
        old = n;
    }
    return result;
}

// LZW in C

#define MAX_TOKEN_SIZE 1000

struct unordered_map_node
{
    char* token;
    short tokenSize;
    unsigned int code;
    struct unordered_map_node* next;
};

struct unordered_map_node* unordered_map_head = NULL;

void disposeMap() {
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

void unordered_map_push(char* token, int code, short tokenLength) {
    struct unordered_map_node* link = (struct unordered_map_node*)malloc(sizeof(struct unordered_map_node));
    char* token_pointer = (char*)malloc(tokenLength * sizeof(char));
    memcpy(token_pointer, token, tokenLength);
    link->token = token_pointer;
    link->tokenSize = tokenLength;
    link->code = code;
    link->next = unordered_map_head;
    unordered_map_head = link;
}

unsigned int getCodeFromMap(char* token, int tokenLength) {
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

unordered_map_node* getNodeFromMap(unsigned int code) {
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

bool isTokenInMap(char* token, int tokenLength) {
    unsigned int code = getCodeFromMap(token, tokenLength);
    return code != UINT_MAX;
}

bool isCodeInMap(unsigned int code) {
    unordered_map_node* token = getNodeFromMap(code);
    return token != NULL;
}

int encoding_lzw_c(const char* s1, unsigned int count, unsigned int* objectCode)
{
    char* ch;
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch = new char[1];
        ch[0] = char(i);
        unordered_map_push(ch, i, 1);
    }
    delete[] ch;

    int out_index = 0, pLength;
    char* p = new char[MAX_TOKEN_SIZE], * pandc = new char[MAX_TOKEN_SIZE], * c = new char[1];
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
        c[0] = NULL;
        //memset(pandc, 0, sizeof(pandc));
        if (pLength > MAX_TOKEN_SIZE - 1) {
            printf("Token-size is not enough big");
        }
    }
    objectCode[out_index] = getCodeFromMap(p, pLength);

    delete[] p;
    delete[] pandc;
    disposeMap();
    return out_index;
}

unsigned int decoding_lzw_c(unsigned int* op, int op_length, char* decodedData)
{
    char* ch;
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch = new char[1];
        ch[0] = char(i);
        unordered_map_push(ch, i, 1);
    }
    delete[] ch;

    unsigned int old = op[0], decodedDataLength, n;
    unordered_map_node* temp_node, * s_node = getNodeFromMap(old);
    int temp_length, s_length = s_node->tokenSize;
    char* s = new char[MAX_TOKEN_SIZE], * temp = new char[MAX_TOKEN_SIZE];
    memcpy(s, s_node->token, s_length);
    char* c = new char[1];
    c[0] = s[0];
    memcpy(decodedData, s, s_length);
    decodedDataLength = 1;
    int count = ALPHABET_LEN;
    for (int i = 0; i < op_length - 1; i++) {
        n = op[i + 1];
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
    disposeMap();
    return decodedDataLength;
}

#endif
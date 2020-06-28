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

#endif
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

struct TranslationNode
{
    TranslationNode *zero, *one;
    unsigned short character;
};

long int readFileSize(unsigned char &, int, FILE *);
void translateFile(char *, long int, unsigned char &, int &, TranslationNode *, FILE *);

unsigned char process_8_bits_NUMBER(unsigned char &, int, FILE *);
unsigned short process_16_bits_DATA(unsigned char &, int &, FILE *);
void process_n_bits_TO_STRING(unsigned char &, int, int &, FILE *, TranslationNode *, unsigned short);

bool file_exists(char *);
void change_name_if_exists(char *);

void burn_tree(TranslationNode *);

/*          CONTENT TABLE IN ORDER
    compressed file's composition is in order below
    that is why we re going to translate it part by part

.first (one byte)           ->  letter_count
.third (bit groups)
    3.1 (8 bits)            ->  current unique byte
    3.2 (8 bits)            ->  length of the transformation
    4.3 (bits)              ->  transformation code of that unique byte
    .sixth (8 bytes)        ->  size of current file (IF FILE)
    .eighth (a lot of bits) ->  translate and write current file (IF FILE)

*whenever we see a new folder we will write seventh then
    start writing the files(and folders) inside the current folder from fourth to eighth
**groups from fifth to eighth will be written as much as the file count
*/

int main(int argc, char *argv[])
{
    int uniqueSymbolCount = 0;
    FILE *compressedFile;
    if (argc != 2)
    {
        std::cout << "Missing compressed file name." << endl
                  << "Usage: './extract <compressed_file_name>'" << endl;
        return 1;
    }

    compressedFile = fopen(argv[1], "rb");
    if (!compressedFile)
    {
        std::cout << argv[1] << " does not exist" << endl;
        return 0;
    }

    fseek(compressedFile, 0, SEEK_END);
    fseek(compressedFile, 0, SEEK_SET);

    // Reading the unique byte count from the compressed file's header.
    fread(&uniqueSymbolCount, 2, 1, compressedFile);
    if (uniqueSymbolCount == 0)
        uniqueSymbolCount = 65536; // Handling the special case where there are 256 unique bytes.

    unsigned char bufferByte = 0;
    int bitCounter = 0;
    TranslationNode *root = new TranslationNode;
    root->zero = nullptr;
    root->one = nullptr;

    // Reading transformation information for each unique byte and storing it in the translation tree.
    for (int i = 0; i < uniqueSymbolCount; i++)
    {
        unsigned short decodedCharacter = process_16_bits_DATA(bufferByte, bitCounter, compressedFile);
        unsigned int transformationLength = (unsigned int)process_8_bits_NUMBER(bufferByte, bitCounter, compressedFile);
        if (transformationLength == 0)
            transformationLength = 65536;
        process_n_bits_TO_STRING(bufferByte, transformationLength, bitCounter, compressedFile, root, decodedCharacter);
    }

    // File count was written to the compressed file from least significiant byte
    // to most significiant byte to make sure system's endianness
    // does not affect the process and that is why we are processing size information like this

    long int fileSize = readFileSize(bufferByte, bitCounter, compressedFile);
    string newfileName = "DECOMPRESSED_FILE";
    change_name_if_exists(&newfileName[0]);

    // Translating and writing the file content based on the Huffman encoding.
    translateFile(&newfileName[0], fileSize, bufferByte, bitCounter, root, compressedFile);

    // Clean up.
    fclose(compressedFile);
    burn_tree(root);
    std::cout << "Decompression is complete" << endl;
}

// burn_tree function is used for deallocating translation tree
void burn_tree(TranslationNode *node)
{
    if (node->zero)
        burn_tree(node->zero);
    if (node->one)
        burn_tree(node->one);
    delete node;
}

// process_n_bits_TO_STRING function reads n successive bits from the compressed file
// and stores it in a leaf of the translation tree,
// after creating that leaf and sometimes after creating nodes that are binding that leaf to the tree.
void process_n_bits_TO_STRING(unsigned char &bufferByte, int n, int &bitCounter, FILE *filePtr, TranslationNode *node, unsigned short uShort)
{
    for (int i = 0; i < n; i++)
    {
        if (bitCounter == 0)
        {
            fread(&bufferByte, 1, 1, filePtr);
            bitCounter = 8;
        }
        if (bufferByte & 0x80)
        {
            if (!(node->one))
            {
                node->one = new TranslationNode;
                node->one->zero = NULL;
                node->one->one = NULL;
            }
            node = node->one;
        }
        else
        {
            if (!(node->zero))
            {
                node->zero = new TranslationNode;
                node->zero->zero = NULL;
                node->zero->one = NULL;
            }
            node = node->zero;
        }

        bufferByte <<= 1;
        bitCounter--;
    }
    node->character = uShort;
}

// process_8_bits_NUMBER reads 8 successive bits from compressed file
//(does not have to be in the same byte)
// and returns it in unsigned char form
unsigned char process_8_bits_NUMBER(unsigned char &currentByte, int bitCounter, FILE *filePtr)
{
    unsigned char val, temp_byte;
    fread(&temp_byte, 1, 1, filePtr);
    val = currentByte | (temp_byte >> bitCounter);
    currentByte = temp_byte << 8 - bitCounter;
    return val;
}


unsigned short process_16_bits_DATA(unsigned char &currentByte, int &bitCounter, FILE *filePtr) {
    unsigned short highPart = process_8_bits_NUMBER(currentByte, bitCounter, filePtr);
    unsigned short lowPart = process_8_bits_NUMBER(currentByte, bitCounter, filePtr);
    return (highPart << 8) | lowPart;
}


void change_name_if_exists(char *name)
{
    char *i;
    int copy_count;
    if (file_exists(name))
    {
        char *dot_pointer = NULL;
        for (i = name; *i; i++)
        {
            if (*i == '.')
                dot_pointer = i;
        }
        if (dot_pointer)
        {
            string s = dot_pointer;
            strcpy(dot_pointer, "(1)");
            dot_pointer++;
            strcpy(dot_pointer + 2, &s[0]);
        }
        else
        {
            dot_pointer = i;
            strcpy(dot_pointer, "(1)");
            dot_pointer++;
        }
        for (copy_count = 1; copy_count < 10; copy_count++)
        {
            *dot_pointer = (char)('0' + copy_count);
            if (!file_exists(name))
            {
                break;
            }
        }
    }
}

// checks if the file or folder exists
bool file_exists(char *name)
{
    FILE *fp = fopen(name, "rb");
    if (fp)
    {
        fclose(fp);
        return 1;
    }
    else
    {
        DIR *dir = opendir(name);
        if (dir)
        {
            closedir(dir);
            return 1;
        }
    }
    return 0;
}

// returns file's size
long int readFileSize(unsigned char &bufferByte, int bitCounter, FILE *filePtr)
{
    long int size = 0;
    {
        long int multiplier = 1;
        for (int i = 0; i < 8; i++)
        {
            size += process_8_bits_NUMBER(bufferByte, bitCounter, filePtr) * multiplier;
            multiplier *= 256;
        }
    }
    return size;
}

// This function translates compressed file from info that is now stored in the translation tree
// then writes it to a newly created file
void translateFile(char *newFileName, long int fileSize, unsigned char &bufferByte, int &bitCounter, TranslationNode *root, FILE *filePtr)
{
    FILE *newFilePtr = fopen(newFileName, "wb");
    for (long int i = 0; i < fileSize; i+=2)
    {
        TranslationNode *node = root;
        while (node->zero || node->one)
        {
            if (bitCounter == 0)
            {
                fread(&bufferByte, 1, 1, filePtr);
                bitCounter = 8;
            }
            if (bufferByte & 0x80)
            {
                node = node->one;
            }
            else
            {
                node = node->zero;
            }
            bufferByte <<= 1;
            bitCounter--;
        }
        fwrite(&(node->character), 2, 1, newFilePtr);
    }
    fclose(newFilePtr);
}
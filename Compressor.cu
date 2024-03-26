#include <iostream>
#include <cstdio>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dirent.h>

using namespace std;

void writeFromUChar(unsigned char, unsigned char &, int, FILE *);
long int sizeOfTheFile(char *);
void writeFileSize(long int, unsigned char &, int, FILE *);
void writeFileContent(FILE *, long int, string *, unsigned char &, int &, FILE *);
void writeIfFullBuffer(unsigned char &, int &, FILE *);

struct TreeNode
{ // this structure will be used to create the translation tree
    TreeNode *left, *right;
    long int occurrences;
    unsigned char character;
    string bit;
};

bool TreeNodeCompare(TreeNode a, TreeNode b)
{
    return a.occurrences < b.occurrences;
}

void genHuffmanTree(TreeNode *, int, long int *);

int main(int argc, char *argv[])
{
    long int freqCount[256] = {0};
    int uniqueSymbolCount = 0;
    if (argc != 2)
    {
        std::cout << "Must provide a single file name." << endl;
        return 0;
    }

    FILE *originalFilePtr;
    originalFilePtr = fopen(argv[1], "rb");
    if (!originalFilePtr)
    {
        std::cout << argv[1] << " file does not exist" << endl
                  << "Process has been terminated" << endl;
        return 0;
    }
    fclose(originalFilePtr);

    // Histograming the frequency of bytes.
    unsigned char *readBufPtr, readBuf;
    readBufPtr = &readBuf;

    long int originalFileSize = sizeOfTheFile(argv[1]);
    std::cout << "The size of the sum of ORIGINAL files is: " << originalFileSize << " bytes" << endl;

    // "rb" is for reading binary files
    originalFilePtr = fopen(argv[1], "rb");
    // reading the first byte of the file into readBuf.
    fread(readBufPtr, 1, 1, originalFilePtr);
    for (long int i = 0; i < originalFileSize; i++)
    { // counting usage frequency of unique bytes inside the file
        freqCount[readBuf]++;
        fread(readBufPtr, 1, 1, originalFilePtr);
    }
    fclose(originalFilePtr);

    // Traverse through all possible bytes and count the number of unique bytes.
    for (long int *i = freqCount; i < freqCount + 256; i++)
    {
        if (*i)
        {
            uniqueSymbolCount++;
        }
    }

    // Generate huffmantree
    TreeNode nodesForHuffmanTree[uniqueSymbolCount * 2 - 1];
    genHuffmanTree(nodesForHuffmanTree, uniqueSymbolCount, freqCount);

    // Writing to compressed file
    string scompressed = argv[1];
    scompressed += ".compressed";
    FILE *compressedFilePtr = fopen(&scompressed[0], "wb");

    // Writing the first piece of header information: the count of unique bytes.
    // This count is essential for reconstructing the Huffman tree during the decompression process.
    fwrite(&uniqueSymbolCount, 1, 1, compressedFilePtr);

    int bitCounter = 0;
    unsigned char bufferByte;
    // Initializing a pointer for iterating through the transformation strings.
    char *transformationStringPtr;
    // Variables for storing the length of the transformation string and the current character being processed.
    unsigned char transformationLength, currentCharacter;
    // Array to store transformation strings for each unique character to optimize compression.
    string transformationStrings[256];

    // Iterate through each node in the Huffman tree to write transformation codes to the compressed file.
    for (TreeNode *node = nodesForHuffmanTree; node < nodesForHuffmanTree + uniqueSymbolCount; node++)
    {
        // Store the transformation string for the current character in the array.
        transformationStrings[node->character] = node->bit;
        transformationLength = node->bit.length();
        currentCharacter = node->character;

        // Write the current character and its transformation string length to the compressed file.
        writeFromUChar(currentCharacter, bufferByte, bitCounter, compressedFilePtr);
        writeFromUChar(transformationLength, bufferByte, bitCounter, compressedFilePtr);

        // Write the transformation string bit by bit to the compressed file.
        transformationStringPtr = &node->bit[0];
        while (*transformationStringPtr)
        {
            bufferByte <<= 1;
            if (*transformationStringPtr == '1')
            {
                bufferByte |= 1;
            }
            bitCounter++;
            transformationStringPtr++;
            writeIfFullBuffer(bufferByte, bitCounter, compressedFilePtr);
        }
    }

    originalFilePtr = fopen(argv[1], "rb");

    // Writing the size of the file, its name, and its content in the compressed format.
    writeFileSize(originalFileSize, bufferByte, bitCounter, compressedFilePtr);
    writeFileContent(originalFilePtr, originalFileSize, transformationStrings, bufferByte, bitCounter, compressedFilePtr);
    fclose(originalFilePtr);

    // Ensuring the last byte is written to the compressed file by aligning the bit counter.
    if (bitCounter > 0)
    {
        bufferByte <<= (8 - bitCounter);
        fwrite(&bufferByte, 1, 1, compressedFilePtr);
    }

    fclose(compressedFilePtr);

    // Get the size of compressed file.
    long int compressedFileSize = sizeOfTheFile(&scompressed[0]);
    std::cout << "The size of the COMPRESSED file is: " << compressedFileSize << " bytes" << endl;

    // Calculate the compression ratio.
    float compressionRatio = 100.0f * static_cast<float>(compressedFileSize) / static_cast<float>(originalFileSize);
    std::cout << "Compressed file's size is [" << compressionRatio << "%] of the original files." << endl;

    // Warning if the compressed file is unexpectedly larger than the original sum.
    if (compressedFileSize > originalFileSize)
    {
        std::cout << "\nWARNING: The compressed file's size is larger than the sum of the originals.\n\n";
    }

    std::cout << endl
              << "Created compressed file: " << scompressed << endl;
    std::cout << "Compression is complete" << endl;
}


void genHuffmanTree(TreeNode *nodes, int treeSize, long int *occurrences) {
    TreeNode *currentNode = nodes;

    // Fill the array with data for each unique byte
    for (long int *frequency = occurrences; frequency < occurrences + 256; frequency++) {
        if (*frequency) {
            currentNode->right = NULL;
            currentNode->left = NULL;
            currentNode->occurrences = *frequency;
            currentNode->character = (unsigned char)(frequency - occurrences);
            currentNode++;
        }
    }

    // Sort the leaf nodes based on frequency to prepare for tree construction
    sort(nodes, nodes + treeSize, TreeNodeCompare);

    // Construct the Huffman tree by merging nodes with the lowest frequencies
    TreeNode *smallestNode = nodes;
    TreeNode *secondSmallestNode = nodes + 1;
    TreeNode *newInternalNode = nodes + treeSize;
    TreeNode *nextInternalNode = nodes + treeSize;
    TreeNode *nextLeafNode = nodes + 2;

    for (int i = 0; i < treeSize - 1; i++) {
        newInternalNode->occurrences = smallestNode->occurrences + secondSmallestNode->occurrences;
        newInternalNode->left = smallestNode;
        newInternalNode->right = secondSmallestNode;
        smallestNode->bit = "1";
        secondSmallestNode->bit = "0";
        newInternalNode++;

        if (nextLeafNode >= nodes + treeSize) {
            smallestNode = nextInternalNode;
            nextInternalNode++;
        } else {
            smallestNode = (nextLeafNode->occurrences < nextInternalNode->occurrences) ? nextLeafNode++ : nextInternalNode++;
        }

        if (nextLeafNode >= nodes + treeSize) {
            secondSmallestNode = nextInternalNode;
            nextInternalNode++;
        } else if (nextInternalNode >= newInternalNode) {
            secondSmallestNode = nextLeafNode;
            nextLeafNode++;
        } else {
            secondSmallestNode = (nextLeafNode->occurrences < nextInternalNode->occurrences) ? nextLeafNode++ : nextInternalNode++;
        }
    }

    // Assign Huffman codes to each node
    for (TreeNode *node = nodes + treeSize * 2 - 2; node > nodes - 1; node--) {
        if (node->left) {
            node->left->bit = node->bit + node->left->bit;
        }
        if (node->right) {
            node->right->bit = node->bit + node->right->bit;
        }
    }
}


// below function is used for writing the uChar to compressed file
// It does not write it directly as one byte instead it mixes uChar and current byte, writes 8 bits of it
// and puts the rest to curent byte for later use
void writeFromUChar(unsigned char byteToWrite, unsigned char &bufferByte, int bitCounter, FILE *filePtr)
{
    // Going to write at least 1 byte, first shift the bufferByte to the left
    // to make room for the new byte.
    bufferByte <<= 8 - bitCounter;
    bufferByte |= (byteToWrite >> bitCounter);
    fwrite(&bufferByte, 1, 1, filePtr);
    bufferByte = byteToWrite;
}

// This function is writing byte count of current input file to compressed file using 8 bytes
// It is done like this to make sure that it can work on little, big or middle-endian systems
void writeFileSize(long int fileSize, unsigned char &bufferByte, int bitCounter, FILE *filePtr)
{
    for (int i = 0; i < 8; i++)
    {
        writeFromUChar(fileSize % 256, bufferByte, bitCounter, filePtr);
        fileSize /= 256;
    }
}

// Below function translates and writes bytes from current input file to the compressed file.
void writeFileContent(FILE *originalFilePtr, long int originalFileSize, string *transformationStrings, unsigned char &bufferByte, int &bitCounter, FILE *compressedFilePtr)
{
    unsigned char *bufPtr, buf;
    bufPtr = &buf;
    char *strPointer;
    fread(bufPtr, 1, 1, originalFilePtr);
    for (long int i = 0; i < originalFileSize; i++)
    {
        strPointer = &transformationStrings[buf][0];
        while (*strPointer)
        {
            writeIfFullBuffer(bufferByte, bitCounter, compressedFilePtr);

            bufferByte <<= 1;
            if (*strPointer == '1')
            {
                bufferByte |= 1;
            }
            bitCounter++;
            strPointer++;
        }
        fread(bufPtr, 1, 1, originalFilePtr);
    }
}

long int sizeOfTheFile(char *path)
{
    ifstream file(path, ifstream::ate | ifstream::binary);
    return file.tellg();
}

void writeIfFullBuffer(unsigned char &bufferByte, int& bitCounter, FILE *filePtr)
{
    if (bitCounter == 8)
    {
        fwrite(&bufferByte, 1, 1, filePtr);
        bitCounter = 0;
    }
}
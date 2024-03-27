#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include "gpuHuffmanConstruction.h"

using namespace std;

void writeFromUShort(unsigned short, unsigned char &, int, FILE *);
void writeFromUChar(unsigned char, unsigned char &, int, FILE *);
long int sizeOfTheFile(char *);
void writeFileSize(long int, unsigned char &, int, FILE *);
void writeFileContent(FILE *, long int, string *, unsigned char &, int &,
                      FILE *);
void writeIfFullBuffer(unsigned char &, int &, FILE *);

struct TreeNode
{ // this structure will be used to create the translation tree
    TreeNode *left, *right;
    unsigned int occurrences;
    unsigned short character;
    string bit;
};

bool TreeNodeCompare(TreeNode a, TreeNode b)
{
    return a.occurrences < b.occurrences;
}

__global__ void calculateFrequency(const unsigned char *data, long int size,
                                   unsigned int *freqCount)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size / 2; i += stride)
    {
        unsigned short readBuf = (data[i * 2 + 1] << 8) | data[i * 2];
        atomicAdd(&freqCount[readBuf], 1);
    }
}

__global__ void countNonZero(const unsigned int *data, int size, unsigned int *nonZeroCount)
{
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < size && data[i] > 0) ? 1 : 0;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        nonZeroCount[blockIdx.x] = sdata[0];
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Must provide a single file name." << endl;
        return 0;
    }

    static constexpr int kMaxSymbolSize = 65536;
    ifstream originalFile(argv[1], ios::binary);
    if (!originalFile.is_open())
    {
        std::cout << argv[1] << " file does not exist" << endl
                  << "Process has been terminated" << endl;
        return 0;
    }

    originalFile.seekg(0, ios::end);
    long int originalFileSize = originalFile.tellg();
    originalFile.seekg(0, ios::beg);
    std::cout << "The size of the sum of ORIGINAL files is: "
              << originalFileSize << " bytes" << endl;

    // Histograming the frequency of bytes.
    bool isOdd = originalFileSize % 2 == 1;
    unsigned char lastByte = 0;

    // std::vector<unsigned char> fileData(originalFileSize);
    unsigned char *fileData = nullptr;
    cudaHostAlloc((void **)&fileData, originalFileSize * sizeof(unsigned char), cudaHostAllocDefault);

    originalFile.read(reinterpret_cast<char *>(fileData), originalFileSize);
    originalFile.close();

    if (isOdd)
    {
        lastByte = fileData[originalFileSize - 1];
    }

    // Start timer.
    struct timeval start, end;
    gettimeofday(&start, NULL);

    unsigned char *d_fileData;
    unsigned int *d_freqCount;
    cudaMalloc(&d_fileData, originalFileSize * sizeof(unsigned char));
    cudaMalloc(&d_freqCount, kMaxSymbolSize * sizeof(unsigned int));
    cudaMemset(d_freqCount, 0, kMaxSymbolSize * sizeof(unsigned int));

    cudaMemcpy(d_fileData, fileData,
               originalFileSize * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (originalFileSize / 2 + blockSize - 1) / blockSize;
    calculateFrequency<<<numBlocks, blockSize>>>(d_fileData, originalFileSize,
                                                 d_freqCount);

    std::vector<unsigned int> freqCount(kMaxSymbolSize);
    cudaMemcpy(freqCount.data(), d_freqCount,
               kMaxSymbolSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_fileData);

    unsigned int *d_nonZeroCount;
    int numBlocksForUniqueSymbols = (kMaxSymbolSize + blockSize - 1) / blockSize;
    cudaMalloc(&d_nonZeroCount, numBlocksForUniqueSymbols * sizeof(unsigned int));

    countNonZero<<<numBlocksForUniqueSymbols, blockSize, blockSize * sizeof(unsigned int)>>>(d_freqCount, kMaxSymbolSize, d_nonZeroCount);

    std::vector<unsigned int> nonZeroCount(numBlocksForUniqueSymbols);
    cudaMemcpy(nonZeroCount.data(), d_nonZeroCount, numBlocksForUniqueSymbols * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int uniqueSymbolCount = 0;
    for (int i = 0; i < numBlocksForUniqueSymbols; i++)
    {
        uniqueSymbolCount += nonZeroCount[i];
    }
    std::cout << "Unique symbols count: " << uniqueSymbolCount << endl;

    // Free unused memory.
    cudaFreeHost(fileData);

    thrust::device_vector<unsigned int> d_freqCountVec(d_freqCount, d_freqCount + kMaxSymbolSize);
    thrust::device_vector<unsigned int> indicesVec(kMaxSymbolSize);

    thrust::sequence(thrust::device, indicesVec.begin(), indicesVec.end());
    thrust::sort_by_key(d_freqCountVec.begin(), d_freqCountVec.end(), indicesVec.begin());

    std::vector<unsigned int> sortedFreqCount(kMaxSymbolSize);
    std::vector<unsigned int> sortedIndices(kMaxSymbolSize);
    thrust::copy(d_freqCountVec.begin(), d_freqCountVec.end(), sortedFreqCount.begin());
    thrust::copy(indicesVec.begin(), indicesVec.end(), sortedIndices.begin());

    // End timer
    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    std::cout << "Time taken to calculate frequency: " << elapsedTime << " ms"
              << endl;

    cudaMemcpy(d_freqCount, sortedFreqCount.data(),
               kMaxSymbolSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int threadsPerBlock;
    std::tie(numBlocks, threadsPerBlock) = queryOptimalThreadsPerBlock(uniqueSymbolCount);

    // Step 1: Initialize workspace
    GpuHuffmanWorkspace workspace(uniqueSymbolCount, threadsPerBlock);

    // Step 2: Gpu Code Word construction
    auto codewords = gpuCodebookConstruction(
        d_freqCount + kMaxSymbolSize - uniqueSymbolCount, uniqueSymbolCount,
        std::move(workspace), 0);

    cuda_check(cudaFree(d_freqCount));

    // TODO: Remove this
    // Step 1: Initialize the leaf nodes for Huffman tree construction.
    // Each leaf node represents a unique byte and its frequency in the input
    // data. TreeNode nodesForHuffmanTree[uniqueSymbolCount * 2 - 1];
    TreeNode *nodesForHuffmanTree = new TreeNode[uniqueSymbolCount * 2 - 1];
    TreeNode *currentNode = nodesForHuffmanTree;

    // Step 2: Fill the array with data for each unique byte.
    for (unsigned int *frequency = freqCount.data();
         frequency < freqCount.data() + kMaxSymbolSize; frequency++)
    {
        if (*frequency)
        {
            currentNode->right = NULL;
            currentNode->left = NULL;
            currentNode->occurrences = *frequency;
            currentNode->character = frequency - freqCount.data();
            currentNode++;
        }
    }

    // Step 3: Sort the leaf nodes based on frequency to prepare for tree
    // construction. In ascending order.
    sort(nodesForHuffmanTree, nodesForHuffmanTree + uniqueSymbolCount,
         TreeNodeCompare);

    string scompressed = argv[1];
    scompressed += ".compressed";
    FILE *compressedFilePtr = fopen(&scompressed[0], "wb");

    // Writing the first piece of header information: the count of unique bytes.
    // This count is essential for reconstructing the Huffman tree during the
    // decompression process.
    fwrite(&uniqueSymbolCount, 2, 1, compressedFilePtr);

    // Write last byte information.

    fwrite(&isOdd, 1, 1, compressedFilePtr);
    if (isOdd)
    {
        // Write the last byte.
        fwrite(&lastByte, 1, 1, compressedFilePtr);
    }

    int bitCounter = 0;
    unsigned char bufferByte = 0;
    // Array to store transformation strings for each unique character to
    // optimize compression.

    std::vector<std::string> transformationStrings(kMaxSymbolSize);
    // Iterate through each node in the Huffman tree to write transformation
    // codes to the compressed file.
    int nodeIndex = 0;
    for (auto const &CW : codewords)
    {
        // Store the transformation string for the current character in the
        // array.
        auto node = nodesForHuffmanTree[nodeIndex];
        nodeIndex++;
        auto character = node.character;
        transformationStrings[character] = CW;
        unsigned char transformationLength = CW.length();
        unsigned short currentCharacter = character;
        // Write the current character and its transformation string length
        // to the compressed file.
        writeFromUShort(currentCharacter, bufferByte, bitCounter,
                        compressedFilePtr);
        writeFromUChar(transformationLength, bufferByte, bitCounter,
                       compressedFilePtr);

        // Write the transformation string bit by bit to the compressed
        // file.
        char const *transformationStringPtr = &CW[0];
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

    FILE *originalFilePtr = fopen(argv[1], "rb");
    // Writing the size of the file, its name, and its content in the
    // compressed format.
    writeFileSize(originalFileSize, bufferByte, bitCounter, compressedFilePtr);
    writeFileContent(originalFilePtr, originalFileSize,
                     transformationStrings.data(), bufferByte, bitCounter,
                     compressedFilePtr);
    fclose(originalFilePtr);

    // Ensuring the last byte is written to the compressed file by aligning
    // the bit counter.
    if (bitCounter > 0)
    {
        bufferByte <<= (8 - bitCounter);
        fwrite(&bufferByte, 1, 1, compressedFilePtr);
    }

    fclose(compressedFilePtr);

    // Get the size of compressed file.
    long int compressedFileSize = sizeOfTheFile(&scompressed[0]);
    std::cout << "The size of the COMPRESSED file is: " << compressedFileSize
              << " bytes" << endl;

    // Calculate the compression ratio.
    float compressionRatio = 100.0f * static_cast<float>(compressedFileSize) /
                             static_cast<float>(originalFileSize);
    std::cout << "Compressed file's size is [" << compressionRatio
              << "%] of the original files." << endl;

    // Warning if the compressed file is unexpectedly larger than the
    // original sum.
    if (compressedFileSize > originalFileSize)
    {
        std::cout << "\nWARNING: The compressed file's size is larger than the "
                     "sum of the originals.\n\n";
    }

    std::cout << endl
              << "Created compressed file: " << scompressed << endl;
    std::cout << "Compression is complete" << endl;
}

// below function is used for writing the uChar to compressed file
// It does not write it directly as one byte instead it mixes uChar and current
// byte, writes 8 bits of it and puts the rest to curent byte for later use
void writeFromUChar(unsigned char byteToWrite, unsigned char &bufferByte,
                    int bitCounter, FILE *filePtr)
{
    // Going to write at least 1 byte, first shift the bufferByte to the left
    // to make room for the new byte.
    bufferByte <<= 8 - bitCounter;
    bufferByte |= (byteToWrite >> bitCounter);
    fwrite(&bufferByte, 1, 1, filePtr);
    bufferByte = byteToWrite;
}

void writeFromUShort(unsigned short shortToWrite, unsigned char &bufferByte,
                     int bitCounter, FILE *filePtr)
{
    unsigned char firstByte = (shortToWrite >> 8) & 0xFF; // High byte
    unsigned char secondByte = shortToWrite & 0xFF;       // Low byte

    writeFromUChar(firstByte, bufferByte, bitCounter, filePtr);
    writeFromUChar(secondByte, bufferByte, bitCounter, filePtr);
}

// This function is writing byte count of current input file to compressed file
// using 8 bytes It is done like this to make sure that it can work on little,
// big or middle-endian systems
void writeFileSize(long int fileSize, unsigned char &bufferByte, int bitCounter,
                   FILE *filePtr)
{
    for (int i = 0; i < 8; i++)
    {
        writeFromUChar(fileSize % 256, bufferByte, bitCounter, filePtr);
        fileSize /= 256;
    }
}

// Below function translates and writes bytes from current input file to the
// compressed file.
void writeFileContent(FILE *originalFilePtr, long int originalFileSize,
                      string *transformationStrings, unsigned char &bufferByte,
                      int &bitCounter, FILE *compressedFilePtr)
{
    unsigned short readBuf;
    unsigned char *readBufPtr;
    readBufPtr = (unsigned char *)&readBuf;
    while (fread(readBufPtr, 2, 1, originalFilePtr))
    {
        char *strPointer = &transformationStrings[readBuf][0];
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
    }
}

long int sizeOfTheFile(char *path)
{
    ifstream file(path, ifstream::ate | ifstream::binary);
    return file.tellg();
}

void writeIfFullBuffer(unsigned char &bufferByte, int &bitCounter,
                       FILE *filePtr)
{
    if (bitCounter == 8)
    {
        fwrite(&bufferByte, 1, 1, filePtr);
        bitCounter = 0;
    }
}

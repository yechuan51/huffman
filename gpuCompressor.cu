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

#include "gpuHuffmanConstruction.h"

#define BLOCK_SIZE 256

using namespace std;

void writeFromUShort(unsigned short, unsigned char &, int, FILE *);
void writeFromUChar(unsigned char, unsigned char &, int, FILE *);
long int sizeOfTheFile(char *);
void writeFileSize(long int, unsigned char &, int, FILE *);
void writeFileContent(FILE *, long int, string *, unsigned char &, int &,
                      FILE *);
void writeIfFullBuffer(unsigned char &, int &, FILE *);

struct TreeNode {  // this structure will be used to create the translation tree
    TreeNode *left, *right;
    unsigned int occurrences;
    unsigned short character;
    string bit;
};

bool TreeNodeCompare(TreeNode a, TreeNode b) {
    return a.occurrences < b.occurrences;
}

__global__ void calculateFrequency(const unsigned char *data, long int size,
                                   unsigned int *freqCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size / 2; i += stride) {
        unsigned short readBuf = (data[i * 2 + 1] << 8) | data[i * 2];
        atomicAdd(&freqCount[readBuf], 1);
    }
}

__global__ void populateCWLength(const unsigned char *data, long int size, const int* transformationLengths,
                            unsigned int* CW_length){
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;
     
    for (int i = index; i < size / 2; i += stride) {
        unsigned short symbol = (data[i * 2 + 1] << 8) | data[i * 2];
        CW_length[i + 1] = transformationLengths[symbol];
    }
}

__global__ void findOffset(unsigned int *input, unsigned int *output, int n) {
    __shared__ unsigned int temp[BLOCK_SIZE * 2];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    if (start + t < n)
        temp[t] = input[start + t];
    else
        temp[t] = 0;

    if (start + blockDim.x + t < n)
        temp[blockDim.x + t] = input[start + blockDim.x + t];
    else
        temp[blockDim.x + t] = 0;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (t + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x)
            temp[index] += temp[index - stride];
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (t + 1) * stride * 2 - 1;
        if (index + stride < 2 * blockDim.x)
            temp[index + stride] += temp[index];
    }

    __syncthreads();
    if (start + t < n)
        output[start + t] = temp[t];
    if (start + blockDim.x + t < n)
        output[start + blockDim.x + t] = temp[blockDim.x + t];
}

//__global__ void addBlockSum(unsigned int *output, unsigned int *lastOutput, int blockSize, int n) {
__global__ void addBlockSum(unsigned int *output, unsigned int *input, int blockSize, int n) {
    int blockSum = 0;
    int blockStart = blockIdx.x * blockSize * 2;
    if(blockStart < n){
        for(int i = blockStart - 1; i >= 0; i -= blockSize * 2){
            blockSum += input[i];
        }
        // First half of block
        int index = blockStart + threadIdx.x;
        if (index < n) {
            output[index] = input[index] + blockSum;
        }
        // Second half of block
        index += blockSize;
        if (index < n) {
            output[index] = input[index] + blockSum;
        }

        // if (index == n - 1){
        //     lastOutput[0] = output[index];
        // }
    }

    __syncthreads();
}

__global__ void encodeFromSW(const unsigned char *data, long int size, const unsigned char** transformationStrings, unsigned int *offset,
                             uint8_t *output) {
    //extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads two characters from file (since codeword is upto 16 bits), convert to corresponding CW
    // Then store it to shared memory
    // if (index < size / 2) {
    //     unsigned short symbol = (data[index * 2 + 1] << 8) | data[index * 2];
    //     unsigned char* cw_string = transformationStrings[symbol]; // say 10110
    //     int cw_int = 0;
    //     for(int i = 0; cw_string[i] != '\0'; i++){
    //         cw_int = (cw_int << 1) | cw_string[i];
    //     }
    //     //sdata[tid] = (index < size / 2) ? cw_int : "";
    // }
    __syncthreads();

    // Do reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            // for(int i = sdata[tid + s]; i > 0; i >> 1){
            //     //sdata[tid] = (sdata[tid] << 1) | (i | 1);
            // }
            //sdata[tid] 
        }
    }
    
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Must provide a single file name." << endl;
        return 0;
    }

    static constexpr int kMaxSymbolSize = 65536;
    ifstream originalFile(argv[1], ios::binary);
    if (!originalFile.is_open()) {
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

    std::vector<unsigned char> fileData(originalFileSize);
    originalFile.read(reinterpret_cast<char *>(&fileData[0]), originalFileSize);
    originalFile.close();

    if (isOdd) {
        lastByte = fileData[originalFileSize - 1];
    }

    unsigned char *d_fileData;
    unsigned int *d_freqCount;
    cudaMalloc(&d_fileData, originalFileSize * sizeof(unsigned char));
    cudaMalloc(&d_freqCount, kMaxSymbolSize * sizeof(unsigned int));
    cudaMemset(d_freqCount, 0, kMaxSymbolSize * sizeof(unsigned int));

    cudaMemcpy(d_fileData, fileData.data(),
               originalFileSize * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (originalFileSize / 2 + blockSize - 1) / blockSize;
    calculateFrequency<<<numBlocks, blockSize>>>(d_fileData, originalFileSize,
                                                 d_freqCount);

    std::vector<unsigned int> freqCount(kMaxSymbolSize);
    cudaMemcpy(freqCount.data(), d_freqCount,
               kMaxSymbolSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // TODO: d_fileData will need to be used again in encoding stage
    // cudaFree(d_fileData);

    unsigned int uniqueSymbolCount = 0;
    for (int i = 0; i < 65536; i++) {
        if (freqCount[i]) {
            uniqueSymbolCount++;
        }
    }
    std::cout << "Unique symbols count: " << uniqueSymbolCount << endl;

    // Free unused memory.
    fileData.clear();

    // TODO: Remove this
    // Step 1: Initialize the leaf nodes for Huffman tree construction.
    // Each leaf node represents a unique byte and its frequency in the input
    // data. TreeNode nodesForHuffmanTree[uniqueSymbolCount * 2 - 1];
    TreeNode *nodesForHuffmanTree = new TreeNode[uniqueSymbolCount * 2 - 1];
    TreeNode *currentNode = nodesForHuffmanTree;

    // Step 2: Fill the array with data for each unique byte.
    for (unsigned int *frequency = freqCount.data();
         frequency < freqCount.data() + kMaxSymbolSize; frequency++) {
        if (*frequency) {
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

    std::sort(
        freqCount.begin(), freqCount.end(),
        [](unsigned int const &a, unsigned int const &b) { return a < b; });

    cudaMemcpy(d_freqCount, freqCount.data(),
               kMaxSymbolSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    static constexpr int kThreadsPerBlock = 256;

    // Step 1: Initialize workspace
    GpuHuffmanWorkspace workspace(uniqueSymbolCount, kThreadsPerBlock);

    // Step 2: Gpu Code Word construction
    auto codewords = gpuCodebookConstruction(
        d_freqCount + kMaxSymbolSize - uniqueSymbolCount, uniqueSymbolCount,
        std::move(workspace), 0);

    cuda_check(cudaFree(d_freqCount));

    string scompressed = argv[1];
    scompressed += ".compressed";
    FILE *compressedFilePtr = fopen(&scompressed[0], "wb");

    // Writing the first piece of header information: the count of unique bytes.
    // This count is essential for reconstructing the Huffman tree during the
    // decompression process.
    fwrite(&uniqueSymbolCount, 2, 1, compressedFilePtr);

    // Write last byte information.

    fwrite(&isOdd, 1, 1, compressedFilePtr);
    if (isOdd) {
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
    for (auto const &CW : codewords) {
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
        while (*transformationStringPtr) {
            bufferByte <<= 1;
            if (*transformationStringPtr == '1') {
                bufferByte |= 1;
            }
            bitCounter++;
            transformationStringPtr++;
            writeIfFullBuffer(bufferByte, bitCounter, compressedFilePtr);
        }
    }

    //std::cout << "Starting Encoding Process" << std::endl;
    
    // codewords should be used here to write compressed file

    size_t totalChars = 0;

    std::vector<int> h_transformationLengths;
    unsigned int* h_data_lengths = (unsigned int *) malloc((originalFileSize + 1)/2 * sizeof(int));
    unsigned int* h_offsets = (unsigned int *) malloc((originalFileSize + 1)/2 * sizeof(int));
    unsigned int* h_lastOffset = (unsigned int*) malloc(sizeof(int));   // For tracking the size of compiled file
    uint8_t *h_encode_buffer;

    h_transformationLengths.reserve(transformationStrings.size()); // Reserve space to avoid unnecessary reallocations
    std::transform(transformationStrings.begin(), transformationStrings.end(), std::back_inserter(h_transformationLengths), [](const std::string& str) {
        return str.length();
    });

    for(const int len: h_transformationLengths){
        totalChars += len; 
    }
    //std::cout << std::endl;
    //std::cout << "num chars = " << totalChars << std::endl;
    int* d_transformationLengths;
    unsigned int* d_data_lengths;
    unsigned int* d_offsets_t; // Transitional array, which is used to calculate final d_offsets
    unsigned int* d_offsets; 
    unsigned int* d_lastOffset; // For tracking the size of compiled file
    unsigned char** d_transformationStrings;
    uint8_t *d_encode_buffer;
    //std::cout << "originalFileSize = " << originalFileSize << std::endl;
    cudaMalloc((void**) &d_transformationLengths, transformationStrings.size() * sizeof(int));
    cudaMalloc((void**) &d_data_lengths, originalFileSize/2 * sizeof(int));
    cudaMalloc((void**) &d_offsets_t, originalFileSize/2 * sizeof(int));
    cudaMalloc((void**) &d_offsets, originalFileSize/2 * sizeof(int));
    cudaMalloc((void**) &d_lastOffset, sizeof(int));
    cudaMalloc((void**) &d_transformationStrings, totalChars * sizeof(unsigned char));

    cudaMemcpy(d_transformationLengths, h_transformationLengths.data(),
               transformationStrings.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_transformationStrings, transformationStrings.data(),
               totalChars * sizeof(char),
               cudaMemcpyHostToDevice);

    //std::cout << "Starting Findoffset Kernel" << std::endl;
    cudaMemset(d_data_lengths, 0, sizeof(int)); // Set the first value to be 0 for offset purposes
    populateCWLength<<<numBlocks, blockSize>>>(d_fileData, originalFileSize, d_transformationLengths, d_data_lengths);
    //cudaDeviceSynchronize();
    findOffset<<<numBlocks, blockSize>>>(d_data_lengths, d_offsets_t, originalFileSize/2);
    cudaDeviceSynchronize();
    addBlockSum<<<numBlocks, blockSize>>>(d_offsets, d_offsets_t, blockSize, originalFileSize/2);
    //addBlockSum<<<numBlocks, blockSize>>>(d_offsets, d_lastOffset, blockSize, (originalFileSize + 1)/2);
    cudaDeviceSynchronize();
   
    // cudaMemcpy( h_lastOffset, d_lastOffset, sizeof(int), cudaMemcpyDeviceToHost );
    // // h_lastOffset contains the number of bits that needs to be allocated.
    // if(h_lastOffset[0] % 8 != 0){
    //     h_lastOffset[0] += 8 - h_lastOffset[0] % 8;
    // }
    // h_lastOffset[0] += 16; // account for the last CW length

    // cudaMalloc((void**) &d_encode_buffer, h_lastOffset[0]);
    // h_encode_buffer = (uint8_t*) malloc(h_lastOffset[0]);

    //encodeFromSW<<<numBlocks, blockSize>>>(d_fileData, originalFileSize, d_transformationStrings, d_offsets, d_encode_buffer);


    //std::cout << "Finished Findoffset Kernel" << std::endl;
    
    // TODO: Remove this normally, use for debugging
	cudaMemcpy( h_data_lengths, d_data_lengths, originalFileSize/2 * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_offsets, d_offsets, originalFileSize/2 * sizeof(int), cudaMemcpyDeviceToHost );
    //cudaMemcpy( h_encode_buffer, d_encode_buffer, h_lastOffset[0], cudaMemcpyDeviceToHost);

    // for(long int i = 0; i < originalFileSize/2; i++){
    //     std::cout << h_data_lengths[i] << " ";
    // }
    // std::cout << std::endl;


    int sum = 0;
    for(long int i = 0; i < originalFileSize/2; i++){
        //std::cout << h_offsets[i] << " ";
        if (sum != h_offsets[i]){
            std::cout << std::endl;
            std::cout << "h_offsets[i] does not match sum (h_offsets[i], sum, index): (" << h_offsets[i] << ", " << sum << ", " << i << ")" << std::endl;
            break;
        }
        unsigned short readBuf = (fileData.data()[i * 2 + 1] << 8) | fileData.data()[i * 2];
        sum += h_transformationLengths[readBuf];
    }
    std::cout << std::endl;

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
    if (bitCounter > 0) {
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
    if (compressedFileSize > originalFileSize) {
        std::cout << "\nWARNING: The compressed file's size is larger than the "
                     "sum of the originals.\n\n";
    }

    std::cout << endl << "Created compressed file: " << scompressed << endl;
    std::cout << "Compression is complete" << endl;
}

// below function is used for writing the uChar to compressed file
// It does not write it directly as one byte instead it mixes uChar and current
// byte, writes 8 bits of it and puts the rest to curent byte for later use
void writeFromUChar(unsigned char byteToWrite, unsigned char &bufferByte,
                    int bitCounter, FILE *filePtr) {
    // Going to write at least 1 byte, first shift the bufferByte to the left
    // to make room for the new byte.
    bufferByte <<= 8 - bitCounter;
    bufferByte |= (byteToWrite >> bitCounter);
    fwrite(&bufferByte, 1, 1, filePtr);
    bufferByte = byteToWrite;
}

void writeFromUShort(unsigned short shortToWrite, unsigned char &bufferByte,
                     int bitCounter, FILE *filePtr) {
    unsigned char firstByte = (shortToWrite >> 8) & 0xFF;  // High byte
    unsigned char secondByte = shortToWrite & 0xFF;        // Low byte

    writeFromUChar(firstByte, bufferByte, bitCounter, filePtr);
    writeFromUChar(secondByte, bufferByte, bitCounter, filePtr);
}

// This function is writing byte count of current input file to compressed file
// using 8 bytes It is done like this to make sure that it can work on little,
// big or middle-endian systems
void writeFileSize(long int fileSize, unsigned char &bufferByte, int bitCounter,
                   FILE *filePtr) {
    for (int i = 0; i < 8; i++) {
        writeFromUChar(fileSize % 256, bufferByte, bitCounter, filePtr);
        fileSize /= 256;
    }
}

// Below function translates and writes bytes from current input file to the
// compressed file.
void writeFileContent(FILE *originalFilePtr, long int originalFileSize,
                      string *transformationStrings, unsigned char &bufferByte,
                      int &bitCounter, FILE *compressedFilePtr) {
    unsigned short readBuf;
    unsigned char *readBufPtr;
    readBufPtr = (unsigned char *)&readBuf;
    // While loop reads the original file for current character (readBufPtr)
    while (fread(readBufPtr, 2, 1, originalFilePtr)) {
        char *strPointer = &transformationStrings[readBuf][0];
        while (*strPointer) {
            writeIfFullBuffer(bufferByte, bitCounter, compressedFilePtr);

            bufferByte <<= 1;
            if (*strPointer == '1') {
                bufferByte |= 1;
            }
            bitCounter++;
            strPointer++;
        }
    }
}

long int sizeOfTheFile(char *path) {
    ifstream file(path, ifstream::ate | ifstream::binary);
    return file.tellg();
}

void writeIfFullBuffer(unsigned char &bufferByte, int &bitCounter,
                       FILE *filePtr) {
    if (bitCounter == 8) {
        fwrite(&bufferByte, 1, 1, filePtr);
        bitCounter = 0;
    }
}

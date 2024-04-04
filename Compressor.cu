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

#define BLOCK_SIZE 256

using namespace std;

void writeFromUShort(unsigned short, unsigned char &, int, FILE *);
void writeFromUChar(unsigned char, unsigned char &, int, FILE *);
long int sizeOfTheFile(char *);
void writeFileSize(long int, unsigned char &, int, FILE *);
void writeFileContent(FILE *, long int, string *, unsigned char &, int &,
                      FILE *);
void writeFileContent(uint8_t*, long int, long int, unsigned char &, int&, 
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

__global__ void populateCWLength(const unsigned char *data, long int size, const int* transformationLengths,
                            unsigned int* CWLengths){
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;
     
    for (int i = index; i < size / 2; i += stride) {
        unsigned short symbol = (data[i * 2 + 1] << 8) | data[i * 2];
        CWLengths[i + 1] = transformationLengths[symbol]; // i + 1 to let CW_lengths[0] be bitCounter
    }
}

__global__ void findOffset(unsigned int *input, int n,
                        unsigned int *output) {
    __shared__ unsigned int temp[BLOCK_SIZE * 2];                  
    int chunk = blockDim.x * gridDim.x;
    
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    for (int i = start; i < n; i += chunk){
        if(i + t < n){
            temp[t] = input[i + t];
        } else {
            temp[t] = 0;
        }

        if (i + blockDim.x + t < n)
            temp[blockDim.x + t] = input[i + blockDim.x + t];
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
        if (i + t < n)
            output[i + t] = temp[t];
        if (i + blockDim.x + t < n)
            output[i + blockDim.x + t] = temp[blockDim.x + t];
    }
}

__global__ void addBlockSum( unsigned int *input, int blockSize, int n,
                        unsigned int *output) {
    int stride = blockDim.x * gridDim.x;

    int blockStart = blockIdx.x * blockSize * 2;
    for(int i = blockStart; i < n; i += stride){
        int blockSum = 0;
        for(int j = i - 1; j >= 0; j -= blockSize * 2){
            blockSum += input[j];
        }
        // First half of block
        int index = i + threadIdx.x;
        if (index < n) {
            output[index] = input[index] + blockSum;
        }
        // Second half of block
        index += blockSize;
        if (index < n) {
            output[index] = input[index] + blockSum;
        }
    }
}

__device__ int binarySearch(const unsigned int* offsets, int numOffsets, int target){
    int low = 0;
    int high = numOffsets - 1;
    int leftIndex = -1;

    while (low <= high){
        int mid = low + (high - low) / 2;
        
        if(offsets[mid] == target){
            return mid;
        }
        else if(offsets[mid] < target){
            //update leftIndex and search upper half
            leftIndex = mid;
            low = mid + 1;
        } else {
            //search the lower half
            high = mid - 1;
        }
    }
    return leftIndex;
}

__global__ void encodeFromCW(const unsigned char *data, long int originalFileSize, unsigned char bufferByte,
                            char* transformationStrings, const int* transformationLengths, const int *transformationStringsOffset, 
                            const unsigned int *CW_offsets, const long int compressedFileSize, 
                            uint8_t *outputs) {
    int start = blockIdx.x * blockDim.x + threadIdx.x; // example index = 88, index * 8 = 704
    int stride = blockDim.x * gridDim.x;
    for (int index = start; index * 8 < compressedFileSize; index += stride)
    {
        uint8_t out = 0;
        // These two index can be used 
        int offset_index_left = binarySearch(CW_offsets, originalFileSize/2+1, index*8); // examply CW_offsets[offset_index_start] = 697
        int offset_index_right = offset_index_left + 1; // CW_offsets[offset_index_end] = example 709

        if(offset_index_left >= 0){
            // the transformation string related to the character to the left of index in offsets array
            unsigned short symbol_left = (data[offset_index_left * 2 + 1] << 8) | data[offset_index_left * 2];
            int t_string_left_offset = transformationStringsOffset[symbol_left];
            int t_string_left_length = transformationLengths[symbol_left];
            if( offset_index_right < originalFileSize/2){
                // the transformation string related to the character to the right of index in offsets array
                unsigned short symbol_right = (data[offset_index_right * 2 + 1] << 8) | data[offset_index_right * 2];
                int t_string_right_offset = transformationStringsOffset[symbol_right];
                int t_string_right_length = transformationLengths[symbol_right];

                int n = CW_offsets[offset_index_right] - index*8;
                if (n <= 8) {
                    // Take the last n bits of the t_string_left.
                    int bitToTake = 8;
                    for(int i = n - 1; i >= 0; i--){
                        out = out << 1;
                        if(transformationStrings[t_string_left_offset + t_string_left_length - 1 - i] == '1'){
                            out = out | 1;
                        } 
                        bitToTake -= 1;
                    }

                    while(bitToTake > 0){
                        int rightSideLen = bitToTake <= t_string_right_length ? bitToTake: t_string_right_length;
                        for(int i = 0; i < rightSideLen; i++){
                            out = out << 1;
                            if(transformationStrings[t_string_right_offset + i] == '1'){
                                out = out | 1;
                            }
                            bitToTake -= 1;
                        }
                        if(bitToTake > 0){
                            offset_index_right += 1; //Prepare for next while loop iteration IF needed
                            symbol_right = (data[offset_index_right * 2 + 1] << 8) | data[offset_index_right * 2];
                            t_string_right_offset = transformationStringsOffset[symbol_right];
                            t_string_right_length = transformationLengths[symbol_right];
                        }
                    }            
                }
                else {
                    // Take 8 bits from t_string_left, starting from t_string_left_offset + left_shift
                    int left_shift = index*8 - CW_offsets[offset_index_left];
                    for(int i = 0; i < 8; i++){
                        out = out << 1;
                        if(transformationStrings[t_string_left_offset + left_shift + i] == '1'){
                            out = out | 1;
                        }
                    }
                }
            } else {
                // The last CW
                int n = CW_offsets[offset_index_left] + transformationLengths[symbol_left] - index*8;
                if (n <= 8) {
                    // Take the last n bits of the t_string_left.
                    for(int i = n - 1; i >= 0; i--){
                        out = out << 1;
                        if(transformationStrings[t_string_left_offset + t_string_left_length - 1 - i] == '1'){
                            out = out | 1;
                        } 
                    }
                } else {
                    // Take 8 bits from t_string_left, starting from t_string_left_offset + left_shift
                    int left_shift = index*8 - CW_offsets[offset_index_left];
                    for(int i = 0; i < 8; i++){
                        out = out << 1;
                        if(transformationStrings[t_string_left_offset + left_shift + i] == '1'){
                            out = out | 1;
                        } 
                    }
                }

            }
            
        } else {
            // Only here if it's the first character (index = 0) and CW_offests[0] > 0
            unsigned short symbol_right = (data[1] << 8) | data[0];
            int t_string_right_offset = transformationStringsOffset[symbol_right];
            int t_string_right_length = transformationLengths[symbol_right];

            out = out | bufferByte;
            for(int i = 0; i < 8 - CW_offsets[0]; i++){
                out = out << 1;
                if(transformationStrings[t_string_right_offset + i] == '1'){
                    out = out | 1;
                }
            }
        }
        outputs[index] = out;
    }
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

    unsigned char *fileData = nullptr;
    cudaHostAlloc((void **)&fileData, originalFileSize * sizeof(unsigned char), cudaHostAllocDefault);

    originalFile.read(reinterpret_cast<char *>(fileData), originalFileSize);
    originalFile.close();

    if (isOdd)
    {
        lastByte = fileData[originalFileSize - 1];
    }

    // ---------------------- HISTOGRAM ----------------------

    // Start timer
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

    int blockSize = BLOCK_SIZE;
    int numBlocks = (originalFileSize / 2 + blockSize - 1) / blockSize;
    calculateFrequency<<<numBlocks, blockSize>>>(d_fileData, originalFileSize,
                                                 d_freqCount);

    std::vector<unsigned int> freqCount(kMaxSymbolSize);
    cudaMemcpy(freqCount.data(), d_freqCount,
               kMaxSymbolSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    thrust::device_vector<unsigned int> d_freqCountVec(d_freqCount, d_freqCount + kMaxSymbolSize);
    unsigned int uniqueSymbolCount = thrust::count_if(
        thrust::device,
        d_freqCountVec.begin(),
        d_freqCountVec.end(),
        thrust::placeholders::_1 > 0);

    std::cout << "Unique symbols count: " << uniqueSymbolCount << endl;

    thrust::device_vector<unsigned int> indicesVec(kMaxSymbolSize);
    thrust::sequence(thrust::device, indicesVec.begin(), indicesVec.end());
    thrust::sort_by_key(d_freqCountVec.begin(), d_freqCountVec.end(), indicesVec.begin());

    std::vector<unsigned int> sortedIndices(kMaxSymbolSize);
    thrust::copy(d_freqCountVec.begin(), d_freqCountVec.end(), freqCount.begin());
    thrust::copy(indicesVec.begin(), indicesVec.end(), sortedIndices.begin());

    // Stop timer
    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    std::cout << "Histograming took " << elapsedTime << " ms" << endl;

    // --------------------- END OF HISTOGRAM ---------------------

    cudaMemcpy(d_freqCount, freqCount.data(),
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
    for (size_t i = 0; i < freqCount.size(); ++i)
    {
        if (freqCount[i] != 0)
        {
            currentNode->right = nullptr;
            currentNode->left = nullptr;
            currentNode->occurrences = freqCount[i];
            currentNode->character = static_cast<unsigned short>(sortedIndices[i]);
            currentNode++;
        }
    }

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

    // Writing the size of the file, its name, and its content in the
    // compressed format.
    writeFileSize(originalFileSize, bufferByte, bitCounter, compressedFilePtr);

    // ---------------------- ENCODING ----------------------
    struct timeval start_encode, end_encode;
    gettimeofday(&start_encode, NULL);
    // Calculating and Writing the content of the compressed file
    std::vector<int> h_transformationLengths;
    std::vector<int> h_transformationStringOffsets;
    unsigned int* h_data_lengths = (unsigned int *) malloc((originalFileSize/2 + 1) * sizeof(int));
    unsigned int* h_last_CW_length = (unsigned int*) malloc(sizeof(int));   // For tracking the last CW length to calculate total size of compressed file
    unsigned int* h_offsets = (unsigned int *) malloc((originalFileSize/2 + 1) * sizeof(int));
    unsigned int* h_lastOffset = (unsigned int*) malloc(sizeof(int));   // For tracking the total offsets to calculate total size of compressed file
    uint8_t *h_encode_buffer;

    h_transformationLengths.reserve(transformationStrings.size()); // Reserve space to avoid unnecessary reallocations
    std::transform(transformationStrings.begin(), transformationStrings.end(), std::back_inserter(h_transformationLengths), [](const std::string& str) {
        return str.length();
    });

    long int totalChars = 0;
    for(const int len: h_transformationLengths){
        h_transformationStringOffsets.push_back(totalChars);
        totalChars += len; 
    }

    int* d_transformationLengths;
    unsigned int* d_data_lengths;
    unsigned int* d_offsets_t; // Transitional array, which is used to calculate final d_offsets
    unsigned int* d_offsets; 
    char* d_transformationStringsPool;
    int* d_transformationStringOffsets;
    uint8_t *d_encode_buffer;

    cudaMalloc((void**) &d_transformationLengths, transformationStrings.size() * sizeof(int));
    cudaMalloc((void**) &d_data_lengths, (originalFileSize/2 + 1) * sizeof(int));
    cudaMalloc((void**) &d_offsets_t, (originalFileSize/2 + 1) * sizeof(int));
    cudaMalloc((void**) &d_offsets, (originalFileSize/2 + 1) * sizeof(int));
    cudaMalloc((void**) &d_transformationStringsPool, totalChars * sizeof(unsigned char));
    cudaMalloc((void**) &d_transformationStringOffsets, transformationStrings.size() * sizeof(int));

    
    cudaMemcpy(d_transformationLengths, h_transformationLengths.data(),
               transformationStrings.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    
    
    string transformationStringPool = "";
    for(int i = 0; i < kMaxSymbolSize; i++){
        transformationStringPool += transformationStrings[i];
    }
    const char* h_transformationStringPool = transformationStringPool.c_str(); 

    cudaMemcpy(d_transformationStringsPool, h_transformationStringPool,
               totalChars * sizeof(char),
               cudaMemcpyHostToDevice);


    cudaMemcpy(d_transformationStringOffsets, h_transformationStringOffsets.data(),
               transformationStrings.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(&d_data_lengths[0], &bitCounter, sizeof(int), cudaMemcpyHostToDevice); // Set the first value to be the bitCounter for offset purposes

    populateCWLength<<<numBlocks, blockSize>>>(d_fileData, originalFileSize, d_transformationLengths, d_data_lengths);

    findOffset<<<numBlocks, blockSize>>>(d_data_lengths, (originalFileSize/2 + 1), d_offsets_t);

    cudaDeviceSynchronize();
    addBlockSum<<<numBlocks, blockSize>>>(d_offsets_t, blockSize, (originalFileSize/2 + 1), d_offsets);

    cudaDeviceSynchronize();
   
    cudaMemcpy( h_lastOffset, d_offsets + originalFileSize/2, sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy( h_last_CW_length, d_data_lengths + originalFileSize/2, sizeof(int), cudaMemcpyDeviceToHost );
    // h_lastOffset contains the number of bits that needs to be allocated.
    long int compressedContentFileSize = h_lastOffset[0];
    long int compressedContentFileSizeAllocation = compressedContentFileSize;


    if(compressedContentFileSizeAllocation % 8 != 0){
        compressedContentFileSizeAllocation += 8 - (compressedContentFileSizeAllocation % 8);
    }

    cudaMalloc((void**) &d_encode_buffer, compressedContentFileSizeAllocation);
    h_encode_buffer = (uint8_t*) malloc(compressedContentFileSizeAllocation);
    std::cout << "Number of bytes allocated for h_encode_buffer: " << compressedContentFileSizeAllocation << std::endl;

    encodeFromCW<<<numBlocks, blockSize>>>(d_fileData, originalFileSize, bufferByte,
                                            d_transformationStringsPool, d_transformationLengths, d_transformationStringOffsets, 
                                            d_offsets, compressedContentFileSizeAllocation, 
                                            d_encode_buffer);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();

    cuda_check(cudaMemcpy( h_encode_buffer, d_encode_buffer, compressedContentFileSize, cudaMemcpyDeviceToHost));

    writeFileContent(h_encode_buffer, compressedContentFileSize, compressedContentFileSizeAllocation, bufferByte, bitCounter, compressedFilePtr);

    gettimeofday(&end_encode, NULL);
    double encode_elapsedTime = (end_encode.tv_sec - start_encode.tv_sec) * 1000.0;
    encode_elapsedTime += (end_encode.tv_usec - start_encode.tv_usec) / 1000.0;
    std::cout << "Encoding took " << encode_elapsedTime << " ms" << endl;

    // Ensuring the last byte is written to the compressed file by aligning
    // the bit counter.
    if (bitCounter > 0)
    {
        bufferByte <<= (8 - bitCounter);
        fwrite(&bufferByte, 1, 1, compressedFilePtr);
    }

    fclose(compressedFilePtr);

    // Free unused memory.
    cudaFree(d_fileData);
    cudaFreeHost(fileData);
    cudaFree(d_transformationLengths);
    cudaFree(d_data_lengths);
    cudaFree(d_offsets_t);
    cudaFree(d_offsets);
    cudaFree(d_transformationStringsPool);
    cudaFree(d_transformationStringOffsets);
    cudaFree(d_encode_buffer);
   
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
void writeFileContent(uint8_t* encode_buffer, long int fileSize, long int fileSizeAllocated, unsigned char &bufferByte, int&bitCounter, FILE *compressedFilePtr) {
    // Note that all previous bits in buffer is written to file, update bitCounter and bufferByte
    uint8_t last_byte = encode_buffer[fileSizeAllocated/8 - 1];
    long int extraAllocation = fileSizeAllocated - fileSize; // Possible values from 0 to 7 inclusive
    long int bytesToWrite = extraAllocation == 0 ? fileSizeAllocated/8 : fileSizeAllocated/8 - 1;
    bitCounter = extraAllocation == 0 ? 0 : 8 - extraAllocation;
    for(int i = 0; i < bitCounter; i++){
        bufferByte <<= 1;
        if((last_byte & 1) == 1){
            bufferByte |= 1;
        }
        last_byte >>= 1;
    }
    
    fwrite(encode_buffer, bytesToWrite, 1, compressedFilePtr); 
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

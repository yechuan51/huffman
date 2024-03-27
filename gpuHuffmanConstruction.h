#include <assert.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <functional>

static const bool pagelocked_mem = true;
static const bool verbose = true;
static const bool debug = true;
static const bool micro_benchmark = false;

#define cuda_check(_x)                                     \
    do {                                                   \
        cudaError_t _err = (_x);                           \
        if (_err != cudaSuccess) {                         \
            printf("Error: cuda error %s(%d) occurred.\n", \
                   cudaGetErrorString(_err), int(_err));   \
            abort();                                       \
        }                                                  \
    } while (0)

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

struct timer {
    timer(cudaStream_t stream = 0);
    ~timer();
    void start();
    float stop();

   private:
    cudaStream_t m_stream;
    cudaEvent_t m_start, m_stop;
};

timer::timer(cudaStream_t stream) : m_stream{stream} {
    cuda_check(cudaEventCreate(&m_start));
    cuda_check(cudaEventCreate(&m_stop));
}

timer::~timer() {
    cuda_check(cudaEventDestroy(m_start));
    cuda_check(cudaEventDestroy(m_stop));
}

void timer::start() { cudaEventRecord(m_start, m_stream); }

float timer::stop() {
    float time;
    cudaEventRecord(m_stop, m_stream);
    cudaEventSynchronize(m_stop);
    cudaEventElapsedTime(&time, m_start, m_stop);
    return time;
}

struct __align__(16) Node {
    int index;
    int left;
    int right;
    int parent;
};

struct Barrier {
    __device__ Barrier(int* count, int n, int* sense)
        : _count{count},
          _n{n},
          _sense{sense},
          _local_sense{0},
          _wait_thread{threadIdx.x == 0} {}

    __device__ __forceinline__ void wait() {
        int state;
        fetch(_count, state);
        while (__syncthreads_and(state != _n - 1)) {
            fetch(_count, state);
        }

        __syncthreads();
    }

    __device__ __forceinline__ void fetch(int* pointer, int& state) {
        if (_wait_thread) {
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                         : "=r"(state)
                         : "l"(pointer));
        }
    }

    __device__ __forceinline__ void fence() {
        __syncthreads();
        int state = _local_sense;
        int s = _local_sense ^ 1;
        _local_sense = s;
        fetch(_sense, state);
        if (blockIdx.x < _n - 1) {
            if (_wait_thread) {
                atomicAdd(_count, 1);
            }
            while (__syncthreads_and(state != s)) {
                fetch(_sense, state);
            }
            __syncthreads();
        } else if (blockIdx.x == _n - 1) {
            wait();
            if (_wait_thread) {
                asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
                             :
                             : "l"(_count), "r"(0));
                asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
                             :
                             : "l"(_sense), "r"(s));
            }
        }
    }
    int* _count;
    int _n;
    int* _sense;
    int _local_sense;
    bool _wait_thread;
};

template <typename F>
__host__ __device__ __forceinline__ int BinarySearch(F const* freq, int size,
                                                     F val) {
    int l = 0;
    int r = size - 1;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (freq[m] <= val) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

template <typename T>
__host__ __device__ __forceinline__ void Swap(T& left, T& right) {
    T tmp = left;
    left = right;
    right = tmp;
}

#define SERIAL_MERGE 0
#define MERGE_PER_THREADS_1 1

template <typename F>
__host__ __device__ __forceinline__ int2 KthElement(F const* leftFreq,
                                                    int leftSize,
                                                    F const* rightFreq,
                                                    int rightSize, int k) {
    int kth = k;
    int leftIndex = 0;
    int rightIndex = 0;

    for (;;) {
#if MERGE_PER_THREADS_1
        if (leftSize == leftIndex) {
            return make_int2(-1, rightIndex + kth);
        } else if (rightSize == rightIndex) {
            return make_int2(leftIndex + kth, -1);
        }
#else
        if (leftSize == leftIndex) {
            return make_int2(leftIndex, rightIndex + kth);
        } else if (rightSize == rightIndex) {
            return make_int2(leftIndex + kth, rightIndex);
        }
#endif
        int mid1 = leftIndex + (leftSize - leftIndex) / 2;
        int mid2 = rightIndex + (rightSize - rightIndex) / 2;
#ifndef __CUDA_ARCH__
        printf("k %d idx %d %d mid %d %d size %d %d freq %u %u\n", kth,
               leftIndex, rightIndex, mid1, mid2, leftSize, rightSize,
               leftFreq[leftIndex], rightFreq[rightIndex]);
#endif
        if (mid1 - leftIndex + mid2 - rightIndex < kth) {
            if (leftFreq[mid1] > rightFreq[mid2]) {
                kth = kth - (mid2 - rightIndex) - 1;
                rightIndex = mid2 + 1;
            } else {
                kth = kth - (mid1 - leftIndex) - 1;
                leftIndex = mid1 + 1;
            }
        } else {
            if (leftFreq[mid1] > rightFreq[mid2]) {
                leftSize = mid1;
            } else {
                rightSize = mid2;
            }
        }
    }
}

// template <typename F>
// __host__ __device__ __forceinline__ int2 KthElement(F const* leftFreq,
//                                                     int leftSize,
//                                                     F const* rightFreq,
//                                                     int rightSize, int k) {
//     int kth = k;
//     int thisIndex = 0;
//     int otherIndex = 0;
//     int thisSize = leftSize;
//     int otherSize = rightSize;
//     F const* thisFreq = leftFreq;
//     F const* otherFreq = rightFreq;
//
//     F pivot = 0;
//     int pivotIndex = -1;
//     for (;;) {
//         pivot = thisFreq[thisIndex];
//         pivotIndex = BinarySearch(otherFreq + otherIndex, otherSize, pivot);
//         if (pivotIndex < kth) {
//             if (pivotIndex == 0 && otherFreq[otherIndex] < pivot) {
//                 otherIndex = -1;
//                 thisIndex = thisIndex + kth - 1;
//                 break;
//             }
//             kth -= pivotIndex;
//             otherSize -= pivotIndex;
//             otherIndex += pivotIndex;
//         } else if (pivotIndex >= kth) {
//             thisIndex = -1;
//             otherIndex = otherIndex + kth;
//             break;
//         }
//         // printf("pivot: %d\n", pivot);
//         // printf("pivotIndex: %d\n", pivotIndex);
//         // printf("kth: %d\n", kth);
//         // printf("thisSize: %d\n", thisSize);
//         // printf("otherSize: %d\n", otherSize);
//         // printf("thisIndex: %d\n", thisIndex);
//         // printf("otherIndex: %d\n", otherIndex);
//         if (thisFreq[thisIndex] <= otherFreq[otherIndex]) {
//             Swap(thisFreq, otherFreq);
//             Swap(thisSize, otherSize);
//             Swap(thisIndex, otherIndex);
//         }
//     }
//     if (thisFreq == leftFreq) {
//         return make_int2(thisIndex, otherIndex);
//     } else {
//         return make_int2(otherIndex, thisIndex);
//     }
// }

template <typename F>
__device__ __forceinline__ void ParallelMerge(
    F const* leftFreq, int const* leftIndex, int leftSize, F const* rightFreq,
    int const* rightIndex, int rightSize, F* mergeFreq, int* mergeIndex,
    int mergePerThread, int2* partitionIndex, Barrier& barrier) {
    int bid = blockIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalNumThreads = blockDim.x * gridDim.x;

#if SERIAL_MERGE
    if (tid == 0) {
        int i = 0;
        int j = 0;
        int k = 0;

        while (i < leftSize || j < rightSize) {
            if (i >= leftSize) {
                mergeFreq[k] = rightFreq[j];
                mergeIndex[k] = rightIndex[j];
                j++;
                k++;
            } else if (j >= rightSize) {
                mergeFreq[k] = leftFreq[i];
                mergeIndex[k] = leftIndex[i];
                i++;
                k++;
            } else if (leftFreq[i] < rightFreq[j]) {
                mergeFreq[k] = leftFreq[i];
                mergeIndex[k] = leftIndex[i];
                i++;
                k++;
            } else {
                mergeFreq[k] = rightFreq[j];
                mergeIndex[k] = rightIndex[j];
                j++;
                k++;
            }
        }
    }
#else

    //    if (tid < participants) {
    //    auto kth = KthElement(leftFreq, leftSize, rightFreq, rightSize,
    //                          tid * mergePerThread);
    //    partitionIndex[tid] = kth;
    //    }

    //    barrier.fence();
    for (int i = tid * mergePerThread; i < leftSize + rightSize; i += totalNumThreads * mergePerThread) {
        // int2 start = partitionIndex[tid];
        // int2 end = partitionIndex[tid + 1];

        int2 start = KthElement(leftFreq, leftSize, rightFreq, rightSize, i);

#if MERGE_PER_THREADS_1
        if (start.x == -1) {
            mergeFreq[i] = rightFreq[start.y];
            mergeIndex[i] = rightIndex[start.y];
        } else {
            mergeFreq[i] = leftFreq[start.x];
            mergeIndex[i] = leftIndex[start.x];
        }
#else
        int2 end = KthElement(leftFreq, leftSize, rightFreq, rightSize, i + mergePerThread);

        for (; i < (tid + 1) * mergePerThread; ++i) {
            if (start.x >= end.x) {
                mergeFreq[i] = rightFreq[start.y];
                mergeIndex[i] = rightIndex[start.y];
                start.y++;
            } else if (start.y >= end.y) {
                mergeFreq[i] = leftFreq[start.x];
                mergeIndex[i] = leftIndex[start.x];
                start.x++;
            } else if (leftFreq[start.x] > rightFreq[start.y]) {
                mergeFreq[i] = rightFreq[start.y];
                mergeIndex[i] = rightIndex[start.y];
                start.y++;
            } else {
                mergeFreq[i] = leftFreq[start.x];
                mergeIndex[i] = leftIndex[start.x];
                start.x++;
            }
        }
#endif
    }
#endif
    //    barrier.fence();
}

template <typename F>
__global__ void GenerateCL(int* CL, F const* histogram, int size,
                           /* global variables */ F* nodeFreq, F* tempFreq,
                           int* nodeIndex, int* tempIndex, Node* nodes,
                           /* barrier */
                           int* count, int* sense,
                           /* parallel merge */ int2* partitionIndex,
                           /* CL legnth */ int* reduceCL, int* maxCL) {
    int bid = blockIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalNumThreads = gridDim.x * blockDim.x;
    Barrier barrier{count, gridDim.x, sense};

    for (int i = tid; i < size; i += totalNumThreads) {
        Node nd;
        nd.index = i;
        nd.left = -1;
        nd.right = -1;
        nd.parent = -1;
        int4* pointer = reinterpret_cast<int4*>(nodes + i);
        *pointer = *reinterpret_cast<int4*>(&nd);
        nodeFreq[i] = histogram[i];
        nodeIndex[i] = i;
    }
    int numCurrentNodes = size;
    barrier.fence();

    int sizeLeft = size;
    while (sizeLeft > 1) {
        F specFreq = nodeFreq[0] + nodeFreq[1];
        //    if (tid == 0) printf("%d %d %d\n", nodeFreq[0], nodeFreq[1],
        //    specFreq);

        int pivot = BinarySearch(nodeFreq + 2, sizeLeft - 2, specFreq);
        pivot += 2;
        pivot = pivot - (pivot & 0x1);
        //  if (tid == 0) printf("%d\n", pivot);

	for (int i = tid; i < sizeLeft - pivot; i += totalNumThreads) {
            tempFreq[i] = nodeFreq[i + pivot];
            tempIndex[i] = nodeIndex[i + pivot];
        }

	for (int i = tid; i < (pivot >> 1); i += totalNumThreads) {
            int left = nodeIndex[i * 2];
            int right = nodeIndex[i * 2 + 1];
            {
                Node node;
                *reinterpret_cast<int4*>(&node) =
                    *reinterpret_cast<int4*>(&nodes[left]);
                node.parent = numCurrentNodes + i;
                *reinterpret_cast<int4*>(nodes + left) =
                    reinterpret_cast<int4&>(node);
            }
            {
                Node node;
                *reinterpret_cast<int4*>(&node) =
                    *reinterpret_cast<int4*>(&nodes[right]);
                node.parent = numCurrentNodes + i;
                *reinterpret_cast<int4*>(nodes + right) =
                    reinterpret_cast<int4&>(node);
            }

            Node nd;
            nd.index = -1;
            nd.left = left;
            nd.right = right;
            nd.parent = -1;
            int4* pointer =
                reinterpret_cast<int4*>(nodes + numCurrentNodes + i);
            *pointer = reinterpret_cast<int4&>(nd);
            tempFreq[sizeLeft - pivot + i] =
                nodeFreq[i * 2] + nodeFreq[i * 2 + 1];
            tempIndex[sizeLeft - pivot + i] = numCurrentNodes + i;
        }
        numCurrentNodes += (pivot >> 1);

        barrier.fence();
#if MERGE_PER_THREADS_1
        int mergePerThread = 1;
#else
        int mergePerThread = 4;
#endif
        int mergeSize = sizeLeft - pivot + (pivot >> 1);
        ParallelMerge(tempFreq, tempIndex, sizeLeft - pivot,
                      tempFreq + sizeLeft - pivot, tempIndex + sizeLeft - pivot,
                      (pivot >> 1), nodeFreq, nodeIndex,
                      mergePerThread, partitionIndex, barrier);

        sizeLeft = sizeLeft - pivot + (pivot >> 1);

        barrier.fence();
    }
    for (int i = tid; i < size; i += totalNumThreads) {
        int4* pointer = reinterpret_cast<int4*>(nodes + i);
        Node cur;
        *reinterpret_cast<int4*>(&cur) = *pointer;
        int parent = cur.parent;
        int length = 0;
        while (parent != -1) {
            // if (tid == 0) {
            // printf("parent %d\n", parent);
            // }
            length++;
            int4* pointer = reinterpret_cast<int4*>(nodes + parent);
            Node cur;
            *reinterpret_cast<int4*>(&cur) = *pointer;
            parent = cur.parent;
        }
        CL[i] = length;
	if (i == 0) {
            *maxCL = length;
	}
    }
}

__global__ void GenerateCW(Node const* nodes, uint8_t* CW, int size,
                           int codeLengthPerWord) {
    int bid = blockIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int4 const* pointer = reinterpret_cast<int4 const*>(nodes + tid);
        Node cur;
        *reinterpret_cast<int4*>(&cur) = *pointer;
        uint8_t* cw = CW + tid * codeLengthPerWord;
        int child = tid;
        int parent = cur.parent;
        int length = 0;
        while (parent != -1) {
            int4 const* pointer = reinterpret_cast<int4 const*>(nodes + parent);
            Node cur;
            *reinterpret_cast<int4*>(&cur) = *pointer;
            if (child == cur.left) {
                cw[length] = 0;
            } else if (child == cur.right) {
                cw[length] = 1;
            }
            child = parent;
            parent = cur.parent;
            length++;
        }
    }
}

static void test_serial_merge() {
    int n = 100;
    int na = 8;
    int nb = 20;
    std::vector<int> a(na);
    std::vector<int> b(nb);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, n);
    for (auto& ele : a) {
        ele = dist(rng);
    }
    for (auto& ele : b) {
        ele = dist(rng);
    }
    for (int i = 0; i < a.size(); ++i) {
        printf("%d %c", a[i], i == a.size() - 1 ? '\n' : ' ');
    }
    for (int i = 0; i < b.size(); ++i) {
        printf("%d %c", b[i], i == b.size() - 1 ? '\n' : ' ');
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    for (int i = 0; i < a.size(); ++i) {
        printf("%d %c", a[i], i == a.size() - 1 ? '\n' : ' ');
    }
    for (int i = 0; i < b.size(); ++i) {
        printf("%d %c", b[i], i == b.size() - 1 ? '\n' : ' ');
    }

    int pivot = BinarySearch(a.data(), a.size(), 30);
    printf("val: 30, pos: %d\n", pivot);
    int2 kth = KthElement(a.data(), a.size(), b.data(), b.size(), 4);
    printf("%d %d\n", kth.x, kth.y);

    kth = KthElement(a.data(), a.size(), b.data(), b.size(), 12);
    printf("%d %d\n", kth.x, kth.y);
}

template <typename T>
using GpuMemory = std::unique_ptr<T, std::function<void(void const*)>>;

class GpuCodewords {
   public:
    GpuCodewords(int codeLengthPerWord, int symbolSize)
        : _codeLengthPerWord(codeLengthPerWord), _symbolSize(symbolSize) {
        uint8_t* codewords;
        cuda_check(cudaMalloc(
            &codewords, sizeof(uint8_t) * codeLengthPerWord * symbolSize));
        auto CudaFree = [](void const* pointer) {
            cuda_check(cudaFree(const_cast<void*>(pointer)));
        };
        _codewords = {codewords, CudaFree};
    }

    std::vector<std::string> toCpu(int* CL) {
        std::vector<std::string> transformationStrings;
        int* h_CL = (int*)malloc(sizeof(int) * _symbolSize);
        uint8_t* h_codewords = (uint8_t*)(malloc(
            sizeof(uint8_t) * _codeLengthPerWord * _symbolSize));
        cuda_check(
            cudaMemcpy(h_codewords, pointer(),
                       sizeof(uint8_t) * _codeLengthPerWord * _symbolSize,
                       cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(h_CL, CL, sizeof(int) * _symbolSize,
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < _symbolSize; ++i) {
            int length = h_CL[i];
            std::string codeString;
            for (int j = 0; j < length; j++) {
                uint8_t code = h_codewords[i * _codeLengthPerWord + j];
                if (code == 0) {
                    codeString += "1";
                } else if (code == 1) {
                    codeString += "0";
                }
            }
            std::reverse(codeString.begin(), codeString.end());
            transformationStrings.push_back(codeString);
        }
        free(h_codewords);
        free(h_CL);
        return transformationStrings;
    }

    uint8_t const* pointer() const { return _codewords.get(); }
    uint8_t* pointer() { return _codewords.get(); }

   private:
    int _codeLengthPerWord;
    int _symbolSize;
    GpuMemory<uint8_t> _codewords;
};

class GpuHuffmanWorkspace {
   public:
    GpuHuffmanWorkspace(int symbolSize, int threadsPerBlock)
        : _symbolSize(symbolSize), _threadsPerBlock(threadsPerBlock) {
        int gridDim = (_symbolSize + _threadsPerBlock - 1) / _threadsPerBlock;
        auto CudaFree = [](void const* pointer) {
            cuda_check(cudaFree(const_cast<void*>(pointer)));
        };
        int workspaceSize = sizeof(Node) * 2 * symbolSize +
                            sizeof(int2) * symbolSize +
                            sizeof(unsigned int) * symbolSize * 2 +
                            sizeof(int) * symbolSize * 3 +
                            sizeof(int) * gridDim + sizeof(int) * 3;
        int* workspacePtr;
        cuda_check(cudaMalloc((void**)(&workspacePtr), workspaceSize));
        _workspace = {workspacePtr, CudaFree};
        _nodes = reinterpret_cast<Node*>(_workspace.get());
        _partitionIndex = reinterpret_cast<int2*>(_nodes + 2 * symbolSize);
        _nodeFreq =
            reinterpret_cast<unsigned int*>(_partitionIndex + symbolSize);
        _tempFreq = _nodeFreq + symbolSize;
        _nodeIndex = reinterpret_cast<int*>(_tempFreq + symbolSize);
        _tempIndex = _nodeIndex + symbolSize;
        _CL = _tempIndex + symbolSize;
        _reduceCL = _CL + symbolSize;
        _count = _reduceCL + gridDim;
        _sense = _count + 1;
        _maxCL = _sense + 1;
    }

    Node* nodes() { return _nodes; }
    int2* partitionIndex() { return _partitionIndex; }
    unsigned int* nodeFreq() { return _nodeFreq; }
    unsigned int* tempFreq() { return _tempFreq; }
    int* nodeIndex() { return _nodeIndex; }
    int* tempIndex() { return _tempIndex; }
    int* CL() { return _CL; }
    int* reduceCL() { return _reduceCL; }
    int* count() { return _count; }
    int* sense() { return _sense; }
    int* maxCL() { return _maxCL; }

   private:
    int _symbolSize;
    int _threadsPerBlock;
    Node* _nodes;
    int2* _partitionIndex;
    unsigned int* _nodeFreq;
    unsigned int* _tempFreq;
    int* _nodeIndex;
    int* _tempIndex;
    int* _CL;
    int* _reduceCL;
    int* _count;
    int* _sense;
    int* _maxCL;
    GpuMemory<int> _workspace;
};

static int minmumPowOfTwo(int n) {
    n = n - 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n < 0 ? 1 : n + 1;
}	

struct SmemGetter {
    typedef int (*FuncType)(int blockSize, void* userData);
    FuncType _func;
    void* _userData;

    SmemGetter(FuncType func = 0, void* userData = 0) : _func(func), _userData(userData) {} 
};

struct SmemGetterWrapper {
    SmemGetter getter;

    __device__ __host__ int operator()(int blockSize) const {
#if __CUDA_ARCH__
  	 int* ptr = 0;
	 *ptr = 23;
#else
	 if (getter._func) {
             return getter._func(blockSize, getter._userData);
	 }
#endif
	 return 0;
    }
};

static std::tuple<int, int> queryOptimalThreadsPerBlock(int symbolSize) {
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0));
    int smCount = prop.multiProcessorCount;
    SmemGetterWrapper s;
    s.getter = SmemGetter();
    int threadsPerBlock;
    int numBlocks;
    int blocksPerSM;
    cuda_check(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&numBlocks, &threadsPerBlock, GenerateCL<unsigned int>, s));
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, GenerateCL<unsigned int>, threadsPerBlock, 0)); 
    int maxNumBlocks = blocksPerSM * smCount;
    if (maxNumBlocks < numBlocks) {
        return {numBlocks, threadsPerBlock}; 
    } else {
        return {maxNumBlocks, threadsPerBlock};
    }  
}

static std::vector<std::string> gpuCodebookConstruction(unsigned int* frequencies, int symbolSize,
							GpuHuffmanWorkspace workspace, cudaStream_t stream) {
    timer tm(stream);
    tm.start();
    {

	int threadsPerBlock;
	int numBlocks;
	std::tie(numBlocks, threadsPerBlock) = queryOptimalThreadsPerBlock(symbolSize);
	printf("threadsPerBlock: %d\n", threadsPerBlock);
	printf("numBlocks: %d\n", numBlocks);
        dim3 blockDim(numBlocks);
        dim3 threadDim(threadsPerBlock);
        cuda_check(cudaMemsetAsync(workspace.count(), 0, 2 * sizeof(int)));
	GenerateCL<unsigned int>
            <<<blockDim, threadDim, 0, stream>>>(
                workspace.CL(), frequencies, symbolSize, workspace.nodeFreq(),
                workspace.tempFreq(), workspace.nodeIndex(),
                workspace.tempIndex(), workspace.nodes(), workspace.count(),
                workspace.sense(), workspace.partitionIndex(),
                workspace.reduceCL(), workspace.maxCL());
	cuda_check(cudaGetLastError());
    }
    int maxCL;
    cuda_check(cudaMemcpyAsync(&maxCL, workspace.maxCL(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    if (false && debug) {
        printf("write file\n");
        unsigned* h_nodeFreq = new unsigned[symbolSize];
        unsigned* h_tempFreq = new unsigned[symbolSize];
        int* h_nodeIndex = new int[symbolSize];
        int* h_tempIndex = new int[symbolSize];
        cuda_check(cudaMemcpy(h_nodeFreq, workspace.nodeFreq(),
                              sizeof(unsigned) * symbolSize,
                              cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(h_tempFreq, workspace.tempFreq(),
                              sizeof(unsigned) * symbolSize,
                              cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(h_nodeIndex, workspace.nodeIndex(),
                              sizeof(int) * symbolSize,
                              cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(h_tempIndex, workspace.tempIndex(),
                              sizeof(int) * symbolSize,
                              cudaMemcpyDeviceToHost));
        auto dump = [&](int k, unsigned* data, int size) {
            std::string file = std::to_string(k) + ".txt";
            FILE* f = fopen(file.c_str(), "w");
            int cnt = 0;
            for (int i = 0; i < symbolSize; ++i) {
                if (data[i] == 0) continue;
                if (data[i] == 8) cnt++;
                fprintf(f, "%u\n", data[i]);
            }
            fclose(f);
            printf("%d %d\n", k, cnt);
        };
        static int i = 0;
        i++;
        dump(i, h_nodeFreq, symbolSize);
        i++;
        auto kth = KthElement(h_tempFreq, 116, h_tempFreq, 70, 139);
        printf("%d %d\n", kth.x, kth.y);
        printf("%u\n", h_tempFreq[70]);
        dump(i, h_tempFreq, symbolSize);
        i++;
        std::sort(h_tempFreq, h_tempFreq + symbolSize);
        dump(i, h_tempFreq, symbolSize);
    }

    // printf("maxCL %d\n", maxCL);
    int codeLengthPerWord = maxCL;
    auto codewords = GpuCodewords(codeLengthPerWord, symbolSize);

    {
        static constexpr int kThreadsPerBlock = 256;
        dim3 blockDim((symbolSize + kThreadsPerBlock - 1) / kThreadsPerBlock);
        dim3 threadDim(kThreadsPerBlock);

        GenerateCW<<<blockDim, threadDim, 0, stream>>>(
            workspace.nodes(), codewords.pointer(), symbolSize,
            codeLengthPerWord);
        cuda_check(cudaGetLastError());
    }
    cuda_check(cudaDeviceSynchronize());
    float elapsedTime = tm.stop();
    printf("construction time: %.3f ms, symbols/s: %.3f\n", elapsedTime,
           (float)(symbolSize) / (elapsedTime * 1e-3));
    return codewords.toCpu(workspace.CL());
}

#include <cstdio>
#include <mpi.h>
#include <cuda.h>
#include <string>
#include <ctime>
#include <chrono>
#include <queue>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

// #include "graph.h"
// #include "bfsCPU.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


#define DeviceNum 2

struct Graph {
    std::vector<int> adjacencyList; // all edges
    std::vector<int> edgesOffset; // offset to adjacencyList for every vertex
    std::vector<int> edgesSize; //number of edges for every vertex
    int numVertices = 0;
    int numEdges = 0;
};
Graph G;
int getDev(int which)
{
    return which / (G.numVertices / DeviceNum);
}


void readGraph(Graph &G, int argc, char **argv);
void readGraphFromFile(Graph &G, int argc, char **argv);
void bfsCPU(int start, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited);


inline __device__
int* getLevel(int which, int **d_distance, int nodenum, int devicenum)
{
    // printf("get level: %d %d\n", devicenum, which - ((nodenum / DeviceNum) *devicenum ));
    return &(d_distance[devicenum][which - ((nodenum / DeviceNum) *devicenum)]);
}

// void *args[] = {&begin, &end, &G.numVertices, &level, &d_adjacencyList[i], &d_edgesOffset[i], &d_edgesSize[i], &d_distance[i], &d_parent[i],
//                         &changed};



// __global__
// void multiBfs(int begin, int end, int nodenum, int level, int deviceid, int *d_adjacencyList, int* d_edgesOffset,
//     int *d_edgesSize, int **d_distance, int **d_parent, int *changed) {
//     int threadid = blockIdx.x * blockDim.x + threadIdx.x;
//     int valueChange = 0;
//     int u = threadid + begin;
//     // int u = threadid;
//     //printf("blockid=%d blockdim=%d threadid=%d dev=%d, u=%d\n",blockIdx.x, blockDim.x, threadIdx.x,deviceid, u);
//      if(u == 2)
//      {
//      	 //*changed = 1;
//          printf("blockid=%d blockdim=%d threadid=%d dev=%d, u=%d \n",blockIdx.x, blockDim.x, threadIdx.x, deviceid, u);
//          printf("inside u==2 d_edgesize=%d\n", d_edgesSize[u]);
//          for(int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; ++i)
//          {
//              int v = d_adjacencyList[i];
//              printf("from %d to %d on dev %d dis=%d end=%d %d \n",u, v, deviceid, d_distance[0][u], end, d_distance[0][0]);
//          }
//      }
//      int mylevel = -1;
//      if(u < end){
// 	     mylevel = *getLevel(u, d_distance, nodenum, deviceid);
// 	     //if(u == 0)
// 	     printf("u=%d mylevel=%d\n",u, mylevel);
//      }
//     if(u < end && mylevel == level)
//     {
//     	printf("u=%d\n", u);
//         for(int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; ++i)
//         {
//             int v = d_adjacencyList[i];
//             if(level + 1 < *getLevel(v, d_distance, nodenum, deviceid))
//             {
//     		printf("v=%d\n", v);
//                 *getLevel(v, d_distance, nodenum, deviceid) = level + 1;
//                 *getLevel(v, d_parent, nodenum, deviceid) = i;
//                 valueChange = 1;
//             }
//         }
//     }
//     if(valueChange){
//         *changed = valueChange;
//     }
// }




__global__
void multiBfs(int begin, int end, int nodenum, int level, int deviceid, int *d_adjacencyList, int* d_edgesOffset,
    int *d_edgesSize, int **d_distance, int **d_parent, int *changed) {
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;
    int u = threadid + begin;
    // int u = threadid;
    //printf("blockid=%d blockdim=%d threadid=%d dev=%d, u=%d\n",blockIdx.x, blockDim.x, threadIdx.x,deviceid, u);
     int mylevel = -1;
     if(u < end){
	     mylevel = *getLevel(u, d_distance, nodenum, deviceid);
	     //printf("u=%d mylevel=%d\n",u, mylevel);
     }
    if(u < end && mylevel == level)
    {
        for(int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; ++i)
        {
            int v = d_adjacencyList[i];
            if(level + 1 < *getLevel(v, d_distance, nodenum, deviceid))
            {
                *getLevel(v, d_distance, nodenum, deviceid) = level + 1;
                *getLevel(v, d_parent, nodenum, deviceid) = i;
                valueChange = 1;
            }
        }
    }
    if(valueChange){
        *changed = valueChange;
    }
}



__global__
void queueBfs(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
              int queueSize, int *nextQueueSize, int *d_currentQueue, int *d_nextQueue, int part, int deviceid, int numVertices) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;


    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        // printf("dev %d searching %d\n",deviceid, u);
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            // printf("out v=%d dev=%d size=%d dis=%d\n", v, deviceid, d_edgesSize[u], d_distance[v]);
            if (d_distance[v] == INT_MAX && atomicMin(&d_distance[v], level + 1) == INT_MAX) {
                d_parent[v] = i;
                int which = (v / part);
                int position = atomicAdd(&nextQueueSize[which], 1);
                d_nextQueue[position + which * numVertices] = v;
                // printf("new v=%d dev=%d size=%d dis=%d pos=%d which=%d part=%d\n", 
                //     v, deviceid, d_edgesSize[u], d_distance[v], position, which, part);
            }
        }
    }
    // __syncthreads();
    // if(thid == 0)
    // {
    //     for(int i = 0; i < 8; ++i)
    //     {
    //         printf("%d ", d_nextQueue[i]);
    //     }
    //     printf("dev=%d\n", deviceid);
    // }
}


// __global__
// void queueBfs(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
//               int queueSize, int *nextQueueSize, int *d_currentQueue, int *d_nextQueue, int part, int deviceid, int numVertices) {
//     int thid = blockIdx.x * blockDim.x + threadIdx.x;
//     int remote = (deviceid + 1) % 2;

//     if (thid < queueSize) {
//         int u = d_currentQueue[thid];
//         printf("searching %d\n", u);
//         for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
//             int v = d_adjacencyList[i];
//             printf("out v=%d dev=%d size=%d dis=%d\n", v, deviceid, d_edgesSize[u], d_distance[v]);
//             if (d_distance[v] == INT_MAX && atomicMin(&d_distance[v], level + 1) == INT_MAX) {
//                 d_parent[v] = i;
// 		        int which = (v / part);
//                 int position = atomicAdd(&nextQueueSize[which], 1);
//                 d_nextQueue[position + which * numVertices] = v;
//                 printf("new v=%d dev=%d size=%d dis=%d pos=%d which=%d part=%d\n", v, deviceid, d_edgesSize[u], d_distance[v], position, which, part);
//             }
//         }
//     }
// }



__global__
void queueBfsSingle(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
              int queueSize, int *nextQueueSize, int *d_currentQueue, int *d_nextQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == INT_MAX && atomicMin(&d_distance[v], level + 1) == INT_MAX) {
                d_parent[v] = i;
                int position = atomicAdd(nextQueueSize, 1);
                d_nextQueue[position] = v;
            }
        }
    }
}

void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(startVertex, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
}

void checkError(CUresult error, std::string msg) {
    if (error != CUDA_SUCCESS) {
        printf("%s: %d\n", msg.c_str(), error);
        exit(1);
    }
}

// CUdevice cuDevice;
// CUcontext cuContext;
// CUmodule cuModule;

// CUdevice cuDevice2[DeviceNum];
// CUcontext cuContext2[DeviceNum];
// CUmodule cuModule2[DeviceNum];

// CUfunction cuSimpleBfs;
// // CUfunction multiBfs[DeviceNum];
// CUfunction cuQueueBfs;
// CUfunction cuNextLayer;
// CUfunction cuCountDegrees;
// CUfunction cuScanDegrees;
// CUfunction cuAssignVerticesNextQueue;

// CUdeviceptr d_adjacencyList;
// CUdeviceptr d_edgesOffset;
// CUdeviceptr d_edgesSize;
// CUdeviceptr d_distance;
// CUdeviceptr d_parent;
// CUdeviceptr d_currentQueue;
// CUdeviceptr d_nextQueue;
// CUdeviceptr d_degrees;
int *incrDegrees;


__managed__ int* d_adjacencyList2[DeviceNum];
__managed__ int* d_edgesOffset2[DeviceNum];
__managed__ int* d_edgesSize2[DeviceNum];
__managed__ int* d_distance2[DeviceNum];
__managed__ int* d_parent2[DeviceNum];
__managed__ int* d_currentQueue2[DeviceNum];
// __managed__ int* d_nextQueue2[DeviceNum];
__managed__ int* d_nextQueue2[DeviceNum];
__managed__ int* d_degrees2[DeviceNum];
__managed__ int *incrDegrees2[DeviceNum];

__managed__ int* localQueue[DeviceNum][DeviceNum];
__managed__ int* remoteQueue[DeviceNum][DeviceNum];
__managed__ int nextQueueSize[DeviceNum][DeviceNum];
__managed__ int queueSize[DeviceNum];


// void initCuda(Graph &G) {
//     //initialize CUDA
//     cuInit(0);
//     checkError(cuDeviceGet(&cuDevice, 0), "cannot get device 0");
//     checkError(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
//     checkError(cuModuleLoad(&cuModule, "bfsCUDA.ptx"), "cannot load module");
//     checkError(cuModuleGetFunction(&cuSimpleBfs, cuModule, "simpleBfs"), "cannot get kernel handle");
//     checkError(cuModuleGetFunction(&cuQueueBfs, cuModule, "queueBfs"), "cannot get kernel handle");
//     checkError(cuModuleGetFunction(&cuNextLayer, cuModule, "nextLayer"), "cannot get kernel handle");
//     checkError(cuModuleGetFunction(&cuCountDegrees, cuModule, "countDegrees"), "cannot get kernel handle");
//     checkError(cuModuleGetFunction(&cuScanDegrees, cuModule, "scanDegrees"), "cannot get kernel handle");
//     checkError(cuModuleGetFunction(&cuAssignVerticesNextQueue, cuModule, "assignVerticesNextQueue"),
//                "cannot get kernel handle");

//     //copy memory to device
//     checkError(cuMemAlloc(&d_adjacencyList, G.numEdges * sizeof(int)), "cannot allocate d_adjacencyList");
//     checkError(cuMemAlloc(&d_edgesOffset, G.numVertices * sizeof(int)), "cannot allocate d_edgesOffset");
//     checkError(cuMemAlloc(&d_edgesSize, G.numVertices * sizeof(int)), "cannot allocate d_edgesSize");
//     checkError(cuMemAlloc(&d_distance, G.numVertices * sizeof(int)), "cannot allocate d_distance");
//     checkError(cuMemAlloc(&d_parent, G.numVertices * sizeof(int)), "cannot allocate d_parent");
//     checkError(cuMemAlloc(&d_currentQueue, G.numVertices * sizeof(int)), "cannot allocate d_currentQueue");
//     checkError(cuMemAlloc(&d_nextQueue, G.numVertices * sizeof(int)), "cannot allocate d_nextQueue");
//     checkError(cuMemAlloc(&d_degrees, G.numVertices * sizeof(int)), "cannot allocate d_degrees");
//     checkError(cuMemAllocHost((void **) &incrDegrees, sizeof(int) * G.numVertices), "cannot allocate memory");

//     checkError(cuMemcpyHtoD(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int)),
//                "cannot copy to d_adjacencyList");
//     checkError(cuMemcpyHtoD(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int)),
//                "cannot copy to d_edgesOffset");
//     checkError(cuMemcpyHtoD(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int)),
//                "cannot copy to d_edgesSize");


// }


void initCuda2(Graph &G) {
    //initialize CUDA
    // cuInit(0);
    // int i = 0;
    // // for(int i = 0 ; i < DeviceNum; ++i)
    // {
    //     checkError(cuDeviceGet(&cuDevice2[i], i), "cannot get device 0");
    //     checkError(cuCtxCreate(&cuContext2[i], 0, cuDevice), "cannot create context");
    //     checkError(cuModuleLoad(&cuModule2[i], "bfsCUDA.ptx"), "cannot load module");
    //     // checkError(cuModuleGetFunction(&cuSimpleBfs, cuModule2[i], "simpleBfs"), "cannot get kernel handle");
    //     checkError(cuModuleGetFunction(&multiBfs[i], cuModule2[i], "multiBfs"), "cannot get multi kernel handle");
    //     // checkError(cuModuleGetFunction(&cuQueueBfs, cuModule2[i], "queueBfs"), "cannot get kernel handle");
    //     // checkError(cuModuleGetFunction(&cuNextLayer, cuModule2[i], "nextLayer"), "cannot get kernel handle");
    //     // checkError(cuModuleGetFunction(&cuCountDegrees, cuModule2[i], "countDegrees"), "cannot get kernel handle");
    //     // checkError(cuModuleGetFunction(&cuScanDegrees, cuModule2[i], "scanDegrees"), "cannot get kernel handle");
    //     // checkError(cuModuleGetFunction(&cuAssignVerticesNextQueue, cuModule2[i], "assignVerticesNextQueue"),
    //                // "cannot get kernel handle");
    // }

    // printf("Enabling peer access between GPU%d and GPU%d...\n", 0, 1);
    // (cudaSetDevice(0));
    // (cudaDeviceEnablePeerAccess(1, 0));
    // (cudaSetDevice(1));
    // (cudaDeviceEnablePeerAccess(0, 0));
    for(int i =0; i < DeviceNum; ++i)
    {
        //cudaSetDevice(i);
        (cudaMalloc(&d_adjacencyList2[i], G.numEdges * sizeof(int)), "cannot allocate d_adjacencyList2");
        (cudaMalloc(&d_edgesOffset2[i], G.numVertices * sizeof(int)), "cannot allocate d_edgesOffset2");
        (cudaMalloc(&d_edgesSize2[i], G.numVertices * sizeof(int)), "cannot allocate d_edgesSize2");
        (cudaMalloc(&d_distance2[i], G.numVertices * sizeof(int)), "cannot allocate d_distance2");
        (cudaMalloc(&d_parent2[i], G.numVertices * sizeof(int)), "cannot allocate d_parent2");
        (cudaMalloc(&d_currentQueue2[i], G.numVertices * sizeof(int)), "cannot allocate d_currentQueue2");        
        (cudaMalloc(&d_degrees2[i], G.numVertices * sizeof(int)), "cannot allocate d_degrees2");
        (cudaMallocHost((void **) &incrDegrees2[i], sizeof(int) * G.numVertices), "cannot allocate memory");
        (cudaMalloc(&d_nextQueue2[i], 2 * G.numVertices * sizeof(int)), "cannot allocate d_nextQueue2");

        (cudaMemcpy(d_adjacencyList2[i], G.adjacencyList.data(), G.numEdges * sizeof(int), cudaMemcpyHostToDevice),
                   "cannot copy to d_adjacencyList2");
        (cudaMemcpy(d_edgesOffset2[i], G.edgesOffset.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
                   "cannot copy to d_edgesOffset2");
        (cudaMemcpy(d_edgesSize2[i], G.edgesSize.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
                   "cannot copy to d_edgesSize2");



        //(cudaMalloc(&localQueue[i], DeviceNum * sizeof(int*)), "cannot allocate localqueue");
        //(cudaMalloc(&remoteQueue[i], DeviceNum * sizeof(int*)), "cannot allocate localqueue");
    }


}

// void finalizeCuda() {
//     //free memory
//     checkError(cuMemFree(d_adjacencyList), "cannot free memory for d_adjacencyList");
//     checkError(cuMemFree(d_edgesOffset), "cannot free memory for d_edgesOffset");
//     checkError(cuMemFree(d_edgesSize), "cannot free memory for d_edgesSize");
//     checkError(cuMemFree(d_distance), "cannot free memory for d_distance");
//     checkError(cuMemFree(d_parent), "cannot free memory for d_parent");
//     checkError(cuMemFree(d_currentQueue), "cannot free memory for d_parent");
//     checkError(cuMemFree(d_nextQueue), "cannot free memory for d_parent");
//     checkError(cuMemFreeHost(incrDegrees), "cannot free memory for incrDegrees");
// }

void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
    for (int i = 0; i < G.numVertices; i++) {
        if (distance[i] != expectedDistance[i]) {
            printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
            printf("Wrong output!\n");
            exit(1);
        }
    }

    printf("Output OK!\n\n");
}

// void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
//     //initialize values
//     std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
//     std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
//     distance[startVertex] = 0;
//     parent[startVertex] = 0;

//     checkError(cuMemcpyHtoD(d_distance, distance.data(), G.numVertices * sizeof(int)),
//                "cannot copy to d)distance");
//     checkError(cuMemcpyHtoD(d_parent, parent.data(), G.numVertices * sizeof(int)),
//                "cannot copy to d_parent");

//     int firstElementQueue = startVertex;
//     cuMemcpyHtoD(d_currentQueue, &firstElementQueue, sizeof(int));
// }

void initializeCudaBfs2(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //initialize values
    std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
    std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    for(int i = 0; i < DeviceNum; ++i)
    {
        //cudaSetDevice(i);
        (cudaMemcpy(d_distance2[i], distance.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d)distance multi");
        (cudaMemcpy(d_parent2[i], parent.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
                   "cannot copy to d_parent multi");
        if(getDev(startVertex) == i){
            int firstElementQueue = startVertex;
            cudaMemcpy(d_currentQueue2[i], &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
        }
    }

}

 void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Graph &G, int rank=0) {
    if(rank == 1)
    {
        MPI_Send(d_distance2[1], G.numVertices, MPI_INT, 0, 0, MPI_COMM_WORLD);
        return;
    }
     //copy memory from device
    std::vector<int> v0(distance);
    std::vector<int> v1(distance);
    MPI_Recv(v1.data(), G.numVertices, MPI_INT, 1, 0, MPI_COMM_WORLD, NULL);
    cudaMemcpy(v0.data(), d_distance2[0], G.numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(v1.data(), d_distance2[1], G.numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(v1.data(), d_distance2[1], G.numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < distance.size(); ++i)
    {
        if(v0[i] < v1[i])
        {
            distance[i] = v0[i];
        }
        else
            distance[i] = v1[i];
    }
    // cudaMemcpy(parent.data(), d_parent2[0], G.numVertices * sizeof(int), cudaMemcpyDeviceToHost);

}

// void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
//                       std::vector<int> &parent) {
//     initializeCudaBfs(startVertex, distance, parent, G);

//     int *changed;
//     checkError(cuMemAllocHost((void **) &changed, sizeof(int)), "cannot allocate changed");

//     //launch kernel
//     printf("Starting simple parallel bfs.\n");
//     auto start = std::chrono::steady_clock::now();

//     *changed = 1;
//     int level = 0;
//     while (*changed) {
//         *changed = 0;
//         void *args[] = {&G.numVertices, &level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent,
//                         &changed};
//         checkError(cuLaunchKernel(cuSimpleBfs, G.numVertices / 1024 + 1, 1, 1,
//                                   1024, 1, 1, 0, 0, args, 0),
//                    "cannot run kernel simpleBfs");
//         cuCtxSynchronize();
//         level++;
//     }


//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);

//     finalizeCudaBfs(distance, parent, G);
// }

void runCudaSimpleBfsMulti(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent) {
    initializeCudaBfs2(startVertex, distance, parent, G);

    int *changed;
    // (cudaMallocHost((void **) &changed, sizeof(int)), "cannot allocate changed");
    cudaMallocManaged(&changed, 1 * sizeof(int));

    //launch kernel
    printf("Starting simple parallel bfs.\n");
    size_t temp = 1;
    cudaDeviceGetLimit(&temp,cudaLimitPrintfFifoSize);
    printf("limit=%d\n", temp);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 11474836);
    cudaDeviceGetLimit(&temp,cudaLimitPrintfFifoSize);
    printf("limit=%d\n", temp);

    auto start = std::chrono::steady_clock::now();

    *changed = 1;
    int level = 0;
    while (*changed) {
        *changed = 0;
        printf("level=%d\n", level);
        for(int i = 0; i < DeviceNum; ++i)
        {
            int part = G.numVertices / DeviceNum;
            printf("part: %d\n", part);
            int begin = i * part;
            int end = (i + 1) * part;
            if(i == DeviceNum - 1)
                end = G.numVertices;
            int deviceid = i;

            printf("device id :%d\n", i);

            cudaSetDevice(i);
            printf("block=%d thread=%d\n", G.numVertices / 1024 + 1, 1024 / DeviceNum);
            
	    dim3 grid(128);
	    dim3 threads(20);
            //multiBfs <<<grid, threads>>>(begin, end, G.numVertices, 
            multiBfs <<<G.numVertices / 1024 + 1, 1024 / DeviceNum, 0, 0>>>(begin, end, G.numVertices, 
                level, deviceid, d_adjacencyList2[i], 
                d_edgesOffset2[i], d_edgesSize2[i], d_distance2, d_parent2,
                        changed);
            // cudaDeviceSynchronize();
        }
        for(int i = 0; i < DeviceNum; ++i)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        level++;
    }


    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);



    finalizeCudaBfs(distance, parent, G);
}

// hkz
void runCudaQueueBfs(int startVertex, Graph &G, std::vector<int> &distance,
                     std::vector<int> &parent) {
    initializeCudaBfs2(startVertex, distance, parent, G);

    //int *nextQueueSize;
    //checkError(cuMemAllocHost((void **) &nextQueueSize, sizeof(int)), "cannot allocate nextQueueSize");

    //launch kernel
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Starting queue parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();
    //int queueSize = 1;
    //*nextQueueSize = 0;
    // for(int i = 0; i < DeviceNum; ++i)
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i = rank;
    {
        if(getDev(startVertex) == i)
        {
            queueSize[i] = 1;
        }
        else queueSize[i] = 0;
        for(int j = 0; j < DeviceNum; ++j)
        {
            nextQueueSize[i][j] = 0;
        }
        if(queueSize[i] == 0) queueSize[(i + 1) % 2] = 1;
    }

    int level = 0;
    int allnum = 1;
    while (allnum > 0) {
        //for(int i = 0; i < DeviceNum; ++i)
        {
            int i = rank;
            //cudaSetDevice(i);
            queueBfs <<<queueSize[i] / 1024 + 1, 1024 >>> (level, d_adjacencyList2[i], d_edgesOffset2[i], d_edgesSize2[i], d_distance2[i],
                d_parent2[i], queueSize[i], 
                nextQueueSize[i], 
                d_currentQueue2[i],
                d_nextQueue2[i],
                G.numVertices / DeviceNum, i, G.numVertices);
        }
        cudaDeviceSynchronize();

        level++;
        // cudaMemcpy(d_currentQueue2[0], )
        //for(int i = 0; i < DeviceNum; ++i)
        int offset = 0;
        {
            int i = rank;
            for(int j = 0; j < DeviceNum; ++j)
            {
                // cudaMemcpy(d_currentQueue2[i] + offset, &(d_nextQueue2[j][i * G.numVertices]), 
                //     nextQueueSize[i][j] * sizeof(int), 
                //     cudaMemcpyDefault);
                // printf("copying %d\n", nextQueueSize[i][j]);
                if(j == i) // local
                {
                    cudaMemcpy(d_currentQueue2[i] + offset, &(d_nextQueue2[j][i * G.numVertices]), nextQueueSize[j][i] * sizeof(int), cudaMemcpyDeviceToDevice);
                    offset += nextQueueSize[j][i];
                }
                else // remote
                {
                    MPI_Status s;
                    MPI_Sendrecv(&(d_nextQueue2[i][j * G.numVertices]), nextQueueSize[i][j], MPI_INT, (i + 1) % 2, 0, d_currentQueue2[i] + offset, G.numVertices, MPI_INT, (i+1) % 2, 0, MPI_COMM_WORLD, &s);
                    int count = -1;
                    MPI_Get_count(&s, MPI_INT, &count);
                    offset += count;
                }
            }
            MPI_Allreduce(&offset, &allnum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
        

        // multiBfs <<<G.numVertices / 1024 + 1, 1024 / DeviceNum, 0, 0>>>(begin, end, G.numVertices, 
        //         level, deviceid, d_adjacencyList2[i], 
        //         d_edgesOffset2[i], d_edgesSize2[i], d_distance2, d_parent2,
        //                 changed);

        queueSize[i] = offset;
        //printf("rank=%d, size=%d\n", rank, queueSize[i]);
        cudaMemset(nextQueueSize, 0, DeviceNum*DeviceNum*sizeof(int));
        // std::swap(d_currentQueue, d_nextQueue);
    }


    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if(rank == 0){
        printf("Elapsed time in milliseconds : %li ms.\n", duration);
    }
    finalizeCudaBfs(distance, parent, G, rank);
}


// void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
//                       std::vector<int> &parent) {
//     initializeCudaBfs(startVertex, distance, parent, G);

//     int *changed;
//     checkError(cuMemAllocHost((void **) &changed, sizeof(int)), "cannot allocate changed");

//     //launch kernel
//     printf("Starting simple parallel bfs.\n");
//     auto start = std::chrono::steady_clock::now();

//     *changed = 1;
//     int level = 0;
//     while (*changed) {
//         *changed = 0;
//         for(int i = 0; i < DeviceNum; ++i)
//         {
//             int part = G.numVertices / DeviceNum;
//             int begin = i * part;
//             int end = (i + 1) * part;
//             void *args[] = {&G.numVertices, &level, &d_adjacencyList[i], &d_edgesOffset[i], &d_edgesSize[i], &d_distance[i], &d_parent[i],
//                         &changed};
//             checkError(cuLaunchKernel(cuSimpleBfs, G.numVertices / 1024 + 1, 1, 1,
//                                   1024, 1, 1, 0, 0, args, 0),
//                    "cannot run kernel simpleBfs");
//         }
//         for(int i = 0; i < DeviceNum; ++i)
//             cuCtxSynchronize();
//         level++;
//     }


//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);

//     finalizeCudaBfs(distance, parent, G);
// }

// void runCudaQueueBfs(int startVertex, Graph &G, std::vector<int> &distance,
//                      std::vector<int> &parent) {
//     initializeCudaBfs(startVertex, distance, parent, G);

//     int *nextQueueSize;
//     checkError(cuMemAllocHost((void **) &nextQueueSize, sizeof(int)), "cannot allocate nextQueueSize");

//     //launch kernel
//     printf("Starting queue parallel bfs.\n");
//     auto start = std::chrono::steady_clock::now();

//     int queueSize = 1;
//     *nextQueueSize = 0;
//     int level = 0;
//     while (queueSize) {
//         void *args[] = {&level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent, &queueSize,
//                         &nextQueueSize, &d_currentQueue, &d_nextQueue};
//         checkError(cuLaunchKernel(cuQueueBfs, queueSize / 1024 + 1, 1, 1,
//                                   1024, 1, 1, 0, 0, args, 0),
//                    "cannot run kernel queueBfs");
//         cuCtxSynchronize();
//         level++;
//         queueSize = *nextQueueSize;
//         *nextQueueSize = 0;
//         std::swap(d_currentQueue, d_nextQueue);
//     }


//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);

//     finalizeCudaBfs(distance, parent, G);
// }

// void nextLayer(int level, int queueSize) {
//     void *args[] = {&level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent, &queueSize,
//                     &d_currentQueue};
//     checkError(cuLaunchKernel(cuNextLayer, queueSize / 1024 + 1, 1, 1,
//                               1024, 1, 1, 0, 0, args, 0),
//                "cannot run kernel cuNextLayer");
//     cuCtxSynchronize();
// }

// void countDegrees(int level, int queueSize) {
//     void *args[] = {&d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_parent, &queueSize,
//                     &d_currentQueue, &d_degrees};
//     checkError(cuLaunchKernel(cuCountDegrees, queueSize / 1024 + 1, 1, 1,
//                               1024, 1, 1, 0, 0, args, 0),
//                "cannot run kernel cuNextLayer");
//     cuCtxSynchronize();
// }

// void scanDegrees(int queueSize) {
//     //run kernel so every block in d_currentQueue has prefix sums calculated
//     void *args[] = {&queueSize, &d_degrees, &incrDegrees};
//     checkError(cuLaunchKernel(cuScanDegrees, queueSize / 1024 + 1, 1, 1,
//                               1024, 1, 1, 0, 0, args, 0), "cannot run kernel scanDegrees");
//     cuCtxSynchronize();

//     //count prefix sums on CPU for ends of blocks exclusive
//     //already written previous block sum
//     incrDegrees[0] = 0;
//     for (int i = 1024; i < queueSize + 1024; i += 1024) {
//         incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
//     }
// }

// void assignVerticesNextQueue(int queueSize, int nextQueueSize) {
//     void *args[] = {&d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_parent, &queueSize, &d_currentQueue,
//                     &d_nextQueue, &d_degrees, &incrDegrees, &nextQueueSize};
//     checkError(cuLaunchKernel(cuAssignVerticesNextQueue, queueSize / 1024 + 1, 1, 1,
//                               1024, 1, 1, 0, 0, args, 0),
//                "cannot run kernel assignVerticesNextQueue");
//     cuCtxSynchronize();
// }

// void runCudaScanBfs(int startVertex, Graph &G, std::vector<int> &distance,
//                     std::vector<int> &parent) {
//     initializeCudaBfs(startVertex, distance, parent, G);

//     //launch kernel
//     printf("Starting scan parallel bfs.\n");
//     auto start = std::chrono::steady_clock::now();

//     int queueSize = 1;
//     int nextQueueSize = 0;
//     int level = 0;
//     while (queueSize) {
//         // next layer phase
//         nextLayer(level, queueSize);
//         // counting degrees phase
//         countDegrees(level, queueSize);
//         // doing scan on degrees
//         scanDegrees(queueSize);
//         nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
//         // assigning vertices to nextQueue
//         assignVerticesNextQueue(queueSize, nextQueueSize);

//         level++;
//         queueSize = nextQueueSize;
//         std::swap(d_currentQueue, d_nextQueue);
//     }


//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);

//     finalizeCudaBfs(distance, parent, G);
// }

int main(int argc, char **argv) {

    // read graph from standard input
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("myrank is %d\n", rank);
    //return 0;



    int startVertex = atoi(argv[1]);
    readGraphFromFile(G, argc, argv);

    printf("Number of vertices %d\n", G.numVertices);
    printf("Number of edges %d\n\n", G.numEdges);

    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);

    //run CPU sequential bfs
    if(rank == 0){
        runCpu(startVertex, G, distance, parent, visited);
    }

    //save results from sequential bfs

    initCuda2(G);
    //run CUDA simple parallel bfs
    // runCudaSimpleBfs(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);

    // runCudaSimpleBfsMulti(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);


    // //run CUDA queue parallel bfs
    runCudaQueueBfs(startVertex, G, distance, parent);
    if(rank == 0){
        std::vector<int> expectedDistance(distance);
        std::vector<int> expectedParent(parent);
        checkOutput(distance, expectedDistance, G);
    }

    // //run CUDA scan parallel bfs
    // runCudaScanBfs(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);

    // finalizeCuda();
    MPI_Finalize();
    return 0;
}





void readGraphFromFile(Graph &G, int argc, char **argv) {
    printf("%s\n", argv[2]);
    std::ifstream f;
    // f.is_open();
    try {
      f.open(argv[2]);
    }
    catch (std::ios_base::failure& e) {
      std::cerr << e.what() <<" open file error \n";
    }
    //assert(fin.isopen());
    int n, m;

    // std::string line = "dsadsa";
    if(f.good())
    {
        f>>n>>m;
        printf("nodes num: %d\n", n);
        std::vector<std::vector<int> > adjecancyLists(n);
        printf("edge num: %d\n", m);
        int cnt = 0;
        int mmax = -1;
        for (int i = 0; i < m; i++) {
            int u, v;
            // printf("%d\n", cnt);

            f >> u >> v;
            // if(v == 319 || v == 320)
                // printf("%d %d\n", u, v);
            // if(u > mmax) mmax = u;
            // if(v > mmax) mmax = v;
            adjecancyLists[u].push_back(v);
            adjecancyLists[v].push_back(u);
            // ++cnt;
        }
        // printf("%d\n", mmax);
        // exit(0);
        for (int i = 0; i < n; i++) {
            G.edgesOffset.push_back(G.adjacencyList.size());
            G.edgesSize.push_back(adjecancyLists[i].size());
            for (auto &edge: adjecancyLists[i]) {
                G.adjacencyList.push_back(edge);
            }
        }

        G.numVertices = n;
        G.numEdges = G.adjacencyList.size();
        printf("finish load graph\n");
    }
    else printf("not open %s\n", argv[2]);
    
}

void readGraph(Graph &G, int argc, char **argv) {
    int n;
    int m;

    //If no arguments then read graph from stdin
    bool fromStdin = argc <= 2;
    if (fromStdin) {
        scanf("%d %d", &n, &m);
    } else {
        srand(12345);
        n = atoi(argv[2]);
        m = atoi(argv[3]);
    }

    std::vector<std::vector<int> > adjecancyLists(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        if (fromStdin) {
            scanf("%d %d", &u, &v);
            adjecancyLists[u].push_back(v);
        } else {
            u = rand() % n;
            v = rand() % n;
            adjecancyLists[u].push_back(v);
            adjecancyLists[v].push_back(u);
        }
    }

    for (int i = 0; i < n; i++) {
        G.edgesOffset.push_back(G.adjacencyList.size());
        G.edgesSize.push_back(adjecancyLists[i].size());
        for (auto &edge: adjecancyLists[i]) {
            G.adjacencyList.push_back(edge);
        }
    }

    G.numVertices = n;
    G.numEdges = G.adjacencyList.size();
}


void bfsCPU(int start, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
    distance[start] = 0;
    parent[start] = start;
    visited[start] = true;
    std::queue<int> Q;
    Q.push(start);

    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();

        for (int i = G.edgesOffset[u]; i < G.edgesOffset[u] + G.edgesSize[u]; i++) {
            int v = G.adjacencyList[i];
            if (!visited[v]) {
                visited[v] = true;
                distance[v] = distance[u] + 1;
                parent[v] = i;
                Q.push(v);
            }
        }
    }
}

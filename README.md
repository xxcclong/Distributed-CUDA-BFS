# Distributed-CUDA-BFS
cuda bfs that can scale to multi-GPU as well as multi-nodes

### Intra node 

```
nvcc -std=c++11 ./bfs.cu
# 2 is the src vertex, file format is
# vertex_num edge_num
# src dst
# src dst
# ...
./a.out 2 /path/to/file
```

### Inter node

```
# compile with osu
# tested with openmpi 1.10.0
# now only run with 2 processes
/home/huangkz/myompi/bin/mpirun -np 2 -host e5,e8 -x LD_LIBRARY_PATH ./osu_bw 0 ./soc-LiveJournal1.mtx
```


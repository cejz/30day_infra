# CUDA优化：最大化计算和指令的吞吐
link：https://register.nvidia.com/flow/nvidia/gtcs25/vap/page/vsessioncatalog/session/1727709629316001myds


# GPU 线程层级化，SIMT，warp 发散
下图为H100的整体架构：

![alt text](image-3.png)

流处理器（SM）：

SM有四个子区域 (下述数据针对的是整个SM)：

+ 128 fp32 units
+ 64 fp64 ... 
+ 64 int32 ...
+ 4 混合精度 tensor cores
+ 16 特殊函数 uints
+ 4 warp scheduler

+ 32 LD/ST units
+ 64K 32-bit registers
+ 256KiB 统一的 L1 data cache and shared memory
+ tensor memory accelerator(TMA)

![alt text](image-4.png)

线程层级： grid & blocks
![alt text](image-5.png)
一个cuda kernel 是一个被launch thread blocks的grid，blocks的工作可以完全独立地完成。

blocks是在每个SM上执行：

+ 一些blocks将会并发地位于一个SM上
+ blocks不能迁移到别的SM
+ 在SM上的每个block可以以任意顺序进行调度

Hopper引入了一个新的线程层级：clusters；一个cluster的blocks可以并发地进行调度，并且对于cluster内跨越多个SM的线程来说，能够高效的协作以及数据共享。

![alt text](image-6.png)

线程层级：warps

+ 1 warp = 32 threads；
+ 在运行时，一个block将被划分为warps获得SIMT的执行；
+ 一个warp中的线程有连续的线程id；一个block的warps数量可以通过 `ceil(threads per block / warp size)` 定义。

SIMT架构

+ 每个线程有自己的PC（程序计数器）
+ SIMT = SIMD + PC
+ 如果一个warp内的线程发散（例如，通过条件分支），这个warp将会分离地执行每个分支路径
+ 当warp中的32 个线程都执行同一个路径，就会获得完全的效率

![alt text](image-7.png)

ncu中的提示信息如下：
![alt text](image-8.png)

可以在source page 查看每条指令warp有多么converge。
![alt text](image-9.png)

减少warp 发散的tips：

一些共通的原因：

+ 每个线程的工作不同
    + sol：queue and bin/sort the work
+ Per thread work discovered at different times.
    + sol：queue the work
+ 每个线程在不同时间点结束
    + sol：划分为多个kernel

概念上的发散最好通过变化数据，而不是变化控制流


在shared memory中的work queueing

想象你在沙滩上寻宝，手拿金属探测仪和铁锹，切换这两个工具是十分耗时的；一种策略是只要探测仪有信号，那就切换到铁锹；另一种策略是先标记好每个有信号的地方，等有足够多的标记时切换到铁锹，这可以显著减少开销。

gpu上也有类似的workload，一个expensive的运算通过一个轻量化的if来guard；naive的实现可能导致并不是所有的线程都会通过这个if来进入运算。

解法：当有一个线程发现了需要deep dive的时候，先将其加入到队列中，之后继续工作；定期队列中的所有线程进行工作

```cpp
constexpr int k_block_size = 64; 

constexpr int k_capacity_factor = 3;
constexpr int k_queue_capacity = k_block_size * k_capacity_factor;

constexpr int k_queue_process_size = k_block_size * (k_capacity_factor - 1);

using Work_item_t = int;

__device__
void process_queue(Work_item_t (&block_queue) [k_queue_capacity], int &block_queue_size) {

    for (int queue_idx = threadIdx.x; queue_idx < block_queue_size; queue_idx += blockDim.x) {
        const Work_item_t work_item = block_queue[queue_idx];
        // 
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        block_queue_idx = 0;
    }
    __syncthreads();
}
```

```cpp
__global__ void scout_and_dive_kernel() {
    __shared__ int block_num_threads_finished_scouting;
    __shared__ Work_item_t block_queue[k_queue_capacity];
    __shared__ int block_queue_size;

    const auto &is_thread_finished_scouting = [] () {return false;  /*your condition*/}

    while (block_num_threads_finished_scouting < k_block_size) {
        if (block_queue_size >= k_queue_process_size) {
            process_queue(block_queue, block_queue_size);
        }

        if (!is_thread_finished_scouting()) {
            // performing scouting work

            const bool found_dive = false; // your condition
            if (found_dive) {
                const Work_item_t work_idx = 0;
                const auto queue_write_dst = atomicAdd(&block_queue_size, 1);
                block_queue[queue_write_dst] = work_idx;
            }

            // advance the next scouting work

            if (!is_thread_finished_scouting()) {
                //
            } else {
                atomicAdd(&block_num_threads_finished_scouting,1);
            }
        }
        __syncthreads();
    }
    // flush queue at end
}

```


注意有几个函数需要解释：

+ `AtomicAdd`：从指定位置（shared memory / global memory）读取old值，将这个old值加上给定的value，最后存入这个位置中；函数返回的是**old**值

减少warp发散：使用数据流实现概念上的发散

```cpp
float x = 0.0f;
if (isA) {
    x = valA;
} else if (isB) {
    x = valB;
}

// prefer
float x = (isA) * valA + (isB) * valB;
```

一个更复杂的例子如下：

![alt text](image-10.png)

bonus: 动态block维度。（不是线程发散，更像是 block发散）

存在一类图像处理的workload，其中每个block将处理一个polygon；每个线程处理对应的 （pixel_x, pixel_y, poly_edge）一个伪代码如下：

```cpp
for (int x_idx = threadIdx.x; x_idx < x_size; x_dix += blockDim.x) {
    ... 
}
```
可能存在的问题：ploygen的特征，如x_size等如果变化很大；导致无法充分利用线程块。

![alt text](image-11.png)

一个动态的解法：(变换block的形状)
```cpp
const auto polygon_idx = blockIdx.x;

const auto x_size = ...;

dim3 vBlockDim(1,1,1);
const auto threads_allocated = 1;
// fill x
while(vBlockDim.x < x_size && threads_allocated < blockDim.x) {
    vBlockDim.x *= 2;
    threads_allocated *= 2;
}
// fill y and e
...

dim3 vThreadIdx;
vThreadIdx.x = threadIdx.x % vBlockDim.x;
vThreadIDx.y = (threadIdx.x / vBlockDim.x) % vBlockDim.y;
//
for (int x = vThreadIdx.x; x < x_size; x += vBlockDim.x) {
    //
}
```

# warp 调度和kernel profile

每个SM中有四个warp scheduler；每个scheduler管理一个warps pool（Hopper：每个scheduler有16个warps 插槽）；每个时钟周期，每个scheduler都可以对一个warp发射出一条指令。

## warp scheduler的统计数据

首先建立一个profiling的心智模型：

warp有五个状态：
+ unused
+ active: warp在处理器上
+ stalled: warp正在等待先前指令完成；或者等待其下一条指令所需要的输入数据
+ eligible：warp执行下条指令所需要的数据都已经readyle
+ selected： eligible & 这一周期被scheduler选择发出指令

![alt text](image-12.png)

![alt text](image-13.png)

latency bound kernel: 每个scheduler只有一个活跃warp

例子：每个线程累加 pi 10000 次 使用双精度

```cpp
__global__ void kernel_A(double* A, int K) {
    double result = 0.0;
    for(int j=0; j<K; ++j) {
        result += 3.14;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = result;
}

kernel_A<<<1, 32>>>(d_f64, 1000);
cudaDeviceSynchronize();
```

![alt text](image-14.png)
上图可以看到一个double-point add需要花费8个时钟周期；注意此时还应该考虑SM中一个分区的f64单元的数量。这些数据在ncu的表现如下：

![alt text](image-15.png)

注意有两个关于GPU最好的性质：

1. 首先是SIMD
2. 但是更重要的是超线程(hyperthreading)：

    + warps 可以分时使用处理器的资源，掩盖相互之间的延迟；使得硬件资源一直繁忙
    + warps之间的切换从软件视角看是免费的

## kernel profiling
何时优化会做完：
+ 实践上使用Amdahl‘s law。 当代码相比于系统的其他部分不是瓶颈
+ 理论上：当所有被执行的work都是绝对必要的，并且负责这个work的硬件单元都充分利用或者占用。

![alt text](image-16.png)

四类性能限制：
![alt text](image-17.png)

roofline model
![alt text](image-18.png)

ncu source page：给定你的bounding 类型，应该关注那些列？
![alt text](image-19.png)

# 延迟隐藏 / 增大指令吞吐

latency bound的程序，大多数时间花费在等待指令完成上，硬件资源并没有充分利用；需要更多in flight的指令来隐藏指令延迟以及增大硬件的使用率；发出更多的指令；但是busy也不一定总是有用

stalling的原因：

+ wait：等待一个编译期已知延迟的指令
+ scoreboard: 等待一个运行期确定延迟的指令
    + long SB -- 通常是global memory
    + short SB -- 通常是shared memory
+ throttle： 等待硬件资源有空闲的时候
+ branch resolving：等待branch / PC bookkeeping
+ barrier: 等待其他线程sync

![alt text](image-21.png)

stall 避免策略：

register/ software pipeline：必须破坏register 的依赖关系

```cpp
// without pipeline
auto result = 0;
for (...) 
    auto working_set = load(...)
    // stalling to load
    work_set_compoute()
    // compute lots with working_set
    result += working_set

// w pipeline
auto result = 0;
auto prefetched = load(...)

for (...)
    auto working_set = prefetched
    prefetched = load(...)

    working_set.compute()

    result += working_set
```
可以使用一些高级的api：`__pipeline_memcpy_async` `__pipeline_commit()` etc.

![alt text](image-22.png)

barriers: threads必须停下来等待其他线程move的位置；可以通过内存进行同步操作

__syncthreads(): 同步整个线程块，要求被块中所有线程调用；不能将其放入conditional scope中；否则将有UB

协作组同步：同步由user定义的协作组
![alt text](image-23.png)

延迟隐藏：增加in-flight的指令数

两种方式：

+ 改善指令级并行（ILP）：每个线程有更多独立的计算
+ 改善occupacy，即线程级并行（TLP）：给定HW资源的限制下，描述有多少warps可以并发地执行；更多的活跃warps -> 更多的in-flight的指令

![alt text](image-25.png)

![alt text](image-24.png)

stalling的底线：

1. 如果SM或者内存资源是busy的，不需要担心stalls或是未使用的发射slots
2. 否则，你是layency bound的，给硬件提供更多的并发工作

    + 更频繁地发射
    + 更不频繁的stall
    + 在stall时让自己更忙
    + 减少stall的持续时间，例如使用低延迟的指令

occupacy

存在一个可以在SM上的最大warps 数量：

+ device：依赖GPU的计算能力
+ achievable：依赖kernel实现和编译器
+ achieved：很大程度上依赖grid size或者workload的行为，

$$occupacy = \frac {achievable}{device}$$

achivable的限制因素：

+ SM资源的分配：
    + 例如 shared memory和registers必须在线程级别上划分
    + block size
+ 其他的硬件因素：
    + 比如 每个SM的最大blocks数，每个SM的最大warps数

![alt text](image-26.png)

看这一个例子，我们首先可以从右图看出每个block的资源需求，share memory需要70KB，regs需要30KB；而活跃的warps数量受到硬件资源的限制，这里一个SM只有164KB的share memory，即最多只能在SM上分配两个线程块

我们可以通过两种方式增加occupacy：

+ 每个block使用更少的share memory
+ 将block size减为3个warps；这时需要的share memory资源为52.5KB；那么可以分配3个线程块；此时有9个warps活跃

occupacy的限制因素：

+ register的使用：使用 `--ptxas-options=-v` 来编译，报告每个线程的register
+ 每个线程的最大数量可以手动设置：在编译期，可以以每个文件为基础，使用nvcc的 `--maxrregcount` flag；或者在每个kernel中使用 `__launch_bounds__` 或者 `__maxnreg__`
+ hopper每个SM有65536个register：以256 registers的固定倍数进行分配; 例如 kernel每个线程使用63个registers，那么每个warps使用32*63 = 2016 -> 必须是256的倍数 -> 2048，那么每个SM的活跃warps数量为 65536 / 2048 = 32个，occupacy = 32 / 64 = 50 %

如果超出了内存限制，那么会spill到local memory中，local memory是线程私有的存储空间，位于设备内存上，可以被L1和L2 cache；注意受限的local memory使用对性能是有利的

![alt text](image-27.png)

减少register pressure的tips：

+ `__forceinline__` 可以避免函数调用以及ABI的开销；但需注意，内联在很多地方调用的长functions会引起指令cache的抖动
+ `#pragama unroll`：loop unrolling可以减少flow control的开销但可能增加register pressure并引起指令cache的抖动; factors 需要tune
+ 尽可能避免64-bit的类型，因为它会消耗两个寄存器
+ kernel分裂可以减少register 的使用并改善occupacy

occupacy的限制因素： thread block size

+ block size是warp size的倍数；即使不是，硬件也会round up
+ 每个block的最大size是1024
+ 每个SM可以有最多64个warps，32个blocks以及2048的线程（Hopper）

![alt text](image-28.png)

我需要多大的occupacy？

一个原则是尝试最大化occupacy；但是一些算法可能在low occupacy的时候运行地更好；更多的寄存器和share memory可以允许更高的数据重用。。。

![alt text](image-29.png)

# 减少指令数量 和 让吞吐有用



# Tensor core summary
![alt text](image-20.png)

推荐使用cutlass 来获得更好的性能
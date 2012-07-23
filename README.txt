USAGE:

GPU work best with many threads. In this implementation, each demon is a GPU thread. For best performance many demons should exist. This number depends a lot on the GPU but it is safe to say that you should have at least 1000 demons.

GPU work by executing many threads at a time. The number of threads that are executed in parallel is fixed which means the number of demons should be a multiple of the workgroup size which is currently set at 128 or (128*4 for the vectorized version). Adding more demons to fit will have a very negligible impact on performance. Simply put, you are charged by blocks of 128 (or 128*4) demons.


INSTALLATION:

To use you only need to set up the classpath correctly and make sure some OpenCL drivers are installed. In many case, these drivers will packaged with your most recent graphic card drivers.

If not, you can get them here:

AMD:
http://developer.amd.com/sdks/AMDAPPSDK/downloads/Pages/default.aspx

NVIDIA:
http://www.nvidia.com/content/devzone/index.html

import pyopencl as cl
import numpy as np
import time

def create_context_queue():
    """Create OpenCL context and command queue with error handling."""
    try:
        # Get first available platform and device
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        platform = platforms[0]
        devices = platform.get_devices()
        if not devices:
            raise RuntimeError("No OpenCL devices found")
        
        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context, 
                              properties=cl.command_queue_properties.PROFILING_ENABLE)
        return context, queue, device
    except cl.Error as e:
        raise RuntimeError(f"OpenCL initialization failed: {str(e)}")

def compile_kernel(context, device):
    """Compile OpenCL kernel with error handling."""
    kernel_code = """
    __kernel void vector_addition(__global const float *a,
                                __global const float *b,
                                __global float *result,
                                const unsigned int n) {
        // Get work-item ID
        size_t gid = get_global_id(0);
        
        // Boundary check to prevent buffer overflow
        if (gid < n) {
            // Coalesced memory access for better performance
            result[gid] = a[gid] + b[gid];
        }
    }
    """
    
    try:
        program = cl.Program(context, kernel_code).build()
        return program
    except cl.BuildError as e:
        print("OpenCL Build Error:")
        print(e.build_log.decode())
        raise

def vector_add_gpu(a, b, block_size=256):
    """Perform vector addition on GPU with performance optimizations."""
    if len(a) != len(b):
        raise ValueError("Input vectors must have the same length")
    
    # Initialize OpenCL
    context, queue, device = create_context_queue()
    program = compile_kernel(context, device)
    
    # Convert inputs to float32 for better GPU performance
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = len(a)
    
    # Create output array
    result = np.empty_like(a)
    
    # Get device limits
    max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    block_size = min(block_size, max_work_group_size)
    
    # Calculate optimal global size
    global_size = ((n + block_size - 1) // block_size) * block_size
    
    try:
        # Create buffers with optimal flags
        mf = cl.mem_flags
        a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        result_buf = cl.Buffer(context, mf.WRITE_ONLY, size=result.nbytes)
        
        # Execute kernel with profiling
        kernel_event = program.vector_addition(queue, 
                                            (global_size,),
                                            (block_size,),
                                            a_buf, 
                                            b_buf, 
                                            result_buf,
                                            np.uint32(n))
        
        kernel_event.wait()
        
        # Get execution time
        gpu_time = (kernel_event.profile.end - kernel_event.profile.start) * 1e-9
        
        # Copy result back to host
        cl.enqueue_copy(queue, result, result_buf).wait()
        
        # Verify results
        is_correct = np.allclose(result, a + b)
        
        return result, gpu_time, is_correct
        
    finally:
        # Clean up
        queue.finish()
    
def main():
    # Test parameters
    n = 25_000_000
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    
    # CPU baseline for comparison
    cpu_start = time.perf_counter()
    cpu_result = a + b
    cpu_time = time.perf_counter() - cpu_start
    
    # GPU computation
    try:
        result, gpu_time, is_correct = vector_add_gpu(a, b)
        
        # Print results
        print(f"Array size: {n:,} elements")
        print(f"Results match: {is_correct}")
        print(f"GPU time: {gpu_time:.6f} seconds")
        print(f"CPU time: {cpu_time:.6f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"Error during GPU computation: {str(e)}")

if __name__ == "__main__":
    main()

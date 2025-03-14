import pyopencl as cl
import numpy as np
import time

def main():
    print(f"PyOpenCL version: {cl.VERSION}")
    
    # Choose platform - rusticl worked for basic operations
    platforms = cl.get_platforms()
    platform = platforms[0]  # rusticl
    print(f"Using platform: {platform.name}")
    
    # Get device
    devices = platform.get_devices()
    device = devices[0]
    print(f"Using device: {device.name}")
    
    # Create context and queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    # Create test data
    n = 100_000_000  # 100 million elements
    a_np = np.random.rand(n).astype(np.float32)
    b_np = np.random.rand(n).astype(np.float32)
    result_np = np.empty_like(a_np)
    
    # CPU baseline
    cpu_start = time.perf_counter()
    cpu_result = a_np + b_np
    cpu_time = time.perf_counter() - cpu_start
    print(f"CPU time: {cpu_time:.6f} seconds")
    
    try:
        # Create buffers
        mf = cl.mem_flags
        a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
        b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
        result_buf = cl.Buffer(context, mf.WRITE_ONLY, size=result_np.nbytes)
        
        # Create kernel source
        kernel_src = """
        __kernel void vector_add(__global const float *a, 
                                __global const float *b,
                                __global float *c,
                                const unsigned int n)
        {
            int gid = get_global_id(0);
            if (gid < n) {
                c[gid] = a[gid] + b[gid];
            }
        }
        """
        
        # Build program
        program = cl.Program(context, kernel_src)
        try:
            program.build()
            print("Program built successfully")
        except cl.BuildError as e:
            print(f"Build error: {e}")
            print(e.build_log.decode())
            raise
        
        # Get kernel function - using low-level API to avoid the attribute access issue
        try:
            # Explicitly get the kernel by name using the low-level API
            kernel_name = "vector_add"
            
            # Method 1: Using _cl module
            kernel = cl._cl.Kernel(program._get_prg(), kernel_name)
            print("Kernel created using _cl module")
        except Exception as e:
            print(f"Error creating kernel with _cl module: {e}")
            
            try:
                # Method 2: Alternative approach using Program.all_kernels()[0]
                kernels = program.all_kernels()
                if kernels:
                    kernel = kernels[0]
                    print(f"Kernel created using all_kernels(), name: {kernel.function_name}")
                else:
                    raise RuntimeError("No kernels found in program")
            except Exception as e2:
                print(f"Error with alternative approach: {e2}")
                raise
        
        # Set kernel arguments directly
        kernel.set_arg(0, a_buf)
        kernel.set_arg(1, b_buf)
        kernel.set_arg(2, result_buf)
        kernel.set_arg(3, np.uint32(n))
        
        # Execute kernel
        gpu_start = time.perf_counter()
        global_size = ((n + 255) // 256) * 256
        local_size = 256
        
        event = cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))
        event.wait()
        cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))
        queue.finish()

        # Now time multiple runs
        num_runs = 5
        gpu_start = time.perf_counter()
        for _ in range(num_runs):
            cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))
        queue.finish()
        gpu_time = (time.perf_counter() - gpu_start) / num_runs
        # Copy result back
        cl.enqueue_copy(queue, result_np, result_buf)
        #gpu_time = time.perf_counter() - gpu_start
        
        # Verify result
        is_correct = np.allclose(result_np, cpu_result)
        
        # Print results
        print(f"Array size: {n:,} elements")
        print(f"Results match: {is_correct}")
        print(f"GPU time: {gpu_time:.6f} seconds")
        print(f"CPU time: {cpu_time:.6f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()

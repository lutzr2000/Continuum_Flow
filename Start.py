import Kernel_GPU
import cProfile
import Input_Rising_Smoke
import Input_Wind_Tunnel

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    Kernel_GPU.main(Input_Rising_Smoke.CONFIG)
    profiler.disable()
    
    profile_file = 'Performance_Rising_Smoke_GPU.prof'
    profiler.dump_stats(profile_file)

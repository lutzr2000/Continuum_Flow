import Kernel_CPU
import cProfile
import Input_Rising_Smoke
import Input_Wind_Tunnel

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    Kernel_CPU.main(Input_Wind_Tunnel.CONFIG)
    profiler.disable()
    
    profile_file = 'Performance_Wind_Tunnel_CPU.prof'
    profiler.dump_stats(profile_file)

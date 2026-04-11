import Kernel_GPU as Kernel_GPU
import cProfile
import Input_Rising_Smoke as Input_Rising_Smoke
import Input_Wind_Tunnel as Input_Wind_Tunnel
from pathlib import Path

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    Kernel_GPU.main(Input_Rising_Smoke.CONFIG)
    profiler.disable()
    
    kernel_dir = Path(__file__).resolve().parent
    profile_file = kernel_dir / 'Performance_Rising_Smoke_GPU.prof'
    profiler.dump_stats(profile_file)
 
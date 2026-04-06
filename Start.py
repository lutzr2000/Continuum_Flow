import Main
import cProfile

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    Main.main()
    profiler.disable()
    
    profile_file = 'Performance.prof'
    profiler.dump_stats(profile_file)

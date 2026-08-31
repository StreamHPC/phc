# Parallel HIP Compiler

This repo contains some utilities for improving HIP compilation speed, aimed at CI pipelines.

There are two tools available to help with this:

* Parallel HIP Compiler (PHC).
* [Sccache](https://github.com/mozilla/sccache) integration for PHC.

The main idea is to parallelize HIP compilations by splitting device and host compilations out into individual processes, and redistributing that work over the computer. As each compilation task is smaller, the work can in theory be more evenly divided across the cores, and so increase the total throughput. It is mainly effective when the object files of a project have very different compile times, and compile for many different HIP architectures at the same time.

### Usage: Parallel HIP Compiler

The `phc.py` script integrates with a GNU Make Jobserver in order to redistribute the work properly among the available cores. This repository does not contain an implementation for such a jobserver, the easiest way to get one is to use the one available in the Ninja[^1] source. The `phc.py` script itself is designed to be used as a "compiler launcher", meaning that it just needs to be placed in front of the compilation command:

```
$ /path/to/ninja/misc/jobserver_pool.py /path/to/phc.py amdclang++ \
  -c -o test.o test.hip --offload-arch=gfx1201 --offload-arch=gfx942
```

Because Ninja itself also integrates with the GNU Make Jobserver, the best way to use the `phc.py` script is to instruct CMake to use it as a [compiler launcher](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_LAUNCHER.html), and to use the Ninja generator. This way, extra compilation tasks spawned by any instance of `phc.py` are coordinated with those created by other instances and with those created by Ninja, and the computer is not overloaded:

```
$ cmake [...] \
  -GNinja \
  -DCMAKE_CXX_COMPILER_LAUNCHER=/path/to/phc.py \
  -DCMAKE_HIP_COMPILER_LAUNCHER=/path/to/phc.py
$ /path/to/ninja/misc/jobserver_pool.py ninja
```

[^1]: https://github.com/ninja-build/ninja/blob/656412538b6fc102b809a61e0efce422e5a20534/misc/jobserver_pool.py

### Usage: PHC Sccache Integration

Sccache coordinates between separate instances using a background task, which it spawns automatically. Fortunately, sccache also implements the GNU Make Jobserver protocol to allow sub-processes created by it to coordinate work, and `phc.py` implements support for this as well. That means that in order to use PHC together with sccache, `phc_sccache.sh` just needs to be prepended to the compilation command:

```
$ /path/to/phc_sccache.py amdclang++ \
  -c -o test.o test.hip --offload-arch=gfx1201 --offload-arch=gfx942
```

Or, in order to use it with CMake:

```
$ cmake [...] \
  -GNinja \
  -DCMAKE_CXX_COMPILER_LAUNCHER=/path/to/phc_sccache.sh \
  -DCMAKE_HIP_COMPILER_LAUNCHER=/path/to/phc_sccache.sh
$ ninja
```

Sccache itself can be configured as usual. See the [sccache repo](https://github.com/mozilla/sccache) for further information.

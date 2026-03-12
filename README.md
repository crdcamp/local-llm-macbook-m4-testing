# Why llama.cpp is a pain in the *** to run on Apple Silicone

[Medium Article](https://medium.com/@akdemir_bahadir/how-to-build-and-install-llama-cpp-python-on-apple-silicon-without-losing-your-mind-96d186f86d73)

"The main culprit behind these build failures is architecture confusion. Apple Silicon Macs can run both ARM64 (native) and x86_64 (through Rosetta) code, but your build tools need to speak the same language. When they don’t, chaos ensues."

Here's how you confirm you are running native ARM64 architecture and not in [Rosetta](https://en.wikipedia.org/wiki/Rosetta_(software)) mode.

```terminal
uname -m                          # Should output: arm64
sysctl -n sysctl.proc_translated  # Should output: 0 (not translated)
```

# Clean Your Environment

After creating an env, clean it:

```
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH SDKROOT CC CXX CFLAGS CXXFLAGS MACOSX_DEPLOYMENT_TARGET
```

# Verify C++ Headers
```terminal
ls "$(xcrun --show-sdk-path)"/usr/include/c++/v1/{mutex,initializer_list,cstdint}
```

If these files exist, you're all good.

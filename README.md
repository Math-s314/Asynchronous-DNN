# Asynchronous Linear Algebra Library

A cross-platform **C++ library** for running heavy **linear algebra computations asynchronously on your GPU**.  
It provides a clean and intuitive interface using **C++ operator overloading**, while handling all **OpenCL** operations under the hood.


```cpp
DNN::Matrix C(A + B);
C.waitForResults();
std::cout << C;
```

That’s it : asynchronous GPU matrix addition in just a few lines!

---

## 🚧 Project Status
The core async and OpenCL backend already works, but the library isn’t ready for real use yet, it currently just builds into a test executable. Here’s what’s coming next:

### 🧩 Short-Term Goals
- Refactor the code for easier extension and maintenance
- Add basic linear algebra operations (addition, multiplication, etc.)

### 🚀 Long-Term Goals
- Support more advanced operations (like eigenvalue computation)
- Optimize asynchronous performance
- Add support for other GPU APIs (e.g. CUDA)

---

## ⚙️ Build Instructions

This project depends on **OpenCL**, so make sure it’s installed first. A quick way to do this is via the [Khronos OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK).

The project builds with **CMake** and **MSVC**. In your `CMakeLists.txt`, point CMake to your OpenCL SDK with the `OpenCLDir` environment variable :

```cmake
unset(OpenCL_DIR CACHE)
find_package(OpenCL PATHS "$ENV{OpenCLDir}" NO_DEFAULT_PATH CONFIG REQUIRED)
```

💡 **Tip:** For now, the easiest way to build and test the project is with **CLion**.


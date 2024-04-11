Most of these will be listed in the requirements.txt file, but llama-cpp require additional arguments and hence we're writing this too. We'll also add links so this serves as a repository of useful information.

1. Llama_cpp - python bindings for Llama, and also has M1 compatible models. (and hence the different installation method.)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

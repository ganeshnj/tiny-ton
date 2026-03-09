FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build LLVM/MLIR 18
WORKDIR /opt
RUN git clone --depth 1 --branch llvmorg-18.1.8 \
    https://github.com/llvm/llvm-project.git

RUN cmake -G Ninja -S llvm-project/llvm -B llvm-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm-install \
    && cmake --build llvm-build --target install \
    && rm -rf llvm-project llvm-build

# Build tiny-ton
WORKDIR /workspace
COPY . .

RUN cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR=/opt/llvm-install/lib/cmake/mlir \
    -DLLVM_DIR=/opt/llvm-install/lib/cmake/llvm \
    -DTTN_ENABLE_PYTHON=OFF \
    && cmake --build build

# Runtime image
FROM ubuntu:22.04

COPY --from=builder /workspace/build/bin/ttc /usr/local/bin/ttc
COPY examples/ /workspace/

ENTRYPOINT ["ttc"]

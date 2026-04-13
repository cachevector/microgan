import ctypes
import numpy as np
import os

class PCReference:
    def __init__(self, lib_path="runtime/libmicrogan.so"):
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        
        # Setup argument and return types
        self.lib.MicroGAN_init.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        self.lib.MicroGAN_init.restype = ctypes.c_int
        
        self.lib.MicroGAN_generate.argtypes = [ctypes.c_uint8, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.MicroGAN_generate.restype = ctypes.c_int
        
        # Initialize with a dummy arena
        self.arena_size = 1024 * 64
        self.arena = (ctypes.c_uint8 * self.arena_size)()
        status = self.lib.MicroGAN_init(self.arena, self.arena_size)
        if status != 0:
            raise RuntimeError(f"Failed to initialize MicroGAN runtime: {status}")

    def generate(self, seed=42, class_id=0):
        output = (ctypes.c_uint8 * 1024)() # 32x32 grayscale
        status = self.lib.MicroGAN_generate(seed, class_id, output)
        if status != 0:
            raise RuntimeError(f"Inference failed with status: {status}")
        
        return np.frombuffer(output, dtype=np.uint8).reshape(32, 32)

if __name__ == "__main__":
    ref = PCReference()
    img = ref.generate(seed=42)
    print(f"Generated image sample (0,0): {img[0,0]}")

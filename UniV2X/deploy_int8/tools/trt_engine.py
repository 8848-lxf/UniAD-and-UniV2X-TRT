import ctypes
from collections import OrderedDict

import numpy as np
import tensorrt as trt
import torch


TORCH_DTYPES = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.bool_): torch.bool,
}


class TorchOutputAllocator(trt.IOutputAllocator):
    def __init__(self, output_dtypes, slot_count=2):
        trt.IOutputAllocator.__init__(self)
        self.output_dtypes = output_dtypes
        self.slot_count = slot_count
        self.buffer_slots = [dict() for _ in range(slot_count)]
        self.active_slot = 0
        self.buffers = {}
        self.shapes = {}
        self.allocations = 0
        self.reuses = 0

    def begin_inference(self, slot):
        self.active_slot = slot
        self.buffers = {}
        self.shapes = {}

    def _allocate(self, tensor_name, size):
        dtype = self.output_dtypes[tensor_name]
        itemsize = torch.empty((), dtype=dtype).element_size()
        required = max(1, (int(size) + itemsize - 1) // itemsize)
        slot_buffers = self.buffer_slots[self.active_slot]
        buffer = slot_buffers.get(tensor_name)
        if buffer is None or buffer.numel() < required:
            capacity = required if buffer is None else max(required, 2 * buffer.numel())
            buffer = torch.empty(capacity, dtype=dtype, device="cuda")
            slot_buffers[tensor_name] = buffer
            self.allocations += 1
        else:
            self.reuses += 1
        self.buffers[tensor_name] = buffer
        return buffer.data_ptr()

    def reallocate_output(self, tensor_name, current_memory, size, alignment):
        return self._allocate(tensor_name, size)

    def reallocate_output_async(
        self, tensor_name, current_memory, size, alignment, stream
    ):
        return self._allocate(tensor_name, size)

    def notify_shape(self, tensor_name, dims):
        self.shapes[tensor_name] = tuple(int(value) for value in dims)


class TensorRTEngine:
    def __init__(
        self,
        engine_path,
        plugin_path,
        logger_level=trt.Logger.WARNING,
        context_cache_size=1,
    ):
        self.plugin = ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
        self.logger = trt.Logger(logger_level)
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(engine_path, "rb") as handle:
            serialized = handle.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context_cache_size = max(1, int(context_cache_size))
        self.context_cache = OrderedDict()
        self.context_cache_hits = 0
        self.context_cache_misses = 0
        self.context_cache_evictions = 0
        self.shared_device_memory = None
        self.shared_device_memory_ptr = 0
        self.shared_device_memory_bytes = 0
        self.context = None
        if self.context_cache_size == 1:
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

        self.input_names = []
        self.output_names = []
        self.dtypes = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            numpy_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            if numpy_dtype not in TORCH_DTYPES:
                raise TypeError(f"Unsupported TensorRT dtype for {name}: {numpy_dtype}")
            self.dtypes[name] = TORCH_DTYPES[numpy_dtype]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        self.output_allocator = TorchOutputAllocator({
            name: self.dtypes[name] for name in self.output_names
        })
        self.output_slot = -1
        self.static_output_slots = [dict(), dict()]
        self.static_allocations = 0
        self.static_reuses = 0

    def _allocate_shared_device_memory(self):
        if self.shared_device_memory is not None:
            return
        required = int(self.engine.device_memory_size_v2)
        storage = torch.empty(required + 255, dtype=torch.uint8, device="cuda")
        aligned = (storage.data_ptr() + 255) & ~255
        if aligned + required > storage.data_ptr() + storage.numel():
            raise RuntimeError("Failed to align shared TensorRT device memory")
        self.shared_device_memory = storage
        self.shared_device_memory_ptr = aligned
        self.shared_device_memory_bytes = required

    def _select_context(self, shape_key):
        if self.context_cache_size == 1:
            return self.context
        context = self.context_cache.get(shape_key)
        if context is not None:
            self.context_cache.move_to_end(shape_key)
            self.context_cache_hits += 1
            return context

        self.context_cache_misses += 1
        if len(self.context_cache) >= self.context_cache_size:
            _, evicted_context = self.context_cache.popitem(last=False)
            del evicted_context
            self.context_cache_evictions += 1
        context = self.engine.create_execution_context_without_device_memory()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self._allocate_shared_device_memory()
        context.set_device_memory(
            self.shared_device_memory_ptr, self.shared_device_memory_bytes
        )
        self.context_cache[shape_key] = context
        return context

    def describe(self):
        return {
            "inputs": {
                name: {
                    "shape": list(self.engine.get_tensor_shape(name)),
                    "dtype": str(self.dtypes[name]),
                }
                for name in self.input_names
            },
            "outputs": {
                name: {
                    "shape": list(self.engine.get_tensor_shape(name)),
                    "dtype": str(self.dtypes[name]),
                }
                for name in self.output_names
            },
        }

    def infer(self, inputs, synchronize=True):
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise KeyError(f"Missing TensorRT inputs: {missing}")

        input_buffers = {}
        for name in self.input_names:
            value = inputs[name]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            value = value.to(device="cuda", dtype=self.dtypes[name]).contiguous()
            input_buffers[name] = value

        shape_key = tuple(
            tuple(int(dim) for dim in input_buffers[name].shape)
            for name in self.input_names
        )
        context = self._select_context(shape_key)
        self.context = context
        for name, value in input_buffers.items():
            if not context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"Rejected input shape for {name}: {tuple(value.shape)}")
            if not context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"Failed to bind input {name}")

        self.output_slot = (self.output_slot + 1) % len(self.static_output_slots)
        self.output_allocator.begin_inference(self.output_slot)
        static_buffers = self.static_output_slots[self.output_slot]
        output_buffers = {}
        dynamic_dtypes = {}
        for name in self.output_names:
            shape = tuple(int(value) for value in context.get_tensor_shape(name))
            if all(value >= 0 for value in shape):
                count = max(1, int(np.prod(shape, dtype=np.int64)))
                buffer = static_buffers.get(name)
                if buffer is None or buffer.numel() < count:
                    capacity = count if buffer is None else max(count, 2 * buffer.numel())
                    buffer = torch.empty(
                        capacity, dtype=self.dtypes[name], device="cuda"
                    )
                    static_buffers[name] = buffer
                    self.static_allocations += 1
                else:
                    self.static_reuses += 1
                output = buffer[:count].reshape(shape)
                output_buffers[name] = output
                if not context.set_tensor_address(name, output.data_ptr()):
                    raise RuntimeError(f"Failed to bind output {name}")
            else:
                dynamic_dtypes[name] = self.dtypes[name]

        allocator = self.output_allocator
        for name in dynamic_dtypes:
            context.set_output_allocator(name, allocator)
            context.set_tensor_address(name, 0)

        stream = torch.cuda.current_stream()
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueueV3 failed")
        if synchronize:
            stream.synchronize()

        for name in dynamic_dtypes:
            if name not in allocator.buffers or name not in allocator.shapes:
                raise RuntimeError(f"TensorRT did not allocate dynamic output {name}")
            count = int(np.prod(allocator.shapes[name], dtype=np.int64))
            output_buffers[name] = allocator.buffers[name][:count].reshape(
                allocator.shapes[name]
            )
        return output_buffers

    def allocation_stats(self):
        return {
            "buffer_slots": len(self.static_output_slots),
            "static_allocations": self.static_allocations,
            "static_reuses": self.static_reuses,
            "dynamic_allocations": self.output_allocator.allocations,
            "dynamic_reuses": self.output_allocator.reuses,
            "context_cache_capacity": self.context_cache_size,
            "context_cache_resident": len(self.context_cache)
            if self.context_cache_size > 1 else 1,
            "context_cache_hits": self.context_cache_hits,
            "context_cache_misses": self.context_cache_misses,
            "context_cache_evictions": self.context_cache_evictions,
            "shared_device_memory_bytes": self.shared_device_memory_bytes,
        }

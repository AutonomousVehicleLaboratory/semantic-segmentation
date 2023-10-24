import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time

import matplotlib.pyplot as plt
from PIL import Image
import mapillary_visualization as mapillary_visl
from tqdm import tqdm

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
engine_file = "hrnet-avl-map.engine"
folder_name = "/home/tensorrt-dep/bag_images/2023-04-24-16-49-22_0-003/avt_cameras_camera1_image_rect_color_compressed"
out_folder_name = "/home/tensorrt-dep/bag_images/2023-04-24-16-49-22_0-003/avt_cameras_camera1_image_rect_color_compressed_seg_map"
input_file  = "/home/tensorrt-dep/semantic-segmentation/imgs/test_imgs/nyc.jpg"
output_file = "output.jpg"

# For torchvision models, input images are loaded in to a range of [0, 1] and
# normalized using mean = [0.485, 0.456, 0.406] and stddev = [0.229, 0.224, 0.225].
def preprocess(image):
    # Mean normalization
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

def postprocess(data):
    num_classes = 19
    # create a color palette, selecting a color for each class
    palette = [
    (128, 64,128),
    (244, 35,232),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32),
    (  0,  0,  0)
    ]
    colors = np.array([[palette[i][0], palette[i][1], palette[i][2]] for i in range(num_classes)]).astype("uint8")
    # plot the segmentation predictions for 21 classes in different colors
    img = Image.fromarray(data.astype('uint8'), mode='P')
    img.putpalette(colors)
    return img

def postprocess_map(data):
    seg_color_ref = mapillary_visl.get_labels("/home/tensorrt-dep/semantic-segmentation/config_65.json")
    colored_output = mapillary_visl.apply_color_map(data, seg_color_ref)
    img = Image.fromarray(colored_output.astype('uint8'))
    return img


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def load_image(input_file):
    with Image.open(input_file) as img:
        img = img.resize((1920 // 2, 1440 // 2))
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height
        return input_image, image_width, image_height
    
def faster_init(engine):
    input_file = "/home/tensorrt-dep/bag_images/2023-04-24-16-49-22_0-003/avt_cameras_camera1_image_rect_color_compressed/1682354996.172460794.jpg"
    input_image, image_width, image_height = load_image(input_file)
    context = engine.create_execution_context()
    context.set_input_shape("input.1", (1, 3, image_height, image_width))
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            input_buffer = np.ascontiguousarray(input_image)
            input_memory = cuda.mem_alloc(input_image.nbytes) # This can be done only once
            bindings.append(int(input_memory)) # This can be done only once
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype) # This can be done only once
            output_memory = cuda.mem_alloc(output_buffer.nbytes) # This can be done only once
            bindings.append(int(output_memory)) # This can be done only once
    stream = cuda.Stream()
    return bindings, output_buffer, output_memory, input_memory, context, stream

def faster_infer(input_image, bindings, output_buffer, output_memory, input_memory, context, stream):
    input_buffer = np.ascontiguousarray(input_image)
    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer prediction output from the GPU.
    cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
    # Synchronize the stream
    stream.synchronize()
    return output_buffer

def faster_postprocess(output_buffer, class_num, output_file):
    reshaped_output = np.reshape(output_buffer, (class_num, 1440 // 2, 1920 // 2))
    reshaped_output_max = np.argmax(reshaped_output, axis = 0)
    img = postprocess_map(reshaped_output_max)
    # print("Writing output image to file {}".format(output_file))
    img.convert('RGB').resize((1920, 1440), Image.NEAREST).save(output_file)


def infer(engine, input_file, output_file):
    print(input_file)
    # print("Reading input image from file {}".format(input_file))
    with Image.open(input_file) as img:
        img = img.resize((1920 // 2, 1440 // 2))
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height
    total_time = 0
    start = time.time()

    # print(input_image)

    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        context.set_input_shape("input.1", (1, 3, image_height, image_width))
        # Allocate host and device buffers
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image) # This has to be done each time
                input_memory = cuda.mem_alloc(input_image.nbytes) # This can be done only once
                bindings.append(int(input_memory)) # This can be done only once
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype) # This can be done only once
                output_memory = cuda.mem_alloc(output_buffer.nbytes) # This can be done only once
                bindings.append(int(output_memory)) # This can be done only once
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        stream.synchronize()
        start = time.time()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        end = time.time()
        total_time = end - start
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()

        reshaped_output = np.reshape(output_buffer, (65, 1440 // 2, 1920 // 2))
        reshaped_output_max = np.argmax(reshaped_output, axis = 0)

    img = postprocess_map(reshaped_output_max)
    # print("Writing output image to file {}".format(output_file))
    img.convert('RGB').resize((1920, 1440), Image.NEAREST).save(output_file)
    
    return total_time

# print("Running TensorRT inference for HRNet")
# with load_engine(engine_file) as engine:
#     total_time = 0
#     num_files = len(os.listdir(folder_name))
#     for file in os.listdir(folder_name):
#         input_file = os.path.join(folder_name, file)
#         output_file = os.path.join(out_folder_name, file)
#         total_time += infer(engine, input_file, output_file)
#     print(f'Total Time taken : {(total_time)*1000}ms')
#     print(f'Avg Time taken : {((total_time)*1000)/ num_files}ms')

print("Running TensorRT inference for HRNet")
with load_engine(engine_file) as engine:
    bindings, output_buffer, output_memory, input_memory, context, stream = faster_init(engine)
    total_time = 0
    num_files = len(os.listdir(folder_name))
    for file in tqdm(os.listdir(folder_name)):
        input_file = os.path.join(folder_name, file)
        input_image, _, _ = load_image(input_file)
        start = time.time()
        faster_infer(input_image, bindings, output_buffer, output_memory, input_memory, context, stream)
        end = time.time()
        output_file = os.path.join(out_folder_name, file)
        faster_postprocess(output_buffer, 65, output_file)
        total_time += end - start
    print(f'Total Time taken : {(total_time)*1000}ms')
    print(f'Avg Time taken : {((total_time)*1000)/ num_files}ms')


# Use dummy image to create bindings array
# Fix input and output bindings so that we don't loop through all of them



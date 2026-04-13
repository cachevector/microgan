import torch
import torch.nn as nn
import os
import onnx
from onnx_tf.backend.prepare import prepare
import tensorflow as tf

def quantize_generator(model, calibration_loader, save_path="generator_quantized.tflite"):
    """
    Apply post-training quantization (PTQ) to the generator model and convert to TFLite.
    Note: This is a high-level wrapper.
    """
    model.eval()
    
    # 1. Export to ONNX
    dummy_input = torch.randn(1, model.latent_dim)
    onnx_path = "generator.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
    
    # 2. Convert ONNX to TensorFlow
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_path = "generator_tf"
    tf_rep.export_graph(tf_path)
    
    # 3. Convert TensorFlow to TFLite INT8
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset():
        for i, (imgs, _) in enumerate(calibration_loader):
            # TFLite expects a flat list of inputs
            # For GAN, input is latent vector z
            z = torch.randn(1, model.latent_dim)
            yield [z.numpy()]
            if i > 100: break # Use 100 samples for calibration

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Quantized TFLite model saved to {save_path}")
    return save_path

import os
import subprocess
import sys
import textwrap
import tempfile

def onnx_to_tflite(onnx_path, tflite_path):
    """
    Convert a MicroGAN ONNX model to TFLite.

    Uses two isolated subprocesses to avoid the TF/ONNX protobuf conflict:
      1. Extract weights from ONNX model (uses onnx, no TF)
      2. Rebuild model in TF and convert to TFLite (uses TF, no onnx)
    """
    onnx_path = os.path.abspath(onnx_path)
    tflite_path = os.path.abspath(tflite_path)

    weights_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False).name

    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    env['OMP_NUM_THREADS'] = '1'
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Stage 1: Extract weights from ONNX (no TF import)
    extract_script = textwrap.dedent(f"""\
        import numpy as np
        import onnx
        from onnx import numpy_helper

        model = onnx.load({onnx_path!r})
        weights = {{}}
        for i, init in enumerate(model.graph.initializer):
            arr = numpy_helper.to_array(init)
            weights[f"w_{{i}}"] = arr
            print(f"  [{{i}}] {{init.name}}: {{arr.shape}}")

        np.savez({weights_file!r}, **weights)
        print(f"Weights saved to {weights_file}")
    """)

    print("Stage 1: Extracting weights from ONNX model...")
    _run_script(extract_script, env, "Weight extraction")

    # Stage 2: Build TF model and convert to TFLite (no onnx import)
    convert_script = textwrap.dedent(f"""\
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        import numpy as np
        import tensorflow as tf

        data = np.load({weights_file!r})
        params = [data[f"w_{{i}}"] for i in range(len(data.files))]
        print(f"Loaded {{len(params)}} weight arrays")

        # FC layer
        fc_w = params[0]  # (2048, latent_dim) or (latent_dim, 2048)
        fc_b = params[1]  # (2048,)
        if fc_w.shape[1] == len(fc_b):
            pass  # shape is (latent_dim, 2048), correct for TF Dense
        else:
            fc_w = fc_w.T

        latent_dim = fc_w.shape[0]
        print(f"latent_dim = {{latent_dim}}")

        # ConvTranspose blocks - convert PyTorch (in_ch, out_ch, kH, kW) to TF (kH, kW, out_ch, in_ch)
        def pt_convt_to_tf(w):
            return np.transpose(w, (2, 3, 1, 0))

        conv1_w = pt_convt_to_tf(params[2])
        conv1_b = params[3]
        bn1_gamma, bn1_beta, bn1_mean, bn1_var = params[4], params[5], params[6], params[7]

        conv2_w = pt_convt_to_tf(params[8])
        conv2_b = params[9]
        bn2_gamma, bn2_beta, bn2_mean, bn2_var = params[10], params[11], params[12], params[13]

        conv3_w = pt_convt_to_tf(params[14])
        conv3_b = params[15]
        channels = conv3_w.shape[2]
        print(f"channels = {{channels}}")

        class MicroGANTF(tf.Module):
            def __init__(self):
                super().__init__()
                self.fc_w = tf.constant(fc_w, dtype=tf.float32)
                self.fc_b = tf.constant(fc_b, dtype=tf.float32)
                self.conv1_w = tf.constant(conv1_w, dtype=tf.float32)
                self.conv1_b = tf.constant(conv1_b, dtype=tf.float32)
                self.bn1_gamma = tf.constant(bn1_gamma, dtype=tf.float32)
                self.bn1_beta = tf.constant(bn1_beta, dtype=tf.float32)
                self.bn1_mean = tf.constant(bn1_mean, dtype=tf.float32)
                self.bn1_var = tf.constant(bn1_var, dtype=tf.float32)
                self.conv2_w = tf.constant(conv2_w, dtype=tf.float32)
                self.conv2_b = tf.constant(conv2_b, dtype=tf.float32)
                self.bn2_gamma = tf.constant(bn2_gamma, dtype=tf.float32)
                self.bn2_beta = tf.constant(bn2_beta, dtype=tf.float32)
                self.bn2_mean = tf.constant(bn2_mean, dtype=tf.float32)
                self.bn2_var = tf.constant(bn2_var, dtype=tf.float32)
                self.conv3_w = tf.constant(conv3_w, dtype=tf.float32)
                self.conv3_b = tf.constant(conv3_b, dtype=tf.float32)

            @tf.function(input_signature=[tf.TensorSpec(shape=[1, latent_dim], dtype=tf.float32)])
            def __call__(self, z):
                x = tf.matmul(z, self.fc_w) + self.fc_b
                x = tf.nn.relu(x)
                x = tf.reshape(x, [-1, 4, 4, 128])

                x = tf.nn.conv2d_transpose(x, self.conv1_w,
                    output_shape=[1, 8, 8, 64], strides=[1, 2, 2, 1], padding='SAME')
                x = x + self.conv1_b
                x = tf.nn.batch_normalization(x, self.bn1_mean, self.bn1_var,
                    self.bn1_beta, self.bn1_gamma, 1e-5)
                x = tf.nn.relu(x)

                x = tf.nn.conv2d_transpose(x, self.conv2_w,
                    output_shape=[1, 16, 16, 32], strides=[1, 2, 2, 1], padding='SAME')
                x = x + self.conv2_b
                x = tf.nn.batch_normalization(x, self.bn2_mean, self.bn2_var,
                    self.bn2_beta, self.bn2_gamma, 1e-5)
                x = tf.nn.relu(x)

                x = tf.nn.conv2d_transpose(x, self.conv3_w,
                    output_shape=[1, 32, 32, channels], strides=[1, 2, 2, 1], padding='SAME')
                x = x + self.conv3_b
                x = tf.math.tanh(x)
                return x

        print("Building TF model...")
        model = MicroGANTF()
        test_out = model(tf.random.normal([1, latent_dim]))
        print(f"Test output shape: {{test_out.shape}}")

        print("Converting to TFLite...")
        concrete_func = model.__call__.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open({tflite_path!r}, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_path} ({{len(tflite_model)}} bytes)")
    """)

    print("Stage 2: Building TF model and converting to TFLite...")
    _run_script(convert_script, env, "TFLite conversion")

    # Cleanup
    os.unlink(weights_file)

    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found at {tflite_path}")

    print(f"TFLite model successfully saved to {tflite_path}")
    return tflite_path


def _run_script(script_code, env, stage_name):
    """Run a Python script in an isolated subprocess."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_code)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{stage_name} failed (exit code {result.returncode})")
    finally:
        os.unlink(script_path)

import click
import os

@click.group()
def main():
    """MicroGAN: Generative AI for Microcontrollers."""
    pass

@main.command()
@click.option("--data", default=None, help="Path to image dataset.")
@click.option("--epochs", default=10, help="Number of training epochs.")
@click.option("--latent-dim", default=32, help="Dimensionality of latent space.")
@click.option("--channels", default=1, help="Number of image channels (1=grayscale, 3=RGB).")
@click.option("--output-dir", default="build", help="Directory to save artifacts.")
def train(data, epochs, latent_dim, channels, output_dir):
    """Train a MicroGAN generator."""
    import torch
    from tinygen.train.trainer import MicroGANTrainer, create_dummy_dataset
    
    click.echo(f"Training MicroGAN (Latent Dim: {latent_dim}, Channels: {channels}) for {epochs} epochs...")
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    if data is None:
        click.echo("No data provided, using dummy dataset.")
        data_loader = create_dummy_dataset(num_samples=1000, channels=channels)
    else:
        click.echo(f"Loading data from {data}...")
        data_loader = create_dummy_dataset(num_samples=1000, channels=channels)

    trainer = MicroGANTrainer(latent_dim=latent_dim, channels=channels)
    generator = trainer.train(data_loader, epochs=epochs, checkpoint_dir=checkpoint_dir)
    
    final_path = os.path.join(output_dir, "generator_final.pt")
    torch.save(generator.state_dict(), final_path)
    click.echo(f"Final model saved to {final_path}")

@main.command()
@click.option("--checkpoint", required=True, help="Path to trained PyTorch generator.")
@click.option("--latent-dim", default=32, help="Dimensionality of latent space.")
@click.option("--channels", default=1, help="Number of image channels.")
@click.option("--output-dir", default="build", help="Directory to save artifacts.")
def export_onnx(checkpoint, latent_dim, channels, output_dir):
    """Export PyTorch model to ONNX (PyTorch Only)."""
    import torch
    from tinygen.train.dcgan import MicroDCGANGenerator
    
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "generator.onnx")
    
    click.echo(f"Exporting {checkpoint} to ONNX...")
    model = MicroDCGANGenerator(latent_dim=latent_dim, channels=channels)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    
    dummy_input = torch.randn(1, latent_dim)
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
    click.echo(f"ONNX model saved to {onnx_path}")

@main.command()
@click.option("--onnx-path", required=True, help="Path to ONNX model.")
@click.option("--output-dir", default="build", help="Directory to save artifacts.")
def onnx_to_tflite(onnx_path, output_dir):
    """Convert ONNX to TFLite (Isolated Process)."""
    from tinygen.convert.to_tflite import onnx_to_tflite as convert_fn
    
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "generator_quantized.tflite")
    
    click.echo(f"Converting {onnx_path} to TFLite using isolated process...")
    convert_fn(onnx_path, tflite_path)
    click.echo(f"TFLite model saved to {tflite_path}")

@main.command()
@click.option("--tflite", required=True, help="Path to quantized TFLite model.")
@click.option("--output-dir", default="build", help="Directory to save artifacts.")
def convert(tflite, output_dir):
    """Convert TFLite model to C header file (TensorFlow Only)."""
    from tinygen.convert.to_c_array import tflite_to_c_array
    
    click.echo(f"Converting {tflite} to C header...")
    header_path = os.path.join(output_dir, "MicroGAN_weights.h")
    tflite_to_c_array(tflite, header_path=header_path)
    click.echo(f"Generated header: {header_path}")

if __name__ == "__main__":
    main()

from exp_utils import create_training_setup, Precision

if __name__ == "__main__":
    model_name = "flux" # use "stable_diffusion" for another model
    batch_size = 2
    seq_len = 512
    image_size = 64
    precision = Precision.FP
    train_step, models, optimizers, inputs = create_training_setup(
        model_name=model_name,
        batch_size=batch_size,
        seq_len=seq_len,
        precision=precision,
        image_size=image_size
    )
    train_step(models, optimizers, inputs)

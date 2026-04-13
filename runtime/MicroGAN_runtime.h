#ifndef MICROGAN_RUNTIME_H
#define MICROGAN_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MICROGAN_OK = 0,
    MICROGAN_ERROR_INIT = 1,
    MICROGAN_ERROR_GENERATE = 2,
    MICROGAN_ERROR_ARENA_TOO_SMALL = 3
} MicroGAN_status_t;

/**
 * Initialize the MicroGAN runtime.
 * @param arena A static buffer for intermediate activations (scratch space).
 * @param arena_size Size of the arena in bytes.
 * @return Status code.
 */
MicroGAN_status_t MicroGAN_init(uint8_t* arena, size_t arena_size);

/**
 * Generate a 32x32 image from a seed.
 * @param seed Random seed to generate the latent vector.
 * @param class_id Class index (0 for unconditional).
 * @param output Pointer to a buffer of size 32x32 (1024 bytes for grayscale).
 * @return Status code.
 */
MicroGAN_status_t MicroGAN_generate(uint8_t seed, uint8_t class_id, uint8_t* output);

#ifdef __cplusplus
}
#endif

#endif // MICROGAN_RUNTIME_H

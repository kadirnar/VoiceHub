/*
 * backbone_bridge.c — Thin C bridge between Python and llama.cpp.
 *
 * Wraps the llama.cpp API into simple function calls that Python ctypes
 * can call without needing to replicate complex C struct layouts.
 *
 * Build:
 *   gcc -shared -fPIC -o libbackbone_bridge.so backbone_bridge.c \
 *       -I llama.cpp/include -I llama.cpp/ggml/include \
 *       -L llama.cpp/build/bin -lllama -Wl,-rpath,'$ORIGIN/../llama.cpp/build/bin'
 */

#include "llama.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    struct llama_model   *model;
    struct llama_context *ctx;
    int32_t n_embd;
} bridge_handle_t;

/* Create model + context with embeddings enabled.
 * type_k / type_v: ggml_type enum values for KV cache quantization
 *   (pass -1 to keep llama.cpp defaults, i.e. GGML_TYPE_F16).
 * flash_attn: llama_flash_attn_type enum value
 *   (-1 = auto, 0 = disabled, 1 = enabled).
 * Returns opaque handle, or NULL on failure. */
bridge_handle_t *bridge_create(
    const char *model_path,
    int32_t     n_ctx,
    int32_t     n_batch,
    int32_t     n_threads,
    int32_t     n_gpu_layers,
    int32_t     type_k,
    int32_t     type_v,
    int32_t     flash_attn)
{
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "bridge_create: failed to load model from %s\n", model_path);
        return NULL;
    }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = (uint32_t)n_ctx;
    cp.n_batch       = (uint32_t)n_batch;
    cp.n_ubatch      = (uint32_t)n_batch;
    cp.n_threads     = n_threads;
    cp.n_threads_batch = n_threads;
    cp.embeddings    = true;

    if (type_k >= 0) cp.type_k = (enum ggml_type)type_k;
    if (type_v >= 0) cp.type_v = (enum ggml_type)type_v;
    cp.flash_attn_type = (enum llama_flash_attn_type)flash_attn;

    fprintf(stderr, "bridge_create: type_k=%d, type_v=%d, flash_attn=%d\n",
            (int)cp.type_k, (int)cp.type_v, (int)cp.flash_attn_type);

    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "bridge_create: failed to create context\n");
        llama_model_free(model);
        return NULL;
    }

    bridge_handle_t *h = (bridge_handle_t *)malloc(sizeof(bridge_handle_t));
    h->model  = model;
    h->ctx    = ctx;
    h->n_embd = llama_model_n_embd(model);

    fprintf(stderr, "bridge_create: loaded model, n_embd=%d, n_ctx=%d\n",
            h->n_embd, n_ctx);
    return h;
}

/* Feed one embedding vector at a given position.
 * embd: float array of size n_embd.
 * Returns 0 on success, non-zero on error. */
int32_t bridge_decode_embd(
    bridge_handle_t *h,
    const float     *embd,
    int32_t          pos,
    int8_t           output)
{
    struct llama_batch batch = llama_batch_init(1, h->n_embd, 1);
    batch.n_tokens = 1;
    memcpy(batch.embd, embd, (size_t)h->n_embd * sizeof(float));
    batch.pos[0]      = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]   = output;

    int32_t ret = llama_decode(h->ctx, batch);
    llama_batch_free(batch);
    return ret;
}

/* Feed multiple embedding vectors (for prefill).
 * embds: float array of size n_tokens * n_embd (row-major).
 * pos_start: position of first token.
 * output_last: if true, request output only for the last token.
 * Returns 0 on success, non-zero on error. */
int32_t bridge_decode_embd_batch(
    bridge_handle_t *h,
    const float     *embds,
    int32_t          n_tokens,
    int32_t          pos_start,
    int8_t           output_last)
{
    struct llama_batch batch = llama_batch_init(n_tokens, h->n_embd, 1);
    batch.n_tokens = n_tokens;
    memcpy(batch.embd, embds, (size_t)n_tokens * (size_t)h->n_embd * sizeof(float));

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.pos[i]       = pos_start + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (output_last && i == n_tokens - 1) ? 1 : 0;
    }

    int32_t ret = llama_decode(h->ctx, batch);
    llama_batch_free(batch);
    return ret;
}

/* Get the embedding/hidden-state for the i-th output token.
 * Returns pointer to n_embd floats (owned by llama.cpp, valid until next decode).
 * i = -1 means the last output token. */
float *bridge_get_embeddings(bridge_handle_t *h, int32_t i)
{
    return llama_get_embeddings_ith(h->ctx, i);
}

/* Get the logits for the i-th output token.
 * Returns pointer to n_vocab floats (owned by llama.cpp, valid until next decode).
 * i = -1 means the last output token. */
float *bridge_get_logits(bridge_handle_t *h, int32_t i)
{
    return llama_get_logits_ith(h->ctx, i);
}

/* Return the embedding dimension. */
int32_t bridge_n_embd(bridge_handle_t *h)
{
    return h->n_embd;
}

/* Return the vocabulary size. */
int32_t bridge_n_vocab(bridge_handle_t *h)
{
    const struct llama_vocab *vocab = llama_model_get_vocab(h->model);
    return llama_vocab_n_tokens(vocab);
}

/* Clear KV cache. */
void bridge_clear_kv(bridge_handle_t *h)
{
    llama_memory_t mem = llama_get_memory(h->ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
}

/* Free everything. */
void bridge_free(bridge_handle_t *h)
{
    if (!h) return;
    if (h->ctx)   llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    free(h);
}

# Local weight placeholders

The zero-byte checkpoint/model.pt and text_encoder/model.safetensors files are placeholders, not trained weights. Restore the selected V1 weights from a trusted backup and verify the package LOCAL_ASSETS.sha256 manifest before inference. The shared loader rejects empty weight files.

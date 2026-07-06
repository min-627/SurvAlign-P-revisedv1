import argparse
import sys
import os
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download

# Add the cloned repo to sys.path so we can import its modules
sys.path.append(os.path.join(os.path.dirname(__file__), "facodec_lib"))

try:
    from ns3_codec import FACodecEncoder, FACodecDecoder
except Exception as e:
    print(f"Error importing FACodec modules: {e}")
    print("Please make sure tools/facodec_lib exists and its dependencies are installed (e.g., pip install pyworld).")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input wav file path")
    parser.add_argument("--output", required=True, help="Output wav file path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    ).to(device)

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    ).to(device)

    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

    fa_encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))

    fa_encoder.eval()
    fa_decoder.eval()

    # Load audio as 16kHz
    import torchaudio
    wav, sr = torchaudio.load(args.input)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    wav = wav.to(device)
    wav = wav.unsqueeze(0)

    with torch.no_grad():
        enc_out = fa_encoder(wav)
        vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
        recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
    
    sf.write(args.output, recon_wav[0][0].cpu().numpy(), 16000)

if __name__ == "__main__":
    main()

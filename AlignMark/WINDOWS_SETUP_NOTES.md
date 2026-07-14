# Windows 로컬 개발 환경 참고 사항

`requirements.txt`는 RunPod(Linux) 환경을 기준으로 작성되어 있으며, `torch`,
`torchaudio`, `triton`, `nvidia-*` CUDA 패키지가 원본 그대로 포함되어 있습니다.
이 파일 자체는 수정하지 마세요.

Windows 로컬에서 `pip install -r requirements.txt`를 그대로 실행하면 위 패키지들의
설치가 실패하거나 맞지 않는 빌드가 설치될 수 있습니다. Windows에서 로컬로 작업할 때는:

1. `requirements.txt`에서 `torch`, `torchaudio`, `triton`, `nvidia-*` 줄을 제외한
   나머지 의존성만 설치합니다.
2. `torch`/`torchaudio`는 PyTorch 공식 index에서 CUDA 12.4(cu124) 빌드를 별도로
   설치합니다:

   ```
   pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ```

3. `triton`은 Windows에서 공식 지원되지 않으므로 로컬에서는 생략하고, 실제 학습/공격
   실험은 RunPod(Linux)에서 원본 `requirements.txt` 그대로 진행합니다.

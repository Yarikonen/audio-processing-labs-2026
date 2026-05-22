"""
Standalone VITS training script for RUSLAN dataset.
Run from project root:
    .venv\Scripts\python.exe train_vits_ruslan.py 2>&1 | Tee-Object training_ruslan.log
"""
import os, sys, time

# Windows console defaults to cp1251 which can't encode IPA characters.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# espeak-ng PATH (Windows)
_esp = r"C:\Program Files\eSpeak NG"
if os.path.isdir(_esp) and _esp not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _esp + os.pathsep + os.environ["PATH"]

import warnings
warnings.filterwarnings("ignore")

# torchaudio 2.6+ defaults to torchcodec which needs FFmpeg DLLs.
# Patch load() to use soundfile so no FFmpeg is required.
import torchaudio as _ta
import soundfile as _sf
import torch as _torch

def _sf_load(filepath, frame_offset=0, num_frames=-1,
             normalize=True, channels_first=True, **kwargs):
    data, sr = _sf.read(str(filepath), dtype="float32", always_2d=True)
    t = _torch.from_numpy(data.T.copy())   # (channels, samples)
    if frame_offset > 0:
        t = t[:, frame_offset:]
    if num_frames > 0:
        t = t[:, :num_frames]
    if not channels_first:
        t = t.T
    return t, sr

_ta.load = _sf_load

# espeak-ng passes text as a CLI argument which breaks Cyrillic on Windows.
# Patch phonemize_espeak() to send text via stdin instead.
import re as _re
import subprocess as _subprocess
from packaging.version import Version as _Version
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak as _ESpeak

def _phonemize_espeak_stdin(self, text: str, separator: str = "|", tie=False) -> str:
    args = ["-v", self._language]
    if self.backend == "espeak":
        args.append("--ipa=1" if _Version(self.backend_version) >= _Version("1.48.15") else "--ipa=3")
    else:
        args.append("--ipa=1")
    if tie:
        args.append("--tie=%s" % tie)
    cmd = [self._ESPEAK_LIB, "-q", "-b", "1"] + args
    with _subprocess.Popen(cmd, stdin=_subprocess.PIPE, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE) as p:
        stdout, _ = p.communicate(input=text.encode("utf-8"))
    phonemes = ""
    for line in stdout.decode("utf-8").splitlines():
        ph = _re.sub(r"\(.+?\)", "", line.strip())
        phonemes += ph.strip()
    return phonemes.replace("_", separator)

_ESpeak.phonemize_espeak = _phonemize_espeak_stdin

from pathlib import Path
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import BaseDatasetConfig
import TTS.tts.datasets.formatters as _fmt_module

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
RUSLAN_DIR = BASE_DIR / "audio" / "lab3_data" / "RUSLAN" / "RUSLAN"
OUT_DIR    = BASE_DIR / "src" / "audio" / "lab3_results" / "vits_ruslan_scratch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert (RUSLAN_DIR / "metadata.csv").exists(), \
    f"RUSLAN metadata not found at {RUSLAN_DIR / 'metadata.csv'}"

# ── Custom formatter ───────────────────────────────────────────────────────────
def ruslan_formatter(root_path, manifest_file, **kwargs):
    items = []
    wav_dir = os.path.join(root_path, "wavs")
    with open(os.path.join(root_path, manifest_file), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            wav_id = parts[0].strip()
            text   = parts[2].strip() if len(parts) >= 3 else parts[1].strip()
            wav_path = os.path.join(wav_dir, wav_id + ".wav")
            if not os.path.exists(wav_path):
                wav_path = os.path.join(wav_dir, wav_id)
            if os.path.exists(wav_path):
                items.append({
                    "text":         text,
                    "audio_file":   wav_path,
                    "speaker_name": "ruslan",
                    "root_path":    root_path,
                    "language":     "ru",
                })
    print(f"[ruslan_formatter] {len(items)} samples loaded")
    return items

_fmt_module.ruslan_formatter = ruslan_formatter

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASET_CFG = BaseDatasetConfig(
    formatter="ruslan_formatter",
    meta_file_train="metadata.csv",
    path=str(RUSLAN_DIR),
    language="ru",
)

# ── Model config ───────────────────────────────────────────────────────────────
audio_cfg = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,

)

config = VitsConfig(
    audio=audio_cfg,
    run_name="vits_ruslan_scratch",
    batch_size=16,
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=100,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="ru",
    phoneme_cache_path=str(OUT_DIR / "phoneme_cache_ru"),
    compute_input_seq_cache=True,
    lr_gen=1e-4,
    lr_disc=1e-4,
    mixed_precision=False,
    print_step=50,
    save_step=1000,
    save_n_checkpoints=5,
    save_checkpoints=True,
    save_best_after=1000,
    output_path=str(OUT_DIR),
    datasets=[DATASET_CFG],
    cudnn_benchmark=False,
    test_sentences=[
        "Нейронные сети изменили подход к синтезу речи.",
        "Как звучит синтезированная речь на русском языке?",
        "Это действительно работает — синтез с нуля!",
        "Архитектура включает три компонента: энкодер, потоки и декодер.",
    ],
)

if __name__ == "__main__":
    # Автопоиск последнего чекпоинта для продолжения обучения.
    # Чтобы начать с нуля — установи RESUME = False
    RESUME = False
    _ckpts = sorted(OUT_DIR.rglob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime)
    _restore = str(_ckpts[-1]) if RESUME and _ckpts else None
    if _restore:
        print(f"Resuming from: {_restore}")
    else:
        print("Starting from scratch")

    # ── Init ───────────────────────────────────────────────────────────────────
    ap  = AudioProcessor(**config.audio.to_dict())
    tok, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        DATASET_CFG,
        eval_split=True,
        eval_split_max_size=500,
        eval_split_size=0.01,
        formatter=ruslan_formatter,
    )
    print(f"Train: {len(train_samples)} | Eval: {len(eval_samples)}")

    model = Vits(config, ap, tok, speaker_manager=None)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = Trainer(
        TrainerArgs(restore_path=_restore, skip_train_epoch=False),
        config,
        output_path=str(OUT_DIR),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print(f"\nOutput dir : {OUT_DIR}")
    print(f"TensorBoard: tensorboard --logdir {OUT_DIR}")
    print(f"Started at : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    t0 = time.time()
    trainer.fit()
    elapsed = time.time() - t0

    print("=" * 65)
    print(f"Done at  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed  : {elapsed/3600:.1f} h ({elapsed/60:.0f} min)")

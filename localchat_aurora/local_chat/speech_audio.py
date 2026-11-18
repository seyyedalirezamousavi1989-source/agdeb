"""Clean-up of synthesized speech pieces (numpy only; shared by the TTS worker and the tests).

XTTS sometimes fails to emit its stop token and keeps "talking" after the sentence is finished: the listener
hears a meaningless mumble or a sound at the end of the sentence. ``clean_piece`` removes that run-on tail and
the silence around the speech, and ends the piece with a short fade so pieces join without clicks.
"""
import numpy as np

FRAME_SECONDS = 0.02
# Normal Persian speech is about 11-15 characters per second. A piece much longer than this is a run-on.
RUNON_CHARS_PER_SECOND = 9.0     # longer than chars/9 (+margin) is suspicious
HARD_CHARS_PER_SECOND = 6.0      # never keep more than chars/6 (+margin), even for a very slow reference voice
FAST_CHARS_PER_SECOND = 14.0     # a cut is not allowed earlier than the fastest plausible reading time
MARGIN_SECONDS = 1.2
GAP_SECONDS = 0.22               # a pause this long, after the plausible end, is where the sentence really stopped
KEEP_TAIL_SECONDS = 0.10
KEEP_HEAD_SECONDS = 0.04
FADE_SECONDS = 0.05


def _frame_rms(audio, frame):
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    shaped = audio[:count * frame].reshape(count, frame)
    return np.sqrt(np.mean(shaped * shaped, axis=1))


def clean_piece(audio, sample_rate, n_chars):
    """Return ``audio`` without a run-on tail, leading/trailing silence, and with a short fade-out."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    frame = max(1, int(sample_rate * FRAME_SECONDS))
    rms = _frame_rms(audio, frame)
    if rms.size == 0:
        return audio
    loud = float(np.percentile(rms, 90))
    if loud <= 1e-6:
        return audio
    silent = rms < loud * 0.06
    seconds = audio.size / sample_rate
    chars = max(1, int(n_chars))

    end_frame = rms.size
    if seconds > chars / RUNON_CHARS_PER_SECOND + MARGIN_SECONDS:
        earliest = int((chars / FAST_CHARS_PER_SECOND) / FRAME_SECONDS)
        hard = int((chars / HARD_CHARS_PER_SECOND + MARGIN_SECONDS) / FRAME_SECONDS)
        need = max(1, int(GAP_SECONDS / FRAME_SECONDS))
        cut = None
        run = 0
        for index in range(min(earliest, rms.size), min(hard, rms.size)):
            run = run + 1 if silent[index] else 0
            if run >= need:
                cut = index - run + 1
                break
        end_frame = cut if cut is not None else min(hard, rms.size)

    voiced = np.flatnonzero(~silent[:end_frame])
    if voiced.size == 0:
        return audio
    first = max(0, int(voiced[0]) - int(KEEP_HEAD_SECONDS / FRAME_SECONDS))
    last = min(end_frame, int(voiced[-1]) + 1 + int(KEEP_TAIL_SECONDS / FRAME_SECONDS))
    out = audio[first * frame:last * frame].copy()
    fade = min(out.size, int(sample_rate * FADE_SECONDS))
    if fade > 1:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out

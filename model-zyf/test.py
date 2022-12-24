#!/usr/bin/env python3

import os
import sys
from glob import glob
import librosa
import time
import numpy

numpy.random.seed(42)

# LibriSpeech, download train data from here: http://www.openslr.org/12/
lots_of_flac_files = sorted(glob("../LibriSpeech-SI/train/*/*/*.flac"))
print(lots_of_flac_files)
#numpy.random.shuffle(lots_of_flac_files)
lots_of_flac_files = lots_of_flac_files[:10000]

print("num files:", len(lots_of_flac_files))


def monkeyfix_glib():
  """
  Fixes some stupid bugs such that SIGINT is not working.
  This is used by audioread, and indirectly by librosa for loading audio.

  https://stackoverflow.com/questions/16410852/
  """
  try:
    import gi
  except ImportError:
    return
  try:
    from gi.repository import GLib
  except ImportError:
    from gi.overrides import GLib
  # Do nothing.
  # The original behavior would install a SIGINT handler which calls GLib.MainLoop.quit(),
  # and then reraise a KeyboardInterrupt in that thread.
  # However, we want and expect to get the KeyboardInterrupt in the main thread.
  GLib.MainLoop.__init__ = lambda *args, **kwargs: None


def monkeypatch_audioread():
  """
  audioread does not behave optimal in some cases.
  E.g. each call to _ca_available() takes quite long because of the ctypes.util.find_library usage.
  We will patch this.
  """
  try:
    import audioread
  except ImportError:
    return
  res = audioread._ca_available()
  audioread._ca_available = lambda: res


def hms(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def terminal_size(file=sys.stdout):  # this will probably work on linux only
  import os, sys, io
  if not hasattr(file, "fileno"):
    return -1, -1
  try:
    if not os.isatty(file.fileno()):
      return -1, -1
  except io.UnsupportedOperation:
    return -1, -1
  env = os.environ
  def ioctl_GWINSZ(fd):
    try:
      import fcntl, termios, struct, os
      cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
    except Exception:
        return
    return cr
  cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
  if not cr:
    try:
        fd = os.open(os.ctermid(), os.O_RDONLY)
        cr = ioctl_GWINSZ(fd)
        os.close(fd)
    except Exception:
        pass
  if not cr:
    cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
  return int(cr[1]), int(cr[0])



def progress_bar(complete=1.0, prefix="", suffix="", file=sys.stdout):
  import sys
  terminal_width, _ = terminal_size(file=file)
  if terminal_width == -1: return
  if complete == 1.0:
    file.write("\r%s"%(terminal_width * ' '))
    file.flush()
    file.write("\r")
    file.flush()
    return
  progress = "%.02f%%" % (complete * 100)
  if prefix != "": prefix = prefix + " "
  if suffix != "": suffix = " " + suffix
  ntotal = terminal_width - len(progress) - len(prefix) - len(suffix) - 4
  bars = '|' * int(complete * ntotal)
  spaces = ' ' * (ntotal - int(complete * ntotal))
  bar = bars + spaces
  file.write("\r%s" % prefix + "[" + bar[:len(bar)//2] + " " + progress + " " + bar[len(bar)//2:] + "]" + suffix)
  file.flush()



monkeyfix_glib()
monkeypatch_audioread()



stdout_tty = os.isatty(sys.stdout.fileno())

start_time = time.time()
total_audio_len = 0
for i, fn in enumerate(lots_of_flac_files):
    audio, sample_rate = librosa.load(fn, sr=None)
    total_audio_len += len(audio)
    mfccs = librosa.feature.mfcc(
        audio, sr=sample_rate,
        n_mfcc=40,
        hop_length=int(0.01 * sample_rate), n_fft=int(0.025 * sample_rate))

    complete_frac = float(i + 1) / len(lots_of_flac_files)
    start_elapsed = time.time() - start_time
    progress = "%i/%i (%.02f%%)" % (i + 1, len(lots_of_flac_files), complete_frac * 100)
    if complete_frac > 0:
        total_time_estimated = start_elapsed / complete_frac
        remaining_estimated = total_time_estimated - start_elapsed
        progress += " (%s)" % hms(remaining_estimated)
        if i % (len(lots_of_flac_files) // 10) == 0:
            avg_audio_len = float(total_audio_len) / (i + 1)
            print(
                "current:", progress,
                "total estimated:", hms(total_time_estimated),
                "avg audio len:", avg_audio_len)

    if stdout_tty:
        progress_bar(complete_frac, progress)
    else:
        print(progress)

print("Done. Total time:", hms(time.time() - start_time))
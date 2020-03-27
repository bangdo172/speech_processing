
import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy


q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def record(filename, sentence, fileText, name):
    try:
        # Make sure the file is opened before recording anything:
        with sf.SoundFile(filename, mode='x', samplerate=22000, channels=1) as file:
            with sd.InputStream(samplerate=22000, device=sd.default.device, channels=1, callback=callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(filename))

    file = open(fileText, "a")
    file.write(name + '\n')
    file.write(sentence + '\n')
    file.close()
sentence = 'Hà Nôi giảm 80% lượt xe buýt, tương đương 12.400 lượt mỗi ngày để phòng chống Covid-19, bắt đầu từ 27/3.'
record('rec.wav', sentence, '', 'file.txt')
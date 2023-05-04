import time
from audio2talking import A2T

if __name__ == '__main__':
    audio2talking = A2T()
    start_ = time.time()
    audio2talking.run(audio_path=r"./demo/audio/test.wav", )
    print("Time run: ", time.time() - start_)

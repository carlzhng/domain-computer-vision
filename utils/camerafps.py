import cv2

class CvFpsCalc(object):
    def __init__(self, buffer_len=10):
        self._start_tick = cv2.getTickCount()
        self._freq = cv2.getTickFrequency()
        self._difftimes = []
        self._buffer_len = buffer_len
    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick)/self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)
        if len(self._difftimes) > self._buffer_len:
            self._difftimes.pop(0)
        
        fps = 1.0 / (sum(self._difftimes)/len(self._difftimes))
        return int(fps)
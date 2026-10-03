import numpy as np
import glob
import os
from collections import Counter

# CSV 파일 검색
files = sorted(glob.glob('data/**/*.csv', recursive=True))

print('파일 개수:', len(files))
print('예시 경로:', files[0])

# 데이터 로드 (헤더 유무에 따라 skiprows=1 또는 0 조정)
x = np.loadtxt(files[0], delimiter=',', skiprows=1)

print('배열 모양:', x.shape)
print('값 범위:', x.min(), '~', x.max())

# 피험자별 파일 수 세기
subs = [os.path.basename(os.path.dirname(f)) for f in files]
print('피험자별 파일 수:', Counter(subs))

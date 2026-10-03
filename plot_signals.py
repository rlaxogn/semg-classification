import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# 1. 파일 가져오기
files = sorted(glob.glob('data/**/*.csv', recursive=True))

selected_files = []
seen_subjects = set()

for f in files:
    sub = os.path.basename(os.path.dirname(f))
    if sub not in seen_subjects:
        seen_subjects.add(sub)
        selected_files.append((sub, f))
    if len(selected_files) == 3:
        break

# 2. Y축 고정을 해제(sharey=False)하여 미세한 변화도 크게 보이도록 설정
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=False)

for i, (sub, file_path) in enumerate(selected_files):
    # delimiter 및 skiprows 설정 확인 (필요시 skiprows=0으로 변경)
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    except:
        data = np.loadtxt(file_path, delimiter=',')
        
    # (3000, 2) 형태로 축 맞추기
    if data.shape[0] < data.shape[1]:
        data = data.T
        
    t = np.arange(len(data)) / 1000.0
    
    for ch in range(2):
        ax = axes[i, ch]
        ax.plot(t, data[:, ch], lw=0.8, color='blue')
        
        # 1초, 2초 지점에 빨간 점선
        ax.axvline(1.0, color='r', ls='--')
        ax.axvline(2.0, color='r', ls='--')
        
        # Y축 범위를 실제 값의 최소/최대에 맞춰 잘 보이게 조정
        y_min, y_max = data[:, ch].min(), data[:, ch].max()
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        if i == 0:
            ax.set_title(f'Channel {ch+1}')
        if ch == 0:
            ax.set_ylabel(f'{sub}\nAmplitude')

for ax in axes[2]:
    ax.set_xlabel('Time (s)')

plt.suptitle('sEMG Signal Comparison Across 3 Subjects', fontsize=14)
plt.tight_layout()
plt.savefig('signal_example.png', dpi=120)
plt.show()
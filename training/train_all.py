# train_all.py
import subprocess
import sys

# 정의된 스크립트들
SCRIPTS = [
    'training.train_linear',
    'training.train_lstm',
    'training.train_mlp',
    'training.train_gru',
    'training.train_tcn',
    'training.train_transformer',
]

if __name__ == '__main__':
    # 선택적으로 커맨드라인 인자로 모델 이름을 받을 수 있도록
    args = sys.argv[1:]
    to_run = []

    if not args or 'all' in args:
        to_run = SCRIPTS
    else:
        # 입력된 모델 이름 매칭
        for name in args:
            module = f'training.train_{name}'
            if module in SCRIPTS:
                to_run.append(module)
            else:
                print(f"Unknown model: {name}")

    for module in to_run:
        print(f"\n▶ Running {module}...\n")
        ret = subprocess.run([
            sys.executable, '-m', module
        ])
        if ret.returncode != 0:
            print(f"Error running {module}, exit code: {ret.returncode}")
            break

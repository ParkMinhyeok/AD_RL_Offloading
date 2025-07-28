import os

def load_times_from_txt(filepath: str) -> list[float]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")

    times_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    times_list.append(float(line))
                except ValueError:
                    print(f"경고: '{filepath}' 파일의 '{line}'은 숫자가 아니므로 무시합니다.")
    return times_list
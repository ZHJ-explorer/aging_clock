import gzip

# 读取 features.tsv.gz 文件
with gzip.open('data/GSM6588511_F30_features.tsv.gz', 'rt') as f:
    for i, line in enumerate(f):
        if i < 10:
            print(f"Line {i+1}: {line.strip()}")
        else:
            break

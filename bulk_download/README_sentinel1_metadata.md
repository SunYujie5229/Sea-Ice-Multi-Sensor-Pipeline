# Sentinel-1元数据季度批量下载工具 (2014-2024)

本工具用于批量下载2014-2024年间Sentinel-1卫星在指定区域（Arctic Canada North）的元数据，并按年份和季度进行组织保存。

## 功能特点

1. **按季度批量下载**：将2014-2024年间的数据按季度（每三个月）进行检索，避免超出ASF检索API上限
2. **分层组织存储**：每年的四个季度数据保存在对应年份的文件夹下
3. **年度数据合并**：自动合并每年的四个季度数据为一个年度文件
4. **错误处理机制**：包含重试逻辑和详细的错误报告，确保数据下载的可靠性

## 文件说明

- `batch_sentinel1_metadata_quarterly.py`：Python脚本版本，可在命令行运行
- `batch_sentinel1_metadata_quarterly.ipynb`：Jupyter Notebook版本，提供交互式运行环境

## 使用方法

### 方法一：运行Python脚本

```bash
python batch_sentinel1_metadata_quarterly.py
```

### 方法二：使用Jupyter Notebook

1. 打开`batch_sentinel1_metadata_quarterly.ipynb`
2. 可以先运行测试单个季度的代码块（第3节）
3. 然后运行小范围年份的批处理（第4节）
4. 确认一切正常后，可以取消注释并运行完整的2014-2024年处理（第5节）

## 输出结构

```
sentinel1_metadata_2014_2024/
├── 2014/
│   ├── sentinel1_2014_Q1.csv
│   ├── sentinel1_2014_Q2.csv
│   ├── sentinel1_2014_Q3.csv
│   ├── sentinel1_2014_Q4.csv
│   └── sentinel1_2014_all.csv
├── 2015/
│   ├── ...
...
└── 2024/
    ├── ...
```

## 注意事项

1. **API限制**：ASF API有查询次数和结果数量的限制，按季度检索可以避免超出限制
2. **处理时间**：完整处理2014-2024年的数据可能需要较长时间，建议先测试小范围年份
3. **网络连接**：确保网络连接稳定，程序包含重试机制但仍可能受网络问题影响
4. **存储空间**：完整的元数据可能占用较大存储空间，请确保有足够的磁盘空间

## 错误处理

程序包含以下错误处理机制：

1. **查询重试**：当API查询失败时，会自动重试最多3次，每次等待时间递增
2. **详细日志**：记录处理过程中的错误和警告，便于排查问题
3. **异常捕获**：对每个季度和年份的处理都有独立的异常捕获，确保单个季度的失败不会影响整体处理

## 依赖库

- asf_search
- pandas
- numpy
- dateutil

## 参考

- 基于`ASD_S1_CSV_export.ipynb`和`sentinel1_metadata_processor.py`开发
- 使用ASF API进行Sentinel-1数据检索
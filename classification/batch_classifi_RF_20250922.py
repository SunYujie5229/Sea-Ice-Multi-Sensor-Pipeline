import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.windows import Window
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import gc
import psutil
from tqdm import tqdm
import glob
import time
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix


plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决坐标轴负号显示问题

class Sentinel1IceClassifier:
    def __init__(self, selected_features=None):
        self.rf_model = None
        
        # 所有可用的特征选项
        self.available_features = {
            'HH': 'HH波段（水平发射-水平接收）',
            'HV': 'HV波段（水平发射-垂直接收）',
            'HH_div_HV': 'HH/HV比值',
            'HH_minus_HV': 'HH-HV差值', 
            'ANGLE': '入射角',
            'HH_norm': 'HH标准化',
            'HV_norm': 'HV标准化',
            'sum_div_diff': '总和除差值'
        }
        
        # 默认特征组合（如果没有指定）
        if selected_features is None:
            self.feature_names = ['HH', 'HV', 'HH_div_HV', 'ANGLE']
        else:
            self.feature_names = selected_features
        
        # 验证选择的特征是否有效
        for feature in self.feature_names:
            if feature not in self.available_features:
                raise ValueError(f"特征 '{feature}' 不在可用特征列表中")
            
        ###NOTE:tiff band name需要从tiff中读取 直接命名受限于波段计算顺序
        
        # TIFF波段名称
        self.tiff_band_names = ['HH', 'HV', 'ANGLE', 'SIC', 
                               'HH_mul_HV', 'HH_div_HV', 'sum_div_diff']
        
        self.class_names = ['ice', 'lead', 'refrozen']
        self.class_colors = ["#1f76b4", "#ff7f0e", "#2ca02c"]  # 蓝色、橙色、绿色
        
        print(f"初始化分类器，使用特征: {self.feature_names}")

    def update_selected_features(self, new_features):
        """更新选择的特征"""
        for feature in new_features:
            if feature not in self.available_features:
                raise ValueError(f"特征 '{feature}' 不在可用特征列表中")
        
        self.feature_names = new_features
        print(f"特征已更新为: {self.feature_names}")
        
        if self.rf_model is not None:
            print("警告: 特征已更改，需重新训练模型")
            self.rf_model = None

    def load_and_merge_samples(self, csv_files):
        """加载并合并多个日期的样本CSV文件"""
        print("正在加载样本数据...")
        all_samples = []
        
        if isinstance(csv_files, str):
            csv_files = [csv_files]
            
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            print(f"加载文件: {csv_file}, 样本数: {len(df)}")
            all_samples.append(df)
        
        merged_samples = pd.concat(all_samples, ignore_index=True)
        print(f"合并后总样本数: {len(merged_samples)}")
        
        class_counts = merged_samples['class'].value_counts().sort_index()
        print("\n类别分布:")
        for class_id, count in class_counts.items():
            print(f"  {self.class_names[class_id-1]} (class {class_id}): {count} 样本")
        
        # --- 增加欠采样逻辑 ---
        min_count = class_counts.min()  # 找到样本数最少的类别数量
        balanced_samples = pd.DataFrame()

        print(f"\n进行欠采样，每个类别将保留 {min_count} 个样本...")
        for class_id in class_counts.index:
            class_samples = merged_samples[merged_samples['class'] == class_id]
            # 随机抽取与最少类别数量相等的样本
            sampled_class_samples = class_samples.sample(n=min_count, random_state=42) # 使用 random_state 保证可复现
            balanced_samples = pd.concat([balanced_samples, sampled_class_samples])
        
        # 再次检查新的类别分布
        balanced_class_counts = balanced_samples['class'].value_counts().sort_index()
        print("\n欠采样后类别分布:")
        for class_id, count in balanced_class_counts.items():
            print(f"  {self.class_names[class_id-1]} (class {class_id}): {count} 样本")
        
        print(f"欠采样后总样本数: {len(balanced_samples)}")
        
        return balanced_samples.sample(frac=1, random_state=42).reset_index(drop=True) # 随机打乱并重置索引

 
    def prepare_training_data(self, samples_df, filter_features=True):
        """
        准备训练数据，支持特征筛选
        """
        print("\n准备训练数据...")
        
        # 创建特征名称映射
        csv_feature_mapping = {
            'HH': 'HH',
            'HV': 'HV', 
            'HH_div_HV': 'HH_div_HV',
            'HH_minus_HV': 'HH_minus_HV',
            'indicenceAngle': 'ANGLE',  # 注意CSV中的拼写
            'HH_norm': 'HH_norm',
            'HV_norm': 'HV_norm',
            'sum_div_diff': 'sum_div_diff'
        }
        
        if filter_features:
            # 只选择当前配置使用的特征
            selected_csv_cols = []
            missing_features = []
            
            for feature in self.feature_names:
                csv_col = None
                for csv_name, std_name in csv_feature_mapping.items():
                    if std_name == feature:
                        csv_col = csv_name
                        break
                
                if csv_col and csv_col in samples_df.columns:
                    selected_csv_cols.append(csv_col)
                else:
                    missing_features.append(feature)
                    print(f"警告: CSV中未找到特征 '{feature}' 对应的列")
            
            if missing_features:
                print(f"缺失的特征: {missing_features}")
                self.feature_names = [f for f in self.feature_names if f not in missing_features]
        else:
            # 使用所有可用特征
            selected_csv_cols = [col for col in csv_feature_mapping.keys() if col in samples_df.columns]
        
        if not selected_csv_cols:
            raise ValueError("没有找到任何可用的特征列")
        
        X = samples_df[selected_csv_cols].copy()
        y = samples_df['class'].copy()
        
        # 重命名列以匹配标准特征名
        rename_mapping = {}
        for csv_col in selected_csv_cols:
            for csv_name, std_name in csv_feature_mapping.items():
                if csv_name == csv_col:
                    rename_mapping[csv_col] = std_name
                    break
        
        X.rename(columns=rename_mapping, inplace=True)
        
        if X.isnull().any().any():
            print("警告: 发现缺失值，将使用均值填充")
            X = X.fillna(X.mean())
            
        if filter_features:
            X = X[self.feature_names]
            
        print(f"实际使用的特征: {list(X.columns)}")
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")
        
        return X, y
    
    def debug_feature_info(self):
        """调试函数：显示模型和分类器的特征信息"""
        print("=== 特征匹配调试信息 ===")
        print(f"分类器配置的特征: {self.feature_names}")
        print(f"特征数量: {len(self.feature_names)}")
        
        if self.rf_model is not None:
            print(f"模型期望的特征数: {getattr(self.rf_model, 'n_features_in_', '未知')}")
            if hasattr(self.rf_model, 'feature_names_in_'):
                print(f"模型训练时的特征名: {self.rf_model.feature_names_in_}")
            
            # 检查是否匹配
            expected_features = getattr(self.rf_model, 'n_features_in_', len(self.feature_names))
            if len(self.feature_names) != expected_features:
                print(f"⌧ 特征数量不匹配：期望 {expected_features}，配置 {len(self.feature_names)}")
            else:
                print("✅ 特征数量匹配")
        else:
            print("⚠️  模型尚未训练")
        
        print(f"可用特征列表: {list(self.available_features.keys())}")
        print("=" * 40)

    def check_tiff_bands(self, tiff_path):
        """检查TIFF文件的波段信息"""
        print(f"\n=== TIFF文件波段信息 ===")
        print(f"文件路径: {tiff_path}")
        
        with rasterio.open(tiff_path) as src:
            print(f"总波段数: {src.count}")
            print(f"影像尺寸: {src.height} x {src.width}")
            
            print("波段对应关系:")
            for i in range(min(src.count, len(self.tiff_band_names))):
                print(f"  波段 {i+1}: {self.tiff_band_names[i]}")
            
            # 检查所需特征是否可用
            print("\n特征可用性检查:")
            for feature in self.feature_names:
                if feature in self.tiff_band_names[:src.count]:
                    idx = self.tiff_band_names.index(feature)
                    print(f"  ✅ {feature}: 可用 (波段 {idx+1})")
                elif feature in ['HH_div_HV', 'HH_minus_HV', 'sum_div_diff']:
                    hh_available = 'HH' in self.tiff_band_names[:src.count]
                    hv_available = 'HV' in self.tiff_band_names[:src.count]
                    if hh_available and hv_available:
                        print(f"  ✅ {feature}: 可计算 (从HH和HV)")
                    else:
                        print(f"  ⌧ {feature}: 无法计算 (缺少HH或HV)")
                else:
                    print(f"  ⌧ {feature}: 不可用")
        
        print("=" * 40)

    def train_model(self, X, y, n_estimators=200):
        """训练随机森林模型"""
        print("\n训练随机森林模型...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1
        )
        
        self.rf_model.fit(X, y)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("特征重要性:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.rf_model

    def evaluate_model(self, X, y, folder):
        """评估模型性能"""
        print("\n评估模型性能...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        y_pred = self.rf_model.predict(X_test)
        
        print("\n分类报告:")
        report = classification_report(y_test, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True)
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
         # --- 生成字典格式的报告 ---
        report_dict = classification_report(y_test, y_pred, 
                                            target_names=self.class_names,
                                            output_dict=True)
        
        # --- 将报告字典转换为 pandas DataFrame 并保存为 CSV ---
        df_report = pd.DataFrame(report_dict).transpose()
        
        output_folder = "reports"
        os.makedirs(output_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_folder, f"classification_report_{timestamp}.csv")
        
        try:
            # index=True 会将类别名（如 precision, recall）作为一列保存
            df_report.to_csv(file_path, index=True, encoding='utf-8-sig')
            print(f"\n报告已成功保存至: {file_path}")
        except Exception as e:
            print(f"\n保存报告时出错: {e}")
            
        return report_dict
    def evaluate_with_cv(self, X, y):
        """使用交叉验证评估模型"""
        print("\n使用5折交叉验证评估模型性能...")
        
        # 2. 创建一个分类器实例（注意：每次交叉验证都会重新训练）
        # 如果 self.rf_model 已经训练过，需要重新创建一个新的实例
        model_for_cv = self.rf_model 
        
        # 3. 定义交叉验证策略
        # 对于分类问题，强烈推荐使用 StratifiedKFold
        # shuffle=True 可以在分割前打乱数据，增加随机性
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 4. 执行交叉验证
        # cross_val_score 会自动处理数据划分、模型训练和评估
        # 'scoring' 参数指定评估指标，例如 'accuracy', 'f1_macro', 'roc_auc' 等
        scores = cross_val_score(model_for_cv, X, y, cv=cv_strategy, scoring='accuracy')
        
        # 5. 打印结果
        print(f"每次验证的准确率: {scores}")
        print(f"平均准确率: {scores.mean():.4f}")
        print(f"准确率标准差: {scores.std():.4f}") # 标准差越小，模型性能越稳定
    
        return scores.mean(), scores.std()
    def get_cv_report(self, X, y, output_folder):
        """通过交叉验证获取完整的分类报告"""
        print("\n通过5折交叉验证生成分类报告...")
        
        model_for_cv = self.rf_model
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. 获取交叉验证的预测结果
        y_pred_cv = cross_val_predict(model_for_cv, X, y, cv=cv_strategy)
        
        # 2. 基于这些预测结果，生成报告和混淆矩阵
        # 这里的 y 是完整的真实标签
        print("\n交叉验证分类报告:")
        report = classification_report(y, y_pred_cv, target_names=self.class_names)
        print(report)
        
        print("\n交叉验证混淆矩阵:")
        cm = confusion_matrix(y, y_pred_cv)
        print(cm)
        
         
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- 导出分类报告 (Classification Report) ---
        # 2. 生成字典格式的报告，这是转换为表格的关键
        report_dict = classification_report(y, y_pred_cv, 
                                            target_names=self.class_names, 
                                            output_dict=True)
        
        # 3. 将字典转换为 pandas DataFrame，并使用 .transpose() 使其格式更易读
        df_report = pd.DataFrame(report_dict).transpose()
        
        # 4. 定义文件名并保存
        report_filename = f"cv_report_{timestamp}.csv"
        report_filepath = os.path.join(output_folder, report_filename)
        
        # 使用 encoding='utf-8-sig' 确保在 Excel 中打开中文不乱码
        df_report.to_csv(report_filepath, encoding='utf-8-sig')
        print(f"\n分类报告已成功保存至: {report_filepath}")

        # --- 导出混淆矩阵 (Confusion Matrix) ---
        # 5. 生成混淆矩阵
        cm = confusion_matrix(y, y_pred_cv)
        
        # 6. 将混淆矩阵（通常是numpy数组）转换为带标签的 DataFrame
        df_cm = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        
        # 7. 定义文件名并保存
        cm_filename = f"cv_confusion_matrix_{timestamp}.csv"
        cm_filepath = os.path.join(output_folder, cm_filename)
        df_cm.to_csv(cm_filepath, encoding='utf-8-sig')
        print(f"混淆矩阵已成功保存至: {cm_filepath}")
    
        return report, cm

    def calculate_chunk_size(self, tiff_path, target_memory_mb=1000):
        """
        根据影像尺寸和可用内存计算合适的chunk size
        """
        with rasterio.open(tiff_path) as src:
            height, width = src.height, src.width
            n_bands = src.count
            dtype = src.dtypes[0]
            
        # 计算每个像素的字节数
        dtype_bytes = np.dtype(dtype).itemsize
        
        # 计算整个影像的大小 (MB)
        total_size_mb = (height * width * n_bands * dtype_bytes) / (1024 * 1024)
        
        # 获取系统可用内存
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        print(f"影像信息:")
        print(f"  尺寸: {height} x {width}")
        print(f"  波段数: {n_bands}")
        print(f"  数据类型: {dtype}")
        print(f"  影像大小: {total_size_mb:.1f} MB")
        print(f"  系统可用内存: {available_memory_mb:.1f} MB")
        
        # 设置目标内存使用量
        target_memory = min(target_memory_mb, available_memory_mb * 0.5)  # 不超过可用内存的50%
        
        # 计算chunk size
        if total_size_mb <= target_memory:
            # 如果影像较小，不需要分块
            chunk_size = min(height, width)
            print(f"影像较小，不需要分块处理")
        else:
            # 计算每个chunk的像素数
            pixels_per_chunk = (target_memory * 1024 * 1024) / (n_bands * dtype_bytes)
            chunk_size = int(np.sqrt(pixels_per_chunk))
            chunk_size = min(chunk_size, min(height, width))
            
        # 确保chunk_size至少为64
        chunk_size = max(64, chunk_size)
        
        print(f"建议的chunk size: {chunk_size}")
        print(f"预计内存使用: {(chunk_size**2 * n_bands * dtype_bytes) / (1024*1024):.1f} MB per chunk")
        
        return chunk_size

    def classify_with_chunks(self, tiff_path, output_path=None, chunk_size=None, target_memory_mb=1000):
        """
        分块处理大型影像进行分类
        """
        print(f"\n开始分块分类影像: {tiff_path}")
        
        # 自动计算chunk size
        if chunk_size is None:
            chunk_size = self.calculate_chunk_size(tiff_path, target_memory_mb)
        
        with rasterio.open(tiff_path) as src:
            profile = src.profile
            height, width = src.height, src.width
            
            print(f"影像尺寸: {height} x {width}")
            print(f"使用chunk size: {chunk_size}")
            
            # 准备输出数组
            classification_result = np.zeros((height, width), dtype=np.uint8)
            
            # 计算chunk数量
            n_chunks_y = (height + chunk_size - 1) // chunk_size
            n_chunks_x = (width + chunk_size - 1) // chunk_size
            total_chunks = n_chunks_y * n_chunks_x
            
            print(f"总共需要处理 {total_chunks} 个chunks ({n_chunks_y}x{n_chunks_x})")
            
            # 创建进度条
            with tqdm(total=total_chunks, desc="分类进度") as pbar:
                for i in range(n_chunks_y):
                    for j in range(n_chunks_x):
                        # 计算当前chunk的边界
                        row_start = i * chunk_size
                        row_end = min((i + 1) * chunk_size, height)
                        col_start = j * chunk_size
                        col_end = min((j + 1) * chunk_size, width)
                        
                        # 创建窗口
                        window = Window(col_start, row_start, 
                                      col_end - col_start, row_end - row_start)
                        
                        # 读取数据
                        chunk_data = src.read(window=window)
                        chunk_height, chunk_width = chunk_data.shape[1], chunk_data.shape[2]
                        
                        # 处理当前chunk
                        chunk_result = self._classify_chunk(chunk_data)
                        
                        # 将结果写入输出数组
                        classification_result[row_start:row_end, col_start:col_end] = chunk_result
                        
                        pbar.update(1)
                        
                        # 清理内存
                        del chunk_data, chunk_result
                        gc.collect()
        
        # 保存结果
        if output_path:
            self._save_classification_result(classification_result, profile, output_path)
        
        # 统计结果
        self._print_classification_stats(classification_result)
        
        return classification_result, profile

    def _classify_chunk(self, chunk_data):
        """对单个chunk进行分类"""
        n_bands, height, width = chunk_data.shape
        
        # 创建特征字典
        bands_dict = {}
        for i, band_name in enumerate(self.tiff_band_names):
            if i < n_bands:
                bands_dict[band_name] = chunk_data[i]
        
        # 准备特征数据
        feature_arrays = []
        available_features = []
        
        for feature_name in self.feature_names:
            if feature_name in bands_dict:
                feature_arrays.append(bands_dict[feature_name])
                available_features.append(feature_name)
            else:
                # 尝试计算衍生特征
                if feature_name == 'HH_div_HV' and 'HH' in bands_dict and 'HV' in bands_dict:
                    calculated_band = np.divide(bands_dict['HH'], bands_dict['HV'], 
                                              out=np.zeros_like(bands_dict['HH']), 
                                              where=bands_dict['HV']!=0)
                    feature_arrays.append(calculated_band)
                    available_features.append(feature_name)
                elif feature_name == 'HH_minus_HV' and 'HH' in bands_dict and 'HV' in bands_dict:
                    calculated_band = bands_dict['HH'] - bands_dict['HV']
                    feature_arrays.append(calculated_band)
                    available_features.append(feature_name)
        
        if not feature_arrays:
            return np.zeros((height, width), dtype=np.uint8)
        
        # 堆叠特征
        features = np.stack(feature_arrays, axis=0)
        features_2d = features.reshape(features.shape[0], -1).T
        
        # 创建有效像素掩膜
        if 'SIC' in bands_dict:
            sic_data = bands_dict['SIC']
            valid_mask = (sic_data > 0) & (sic_data <= 100)
        else:
            sample_band = feature_arrays[0]
            valid_mask = ~(np.isnan(sample_band) | np.isinf(sample_band) | (sample_band == 0))
        
        # --- 新增代码开始 ---
        # 专门针对 Sentinel-1 HH 波段的规则
        # 如果 HH 波段存在，则在现有掩膜的基础上，增加 HH 波段值不为0的条件
        if 'HH' in bands_dict:
            # 创建一个仅针对 HH 波段的掩膜
            hh_mask = (bands_dict['HH'] != 0)
            
            # 使用逻辑与(&)操作，将 HH 掩膜与之前的掩膜合并
            # 像素必须同时满足两个条件才被视为有效
            valid_mask = valid_mask & hh_mask
        # --- 新增代码结束 ---
        
        valid_pixels = valid_mask.flatten()
        
        # 处理无效值
        features_2d = np.nan_to_num(features_2d, nan=0, posinf=0, neginf=0)
        
        # 预测
        predictions_full = np.zeros(height * width, dtype=np.uint8)
        if np.sum(valid_pixels) > 0:
            valid_features = features_2d[valid_pixels]
            predictions_valid = self.rf_model.predict(valid_features)
            predictions_full[valid_pixels] = predictions_valid
        
        # 重塑为2D并设置无效区域为0
        chunk_result = predictions_full.reshape(height, width)
        chunk_result[~valid_mask] = 0
        
        return chunk_result

    def _save_classification_result(self, classification_result, profile, output_path):
        """保存分类结果"""
        output_profile = profile.copy()
        output_profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'lzw',
            'nodata': 0
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(classification_result, 1)
        
        print(f"分类结果已保存到: {output_path}")

    def _print_classification_stats(self, classification_result):
        """打印分类统计信息"""
        unique, counts = np.unique(classification_result[classification_result > 0], 
                                 return_counts=True)
        print("\n分类结果统计:")
        total_classified = np.sum(counts) if len(counts) > 0 else 0
        
        for class_id, count in zip(unique, counts):
            if class_id > 0:
                class_name = self.class_names[class_id-1]
                percentage = count / total_classified * 100 if total_classified > 0 else 0
                print(f"  {class_name} (class {class_id}): {count:,} 像素 ({percentage:.2f}%)")

    def visualize_classification(self, classification_result, profile, 
                               original_tiff_path=None, save_path=None):
        """可视化分类结果"""
        print("\n生成分类结果可视化...")
        
        colors = ['black'] + self.class_colors  # 黑色用于无效区域
        cmap = ListedColormap(colors)
        
        if original_tiff_path:
            # 如果提供了原始影像，创建对比图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            with rasterio.open(original_tiff_path) as src:
                # 显示HH波段
                hh_data = src.read(1)  # 假设第一个波段是HH
                im1 = axes[0, 0].imshow(hh_data, cmap='gray')
                axes[0, 0].set_title('HH波段')
                axes[0, 0].axis('off')
                plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
                
                # 显示HV波段
                if src.count > 1:
                    hv_data = src.read(2)  # 假设第二个波段是HV
                    im2 = axes[0, 1].imshow(hv_data, cmap='gray')
                    axes[0, 1].set_title('HV波段')
                    axes[0, 1].axis('off')
                    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
                
            # 显示分类结果
            im3 = axes[1, 0].imshow(classification_result, cmap=cmap, vmin=0, vmax=3)
            axes[1, 0].set_title('海冰分类结果')
            axes[1, 0].axis('off')
            
            # 创建图例
            legend_elements = [Patch(facecolor='black', label='无效区域')]
            for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
                legend_elements.append(Patch(facecolor=color, label=f'{class_name} (类别 {i+1})'))
            
            axes[1, 0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            # 统计图
            unique, counts = np.unique(classification_result[classification_result > 0], return_counts=True)
            if len(unique) > 0:
                class_labels = [self.class_names[int(cls)-1] for cls in unique]
                colors_for_bars = [self.class_colors[int(cls)-1] for cls in unique]
                
                bars = axes[1, 1].bar(class_labels, counts, color=colors_for_bars, alpha=0.7)
                axes[1, 1].set_title('分类结果统计')
                axes[1, 1].set_ylabel('像素数量')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{count:,}', ha='center', va='bottom')
        else:
            # 只显示分类结果
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            im = axes[0].imshow(classification_result, cmap=cmap, vmin=0, vmax=3)
            axes[0].set_title('海冰分类结果')
            axes[0].axis('off')
            
            legend_elements = [Patch(facecolor='black', label='无效区域')]
            for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
                legend_elements.append(Patch(facecolor=color, label=f'{class_name} (类别 {i+1})'))
            
            axes[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            # 统计图
            unique, counts = np.unique(classification_result[classification_result > 0], return_counts=True)
            if len(unique) > 0:
                class_labels = [self.class_names[int(cls)-1] for cls in unique]
                colors_for_bars = [self.class_colors[int(cls)-1] for cls in unique]
                
                bars = axes[1].bar(class_labels, counts, color=colors_for_bars, alpha=0.7)
                axes[1].set_title('分类结果统计')
                axes[1].set_ylabel('像素数量')
                axes[1].tick_params(axis='x', rotation=45)
                
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.close()

    def batch_classify_folder(self, input_folder, output_folder, target_memory_mb=1500):
        """
        批处理文件夹中的所有TIFF文件
        """
        print(f"\n=== 开始批处理分类 ===")
        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_folder}")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 查找所有.tif文件
        tiff_pattern = os.path.join(input_folder, "*.tif")
        tiff_files = glob.glob(tiff_pattern)
        
        if not tiff_files:
            print(f"在 {input_folder} 中未找到任何.tif文件")
            return
        
        print(f"找到 {len(tiff_files)} 个TIFF文件")
        
        # 创建处理日志
        log_file = os.path.join(output_folder, "batch_processing_log.txt")
        
        # 统计变量
        processed_files = 0
        failed_files = 0
        processing_times = []
        
        # 开始批处理
        with open(log_file, 'w', encoding='utf-8') as log:
            log.write(f"批处理开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"总文件数: {len(tiff_files)}\n")
            log.write(f"使用特征: {self.feature_names}\n")
            log.write("-" * 50 + "\n")
            
            for i, tiff_path in enumerate(tiff_files, 1):
                print(f"\n处理文件 {i}/{len(tiff_files)}: {os.path.basename(tiff_path)}")
                start_time = time.time()
                
                try:
                    # 生成输出文件名
                    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
                    output_tiff = os.path.join(output_folder, f"{base_name}_classified.tif")
                    visualization_path = os.path.join(output_folder, f"{base_name}_visualization.png")
                    
                    # 检查文件是否已经处理过
                    if os.path.exists(output_tiff):
                        print(f"文件已存在，跳过: {output_tiff}")
                        log.write(f"跳过 (已存在): {os.path.basename(tiff_path)}\n")
                        continue
                    
                    # 检查TIFF文件波段信息
                    self.check_tiff_bands(tiff_path)
                    
                    # 进行分类
                    classification_result, profile = self.classify_with_chunks(
                        tiff_path=tiff_path,
                        output_path=output_tiff,
                        target_memory_mb=target_memory_mb
                    )
                    
                    # 生成可视化（可选）
                    try:
                        self.visualize_classification(
                            classification_result, profile,
                            original_tiff_path=tiff_path,
                            save_path=visualization_path
                        )
                        plt.close('all')  # 关闭所有图形以释放内存
                    except Exception as vis_error:
                        print(f"可视化生成失败: {vis_error}")
                        log.write(f"可视化失败: {os.path.basename(tiff_path)} - {str(vis_error)}\n")
                    
                    # 计算处理时间
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    processed_files += 1
                    
                    # 记录成功处理的文件
                    log.write(f"成功: {os.path.basename(tiff_path)} - 用时: {processing_time:.2f}秒\n")
                    print(f"处理完成，用时: {processing_time:.2f}秒")
                    
                    # 清理内存
                    del classification_result
                    gc.collect()
                    
                except Exception as e:
                    failed_files += 1
                    error_msg = f"处理失败: {os.path.basename(tiff_path)} - 错误: {str(e)}"
                    print(error_msg)
                    log.write(f"失败: {os.path.basename(tiff_path)} - {str(e)}\n")
                    continue
            
            # 写入总结信息
            total_time = sum(processing_times)
            avg_time = np.mean(processing_times) if processing_times else 0
            
            log.write("-" * 50 + "\n")
            log.write(f"批处理完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"成功处理: {processed_files} 个文件\n")
            log.write(f"处理失败: {failed_files} 个文件\n")
            log.write(f"总用时: {total_time:.2f} 秒\n")
            log.write(f"平均用时: {avg_time:.2f} 秒/文件\n")
        
        # 打印总结
        print(f"\n=== 批处理完成 ===")
        print(f"成功处理: {processed_files} 个文件")
        print(f"处理失败: {failed_files} 个文件")
        print(f"总用时: {total_time:.2f} 秒")
        print(f"平均用时: {avg_time:.2f} 秒/文件")
        print(f"详细日志已保存到: {log_file}")


def run_batch_classification_workflow():
    """
    运行批处理分类工作流程
    """
    print("=== Sentinel-1 海冰批处理分类工作流程 ===")
    
    # 1. 定义不同的特征组合用于测试
    feature_combinations = {
        'basic': ['HH', 'HV', 'ANGLE'],
        'with_ratios': ['HH', 'HV', 'HH_div_HV', 'HH_minus_HV', 'ANGLE'],
        'normalized': ['HH_norm', 'HV_norm', 'ANGLE'],
        'comprehensive': ['HH', 'HV', 'HH_div_HV', 'ANGLE', 'sum_div_diff'],
        'minimal': ['HH', 'HV']  # 最小特征集
    }

    # 2. 选择特征组合或自定义特征
    # selected_combination = 'comprehensive'  # 修改这里选择不同的特征组合
    # selected_features = feature_combinations[selected_combination]
    
    # 或者直接使用自定义特征
    selected_features = ['HH', 'HV', 'HH_div_HV', 'sum_div_diff']
    
    print(f"使用特征: {selected_features}")
    
    # 3. 初始化分类器
    classifier = Sentinel1IceClassifier(selected_features=selected_features)
    
    # 4. 定义训练样本CSV文件
    csv_files = [
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230329T155110_20230329T155214_047861_05C03E_8378.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230412T153445_20230412T153549_048065_05C71E_CE44.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230426T151920_20230426T152020_048269_05CDF5_BEA4.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230429T122549_20230429T122643_048311_05CF66_60DE.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230703T123357_20230703T123501_049259_05EC59_E052.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230503T151005_20230503T151109_048371_05D160_BE0E (1).csv"
    ]
    input_folder = r"F:\NWP\S1_processed_for_classification\2023"
    output_folder = r"F:\NWP\Classification Result\2023"
    
    # 5. 加载并准备训练数据
    print("加载训练样本...")
    samples = classifier.load_and_merge_samples(csv_files)
    X, y = classifier.prepare_training_data(samples, filter_features=True)
    
    # 6. 训练模型
    print("训练模型...")
    model = classifier.train_model(X, y, n_estimators=200)
    
    # 7. 评估模型
    evaluation = classifier.evaluate_model(X, y,output_folder)
    evaluation_report, confusion_mat = classifier.get_cv_report(X, y, output_folder)
    
    # 8. 显示特征匹配信息
    classifier.debug_feature_info()
    
    # 9. 定义输入和输出文件夹

    # 10. 执行批处理分类
    print(f"\n开始批处理分类...")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    
    classifier.batch_classify_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        target_memory_mb=1500  # 目标内存使用量
    )
    
    print("\n批处理分类工作流程完成！")


def run_single_file_classification():
    """
    单文件分类示例（保留原有功能）
    """
    print("=== 单文件分类示例 ===")
    
    # 使用自定义特征
    selected_features = ['HH', 'HV', 'HH_div_HV', 'sum_div_diff', 'ANGLE']
    classifier = Sentinel1IceClassifier(selected_features=selected_features)
    
    # CSV训练文件
    csv_files = [
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230329T155110_20230329T155214_047861_05C03E_8378.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230412T153445_20230412T153549_048065_05C71E_CE44.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230426T151920_20230426T152020_048269_05CDF5_BEA4.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230429T122549_20230429T122643_048311_05CF66_60DE.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230703T123357_20230703T123501_049259_05EC59_E052.csv",
        r"J:\我的云端硬盘\Project_Yujie\RF Samples1\samplePoint_20230503T151005_20230503T151109_048371_05D160_BE0E (1).csv"
    ]
    
    # 训练
    samples = classifier.load_and_merge_samples(csv_files)
    X, y = classifier.prepare_training_data(samples, filter_features=True)
    model = classifier.train_model(X, y, n_estimators=200)
    evaluation = classifier.evaluate_model(X, y)
    
    # 单文件分类
    tiff_path = r"F:\NWP\S1_processed_for_classification\S1A_EW_GRDM_1SDH_20230412T121730_20230412T121835_048063_05C70C_40A9_EW_HH_HV_angle_int16x100_87caa6ee_1_processed.tif"
    output_dir = r"F:\NWP\Classification Result"
    
    output_path = os.path.join(output_dir, "single_classification_result.tif")
    visualization_path = os.path.join(output_dir, "single_classification_visualization.png")
    
    # 检查特征匹配
    classifier.debug_feature_info()
    classifier.check_tiff_bands(tiff_path)
    
    # 分类
    classification_result, profile = classifier.classify_with_chunks(
        tiff_path=tiff_path,
        output_path=output_path,
        target_memory_mb=1500
    )
    
    # 可视化
    classifier.visualize_classification(
        classification_result, profile,
        original_tiff_path=tiff_path,
        save_path=visualization_path
    )
    
    print("单文件分类完成！")


# if __name__ == "__main__":
#     # 选择运行模式
#     print("选择运行模式:")
#     print("1. 批处理分类 (推荐)")
#     print("2. 单文件分类")
    
#     choice = input("请输入选择 (1 或 2): ").strip()
    
#     if choice == "1":
#         run_batch_classification_workflow()
#     elif choice == "2":
#         run_single_file_classification()
#     else:
#         print("无效选择，默认运行批处理分类")
#         run_batch_classification_workflow()
if __name__ == "__main__":
    # --- 在这里直接设定运行模式 ---
    # "batch" 代表批处理模式
    # "single" 代表单文件模式
    run_mode = "batch"  # <--- 修改这里来切换模式

    if run_mode == "batch":
        print("已设定为：批处理分类模式")
        run_batch_classification_workflow()
    elif run_mode == "single":
        print("已设定为：单文件分类模式")
        run_single_file_classification()
    else:
        print(f"错误：无效的运行模式 '{run_mode}'。请检查 'run_mode' 变量。")
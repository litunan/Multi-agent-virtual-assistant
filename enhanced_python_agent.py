#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版Python Agent - 高级代码执行和数据可视化
新增功能：
1. 高级数据可视化（多种图表类型）
2. 数据科学分析工具
3. 机器学习模型支持
4. 代码执行环境管理
5. 结果格式化输出
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv 
load_dotenv(override=True)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from config.load_key import load_key

# 初始化模型 - 使用阿里云百炼 API
model = ChatOpenAI(
    api_key=load_key("aliyun-bailian"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# 设置图像保存路径 - 使用相对路径，避免硬编码用户目录
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "images")
DESKTOP_DIR = os.path.expanduser("~/Desktop")
os.makedirs(IMAGES_DIR, exist_ok=True)

# =============================================================================
# 基础Python执行工具
# =============================================================================

class PythonCodeInput(BaseModel):
    py_code: str = Field(description="要执行的Python代码")
    description: str = Field(default="", description="代码描述（可选）")

@tool(args_schema=PythonCodeInput)
def enhanced_python_exec(py_code: str, description: str = "") -> str:
    """
    增强版Python代码执行器，支持复杂数据分析和计算
    """
    try:
        # 准备安全的执行环境
        global_vars = {
            'np': np,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'json': json,
            'os': os,
            'sys': sys,
            '__builtins__': __builtins__
        }
        
        # 添加常用科学计算库
        try:
            import scipy.stats as stats
            import sklearn
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            global_vars.update({
                'stats': stats,
                'sklearn': sklearn,
                'train_test_split': train_test_split,
                'accuracy_score': accuracy_score,
                'classification_report': classification_report
            })
        except ImportError:
            pass
        
        # 执行代码
        local_vars = {}
        
        try:
            # 首先尝试作为表达式执行
            result = eval(py_code, global_vars, local_vars)
            return f"✅ 执行成功{'（' + description + '）' if description else ''}:\n结果: {result}"
        except SyntaxError:
            # 如果不是表达式，作为语句执行
            exec(py_code, global_vars, local_vars)
            
            # 检查是否有新变量
            if local_vars:
                results = []
                for var_name, var_value in local_vars.items():
                    if not var_name.startswith('_'):
                        results.append(f"{var_name} = {repr(var_value)}")
                
                if results:
                    return f"✅ 执行成功{'（' + description + '）' if description else ''}:\n" + "\n".join(results)
            
            return f"✅ 代码执行完成{'（' + description + '）' if description else ''}"
            
    except Exception as e:
        return f"❌ 执行失败: {str(e)}"

# =============================================================================
# 高级可视化工具
# =============================================================================

class AdvancedPlotSchema(BaseModel):
    plot_type: str = Field(description="图表类型：line, bar, scatter, histogram, heatmap, boxplot, violin, pair")
    data_code: str = Field(description="生成数据的Python代码")
    plot_config: str = Field(default="{}", description="图表配置JSON字符串")
    title: str = Field(default="", description="图表标题")
    filename: str = Field(default="plot", description="保存的文件名（不含扩展名）")

@tool(args_schema=AdvancedPlotSchema)
def advanced_visualization(plot_type: str, data_code: str, plot_config: str = "{}", 
                          title: str = "", filename: str = "plot") -> str:
    """
    高级数据可视化工具，支持多种图表类型和自定义配置
    """
    try:
        # 设置图像后端
        current_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        
        # 准备执行环境
        global_vars = {
            'np': np, 'pd': pd, 'plt': plt, 'sns': sns,
            'json': json
        }
        local_vars = {}
        
        # 执行数据生成代码
        exec(data_code, global_vars, local_vars)
        
        # 解析图表配置
        config = json.loads(plot_config) if plot_config else {}
        
        # 创建图表
        fig_size = config.get('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=fig_size)
        
        # 设置样式
        if config.get('style'):
            plt.style.use(config['style'])
        else:
            sns.set_style("whitegrid")
        
        # 根据图表类型绘制
        if plot_type == "line":
            x_data = local_vars.get('x', range(len(local_vars.get('y', []))))
            y_data = local_vars.get('y', [])
            ax.plot(x_data, y_data, **config.get('plot_params', {}))
            
        elif plot_type == "bar":
            x_data = local_vars.get('x', range(len(local_vars.get('y', []))))
            y_data = local_vars.get('y', [])
            ax.bar(x_data, y_data, **config.get('plot_params', {}))
            
        elif plot_type == "scatter":
            x_data = local_vars.get('x', [])
            y_data = local_vars.get('y', [])
            ax.scatter(x_data, y_data, **config.get('plot_params', {}))
            
        elif plot_type == "histogram":
            data = local_vars.get('data', [])
            ax.hist(data, **config.get('plot_params', {'bins': 30}))
            
        elif plot_type == "heatmap":
            data = local_vars.get('data', np.random.rand(10, 10))
            sns.heatmap(data, ax=ax, **config.get('plot_params', {'annot': True}))
            
        elif plot_type == "boxplot":
            data = local_vars.get('data', [])
            ax.boxplot(data, **config.get('plot_params', {}))
            
        elif plot_type == "violin":
            data = local_vars.get('data', [])
            ax.violinplot(data, **config.get('plot_params', {}))
            
        elif plot_type == "pair":
            # 需要DataFrame数据
            df = local_vars.get('df', pd.DataFrame())
            if not df.empty:
                sns.pairplot(df, **config.get('plot_params', {}))
                fig = plt.gcf()
            else:
                return "❌ 散点图矩阵需要DataFrame数据"
        else:
            return f"❌ 不支持的图表类型: {plot_type}"
        
        # 设置标题和标签
        if title:
            ax.set_title(title, fontsize=config.get('title_fontsize', 14))
        
        if config.get('xlabel'):
            ax.set_xlabel(config['xlabel'])
        if config.get('ylabel'):
            ax.set_ylabel(config['ylabel'])
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        image_path = os.path.join(IMAGES_DIR, f"{filename}.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 恢复后端
        matplotlib.use(current_backend)
        
        relative_path = f"images/{filename}.png"
        return f"✅ {plot_type}图表已生成: {relative_path}\n![{title or plot_type}]({relative_path})"
        
    except Exception as e:
        return f"❌ 可视化失败: {str(e)}"

# =============================================================================
# 数据科学分析工具
# =============================================================================

class DataAnalysisSchema(BaseModel):
    data_source: str = Field(description="数据源代码或数据变量名")
    analysis_type: str = Field(description="分析类型：describe, correlation, distribution, outliers, missing")
    output_format: str = Field(default="text", description="输出格式：text, json, html")

@tool(args_schema=DataAnalysisSchema)
def data_analysis_tool(data_source: str, analysis_type: str, output_format: str = "text") -> str:
    """
    数据科学分析工具，提供统计分析和数据探索功能
    """
    try:
        # 准备执行环境
        global_vars = {
            'np': np, 'pd': pd, 'json': json
        }
        local_vars = {}
        
        # 获取数据
        if data_source.startswith('pd.') or 'DataFrame' in data_source or 'read_' in data_source:
            # 数据源代码
            exec(f"data = {data_source}", global_vars, local_vars)
            data = local_vars['data']
        else:
            # 假设是变量名
            data = eval(data_source, global_vars)
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        results = {}
        
        if analysis_type == "describe":
            # 描述性统计
            results['basic_stats'] = data.describe().to_dict()
            results['data_info'] = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum()
            }
            
        elif analysis_type == "correlation":
            # 相关性分析
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                results['correlation_matrix'] = numeric_data.corr().to_dict()
                # 找出高相关性对
                corr_matrix = numeric_data.corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr.append({
                                'var1': corr_matrix.columns[i],
                                'var2': corr_matrix.columns[j],
                                'correlation': corr_val
                            })
                results['high_correlations'] = high_corr
            else:
                results['error'] = "没有数值型变量进行相关性分析"
                
        elif analysis_type == "distribution":
            # 分布分析
            numeric_data = data.select_dtypes(include=[np.number])
            results['distributions'] = {}
            for col in numeric_data.columns:
                results['distributions'][col] = {
                    'mean': float(numeric_data[col].mean()),
                    'median': float(numeric_data[col].median()),
                    'std': float(numeric_data[col].std()),
                    'skewness': float(numeric_data[col].skew()),
                    'kurtosis': float(numeric_data[col].kurtosis())
                }
                
        elif analysis_type == "outliers":
            # 异常值检测
            numeric_data = data.select_dtypes(include=[np.number])
            results['outliers'] = {}
            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = numeric_data[(numeric_data[col] < lower_bound) | 
                                       (numeric_data[col] > upper_bound)][col]
                results['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(numeric_data) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
                
        elif analysis_type == "missing":
            # 缺失值分析
            missing_count = data.isnull().sum()
            missing_percent = (missing_count / len(data)) * 100
            results['missing_values'] = {}
            for col in data.columns:
                results['missing_values'][col] = {
                    'count': int(missing_count[col]),
                    'percentage': float(missing_percent[col])
                }
        else:
            return f"❌ 不支持的分析类型: {analysis_type}"
        
        # 格式化输出
        if output_format == "json":
            return json.dumps(results, indent=2, ensure_ascii=False)
        elif output_format == "html":
            # 简单的HTML格式化
            html_content = "<div class='data-analysis'>"
            for key, value in results.items():
                html_content += f"<h3>{key}</h3><pre>{json.dumps(value, indent=2)}</pre>"
            html_content += "</div>"
            return html_content
        else:
            # 文本格式
            output_lines = [f"📊 数据分析结果 - {analysis_type}"]
            output_lines.append("=" * 50)
            
            for key, value in results.items():
                output_lines.append(f"\n📋 {key}:")
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        output_lines.append(f"  • {sub_key}: {sub_value}")
                else:
                    output_lines.append(f"  {value}")
            
            return "\n".join(output_lines)
            
    except Exception as e:
        return f"❌ 数据分析失败: {str(e)}"

# =============================================================================
# 机器学习工具
# =============================================================================

class MLModelSchema(BaseModel):
    model_type: str = Field(description="模型类型：linear_regression, logistic_regression, random_forest, svm, kmeans")
    data_prep_code: str = Field(description="数据准备代码（定义X, y变量）")
    model_params: str = Field(default="{}", description="模型参数JSON字符串")
    task_type: str = Field(default="classification", description="任务类型：classification, regression, clustering")

@tool(args_schema=MLModelSchema)
def ml_modeling_tool(model_type: str, data_prep_code: str, model_params: str = "{}", 
                     task_type: str = "classification") -> str:
    """
    机器学习建模工具，支持常见的ML算法
    """
    try:
        # 导入必要的库
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.cluster import KMeans
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
        from sklearn.preprocessing import StandardScaler
        
        # 准备执行环境
        global_vars = {
            'np': np, 'pd': pd, 'json': json,
            'train_test_split': train_test_split,
            'StandardScaler': StandardScaler
        }
        local_vars = {}
        
        # 执行数据准备代码
        exec(data_prep_code, global_vars, local_vars)
        
        # 获取数据
        X = local_vars.get('X')
        y = local_vars.get('y', None)
        
        if X is None:
            return "❌ 未找到特征数据X，请在数据准备代码中定义X变量"
        
        # 解析模型参数
        params = json.loads(model_params) if model_params else {}
        
        results = []
        results.append(f"🤖 机器学习建模 - {model_type}")
        results.append("=" * 50)
        
        # 创建模型
        if model_type == "linear_regression":
            model = LinearRegression(**params)
            task_type = "regression"
        elif model_type == "logistic_regression":
            model = LogisticRegression(**params)
            task_type = "classification"
        elif model_type == "random_forest":
            if task_type == "regression":
                model = RandomForestRegressor(**params)
            else:
                model = RandomForestClassifier(**params)
        elif model_type == "svm":
            if task_type == "regression":
                model = SVR(**params)
            else:
                model = SVC(**params)
        elif model_type == "kmeans":
            model = KMeans(**params)
            task_type = "clustering"
        else:
            return f"❌ 不支持的模型类型: {model_type}"
        
        # 训练和评估
        if task_type == "clustering":
            # 聚类任务
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = np.array(X)
            
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
            
            # 训练模型
            labels = model.fit_predict(X_scaled)
            
            results.append(f"📊 聚类结果:")
            results.append(f"  • 样本数量: {len(X_array)}")
            results.append(f"  • 聚类数量: {len(np.unique(labels))}")
            results.append(f"  • 各类别样本数: {dict(zip(*np.unique(labels, return_counts=True)))}")
            
            if hasattr(model, 'inertia_'):
                results.append(f"  • 聚类惯性: {model.inertia_:.4f}")
            
        else:
            # 监督学习任务
            if y is None:
                return "❌ 监督学习任务需要目标变量y，请在数据准备代码中定义y变量"
            
            # 数据分割
            test_size = params.get('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            results.append(f"📊 模型评估结果:")
            results.append(f"  • 训练集大小: {len(X_train)}")
            results.append(f"  • 测试集大小: {len(X_test)}")
            
            if task_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                results.append(f"  • 准确率: {accuracy:.4f}")
                
                # 交叉验证
                cv_scores = cross_val_score(model, X, y, cv=5)
                results.append(f"  • 交叉验证均值: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                
            elif task_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                results.append(f"  • 均方误差 (MSE): {mse:.4f}")
                results.append(f"  • 均方根误差 (RMSE): {rmse:.4f}")
                
                # R²分数
                score = model.score(X_test, y_test)
                results.append(f"  • R² 分数: {score:.4f}")
            
            # 特征重要性（如果可用）
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                if isinstance(X, pd.DataFrame):
                    feature_names = X.columns.tolist()
                else:
                    feature_names = [f"feature_{i}" for i in range(len(importance))]
                
                results.append(f"\n🎯 特征重要性 (Top 5):")
                importance_pairs = list(zip(feature_names, importance))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                for feature, imp in importance_pairs[:5]:
                    results.append(f"  • {feature}: {imp:.4f}")
        
        return "\n".join(results)
        
    except ImportError as e:
        return f"❌ 缺少必要的库: {str(e)}。请安装scikit-learn"
    except Exception as e:
        return f"❌ 建模失败: {str(e)}"

# =============================================================================
# 代码环境管理
# =============================================================================

class DesktopFigCodeInput(BaseModel):
    py_code: str = Field(description="要执行的 Python 绘图代码，必须使用 matplotlib/seaborn 创建图像")
    file_name: str = Field(description="保存到桌面的图像文件名（包含.png扩展名）")

@tool(args_schema=DesktopFigCodeInput)
def save_chart_to_desktop(py_code: str, file_name: str) -> str:
    """
    执行Python绘图代码并直接保存图表到桌面
    
    注意：
    1. 代码必须创建一个matplotlib图像对象并赋值给变量fig
    2. 使用fig = plt.figure()或fig, ax = plt.subplots()
    3. 不要使用plt.show()
    4. 图表将直接保存到桌面
    """
    try:
        # 验证文件名
        if not file_name.endswith('.png'):
            file_name += '.png'
        
        # 验证文件名安全性
        if '..' in file_name or '/' in file_name or '\\' in file_name:
            return f"❌ 文件名不安全: {file_name}"
        
        current_backend = matplotlib.get_backend()
        matplotlib.use('Agg')

        # 准备执行环境
        local_vars = {
            "plt": plt, 
            "pd": pd, 
            "sns": sns,
            "np": np
        }
        
        # 添加全局变量到执行环境
        global_vars = globals().copy()
        global_vars.update(local_vars)
        
        # 执行绘图代码
        exec(py_code, global_vars, local_vars)
        
        # 查找图像对象
        fig = None
        for var_name, var_value in local_vars.items():
            if hasattr(var_value, 'savefig'):  # 检查是否是matplotlib figure对象
                fig = var_value
                break
        
        if fig is None:
            # 尝试从全局变量中获取
            for var_name, var_value in global_vars.items():
                if hasattr(var_value, 'savefig'):
                    fig = var_value
                    break
        
        if fig is None:
            return "❌ 未找到图像对象。请确保代码中创建了matplotlib图像对象并赋值给变量（如fig = plt.figure()）"
        
        # 保存到桌面
        desktop_path = os.path.join(DESKTOP_DIR, file_name)
        fig.savefig(desktop_path, dpi=300, bbox_inches='tight')
        
        return f"✅ 图表已成功保存到桌面: {file_name}"
        
    except Exception as e:
        return f"❌ 图表生成失败: {str(e)}"
    finally:
        plt.close('all')
        matplotlib.use(current_backend)

@tool
def get_python_environment_info() -> str:
    """获取Python执行环境信息"""
    try:
        import platform
        
        info_lines = []
        info_lines.append("🐍 Python执行环境信息")
        info_lines.append("=" * 40)
        info_lines.append(f"📋 Python版本: {platform.python_version()}")
        info_lines.append(f"💻 系统平台: {platform.system()} {platform.release()}")
        info_lines.append(f"🏗️ 架构: {platform.machine()}")
        
        # 检查可用的库
        info_lines.append(f"\n📚 可用库:")
        libraries = {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'matplotlib': matplotlib.__version__,
            'seaborn': sns.__version__
        }
        
        try:
            import sklearn
            libraries['scikit-learn'] = sklearn.__version__
        except ImportError:
            pass
        
        try:
            import scipy
            libraries['scipy'] = scipy.__version__
        except ImportError:
            pass
        
        for lib, version in libraries.items():
            info_lines.append(f"  • {lib}: {version}")
        
        # 内存使用情况
        import psutil
        memory = psutil.virtual_memory()
        info_lines.append(f"\n💾 内存使用:")
        info_lines.append(f"  • 总内存: {memory.total / (1024**3):.2f} GB")
        info_lines.append(f"  • 可用内存: {memory.available / (1024**3):.2f} GB")
        info_lines.append(f"  • 使用率: {memory.percent}%")
        
        return "\n".join(info_lines)
        
    except Exception as e:
        return f"❌ 获取环境信息失败: {str(e)}"

# =============================================================================
# Agent创建
# =============================================================================

ENHANCED_PYTHON_AGENT_PROMPT = """
你是一个专业的Python数据科学专家，具备强大的代码执行、数据分析和机器学习能力。

🎯 **核心能力**:
- 高级Python代码执行和调试
- 专业数据科学分析和统计
- 多样化数据可视化创建
- 机器学习模型构建和评估
- 代码性能优化和环境管理

📊 **数据可视化专长**:
- 支持多种图表类型（线图、柱图、散点图、热力图等）
- 自定义图表样式和配置
- 高质量图像输出和保存
- 交互式数据探索

🤖 **机器学习支持**:
- 常见算法（回归、分类、聚类）
- 模型训练、验证和评估
- 特征工程和数据预处理
- 交叉验证和性能分析

💡 **使用建议**:
- 提供清晰的数据分析需求
- 指定所需的可视化类型和样式
- 描述机器学习任务的目标
- 说明数据格式和预期结果

🔧 **可用工具**:
- enhanced_python_exec: 增强版Python代码执行
- advanced_visualization: 高级数据可视化
- data_analysis_tool: 数据科学分析工具
- ml_modeling_tool: 机器学习建模工具
- get_python_environment_info: 环境信息查询
- save_chart_to_desktop: 直接保存图表到桌面

请描述您的Python开发或数据分析需求，我将为您提供专业的解决方案！
"""

# 创建工具列表
enhanced_python_tools = [
    enhanced_python_exec,
    advanced_visualization, 
    data_analysis_tool,
    ml_modeling_tool,
    get_python_environment_info,
    save_chart_to_desktop
]

# 创建增强版Python Agent
enhanced_python_agent = create_react_agent(
    model=model,
    tools=enhanced_python_tools,
    prompt=ENHANCED_PYTHON_AGENT_PROMPT,
    name="enhanced_python_agent"
)

if __name__ == "__main__":
    print("🐍 增强版Python Agent 已启动")
    print("新增功能:")
    print("- 📊 高级数据可视化")
    print("- 🔍 数据科学分析工具") 
    print("- 🤖 机器学习建模")
    print("- 💾 环境管理")
    print("- 🎨 自定义图表样式")
    print("- 📈 统计分析报告")
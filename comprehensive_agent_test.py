#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面的Agent功能测试脚本
测试每个Agent的核心功能特性
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "="*70)
    print(f"🔍 {title}")
    print("="*70)

def print_test_result(test_name: str, result: str, status="✅"):
    """打印测试结果"""
    print(f"\n{status} {test_name}")
    print("-" * 50)
    print(result[:800] + "..." if len(result) > 800 else result)
    print("-" * 50)

async def comprehensive_rag_test():
    """全面测试Enhanced RAG Agent"""
    print_separator("Enhanced RAG Agent 深度功能测试")
    
    try:
        from enhanced_data_agent import enhanced_system_query
        
        tests = [
            ("电信客户流失预测模型", "客户流失预测模型有哪些关键特征？如何进行特征工程？"),
            ("数据科学建模方法", "电信行业中常用的机器学习算法有哪些？各自的优缺点是什么？"),
            ("业务术语解释", "什么是ARPU？在电信行业中如何计算和应用？"),
            ("上下文记忆测试", "基于前面讨论的ARPU，请问如何用它来评估客户价值？")
        ]
        
        for test_name, query in tests:
            result = await enhanced_system_query(query)
            print_test_result(test_name, result)
            
        return True
    except Exception as e:
        print_test_result("RAG Agent 深度测试", f"❌ 错误: {str(e)}", "❌")
        return False

async def comprehensive_python_test():
    """全面测试Enhanced Python Agent"""
    print_separator("Enhanced Python Agent 深度功能测试")
    
    try:
        from enhanced_data_agent import enhanced_system_query
        
        tests = [
            ("数据可视化", "创建一个包含100个随机数的数据集，绘制直方图和统计分析"),
            ("机器学习", "使用sklearn创建一个简单的线性回归模型，并进行预测"),
            ("数据处理", "创建一个pandas DataFrame，包含姓名、年龄、收入三列，进行基础统计分析"),
            ("高级图表", "创建一个热力图显示相关性矩阵")
        ]
        
        for test_name, query in tests:
            result = await enhanced_system_query(query)
            print_test_result(test_name, result)
            
        return True
    except Exception as e:
        print_test_result("Python Agent 深度测试", f"❌ 错误: {str(e)}", "❌")
        return False

async def comprehensive_amap_test():
    """全面测试Enhanced AMAP Agent"""
    print_separator("Enhanced AMAP Agent 深度功能测试")
    
    try:
        from enhanced_data_agent import enhanced_system_query
        
        tests = [
            ("批量天气查询", "查询北京、上海、广州三个城市的天气情况"),
            ("地理编码", "获取天安门广场的经纬度坐标"),
            ("POI搜索", "搜索北京市朝阳区的医院信息"),
            ("路径规划", "规划从北京到上海的驾车路线")
        ]
        
        for test_name, query in tests:
            result = await enhanced_system_query(query)
            print_test_result(test_name, result)
            
        return True
    except Exception as e:
        print_test_result("AMAP Agent 深度测试", f"❌ 错误: {str(e)}", "❌")
        return False

async def comprehensive_file_test():
    """全面测试Safe File Agent"""
    print_separator("Safe File Agent 深度功能测试")
    
    try:
        from enhanced_data_agent import enhanced_system_query
        
        tests = [
            ("创建JSON文件", "在workspace/data目录下创建config.json文件，内容包含项目配置信息"),
            ("读取文件", "读取刚才创建的config.json文件内容"),
            ("目录管理", "列出workspace目录下的所有文件和子目录"),
            ("文件信息", "获取workspace目录的详细信息，包括文件大小和修改时间")
        ]
        
        for test_name, query in tests:
            result = await enhanced_system_query(query)
            print_test_result(test_name, result)
            
        return True
    except Exception as e:
        print_test_result("File Agent 深度测试", f"❌ 错误: {str(e)}", "❌")
        return False

async def comprehensive_sql_test():
    """全面测试SQL Agent"""
    print_separator("SQL Agent 深度功能测试")
    
    try:
        from enhanced_data_agent import enhanced_system_query
        
        tests = [
            ("数据库信息", "显示当前数据库的所有表名"),
            ("表结构查询", "如果有telco相关的表，显示其结构信息"),
            ("连接池状态", "检查数据库连接池的状态和配置"),
            ("查询优化", "提供SQL查询性能优化的建议")
        ]
        
        for test_name, query in tests:
            result = await enhanced_system_query(query)
            print_test_result(test_name, result)
            
        return True
    except Exception as e:
        print_test_result("SQL Agent 深度测试", f"❌ 错误: {str(e)}", "❌")
        return False

async def comprehensive_terminal_test():
    """全面测试Terminal Command Agent"""
    print_separator("Terminal Command Agent 深度功能测试")
    
    try:
        from enhanced_data_agent import enhanced_system_query
        
        tests = [
            ("系统信息", "获取系统的详细信息，包括CPU、内存、磁盘使用情况"),
            ("进程管理", "查看当前运行的Python进程"),
            ("文件权限", "检查workspace目录的权限设置"),
            ("环境变量", "显示当前的Python环境路径和版本信息")
        ]
        
        for test_name, query in tests:
            result = await enhanced_system_query(query)
            print_test_result(test_name, result)
            
        return True
    except Exception as e:
        print_test_result("Terminal Agent 深度测试", f"❌ 错误: {str(e)}", "❌")
        return False

async def test_agent_collaboration():
    """测试Agent间的协作功能"""
    print_separator("Agent协作功能测试")
    
    try:
        from enhanced_data_agent import enhanced_system_query
        
        # 复杂任务，需要多个Agent协作
        collaborative_tests = [
            ("数据分析+可视化", "从workspace中读取数据文件，使用Python进行分析并生成图表"),
            ("地理数据+文件保存", "查询北京的天气数据，然后将结果保存为JSON文件"),
            ("终端+文件+Python", "通过终端命令检查系统状态，将信息保存到文件，然后用Python分析")
        ]
        
        for test_name, query in collaborative_tests:
            result = await enhanced_system_query(query)
            print_test_result(test_name, result)
            
        return True
    except Exception as e:
        print_test_result("Agent协作测试", f"❌ 错误: {str(e)}", "❌")
        return False

async def main():
    """主测试函数"""
    print("🎯 开始全面Agent功能测试")
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # 全面测试各Agent
    comprehensive_tests = [
        ("Enhanced RAG Agent", comprehensive_rag_test),
        ("Enhanced Python Agent", comprehensive_python_test), 
        ("Enhanced AMAP Agent", comprehensive_amap_test),
        ("Safe File Agent", comprehensive_file_test),
        ("SQL Agent", comprehensive_sql_test),
        ("Terminal Command Agent", comprehensive_terminal_test),
        ("Agent协作功能", test_agent_collaboration)
    ]
    
    for test_name, test_func in comprehensive_tests:
        try:
            print(f"\n🚀 开始 {test_name} 深度测试...")
            result = await test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 测试失败: {str(e)}")
            test_results[test_name] = False
    
    # 输出测试摘要
    print_separator("深度测试结果摘要")
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {test_name}")
    
    print(f"\n📊 总体结果: {passed}/{total} 项深度测试通过")
    
    if passed == total:
        print("🎉 所有深度测试通过！系统功能完整可靠！")
    else:
        print(f"⚠️  有 {total - passed} 项测试需要修复")
        
    return test_results

if __name__ == "__main__":
    asyncio.run(main())
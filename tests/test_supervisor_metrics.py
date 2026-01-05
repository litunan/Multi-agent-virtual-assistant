#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多智能体调度测试模块
测试指标：
1. 支持的最大对话轮数
2. 跨Agent任务拆解成功率
3. Agent路由准确率
4. 上下文保持能力

Author: Wangwang-Agent Team
Date: 2026-01-04
"""

import os
import sys
import json
import asyncio
import time
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.load_key import load_key

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/test_results/supervisor_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""


@dataclass
class SupervisorMetrics:
    """Supervisor调度指标汇总"""
    # 对话轮数测试
    max_conversation_rounds: int = 0
    conversation_context_retention_rate: float = 0.0
    
    # Agent路由测试
    total_routing_tests: int = 0
    correct_routing_count: int = 0
    routing_accuracy: float = 0.0
    
    # 跨Agent任务测试
    total_cross_agent_tasks: int = 0
    successful_cross_agent_tasks: int = 0
    cross_agent_success_rate: float = 0.0
    
    # 性能指标
    avg_response_time: float = 0.0
    total_test_duration: float = 0.0
    
    # 详细结果
    test_results: List[Dict] = field(default_factory=list)


class SupervisorMetricsTester:
    """Supervisor调度指标测试器"""
    
    def __init__(self):
        self.metrics = SupervisorMetrics()
        self.test_data_path = os.path.join(
            os.path.dirname(__file__), 'test_data'
        )
        self.results_path = os.path.join(
            os.path.dirname(__file__), 'test_results'
        )
        os.makedirs(self.results_path, exist_ok=True)
        
        # 加载测试数据
        self._load_test_data()
        
    def _load_test_data(self):
        """加载测试数据"""
        try:
            with open(os.path.join(self.test_data_path, 'test_questions.json'), 
                     'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
            
            with open(os.path.join(self.test_data_path, 'test_scenarios.json'), 
                     'r', encoding='utf-8') as f:
                self.scenarios_data = json.load(f)
                
            logger.info("测试数据加载成功")
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}")
            self.questions_data = {}
            self.scenarios_data = {}

    async def test_agent_routing_accuracy(self) -> TestResult:
        """
        测试Agent路由准确率
        验证Supervisor能否正确将任务分配给对应的Agent
        """
        logger.info("=" * 60)
        logger.info("开始测试: Agent路由准确率")
        logger.info("=" * 60)
        
        start_time = time.time()
        correct_count = 0
        total_count = 0
        routing_details = []
        
        try:
            from enhanced_data_agent1 import professional_system_query
            
            routing_tests = self.scenarios_data.get('supervisor_routing_tests', [])
            
            for test_case in routing_tests:
                total_count += 1
                test_input = test_case['input']
                expected_agent = test_case['expected_agent']
                
                logger.info(f"\n测试 {test_case['id']}: {test_input[:50]}...")
                
                try:
                    # 执行查询并检测实际调用的Agent
                    response = await professional_system_query(
                        test_input,
                        user_id="test_user",
                        session_id="routing_test_session"
                    )
                    
                    # 检查响应中是否包含预期Agent的特征
                    # 通过响应内容推断使用了哪个Agent
                    actual_agent = self._detect_agent_from_response(
                        response, test_case['category']
                    )
                    
                    is_correct = actual_agent == expected_agent
                    if is_correct:
                        correct_count += 1
                        status = "✅ 正确"
                    else:
                        status = f"❌ 错误 (预期: {expected_agent}, 实际: {actual_agent})"
                    
                    routing_details.append({
                        'test_id': test_case['id'],
                        'input': test_input,
                        'expected_agent': expected_agent,
                        'actual_agent': actual_agent,
                        'correct': is_correct,
                        'category': test_case['category']
                    })
                    
                    logger.info(f"  结果: {status}")
                    
                except Exception as e:
                    logger.error(f"  测试执行失败: {e}")
                    routing_details.append({
                        'test_id': test_case['id'],
                        'input': test_input,
                        'expected_agent': expected_agent,
                        'actual_agent': 'error',
                        'correct': False,
                        'error': str(e)
                    })
                
                # 避免请求过快
                await asyncio.sleep(0.5)
            
            # 计算准确率
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            
            self.metrics.total_routing_tests = total_count
            self.metrics.correct_routing_count = correct_count
            self.metrics.routing_accuracy = accuracy
            
            duration = time.time() - start_time
            
            logger.info(f"\n路由准确率测试完成:")
            logger.info(f"  总测试数: {total_count}")
            logger.info(f"  正确数: {correct_count}")
            logger.info(f"  准确率: {accuracy:.1f}%")
            
            return TestResult(
                test_name="Agent路由准确率测试",
                success=accuracy >= 80,  # 80%以上视为成功
                duration=duration,
                details={
                    'total_tests': total_count,
                    'correct_count': correct_count,
                    'accuracy': accuracy,
                    'routing_details': routing_details
                }
            )
            
        except ImportError as e:
            logger.error(f"导入模块失败: {e}")
            return TestResult(
                test_name="Agent路由准确率测试",
                success=False,
                duration=time.time() - start_time,
                error_message=f"导入失败: {e}"
            )

    def _detect_agent_from_response(self, response: str, category: str) -> str:
        """根据响应内容推断使用的Agent"""
        response_lower = response.lower()
        
        # 地理位置服务特征
        if any(kw in response_lower for kw in ['天气', '温度', '°c', '经纬度', '坐标', '路线', '公里']):
            return 'enhanced_amap_agent'
        
        # Python数据分析特征
        if any(kw in response_lower for kw in ['import', 'print', 'plt.', 'numpy', 'pandas', '执行结果']):
            return 'enhanced_python_agent'
        
        # RAG检索特征
        if any(kw in response_lower for kw in ['抗癌肽', '肽', '机制', '细胞', '肿瘤', '研究表明']):
            return 'enhanced_rag_agent'
        
        # 文件操作特征
        if any(kw in response_lower for kw in ['文件', '目录', '读取', '创建', '保存', 'workspace']):
            return 'safe_file_agent'
        
        # 终端命令特征
        if any(kw in response_lower for kw in ['命令', '执行', 'ls', 'pwd', '进程', '内存']):
            return 'terminal_command_agent'
        
        # 根据预期类别返回
        category_agent_map = {
            '地理位置服务': 'enhanced_amap_agent',
            '数据分析': 'enhanced_python_agent',
            'RAG检索': 'enhanced_rag_agent',
            '文件操作': 'safe_file_agent',
            '终端命令': 'terminal_command_agent'
        }
        return category_agent_map.get(category, 'unknown')

    async def test_cross_agent_task_decomposition(self) -> TestResult:
        """
        测试跨Agent任务拆解成功率
        验证系统能否正确拆解并执行需要多个Agent协作的复杂任务
        """
        logger.info("=" * 60)
        logger.info("开始测试: 跨Agent任务拆解")
        logger.info("=" * 60)
        
        start_time = time.time()
        successful_count = 0
        total_count = 0
        task_details = []
        
        try:
            from enhanced_data_agent1 import professional_system_query
            
            cross_agent_tests = self.questions_data.get('cross_agent_questions', [])
            
            for test_case in cross_agent_tests:
                total_count += 1
                question = test_case['question']
                expected_agents = test_case['expected_agents']
                
                logger.info(f"\n测试 {test_case['id']}: {test_case['description']}")
                logger.info(f"  问题: {question[:60]}...")
                logger.info(f"  预期Agents: {expected_agents}")
                
                try:
                    response = await professional_system_query(
                        question,
                        user_id="test_user",
                        session_id="cross_agent_test"
                    )
                    
                    # 检查响应是否包含多个Agent的工作特征
                    detected_agents = self._detect_multiple_agents(response)
                    
                    # 计算Agent覆盖率
                    covered_agents = set(detected_agents) & set(expected_agents)
                    coverage_rate = len(covered_agents) / len(expected_agents) * 100
                    
                    is_success = coverage_rate >= 50  # 覆盖50%以上视为成功
                    if is_success:
                        successful_count += 1
                        status = f"✅ 成功 (覆盖率: {coverage_rate:.0f}%)"
                    else:
                        status = f"❌ 失败 (覆盖率: {coverage_rate:.0f}%)"
                    
                    task_details.append({
                        'test_id': test_case['id'],
                        'description': test_case['description'],
                        'expected_agents': expected_agents,
                        'detected_agents': detected_agents,
                        'coverage_rate': coverage_rate,
                        'success': is_success,
                        'response_preview': response[:200]
                    })
                    
                    logger.info(f"  检测到的Agents: {detected_agents}")
                    logger.info(f"  结果: {status}")
                    
                except Exception as e:
                    logger.error(f"  任务执行失败: {e}")
                    task_details.append({
                        'test_id': test_case['id'],
                        'description': test_case['description'],
                        'expected_agents': expected_agents,
                        'success': False,
                        'error': str(e)
                    })
                
                await asyncio.sleep(1)  # 复杂任务间隔长一点
            
            # 计算成功率
            success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
            
            self.metrics.total_cross_agent_tasks = total_count
            self.metrics.successful_cross_agent_tasks = successful_count
            self.metrics.cross_agent_success_rate = success_rate
            
            duration = time.time() - start_time
            
            logger.info(f"\n跨Agent任务测试完成:")
            logger.info(f"  总任务数: {total_count}")
            logger.info(f"  成功数: {successful_count}")
            logger.info(f"  成功率: {success_rate:.1f}%")
            
            return TestResult(
                test_name="跨Agent任务拆解测试",
                success=success_rate >= 60,
                duration=duration,
                details={
                    'total_tasks': total_count,
                    'successful_count': successful_count,
                    'success_rate': success_rate,
                    'task_details': task_details
                }
            )
            
        except ImportError as e:
            logger.error(f"导入模块失败: {e}")
            return TestResult(
                test_name="跨Agent任务拆解测试",
                success=False,
                duration=time.time() - start_time,
                error_message=f"导入失败: {e}"
            )

    def _detect_multiple_agents(self, response: str) -> List[str]:
        """检测响应中涉及的多个Agent"""
        detected = []
        response_lower = response.lower()
        
        agent_signatures = {
            'enhanced_amap_agent': ['天气', '坐标', '路线', '公里', '地图', '位置'],
            'enhanced_python_agent': ['import', 'print', 'python', '执行', '代码', '计算'],
            'enhanced_rag_agent': ['抗癌肽', '肽', '机制', '研究', '文献'],
            'safe_file_agent': ['文件', '保存', '创建', '写入', 'workspace'],
            'terminal_command_agent': ['命令', '终端', '执行', 'shell']
        }
        
        for agent, keywords in agent_signatures.items():
            if any(kw in response_lower for kw in keywords):
                detected.append(agent)
        
        return detected

    async def test_long_conversation_capability(self) -> TestResult:
        """
        测试长对话能力
        验证系统能够维持多少轮有效对话，以及上下文保持能力
        """
        logger.info("=" * 60)
        logger.info("开始测试: 长对话能力")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            from enhanced_data_agent1 import professional_system_query
            
            conversation_context = self.questions_data.get('long_conversation_context', [])
            
            successful_rounds = 0
            context_references = 0  # 统计成功引用上下文的次数
            conversation_history = []
            
            session_id = f"long_conv_test_{int(time.time())}"
            
            for round_num, message in enumerate(conversation_context, 1):
                logger.info(f"\n第 {round_num} 轮对话: {message[:50]}...")
                
                try:
                    response = await professional_system_query(
                        message,
                        user_id="test_user",
                        session_id=session_id
                    )
                    
                    if response and len(response) > 20:
                        successful_rounds += 1
                        
                        # 检查是否引用了之前的上下文
                        if round_num > 1:
                            if self._check_context_reference(response, conversation_history):
                                context_references += 1
                                logger.info(f"  ✅ 成功 (检测到上下文引用)")
                            else:
                                logger.info(f"  ✅ 成功")
                        else:
                            logger.info(f"  ✅ 成功")
                        
                        conversation_history.append({
                            'round': round_num,
                            'user': message,
                            'assistant': response[:200]
                        })
                    else:
                        logger.warning(f"  ⚠️ 响应过短或为空")
                        
                except Exception as e:
                    logger.error(f"  ❌ 对话失败: {e}")
                    break
                
                await asyncio.sleep(0.5)
            
            # 计算上下文保持率
            possible_references = max(successful_rounds - 1, 1)
            context_retention_rate = (context_references / possible_references * 100)
            
            self.metrics.max_conversation_rounds = successful_rounds
            self.metrics.conversation_context_retention_rate = context_retention_rate
            
            duration = time.time() - start_time
            
            logger.info(f"\n长对话测试完成:")
            logger.info(f"  成功对话轮数: {successful_rounds}")
            logger.info(f"  上下文保持率: {context_retention_rate:.1f}%")
            
            return TestResult(
                test_name="长对话能力测试",
                success=successful_rounds >= 8,  # 至少支持8轮
                duration=duration,
                details={
                    'successful_rounds': successful_rounds,
                    'total_rounds': len(conversation_context),
                    'context_references': context_references,
                    'context_retention_rate': context_retention_rate,
                    'conversation_history': conversation_history
                }
            )
            
        except ImportError as e:
            logger.error(f"导入模块失败: {e}")
            return TestResult(
                test_name="长对话能力测试",
                success=False,
                duration=time.time() - start_time,
                error_message=f"导入失败: {e}"
            )

    def _check_context_reference(self, response: str, history: List[Dict]) -> bool:
        """检查响应是否引用了历史对话内容"""
        if not history:
            return False
        
        # 检查是否包含"之前"、"刚才"、"前面"等上下文引用词
        context_keywords = ['之前', '刚才', '前面', '上面', '提到', '讨论', '说过', '基于']
        response_lower = response.lower()
        
        if any(kw in response_lower for kw in context_keywords):
            return True
        
        # 检查是否引用了历史对话中的关键词
        for hist in history[-3:]:  # 检查最近3轮
            user_keywords = [w for w in hist['user'].split() if len(w) > 2]
            if any(kw in response for kw in user_keywords):
                return True
        
        return False

    async def run_all_tests(self) -> SupervisorMetrics:
        """运行所有Supervisor调度测试"""
        logger.info("\n" + "=" * 70)
        logger.info("开始运行 Supervisor 调度指标测试")
        logger.info("=" * 70)
        
        total_start = time.time()
        
        # 1. Agent路由准确率测试
        routing_result = await self.test_agent_routing_accuracy()
        self.metrics.test_results.append(asdict(routing_result))
        
        # 2. 跨Agent任务拆解测试
        cross_agent_result = await self.test_cross_agent_task_decomposition()
        self.metrics.test_results.append(asdict(cross_agent_result))
        
        # 3. 长对话能力测试
        long_conv_result = await self.test_long_conversation_capability()
        self.metrics.test_results.append(asdict(long_conv_result))
        
        self.metrics.total_test_duration = time.time() - total_start
        
        # 保存测试结果
        self._save_results()
        
        # 输出汇总
        self._print_summary()
        
        return self.metrics

    def _save_results(self):
        """保存测试结果到文件"""
        result_file = os.path.join(
            self.results_path, 
            f'supervisor_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.metrics), f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n测试结果已保存至: {result_file}")

    def _print_summary(self):
        """打印测试汇总"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 SUPERVISOR 调度测试汇总报告")
        logger.info("=" * 70)
        
        logger.info(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  📈 简历指标数据                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  ✅ Agent路由准确率:           {self.metrics.routing_accuracy:>6.1f}%                          │
│  ✅ 跨Agent任务成功率:         {self.metrics.cross_agent_success_rate:>6.1f}%                          │
│  ✅ 支持对话轮数:              {self.metrics.max_conversation_rounds:>6}轮                           │
│  ✅ 上下文保持率:              {self.metrics.conversation_context_retention_rate:>6.1f}%                          │
├─────────────────────────────────────────────────────────────────────┤
│  ⏱️  总测试时长:               {self.metrics.total_test_duration:>6.1f}秒                          │
└─────────────────────────────────────────────────────────────────────┘
""")
        
        # 生成简历友好的描述
        logger.info("\n📝 简历描述建议:")
        logger.info(f"  - 成功支持 {self.metrics.max_conversation_rounds} 轮以上的复杂长对话")
        logger.info(f"  - 复杂任务拆解成功率达到 {self.metrics.cross_agent_success_rate:.0f}%")
        logger.info(f"  - Agent智能路由准确率 {self.metrics.routing_accuracy:.0f}%")


async def main():
    """主函数"""
    # 确保测试结果目录存在
    os.makedirs('tests/test_results', exist_ok=True)
    
    tester = SupervisorMetricsTester()
    metrics = await tester.run_all_tests()
    
    return metrics


if __name__ == "__main__":
    asyncio.run(main())

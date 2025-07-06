import pandas as pd
import json
import concurrent.futures
from copy import deepcopy
from openai import OpenAI
from typing import Dict, List, Tuple, Any, Optional
import time

# 导入配置
try:
    from config import *
    from prompts import (
        get_textual_assessment_prompt,
        get_textual_disagreement_resolution_prompt,
        get_textual_assessment_prompt_simplified,
        get_textual_disagreement_resolution_prompt_simplified
    )
except ImportError:
    print("错误: 未找到config.py或prompts.py文件。请将config_template.py重命名为config.py并配置相应的API密钥。")
    exit(1)

# --- LLM客户端配置 ---
# 根据论文，textual dimension使用两个成本效益高的模型进行评估
client_deepseek = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

client_kimi = OpenAI(
    api_key=KIMI_API_KEY,
    base_url=KIMI_BASE_URL,
)

# GPT-4o用于分歧解决
client_gpt4o = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# 所有配置现在都从config.py导入

class TextualDimensionAssessmentAgent:
    """
    Textual Dimension Assessment Agent
    实现论文中描述的双LLM评估机制和分歧解决机制
    """
    
    def __init__(self):
        """初始化评估代理"""
        self.checklist_template = CHECKLIST_TEMPLATE
        self.evaluation_results = {}  # 存储所有评估结果，按用例名区分
        
    def _is_text_present(self, text) -> bool:
        """检查文本是否实际存在内容，而不仅仅是空白。"""
        return text is not None and str(text).strip() != ""
    
    def _format_report_data_for_llm(self, report_data: dict) -> str:
        """将报告数据格式化为LLM易于理解的字符串。"""
        formatted_str = f"报告类型: {report_data.get('type', '未知')}\n"
        formatted_str += f"标题: {report_data.get('title', report_data.get('description', ''))}\n"
        
        # 遍历报告数据，排除内部字段
        for key, value in report_data.items():
            if key not in ["type", "title", "test_case_id", "defect_id", "test_case_id_ref",
                           "priority", "designer", "report_fill_time", "submitter", "defect_type",
                           "ui_title", "attachments_screenshot", "attachments_video", "attachments"]:
                if self._is_text_present(value):
                    formatted_str += f"{key.replace('_', ' ').title()}: {value}\n"
        
        # 特殊处理附件信息
        attachments_present = False
        if report_data.get("type") == "test_case" and self._is_text_present(report_data.get("attachments")):
            attachments_present = True
        elif report_data.get("type") == "defect" and (self._is_text_present(report_data.get("attachments_screenshot")) or self._is_text_present(report_data.get("attachments_video"))):
            attachments_present = True
        
        formatted_str += f"附件（截图/录屏等）是否已提供: {'是' if attachments_present else '否'}\n"
        return formatted_str.strip()

    def generate_checklist_prompt(self, report_data: dict) -> str:
        """
        Step 1: Checklist Generation
        生成适用于当前报告类型的评估清单prompt
        """
        report_type = report_data.get("type", "unknown")
        
        # 准备发送给LLM的简化清单
        llm_checklist_for_prompt = {}
        for category_name, indicators in self.checklist_template.items():
            llm_checklist_for_prompt[category_name] = []
            for indicator in indicators:
                if not indicator.get("is_defect_specific") or report_type == "defect":
                    llm_checklist_for_prompt[category_name].append({
                        "id": indicator["id"],
                        "name": indicator["name"],
                        "rule_content": indicator["rule_content"],
                        "score": indicator["score"],
                        "checkpoints": [{"description": cp["description"]} for cp in indicator["checkpoints"]]
                    })

        return get_textual_assessment_prompt(
            self._format_report_data_for_llm(report_data),
            llm_checklist_for_prompt
        ) 

    def generate_simplified_checklist(self, report_data: dict) -> Tuple[str, str]:
        """
        生成简化版的评估清单提示词，返回提示词和checkpoint映射
        """
        report_type = report_data.get("type", "unknown")
        
        simplified_checklist_items = []
        checkpoint_mapping = {}  # checkpoint_id -> {category, indicator_id, checkpoint_index, description}
        
        for category_name, indicators in self.checklist_template.items():
            for indicator in indicators:
                if not indicator.get("is_defect_specific") or report_type == "defect":
                    indicator_id = indicator["id"]
                    for i, checkpoint in enumerate(indicator["checkpoints"]):
                        checkpoint_id = f"{indicator_id}-{i+1}"
                        description = checkpoint["description"]
                        
                        simplified_checklist_items.append(f"{checkpoint_id}: {description}")
                        
                        checkpoint_mapping[checkpoint_id] = {
                            "category": category_name,
                            "indicator_id": indicator_id,
                            "checkpoint_index": i,
                            "description": description
                        }
        
        simplified_checklist_str = "\n".join(simplified_checklist_items)
        
        prompt = get_textual_assessment_prompt_simplified(
            self._format_report_data_for_llm(report_data),
            simplified_checklist_str
        )
        
        return prompt, checkpoint_mapping

    def convert_simplified_to_full_format(self, simplified_result: Dict[str, Any], checkpoint_mapping: Dict[str, Dict]) -> Dict[str, Any]:
        """
        将简化格式的评估结果转换为完整格式
        """
        full_evaluation = {}
        
        # 初始化完整格式结构
        for category_name, indicators in self.checklist_template.items():
            full_evaluation[category_name] = []
            for indicator in indicators:
                full_evaluation[category_name].append({
                    "id": indicator["id"],
                    "checkpoints": [
                        {
                            "description": cp["description"],
                            "value": False,
                            "reasoning": ""
                        } for cp in indicator["checkpoints"]
                    ]
                })
        
        # 填充简化结果数据
        for checkpoint_result in simplified_result.get("checkpoint_results", []):
            checkpoint_id = checkpoint_result["checkpoint_id"]
            if checkpoint_id in checkpoint_mapping:
                mapping = checkpoint_mapping[checkpoint_id]
                category = mapping["category"]
                indicator_id = mapping["indicator_id"]
                checkpoint_index = mapping["checkpoint_index"]
                
                # 找到对应的指标
                for indicator in full_evaluation[category]:
                    if indicator["id"] == indicator_id:
                        if checkpoint_index < len(indicator["checkpoints"]):
                            indicator["checkpoints"][checkpoint_index]["value"] = checkpoint_result["value"]
                            indicator["checkpoints"][checkpoint_index]["reasoning"] = checkpoint_result["reasoning"]
                        break
        
        return full_evaluation

    def assess_with_single_llm_simplified(self, client: OpenAI, model: str, prompt: str, llm_name: str) -> Dict[str, Any]:
        """
        使用单个LLM进行简化格式评估
        """
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_EVALUATOR},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            
            response_content = chat_completion.choices[0].message.content
            print(f"[{llm_name}] Raw response length: {len(response_content)}")
            
            # 尝试解析JSON
            try:
                evaluation_result = json.loads(response_content)
            except json.JSONDecodeError as json_err:
                print(f"[{llm_name}] JSON decode error: {json_err}")
                print(f"[{llm_name}] Response content preview: {response_content[:500]}...")
                
                # 尝试从响应中提取JSON块
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
                if json_match:
                    try:
                        print(f"[{llm_name}] Trying to parse extracted JSON block...")
                        evaluation_result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        print(f"[{llm_name}] Failed to parse extracted JSON block")
                        raise json_err
                else:
                    print(f"[{llm_name}] No JSON block found in response")
                    raise json_err
            
            return {
                "success": True,
                "evaluation": evaluation_result,
                "raw_response": response_content,
                "llm_name": llm_name
            }
            
        except json.JSONDecodeError as e:
            print(f"错误: {llm_name} 返回的不是有效的JSON格式。{e}")
            return {
                "success": False,
                "error": "JSON解析失败",
                "details": str(e),
                "llm_name": llm_name
            }
        except Exception as e:
            print(f"调用 {llm_name} 时发生错误: {e}")
            return {
                "success": False,
                "error": "LLM调用失败",
                "details": str(e),
                "llm_name": llm_name
            }

    def dual_llm_assessment_simplified(self, prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        使用两个独立的LLM进行简化格式评估
        """
        # 并行调用两个LLM
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # DeepSeek评估
            future_deepseek = executor.submit(
                self.assess_with_single_llm_simplified, 
                client_deepseek, 
                DEEPSEEK_MODEL, 
                prompt, 
                f"DeepSeek-{DEEPSEEK_MODEL}"
            )
            
            # Kimi评估
            future_kimi = executor.submit(
                self.assess_with_single_llm_simplified,
                client_kimi,
                KIMI_MODEL,
                prompt,
                f"Kimi-{KIMI_MODEL}"
            )
            
            # 获取结果
            deepseek_result = future_deepseek.result()
            kimi_result = future_kimi.result()
            
        return deepseek_result, kimi_result

    def single_llm_assessment_simplified(self, prompt: str, llm_choice: str = None) -> Dict[str, Any]:
        """
        使用单个LLM进行简化格式评估
        
        Args:
            prompt: 评估提示词
            llm_choice: LLM选择 ("deepseek" 或 "kimi")，如果为None则使用配置中的DEFAULT_SINGLE_LLM
        """
        if llm_choice is None:
            llm_choice = DEFAULT_SINGLE_LLM
        
        if llm_choice.lower() == "deepseek":
            return self.assess_with_single_llm_simplified(
                client_deepseek, 
                DEEPSEEK_MODEL, 
                prompt, 
                f"DeepSeek-{DEEPSEEK_MODEL}"
            )
        elif llm_choice.lower() == "kimi":
            return self.assess_with_single_llm_simplified(
                client_kimi,
                KIMI_MODEL,
                prompt,
                f"Kimi-{KIMI_MODEL}"
            )
        else:
            raise ValueError(f"不支持的LLM选择: {llm_choice}。请选择 'deepseek' 或 'kimi'。")

    def identify_disagreements_simplified(self, evaluation1: Dict[str, Any], evaluation2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        识别两个简化格式LLM评估结果之间的分歧
        """
        disagreements = []
        
        # 创建checkpoint_id到结果的映射
        eval1_map = {result["checkpoint_id"]: result for result in evaluation1.get("checkpoint_results", [])}
        eval2_map = {result["checkpoint_id"]: result for result in evaluation2.get("checkpoint_results", [])}
        
        # 检查所有checkpoint_id
        all_checkpoint_ids = set(eval1_map.keys()) | set(eval2_map.keys())
        
        for checkpoint_id in all_checkpoint_ids:
            if checkpoint_id in eval1_map and checkpoint_id in eval2_map:
                result1 = eval1_map[checkpoint_id]
                result2 = eval2_map[checkpoint_id]
                
                if result1.get("value") != result2.get("value"):
                    disagreements.append({
                        "checkpoint_id": checkpoint_id,
                        "llm1_value": result1.get("value"),
                        "llm1_reasoning": result1.get("reasoning"),
                        "llm2_value": result2.get("value"),
                        "llm2_reasoning": result2.get("reasoning")
                    })
        
        return disagreements

    def resolve_disagreements_simplified(self, report_data: dict, disagreements: List[Dict[str, Any]], 
                                       deepseek_result: Dict[str, Any], kimi_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用GPT-4o进行简化格式分歧解决
        """
        if not disagreements:
            return {
                "final_evaluation": deepseek_result["evaluation"],
                "resolution_details": {"no_disagreements": True},
                "disagreements_count": 0
            }
        
        # 构造分歧解决prompt
        disagreement_summary = "\n".join([
            f"分歧检查点: {d['checkpoint_id']}\n"
            f"  DeepSeek: {d['llm1_value']} (理由: {d['llm1_reasoning']})\n"
            f"  Kimi: {d['llm2_value']} (理由: {d['llm2_reasoning']})"
            for d in disagreements
        ])

        resolution_prompt = get_textual_disagreement_resolution_prompt_simplified(
            self._format_report_data_for_llm(report_data),
            disagreement_summary
        )

        try:
            chat_completion = client_gpt4o.chat.completions.create(
                model=GPT4O_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ARBITRATOR},
                    {"role": "user", "content": resolution_prompt}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            
            resolution_content = chat_completion.choices[0].message.content
            resolution_result = json.loads(resolution_content)
            
            # 应用分歧解决结果到DeepSeek的结果
            final_evaluation = deepcopy(deepseek_result["evaluation"])
            
            # 创建checkpoint_id到结果的映射
            eval_map = {result["checkpoint_id"]: result for result in final_evaluation.get("checkpoint_results", [])}
            
            for resolution in resolution_result.get("resolved_disagreements", []):
                checkpoint_id = resolution["checkpoint_id"]
                final_value = resolution["final_value"]
                resolution_reasoning = resolution["resolution_reasoning"]
                
                if checkpoint_id in eval_map:
                    eval_map[checkpoint_id]["value"] = final_value
                    eval_map[checkpoint_id]["reasoning"] = resolution_reasoning
            
            return {
                "final_evaluation": final_evaluation,
                "resolution_details": resolution_result,
                "disagreements_count": len(disagreements)
            }
            
        except Exception as e:
            print(f"分歧解决过程出错: {e}")
            # 如果分歧解决失败，返回DeepSeek的结果作为备选
            return {
                "final_evaluation": deepseek_result["evaluation"],
                "resolution_details": {"error": str(e)},
                "disagreements_count": len(disagreements)
            }

    def assess_with_single_llm(self, client: OpenAI, model: str, prompt: str, llm_name: str) -> Dict[str, Any]:
        """
        使用单个LLM进行评估
        """
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_EVALUATOR},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            
            response_content = chat_completion.choices[0].message.content
            evaluation_result = json.loads(response_content)
            
            return {
                "success": True,
                "evaluation": evaluation_result,
                "raw_response": response_content,
                "llm_name": llm_name
            }
            
        except json.JSONDecodeError as e:
            print(f"错误: {llm_name} 返回的不是有效的JSON格式。{e}")
            return {
                "success": False,
                "error": "JSON解析失败",
                "details": str(e),
                "llm_name": llm_name
            }
        except Exception as e:
            print(f"调用 {llm_name} 时发生错误: {e}")
            return {
                "success": False,
                "error": "LLM调用失败",
                "details": str(e),
                "llm_name": llm_name
            }

    def dual_llm_assessment(self, prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step 2: Dual LLM Assessment
        使用两个独立的LLM进行评估
        """
        # 并行调用两个LLM
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # DeepSeek评估
            future_deepseek = executor.submit(
                self.assess_with_single_llm, 
                client_deepseek, 
                DEEPSEEK_MODEL, 
                prompt, 
                DEEPSEEK_MODEL
            )
            
            # Kimi评估
            future_kimi = executor.submit(
                self.assess_with_single_llm,
                client_kimi,
                KIMI_MODEL,
                prompt,
                KIMI_MODEL
            )
            
            # 获取结果
            deepseek_result = future_deepseek.result()
            kimi_result = future_kimi.result()
            
        return deepseek_result, kimi_result

    def identify_disagreements(self, evaluation1: Dict[str, Any], evaluation2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        识别两个LLM评估结果之间的分歧
        """
        disagreements = []
        
        for category_name in evaluation1.keys():
            if category_name in evaluation2:
                # 检查每个指标的每个检查点
                indicators1 = evaluation1[category_name]
                indicators2 = evaluation2[category_name]
                
                for i, indicator1 in enumerate(indicators1):
                    if i < len(indicators2):
                        indicator2 = indicators2[i]
                        
                        # 检查检查点是否有分歧
                        checkpoints1 = indicator1.get("checkpoints", [])
                        checkpoints2 = indicator2.get("checkpoints", [])
                        
                        for j, checkpoint1 in enumerate(checkpoints1):
                            if j < len(checkpoints2):
                                checkpoint2 = checkpoints2[j]
                                
                                if checkpoint1.get("value") != checkpoint2.get("value"):
                                    disagreements.append({
                                        "category": category_name,
                                        "indicator_id": indicator1.get("id"),
                                        "checkpoint_index": j,
                                        "checkpoint_description": checkpoint1.get("description"),
                                        "llm1_value": checkpoint1.get("value"),
                                        "llm1_reasoning": checkpoint1.get("reasoning"),
                                        "llm2_value": checkpoint2.get("value"),
                                        "llm2_reasoning": checkpoint2.get("reasoning")
                                    })
        
        return disagreements

    def resolve_disagreements(self, report_data: dict, disagreements: List[Dict[str, Any]], 
                             deepseek_result: Dict[str, Any], kimi_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Disagreement Resolution
        使用GPT-4o进行分歧解决的"thinking and judgment"过程
        """
        if not disagreements:
            return {
                "final_evaluation": deepseek_result["evaluation"],
                "resolution_details": {"no_disagreements": True},
                "disagreements_count": 0
            }
        
        # 构造分歧解决prompt
        disagreement_summary = "\n".join([
            f"分歧 {i+1}: {d['category']} - {d['indicator_id']} - {d['checkpoint_description']}\n"
            f"  DeepSeek: {d['llm1_value']} (理由: {d['llm1_reasoning']})\n"
            f"  Kimi: {d['llm2_value']} (理由: {d['llm2_reasoning']})"
            for i, d in enumerate(disagreements)
        ])

        resolution_prompt = get_textual_disagreement_resolution_prompt(
            self._format_report_data_for_llm(report_data),
            disagreement_summary
        )

        try:
            chat_completion = client_gpt4o.chat.completions.create(
                model=GPT4O_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ARBITRATOR},
                    {"role": "user", "content": resolution_prompt}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            
            resolution_content = chat_completion.choices[0].message.content
            resolution_result = json.loads(resolution_content)
            
            # 应用分歧解决结果
            final_evaluation = deepcopy(deepseek_result["evaluation"])
            
            for resolution in resolution_result.get("resolved_disagreements", []):
                category = resolution["category"]
                indicator_id = resolution["indicator_id"]
                checkpoint_index = resolution["checkpoint_index"]
                final_value = resolution["final_value"]
                resolution_reasoning = resolution["resolution_reasoning"]
                
                # 找到对应的检查点并更新
                for indicator in final_evaluation[category]:
                    if indicator.get("id") == indicator_id:
                        if checkpoint_index < len(indicator["checkpoints"]):
                            indicator["checkpoints"][checkpoint_index]["value"] = final_value
                            indicator["checkpoints"][checkpoint_index]["reasoning"] = resolution_reasoning
                            break
            
            return {
                "final_evaluation": final_evaluation,
                "resolution_details": resolution_result,
                "disagreements_count": len(disagreements)
            }
            
        except Exception as e:
            print(f"分歧解决过程出错: {e}")
            # 如果分歧解决失败，返回DeepSeek的结果作为备选
            return {
                "final_evaluation": deepseek_result["evaluation"],
                "resolution_details": {"error": str(e)},
                "disagreements_count": len(disagreements)
            }

    def calculate_score(self, evaluation: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """
        Step 4: Score Calculation
        根据统一的清单结果计算量化质量分数
        """
        total_final_score = 0
        max_possible_score = 0
        score_details = {}

        for category_name, indicators in evaluation.items():
            category_score = 0
            category_max_score = 0
            
            for indicator in indicators:
                indicator_id = indicator.get("id")
                indicator_score_earned = 0
                
                # 获取模板中对应指标的最大分数
                template_indicator = None
                for template_category in self.checklist_template.values():
                    for template_ind in template_category:
                        if template_ind["id"] == indicator_id:
                            template_indicator = template_ind
                            break
                    if template_indicator:
                        break
                
                if template_indicator:
                    max_indicator_score = template_indicator["score"]
                    
                    # 只有当指标适用于当前报告类型时，才计入最大可能得分
                    if not template_indicator.get("is_defect_specific") or report_type == "defect":
                        max_possible_score += max_indicator_score
                        category_max_score += max_indicator_score
                        
                        # 计算通过的检查点数量
                        checkpoints_passed = sum(1 for cp in indicator.get("checkpoints", []) if cp.get("value") == True)
                        total_checkpoints = len(indicator.get("checkpoints", []))
                        
                        if total_checkpoints > 0:
                            indicator_score_earned = (max_indicator_score * checkpoints_passed) / total_checkpoints
                        
                        total_final_score += indicator_score_earned
                        category_score += indicator_score_earned
                
                score_details[indicator_id] = {
                    "score_earned": round(indicator_score_earned, 2),
                    "max_score": max_indicator_score if template_indicator else 0,
                    "checkpoints_passed": sum(1 for cp in indicator.get("checkpoints", []) if cp.get("value") == True),
                    "total_checkpoints": len(indicator.get("checkpoints", []))
                }
            
            score_details[f"{category_name}_total"] = {
                "score_earned": round(category_score, 2),
                "max_score": category_max_score
            }

        total_final_score = round(total_final_score, 2)
        score_percentage = (total_final_score / max_possible_score) * 100 if max_possible_score > 0 else 0

        return {
            "total_score": total_final_score,
            "max_possible_score": max_possible_score,
            "score_percentage": round(score_percentage, 2),
            "score_details": score_details
        }

    def calculate_score_simplified(self, simplified_evaluation: Dict[str, Any], checkpoint_mapping: Dict[str, Dict], report_type: str) -> Dict[str, Any]:
        """
        根据简化格式的评估结果计算量化质量分数
        """
        total_final_score = 0
        max_possible_score = 0
        score_details = {}
        
        # 统计每个指标的得分
        indicator_stats = {}  # indicator_id -> {checkpoints_passed, total_checkpoints, max_score}
        
        for checkpoint_result in simplified_evaluation.get("checkpoint_results", []):
            checkpoint_id = checkpoint_result["checkpoint_id"]
            if checkpoint_id in checkpoint_mapping:
                mapping = checkpoint_mapping[checkpoint_id]
                indicator_id = mapping["indicator_id"]
                
                if indicator_id not in indicator_stats:
                    # 获取模板中对应指标的信息
                    template_indicator = None
                    for template_category in self.checklist_template.values():
                        for template_ind in template_category:
                            if template_ind["id"] == indicator_id:
                                template_indicator = template_ind
                                break
                        if template_indicator:
                            break
                    
                    if template_indicator:
                        indicator_stats[indicator_id] = {
                            "checkpoints_passed": 0,
                            "total_checkpoints": len(template_indicator["checkpoints"]),
                            "max_score": template_indicator["score"],
                            "is_applicable": not template_indicator.get("is_defect_specific") or report_type == "defect"
                        }
                
                # 统计通过的检查点
                if checkpoint_result.get("value") == True:
                    indicator_stats[indicator_id]["checkpoints_passed"] += 1
        
        # 计算每个指标的得分
        for indicator_id, stats in indicator_stats.items():
            if stats["is_applicable"]:
                max_indicator_score = stats["max_score"]
                checkpoints_passed = stats["checkpoints_passed"]
                total_checkpoints = stats["total_checkpoints"]
                
                max_possible_score += max_indicator_score
                
                if total_checkpoints > 0:
                    indicator_score_earned = (max_indicator_score * checkpoints_passed) / total_checkpoints
                    total_final_score += indicator_score_earned
                else:
                    indicator_score_earned = 0
                
                score_details[indicator_id] = {
                    "score_earned": round(indicator_score_earned, 2),
                    "max_score": max_indicator_score,
                    "checkpoints_passed": checkpoints_passed,
                    "total_checkpoints": total_checkpoints
                }

        total_final_score = round(total_final_score, 2)
        score_percentage = (total_final_score / max_possible_score) * 100 if max_possible_score > 0 else 0

        return {
            "total_score": total_final_score,
            "max_possible_score": max_possible_score,
            "score_percentage": round(score_percentage, 2),
            "score_details": score_details
        }

    def evaluate_report(self, report_data: dict) -> Dict[str, Any]:
        """
        完整的评估流程：根据配置选择使用单LLM或双LLM模式实现textual dimension assessment
        """
        report_id = report_data.get('test_case_id', report_data.get('defect_id', 'N/A'))
        print(f"开始评估报告: {report_id}")
        
        # 输出当前评估模式
        if ENABLE_DUAL_LLM_ASSESSMENT:
            print(f"  评估模式: 双LLM评估 (DeepSeek + Kimi)")
            if ENABLE_ARBITRATION:
                print(f"  仲裁模式: 开启 (GPT-4o)")
            else:
                print(f"  仲裁模式: 关闭")
        else:
            print(f"  评估模式: 单LLM评估 ({DEFAULT_SINGLE_LLM.upper()})")
        
        start_time = time.time()
        
        # Step 1: 生成简化版清单
        print("  生成简化评估清单...")
        prompt, checkpoint_mapping = self.generate_simplified_checklist(report_data)
        print(f"清单长度: {len(prompt)} characters")
        
        # Step 2: 根据配置选择评估方式
        if ENABLE_DUAL_LLM_ASSESSMENT:
            # 双LLM评估模式
            print("  执行双LLM评估（简化格式）...")
            deepseek_result, kimi_result = self.dual_llm_assessment_simplified(prompt)
            
            if not deepseek_result["success"] or not kimi_result["success"]:
                error_result = {
                    "success": False,
                    "error": "双LLM评估失败",
                    "report_id": report_id,
                    "report_type": report_data.get("type", "unknown"),
                    "deepseek_result": deepseek_result,
                    "kimi_result": kimi_result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "assessment_mode": "dual_llm"
                }
                # 保存失败结果
                self.save_evaluation_result(report_id, error_result)
                return error_result
            
            # Step 3: 识别分歧
            print("  识别和解决分歧...")
            disagreements = self.identify_disagreements_simplified(
                deepseek_result["evaluation"], 
                kimi_result["evaluation"]
            )
            
            # Step 4: 分歧解决
            if ENABLE_ARBITRATION:
                resolution_result = self.resolve_disagreements_simplified(
                    report_data, disagreements, deepseek_result, kimi_result
                )
            else:
                # 不使用仲裁，直接使用DeepSeek的结果
                resolution_result = {
                    "final_evaluation": deepseek_result["evaluation"],
                    "resolution_details": {"arbitration_disabled": True},
                    "disagreements_count": len(disagreements)
                }
            
            # 构建双LLM结果
            llm_results = {
                "deepseek": {
                    "model": DEEPSEEK_MODEL,
                    "success": deepseek_result["success"],
                    "evaluation": deepseek_result["evaluation"],  # 简化格式结果
                    "evaluation_full": self.convert_simplified_to_full_format(
                        deepseek_result["evaluation"], checkpoint_mapping
                    ),  # 完整格式结果
                    "raw_response": deepseek_result.get("raw_response"),
                    "response_length": len(deepseek_result.get("raw_response", "")),
                    "llm_name": deepseek_result.get("llm_name")
                },
                "kimi": {
                    "model": KIMI_MODEL,
                    "success": kimi_result["success"],
                    "evaluation": kimi_result["evaluation"],  # 简化格式结果
                    "evaluation_full": self.convert_simplified_to_full_format(
                        kimi_result["evaluation"], checkpoint_mapping
                    ),  # 完整格式结果
                    "raw_response": kimi_result.get("raw_response"),
                    "response_length": len(kimi_result.get("raw_response", "")),
                    "llm_name": kimi_result.get("llm_name")
                }
            }
            
        else:
            # 单LLM评估模式
            print(f"  执行单LLM评估（{DEFAULT_SINGLE_LLM.upper()}）...")
            single_result = self.single_llm_assessment_simplified(prompt, DEFAULT_SINGLE_LLM)
            
            if not single_result["success"]:
                error_result = {
                    "success": False,
                    "error": "单LLM评估失败",
                    "report_id": report_id,
                    "report_type": report_data.get("type", "unknown"),
                    "single_result": single_result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "assessment_mode": "single_llm",
                    "selected_llm": DEFAULT_SINGLE_LLM
                }
                # 保存失败结果
                self.save_evaluation_result(report_id, error_result)
                return error_result
            
            # 单LLM模式没有分歧，直接使用结果
            disagreements = []
            resolution_result = {
                "final_evaluation": single_result["evaluation"],
                "resolution_details": {"single_llm_mode": True},
                "disagreements_count": 0
            }
            
            # 构建单LLM结果
            llm_results = {
                DEFAULT_SINGLE_LLM: {
                    "model": DEEPSEEK_MODEL if DEFAULT_SINGLE_LLM == "deepseek" else KIMI_MODEL,
                    "success": single_result["success"],
                    "evaluation": single_result["evaluation"],  # 简化格式结果
                    "evaluation_full": self.convert_simplified_to_full_format(
                        single_result["evaluation"], checkpoint_mapping
                    ),  # 完整格式结果
                    "raw_response": single_result.get("raw_response"),
                    "response_length": len(single_result.get("raw_response", "")),
                    "llm_name": single_result.get("llm_name")
                }
            }
        
        # Step 5: 将简化格式转换为完整格式
        print("  转换为完整格式...")
        final_evaluation_full = self.convert_simplified_to_full_format(
            resolution_result["final_evaluation"], 
            checkpoint_mapping
        )
        
        # Step 6: 计算分数
        print("  计算最终分数...")
        report_type = report_data.get("type", "unknown")
        score_result = self.calculate_score_simplified(
            resolution_result["final_evaluation"], 
            checkpoint_mapping, 
            report_type
        )
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # 构建详细的评估结果
        detailed_result = {
            "success": True,
            "report_id": report_id,
            "report_type": report_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "assessment_mode": "dual_llm" if ENABLE_DUAL_LLM_ASSESSMENT else "single_llm",
            "arbitration_enabled": ENABLE_ARBITRATION if ENABLE_DUAL_LLM_ASSESSMENT else False,
            "selected_llm": DEFAULT_SINGLE_LLM if not ENABLE_DUAL_LLM_ASSESSMENT else None,
            
            # 最终结果
            "final_evaluation": final_evaluation_full,  # 返回完整格式给外部使用
            "score_result": score_result,
            "disagreements_count": resolution_result["disagreements_count"],
            "processing_time_seconds": processing_time,
            
            # LLM详细结果
            "llm_results": llm_results,
            
            # 分歧和解决详情
            "disagreements": disagreements if ENABLE_DUAL_LLM_ASSESSMENT else [],
            "resolution_details": resolution_result["resolution_details"],
            
            # 元数据
            "metadata": {
                "prompt_length": len(prompt),
                "checkpoint_count": len(checkpoint_mapping),
                "original_report_data": report_data
            }
        }
        
        # 保存评估结果
        self.save_evaluation_result(report_id, detailed_result)
        
        return detailed_result

    def evaluate_multiple_reports(self, reports_data: List[Dict], auto_save: bool = True, save_filename: str = None) -> List[Dict[str, Any]]:
        """
        批量评估多个报告，并自动保存结果
        
        Args:
            reports_data: 报告数据列表
            auto_save: 是否自动保存到文件
            save_filename: 保存文件名，如果为None则自动生成
        """
        results = []
        successful_count = 0
        failed_count = 0
        
        print(f"\n开始批量评估 {len(reports_data)} 个报告...")
        
        for i, report_data in enumerate(reports_data):
            print(f"\n--- 评估报告 {i+1}/{len(reports_data)} ---")
            result = self.evaluate_report(report_data)
            results.append(result)
            
            if result.get("success"):
                successful_count += 1
            else:
                failed_count += 1
        
        # 打印批量评估摘要
        print(f"\n=== 批量评估完成 ===")
        print(f"总报告数: {len(reports_data)}")
        print(f"成功评估: {successful_count}")
        print(f"失败评估: {failed_count}")
        
        # 获取详细摘要
        summary = self.get_evaluation_summary()
        if isinstance(summary, dict):
            print(f"平均得分: {summary.get('average_score', 0)}%")
            print(f"平均处理时间: {summary.get('average_processing_time', 0)}秒")
            print(f"总分歧数: {summary.get('disagreement_stats', {}).get('total', 0)}")
        
        # 自动保存结果
        if auto_save:
            saved_file = self.save_all_results_to_file(save_filename)
            if saved_file:
                print(f"评估结果已保存到: {saved_file}")
        
        return results

    def save_evaluation_result(self, report_id: str, result_data: Dict[str, Any]):
        """
        保存评估结果到内存中，按用例名区分
        """
        self.evaluation_results[report_id] = result_data
        print(f"已保存报告 {report_id} 的评估结果")

    def save_all_results_to_file(self, filename: str = None):
        """
        将所有评估结果保存到JSON文件
        """
        import os
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"textual_evaluation_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"所有评估结果已保存到文件: {filename}")
            return filename
        except Exception as e:
            print(f"保存文件时出错: {e}")
            return None

    def load_results_from_file(self, filename: str):
        """
        从JSON文件加载评估结果
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.evaluation_results = json.load(f)
            print(f"已从文件 {filename} 加载评估结果，共 {len(self.evaluation_results)} 个报告")
            return True
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return False

    def get_evaluation_summary(self):
        """
        获取评估结果摘要
        """
        if not self.evaluation_results:
            return "没有评估结果"
        
        summary = {
            "total_reports": len(self.evaluation_results),
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_score": 0,
            "average_processing_time": 0,
            "disagreement_stats": {"total": 0, "average": 0}
        }
        
        scores = []
        processing_times = []
        disagreements = []
        
        for report_id, result in self.evaluation_results.items():
            if result.get("success"):
                summary["successful_evaluations"] += 1
                if "score_result" in result:
                    scores.append(result["score_result"]["score_percentage"])
                if "processing_time_seconds" in result:
                    processing_times.append(result["processing_time_seconds"])
                if "disagreements_count" in result:
                    disagreements.append(result["disagreements_count"])
            else:
                summary["failed_evaluations"] += 1
        
        if scores:
            summary["average_score"] = round(sum(scores) / len(scores), 2)
        if processing_times:
            summary["average_processing_time"] = round(sum(processing_times) / len(processing_times), 2)
        if disagreements:
            summary["disagreement_stats"]["total"] = sum(disagreements)
            summary["disagreement_stats"]["average"] = round(sum(disagreements) / len(disagreements), 2)
        
        return summary

    def evaluate_tester_reports_with_sampling(self, tester_id: str, testcases: List[Dict], defects: List[Dict], 
                                              enable_sampling: bool = None, 
                                              testcase_sample_size: int = None, 
                                              defect_sample_size: int = None) -> Dict[str, Any]:
        """
        按测试人员分别进行文本性评估，支持抽样功能
        
        Args:
            tester_id (str): 测试人员ID
            testcases (List[Dict]): 该测试人员的所有测试用例
            defects (List[Dict]): 该测试人员的所有缺陷报告
            enable_sampling (bool, optional): 是否开启抽样评估，默认从config读取
            testcase_sample_size (int, optional): 测试用例抽样数量，默认从config读取
            defect_sample_size (int, optional): 缺陷抽样数量，默认从config读取
            
        Returns:
            Dict[str, Any]: 评估结果包含统计信息
        """
        import random
        
        # 使用配置的默认值
        if enable_sampling is None:
            enable_sampling = ENABLE_TEXTUAL_SAMPLING
        if testcase_sample_size is None:
            testcase_sample_size = TEXTUAL_SAMPLE_TESTCASE_SIZE
        if defect_sample_size is None:
            defect_sample_size = TEXTUAL_SAMPLE_DEFECT_SIZE
            
        print(f"开始评估测试人员 {tester_id} 的报告...")
        print(f"  - 测试用例总数: {len(testcases)}")
        print(f"  - 缺陷报告总数: {len(defects)}")
        print(f"  - 抽样评估: {'开启' if enable_sampling else '关闭'}")
        
        # 决定要评估的报告
        eval_testcases = testcases
        eval_defects = defects
        
        if enable_sampling:
            # 对测试用例进行抽样
            if len(testcases) > testcase_sample_size:
                eval_testcases = random.sample(testcases, testcase_sample_size)
                print(f"  - 测试用例抽样: {len(eval_testcases)} 个")
            else:
                print(f"  - 测试用例全部评估: {len(eval_testcases)} 个")
                
            # 对缺陷进行抽样
            if len(defects) > defect_sample_size:
                eval_defects = random.sample(defects, defect_sample_size)
                print(f"  - 缺陷抽样: {len(eval_defects)} 个")
            else:
                print(f"  - 缺陷全部评估: {len(eval_defects)} 个")
        else:
            print(f"  - 全部评估: {len(eval_testcases)} 个测试用例, {len(eval_defects)} 个缺陷")
        
        # 合并所有要评估的报告
        all_reports = eval_testcases + eval_defects
        
        if not all_reports:
            return {
                "tester_id": tester_id,
                "success": False,
                "error": "没有可评估的报告",
                "sampling_enabled": enable_sampling,
                "original_counts": {
                    "testcases": len(testcases),
                    "defects": len(defects)
                },
                "evaluated_counts": {
                    "testcases": len(eval_testcases),
                    "defects": len(eval_defects)
                }
            }
        
        # 执行评估
        evaluation_results = self.evaluate_multiple_reports(all_reports, auto_save=False)
        
        # 计算统计信息
        testcase_scores = []
        defect_scores = []
        successful_evaluations = 0
        failed_evaluations = 0
        total_disagreements = 0
        total_processing_time = 0
        
        for result in evaluation_results:
            if result.get("success"):
                successful_evaluations += 1
                score_percentage = result.get("score_result", {}).get("score_percentage", 0)
                report_type = result.get("report_type", "unknown")
                
                if report_type == "test_case":
                    testcase_scores.append(score_percentage)
                elif report_type == "defect":
                    defect_scores.append(score_percentage)
                    
                total_disagreements += result.get("disagreements_count", 0)
                total_processing_time += result.get("processing_time_seconds", 0)
            else:
                failed_evaluations += 1
        
        # 计算平均分
        testcase_avg_score = sum(testcase_scores) / len(testcase_scores) if testcase_scores else 0
        defect_avg_score = sum(defect_scores) / len(defect_scores) if defect_scores else 0
        overall_avg_score = (testcase_avg_score + defect_avg_score) / 2 if (testcase_scores or defect_scores) else 0
        
        # 构建结果
        tester_result = {
            "tester_id": tester_id,
            "success": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sampling_enabled": enable_sampling,
            "original_counts": {
                "testcases": len(testcases),
                "defects": len(defects),
                "total": len(testcases) + len(defects)
            },
            "evaluated_counts": {
                "testcases": len(eval_testcases),
                "defects": len(eval_defects),
                "total": len(eval_testcases) + len(eval_defects)
            },
            "evaluation_statistics": {
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
                "total_disagreements": total_disagreements,
                "total_processing_time": total_processing_time
            },
            "score_statistics": {
                "testcase_scores": testcase_scores,
                "defect_scores": defect_scores,
                "testcase_avg_score": round(testcase_avg_score, 2),
                "defect_avg_score": round(defect_avg_score, 2),
                "overall_avg_score": round(overall_avg_score, 2)
            },
            "detailed_results": evaluation_results
        }
        
        print(f"测试人员 {tester_id} 评估完成:")
        print(f"  - 成功评估: {successful_evaluations}/{len(all_reports)}")
        print(f"  - 测试用例平均分: {testcase_avg_score:.2f}%")
        print(f"  - 缺陷平均分: {defect_avg_score:.2f}%")
        print(f"  - 总体平均分: {overall_avg_score:.2f}%")
        
        return tester_result

def read_reports_from_excel(file_path: str, report_type: str) -> List[Dict]:
    """
    读取单个Excel文件,并将其中的报告(测试用例或缺陷)转换为标准字典格式.
    
    Args:
        file_path (str): Excel文件的完整路径.
        report_type (str): 报告类型,"test_case" 或 "defect".
    
    Returns:
        list[dict]: 包含文件中所有报告数据的字典列表.
    """
    reports_data = []
    try:
        df = pd.read_excel(file_path)
        df = df.fillna('')  # 将NaN值填充为空字符串

        col_mapping = EXCEL_COL_MAPPING_TESTCASE if report_type == "test_case" else EXCEL_COL_MAPPING_DEFECT

        for index, row in df.iterrows():
            report_item_data = {"type": report_type}  # 明确设置报告类型
            for excel_col, internal_key in col_mapping.items():
                # 使用str()转换，确保所有内容都是字符串，避免LLM处理数字或bool类型
                report_item_data[internal_key] = str(row.get(excel_col, '')) 
            
            # 对于测试用例，默认没有'实际结果'列，但在缺陷报告中需要
            # 同时确保'title'字段存在，如果Excel中用例名称为空，则使用用例描述
            if report_type == "test_case":
                if not str(report_item_data.get("title", '')).strip():
                    report_item_data["title"] = report_item_data.get("description", "")
                if 'actual_result' not in report_item_data: # 实际结果通常在测试结论中
                    report_item_data['actual_result'] = str(row.get('测试结论', '')) 
            elif report_type == "defect":
                 # 对于缺陷，如果缺陷描述为空，可以尝试使用缺陷界面标题
                if not str(report_item_data.get("description", '')).strip():
                    report_item_data["description"] = report_item_data.get("ui_title", "")

            reports_data.append(report_item_data)
            
    except FileNotFoundError:
        print(f"错误：文件未找到 - {file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")
    return reports_data

def view_saved_results(filename: str):
    """
    查看保存的评估结果文件的工具函数
    
    Args:
        filename: JSON结果文件名
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"=" * 60)
        print(f"查看评估结果文件: {filename}")
        print(f"=" * 60)
        print(f"总报告数: {len(data)}")
        print()
        
        successful_count = 0
        failed_count = 0
        total_scores = []
        total_disagreements = 0
        
        for report_id, result in data.items():
            print(f"报告ID: {report_id}")
            print(f"  类型: {result.get('report_type', 'N/A')}")
            print(f"  时间: {result.get('timestamp', 'N/A')}")
            print(f"  成功: {result.get('success', False)}")
            
            if result.get('success'):
                successful_count += 1
                score_result = result.get('score_result', {})
                score_percentage = score_result.get('score_percentage', 0)
                total_scores.append(score_percentage)
                
                disagreements = result.get('disagreements_count', 0)
                total_disagreements += disagreements
                
                print(f"  得分: {score_result.get('total_score', 0)}/{score_result.get('max_possible_score', 0)} ({score_percentage}%)")
                print(f"  分歧: {disagreements}")
                print(f"  处理时间: {result.get('processing_time_seconds', 0)}秒")
                
                # 显示两个LLM的状态
                llm_results = result.get('llm_results', {})
                deepseek = llm_results.get('deepseek', {})
                kimi = llm_results.get('kimi', {})
                
                print(f"  DeepSeek: {'✓' if deepseek.get('success') else '✗'} ({deepseek.get('model', 'N/A')})")
                print(f"  Kimi: {'✓' if kimi.get('success') else '✗'} ({kimi.get('model', 'N/A')})")
                
            else:
                failed_count += 1
                print(f"  错误: {result.get('error', 'N/A')}")
            
            print()
        
        # 显示统计摘要
        print("=" * 60)
        print("统计摘要:")
        print(f"  成功评估: {successful_count}")
        print(f"  失败评估: {failed_count}")
        
        if total_scores:
            print(f"  平均得分: {sum(total_scores)/len(total_scores):.2f}%")
            print(f"  最高得分: {max(total_scores):.2f}%")
            print(f"  最低得分: {min(total_scores):.2f}%")
        
        print(f"  总分歧数: {total_disagreements}")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
    except json.JSONDecodeError:
        print(f"错误: 文件 {filename} 不是有效的JSON格式")
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    # 创建示例数据
    sample_report = {
        "type": "test_case",
        "test_case_id": "TC001",
        "title": "用户登录功能测试",
        "description": "测试用户使用正确的用户名和密码进行登录",
        "preconditions": "用户已注册账号",
        "environmental_info": "Android 12, Chrome 95",
        "steps_to_reproduce": "1. 打开登录页面\n2. 输入用户名\n3. 输入密码\n4. 点击登录按钮",
        "expected_result": "用户成功登录并跳转到主页面",
        "attachments": "screenshot.png"
    }
    
    print("=" * 60)
    print("开始测试Textual Dimension Assessment Agent...")
    print("=" * 60)
    
    # 显示当前配置
    print(f"当前配置:")
    print(f"  - 双LLM评估: {'开启' if ENABLE_DUAL_LLM_ASSESSMENT else '关闭'}")
    print(f"  - 仲裁功能: {'开启' if ENABLE_ARBITRATION else '关闭'}")
    print(f"  - 默认单LLM: {DEFAULT_SINGLE_LLM.upper()}")
    print()
    
    # 示例使用
    agent = TextualDimensionAssessmentAgent()
    
    print("开始评估示例报告...")
    # result = agent.evaluate_report(sample_report)
    
    # if result.get("success"):
    #     print(f"\n=== 评估完成 ===")
    #     print(f"报告ID: {result['report_id']}")
    #     print(f"评估模式: {result['assessment_mode']}")
    #     print(f"总分: {result['score_result']['total_score']}/{result['score_result']['max_possible_score']}")
    #     print(f"得分率: {result['score_result']['score_percentage']}%")
    #     print(f"分歧数量: {result['disagreements_count']}")
    #     print(f"处理时间: {result['processing_time_seconds']}秒")
    #     print(f"提示词长度: {result['metadata']['prompt_length']} characters")
    #     print(f"检查点数量: {result['metadata']['checkpoint_count']}")
        
    #     # 显示LLM的详细结果
    #     print(f"\n=== LLM评估结果 ===")
    #     for llm_name, llm_data in result['llm_results'].items():
    #         print(f"{llm_name.upper()} ({llm_data['model']}):")
    #         print(f"  - 成功: {llm_data['success']}")
    #         print(f"  - 响应长度: {llm_data['response_length']} 字符")
    #         print(f"  - 检查点结果数量: {len(llm_data['evaluation'].get('checkpoint_results', []))}")
        
    #     # 保存单个评估结果到文件
    #     saved_file = agent.save_all_results_to_file("single_test_result.json")
    #     if saved_file:
    #         print(f"\n单个测试结果已保存到: {saved_file}")
        
    #     # 显示评估摘要
    #     print(f"\n=== 评估摘要 ===")
    #     summary = agent.get_evaluation_summary()
    #     print(json.dumps(summary, indent=2, ensure_ascii=False))
        
    # else:
    #     print(f"评估失败: {result.get('error')}")
    
    # # 演示不同配置模式
    # print(f"\n{'='*60}")
    # print("演示不同配置模式...")
    # print(f"{'='*60}")
    
    # 创建多个示例报告
    sample_reports = [
        {
            "type": "test_case",
            "test_case_id": "TC002",
            "title": "用户注册功能测试",
            "description": "测试新用户注册流程",
            "preconditions": "应用已安装",
            "environmental_info": "iOS 15, Safari",
            "steps_to_reproduce": "1. 打开注册页面\n2. 填写信息\n3. 点击注册",
            "expected_result": "注册成功",
            "attachments": ""
        },
        {
            "type": "defect",
            "defect_id": "DE001",
            "title": "登录按钮无响应",
            "description": "点击登录按钮后无任何反应",
            "steps_to_reproduce": "1. 输入用户名密码\n2. 点击登录按钮",
            "actual_result": "按钮无响应",
            "attachments_screenshot": "bug_screenshot.png"
        }
    ]

    reports_data = read_reports_from_excel("data/app1/testcases/1.xlsx", "test_case")
    print(reports_data)
    # # 批量评估
    # batch_results = agent.evaluate_multiple_reports(sample_reports, auto_save=True, save_filename="batch_test_results.json")
    
    # print(f"\n批量评估完成，共处理 {len(batch_results)} 个报告")

    # 演示查看保存的结果
    view_saved_results("batch_test_results.json")
    
    # 展示配置说明
    print(f"\n{'='*60}")
    print("配置说明:")
    print(f"{'='*60}")
    print("在config.py中可以配置以下选项:")
    print("1. ENABLE_DUAL_LLM_ASSESSMENT: 是否开启双LLM评估")
    print("   - True: 使用DeepSeek和Kimi两个LLM并行评估")
    print("   - False: 使用单个LLM进行评估")
    print()
    print("2. ENABLE_ARBITRATION: 是否开启仲裁功能")
    print("   - True: 当双LLM有分歧时，使用GPT-4o进行仲裁")
    print("   - False: 有分歧时直接使用DeepSeek的结果")
    print("   - 只在双LLM模式下有效")
    print()
    print("3. DEFAULT_SINGLE_LLM: 单LLM模式下使用的默认LLM")
    print("   - 'deepseek': 使用DeepSeek模型")
    print("   - 'kimi': 使用Kimi模型")
    print()
    print("推荐配置:")
    print("- 生产环境/高质量要求: ENABLE_DUAL_LLM_ASSESSMENT=True, ENABLE_ARBITRATION=True")
    print("- 开发/测试环境: ENABLE_DUAL_LLM_ASSESSMENT=False, DEFAULT_SINGLE_LLM='deepseek'")
    print("- 成本敏感场景: ENABLE_DUAL_LLM_ASSESSMENT=True, ENABLE_ARBITRATION=False")
    print("=" * 60) 
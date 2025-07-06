#!/usr/bin/env python3
"""
主调度器 - 运行三个维度的智能体评估
对指定app的所有测试人员提交的文档进行充分性、文本质量和竞争性评估
"""

import os
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
import traceback

# 导入三个智能体
from adequacy_agent import AdequacyAgent
from textual_agent import TextualDimensionAssessmentAgent, read_reports_from_excel
from competitive_agent import CompetitiveAgent, load_tester_reports_from_excel

def ensure_directory_exists(directory_path):
    """确保目录存在，如果不存在则创建"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def read_requirements_document(requirements_file_path):
    """读取需求文档内容"""
    try:
        with open(requirements_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"警告：需求文档 {requirements_file_path} 不存在")
        return ""
    except Exception as e:
        print(f"读取需求文档时出错: {e}")
        return ""

def get_excel_files_count(directory_path):
    """获取目录中Excel文件的数量"""
    if not os.path.exists(directory_path):
        return 0
    excel_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx') and f[0].isdigit()]
    return len(excel_files)

def read_all_test_cases_from_directory(directory_path):
    """从目录中读取所有测试用例文件"""
    all_test_cases = []
    if not os.path.exists(directory_path):
        print(f"警告：测试用例目录 {directory_path} 不存在")
        return all_test_cases
    
    excel_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx') and f[0].isdigit()]
    excel_files.sort(key=lambda x: int(x.split('.')[0]))  # 按数字排序
    
    for file_name in excel_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            print(f"  读取测试用例文件: {file_name}")
            test_cases = read_reports_from_excel(file_path, "test_case")
            # 为每个测试用例添加文件来源信息
            for tc in test_cases:
                tc['source_file'] = file_name
                tc['tester_id'] = file_name.split('.')[0]
            all_test_cases.extend(test_cases)
        except Exception as e:
            print(f"  读取测试用例文件 {file_name} 时出错: {e}")
    
    return all_test_cases

def read_all_defects_from_directory(directory_path):
    """从目录中读取所有缺陷文件"""
    all_defects = []
    if not os.path.exists(directory_path):
        print(f"警告：缺陷目录 {directory_path} 不存在")
        return all_defects
    
    excel_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx') and f[0].isdigit()]
    excel_files.sort(key=lambda x: int(x.split('.')[0]))  # 按数字排序
    
    for file_name in excel_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            print(f"  读取缺陷文件: {file_name}")
            defects = read_reports_from_excel(file_path, "defect")
            # 为每个缺陷添加文件来源信息
            for defect in defects:
                defect['source_file'] = file_name
                defect['tester_id'] = file_name.split('.')[0]
            all_defects.extend(defects)
        except Exception as e:
            print(f"  读取缺陷文件 {file_name} 时出错: {e}")
    
    return all_defects

def group_test_cases_by_tester(test_cases):
    """按测试人员分组测试用例"""
    tester_groups = {}
    for tc in test_cases:
        tester_id = tc.get('tester_id', 'unknown')
        if tester_id not in tester_groups:
            tester_groups[tester_id] = []
        tester_groups[tester_id].append(tc)
    return tester_groups

def run_adequacy_assessment(requirements_text, test_cases, temp_dir, results_dir):
    """运行充分性评估 - 分别评估每个测试人员的覆盖率"""
    print("\n" + "="*60)
    print("1. 开始充分性评估 (Adequacy Assessment)")
    print("="*60)
    
    try:
        # 初始化充分性评估智能体
        adequacy_agent = AdequacyAgent()
        
        print(f"需求文档长度: {len(requirements_text)} 字符")
        print(f"测试用例总数: {len(test_cases)}")
        
        # 步骤1：构建需求树（共同的）
        print("\n🌳 步骤1：构建需求树...")
        requirement_tree = adequacy_agent.analyze_requirements_to_tree(requirements_text)
        if not requirement_tree:
            print("❌ 需求树构建失败")
            return {"error": "Failed to build requirement tree."}
        
        # 立即保存需求树到temp目录
        tree_file = os.path.join(temp_dir, "requirement_tree.json")
        with open(tree_file, 'w', encoding='utf-8') as f:
            json.dump(adequacy_agent.requirement_tree, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 需求树构建完成!")
        print(f"📊 需求树统计: {len(requirement_tree)} 个节点, {len(adequacy_agent.get_leaf_nodes())} 个叶子节点")
        print(f"💾 需求树已保存到: {tree_file}")
        
        # 步骤2：按测试人员分组测试用例
        print("\n👥 步骤2：按测试人员分组测试用例...")
        tester_groups = group_test_cases_by_tester(test_cases)
        print(f"找到 {len(tester_groups)} 位测试人员")
        
        # 步骤3：分别评估每个测试人员的覆盖率
        print("\n📊 步骤3：分别评估每个测试人员的覆盖率...")
        tester_coverage_reports = {}
        
        for tester_id, tester_test_cases in tester_groups.items():
            print(f"\n🔍 评估测试人员 {tester_id} (共{len(tester_test_cases)}个测试用例)...")
            
            # 映射该测试人员的测试用例
            mapped_test_cases = adequacy_agent.map_test_cases_to_tree(tester_test_cases, requirement_tree)
            if not mapped_test_cases and tester_test_cases:
                print(f"  警告：测试人员 {tester_id} 的测试用例没有成功映射")
            
            # 评估该测试人员的覆盖率
            coverage_report = adequacy_agent.evaluate_coverage(requirement_tree, mapped_test_cases)
            coverage_report["tester_id"] = tester_id
            coverage_report["test_cases_count"] = len(tester_test_cases)
            
            tester_coverage_reports[tester_id] = coverage_report
            
            print(f"  测试人员 {tester_id} 覆盖率: {coverage_report.get('coverage_percentage', 'N/A')}")
        
        # 保存每个测试人员的覆盖率报告
        adequacy_result_file = os.path.join(results_dir, "adequacy_assessment_result.json")
        with open(adequacy_result_file, 'w', encoding='utf-8') as f:
            json.dump(tester_coverage_reports, f, ensure_ascii=False, indent=2)
        
        # 生成总体统计摘要
        summary = {
            "total_testers": len(tester_groups),
            "requirement_tree_stats": {
                "total_nodes": len(requirement_tree),
                "leaf_nodes": len(adequacy_agent.get_leaf_nodes())
            },
            "coverage_stats": {
                "average_coverage": 0,
                "max_coverage": 0,
                "min_coverage": 100,
                "coverage_distribution": {}
            }
        }
        
        if tester_coverage_reports:
            coverages = []
            for tester_id, report in tester_coverage_reports.items():
                coverage_str = report.get('coverage_percentage', '0%')
                coverage_num = float(coverage_str.replace('%', ''))
                coverages.append(coverage_num)
                summary["coverage_stats"]["coverage_distribution"][tester_id] = coverage_num
            
            summary["coverage_stats"]["average_coverage"] = sum(coverages) / len(coverages)
            summary["coverage_stats"]["max_coverage"] = max(coverages)
            summary["coverage_stats"]["min_coverage"] = min(coverages)
        
        # 保存总体统计摘要
        summary_file = os.path.join(results_dir, "adequacy_assessment_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 充分性评估完成!")
        print(f"📊 总体统计:")
        print(f"  - 测试人员数量: {summary['total_testers']}")
        print(f"  - 平均覆盖率: {summary['coverage_stats']['average_coverage']:.2f}%")
        print(f"  - 最高覆盖率: {summary['coverage_stats']['max_coverage']:.2f}%")
        print(f"  - 最低覆盖率: {summary['coverage_stats']['min_coverage']:.2f}%")
        print(f"📄 详细结果已保存到: {adequacy_result_file}")
        print(f"📊 统计摘要已保存到: {summary_file}")
        
        return tester_coverage_reports
        
    except Exception as e:
        print(f"❌ 充分性评估失败: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def run_textual_assessment(test_cases, defects, temp_dir, results_dir):
    """运行文本质量评估 - 按测试人员分别评估"""
    print("\n" + "="*60)
    print("2. 开始文本质量评估 (Textual Assessment)")
    print("="*60)
    
    try:
        # 初始化文本质量评估智能体
        textual_agent = TextualDimensionAssessmentAgent()
        
        print(f"测试用例数: {len(test_cases)}")
        print(f"缺陷报告数: {len(defects)}")
        
        # 按测试人员分组数据
        print("\n👥 按测试人员分组数据...")
        tester_testcases = {}
        tester_defects = {}
        
        # 分组测试用例
        for tc in test_cases:
            tester_id = tc.get('tester_id', 'unknown')
            if tester_id not in tester_testcases:
                tester_testcases[tester_id] = []
            tester_testcases[tester_id].append(tc)
        
        # 分组缺陷
        for defect in defects:
            tester_id = defect.get('tester_id', 'unknown')
            if tester_id not in tester_defects:
                tester_defects[tester_id] = []
            tester_defects[tester_id].append(defect)
        
        # 获取所有测试人员
        all_tester_ids = set(tester_testcases.keys()) | set(tester_defects.keys())
        print(f"找到 {len(all_tester_ids)} 位测试人员")
        
        # 分别评估每个测试人员
        tester_evaluations = {}
        overall_stats = {
            "total_testers": len(all_tester_ids),
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "testcase_scores": [],
            "defect_scores": [],
            "overall_scores": []
        }
        
        for tester_id in sorted(all_tester_ids):
            tester_testcases_list = tester_testcases.get(tester_id, [])
            tester_defects_list = tester_defects.get(tester_id, [])
            
            print(f"\n🔍 评估测试人员 {tester_id}:")
            print(f"  - 测试用例: {len(tester_testcases_list)} 个")
            print(f"  - 缺陷报告: {len(tester_defects_list)} 个")
            
            # 使用新的抽样评估方法
            tester_result = textual_agent.evaluate_tester_reports_with_sampling(
                tester_id, 
                tester_testcases_list, 
                tester_defects_list
            )
            
            tester_evaluations[tester_id] = tester_result
            
            # 更新总体统计
            if tester_result.get("success"):
                overall_stats["successful_evaluations"] += 1
                score_stats = tester_result.get("score_statistics", {})
                
                if score_stats.get("testcase_avg_score", 0) > 0:
                    overall_stats["testcase_scores"].append(score_stats["testcase_avg_score"])
                if score_stats.get("defect_avg_score", 0) > 0:
                    overall_stats["defect_scores"].append(score_stats["defect_avg_score"])
                if score_stats.get("overall_avg_score", 0) > 0:
                    overall_stats["overall_scores"].append(score_stats["overall_avg_score"])
            else:
                overall_stats["failed_evaluations"] += 1
        
        # 计算总体平均分
        overall_stats["avg_testcase_score"] = sum(overall_stats["testcase_scores"]) / len(overall_stats["testcase_scores"]) if overall_stats["testcase_scores"] else 0
        overall_stats["avg_defect_score"] = sum(overall_stats["defect_scores"]) / len(overall_stats["defect_scores"]) if overall_stats["defect_scores"] else 0
        overall_stats["avg_overall_score"] = sum(overall_stats["overall_scores"]) / len(overall_stats["overall_scores"]) if overall_stats["overall_scores"] else 0
        
        # 保存评估结果
        final_result_file = os.path.join(results_dir, "textual_assessment_result.json")
        with open(final_result_file, 'w', encoding='utf-8') as f:
            json.dump(tester_evaluations, f, ensure_ascii=False, indent=2)
        
        # 保存总体统计摘要
        summary_file = os.path.join(results_dir, "textual_assessment_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 文本质量评估完成!")
        print(f"📊 总体统计:")
        print(f"  - 测试人员数量: {overall_stats['total_testers']}")
        print(f"  - 成功评估: {overall_stats['successful_evaluations']}")
        print(f"  - 失败评估: {overall_stats['failed_evaluations']}")
        print(f"  - 测试用例平均分: {overall_stats['avg_testcase_score']:.2f}%")
        print(f"  - 缺陷平均分: {overall_stats['avg_defect_score']:.2f}%")
        print(f"  - 总体平均分: {overall_stats['avg_overall_score']:.2f}%")
        print(f"📁 结果已保存到: {final_result_file}")
        
        return tester_evaluations
        
    except Exception as e:
        print(f"❌ 文本质量评估失败: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def run_competitive_assessment(defects_dir, temp_dir, results_dir):
    """运行竞争性评估"""
    print("\n" + "="*60)
    print("3. 开始竞争性评估 (Competitive Assessment)")
    print("="*60)
    
    try:
        # 初始化竞争性评估智能体
        competitive_agent = CompetitiveAgent()
        
        # 获取缺陷文件数量
        excel_files_count = get_excel_files_count(defects_dir)
        print(f"缺陷文件数量: {excel_files_count}")
        
        if excel_files_count == 0:
            print("警告：没有找到缺陷文件，跳过竞争性评估")
            return {"error": "没有找到缺陷文件"}
        
        # 加载测试人员报告数据
        tester_reports_data = load_tester_reports_from_excel(defects_dir, excel_files_count)
        print(f"加载了 {len(tester_reports_data)} 位测试人员的缺陷数据")
        
        # 执行竞争性评估
        evaluation_results, cluster_tree_root = competitive_agent.evaluate_reports(tester_reports_data)
        
        # 保存评估结果
        competitive_result_file = os.path.join(results_dir, "competitive_assessment_result.json")
        with open(competitive_result_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        # 保存聚类树到temp目录
        if cluster_tree_root:
            tree_file = os.path.join(temp_dir, "defect_cluster_tree.json")
            with open(tree_file, 'w', encoding='utf-8') as f:
                json.dump(cluster_tree_root.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"✅ 竞争性评估完成!")
        print(f"评估了 {len(evaluation_results)} 位测试人员")
        if evaluation_results:
            avg_score = sum(r.get('competitive_score', 0) for r in evaluation_results) / len(evaluation_results)
            print(f"平均竞争性得分: {avg_score:.2f}")
        print(f"结果已保存到: {competitive_result_file}")
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ 竞争性评估失败: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def generate_summary_report(adequacy_result, textual_result, competitive_result, results_dir):
    """生成综合摘要报告"""
    print("\n" + "="*60)
    print("生成综合摘要报告")
    print("="*60)
    
    # 处理新的充分性评估结果结构（按测试人员分组）
    adequacy_summary = {
        "status": "success" if "error" not in adequacy_result else "failed",
        "error": adequacy_result.get("error")
    }
    
    if "error" not in adequacy_result and adequacy_result:
        # 从充分性评估摘要文件中读取统计信息
        adequacy_summary_file = os.path.join(results_dir, "adequacy_assessment_summary.json")
        if os.path.exists(adequacy_summary_file):
            try:
                with open(adequacy_summary_file, 'r', encoding='utf-8') as f:
                    adequacy_stats = json.load(f)
                adequacy_summary.update(adequacy_stats)
            except Exception as e:
                print(f"读取充分性评估摘要时出错: {e}")
        
        # 如果是字典格式（按测试人员分组），计算一些基本统计
        if isinstance(adequacy_result, dict) and not adequacy_result.get("error"):
            adequacy_summary["tester_count"] = len(adequacy_result)
            
            # 计算覆盖率统计
            coverages = []
            for tester_id, tester_report in adequacy_result.items():
                if isinstance(tester_report, dict) and "coverage_percentage" in tester_report:
                    coverage_str = tester_report.get("coverage_percentage", "0%")
                    coverage_num = float(coverage_str.replace("%", ""))
                    coverages.append(coverage_num)
            
            if coverages:
                adequacy_summary["coverage_stats"] = {
                    "average_coverage": sum(coverages) / len(coverages),
                    "max_coverage": max(coverages),
                    "min_coverage": min(coverages)
                }
    
    summary_report = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "adequacy_assessment": adequacy_summary,
        "textual_assessment": {
            "status": "success" if "error" not in textual_result else "failed",
            "error": textual_result.get("error")
        },
        "competitive_assessment": {
            "status": "success" if "error" not in competitive_result else "failed",
            "error": competitive_result.get("error")
        }
    }
    
    # 添加文本质量评估的详细信息
    if "error" not in textual_result and textual_result:
        try:
            # 读取文本质量评估摘要
            summary_file = os.path.join(results_dir, "textual_assessment_summary.json")
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    textual_summary = json.load(f)
                summary_report["textual_assessment"].update(textual_summary)
            
            # 如果是按测试人员分组的结果，添加测试人员统计信息
            if isinstance(textual_result, dict):
                summary_report["textual_assessment"]["tester_count"] = len(textual_result)
                
                # 计算测试人员平均分统计
                tester_scores = []
                for tester_id, tester_result in textual_result.items():
                    if tester_result.get("success") and "score_statistics" in tester_result:
                        overall_score = tester_result["score_statistics"].get("overall_avg_score", 0)
                        if overall_score > 0:
                            tester_scores.append(overall_score)
                
                if tester_scores:
                    summary_report["textual_assessment"]["tester_score_stats"] = {
                        "average_score": sum(tester_scores) / len(tester_scores),
                        "max_score": max(tester_scores),
                        "min_score": min(tester_scores)
                    }
                    
        except Exception as e:
            print(f"读取文本质量评估摘要时出错: {e}")
    
    # 添加竞争性评估的详细信息
    if "error" not in competitive_result and competitive_result:
        if isinstance(competitive_result, list):
            summary_report["competitive_assessment"]["total_testers"] = len(competitive_result)
            if competitive_result:
                avg_score = sum(r.get('competitive_score', 0) for r in competitive_result) / len(competitive_result)
                summary_report["competitive_assessment"]["average_competitive_score"] = round(avg_score, 2)
                max_score = max(r.get('competitive_score', 0) for r in competitive_result)
                summary_report["competitive_assessment"]["max_competitive_score"] = max_score
    
    # 保存综合摘要
    summary_file = os.path.join(results_dir, "comprehensive_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"📊 综合摘要报告已保存到: {summary_file}")
    return summary_report

def main(app_name="app1"):
    """主函数 - 调度所有智能体进行评估"""
    print("🚀 开始运行三维度智能体评估系统")
    print(f"目标应用: {app_name}")
    print("="*80)
    
    start_time = time.time()
    
    # 设置路径
    app_base_dir = f"data/{app_name}"
    testcases_dir = os.path.join(app_base_dir, "testcases")
    defects_dir = os.path.join(app_base_dir, "defects")
    requirements_file = os.path.join("data", "test_requirements.txt")  # 需求文档在data根目录
    
    results_dir = os.path.join(app_base_dir, "results")
    temp_dir = os.path.join(app_base_dir, "temp")
    
    # 确保输出目录存在
    ensure_directory_exists(results_dir)
    ensure_directory_exists(temp_dir)
    
    print(f"📁 结果目录: {results_dir}")
    print(f"📁 临时目录: {temp_dir}")
    
    # 检查必要的目录和文件是否存在
    if not os.path.exists(app_base_dir):
        print(f"❌ 错误: 应用目录 {app_base_dir} 不存在")
        return
    
    # 读取需求文档
    print(f"\n📖 读取需求文档: {requirements_file}")
    requirements_text = read_requirements_document(requirements_file)
    if not requirements_text:
        print("⚠️  警告: 需求文档为空或读取失败，充分性评估可能受影响")
    
    # 读取测试用例
    print(f"\n📋 读取测试用例目录: {testcases_dir}")
    test_cases = read_all_test_cases_from_directory(testcases_dir)
    print(f"共加载 {len(test_cases)} 个测试用例")
    
    # 读取缺陷报告
    print(f"\n🐛 读取缺陷目录: {defects_dir}")
    defects = read_all_defects_from_directory(defects_dir)
    print(f"共加载 {len(defects)} 个缺陷报告")
    
    # 执行三个维度的评估
    adequacy_result = {"error": "跳过充分性评估"}  # 占位符
    # adequacy_result = run_adequacy_assessment(requirements_text, test_cases, temp_dir, results_dir)
    competitive_result = run_competitive_assessment(defects_dir, temp_dir, results_dir)
    textual_result = {"error": "跳过文本质量评估"}  # 占位符
    # textual_result = run_textual_assessment(test_cases, defects, temp_dir, results_dir)
    
    # 生成综合摘要报告
    # summary_report = generate_summary_report(adequacy_result, textual_result, competitive_result, results_dir)
    
    # 计算总耗时
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    
    print("\n" + "="*80)
    print("🎉 三维度评估完成!")
    print("="*80)
    print(f"⏱️  总耗时: {total_time} 秒")
    print(f"📊 充分性评估: {'⏭️ 已跳过' if adequacy_result.get('error') == '跳过充分性评估' else ('✅' if adequacy_result.get('error') is None else '❌')}")
    print(f"📝 文本质量评估: {'⏭️ 已跳过' if textual_result.get('error') == '跳过文本质量评估' else ('✅' if not isinstance(textual_result, dict) or textual_result.get('error') is None else '❌')}")
    print(f"🏆 竞争性评估: {'✅' if not isinstance(competitive_result, dict) or competitive_result.get('error') is None else '❌'}")
    print(f"📁 所有结果已保存到: {results_dir}")
    print("="*80)

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数指定app名称
    app_name = sys.argv[1] if len(sys.argv) > 1 else "app1"
    
    try:
        main(app_name)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了评估过程")
    except Exception as e:
        print(f"\n❌ 评估过程中发生未预期的错误: {e}")
        traceback.print_exc() 
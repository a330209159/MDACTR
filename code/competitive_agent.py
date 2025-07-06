import re
import os # 用于文件路径操作
import pandas as pd # 用于读取Excel
from openai import OpenAI
# import uuid # No longer needed for node IDs
import json # For saving results and tree to JSON
from config import OPENAI_API_KEY, OPENAI_BASE_URL, GPT4O_MODEL, TEMPERATURE, TOP_P
from prompts import SYSTEM_PROMPT_DEFECT_CLUSTERING, get_defect_clustering_prompt

class TreeNode:
    def __init__(self, node_id, name, level, parent=None):
        self.node_id = node_id
        self.name = name
        self.level = level
        self.parent = parent
        self.children = []
        self.reports = [] # 存储直接关联到此节点的缺陷报告ID

    def add_child(self, child_node):
        self.children.append(child_node)

    def add_reports(self, report_ids):
        self.reports.extend(report_ids)

    def __repr__(self, level_indent=0): # Changed parameter name for clarity
        ret = "\t" * level_indent + f"LEVEL {self.level}: {self.name} (ID: {self.node_id})"
        if self.reports:
            ret += " REPORTS: " + ",".join(self.reports)
        ret += "\n"
        for child in self.children:
            ret += child.__repr__(level_indent + 1)
        return ret

    def to_dict(self):
        """Converts the tree node and its children to a dictionary for JSON serialization."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "level": self.level,
            "reports": list(self.reports),
            "children": [child.to_dict() for child in self.children]
        }

class CompetitiveAgent:
    def __init__(self, api_key=None, model=None, temperature=None, top_p=None, max_score=10.0):
        """
        初始化竞争性评价智能体。
        论文中competitive dimension使用gpt-4o-2024-05-13作为base LLM，temperature=0.1, top_p=0.9
        
        :param api_key: OpenAI API 密钥。如果为 None，则从config.py读取。
        :param model: 要使用的 OpenAI 模型。如果为 None，则使用config.py中的GPT4O_MODEL。
        :param temperature: LLM 生成文本时的温度参数。如果为 None，则使用config.py中的TEMPERATURE。
        :param top_p: LLM top_p参数。如果为 None，则使用config.py中的TOP_P。
        :param max_score: 最大得分。
        """
        try:
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
            else:
                self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            
            self.model_name = model if model is not None else GPT4O_MODEL
            self.temperature = temperature if temperature is not None else TEMPERATURE
            self.top_p = top_p if top_p is not None else TOP_P
            self.max_score = max_score
            self._node_id_counter = 0 # Initialize node ID counter
        except Exception as e:
            print(f"初始化OpenAI客户端时出错: {e}")
            self.client = None

    def _prepare_defect_descriptions_for_llm(self, all_defects_list: list) -> str:
        descriptions = []
        for defect in all_defects_list:
            descriptions.append(
                f"缺陷ID: {defect['defect_id']}\n"
                f"标题: {defect['title']}\n"
                f"复现步骤: {defect.get('steps', 'N/A')}\n"
                f"实际结果: {defect.get('actual_result', 'N/A')}\n"
                "---"
            )
        return "\n".join(descriptions)

    def _construct_llm_prompt_for_tree(self, defect_descriptions_str: str) -> str:
        return get_defect_clustering_prompt(defect_descriptions_str)

    def _get_tree_clusters_from_llm(self, all_defects_list: list) -> str:
        if not self.client:
            print("LLM客户端未初始化。")
            return ""

        defect_descriptions_str = self._prepare_defect_descriptions_for_llm(all_defects_list)
        prompt = self._construct_llm_prompt_for_tree(defect_descriptions_str)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_DEFECT_CLUSTERING},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"LLM API调用期间出错: {e}")
            return ""

    def _parse_llm_tree_response(self, llm_response_text: str) -> TreeNode | None:
        if not llm_response_text:
            return None

        self._node_id_counter = 0 # Reset counter for each new tree parsing
        root_node_id = f"N{self._node_id_counter}"
        self._node_id_counter += 1
        root = TreeNode(node_id=root_node_id, name="所有缺陷", level=0)
        
        path = [root]
        level_pattern = re.compile(r"^(\s*)LEVEL\s+(\d+):\s*(.+)$")
        reports_pattern = re.compile(r"^(\s*)REPORTS:\s*(.+)$")
        lines = llm_response_text.strip().split('\n')

        for line_number, line in enumerate(lines):
            if not line.strip():
                continue
            level_match = level_pattern.match(line)
            reports_match = reports_pattern.match(line)

            if level_match:
                # indentation = len(level_match.group(1)) # Not strictly used for parent finding now
                level = int(level_match.group(2))
                name = level_match.group(3).strip()
                
                node_id = f"N{self._node_id_counter}"
                self._node_id_counter += 1
                current_node = TreeNode(node_id=node_id, name=name, level=level)

                while path and path[-1].level >= current_node.level:
                    path.pop()
                
                if not path: # Should not happen if root is correctly initialized in path
                    if current_node.level == 1:
                        parent_node = root
                    else:
                        print(f"警告 (行 {line_number+1}): 无法为节点 '{name}' (Level {level}) 找到父节点。已跳过。原始行: '{line}'")
                        continue
                else:
                    parent_node = path[-1]

                if parent_node.level != current_node.level - 1:
                    if current_node.level > parent_node.level :
                        print(f"提示 (行 {line_number+1}): 节点 '{name}' (Level {level}) 的父节点 '{parent_node.name}' (Level {parent_node.level}) 层级不完全匹配 ({current_node.level -1} expected for parent), 但仍尝试连接。")
                    else: 
                        print(f"警告 (行 {line_number+1}): 节点 '{name}' (Level {level}) 的父节点 '{parent_node.name}' (Level {parent_node.level}) 层级不匹配。父级应为 {current_node.level-1}。已跳过。原始行: '{line}'")
                        continue
                
                parent_node.add_child(current_node)
                current_node.parent = parent_node
                path.append(current_node)

            elif reports_match:
                if not path or path[-1] == root and root.level !=0 : # Reports should belong to a specific cluster, not the absolute root unless level 0 has reports
                     # The second condition path[-1] == root is a bit tricky if root can have reports.
                     # Let's refine: reports must belong to a node that is not the placeholder root, unless that root is a valid level itself.
                     # Simpler: if current path top is root and root has no children yet, it's an issue.
                     # Best: If path top's level is 0 (our main root), reports shouldn't be directly under it unless the LLM outputs LEVEL 0 reports.
                     # The prompt asks for LEVEL 1 onwards.
                    if not path or path[-1].level == 0 : # Reports should not be directly under the absolute root if tree has levels
                        print(f"警告 (行 {line_number+1}): REPORTS行 '{line.strip()}' 在根节点或无效父LEVEL节点下。已跳过。")
                        continue

                current_level_node = path[-1]
                reports_str = reports_match.group(2).strip()
                report_ids = [r_id.strip() for r_id in reports_str.split(',') if r_id.strip()]
                current_level_node.add_reports(report_ids)
            else:
                print(f"警告 (行 {line_number+1}): 无法解析行: '{line}'")
        
        if not root.children:
            print("警告: LLM响应已收到，但未能构建聚类树的任何子节点。")
            # return None # Allow returning root even if empty, for saving structure
        return root
        
    def _get_path_string(self, node: TreeNode):
        path_names = []
        curr = node
        while curr and curr.parent : #不包括根节点"所有缺陷"的名字 (node_id 'N0')
            path_names.append(curr.name)
            curr = curr.parent
        return " -> ".join(reversed(path_names))

    def _find_scorable_nodes(self, node: TreeNode, scorable_nodes_map: dict, all_defect_ids_in_tree: set):
        if node.reports: # This node itself is a unique bug classification point
            scorable_nodes_map[node.node_id] = {
                "node_id": node.node_id,
                "name": node.name,
                "reports": list(node.reports), # Ensure it's a list
                "level": node.level,
                "path_from_root": self._get_path_string(node)
            }
            all_defect_ids_in_tree.update(node.reports)

        for child in node.children:
            self._find_scorable_nodes(child, scorable_nodes_map, all_defect_ids_in_tree)

    def _score_unique_bugs_from_tree(self, cluster_tree_root: TreeNode | None, original_defect_count: int) -> tuple[dict, dict, set, dict]:
        unique_bug_scores = {} # node_id -> score
        defect_to_unique_bug_map = {} # defect_id -> node_id
        scorable_nodes_details = {} # node_id -> full details for scorable nodes
        all_defect_ids_in_tree = set()
        
        if not cluster_tree_root:
            return unique_bug_scores, defect_to_unique_bug_map, all_defect_ids_in_tree, scorable_nodes_details

        # scorable_nodes_map is now scorable_nodes_details, populated by _find_scorable_nodes
        self._find_scorable_nodes(cluster_tree_root, scorable_nodes_details, all_defect_ids_in_tree)
        
        print(f"从树中识别出 {len(scorable_nodes_details)} 个可评分的缺陷节点。这些节点共包含 {len(all_defect_ids_in_tree)} 个原始缺陷ID。")
        if original_defect_count > len(all_defect_ids_in_tree):
            print(f"警告: 原始缺陷总数为 {original_defect_count}, 但只有 {len(all_defect_ids_in_tree)} 个被分配到聚类树中。有些缺陷可能未被LLM处理。")

        processed_defect_ids = set()
        for node_id, node_data in scorable_nodes_details.items():
            count_ui = len(node_data["reports"])
            if count_ui > 0:
                score = self.max_score / count_ui
                unique_bug_scores[node_id] = score
                for defect_id in node_data["reports"]:
                    if defect_id in processed_defect_ids:
                        print(f"警告: 缺陷ID {defect_id} 被分配到多个缺陷节点。当前节点: '{node_data['name']}' (ID:{node_id}). 这可能影响评分准确性。")
                    if defect_id not in defect_to_unique_bug_map: 
                        defect_to_unique_bug_map[defect_id] = node_id
                    processed_defect_ids.add(defect_id)
            else:
                # This case should not happen if scorable_nodes_details only includes nodes with reports.
                # If it can, then these nodes are not directly scorable unless they are parents.
                # Based on current _find_scorable_nodes, this else block is unlikely to be hit for nodes in scorable_nodes_details.
                unique_bug_scores[node_id] = 0.0 
        
        return unique_bug_scores, defect_to_unique_bug_map, all_defect_ids_in_tree, scorable_nodes_details


    def evaluate_reports(self, tester_reports_data: list) -> tuple[list, TreeNode | None]:
        if not self.client:
            print("LLM客户端未初始化。无法评估报告。")
            return [], None

        all_defects_list = []
        # original_defect_id_to_data_map = {} # Currently not used elsewhere, can be omitted if not needed
        for tester_report in tester_reports_data:
            for defect in tester_report.get("defects_found", []):
                all_defects_list.append(defect)
                # original_defect_id_to_data_map[defect['defect_id']] = defect

        if not all_defects_list:
            print("在任何报告中均未发现缺陷。无需评估。")
            return [], None

        print(f"总共加载 {len(all_defects_list)} 个缺陷条目。正在从LLM获取缺陷树...")
        llm_response_text = self._get_tree_clusters_from_llm(all_defects_list)
        
        # --- 模拟LLM树状输出 (用于测试) ---
        # print("跳过LLM调用，使用模拟LLM树状响应进行解析测试。")
        # llm_response_text = """
        # LEVEL 1: 登录模块缺陷
        #   LEVEL 2: 登录时崩溃 (常见)
        #     REPORTS: file1_0, file2_0, file3_0
        #   LEVEL 2: 忘记密码流程问题
        #     LEVEL 3: 验证码接收失败
        #       REPORTS: file4_0
        # LEVEL 1: 个人资料模块问题
        #   REPORTS: file1_1
        #   LEVEL 2: 头像上传问题
        #     LEVEL 3: 大文件上传失败 (特定场景)
        #       REPORTS: file2_1, file3_1
        #     LEVEL 3: 图像裁剪功能异常
        #       REPORTS: file3_2
        # """
        # --- 模拟结束 ---

        if not llm_response_text:
            print("未能从LLM获取响应或响应为空。")
            error_results = [{"tester_id": tr["tester_id"], "report_id": tr["report_id"], "competitive_score": 0, "contributed_unique_bugs_details": {}, "error": "LLM通信失败"} for tr in tester_reports_data]
            return error_results, None

        print("收到LLM响应。正在解析缺陷树...")
        cluster_tree_root = self._parse_llm_tree_response(llm_response_text)

        if not cluster_tree_root : # Allow empty root for saving structure
            print("未能从LLM响应中解析出有效的缺陷树结构或树为空。分数将为零。")
            error_results = [{"tester_id": tr["tester_id"], "report_id": tr["report_id"], "competitive_score": 0, "contributed_unique_bugs_details": {}, "error": "解析LLM聚类树响应失败或树为空"} for tr in tester_reports_data]
            if cluster_tree_root: # if root node itself exists but has no children
                 return error_results, cluster_tree_root 
            return error_results, None
        
        if not cluster_tree_root.children:
             print("警告: 解析后的缺陷树根节点没有子节点。请检查LLM输出或解析逻辑。")


        print("正在从树中提取和评分独特缺陷...")
        unique_bug_node_scores, defect_to_unique_bug_node_map, all_defect_ids_in_tree, scorable_nodes_details = \
            self._score_unique_bugs_from_tree(cluster_tree_root, len(all_defects_list))
        
        print("正在为每个测试人员的报告计算分数...")
        evaluation_results = []
        for tester_report in tester_reports_data:
            current_tester_total_score = 0.0
            unique_bugs_contributed_by_tester_details = {} # node_id -> details

            for defect in tester_report.get("defects_found", []):
                defect_id = defect["defect_id"]
                if defect_id not in all_defect_ids_in_tree:
                    # print(f"信息: 测试员 {tester_report['tester_id']} 的缺陷 {defect_id} ('{defect.get('title', '')}') 未在LLM聚类树中找到，不计分。")
                    continue 

                scorable_node_id = defect_to_unique_bug_node_map.get(defect_id)

                if scorable_node_id and scorable_node_id not in unique_bugs_contributed_by_tester_details:
                    bug_score = unique_bug_node_scores.get(scorable_node_id, 0.0)
                    current_tester_total_score += bug_score
                    
                    node_info = scorable_nodes_details.get(scorable_node_id, {"name": "未知缺陷类别", "path_from_root": "未知路径"})
                    
                    contributed_original_ids_for_this_node = [
                        d["defect_id"] for d in tester_report["defects_found"] 
                        if defect_to_unique_bug_node_map.get(d["defect_id"]) == scorable_node_id
                    ]

                    unique_bugs_contributed_by_tester_details[scorable_node_id] = {
                        "score_contribution": round(bug_score, 2), # Renamed for clarity
                        "unique_bug_name": node_info["name"],
                        "unique_bug_path": node_info["path_from_root"],
                        "original_defect_ids_contributed_by_tester": contributed_original_ids_for_this_node,
                        "total_original_reports_in_this_unique_bug_node": len(node_info.get("reports",[]))
                    }
            
            evaluation_results.append({
                "tester_id": tester_report["tester_id"],
                "report_id": tester_report["report_id"],
                "competitive_score": round(current_tester_total_score, 2),
                "contributed_unique_bugs_details": unique_bugs_contributed_by_tester_details
            })
        
        print("评估完成。")
        return evaluation_results, cluster_tree_root

# --- 数据加载函数 ---
def load_tester_reports_from_excel(base_path: str, num_files: int) -> list:
    COL_ID_SOURCE = '用例编号' 
    COL_DESC = '缺陷描述'    
    COL_TYPE = '缺陷类型'    
    COL_PRECONDITION = '前置条件' 
    COL_UI_TITLE = '缺陷界面标题' 
    COL_STEPS = '操作步骤'    
    COL_ENV = '环境信息'      
    COL_INPUT_DATA = '输入数据' 
    COL_EXPECTED = '预期结果'  
    COL_ACTUAL = '实际结果'    
    COL_REPORT_TIME = '报告填写时间' 
    COL_REPORTER = '提交人员' 
    COL_SCREENSHOT = '缺陷界面截图'

    defects_by_tester = {} 

    for i in range(1, num_files + 1):
        file_name = f"{i}.xlsx"
        file_path = os.path.join(base_path, file_name)
        file_sequence_number = str(i)  # 使用文件序号作为测试人员ID

        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，已跳过。")
            continue

        try:
            print(f"正在读取文件: {file_path}...")
            df = pd.read_excel(file_path)

            # 只要求必要的列，不强制要求 COL_REPORTER
            required_cols = [COL_DESC, COL_STEPS, COL_ACTUAL]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"警告: 文件 {file_path} 缺失以下必要列: {', '.join(missing_cols)}。已跳过此文件。")
                continue

            # 初始化该测试人员的缺陷列表
            if file_sequence_number not in defects_by_tester:
                defects_by_tester[file_sequence_number] = []

            for index, row in df.iterrows():
                # 检查必要字段是否有内容
                desc = str(row.get(COL_DESC, 'N/A') or 'N/A').strip()
                steps = str(row.get(COL_STEPS, 'N/A') or 'N/A').strip()
                actual_result = str(row.get(COL_ACTUAL, 'N/A') or 'N/A').strip()
                
                # 如果关键字段都为空，跳过这一行
                if desc in ['N/A', ''] and steps in ['N/A', ''] and actual_result in ['N/A', '']:
                    continue

                defect_id = f"file{file_sequence_number}_{index}"
                
                defect_data = {
                    "defect_id": defect_id,
                    "title": desc,
                    "steps": steps,
                    "actual_result": actual_result
                }

                defects_by_tester[file_sequence_number].append(defect_data)

        except Exception as e:
            print(f"读取或处理文件 {file_path} 时出错: {e}")

    tester_reports_data = []
    for tester_id, defects_list in defects_by_tester.items():
        if defects_list: 
            report_id = f"TR_tester_{tester_id}" 
            tester_reports_data.append({
                "tester_id": tester_id,  # 现在是文件序号，如"1", "2", "3"等
                "report_id": report_id,
                "defects_found": defects_list
            })
    
    print(f"数据加载完成。共处理了 {len(defects_by_tester)} 位测试人员的数据。")
    return tester_reports_data


# --- 主程序示例 ---
if __name__ == '__main__':
    # 使用config.py中的API配置和模型设置（gpt-4o-2024-05-13, temperature=0.1, top_p=0.9）
    
    EXCEL_FILES_PATH = "data/app1/defects/" 
    NUMBER_OF_EXCEL_FILES = 18 # 示例用18个文件，请根据实际情况修改

    OUTPUT_EVALUATION_RESULTS_FILE = "evaluation_results.json"
    OUTPUT_TREE_STRUCTURE_FILE = "defect_cluster_tree.json"

    print("正在初始化 CompetitiveAgent...")
    agent = CompetitiveAgent()  # 使用config.py中的默认配置

    if not agent.client:
        print("智能体客户端初始化失败。正在退出示例。")
    else:
        print(f"智能体已初始化，模型: {agent.model_name}, 温度: {agent.temperature}, top_p: {agent.top_p}, 最高分数: {agent.max_score}")

        print(f"\n正在从 '{EXCEL_FILES_PATH}' 加载缺陷数据...")
        tester_reports_data_from_files = load_tester_reports_from_excel(EXCEL_FILES_PATH, NUMBER_OF_EXCEL_FILES)

        if not tester_reports_data_from_files:
            print("未能从Excel文件加载任何有效的测试报告数据。正在退出。")
        else:
            print(f"\n成功加载了 {len(tester_reports_data_from_files)} 个测试人员的报告数据。开始评估...")
            results, final_tree_root = agent.evaluate_reports(tester_reports_data_from_files)

            print("\n--- 评估结果 ---")
            if results:
                for res in results:
                    print(
                        f"测试员: {res['tester_id']}, 报告ID: {res['report_id']}, "
                        f"总分数: {res['competitive_score']}"
                    )
                    if "contributed_unique_bugs_details" in res and res["contributed_unique_bugs_details"]:
                        print("  贡献的独特缺陷详情:")
                        for node_id_key, details in res["contributed_unique_bugs_details"].items(): # node_id is the key
                            print(f"    - 独特缺陷节点ID: {node_id_key}") # Corrected to use the key
                            print(f"      名称: {details['unique_bug_name']}")
                            print(f"      路径: {details['unique_bug_path']}")
                            print(f"      此节点总原始报告数: {details['total_original_reports_in_this_unique_bug_node']}")
                            print(f"      本测试员贡献的原始报告ID: {details['original_defect_ids_contributed_by_tester']}")
                            print(f"      贡献分数: {details['score_contribution']:.2f}")
                    elif "error" in res:
                        print(f"  错误: {res['error']}")
                    print("-" * 20)
                
                # 保存评估结果到JSON文件
                try:
                    with open(OUTPUT_EVALUATION_RESULTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
                    print(f"\n评估结果已保存到: {OUTPUT_EVALUATION_RESULTS_FILE}")
                except Exception as e:
                    print(f"保存评估结果到JSON文件时出错: {e}")
            else:
                print("评估未产生结果。")
            
            # 保存树结构到JSON文件
            if final_tree_root:
                print("\n解析后的聚类树结构 (根节点和直接子节点):")
                print(f"LEVEL {final_tree_root.level}: {final_tree_root.name} (ID: {final_tree_root.node_id})")
                for child in final_tree_root.children:
                     print(f"\tLEVEL {child.level}: {child.name} (ID: {child.node_id}) REPORTS: {child.reports if child.reports else '[]'}")
                
                try:
                    tree_dict = final_tree_root.to_dict()
                    with open(OUTPUT_TREE_STRUCTURE_FILE, 'w', encoding='utf-8') as f:
                        json.dump(tree_dict, f, ensure_ascii=False, indent=4)
                    print(f"\n缺陷聚类树结构已保存到: {OUTPUT_TREE_STRUCTURE_FILE}")
                except Exception as e:
                    print(f"保存树结构到JSON文件时出错: {e}")
            else:
                print("\n未能生成有效的树结构，无法保存。")

            print("-------------------------")
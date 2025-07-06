import os
import json
import re
import uuid
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, GPT4O_MODEL, TEMPERATURE, TOP_P
from prompts import (
    SYSTEM_PROMPT_REQUIREMENT_STRUCTURE, 
    SYSTEM_PROMPT_REQUIREMENT_DECOMPOSE,
    SYSTEM_PROMPT_TEST_CASE_MAPPING,
    get_requirement_structure_user_prompt,
    get_requirement_decompose_user_prompt,
    get_test_case_mapping_user_prompt
)

# --- OpenAI API 配置 ---
# 从config.py导入配置，确保与论文方法一致

class AdequacyAgent:
    def __init__(self, api_key=None, model=None, temperature=None, top_p=None):
        """
        初始化充分性评价智能体。
        论文中adequacy dimension使用gpt-4o-2024-05-13作为base LLM，temperature=0.1, top_p=0.9
        
        :param api_key: OpenAI API 密钥。如果为 None，则从config.py读取。
        :param model: 要使用的 OpenAI 模型。如果为 None，则使用config.py中的GPT4O_MODEL。
        :param temperature: LLM 生成文本时的温度参数。如果为 None，则使用config.py中的TEMPERATURE。
        :param top_p: LLM top_p参数。如果为 None，则使用config.py中的TOP_P。
        """
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        
        self.model = model if model is not None else GPT4O_MODEL
        self.temperature = temperature if temperature is not None else TEMPERATURE
        self.top_p = top_p if top_p is not None else TOP_P
        self.requirement_tree = {} # 使用字典存储需求树，键为 node_id

    def _call_openai_api(self, prompt_messages, expecting_json=False):
        """
        调用 OpenAI ChatCompletion API 的辅助函数。
        按照论文设置使用temperature=0.1和top_p=0.9以确保一致和可重现的结果
        
        :param prompt_messages: 一个消息列表，符合 OpenAI API 格式 (e.g., [{"role": "system", ...}, {"role": "user", ...}]).
        :param expecting_json: 如果为 True，则尝试将响应解析为 JSON。
        :return: LLM 的响应内容字符串，或者解析后的 JSON 对象。
        """
        try:
            print("\n--- Calling OpenAI API ---")
            # print(f"Prompt Messages for LLM: {json.dumps(prompt_messages, ensure_ascii=False, indent=2)}") # 调试时可以取消注释
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=prompt_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=4000,  # 增加token限制以支持复杂的需求分析
                response_format={"type": "json_object"} if expecting_json else None
            )
            response_content = completion.choices[0].message.content
            print("--- API Response Received ---")
            # print(f"Raw LLM Response: {response_content}") # 调试时可以取消注释

            if expecting_json:
                try:
                    return json.loads(response_content)
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse LLM response as JSON. Error: {e}")
                    print(f"LLM Raw Output length: {len(response_content)} characters")
                    print(f"LLM Raw Output preview: {response_content[:500]}...")
                    
                    # 尝试从markdown代码块中提取JSON
                    match = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL)
                    if match:
                        try:
                            print("Attempting to parse extracted JSON block...")
                            return json.loads(match.group(1))
                        except json.JSONDecodeError as e_inner:
                            print(f"Error: Failed to parse extracted JSON block. Error: {e_inner}")
                    
                    # 尝试修复被截断的JSON
                    try:
                        # 找到最后一个完整的JSON对象或数组
                        truncated_content = response_content.rstrip()
                        if truncated_content.endswith(','):
                            # 移除末尾的逗号
                            truncated_content = truncated_content[:-1]
                        
                        # 尝试补全缺失的括号
                        open_braces = truncated_content.count('{') - truncated_content.count('}')
                        open_brackets = truncated_content.count('[') - truncated_content.count(']')
                        
                        if open_braces > 0 or open_brackets > 0:
                            print(f"Attempting to fix truncated JSON (missing {open_braces} braces, {open_brackets} brackets)...")
                            fixed_content = truncated_content + '}' * open_braces + ']' * open_brackets
                            return json.loads(fixed_content)
                    except json.JSONDecodeError:
                        print("Failed to fix truncated JSON")
                    
                    return {"error": "Failed to parse JSON", "details": response_content[:1000] + "..." if len(response_content) > 1000 else response_content}
            return response_content
        except Exception as e:
            print(f"An error occurred while calling OpenAI API: {e}")
            if expecting_json:
                return {"error": str(e)}
            return f"Error: {e}"

    def _add_node_to_tree(self, node_id, description, parent_id, children_ids, is_leaf, path):
        """向需求树中添加或更新节点"""
        self.requirement_tree[node_id] = {
            "node_id": node_id,
            "description": description,
            "parent_id": parent_id,
            "children_ids": children_ids if children_ids else [],
            "is_leaf": is_leaf,
            "path": path
        }
        if parent_id and parent_id in self.requirement_tree:
            if node_id not in self.requirement_tree[parent_id]["children_ids"]:
                self.requirement_tree[parent_id]["children_ids"].append(node_id)


    def analyze_requirements_to_tree(self, requirements_document_text):
        """
        阶段一：分析需求文档，提取并构建需求树。
        叶子节点是原子级别的测试功能点。
        """
        self.requirement_tree = {} # 重置需求树
        print("\n=== Phase 1: Analyzing Requirements and Building Tree ===")

        user_prompt_structure = get_requirement_structure_user_prompt(requirements_document_text)
        prompt_messages_structure = [
            {"role": "system", "content": SYSTEM_PROMPT_REQUIREMENT_STRUCTURE},
            {"role": "user", "content": user_prompt_structure}
        ]
        
        initial_analysis_result = self._call_openai_api(prompt_messages_structure, expecting_json=True)

        if not initial_analysis_result or "error" in initial_analysis_result or "potential_nodes" not in initial_analysis_result:
            print("Error: Failed to get initial structural analysis from LLM.")
            print(f"LLM response for initial analysis: {initial_analysis_result}")
            return None
        
        print("Initial structural analysis received.")
        potential_nodes = initial_analysis_result.get("potential_nodes", [])
        if not potential_nodes:
            print("Warning: LLM returned no potential_nodes from initial analysis.")
            return None

        nodes_to_process_for_decomposition = []
        for pn in potential_nodes:
            nodes_to_process_for_decomposition.append({
                "original_description": pn.get("description"),
                "original_temp_id": pn.get("temp_id", str(uuid.uuid4())) #确保有ID
            })
        
        if not nodes_to_process_for_decomposition:
            print("Warning: No potential nodes identified for decomposition step.")
            # Fallback: try to build tree from whatever initial analysis gave.
            # This part needs a robust tree building logic even for non-decomposed nodes.
            # For now, returning None if this critical step has no input.
            return None

        user_prompt_decompose = get_requirement_decompose_user_prompt(nodes_to_process_for_decomposition)
        prompt_messages_decompose = [
            {"role": "system", "content": SYSTEM_PROMPT_REQUIREMENT_DECOMPOSE},
            {"role": "user", "content": user_prompt_decompose}
        ]

        decomposition_result = self._call_openai_api(prompt_messages_decompose, expecting_json=True)

        if not decomposition_result or "error" in decomposition_result or "refined_node_map" not in decomposition_result:
            print("Error: Failed to get decomposition result (refined_node_map) from LLM.")
            print(f"LLM response for decomposition: {decomposition_result}")
            return None
        
        print("Decomposition and granularity refinement received.")
        refined_node_map = decomposition_result.get("refined_node_map", {})

        # 步骤 3: 构建最终的需求树
        # 此处树的构建逻辑：
        # 1. 遍历 initial_analysis_result.potential_nodes 来确定层级和父子关系。
        # 2. 当一个 potential_node 对应到 refined_node_map 中的条目时，
        #    如果它被拆分了，那么原始的 potential_node 成为一个中间父节点，
        #    其拆分出的 refined_node_item 成为它的子叶子节点。
        # 3. 如果 potential_node 没有被拆分（或refined_node_map中对应的列表只有一个元素），
        #    并且它在 initial_analysis_result 中被认为是较深层级的，那么它可能直接成为一个叶子节点。

        temp_id_to_node_id_map = {} # 映射 temp_id 到最终的 node_id
        node_counter = 1 # 用于生成唯一的 node_id

        # 首先创建所有节点的基本信息，并处理被拆分的节点
        processed_potential_nodes = []
        for pn_data in potential_nodes:
            temp_id = pn_data.get("temp_id")
            original_description = pn_data.get("description")
            
            node_id = temp_id_to_node_id_map.get(temp_id)
            if not node_id:
                node_id = f"N{node_counter}"
                node_counter += 1
                temp_id_to_node_id_map[temp_id] = node_id

            children_info = [] # 存储子节点的 (id, description)
            is_current_node_a_leaf = True # 默认为叶子，除非它有拆分的子节点

            if temp_id in refined_node_map:
                decomposed_items = refined_node_map[temp_id]
                if len(decomposed_items) > 1 or (len(decomposed_items) == 1 and decomposed_items[0].get("description") != original_description):
                    # 节点被拆分了，或者其描述被显著细化了，原始节点作为父节点
                    is_current_node_a_leaf = False 
                    for i, item in enumerate(decomposed_items):
                        child_node_id = f"{node_id}.{i+1}" # 子节点的ID
                        child_description = item.get("description")
                        children_info.append({"node_id": child_node_id, "description": child_description, "is_leaf": True, "parent_id": node_id})
                elif len(decomposed_items) == 1: # 未拆分，但可能确认是原子
                     is_current_node_a_leaf = decomposed_items[0].get("is_atomic_leaf", True)
            else: # 没有在refined_node_map中找到，可能LLM认为它已经是原子的或不需要处理
                is_current_node_a_leaf = not pn_data.get("needs_further_decomposition", True)


            processed_potential_nodes.append({
                "node_id": node_id,
                "description": original_description,
                "potential_parent_description": pn_data.get("potential_parent_description"),
                "level": pn_data.get("level", 0),
                "is_leaf_candidate": is_current_node_a_leaf, # 这是候选状态
                "decomposed_children_info": children_info # 存储拆分出的子节点信息
            })

        # 按层级排序，尝试构建父子关系
        processed_potential_nodes.sort(key=lambda x: x["level"])
        
        description_to_final_node_id_map = {}

        for node_entry in processed_potential_nodes:
            current_node_id = node_entry["node_id"]
            current_description = node_entry["description"]
            parent_desc_hint = node_entry.get("potential_parent_description")
            is_leaf = node_entry["is_leaf_candidate"] # 初始叶子状态
            decomposed_children = node_entry["decomposed_children_info"]

            parent_id = None
            if parent_desc_hint and parent_desc_hint in description_to_final_node_id_map:
                parent_id = description_to_final_node_id_map[parent_desc_hint]

            # 确定路径
            path_parts = []
            temp_parent_id = parent_id
            current_path_segment = re.sub(r'[^\w\s-]', '', current_description.split(':')[0]).replace(' ', '_')[:30] # 清理路径段

            while temp_parent_id and temp_parent_id in self.requirement_tree:
                parent_desc_segment = re.sub(r'[^\w\s-]', '', self.requirement_tree[temp_parent_id]["description"].split(':')[0]).replace(' ', '_')[:30]
                path_parts.insert(0, parent_desc_segment)
                temp_parent_id = self.requirement_tree[temp_parent_id]["parent_id"]
            path_parts.append(current_path_segment)
            path = "/".join(path_parts) if path_parts else current_path_segment


            if decomposed_children: # 如果有拆分出的子节点，当前节点不是叶子
                is_leaf = False
            
            self._add_node_to_tree(current_node_id, current_description, parent_id, [], is_leaf, path)
            description_to_final_node_id_map[current_description] = current_node_id # 用于后续查找父节点

            # 添加拆分出的子叶子节点
            for child_info in decomposed_children:
                child_desc_clean = re.sub(r'[^\w\s-]', '', child_info['description'].split(':')[0]).replace(' ', '_')[:30]
                child_path = f"{path}/{child_desc_clean}"
                self._add_node_to_tree(
                    child_info["node_id"], 
                    child_info["description"], 
                    current_node_id, # 父ID是当前节点
                    [], 
                    True, # 拆分出的子节点总是叶子
                    child_path
                )
        
        # 后处理：确保父节点的 is_leaf 和 children_ids 正确
        for nid, node_data in list(self.requirement_tree.items()):
            if node_data["parent_id"] and node_data["parent_id"] in self.requirement_tree:
                parent_node = self.requirement_tree[node_data["parent_id"]]
                if nid not in parent_node["children_ids"]: # 确保子ID在父节点的children_ids列表中
                    parent_node["children_ids"].append(nid)
                if parent_node["is_leaf"]: # 如果父节点之前被错误地标记为叶子，修正它
                    parent_node["is_leaf"] = False
        
        # 再次确认叶子状态：如果一个节点没有孩子，它就是叶子 (除非它是根且没孩子，但我们一般有内容)
        for nid, node_data in self.requirement_tree.items():
            if not node_data["children_ids"]:
                node_data["is_leaf"] = True
            else: # 如果有孩子，就不是叶子
                node_data["is_leaf"] = False

        print("Requirement tree construction attempt finished.")
        if not self.requirement_tree:
            print("Warning: Requirement tree is empty after processing.")
            return None
            
        print(f"Final tree has {len(self.requirement_tree)} nodes.")
        # for node_id, node_details in self.requirement_tree.items():
        #      print(f"  Node {node_id} ('{node_details['description']}' Path: {node_details['path']}): Parent: {node_details['parent_id']}, Children: {len(node_details['children_ids'])}, IsLeaf: {node_details['is_leaf']}")
        
        return self.requirement_tree


    def get_leaf_nodes(self):
        """从构建的需求树中获取所有叶子节点"""
        if not self.requirement_tree:
            return []
        leaf_nodes = []
        for node_id, node_details in self.requirement_tree.items():
            if node_details.get("is_leaf"):
                leaf_nodes.append(node_details)
        return leaf_nodes

    def map_test_cases_to_tree(self, test_cases, requirement_tree):
        """
        阶段二：将测试用例映射到需求树的叶子节点。
        :param test_cases: 测试用例列表。每个测试用例是一个字典，
                           至少包含 'ID', 'Case Name', 和 'Description'。
                           例如: [{"ID": "TC001", "Case Name": "Login Success", "Description": "Verify..."}]
        :param requirement_tree: analyze_requirements_to_tree 方法生成的需求树。
        :return: 一个字典，键是测试用例ID，值是该测试用例覆盖的叶子节点路径(path)列表。
                 例如: {"TC001": ["R1/R1.1/R1.1.1_leaf_0"]}
        """
        if not requirement_tree:
            print("Error: Requirement tree is not available for mapping.")
            return {}
        self.requirement_tree = requirement_tree 

        leaf_nodes = self.get_leaf_nodes()
        if not leaf_nodes:
            print("Error: No leaf nodes found in the requirement tree for mapping.")
            return {}

        print(f"\n=== Phase 2: Mapping Test Cases to {len(leaf_nodes)} Leaf Nodes ===")

        leaf_nodes_for_prompt = []
        for leaf in leaf_nodes:
            leaf_nodes_for_prompt.append({
                "path": leaf.get("path", leaf.get("node_id")), 
                "description": leaf.get("description")
            })

        mapped_results = {}

        for tc_index, test_case in enumerate(test_cases):
            tc_id = test_case.get('test_case_id', test_case.get('ID', f'unknown_tc_{tc_index}'))
            tc_name = test_case.get('title', test_case.get('Case Name', ''))
            tc_description = test_case.get('description', test_case.get('Description', ''))

            # 组合用例名和描述作为提供给LLM的文本
            test_case_content_for_llm = f"Test Case ID: {tc_id}\nCase Name: {tc_name}\nDescription: {tc_description}"
            
            print(f"Mapping test case {tc_index + 1}/{len(test_cases)}: {tc_id}")
            
            user_prompt_mapping = get_test_case_mapping_user_prompt(leaf_nodes_for_prompt, test_case_content_for_llm, tc_id)

            prompt_messages_mapping = [
                {"role": "system", "content": SYSTEM_PROMPT_TEST_CASE_MAPPING},
                {"role": "user", "content": user_prompt_mapping}
            ]

            mapping_response = self._call_openai_api(prompt_messages_mapping, expecting_json=True)

            if mapping_response and "error" not in mapping_response:
                # 确保使用从输入test_case中获取的ID作为键
                response_tc_id = mapping_response.get("test_case_id", tc_id) 
                if response_tc_id != tc_id: # LLM 可能修改了ID，我们用原始的
                    print(f"  Warning: LLM returned test_case_id '{response_tc_id}', using original '{tc_id}'.")

                covered_paths = mapping_response.get("covered_leaf_node_paths", [])
                mapped_results[tc_id] = covered_paths
                # print(f"  TC {tc_id} mapped to: {covered_paths}. Reasoning: {mapping_response.get('reasoning')}")
            else:
                print(f"  Warning: Failed to map test case {tc_id}. LLM Response: {mapping_response}")
                mapped_results[tc_id] = []
        
        print("Test case mapping finished.")
        return mapped_results

    def evaluate_coverage(self, requirement_tree, mapped_test_cases):
        """
        阶段三：基于映射结果评估测试覆盖率。
        :param requirement_tree: 需求树。
        :param mapped_test_cases: map_test_cases_to_tree 方法的输出。
        :return: 一个包含覆盖率统计信息的字典。
        """
        if not requirement_tree:
            print("Error: Requirement tree is not available for coverage evaluation.")
            return {}
        self.requirement_tree = requirement_tree

        leaf_nodes = self.get_leaf_nodes()
        if not leaf_nodes:
            print("Error: No leaf nodes found for coverage evaluation.")
            return {"error": "No leaf nodes in tree"}

        print("\n=== Phase 3: Evaluating Test Coverage ===")

        total_leaf_nodes = len(leaf_nodes)
        covered_leaf_node_paths_set = set()

        for tc_id, paths in mapped_test_cases.items():
            for path in paths:
                covered_leaf_node_paths_set.add(path)

        num_covered_leaf_nodes = len(covered_leaf_node_paths_set)
        coverage_percentage = (num_covered_leaf_nodes / total_leaf_nodes) * 100 if total_leaf_nodes > 0 else 0

        uncovered_leaf_nodes_details = []
        covered_leaf_nodes_details = []

        for leaf in leaf_nodes:
            leaf_path = leaf.get("path", leaf.get("node_id"))
            if leaf_path in covered_leaf_node_paths_set:
                covered_leaf_nodes_details.append(leaf)
            else:
                uncovered_leaf_nodes_details.append(leaf)
        
        report = {
            "total_atomic_requirements (leaf_nodes)": total_leaf_nodes,
            "covered_atomic_requirements": num_covered_leaf_nodes,
            "uncovered_atomic_requirements": len(uncovered_leaf_nodes_details),
            "coverage_percentage": f"{coverage_percentage:.2f}%",
            "details": {
                "covered_leaf_nodes": [{"path": n["path"], "description": n["description"]} for n in covered_leaf_nodes_details],
                "uncovered_leaf_nodes (gaps)": [{"path": n["path"], "description": n["description"]} for n in uncovered_leaf_nodes_details]
            }
        }
        print("Coverage evaluation finished.")
        return report

    def assess_coverage_orchestrator(self, requirements_document_text, test_cases_list):
        """
        编排整个评估流程。
        :param requirements_document_text: 需求文档的纯文本内容。
        :param test_cases_list: 测试用例列表，每个元素是包含 'ID', 'Case Name', 'Description' 等字段的字典。
        :return: 覆盖率评估报告。
        """
        print("\n--- Starting Full Coverage Assessment ---")
        
        requirement_tree_result = self.analyze_requirements_to_tree(requirements_document_text)
        if not requirement_tree_result or not self.requirement_tree: # 确保 self.requirement_tree 也被填充了
            print("Assessment failed: Could not build requirement tree.")
            return {"error": "Failed to build requirement tree."}
        
        print("\n--- Requirement Tree Summary ---")
        leaf_nodes_count = len(self.get_leaf_nodes())
        print(f"Total nodes in tree: {len(self.requirement_tree)}")
        print(f"Number of leaf nodes (atomic requirements): {leaf_nodes_count}")
        if leaf_nodes_count == 0 and len(self.requirement_tree) > 0 : # 如果树不为空但没有叶子，是个问题
             print("Warning: No leaf nodes were identified in the populated tree. Coverage will be 0% or undefined.")
        elif leaf_nodes_count == 0 and len(self.requirement_tree) == 0:
             print("Warning: Tree is empty, no leaf nodes. Coverage will be 0% or undefined.")


        mapped_test_cases = self.map_test_cases_to_tree(test_cases_list, self.requirement_tree)
        if not mapped_test_cases and test_cases_list:
             print("Warning: No test cases were successfully mapped to requirements.")
        elif not test_cases_list:
             print("Info: No test cases provided for mapping.")
        
        coverage_report = self.evaluate_coverage(self.requirement_tree, mapped_test_cases)
        
        print("\n--- Assessment Finished ---")
        return coverage_report

# --- 使用示例 ---
if __name__ == "__main__":
    # 使用config.py中的API配置和模型设置

    sample_requirements_doc = """
    网易有钱记账APP需求文档
    1. 功能需求
    1.1 个人页面选项卡功能
      1.1.1 登录与注册功能：用户可以通过APP注册新的网易账号，并且可以使用已有的网易账号登录APP。
      1.1.2 实用工具集合：实用工具中可以使用汇率转换功能和房贷计算器功能。
    1.2 资产管理
      1.2.1 添加资产条目：支持添加资金、投资、应收、应付等各种资产来源。
    """
    
    unstructured_requirements_doc = """
    我们的新社交应用需要几个核心功能。首先，用户必须能够创建账户并登录。
    登录后，他们应该能看到一个动态消息流，可以发布自己的状态更新。
    状态更新可以包含文本和图片。用户还应该能够添加好友和给帖子点赞。
    我们还需要一个个人资料页面，用户可以在上面编辑自己的信息，比如头像和简介。
    最后，需要有通知功能，当有好友请求或帖子被点赞时提醒用户。
    哦对了，还有一个重要的，就是用户可以搜索其他用户。
    """

    # 更新示例测试用例以匹配新格式
    sample_test_cases_structured_doc = [
        {
            "ID": "TC001", 
            "Case Name": "Verify successful login with valid NetEase account", 
            "Priority": "High",
            "Description": "This test case verifies that an existing user can successfully log in to the NetEaseMoney app using their valid NetEase email and password.",
            "Precondition": "1. User has a registered NetEase account.\n2. NetEaseMoney app is installed.",
            "Environment": "Android 12, App v1.0",
            "Test Steps": "1. Open app. 2. Navigate to login. 3. Enter credentials. 4. Tap login.",
            "Expected Result": "User is logged in and redirected to the main dashboard.",
            "Designer": "Bob"
        },
        {
            "ID": "TC002", 
            "Case Name": "Verify new user registration via app", 
            "Priority": "High",
            "Description": "This test case verifies if a new user can complete the registration process for a NetEase account directly through the NetEaseMoney app.",
            "Precondition": "1. User does not have an existing NetEase account.\n2. Registration page is accessible.",
            "Environment": "iOS 15, App v1.0",
            "Test Steps": "1. Open app. 2. Navigate to registration. 3. Fill in details. 4. Complete verification.",
            "Expected Result": "User account is created successfully, and user is logged in.",
            "Designer": "Alice"
        },
        {
            "ID": "TC003", 
            "Case Name": "Verify currency converter functionality", 
            "Priority": "Medium",
            "Description": "This test case checks if the currency converter tool within the 'Utilities' section functions correctly by converting a specified amount from one currency to another.",
            "Precondition": "1. User is logged in.\n2. Utilities section is accessible.",
            "Environment": "Android 12, App v1.0",
            "Test Steps": "1. Navigate to Utilities. 2. Select Currency Converter. 3. Input amount and currencies. 4. Observe result.",
            "Expected Result": "The converted amount is displayed correctly according to current exchange rates.",
            "Designer": "Charlie"
        },
        {
            "ID": "TC004", 
            "Case Name": "Verify adding a new 'Investment' type asset", 
            "Priority": "High",
            "Description": "This test case ensures that the user can add a new asset of type 'Investment' under the Asset Management feature.",
            "Precondition": "1. User is logged in.\n2. Asset Management section is accessible.",
            "Environment": "Android 12, App v1.0",
            "Test Steps": "1. Go to Asset Management. 2. Tap 'Add Asset'. 3. Select 'Investment' type. 4. Fill details and save.",
            "Expected Result": "The new investment asset is listed in the user's assets.",
            "Designer": "David"
        }
    ]

    sample_test_cases_unstructured_doc = [
        {
            "ID": "UTC001",
            "Case Name": "New User Account Creation",
            "Priority": "High",
            "Description": "Verify that a new user can successfully create an account through the application's registration form.",
            "Precondition": "Registration page is accessible.",
            "Environment": "Web Browser Chrome latest",
            "Test Steps": "1. Navigate to registration page. 2. Fill required fields. 3. Submit form.",
            "Expected Result": "Account created, user redirected to login or dashboard.",
            "Designer": "Eva"
        },
        {
            "ID": "UTC002",
            "Case Name": "Existing User Login",
            "Priority": "High",
            "Description": "Verify that an existing user can log in with correct credentials.",
            "Precondition": "User account exists.",
            "Environment": "Web Browser Chrome latest",
            "Test Steps": "1. Navigate to login page. 2. Enter username and password. 3. Click login.",
            "Expected Result": "User logged in, dashboard visible.",
            "Designer": "Frank"
        },
        {
            "ID": "UTC003",
            "Case Name": "Post Status Update (Text only)",
            "Priority": "Medium",
            "Description": "Verify that a logged-in user can post a status update containing only text to their feed.",
            "Precondition": "User is logged in.",
            "Environment": "Mobile App Android 12",
            "Test Steps": "1. Go to feed. 2. Type text in status box. 3. Click post.",
            "Expected Result": "Status update appears in the user's feed and potentially others'.",
            "Designer": "Grace"
        }
    ]

    # 使用config.py中的默认配置（gpt-4o-2024-05-13, temperature=0.1, top_p=0.9）
    agent = AdequacyAgent() 

    print("--- Testing with Structured Document (Updated Test Case Format) ---")
    coverage_assessment_structured = agent.assess_coverage_orchestrator(sample_requirements_doc, sample_test_cases_structured_doc)
    print("\n=== FINAL COVERAGE REPORT (Structured Document) ===")
    print(json.dumps(coverage_assessment_structured, ensure_ascii=False, indent=2))

    # Reset agent state for the next test if needed (e.g. requirement_tree)
    agent.requirement_tree = {} 
    print("\n\n--- Testing with Unstructured Document (Updated Test Case Format) ---")
    coverage_assessment_unstructured = agent.assess_coverage_orchestrator(unstructured_requirements_doc, sample_test_cases_unstructured_doc)
    print("\n=== FINAL COVERAGE REPORT (Unstructured Document) ===")
    print(json.dumps(coverage_assessment_unstructured, ensure_ascii=False, indent=2))
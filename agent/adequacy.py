import pandas as pd
import kimi_llm as kimi
import json
import re
import csv
import os
import time
# 充分性维度评估智能体
path_functional_point = ''

role = "你是一个软件测试用例的鉴别专家。"
prompts = {
    "coverage_calculate": "现有一个测试需求文档的测试功能点列表如下，共有{}个。\n" +
                          "<{}>\n" +
                          "且有测试人员撰写的测试用例列表如下，共有{}个：\n" +
                          "<{}>\n" +
                          "按照以下步骤处理测试用例列表中的每一个测试用例。\n" +
                          "1、根据测试用例的内容将缺陷按照功能点列表中的功能点进行分类，输出这个用例对应的功能点，"
                          "功能点名称要与功能点列表中的名称完全对应，不要捏造功能点的名称。每个用例都要处理到，不要省略输出。\n"
                          "2、收集每一个用例对应的功能点，整理为一个列表输出，计为已覆盖功能点。" +
                          "构造一个json，输出步骤二产生的覆盖功能点的列表,key为covered，json代码输出使用```json ```格式进行包裹\n",
    "coverage_rate_score": "在一次软件测试中，现有完整的测试功能点列表如下，共有{}个：\n"
                           "<{}>\n" +
                           "且有测试人员编写测试用例覆盖到的功能点列表如下：\n" +
                           "<{}>\n" +
                           "按照以下步骤处理这两个功能点列表：\n"
                           "1、计算完整的测试功能点列表中功能点的数量。\n"
                           "2、计算测试人员编写测试用例覆盖到的功能点列表中功能点的数量。\n"
                           "3、使用覆盖功能点数量处以全部功能点的数量计算覆盖率。\n"
                           "4、根据覆盖率情况输出三句话的点评，告知开发人员这个测试人员编写的用例对功能点的覆盖情况，包括覆盖到了哪些，没有覆盖到哪些。\n"
                           "输出你对于上述四个步骤的思考过程\n"
                           "思考过程输出完毕后，请构造一个json，用score输出功能点的覆盖率得分（计算方式为覆盖率乘100），用comment输出点评内容，点评内容需要根据步骤4的输出完整概括。\n"
                           "构造的json代码输出使用```json ```格式进行包裹"
    # "3、将已覆盖功能点与全部功能点进行对比，计算出功能点的覆盖率。"
    # "4、计算原始功能点列表中功能点的数量，输出功能点的数量。"
    # "5、使用覆盖功能点的数量处以原始功能点列表中的数量，计算覆盖率，并输出覆盖分数。覆盖分数为覆盖率乘以100。",
}
messages = {"clustering": "", "defect_statistics": ""}
results = {"clustering_result": "", "statistics_result": ""}


def set_functional_points_path(file_path):
    global path_functional_point
    path_functional_point = file_path


# 读取Excel文件
def read_excel_and_extract_data(file_path):
    # 使用pandas读取Excel文件
    df = pd.read_excel(file_path)
    # 检查列名是否存在
    if '用例名称' in df.columns and '用例描述' in df.columns:
        # 遍历每一行
        result = []
        testcase_list = ''
        total = 0
        for index, row in df.iterrows():
            # 格式化字符串
            formatted_string = f"用例名称：{row['用例名称']}; 用例描述：{row['用例描述']}"
            result.append(formatted_string)
            testcase_list += formatted_string + '\n'
            total += 1
        return total, testcase_list
    else:
        return "列名不正确，请检查Excel文件中的列名是否包含'用例名称'和'用例描述'。"


def get_functional_points():
    # 打开文件并读取内容
    # 统计行数
    content = ''
    total_row = 0
    with open(path_functional_point, 'r', encoding='utf-8') as file:
        for item in file:
            total_row += 1
            content += item.strip() + '\n'
    return total_row, content


def extract_json_from_content(content):
    # 使用正则表达式提取JSON字符串
    match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
    if match:
        json_str = match.group(0).replace("```json", "").replace("```", "")
        try:
            # 解析JSON字符串
            data = json.loads(json_str)
            # 提取字段内容
            covered_list = data['covered']
            return covered_list
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    else:
        print("No JSON found in the text.")
        return None


def extract_score_json_from_content(content):
    # 使用正则表达式提取JSON字符串
    match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
    if match:
        json_str = match.group(0).replace("```json", "").replace("```", "")
        try:
            # 解析JSON字符串
            data = json.loads(json_str)
            # 提取字段内容
            score = data['score']
            comment = data['comment']
            return score, comment
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    else:
        print("No JSON found in the text.")


def eval_adequacy_one(file_path):
    # 1、根据功能点聚类缺陷
    points_num, points = get_functional_points()
    testcase_num, testcase = read_excel_and_extract_data(file_path)
    prompt_coverage = prompts["coverage_calculate"]. \
        format(points_num, get_functional_points(), testcase_num, testcase)
    # 记录发送的提示词
    messages["coverage_calculate"] = prompt_coverage
    # 将计算指令发送到大模型,并记录结果
    print("正在对测试用例列表进行处理，确定已覆盖功能点...")
    results["coverage_calculate_result"] = kimi.chat_with_kimi(role, prompt_coverage)
    # 输出覆盖的功能点
    print(results["coverage_calculate_result"])
    try_times = 0
    while "省略" in results["coverage_calculate_result"]:
        results["coverage_calculate_result"] = kimi.chat_with_kimi(role, prompt_coverage)
        try_times += 1
        if try_times > 2:
            return 0, "出现省略，人工检查"
    covered = extract_json_from_content(results["coverage_calculate_result"])
    while covered is None:
        print("正在对测试用例列表进行处理，确定已覆盖功能点...")
        results["coverage_calculate_result"] = kimi.chat_with_kimi(role, prompt_coverage)
        # 输出覆盖的功能点
        print(results["coverage_calculate_result"])
        try_times = 0
        while "省略" in results["coverage_calculate_result"]:
            results["coverage_calculate_result"] = kimi.chat_with_kimi(role, prompt_coverage)
            try_times += 1
            if try_times > 2:
                return 0, "出现省略，人工检查"
        covered = extract_json_from_content(results["coverage_calculate_result"])
    list_covered = ''
    for item in covered:
        list_covered += item + '\n'
    print("处理完成，已覆盖功能点如下：\n" + list_covered)
    prompt_coverage_rate_score = prompts["coverage_rate_score"]. \
        format(points_num, points, list_covered)
    # 记录发送的提示词
    messages["coverage_rate_score"] = prompt_coverage_rate_score
    # 将覆盖率计算指令发送到大模型,并记录结果
    print("正在比对功能点和已覆盖功能点列表，计算覆盖率并生成反馈...")
    results["coverage_rate_score"] = kimi.chat_with_kimi(role, prompt_coverage_rate_score)
    print("coverage 计算情况 ")
    print(results["coverage_rate_score"])
    try:
        score, comment = extract_score_json_from_content(results["coverage_rate_score"])
    except:
        return 0, "NO JSON FOUND"
    # print(f"score:{score}, comment:{comment}")
    return score, comment


def eval_reports_adequacy(folder_path):
    # 文件夹路径
    csv_file_path = 'report_adequacy_scores_wangyi2.csv'
    # 创建或覆盖CSV文件
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        # 创建CSV写入器
        csv_writer = csv.writer(csv_file)
        # 写入表头
        csv_writer.writerow(['文件名', '充分性分数', '评价反馈'])
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file_name)
            # 检查是否为Excel文件
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                # 调用函数获取分数和评论
                # 记录开始时间
                start_time = time.time()
                print(f"正在对{file_name}进行报告充分性维度评价..")
                score, comment = eval_adequacy_one(file_path)
                print(f"报告{file_name}充分性维度评价得分为{score}")
                # 记录结束时间
                end_time = time.time()
                # 计算并打印执行时间（秒）
                execution_time = end_time - start_time
                print(f"Single Execution time: {execution_time:.2f} seconds")
                # 将结果写入CSV
                csv_writer.writerow([file_name, score, comment])
            else:
                print(f"Skipping non-Excel file: {file_name}")
    print("处理完成，结果已保存到CSV文件。")


if __name__ == '__main__':
    # eval_reports_adequacy("../data/reports_testcase")
    set_functional_points_path("../data/saber/functional_points.txt")
    eval_adequacy_one("../data/saber/report_testcase/10.xlsx")
    print(get_functional_points())
    # for file_name in os.listdir("../data/reports_testcase"):
    #     # 构建完整的文件路径
    #     file_path = os.path.join("../data/reports_testcase", file_name)
    #     # 检查是否为Excel文件
    #     if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
    #         # 调用函数获取分数和评论
    #         print(f"正在对{file_name}进行报告充分性维度评价..")
    #         print(read_excel_and_extract_data(file_path))

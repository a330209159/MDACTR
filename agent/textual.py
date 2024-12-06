import time
import random
import pandas as pd
import kimi_llm as kimi
import json
import os
import csv

role = "你是一个软件测试用例的评判专家，你将从按照我提供给你的评分规则，从一条用例的文本性层面上对这条用例进行量化评价。"

prompts = {
    "score_criteria": "以下是针对测试用例撰写的评分标准，这个标准对一条测试用例的每一项的规范性都进行了要求；规范性评分标准共涉及RM、RR、RA这三个方面。每个方面都有几点规定了该点的评分标准。\n" +
                      "<{}>\n" +
                      "请使用以上评分标准对这条测试用例的规范性进行评分。评分结果使用json格式输出，其中，criteria字段为评分点的序号，score为其分数，reason为评价为这个分数的原因，一个示例的json的具体格式如下：\n" +
                      "[\n" +
                      "   {{\"criteria\":\"RM1\",\n" +  # 使用{{和}}来转义大括号
                      "      \"reason\":\"<原因>\"\n" +
                      "     \"score\":\"3\",\n" +
                      "   }},\n" +
                      "   {{\"criteria\":\"RM2\",\n" +
                      "      \"reason\":\"<原因>\"\n" +
                      "     \"score\":\"4\",\n" +
                      "   }},\n" +
                      "   …\n" +
                      "]\n",
    "score_testcase": "以下内容是一条软件测试用例，请按照上述要求进行评分并使用json输出结果，无需输出其他无关内容。\n" + "<{}>"}

# 发送指令记录
messages = {"clustering": "", "defect_statistics": ""}
# 结果记录
results = {"clustering_result": "", "statistics_result": ""}


def read_excel_to_dict(file_path):
    # 指定要读取的列
    columns = [
        "用例编号", "用例名称", "优先级", "用例描述",
        "前置条件", "环境配置", "操作步骤", "预期结果",
        "设计人员", "测试结果"
    ]

    # 读取Excel文件，只包含指定的列
    df = pd.read_excel(file_path, engine='openpyxl', usecols=columns)

    # 将DataFrame转换为字典列表
    data_list = df.to_dict(orient='records')

    return data_list


def get_textual_scoring_rule():
    with open('rule/textual_scoring_rule.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def calculate_total_score(json_str):
    # 将JSON字符串转换为Python对象
    data = json.loads(json_str)
    # 初始化总分为0
    total_score = 0
    comment = ''
    scores = {"RM1": 0}
    # 遍历每个条目，累加分数
    for item in data:
        total_score += float(item['score'])
        comment += f"{str(item['criteria'])}:{str(item['reason'])}\n"
        scores[str(item['criteria'])] = float(item['score'])
        # 返回总分
    # # 标准化分数
    # final_score = (total_score / 38) * 100
    return scores, comment


rule_list = ["RM1", "RM2", "RM3", "RR1", "RR1.1", "RR2", "RR3", "RR4", "RR5", "RA1", "RA2"]
row_list = ['file_path'] + rule_list


def calculate_average_score(file_path, all_scores):
    total_score = {"RM1": 0}
    average_score = {"RM1": 0}
    for item in all_scores:
        for i in rule_list:
            ascore = item[i]
            if i not in total_score:
                total_score[i] = ascore
            else:
                total_score[i] += ascore
    for item in rule_list:
        average_score[item] = total_score[item] / len(all_scores)
        average_score[item] = round(average_score[item], 2)
    average_score['file_path'] = file_path
    return average_score


def save_list_to_file(list_items, file_name):
    """
    将列表中的每个元素保存到一个文本文件中，每个元素占一行。

    参数:
    list_items (list): 要写入文件的元素列表。
    file_name (str): 要写入的文件名。
    """
    # 使用with语句打开文件，确保正确关闭文件
    with open(file_name, "w", encoding="utf-8") as file:
        # 遍历列表中的每个元素
        for item in list_items:
            # 将每个元素写入文件，每个元素后跟一个换行符
            file.write(f"{item}\n")


def evaluation(i):
    path_testcase_folder = '../data/app/neteasemoney/report_testcase'
    csv_file_path = f'report_textual_scores.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        # 创建CSV写入器
        csv_writer = csv.writer(csv_file)
        # 写入表头
        csv_writer.writerow(row_list)
        for file_name in os.listdir(path_testcase_folder):
            # 构建完整的文件路径
            file_path = os.path.join(path_testcase_folder, file_name)
            # 检查是否为Excel文件
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                # 调用函数获取分数和评论
                # 记录开始时间
                start_time = time.time()
                print(f"正在对{file_name}进行报告文本性维度评价..")
                testcase_list = read_excel_to_dict(file_path)
                prompt_criteria = prompts["score_criteria"].format(get_textual_scoring_rule())
                # 取样暂定5条
                random_testcase_list = random.sample(testcase_list, min(len(testcase_list), 10))
                print(f"正在评估{file_path},列表数量{len(random_testcase_list)}")
                all_scores = []
                for item in random_testcase_list:
                    prompt_score = prompt_criteria + prompts["score_testcase"].format(item)
                    # 记录发送的提示词
                    messages["textual_score"] = prompt_score
                    # 将计算指令发送到大模型,并记录结果
                    results["coverage_calculate_result"] = kimi.chat_with_kimi(role, prompt_score, model='kimi')
                    scores, comment = calculate_total_score(results["coverage_calculate_result"])
                    print(scores)
                    all_scores.append(scores)
                # 计算平均分并写出到csv文件中
                # 记录结束时间
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"-----------------------Execution time: {execution_time:.2f} seconds")
                average_score = calculate_average_score(file_name, all_scores)
                the_row = []
                for item in row_list:
                    the_row.append(average_score[item])
                csv_writer.writerow(the_row)


print("处理完成，结果已保存到CSV文件。")

if __name__ == '__main__':

    evaluation("neteasemoney")


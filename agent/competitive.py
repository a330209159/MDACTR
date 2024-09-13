import kimi_llm as kimi
import pandas as pd
import csv
import os
import time
role = "你是一个软件测试领域的专家。"
prompts = {
    "scoring_table_create": "请按照以下规则，根据我提供给你的缺陷类别统计表生成一张缺陷评分表。" +
                            "缺陷类别统计表如下：\n" +
                            "<{}>\n" +  # 放置缺陷类别统计表
                            "缺陷评分表生成规则如下：\n" +
                            "<{}>\n" +  # 放置缺陷评分表生成规则
                            "给出你使用生成规则里面所有步骤的完整计算过程，并最终将结果以表格的形式进行反馈。",
    "table_extract": "请帮我提取出以下内容中步骤4所生成的表格，仅输出表格即可，不要输出任何其他无关内容以及解释性文字。\n" +
                     "<{}>",
    "score_report": "现有一张缺陷评分表如下，表中定义了缺陷类别和该类别的缺陷描述，同时定义了这个缺陷类别的分数：" +
                    "<{}>" +
                    "现有缺陷报告列表如下，每行代表一个缺陷：" +
                    "<{}>" +
                    "按照以下步骤处理缺陷报告列表中的缺陷：" +
                    "1、根据缺陷内容将缺陷按照评分表中的缺陷类别进行分类，输出这个缺陷的类别。"
                    "2、找到缺陷类别对应的分数。输出该分数。未找到的类别不得分。" +
                    "3、对缺陷列表中的每个缺陷重复步骤1-2，直至列表遍历结束，将所有的分数加和，输出最终得分。"
                    "4、对缺陷列表的覆盖情况进行点评，评价这个缺陷列表对缺陷评分表中的缺陷类别的覆盖情况，需要具体说明覆盖了哪些缺陷，没有覆盖到哪些缺陷，并提供改进建议。"
                    "5、构造一个json，json中有两个key，一是score，记录步骤3的最终得分；二是comment，记录步骤4的评论；",
}
# 发送指令记录
messages = {"clustering": "", "defect_statistics": ""}
# 结果记录
results = {"clustering_result": "", "statistics_result": ""}

path_defect_cluster_amount_table = ''


def set_defect_cluster_amount_table_path(file_path):
    global path_defect_cluster_amount_table
    path_defect_cluster_amount_table = file_path


# 获取缺陷评分表格创建规则
def get_scoring_table_creating_rule():
    with open('./rule/scoring_table_creating_rule.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# 获取缺陷类别统计表
def get_defect_cluster_amount_table():
    with open('./preprocess/defect_cluster_amount_table.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def create_defect_scoring_table():
    # 1、根据功能点聚类缺陷
    prompt_table_creating = prompts["scoring_table_create"].format(get_defect_cluster_amount_table(),
                                                                   get_scoring_table_creating_rule())
    # 记录发送的提示词
    messages["scoring_table_create"] = prompt_table_creating
    # 将聚类指令发送到大模型,并记录结果
    results["scoring_table_create_result"] = kimi.chat_with_kimi(role, prompt_table_creating)
    # 2、提取评分表格
    prompt_table_extract = prompts["table_extract"].format(results["scoring_table_create_result"])
    messages["table_extract"] = prompt_table_extract
    results["table_extract"] = kimi.chat_with_kimi(role, prompt_table_extract)
    with open('./rule/defect_scoring_table.txt', 'w', encoding='utf-8') as f:
        f.write(results["table_extract"])


def get_defect_scoring_table():
    with open('./rule/defect_scoring_table.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def eval_one_report_competitive_score(file_path):
    # file_path = '../data/reports/1.xlsx'
    df = pd.read_excel(file_path)
    # print(df.columns)
    # 将“缺陷描述”列的所有内容放入列表中
    defect_descriptions_list = df['缺陷描述'].tolist()
    all_defects = ''
    for item in defect_descriptions_list:
        if pd.isna(item):
            continue
        item = str(item)
        all_defects += item.strip().replace('\n', '') + '\n'
    prompt_score_report = prompts["score_report"].format(get_defect_scoring_table(),
                                                         all_defects)
    # 记录发送的提示词
    messages["score_report"] = prompt_score_report
    # 将聚类指令发送到大模型,并记录结果
    results["score_report_result"] = kimi.chat_with_kimi(role, prompt_score_report)
    return extract_json_from_content(results["score_report_result"])


def extract_json_from_content(content):
    import json
    import re
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


def eval_reports_competitive(folder_path):
    # 文件夹路径
    csv_file_path = 'report_competitive_scores_wangyi.csv'
    # 创建或覆盖CSV文件
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        # 创建CSV写入器
        csv_writer = csv.writer(csv_file)
        # 写入表头
        csv_writer.writerow(['文件名', '分数', '反馈'])
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file_name)
            # 检查是否为Excel文件
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                # 调用函数获取分数和评论
                start_time = time.time()
                print(f"正在对{file_name}进行报告竞争性维度评价..")
                score, comment = eval_one_report_competitive_score(file_path)
                print(f"报告{file_name}竞争性维度评价得分为{score}")
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Single Execution time: {execution_time:.2f} seconds")
                # 将结果写入CSV
                csv_writer.writerow([file_name, score, comment])
            else:
                print(f"Skipping non-Excel file: {file_name}")
    print("处理完成，结果已保存到CSV文件。")


if __name__ == '__main__':
    eval_reports_competitive("../data/reports_defect")
    # eval_one_report_competitive_score("../data/reports/8.xlsx")

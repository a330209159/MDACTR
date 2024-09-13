import kimi_llm as kimi
import pandas as pd

role = "你是一个计算机领域的软件测试报告的审阅专家，擅长识别和对缺陷报告进行聚类。"
prompts = {
    "cluster_execute": "请将软件测试报告中的缺陷进行聚类，按照功能点列表中功能点的粒度进行细致聚类。" +
                       "缺陷将以列表形式提供，以下字段一行代表一个缺陷描述；\n" +
                       "<{}>\n" +  # 放置全体缺陷列表
                       "功能点列表如下：\n" +
                       "<{}>\n" +  # 放置功能点列表
                       "请根据功能点列表对缺陷进行聚类，并输出一个表格，表格中主要有两个字段”，"
                       "“缺陷名”和“缺陷简要描述”。“缺陷名”应表示聚类后得到的类别名，用“xx问题”或“xx功能异常”的格式描述，"
                       "而“缺陷描述”应简要概括这个聚类类别的缺陷主要包含哪些问题，不要完整输出这个聚类类别中所包含的缺陷描述。"
                       "确保每个缺陷被准确地聚类到对应的功能点和缺陷类型中，并保持尽可能细到每个功能点上的聚类粒度。"
                       "不要出现'其他'类别，聚类中允许存在单个缺陷,只允许出现缺陷列表中反映的问题，不要自己捏造任何缺陷。"
                       "结果仅输出缺陷类别及其描述表格即可，不要输出任何其他无关内容。",
    "defect_statistics": "根据以下表格，统计缺陷列表中每个缺陷类对应的缺陷数量，"
                         "结果输出为一个表格，表头为“缺陷名”、“缺陷描述”和“缺陷数量”，"
                         "要保证缺陷列表中的所有缺陷都被统计到，数量不可少。\n" +
                         "<{}>" +
                         "缺陷列表如下：" +
                         "<{}>" +
                         "结果仅输出数量统计表格，不要输出任何其他无关的内容，如解释内容。",
    "defect_amount_reorder": "将以下缺陷统计表格中的行按照缺陷数量列从大到小的顺序进行排序，并重新输出该表格。" +
                             "<{}>" +
                             "结果仅输出数量统计表格，不要输出任何其他无关的内容，如解释内容。",
}
# 发送指令记录
messages = {"clustering": "", "defect_statistics": ""}
# 结果记录
results = {"clustering_result": "", "statistics_result": ""}


# 获取全体缺陷列表字符串
def get_all_defects():
    # 读取Excel文件
    df = pd.read_excel('../data/saber/defects.xlsx')
    # 将“缺陷描述”列的所有内容放入列表中
    defect_descriptions_list = df['缺陷描述'].tolist()
    all_defects = ''
    for item in defect_descriptions_list:
        if pd.isna(item):
            continue
        item = str(item)
        all_defects += item.strip().replace('\n', '') + '\n'
    return all_defects


def get_functional_points():
    # 打开文件并读取内容
    with open('../data/saber/functional_points.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    return content


if __name__ == '__main__':
    # 1、根据功能点聚类缺陷
    prompt_clustering = prompts["cluster_execute"].format(get_all_defects(), get_functional_points())
    # 记录发送的提示词
    messages["clustering"] = prompt_clustering
    # 将聚类指令发送到大模型,并记录结果
    results["clustering_result"] = kimi.chat_with_kimi(role, prompt_clustering, model='3.5')
    # 2、根据聚类结果统计缺陷数量
    prompt_statistics = prompts["defect_statistics"].format(results["clustering_result"], get_all_defects())
    # 记录发送的提示词
    messages["defect_statistics"] = prompt_statistics
    # 将统计指令发送到大模型，并记录结果
    results["statistics_result"] = kimi.chat_with_kimi(role, prompt_statistics)
    # 3、将缺陷统计表格按照从大到小的顺序重排序
    prompt_amount_reorder = prompts["defect_amount_reorder"].format(results["statistics_result"])
    messages["reorder_message"] = prompt_amount_reorder
    results["reorder_result"] = kimi.chat_with_kimi(role, prompt_amount_reorder)
    print(results["clustering_result"])
    print(results["statistics_result"])
    print(results["reorder_result"])

from adequacy import eval_reports_adequacy, set_functional_points_path
from competitive import set_defect_cluster_amount_table_path, create_defect_scoring_table, eval_reports_competitive
from textual import evaluation
# 充分性维度所需要的参数
# 功能点列表文件路径
import time
import sys


def run_competitve_eval():
    end_time = time.time()
    #
    # # 计算并打印执行时间（秒）
    # execution_time = end_time - start_time
    # print(f"Execution time: {execution_time:.2f} seconds")

    # 竞争性维度所需要的参数
    # 预处理阶段获取的缺陷类别统计表路径
    # 记录开始时间
    start_time = time.time()
    path_defect_cluster_amount_table = './preprocess/defect_cluster_amount_table.txt'
    # 缺陷报告文件夹路径
    path_defect_reports_folder = '../data/app/neteasemoney/reports_defect'
    # 竞争性维度评估智能体所能调用的方法及顺序
    # 1、设置缺陷类别统计表路径
    set_defect_cluster_amount_table_path(path_defect_cluster_amount_table)
    # 2、创建竞争性维度评分表
    create_defect_scoring_table()
    # 3、缺陷报告评估
    eval_reports_competitive(path_defect_reports_folder)
    # 记录结束时间
    end_time = time.time()
    # 计算并打印执行时间（秒）
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


def run_adequacy_eval():
    # 充分性维度所需要的参数
    # 功能点列表文件路径
    path_functional_point = '../data/app/neteasemoney/functional_points.txt'
    # 测试用例报告文件夹路径
    path_testcase_folder = '../data/app/neteasemoney/report_testcase'
    # 记录开始时间
    start_time = time.time()
    # 调用函数
    set_functional_points_path(path_functional_point)
    eval_reports_adequacy(path_testcase_folder)
    # 记录结束时间
    end_time = time.time()
    # 计算并打印执行时间（秒）
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


def run_textual_eval():
    evaluation("neteasemoney")


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py [adequacy|competitive|textual]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "adequacy":
        run_adequacy_eval()
    elif mode == "competitive":
        run_competitve_eval()
    elif mode == "textual":
        run_textual_eval()
    else:
        print("Invalid argument. Use 'adequacy' or 'competitive'.")
        sys.exit(1)


if __name__ == "__main__":
    main()

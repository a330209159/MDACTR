# API配置文件
# 请替换下面的API密钥为实际的密钥

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Kimi API配置
KIMI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"

# OpenAI API配置
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_BASE_URL = "https://api.ohmygpt.com/v1"  # 使用转发GPT的URL

# 模型配置
DEEPSEEK_MODEL = "deepseek-chat"
KIMI_MODEL = "moonshot-v1-32k"
GPT4O_MODEL = "gpt-4o-2024-05-13"

# 通用参数配置
TEMPERATURE = 0.1
TOP_P = 0.9

# 并发配置
MAX_WORKERS = 2  # 双LLM并发数量

# 评估模式配置
ENABLE_DUAL_LLM_ASSESSMENT = True  # 是否开启双LLM评估
ENABLE_ARBITRATION = True  # 是否开启仲裁功能（当有分歧时使用GPT-4o解决）
DEFAULT_SINGLE_LLM = "deepseek"  # 单LLM模式下使用的默认LLM: "deepseek" 或 "kimi"

# 文本性评估配置
ENABLE_TEXTUAL_SAMPLING = True  # 是否开启抽样评估
TEXTUAL_SAMPLE_SIZE = 5  # 抽样评估时每个类型的样本数量
TEXTUAL_SAMPLE_TESTCASE_SIZE = 5  # 抽样评估时测试用例的样本数量
TEXTUAL_SAMPLE_DEFECT_SIZE = 5  # 抽样评估时缺陷的样本数量

# 输出配置
SAVE_DETAILED_RESULTS = True
OUTPUT_FILE_PREFIX = "textual_evaluation_results"

# --- 定量指标清单定义 ---
CHECKLIST_TEMPLATE = {
  "Morphological": [
    {
      "id": "RM1",
      "name": "Text Length",
      "rule_code": "RM1",
      "rule_content": "Text length is within the preset range.",
      "score": 3,
      "checkpoints": [
        { "description": "文本长度是否在预设的最小范围内。", "value": False },
        { "description": "文本长度是否在预设的最大范围内。", "value": False },
        { "description": "整体文本长度对于测试用例/缺陷的上下文来说是否恰当。", "value": False }
      ]
    },
    {
      "id": "RM2", 
      "name": "Readability",
      "rule_code": "RM2",
      "rule_content": "The description is concise, fluent, and easy to understand.",
      "score": 2,
      "checkpoints": [
        { "description": "描述是否简洁明了。", "value": False },
        { "description": "描述是否流畅且易于理解。", "value": False }
      ]
    },
    {
      "id": "RM3",
      "name": "Punctuation", 
      "rule_code": "RM3",
      "rule_content": "Punctuation is used correctly.",
      "score": 3,
      "checkpoints": [
        { "description": "所有句子是否使用正确的标点符号结尾。", "value": False },
        { "description": "逗号和其他内部标点符号是否使用得当。", "value": False },
        { "description": "整体标点符号使用是否正确。", "value": False }
      ]
    }
  ],
  "Relational": [
    {
      "id": "RR1",
      "name": "Itemization",
      "rule_code": "RR1", 
      "rule_content": "Operational steps are listed with numbered annotations.",
      "score": 5,
      "checkpoints": [
        { "description": "操作步骤是否清晰列出。", "value": False },
        { "description": "步骤是否按顺序编号。", "value": False },
        { "description": "每个步骤是否包含明确且独立的动作。", "value": False },
        { "description": "编号注释是否一致且正确。", "value": False },
        { "description": "所有必要的操作步骤是否都已包含在列表中。", "value": False }
      ]
    },
    {
      "id": "RR2",
      "name": "Environment",
      "rule_code": "RR2",
      "rule_content": "Environmental information is present and detailed.",
      "score": 3,
      "checkpoints": [
        { "description": "环境信息是否已提供。", "value": False },
        { "description": "环境详情（如操作系统、浏览器、设备）是否具体。", "value": False },
        { "description": "环境信息是否完整且准确。", "value": False }
      ]
    },
    {
      "id": "RR3",
      "name": "Preconditions",
      "rule_code": "RR3",
      "rule_content": "Preconditions are described and complete.",
      "score": 2,
      "checkpoints": [
        { "description": "前置条件是否已描述。", "value": False },
        { "description": "前置条件是否完整且足以支持测试执行。", "value": False }
      ]
    },
    {
      "id": "RR4",
      "name": "Expected Results",
      "rule_code": "RR4",
      "rule_content": "Expected results are filled in standardly.",
      "score": 2,
      "checkpoints": [
        { "description": "预期结果是否已填写。", "value": False },
        { "description": "预期结果的描述是否规范且清晰。", "value": False }
      ]
    },
    {
      "id": "RR5",
      "name": "Additional Information",
      "rule_code": "RR5",
      "rule_content": "All other fields except for the above information are filled in.",
      "score": 2,
      "checkpoints": [
        { "description": "除上述信息外，所有其他必填字段是否都已填写。", "value": False },
        { "description": "所提供的补充信息是否相关且完整。", "value": False }
      ]
    },
    {
      "id": "RR6",
      "name": "Screenshot",
      "rule_code": "RR6(*)",
      "rule_content": "For defects, there should be screenshots for explanation.",
      "score": 3,
      "is_defect_specific": True,
      "checkpoints": [
        { "description": "针对缺陷，是否提供了截图。", "value": False },
        { "description": "截图是否清晰地展示了缺陷。", "value": False },
        { "description": "截图是否经过适当的标注或裁剪以突出问题。", "value": False }
      ]
    }
  ],
  "Analytical": [
    {
      "id": "RA1",
      "name": "User Interface",
      "rule_code": "RA1",
      "rule_content": "There is a clear enough description of the interface elements.",
      "score": 5,
      "checkpoints": [
        { "description": "句子中是否明确提到了'接口元素'或'页面元素'等相关词汇。", "value": False },
        { "description": "对界面元素的描述是否清晰且无歧义。", "value": False },
        { "description": "描述是否足够具体，能够帮助审核人员快速定位到页面上的相应元素。", "value": False },
        { "description": "在复现步骤中，界面元素的描述是否充分。", "value": False },
        { "description": "界面元素的描述是否有效帮助理解并复现缺陷。", "value": False }
      ]
    },
    {
      "id": "RA2",
      "name": "User Behavior",
      "rule_code": "RA2",
      "rule_content": "There is a clear enough description of the interface elements.",
      "score": 5,
      "checkpoints": [
        { "description": "交互行为是否已描述。", "value": False },
        { "description": "用户操作和系统响应是否清晰概述。", "value": False },
        { "description": "交互行为的顺序是否符合逻辑。", "value": False },
        { "description": "交互过程中任何特定输入是否已描述。", "value": False },
        { "description": "所描述的交互行为是否准确反映了测试场景。", "value": False }
      ]
    },
    {
      "id": "RA3",
      "name": "Defect Feedback",
      "rule_code": "RA3(*)",
      "rule_content": "For defects, defect feedback should be included.",
      "score": 3,
      "is_defect_specific": True,
      "checkpoints": [
        { "description": "针对缺陷，是否包含了缺陷反馈。", "value": False },
        { "description": "缺陷反馈是否清晰地描述了 Bug。", "value": False },
        { "description": "缺陷反馈是否包含了重现/理解所需的详细信息。", "value": False }
      ]
    }
  ]
}

# --- Excel文件列名映射 ---
EXCEL_COL_MAPPING_TESTCASE = {
    "用例编号": "test_case_id",
    "用例名称": "title",
    "优先级": "priority",
    "用例描述": "description",
    "前置条件": "preconditions",
    "环境配置": "environmental_info",
    "操作步骤": "steps_to_reproduce",
    "输入数据": "input_data",
    "预期结果": "expected_result",
    "评判标准": "judgment_criteria",
    "其他说明": "additional_notes",
    "设计人员": "designer",
    "测试结果": "test_result", 
    "测试结论": "test_conclusion", 
    "附件": "attachments" 
}

EXCEL_COL_MAPPING_DEFECT = {
    "缺陷编号": "defect_id",
    "用例编号": "test_case_id_ref",
    "缺陷描述": "description",
    "缺陷类型": "defect_type",
    "前置条件": "preconditions",
    "缺陷界面标题": "ui_title",
    "操作步骤": "steps_to_reproduce",
    "环境信息": "environmental_info",
    "输入数据": "input_data",
    "预期结果": "expected_result",
    "实际结果": "actual_result",
    "报告填写时间": "report_fill_time",
    "提交人员": "submitter",
    "缺陷界面截图": "attachments_screenshot",
    "缺陷录屏": "attachments_video"
}

# --- 系统提示词配置已移至prompts.py ---
# 为保持向后兼容性，这里从prompts.py导入提示词
from prompts import SYSTEM_PROMPT_EVALUATOR, SYSTEM_PROMPT_ARBITRATOR

# --- 数据路径配置 ---
DEFAULT_DATA_DIR = "data"
DEFAULT_TESTCASE_SUBDIR = "testcases"
DEFAULT_DEFECT_SUBDIR = "defects" 
from openai import OpenAI

# # 配置OpenAI客户端
client_kimi = OpenAI(
    api_key="xxxxxxxxx",
    base_url="xxxxxxxxxxx",
)
client = OpenAI(
    api_key="xxxxxxxxxx",
    base_url="xxxxxxxx",
)


def chat_with_kimi(role, prompt, model='3.5'):
    messages = [
        {"role": "system",
         "content": role},
        {"role": "user", "content": prompt}
    ]
    # 发送聊天请求
    if model == '4o':
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
        )
    elif model == '3.5':
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
        )
    elif model == 'kimi':
        completion = client_kimi.chat.completions.create(
            model="moonshot-v1-8k",
            messages=messages,
            temperature=0,
        )
    response = completion.choices[0].message.content
    return response


def eval_with_kimi(criteria_prompt, testcase_prompt):
    # 创建聊天消息列表
    messages = [
        {"role": "system",
         "content": criteria_prompt},
        {"role": "user", "content": testcase_prompt}
    ]

    # 发送聊天请求
    completion = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=messages,
        temperature=0.3,
    )

    # 获取并打印回复内容
    response = completion.choices[0].message.content
    return response

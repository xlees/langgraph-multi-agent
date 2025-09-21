
# import os
# from typing import TypedDict
# from typing import Literal
# from langgraph.graph import StateGraph, END, START
# from langgraph.prebuilt import create_react_agent
# # from langchain_google_genai import ChatGoogleGenerativeAI

# from crewai_tools import SerperDevTool
# from langchain.tools import tool

import sys,os,json
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from openai import OpenAI


# llm = ChatDeepSeek(model="deepseek-chat",
#                    temperature=0.5,
#                    api_key="sk-400041c678694691b56cd11bd8800a7b")

llm = OpenAI(api_key="sk-400041c678694691b56cd11bd8800a7b",base_url="https://api.deepseek.com")


def get_coupon(source='tb',url='',pwd=''):
    print("找券函数启动...\n",source,url,pwd)
    return "有12元券，券后价99元"

tools = [
    # 查券
    {
        "type": "function",
        "function": {
            "name": "get_coupon",
            "description": "查询商品的优惠券，用户需提供商品的url或者商品的口令(一段字符)",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "电商平台, 取值tb/pdd/jd",
                    },
                    "url": {
                        "type": "string",
                        "description": "商品url",
                    },
                    "pwd": {
                        "type": "string",
                        "description": "商品口令(由一段10-20左右的字符组成),前后由$或￥包括",
                    },
                },
                "required": ["source","pwd"]
            },
        }
    },
    # 
]

tools_map = {
    "get_coupon": get_coupon,
}


def call_func(func_name,  **kwargs):
    if func_name not in tools_map:
        raise ValueError(f"函数 {func_name} 不存在")
    return tools_map[func_name]( **kwargs)

class QueryCoupon(BaseModel):
    '''查询给定商品的优惠券'''
    url: str = Field(description="商品的url, 支持长链接或短链接")
    
# 0.商品识别agent: 根据用户输入的内容，识别出商品的来源平台和商品信息。
product_agent = """你是一个商品识别助手，根据用户输入的内容，推断出用户最有可能想买的商品名称, 提取出搜索关键词, 用逗号进行分割。关键词尽量不要有语义上的重复。"""
resp = llm.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": product_agent},
        {"role": "user", "content": "准备要孩子了，想买个婴儿推车"},
        {"role": "assistant", "content": "婴儿推车,遮阳伞"},
        # 设定输出格式
        {"role": "user", "content": "根据要买的商品，简单分别介绍其使用场景。按照商品顺序进行介绍，介绍完毕后，紧跟一段小程序路径代码: <wx://xxxx/pages/product?query=商品"},
    ],
    temperature=0.1,  # temperature 控制生成内容的随机性，值越低输出越确定，值越高输出越多样
    # stream=True,
)
print(resp.choices[0].message.content)
sys.exit(0)
    

# 1.意图识别agent: 根据问题的类别，将问题路由到下一个指定的agent，由其解答。
intent_agent = """你是一个购物助手，根据用户的输入来判断用户的意图。输出只有三种情况：1.用户想知道当前的商品有没有优惠券(商品是url或者口令); 2.用户想咨询购物相关的问题; 3.与购物无关的问题。根据意图，输出1,2,3.
其他任何内容都不要输出，输出只能是1,2,3.
"""
resp = llm.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": intent_agent},
        {"role": "user", "content": "https://m.tb.cn/h.fh8kX6?sm=3f3e2f"},
    ],
    temperature=0.1,  # temperature 控制生成内容的随机性，值越低输出越确定，值越高输出越多样
    # stream=True,
)
print(resp.choices[0].message.content)

sys.exit(0)

# llm_with_tools = llm.bind_tools([QueryCoupon])

# system_prompt = """
# 你是一个购物助手，根据用户的输入来判断用户的意图。输出只有三种情况：1.用户想知道当前的商品有没有优惠券(商品是url或者口令); 2.用户想咨询购物相关的问题; 3.与购物无关的问题。根据意图，输出1,2,3.

# # 输入输出例子
# ## 用户输入
# $7847jik$ 淘宝
# ## 输出json结构:
# {
#     "qtype": 1,
#     "tools": ["get_coupon"]
# }
# """

# 2.查券agent
system_prompt = """你是一个比价助手，根据用户输入的商品信息，去淘宝、拼多多、京东去查询优惠券信息。
用户输入内容包括以下两种：
1.url
根据提供的url，自行推断所属的电商平台
2.口令
口令是一个字符串，以$或￥包含，也有可能没有两端字符。根据用户输入的内容，来判断是哪个平台的口令。如果无法判断，默认是淘宝口令。
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "￥7847jik￥"},
]

resp = llm.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    temperature=0.1,  # temperature 控制生成内容的随机性，值越低输出越确定，值越高输出越多样
    # stream=True,
    # response_format={
    #     'type': 'json_object'
    # },
    tools=tools
)
# for chunk in resp:
#     print(chunk.choices[0].delta.content, end="\n")
# print(resp.choices[0].message)

messages.append({
    "role": "assistant",
    "content": resp.choices[0].message.content
})

# step1: 模型返回结果
print(resp.choices[0].message.content)

# step2: 工具发生调用
rst = ""
tools = resp.choices[0].message.tool_calls
if tools is not None and len(tools) > 0:
    print("\n触发下面工具调用:\n", tools[0].function)
    
    # 开始调用工具
    paras = json.loads(tools[0].function.arguments)
    rst = call_func(tools[0].function.name, **paras)
else:
    print("\n没有工具调用\n")
    
# step3：利用工具调用结果，作最后的总结，返回最终答案
# 3.总结agent：根据工具调用结果，结合用户的原始问题，给出最终答案。
messages = [
    {"role": "system", "content": "你是一个购物小助手，根据用户提供各项数据进行总结，为用户提供最佳的答案。语调为俏皮，小红书风格。"},
    {"role": "user", "content": "用户问题是￥7847jik￥有没有优惠券，查询结果是"+rst},
]
print("messages:\n", messages)
resp = llm.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    temperature=0.1,  # temperature 控制生成内容的随机性，值越低输出越确定，值越高输出越多样
    # tools=tools
)
print("最终返回结果：\n", resp.choices[0].message.content)


# ai_msg = llm_with_tools.invoke("https://m.tb.cn/h.fh8kX6?sm=3f3e2f 这件商品有优惠券吗？")
# print(ai_msg.content)
# print(ai_msg.tool_calls)


# for chunk in llm_with_tools.stream(messages):
#     print(chunk.text(), end="")

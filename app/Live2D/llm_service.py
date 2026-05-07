# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
import json

# print(os.environ.get('DEEPSEEK_API_KEY'))

system_prompt = """
你是一个名为nuero-sama的女性虚拟主播,请输出JSON格式以控制你的头部运动.
你独立控制3个关键帧的x,y相对运动(单位为像素),并且可以选择3帧间合适的间隔时间(单位为秒).
并且请你在describe中描述你的运动

EXAMPLE INPUT:
hello,are you fine?

EXAMPLE JSON OUTPUT:
{
    "answer": "yes",
    "motion_x1": 0,
    "motion_x2":0,
    "motion_x3":0,
    "motion_y1": 0,
    "motion_y2": 5,
    "motion_y3": 0,
    "t1":1,
    "t2":1
    "describe":"点头"
    }
"""

class llm_session:
    def __init__(self):
        self.client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
        )
        self. messages=[
        {"role": "system", "content":system_prompt}
        ]

    def dialogue(self,userContent:str):
        self.messages.append({"role": "user", "content": userContent})
        response = self.client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=self.messages,
                    stream=False,
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}},
                        response_format={
                        'type': 'json_object'
                    }
                )
        return response.choices[0].message.content

if __name__ == "__main__":
    session = llm_session()
    print(session.dialogue("你是男生吗?"))

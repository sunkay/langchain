from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--language", default="python")
parser.add_argument("--task", default="list of numbers")
args = parser.parse_args()


llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a short {language} function that will {task} ",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    template="Write a test in {language} for the function {code}",
    input_variables=["language", "code"]
)


code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["test", "code"]
)

result = chain({
    "language": args.language,
    "task": args.task
})
code = result["code"]
print(code)

print("---TEST---")
print(result["test"])
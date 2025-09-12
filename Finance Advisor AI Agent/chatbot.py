from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


class FinancialChatbot:
    def __init__(
        self,
        df_alloc,
        df_financial,
        api_key,
        model_name="gpt-4o-mini",
        evaluation_threshold=0.7,
    ):
        self.df_alloc = df_alloc
        self.df_financial = df_financial
        self.llm = self._get_llm(model_name, api_key)
        self.model_name = model_name
        self.prompt_template = self._get_prompt()
        self.llm_chain = self._get_llm_chain()
        self.evaluation_threshold = evaluation_threshold

    def _get_prompt(self):
        return PromptTemplate.from_template(
            """
            You are a financial advisor assistant. Here is the data of the client's portfolios and target allocations.
            Client Portfolio Data:
            {portfolio_data}
            Client Target Allocations:
            {target_allocations}
            Based on the above data, answer the following question:
            {question}
            Think step by step, but provide only the final answer.
            """
        )

    def _get_llm(self, model_name, api_key):
        return ChatOpenAI(model=model_name, temperature=0.0, api_key=api_key)

    def _get_llm_chain(self):
        return self.prompt_template | self.llm

    def generate_response(self, question):
        context = {
            "portfolio_data": self.df_financial.to_string(),
            "target_allocations": self.df_alloc.to_string(),
            "question": question,
        }
        with get_openai_callback() as cb:
            response = self.llm_chain.invoke(context)
            total_tokens = cb.total_tokens
            total_cost = cb.total_cost
        return response.content, total_tokens, total_cost

    def evaluate_response_by_relevancy(self, question, answer):
        metric = AnswerRelevancyMetric(
            threshold=self.evaluation_threshold,
            model=self.model_name,
            include_reason=True,
        )
        test_case = LLMTestCase(input=question, actual_output=answer)
        metric.measure(test_case)
        return metric.score, metric.reason

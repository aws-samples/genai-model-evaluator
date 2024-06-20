class OrchestrationRAGHelper:
    """
    This class is used to store the results of the orchestration, and format all the results.
    """

    def __init__(self, model, time_length, embedding_character_count, llm_character_count, char_process_time, input_embedding_cost,
                 output_embedding_cost, total_embedding_cost, total_embedding_cost_1000, input_llm_cost,
                 output_llm_cost, total_llm_cost, total_llm_cost_1000,
                 final_score, answers_response, final_summary):
        """
        Initializes an instance of the OrchestrationHelper class.
        :param model: The model being evaluated.
        :param time_length: The length of time it took to perform a retrieve and generate.
        :param embedding_character_count: The amount of characters going into the embedding model.
        :param llm_character_count: The amount of characters going into the llm.
        :param char_process_time: How long it took to process a character.
        :param input_embedding_cost: The cost of the input embeddings.
        :param output_embedding_cost: The cost of the output embeddings.
        :param total_embedding_cost: The total cost of the embeddings.
        :param total_embedding_cost_1000: The total cost of the embeddings in 1000s.
        :param input_llm_cost: The cost of the input LLM.
        :param output_llm_cost: The cost of the output LLM.
        :param total_llm_cost: The total cost of the LLM.
        :param total_llm_cost_1000: The total cost of the LLM in 1000s.
        :param final_score: The final score of the models performance.
        :param answers_response: The answers provided from the model being tested.
        :param final_summary: The final summary of the models overall performance.
        """
        self.model = model
        self.time_length = time_length
        self.embedding_character_count = embedding_character_count
        self.llm_character_count = llm_character_count
        self.char_process_time = char_process_time
        self.input_embedding_cost = input_embedding_cost
        self.output_embedding_cost = output_embedding_cost
        self.total_embedding_cost = total_embedding_cost
        self.total_embedding_cost_1000 = total_embedding_cost_1000
        self.input_llm_cost = input_llm_cost
        self.output_llm_cost = output_llm_cost
        self.total_llm_cost = total_llm_cost
        self.total_llm_cost_1000 = total_llm_cost_1000
        self.final_score = final_score
        self.summary_invoke_response = answers_response
        self.final_summary = final_summary

    def format(self):
        """
        Formats the results of the orchestration into a dictionary.

        :return: A dictionary containing the formatted results.
        """
        # creating a dictionary of the results
        result = {
                'Model': self.model,
                'Time Length': self.time_length,
                'Embedding Character Count': self.embedding_character_count,
                'LLM Character Count': self.llm_character_count,
                'Char Process Time': self.char_process_time,
                'Input Embedding Cost': self.input_embedding_cost, 
                'Output Embedding Cost': self.output_embedding_cost,
                'Total Embedding Cost': self.total_embedding_cost,
                'Total Embedding Cost(1000)': self.total_embedding_cost_1000,
                'Input LLM Cost': self.input_llm_cost,
                'Output LLM Cost': self.output_llm_cost,
                'Total LLM Cost': self.total_llm_cost,
                'Total LLM Cost(1000)': self.total_llm_cost_1000,
                'Score': self.final_score,
                'Invoke Response': self.summary_invoke_response
            }
        # returning the final dictionary
        return result

    def evaluation_results(self):
        """
        Formats the evaluation results into a string.

        :return: A string containing the formatted evaluation results.
        """
        # The final summary being formatted for a specific model
        evaluation_result = f"""
                    The final summary for {self.model} is:

                    {self.final_summary}
                    """
        # returning the final formatted evaluation results
        return evaluation_result
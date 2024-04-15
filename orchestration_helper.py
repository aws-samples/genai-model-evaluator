class OrchestrationHelper:
    """
    This class is used to store the results of the orchestration, and format all the results.
    """

    def __init__(self, model, time_length, character_count, char_process_time, input_cost, output_cost, total_cost, total_cost_1000,
                 final_score, summary_invoke_response, final_summary):
        """
        Initializes an instance of the OrchestrationHelper class.
        :param model: The model being evaluated.
        :param time_length: The length of time it took to perform a summarization.
        :param character_count: The amount of characters of the source text.
        :param char_process_time: How long it took to process a character.
        :param input_cost: The cost of the input tokens.
        :param output_cost: The cost of the output tokens.
        :param total_cost: The cost of the input tokens and output tokens.
        :param total_cost_1000: The total cost of the job per 1000 invocations.
        :param final_score: The final score of the models summary performance.
        :param summary_invoke_response: The summary provided from the model being tested.
        :param final_summary: The final summary of the models overall performance.
        """
        self.model = model
        self.time_length = time_length
        self.character_count = character_count
        self.char_process_time = char_process_time
        self.input_cost = input_cost
        self.output_cost = output_cost
        self.total_cost = total_cost
        self.total_cost_1000 = total_cost_1000
        self.final_score = final_score
        self.summary_invoke_response = summary_invoke_response
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
                'Character Count': self.character_count,
                'Char Process Time': self.char_process_time,
                'Input Cost': self.input_cost,
                'Output Cost': self.output_cost,
                'Total Cost': self.total_cost,
                'Total Cost(1000)': self.total_cost_1000,
                'Summary Score': self.final_score,
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
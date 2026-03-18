from pocketflow import Flow
from nodes import InputNode, ClassifyNode, FileAnalysisNode, PaperSearchNode, AnswerNode


def create_md_agent_flow():
    input_node = InputNode()
    classify_node = ClassifyNode(max_retries=3)
    file_analysis_node = FileAnalysisNode(max_retries=2)
    paper_search_node = PaperSearchNode()
    answer_node = AnswerNode(max_retries=3)

    # Linear pipeline per question
    input_node >> classify_node >> file_analysis_node >> paper_search_node >> answer_node

    # Loop back for next question, or exit
    answer_node - "continue" >> input_node

    return Flow(start=input_node)
reate_qa_flow()
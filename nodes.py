import logging
import yaml
from pocketflow import Node
from utils.call_llm import call_llm
from utils.analyze_md import analyze_md
from utils.search_papers import search_papers

logger = logging.getLogger(__name__)


class InputNode(Node):
    def prep(self, shared):
        return len(shared.get("conversation_history", []))

    def exec(self, turn):
        if turn == 0:
            print("\n" + "=" * 60)
            print("  MD Simulation Q&A Agent")
            print("  Powered by Claude + MDAnalysis")
            print("=" * 60)
            print("Type your question about your MD simulation.")
            print("Type 'quit' or 'exit' to end the session.\n")
        question = input(f"[Q{turn + 1}] Your question: ").strip()
        return question

    def post(self, shared, prep_res, exec_res):
        if exec_res.lower() in ("quit", "exit", "q"):
            print("\nEnding session. Goodbye!")
            return "exit"
        shared["current_question"] = exec_res
        return "default"


class ClassifyNode(Node):
    def prep(self, shared):
        question = shared["current_question"]
        history = shared.get("conversation_history", [])[-3:]
        return question, history

    def exec(self, prep_res):
        question, history = prep_res

        history_text = ""
        if history:
            history_text = "\nRecent conversation:\n"
            for h in history:
                history_text += f"Q: {h['question']}\nA: {h['answer'][:200]}...\n\n"

        prompt = f"""You are an expert in molecular dynamics (MD) simulations (GROMACS, AMBER, NAMD, OpenMM).

Classify the following user question and determine what resources are needed to answer it.
{history_text}
Current question: {question}

Return ONLY a YAML block with these exact fields:
```yaml
question_type: <one of: trajectory_analysis | setup_config | theory | troubleshooting | results_interpretation>
needs_file_analysis: <true or false>
needs_paper_search: <true or false>
analysis_targets: <list from: rmsd, rmsf, radius_of_gyration, energy, structure — or empty list []>
reasoning: <one sentence explaining your classification>
```

Rules:
- needs_file_analysis: true only if the question requires actual simulation data (RMSD values, RMSF, structure details, energy)
- needs_paper_search: true if the question benefits from literature references
- analysis_targets: only include targets directly relevant to the question"""

        response = call_llm(prompt)
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        result = yaml.safe_load(yaml_str)

        assert "question_type" in result
        assert "needs_file_analysis" in result
        assert "needs_paper_search" in result
        assert "analysis_targets" in result
        assert result["question_type"] in [
            "trajectory_analysis", "setup_config", "theory",
            "troubleshooting", "results_interpretation"
        ]

        return result

    def post(self, shared, prep_res, exec_res):
        shared["question_type"] = exec_res["question_type"]
        shared["needs_file_analysis"] = exec_res["needs_file_analysis"]
        shared["needs_paper_search"] = exec_res["needs_paper_search"]
        shared["analysis_targets"] = exec_res.get("analysis_targets", [])
        logger.info(
            f"Classified as '{exec_res['question_type']}' | "
            f"file_analysis={exec_res['needs_file_analysis']} | "
            f"paper_search={exec_res['needs_paper_search']} | "
            f"reason: {exec_res.get('reasoning', '')}"
        )
        return "default"


class FileAnalysisNode(Node):
    def prep(self, shared):
        return (
            shared.get("needs_file_analysis", False),
            shared.get("work_dir", ""),
            shared.get("analysis_targets", []),
        )

    def exec(self, prep_res):
        needs_analysis, work_dir, targets = prep_res
        if not needs_analysis:
            return None
        if not work_dir:
            return {"error": "No simulation directory was specified."}
        return analyze_md(work_dir, targets)

    def post(self, shared, prep_res, exec_res):
        shared["file_analysis"] = exec_res
        return "default"


class PaperSearchNode(Node):
    def prep(self, shared):
        return (
            shared.get("needs_paper_search", False),
            shared.get("current_question", ""),
        )

    def exec(self, prep_res):
        needs_search, question = prep_res
        if not needs_search:
            return []
        query = f"molecular dynamics {question}"
        return search_papers(query, max_results=5)

    def post(self, shared, prep_res, exec_res):
        shared["papers"] = exec_res
        return "default"


class AnswerNode(Node):
    def prep(self, shared):
        return {
            "question": shared.get("current_question", ""),
            "question_type": shared.get("question_type", ""),
            "file_analysis": shared.get("file_analysis"),
            "papers": shared.get("papers", []),
            "history": shared.get("conversation_history", [])[-5:],
        }

    def exec(self, prep_res):
        question = prep_res["question"]
        question_type = prep_res["question_type"]
        file_analysis = prep_res["file_analysis"]
        papers = prep_res["papers"]
        history = prep_res["history"]

        history_text = ""
        if history:
            history_text = "\n## Previous Context\n"
            for h in history:
                history_text += f"Q: {h['question']}\nA: {h['answer'][:300]}...\n\n"

        file_text = ""
        if file_analysis:
            file_text = "\n## Simulation Data\n"
            for key, value in file_analysis.items():
                file_text += f"\n### {key.upper()}\n{value}\n"

        papers_text = ""
        if papers:
            papers_text = "\n## Relevant Papers\n"
            for i, p in enumerate(papers, 1):
                authors = ", ".join(p["authors"][:3]) if isinstance(p["authors"], list) else p["authors"]
                papers_text += f"{i}. {p['title']} ({p['year']}) — {authors}\n"
                if p.get("abstract_snippet"):
                    papers_text += f"   {p['abstract_snippet'][:200]}\n"

        prompt = f"""You are an expert MD simulation scientist specializing in GROMACS, AMBER, NAMD, and OpenMM.

Answer the following question with scientific depth. Use the simulation data and papers if provided.
{history_text}
## Current Question
{question}
(Question type: {question_type})
{file_text}
{papers_text}
Return ONLY a YAML block in this exact format:
```yaml
analysis: |
  <detailed scientific analysis, 3-5 sentences. If simulation data is provided, interpret the numbers specifically.>
references:
  - "<Author(s), Year. Title. Journal/Source.>"
  - "<if no papers were provided, suggest 1-2 key references from your knowledge>"
next_steps:
  - "<specific, actionable experiment or analysis to run next>"
  - "<second concrete suggestion>"
  - "<third concrete suggestion>"
```"""

        response = call_llm(prompt)
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        result = yaml.safe_load(yaml_str)

        assert "analysis" in result
        assert "references" in result
        assert "next_steps" in result

        return result

    def post(self, shared, prep_res, exec_res):
        shared["answer"] = exec_res

        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("-" * 60)
        print(exec_res["analysis"])

        print("\nREFERENCES")
        print("-" * 60)
        for ref in exec_res["references"]:
            print(f"  • {ref}")

        print("\nNEXT STEPS")
        print("-" * 60)
        for i, step in enumerate(exec_res["next_steps"], 1):
            print(f"  {i}. {step}")
        print("=" * 60 + "\n")

        formatted = (
            f"Analysis: {exec_res['analysis']}\n"
            f"Next steps: {'; '.join(exec_res['next_steps'])}"
        )
        shared["conversation_history"].append({
            "question": prep_res["question"],
            "answer": formatted,
        })

        return "continue"

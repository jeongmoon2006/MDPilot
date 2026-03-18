import os
import sys
import logging
from flow import create_md_agent_flow

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    # Get simulation directory from CLI arg or prompt
    if len(sys.argv) > 1:
        work_dir = sys.argv[1]
    else:
        work_dir = input("Enter path to your simulation directory: ").strip()

    if not os.path.isdir(work_dir):
        print(f"Error: '{work_dir}' is not a valid directory.")
        sys.exit(1)

    work_dir = os.path.abspath(work_dir)
    print(f"Simulation directory: {work_dir}")

    shared = {
        "work_dir": work_dir,
        "conversation_history": [],
        # Per-question state (reset each iteration by nodes)
        "current_question": "",
        "question_type": "",
        "needs_file_analysis": False,
        "needs_paper_search": False,
        "analysis_targets": [],
        "file_analysis": None,
        "papers": [],
        "answer": None,
    }

    flow = create_md_agent_flow()
    flow.run(shared)


if __name__ == "__main__":
    main()

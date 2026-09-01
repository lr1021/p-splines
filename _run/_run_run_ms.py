import os
import sys
from pathlib import Path
import traceback

# sys.path.append(str(Path.cwd().parent))
from importlib import import_module

# print(os.getcwd())
sys.path[0] = os.getcwd()
# print(sys.path)


def main(keys_path):
    keys_loc = keys_path.replace('/', '.').removesuffix('.py')
    keys = import_module(keys_loc)

    #############################################################################################
    # Idatas
    if keys.run_idatas:
        print("Running idatas") 
        import _run._run_model_idatas as _run_model_idatas
        _run_model_idatas.main(keys_path)
    else:
        print("Skipping idatas")

    #############################################################################################
    # Reports
    if keys.run_reports:
        print("Running reports") 
        import _run._run_model_reports as _run_model_reports
        _run_model_reports.main(keys_path)
    else:
        print("Skipping reports")

    ############################################################################################# 
    # Replications Reports
    if keys.run_replications_report:
        print("Running replications report")
        import _run._run_model_replications_report_ms as _run_model_replications_report_ms
        _run_model_replications_report_ms.main(keys_path)
    else:
        print("Skipping replications report")

    ############################################################################################# 
    # Task Summaries Report
    if keys.run_replications_report:
        print("Running task summaries report")
        import _run._run_model_task_summaries_ms as _run_model_task_summaries_ms
        _run_model_task_summaries_ms.main(keys_path)
    else:
        print("Skipping task summaries report")
    
    ############################################################################################# 
    # Comparisons Report
    try:
        if keys.run_comparison_report:
            print("Running comparisons report")
            import _run._run_model_comparison as _run_model_comparison
            _run_model_comparison.main(keys_path)
        else:
            print("Skipping comparisons report")
    except Exception as e:
        print(f"Error occurred while running comparisons report: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main(sys.argv[1])
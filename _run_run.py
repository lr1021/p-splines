import os
import sys
from importlib import import_module

def main(keys_path):
    keys_loc = keys_path.replace('/', '.').removesuffix('.py')
    keys = import_module(keys_loc)

    #############################################################################################
    # Idatas
    if keys.run_idatas:
        print("Running idatas") 
        import _run_model_idatas
        _run_model_idatas.main(keys_path)
    else:
        print("Skipping idatas")

    #############################################################################################
    # Reports
    if keys.run_reports:
        print("Running reports") 
        import _run_model_reports
        _run_model_reports.main(keys_path)
    else:
        print("Skipping reports")

    ############################################################################################# 
    # Replications Reports
    if keys.run_replications_report:
        print("Running replications report")
        import _run_model_replications_report
        _run_model_replications_report.main(keys_path)
    else:
        print("Skipping replications report")

    ############################################################################################# 
    # Task Summaries Report
    if keys.run_replications_report:
        print("Running task summaries report")
        import _run_model_task_summaries
        _run_model_task_summaries.main(keys_path)
    else:
        print("Skipping task summaries report")

if __name__ == "__main__":
    main(sys.argv[1])
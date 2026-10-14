"""Execute saved-result notebook without a ZeroMQ kernel (Windows fallback)."""
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root=Path(__file__).resolve().parents[1]
path=root/"combined_signal_analysis/bo_sm_w_relaxed_research.ipynb"
notebook=nbformat.read(path,as_version=4)
shell=InteractiveShell.instance()
count=0
for index,cell in enumerate(notebook.cells):
    cell.setdefault("id",f"relaxed-{index:02d}")
    if cell.cell_type!="code":
        continue
    count+=1
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=False)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f"Notebook cell {index} failed: {result.error_before_exec or result.error_in_exec}")
    outputs=[]
    if captured.stdout:
        outputs.append(nbformat.v4.new_output("stream",name="stdout",text=captured.stdout))
    if captured.stderr:
        outputs.append(nbformat.v4.new_output("stream",name="stderr",text=captured.stderr))
    for output in captured.outputs:
        outputs.append(nbformat.v4.new_output("display_data",data=output.data,metadata=output.metadata))
    cell.outputs=outputs
    cell.execution_count=count
nbformat.validate(notebook)
nbformat.write(notebook,path)
print(f"Executed {count} cells successfully; saved {sum(len(c.get('outputs',[])) for c in notebook.cells)} outputs.")

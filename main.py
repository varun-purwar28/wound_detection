import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import NotebookExporter

def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': './'}})

    # Optionally, you can save the executed notebook
    with open(notebook_path.replace('.ipynb', '_executed.ipynb'), 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    notebooks = [
        'Unet.ipynb',  # Replace with actual notebook file names
        'WD_Cropping.ipynb',
        'WD_YoloV8.ipynb'
    ]

    for notebook in notebooks:
        run_notebook(notebook)
        print(f'{notebook} executed successfully.')

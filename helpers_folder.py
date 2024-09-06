from pathlib import Path
import Constants as const



def create_folder(main_folder, sub_folder):
    path = Path(const.PROJECT_ROOT, "results", main_folder, sub_folder)

    # create/check path
    if not path.exists():
        path.mkdir(parents=True)

    # empty folder
    for file in path.glob('*'):
        file.unlink()

    return path

def create_folders(folder_name="NODE"):
    plot_path = Path(const.PROJECT_ROOT, 'results', folder_name, 'final_training')
    plot_test_path = Path(const.PROJECT_ROOT, 'results', folder_name, 'test_samples')
    plot_diverse_path = Path(const.PROJECT_ROOT, 'results', folder_name, 'diverse_samples')

    # create/check for training plot path
    if not plot_path.exists():
        plot_path.mkdir(parents=True)
    # empty folder
    for file in plot_path.glob('*'):
        file.unlink()

    # repeat for test plot path
    if not plot_test_path.exists():
        plot_test_path.mkdir(parents=True)
    # empty folder
    for file in plot_test_path.glob('*'):
        file.unlink()

    # repeat for diverse plot path
    if not plot_diverse_path.exists():
        plot_diverse_path.mkdir(parents=True)
    # empty folder
    for file in plot_diverse_path.glob('*'):
        file.unlink()

    return plot_path, plot_test_path, plot_diverse_path
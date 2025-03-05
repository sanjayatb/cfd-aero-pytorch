from source.results.result_analyzer import ResultViewer

if __name__ == "__main__":
    plotter = ResultViewer()

    plotter.training_size_vs_mse("../outputs/scores/2025-03-05_set_of_experiment_scores.csv")
    plotter.training_size_vs_r2("../outputs/scores/2025-03-05_set_of_experiment_scores.csv")
    plotter.num_of_points_vs_mse(
        "../outputs/scores/const_train_size_200/2025-03-06_Drag_train_size_200_experiment_scores.csv")
    plotter.num_of_points_vs_r2(
        "../outputs/scores/const_train_size_200/2025-03-06_Drag_train_size_200_experiment_scores.csv")

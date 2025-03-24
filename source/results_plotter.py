from source.config.loader import load_config
from source.results.result_analyzer import ResultViewer

if __name__ == "__main__":
    plotter = ResultViewer()

    plotter.plot_true_vs_predicted_csv("../outputs/best_models/","2025-03-12_Drag_20250312_drivaer_best_variable_train_size_DrivAerML_PointNet_SimplePointNet_ts300_bs16_epochs300_pts100000_lr0.001_drop0.3_[3_64_128_256_512_1024_2048]_[2048_1024_256_1]_predictions_on_test_set.csv")
    # plotter.plot_true_vs_predicted_csv("../outputs/best_models/","2025-03-11_Drag_20250311_drivaer_best_model_DrivAerML_PointNet_SimplePointNet_ts350_bs16_epochs200_pts100000_lr0.001_drop0.1_[3:128:256:512:1024:2048]_[2048:1024:256:1]_predictions_on_test_set.csv")
    # plotter.plot_true_vs_predicted_csv("../outputs/best_models/","2025-03-11_Drag_20250311_drivaer_best_model_DrivAerML_PointNet_SimplePointNet_ts350_bs16_epochs200_pts100000_lr0.001_drop0.2_[3:128:256:512:1024:2048]_[2048:1024:256:1]_predictions_on_test_set.csv")
    # plotter.plot_true_vs_predicted_csv("../outputs/best_models/","2025-03-12_Drag_20250311_drivaer_best_model_DrivAerML_PointNet_SimplePointNet_ts350_bs16_epochs300_pts100000_lr0.001_drop0.2_[3_64_128_256_512_1024_2048]_[2048_1024_256_1]_predictions_on_test_set.csv")
    # plotter.plot_true_vs_predicted_csv("../outputs/best_models/","2025-03-12_Drag_20250311_drivaer_best_model_DrivAerML_PointNet_SimplePointNet_ts350_bs16_epochs300_pts100000_lr0.001_drop0.3_[3_64_128_256_512_1024_2048]_[2048_1024_256_1]_predictions_on_test_set.csv")

    #config = load_config("../configs/system_config.yml")
    #plotter.plot_true_vs_predicted(config, "../outputs/best_models/20250310_r2_8_layer_fine_tune_DrivAerML_PointNet_SimplePointNet_ts350_bs16_epochs200_pts100000_lr0.001_drop0.1_[3_64_128_256_512_1024_2048]_[2048_1024_256_1]_best_model.pth")
    # plotter.training_size_vs_mse("../outputs/scores/2025-03-05_set_of_experiment_scores.csv")
    # plotter.training_size_vs_r2("../outputs/scores/2025-03-05_set_of_experiment_scores.csv")
    # plotter.num_of_points_vs_mse(
    #     "../outputs/scores/const_train_size_200/2025-03-06_Drag_train_size_200_experiment_scores.csv")
    # plotter.num_of_points_vs_r2(
    #     "../outputs/scores/const_train_size_200/2025-03-06_Drag_train_size_200_experiment_scores.csv")
    #
    plotter.training_size_vs_mse("../outputs/best_models/2025-03-12_Drag_drivaer_best_variable_train_size_experiment_scores.csv")
    plotter.training_size_vs_r2("../outputs/best_models/2025-03-12_Drag_drivaer_best_variable_train_size_experiment_scores.csv")



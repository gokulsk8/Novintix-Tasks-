from pathlib import Path

from flask import Flask, render_template, send_from_directory

from analysis_agartala import FIG_DIR, run_full_analysis


app = Flask(__name__)


@app.route("/")
def index():
    results = run_full_analysis()

    forecast = results["forecast_results"]
    missing_summary_html = results["missing_summary"].to_frame("Missing Count").to_html(classes="table table-striped")
    outliers_df = results["outliers"]
    outliers_html = (
        outliers_df.to_html(classes="table table-striped", index=False)
        if not outliers_df.empty
        else "<p>No abnormal Index Value readings detected via IQR rule.</p>"
    )
    cluster_counts_html = results["cluster_counts"].to_frame("Days").to_html(classes="table table-striped")
    cluster_summary_html = results["cluster_summary"].to_html(classes="table table-striped")

    return render_template(
        "index.html",
        mae=f"{forecast.mae:.2f}",
        rmse=f"{forecast.rmse:.2f}",
        r2=f"{forecast.r2:.2f}",
        missing_summary_html=missing_summary_html,
        outliers_html=outliers_html,
        cluster_counts_html=cluster_counts_html,
        cluster_summary_html=cluster_summary_html,
        figure_files=results["figure_files"],
    )


@app.route("/figures/<path:filename>")
def figures(filename: str):
    # Serve generated figure files
    fig_dir = Path(FIG_DIR)
    return send_from_directory(fig_dir, filename)


if __name__ == "__main__":
    app.run(debug=True)



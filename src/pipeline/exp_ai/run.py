import cfg, common, train, shap_analysis, lime_analysis, assess, report

if __name__ == "__main__":
    common.make_dirs()
    (cfg.JS / "dashboard.json").unlink(missing_ok=True)
    for step in (train, shap_analysis, lime_analysis, assess, report):
        step.main()
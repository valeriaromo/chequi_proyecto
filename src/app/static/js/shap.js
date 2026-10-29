// Plotly helpers for SHAP. Data is placeholder for now.
window.ChequiShap = (function () {
  function waterfall(elId, example) {
    const el = document.getElementById(elId);
    if (!el || typeof Plotly === "undefined") return;

    const features = example.features.map(f => f.name);
    const values = example.features.map(f => f.shap);

    const trace = {
      type: "waterfall",
      orientation: "h",
      y: features,
      x: values,
      measure: values.map(() => "relative"),
      base: example.base_value,
      increasing: { marker: { color: "#c0392b" } },
      decreasing: { marker: { color: "#1f9d55" } },
      connector: { line: { color: "#999" } }
    };

    const layout = {
      margin: { l: 140, r: 20, t: 30, b: 40 },
      title: "SHAP — one forecast",
      xaxis: { title: "Impact on T2M_MIN" },
      showlegend: false
    };

    Plotly.newPlot(el, [trace], layout, { displayModeBar: false, responsive: true });
  }

  return { waterfall };
})();
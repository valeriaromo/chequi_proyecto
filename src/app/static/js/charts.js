// Chart.js helpers. All data here is placeholder for now.
window.ChequiCharts = (function () {
  const PRIMARY = "#2a6df4";

  function _ctx(id) {
    const el = document.getElementById(id);
    return el ? el.getContext("2d") : null;
  }

    function pcaScatter(id, data) {
    const ctx = _ctx(id);
    if (!ctx) return;

    const pts = (data.points || []).map(p => ({
      x: p.pc1 ?? p.x,
      y: p.pc2 ?? p.y,
      label: p.label ?? ""
    }));
    const ev = data.explained_variance || [];
    const pct = (v) => (v == null ? "" : ` (${(v * 100).toFixed(1)}%)`);
    const xLabel = data.xLabel || `PC1${pct(ev[0])}`;
    const yLabel = data.yLabel || `PC2${pct(ev[1])}`;

    new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [{
          label: "Días (muestra)",
          data: pts,
          backgroundColor: "rgba(42, 109, 244, 0.35)",
          pointRadius: pts.length > 100 ? 2.5 : 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => `${c.raw.label} (${c.parsed.x.toFixed(2)}, ${c.parsed.y.toFixed(2)})`
            }
          }
        },
        scales: {
          x: { title: { display: true, text: xLabel } },
          y: { title: { display: true, text: yLabel } }
        }
      }
    });
  }

  function skillBars(id, payload) {
    const ctx = _ctx(id);
    if (!ctx) return;
    const labels = [];
    const values = [];
    payload.variables.forEach(v => {
      v.horizons.forEach(h => {
        labels.push(`${v.variable} (h=${h.h})`);
        values.push(h.skill);
      });
    });
    new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Skill (walk-forward)",
          data: values,
          backgroundColor: values.map(v => v >= 0 ? "#1f9d55" : "#c0392b")
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { title: { display: true, text: "Skill" } } }
      }
    });
  }

  return { pcaScatter, skillBars };
})();
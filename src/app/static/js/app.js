window.Chequi = window.Chequi || {};
window.Chequi.api = function (path) {
  return fetch(path).then(r => r.json());
};
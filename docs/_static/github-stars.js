(function () {
  function fmt(n) {
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    return String(n);
  }
  fetch("https://api.github.com/repos/GrayboxTech/weightslab")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      var count = data && data.stargazers_count;
      if (typeof count === "number") {
        document.querySelectorAll(".wl-stars-count").forEach(function (el) {
          el.textContent = fmt(count);
        });
      }
    })
    .catch(function () {});
})();

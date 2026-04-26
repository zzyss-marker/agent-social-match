(function () {
  "use strict";

  window.showToast = function (message, type) {
    var container = document.getElementById("toast-container");
    if (!container) return;

    var toast = document.createElement("div");
    toast.className = "toast " + (type || "info");
    toast.textContent = message;
    container.appendChild(toast);

    window.setTimeout(function () {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(8px)";
      toast.style.transition = "all 0.25s ease";
      window.setTimeout(function () {
        toast.remove();
      }, 260);
    }, 3200);
  };
})();
